# Source: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds," CVPR 2019.
# Converts a raw LiDAR point cloud into a dense pillar tensor ready for PointNet encoding.

import torch


def discretize_point_cloud(points: torch.Tensor, voxel_size: tuple, point_cloud_range: tuple):
    """
    Assign each point to a pillar grid cell.

    points: (N, 4) — x, y, z, intensity
    voxel_size: (x_size, y_size, z_size)
    point_cloud_range: ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    returns: (N, 2) pillar grid indices (ix, iy) per point
    """
    (x_min, x_max), (y_min, y_max), _ = point_cloud_range
    x_size, y_size, _ = voxel_size
    grid_w = round((x_max - x_min) / x_size)
    grid_h = round((y_max - y_min) / y_size)

    # A point can pass the caller's `< x_max` range filter (in float32) yet
    # still floor to exactly grid_w here, since the filter's comparison and
    # this division/floor round differently right at the boundary. Rare, but
    # with hundreds of thousands of points per sample (multi-sweep LiDAR) it
    # does happen — clamp rather than let a downstream scatter go out of bounds.
    ix = ((points[:, 0] - x_min) / x_size).floor().long().clamp(0, grid_w - 1)
    iy = ((points[:, 1] - y_min) / y_size).floor().long().clamp(0, grid_h - 1)

    return torch.stack((ix, iy), dim=1)


def get_pillar_centers(pillar_indices: torch.Tensor, voxel_size: tuple, point_cloud_range: tuple):
    """
    Compute the geometric center (x, y) of each pillar from its grid index.

    pillar_indices: (N, 2) — (ix, iy) per point
    returns: (N, 2) — (x_center, y_center) in ego-frame meters per point
    """
    (x_min, _), (y_min, _), _ = point_cloud_range
    x_size, y_size, _ = voxel_size

    cx = x_min + (pillar_indices[:, 0] + 0.5) * x_size
    cy = y_min + (pillar_indices[:, 1] + 0.5) * y_size

    return torch.stack((cx, cy), dim=1)


def augment_pillars(points: torch.Tensor, pillar_centers: torch.Tensor, cluster_centers: torch.Tensor):
    """
    Augment each point with offset features as described in the PointPillars paper.

    x_c, y_c, z_c — offset from the arithmetic mean of all points in the pillar (cluster centroid)
    x_p, y_p      — offset from the geometric pillar center

    points:          (N, 4) — x, y, z, intensity
    pillar_centers:  (N, 2) — geometric pillar center per point
    cluster_centers: (N, 3) — mean x, y, z of all points in the pillar per point

    returns: (N, 9)
    """
    x_c = points[:, 0] - cluster_centers[:, 0]
    y_c = points[:, 1] - cluster_centers[:, 1]
    z_c = points[:, 2] - cluster_centers[:, 2]

    x_p = points[:, 0] - pillar_centers[:, 0]
    y_p = points[:, 1] - pillar_centers[:, 1]

    return torch.cat([
        points,
        x_c.unsqueeze(1), y_c.unsqueeze(1), z_c.unsqueeze(1),
        x_p.unsqueeze(1), y_p.unsqueeze(1),
    ], dim=1)


def optimize_pillars(points: torch.Tensor, pillar_indices: torch.Tensor, max_points_per_pillar: int):
    """
    Group points into pillars and pad/truncate to a fixed size.

    points:         (N, 9) — augmented point features
    pillar_indices: (N, 2) — (ix, iy) grid index per point
    max_points_per_pillar: fixed pillar size N

    returns:
        pillar_features: (P, N, 9)
        unique_indices:  (P, 2) — (ix, iy) per pillar
    """
    unique_indices, inverse = torch.unique(pillar_indices, dim=0, return_inverse=True)
    P = unique_indices.shape[0]

    # Vectorized per-pillar slot assignment instead of a Python loop over pillars
    # (see _cluster_centers in point_pillars.py for why). Rank = each point's
    # position within its own pillar, in original point order — computed via a
    # stable sort by pillar so equal-pillar points keep their relative order,
    # then position-within-run via cumcount-per-group.
    order          = torch.argsort(inverse, stable=True)
    sorted_inverse = inverse[order]
    is_group_start = torch.ones_like(sorted_inverse, dtype=torch.bool)
    is_group_start[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    group_start_pos = torch.zeros(P, dtype=torch.long, device=points.device)
    group_start_pos[sorted_inverse[is_group_start]] = torch.nonzero(is_group_start, as_tuple=True)[0]
    rank_sorted = torch.arange(len(sorted_inverse), device=points.device) - group_start_pos[sorted_inverse]
    rank = torch.empty_like(rank_sorted)
    rank[order] = rank_sorted

    keep = rank < max_points_per_pillar
    pillar_features = torch.zeros(P, max_points_per_pillar, points.shape[1],
                                  device=points.device, dtype=points.dtype)
    pillar_features[inverse[keep], rank[keep]] = points[keep]

    return pillar_features, unique_indices