# Source: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds," CVPR 2019.
# Orchestrates pillarization → PointNet encoding → BEV scatter → backbone → detection head.

import torch
import torch.nn as nn

from .pillarize import discretize_point_cloud, get_pillar_centers, augment_pillars, optimize_pillars
from .simple_pointnet import SimplifiedPointNet
from .backbone import PillarBackbone

class PointPillars(nn.Module):
    def __init__(
        self,
        voxel_size: tuple = (0.16, 0.16, 4.0),
        point_cloud_range: tuple = ((-51.2, 51.2), (-51.2, 51.2), (-5.0, 3.0)),
        max_points_per_pillar: int = 32,
        C: int = 64,
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_pillar = max_points_per_pillar

        (x_min, x_max), (y_min, y_max), _ = point_cloud_range
        self.grid_width  = round((x_max - x_min) / voxel_size[0])
        self.grid_height = round((y_max - y_min) / voxel_size[1])
        self.C = C

        self.pointnet = SimplifiedPointNet(input_dim=9, output_dim=C)
        self.backbone = PillarBackbone(C=C)

    def _cluster_centers(self, points: torch.Tensor, pillar_indices: torch.Tensor) -> torch.Tensor:
        unique, inverse = torch.unique(pillar_indices, dim=0, return_inverse=True)
        centers = torch.zeros(len(points), 3, device=points.device)
        for i in range(len(unique)):
            mask = inverse == i
            centers[mask] = points[mask, :3].mean(dim=0)
        return centers

    def _scatter(self, pillar_features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        bev = torch.zeros(self.C, self.grid_height, self.grid_width, device=pillar_features.device)
        bev[:, indices[:, 1], indices[:, 0]] = pillar_features.T
        return bev

    def forward(self, points: torch.Tensor):
        """
        points: (N, 4) — x, y, z, intensity in ego frame
        returns: cls (B, num_anchors * num_classes, H, W)
                 reg (B, num_anchors * 7, H, W)
        """
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.point_cloud_range
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        )
        points = points[mask]

        pillar_indices = discretize_point_cloud(points, self.voxel_size, self.point_cloud_range)
        pillar_centers = get_pillar_centers(pillar_indices, self.voxel_size, self.point_cloud_range)
        cluster_centers = self._cluster_centers(points, pillar_indices)

        augmented = augment_pillars(points, pillar_centers, cluster_centers)
        pillars, unique_indices = optimize_pillars(augmented, pillar_indices, self.max_points_per_pillar)

        encoded = self.pointnet(pillars)                      # (P, C)
        bev = self._scatter(encoded, unique_indices)          # (C, H, W)
        bev = bev.unsqueeze(0)                               # (1, C, H, W)

        bev_features = self.backbone(bev)                    # (1, 6C, H/2, W/2)

        return bev_features
