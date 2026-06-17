"""
Code from PointPillar Implementation
"""
import torch
from torch import nn

def discretize_point_cloud(points: torch.Tensor, voxel_size: tuple, point_cloud_range: tuple):
    """
    Discretize the point cloud into pillars.

    Args:
        points (torch.Tensor): The input point cloud of shape (N, 3) where N is the number of points.
        voxel_size (tuple): The size of each voxel in the format (x_size, y_size, z_size).
        point_cloud_range (tuple): The range of the point cloud in the format ((x_min, x_max), (y_min, y_max), (z_min, z_max)).

    Returns:
        torch.Tensor: A tensor containing the pillar indices for each point.
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = point_cloud_range
    x_size, y_size, z_size = voxel_size

    ix = ((points[:, 0] - x_min) / x_size).floor().long()
    iy = ((points[:, 1] - y_min) / y_size).floor().long()
    iz = ((points[:, 2] - z_min) / z_size).floor().long()

    pillar_indices = torch.stack((ix, iy), dim=1)

    return pillar_indices

def get_pillar_centers(pillar_indices: torch.Tensor, voxel_size: tuple, point_cloud_range: tuple):
    """
    Calculate the centers of the pillars based on their indices.

    Args:
        pillar_indices (torch.Tensor): The indices of the pillars for each point.
        voxel_size (tuple): The size of each voxel in the format (x_size, y_size, z_size).
        point_cloud_range (tuple): The range of the point cloud in the format ((x_min, x_max), (y_min, y_max), (z_min, z_max)).

    Returns:
        torch.Tensor: A tensor containing the center coordinates of each pillar.
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = point_cloud_range
    x_size, y_size, z_size = voxel_size

    pillar_centers_x = x_min + (pillar_indices[:, 0] + 0.5) * x_size
    pillar_centers_y = y_min + (pillar_indices[:, 1] + 0.5) * y_size

    pillar_centers = torch.stack((pillar_centers_x, pillar_centers_y), dim=1)

    return pillar_centers

def augment_pillars(points: torch.Tensor, pillar_centers: torch.Tensor):
    """
    - points in each pillar augmented w/ x_c, y_c, z_c, x_p, and y_p
    - c subscript is distance to arithmetic mean of all point in pillar
    - p subscript denotes offset from the pillar x,y center
    - augmented lidar point l is now D = 9 dimensional

    Args:
        points (torch.Tensor): The input point cloud of shape (N, 3) where N is the number of points.
        pillar_centers (torch.Tensor): The center coordinates of each pillar.

    Returns:
        torch.Tensor: A tensor containing the augmented point features.
    """
    x_c = points[:, 0] - pillar_centers[:, 0]
    y_c = points[:, 1] - pillar_centers[:, 1]
    z_c = points[:, 2] - pillar_centers[:, 2]

    x_p = points[:, 0] - pillar_centers[:, 0]
    y_p = points[:, 1] - pillar_centers[:, 1]

    augmented_points = torch.cat((points, x_c.unsqueeze(1), y_c.unsqueeze(1), z_c.unsqueeze(1), x_p.unsqueeze(1), y_p.unsqueeze(1)), dim=1)

    return augmented_points

class PointPillars(nn.Module):
    pass