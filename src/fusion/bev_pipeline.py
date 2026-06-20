# Source: Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation," 2022.
# Top-level pipeline: camera images + LiDAR point cloud → fused BEV → detection outputs.

import torch
import torch.nn as nn

from backbones.lss_model import compile_model
from lidar.point_pillars import PointPillars
from fusion.bev_encoder import BEVEncoder
from lidar.detection_head import SSD


class BEVFusion(nn.Module):
    def __init__(self, lss_weights: str, grid_conf: dict, data_aug_conf: dict,
                 lidar_channels: int = 384, camera_channels: int = 1,
                 fused_channels: int = 256, num_classes: int = 3, num_anchors: int = 2):
        super().__init__()

        self.camera_encoder = compile_model(grid_conf, data_aug_conf, outC=camera_channels)
        self.camera_encoder.load_state_dict(torch.load(lss_weights, map_location="cpu"))

        self.lidar_encoder = PointPillars(
            voxel_size=(0.25, 0.25, 4.0),
            point_cloud_range=((-50.0, 50.0), (-50.0, 50.0), (-5.0, 3.0)),
        )

        self.bev_encoder = BEVEncoder(
            camera_channels=camera_channels,
            lidar_channels=lidar_channels,
            out_channels=fused_channels,
        )

        self.head = SSD(in_channels=fused_channels, num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, images, rots, trans, intrins, post_rots, post_trans, points):
        """
        images:    (B, N, 3, H, W)
        rots:      (B, N, 3, 3)
        trans:     (B, N, 3)
        intrins:   (B, N, 3, 3)
        post_rots: (B, N, 3, 3)
        post_trans:(B, N, 3)
        points:    (N, 4) — x, y, z, intensity

        returns: cls (B, num_anchors * num_classes, H, W)
                 reg (B, num_anchors * 7, H, W)
        """
        camera_bev = self.camera_encoder(images, rots, trans, intrins, post_rots, post_trans)
        lidar_bev  = self.lidar_encoder(points)
        fused_bev  = self.bev_encoder(camera_bev, lidar_bev)
        cls, reg   = self.head(fused_bev)

        return cls, reg
