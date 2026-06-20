# Source: Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation," 2022.
# Fuses camera and LiDAR BEV features via concatenation followed by a conv encoder
# to correct for spatial misalignment caused by inaccurate depth estimation in LSS.

import torch
import torch.nn as nn


class BEVEncoder(nn.Module):
    def __init__(self, camera_channels: int, lidar_channels: int, out_channels: int = 256):
        """
        camera_channels: channels from LSS output
        lidar_channels:  channels from PointPillars backbone output (6*C = 384)
        out_channels:    fused BEV feature channels passed to task heads
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(camera_channels + lidar_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        camera_bev: (B, camera_channels, H, W)
        lidar_bev:  (B, lidar_channels,  H, W)
        returns:    (B, out_channels, H, W)
        """
        return self.encoder(torch.cat([camera_bev, lidar_bev], dim=1))
