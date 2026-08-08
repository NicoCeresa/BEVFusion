# Source: Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation," 2022.
# Top-level pipeline: camera images + LiDAR point cloud → fused BEV → detection outputs.

import torch
import torch.nn as nn

from camera.lss import compile_model
from lidar.point_pillars import PointPillars
from fusion.bev_encoder import BEVEncoder
from fusion.detection_head import SSD


# Constants LSS derives from grid_conf/data_aug_conf in __init__ rather than
# learning. They live in the state dict, and `frustum`'s shape depends on the
# input image size — so loading pretrained weights saved at a different
# resolution fails on shape mismatch. They're already correct from
# construction, so drop them from the load instead of forcing a resize.
LSS_DERIVED_KEYS = ("frustum", "dx", "bx", "nx")


def load_checkpoint(model, path, device):
    """Load a full BEVFusion checkpoint, ignoring LSS's derived constants.

    Only `frustum` (and dx/bx/nx) depend on input resolution; every learned
    weight is convolutional and therefore resolution-agnostic. Dropping the
    derived keys lets a checkpoint trained at one image size load at another
    — the constants are rebuilt correctly in __init__ either way.

    Note this makes such a checkpoint *load*, not necessarily perform well:
    features learned at one scale won't transfer perfectly to another, so
    retraining is still advisable after a resolution change.
    """
    state = torch.load(path, map_location=device)
    state = {k: v for k, v in state.items()
             if k.rsplit(".", 1)[-1] not in LSS_DERIVED_KEYS}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k.rsplit(".", 1)[-1] not in LSS_DERIVED_KEYS]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch — missing: {missing}, unexpected: {unexpected}")
    return model


class BEVFusion(nn.Module):
    def __init__(self, lss_weights: str, grid_conf: dict, data_aug_conf: dict,
                 lidar_channels: int = 384, camera_channels: int = 1,
                 fused_channels: int = 256, num_classes: int = 3, num_anchors: int = 2):
        super().__init__()

        self.camera_encoder = compile_model(grid_conf, data_aug_conf, outC=camera_channels)
        lss_state = torch.load(lss_weights, map_location="cpu")
        lss_state = {k: v for k, v in lss_state.items() if k not in LSS_DERIVED_KEYS}
        missing, unexpected = self.camera_encoder.load_state_dict(lss_state, strict=False)
        unexpected = [k for k in unexpected if k not in LSS_DERIVED_KEYS]
        missing = [k for k in missing if k not in LSS_DERIVED_KEYS]
        if missing or unexpected:
            raise RuntimeError(f"LSS weight mismatch — missing: {missing}, unexpected: {unexpected}")

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
                 reg (B, num_anchors * 9, H, W)
        """
        camera_bev = self.camera_encoder(images, rots, trans, intrins, post_rots, post_trans)
        lidar_bev  = self.lidar_encoder(points)
        fused_bev  = self.bev_encoder(camera_bev, lidar_bev)
        cls, reg   = self.head(fused_bev)

        return cls, reg
