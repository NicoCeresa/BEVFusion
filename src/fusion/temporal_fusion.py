# Temporal BEV fusion, inspired by BEVFormer's temporal self-attention but
# implemented as an explicit ego-motion warp instead of learned attention —
# the fused BEV grid here is a fixed metric grid (see pipeline.py), so the
# alignment between two frames is exactly known rather than something a
# network needs to learn to attend across.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusion(nn.Module):
    def __init__(self, channels: int = 256, xbound=(-50.0, 50.0, 0.5), ybound=(-50.0, 50.0, 0.5)):
        """
        channels: fused BEV channels per frame (current + warped-prev get
                  concatenated to 2*channels, then mixed back down to channels)
        xbound/ybound: [min, max, step] metric bounds of the BEV grid — the
                  same config.yaml camera.xbound/ybound pipeline.py already
                  gets as grid_conf. Duplicated here (rather than imported
                  from scripts/common.py) because src/ has no dependency on
                  scripts/ and common.py itself imports from fusion.pipeline
                  — importing back would be circular.
        """
        super().__init__()
        self.x_max = xbound[1]
        self.y_max = ybound[1]
        assert xbound[0] == -xbound[1] and ybound[0] == -ybound[1], \
            "warp math below assumes a grid symmetric about the ego origin"

        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fused_t0: torch.Tensor, fused_t1: torch.Tensor,
                ego_transform: torch.Tensor, has_prev: torch.Tensor) -> torch.Tensor:
        """
        fused_t0, fused_t1: (B, channels, H, W) — current and previous frame's
                             fused BEV, both still in their own frame's ego coords.
        ego_transform:      (B, 3) — (dx, dy, dyaw): the previous ego frame's
                             origin/heading expressed in current-ego metric
                             coordinates (see dataloader.py's __getitem__).
        has_prev:            (B,) bool — False at scene starts, where dataloader.py
                             already set ego_transform to zero and duplicated the
                             current frame as "prev". The transform is then
                             provably identity, so skip the warp (and the caller
                             should skip the redundant encoder pass) entirely.
        returns: (B, channels, H, W)
        """
        if not has_prev.any():
            return fused_t0

        # Warping fused_t1 into the current frame means resampling it at each
        # *current*-frame output location — i.e. we need the inverse of the
        # (dx, dy, dyaw) transform, which maps prev-ego -> current-ego:
        #   p_cur = R(dyaw) @ p_prev + (dx, dy)
        #   p_prev = R(dyaw)^-1 @ (p_cur - (dx, dy)) = R(-dyaw) @ p_cur + t'
        #   where t' = -R(-dyaw) @ (dx, dy)
        # affine_grid/grid_sample work in normalized [-1, 1] coordinates. The
        # grid here is symmetric (xbound[0] == -xbound[1], asserted above) so
        # metric -> normalized is a uniform scale by x_max/y_max; a uniform
        # scale commutes with rotation, so the rotation block of theta is
        # identical to the metric one and only the translation needs rescaling.
        dx, dy, dyaw = ego_transform[:, 0], ego_transform[:, 1], ego_transform[:, 2]
        cos, sin = torch.cos(-dyaw), torch.sin(-dyaw)

        tx = -(cos * dx - sin * dy) / self.x_max
        ty = -(sin * dx + cos * dy) / self.y_max

        theta = torch.stack([
            torch.stack([cos, -sin, tx], dim=-1),
            torch.stack([sin,  cos, ty], dim=-1),
        ], dim=1)  # (B, 2, 3)

        grid = F.affine_grid(theta, fused_t1.shape, align_corners=False)
        # zeros, not border: where the warp reveals area the previous frame
        # never observed, "no information" is the honest signal.
        warped_t1 = F.grid_sample(fused_t1, grid, padding_mode='zeros', align_corners=False)

        return self.fuse(torch.cat([fused_t0, warped_t1], dim=1))
