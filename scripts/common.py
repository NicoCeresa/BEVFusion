"""
Shared model configuration and construction.

These lived in train.py, which meant the inference scripts imported anchor
definitions and model setup from a *training* script, and each script kept its
own copy of the GRID_CONF/DATA_AUG_CONF boilerplate. Both are consolidated
here so config.yaml stays the single source of truth and the anchor geometry
can't drift between training and inference.
"""
import sys
import yaml
import torch
from pathlib import Path
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion.pipeline import BEVFusion, load_checkpoint
from dataloader import available_scene_names

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {k: cfg['camera'][k] for k in ('xbound', 'ybound', 'zbound', 'dbound')}
DATA_AUG_CONF = {'final_dim': tuple(cfg['camera']['image_size'])}

# BEV grid, derived from the same bounds the model and C++ engine use.
X_MIN, X_MAX, X_STEP = cfg['camera']['xbound']
Y_MIN, Y_MAX, Y_STEP = cfg['camera']['ybound']
BEV_W = int((X_MAX - X_MIN) / X_STEP)
BEV_H = int((Y_MAX - Y_MIN) / Y_STEP)
NUM_CLASSES = cfg['lidar']['num_classes']

# Per-class anchors: (class_idx, w, l, h, rotation_rad)
# Each class gets anchors sized to match its typical object dimensions.
ANCHORS = [
    (0, 4.73, 2.08, 1.77, 0.0),     # car, 0°
    (0, 4.73, 2.08, 1.77, 1.5708),  # car, 90°
    (1, 0.76, 0.76, 1.73, 0.0),     # pedestrian (symmetric — 1 rotation)
    (2, 1.76, 0.60, 1.73, 0.0),     # bicycle, 0°
    (2, 1.76, 0.60, 1.73, 1.5708),  # bicycle, 90°
]
NUM_ANCHORS    = len(ANCHORS)
ANCHOR_CLASSES = [a[0] for a in ANCHORS]
ANCHOR_Z       = -1.0

CLASS_NAMES = ['car', 'pedestrian', 'bicycle']


def generate_anchors(device: torch.device) -> torch.Tensor:
    """Returns (BEV_H * BEV_W * NUM_ANCHORS, 7). Ordered (H, W, A) so .view works."""
    xs = torch.linspace(X_MIN, X_MAX, BEV_W + 1)[:-1] + (X_MAX - X_MIN) / BEV_W / 2
    ys = torch.linspace(Y_MIN, Y_MAX, BEV_H + 1)[:-1] + (Y_MAX - Y_MIN) / BEV_H / 2
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    per_anchor = []
    for _, w, l, h, rot in ANCHORS:
        per_anchor.append(torch.stack([
            grid_x,
            grid_y,
            torch.full_like(grid_x, ANCHOR_Z),
            torch.full_like(grid_x, w),
            torch.full_like(grid_x, l),
            torch.full_like(grid_x, h),
            torch.full_like(grid_x, rot),
        ], dim=-1))  # (H, W, 7)

    return torch.stack(per_anchor, dim=2).view(-1, 7).to(device)  # (H*W*A, 7)


def build_model(device, ckpt_path=None, eval_mode=True):
    """Construct BEVFusion and optionally load a checkpoint."""
    model = BEVFusion(
        lss_weights   = cfg['weights']['lss'],
        grid_conf     = GRID_CONF,
        data_aug_conf = DATA_AUG_CONF,
        num_anchors   = NUM_ANCHORS,
    ).to(device)
    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path, device)
    model.eval() if eval_mode else model.train()
    return model


def split_scene_names(nusc, split):
    """nuScenes' own scene split, intersected with what's present on disk.

    'val'/'train' map to the mini equivalents automatically so the same call
    works against either dataset version.
    """
    if nusc.version == 'v1.0-mini' and not split.startswith('mini_'):
        split = f'mini_{split}'
    return set(create_splits_scenes()[split]) & available_scene_names(nusc)
