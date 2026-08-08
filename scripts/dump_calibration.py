"""
Dump INT8 calibration batches — one file per (engine, sample).

TensorRT derives INT8 scale factors by observing real activation ranges, so
each sub-model needs representative inputs. Those inputs are intermediate
tensors (pooled BEV, pillar features, fused BEV, ...), so they're produced by
running the PyTorch pipeline rather than read off disk.

Samples are strided across the val split, not taken consecutively: adjacent
nuScenes frames are 0.5s apart and nearly identical, which would give TensorRT
a misleadingly narrow view of the activation range.

Usage: python scripts/dump_calibration.py [num_samples]
"""
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion.pipeline import BEVFusion
from dataloader import NuScenesDataset, available_scene_names
from train import NUM_ANCHORS

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {k: cfg['camera'][k] for k in ('xbound', 'ybound', 'zbound', 'dbound')}
DATA_AUG_CONF = {'final_dim': (128, 352)}
CALIB_DIR = ROOT / "data" / "calib"

MAX_PILLARS, MAX_PTS = 10000, 32


def write(path, tensor):
    np.asarray(tensor, dtype=np.float32).ravel().tofile(path)


def pillar_inputs(lidar_encoder, points):
    """Reproduces PointPillars.forward up to the pointnet call, returning the
    padded pillar tensor and the scattered grid the backbone consumes."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = lidar_encoder.point_cloud_range
    mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] < z_max))
    pts = points[mask]

    from lidar.pillarize import (discretize_point_cloud, get_pillar_centers,
                                 augment_pillars, optimize_pillars)
    idx = discretize_point_cloud(pts, lidar_encoder.voxel_size, lidar_encoder.point_cloud_range)
    centers = get_pillar_centers(idx, lidar_encoder.voxel_size, lidar_encoder.point_cloud_range)
    clusters = lidar_encoder._cluster_centers(pts, idx)
    augmented = augment_pillars(pts, centers, clusters)
    pillars, unique_idx = optimize_pillars(augmented, idx, lidar_encoder.max_points_per_pillar)

    # Pad to the fixed shape the exported engine expects.
    padded = torch.zeros(MAX_PILLARS, MAX_PTS, 9, device=pillars.device, dtype=pillars.dtype)
    n = min(pillars.shape[0], MAX_PILLARS)
    padded[:n] = pillars[:n]

    encoded = lidar_encoder.pointnet(pillars)
    scattered = lidar_encoder._scatter(encoded, unique_idx).unsqueeze(0)
    return padded, scattered


def main(num_samples=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    split_key = 'mini_val' if nusc.version == 'v1.0-mini' else 'val'
    scenes = set(create_splits_scenes()[split_key]) & available_scene_names(nusc)
    dataset = NuScenesDataset(nusc, scene_names=scenes)

    stride = max(1, len(dataset) // num_samples)
    indices = [i * stride for i in range(num_samples) if i * stride < len(dataset)]
    print(f"Calibrating on {len(indices)} samples strided across {len(dataset)} val samples")

    model = BEVFusion(lss_weights=cfg['weights']['lss'], grid_conf=GRID_CONF,
                      data_aug_conf=DATA_AUG_CONF, num_anchors=NUM_ANCHORS).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "bevfusion_epoch10.pt", map_location=device))
    model.eval()
    cam = model.camera_encoder

    for n, idx in enumerate(indices):
        s = dataset[idx]
        images = s['images'].unsqueeze(0).to(device)
        args = [s[k].unsqueeze(0).to(device) for k in
                ('rots', 'trans', 'intrins', 'post_rots', 'post_trans')]
        points = s['lidar_points'].to(device)

        with torch.no_grad():
            write(CALIB_DIR / f"cam_encode_{n}.bin", s['images'].numpy())

            geom = cam.get_geometry(*args)
            pooled = cam.voxel_pooling(geom, cam.get_cam_feats(images))
            write(CALIB_DIR / f"bev_encode_{n}.bin", pooled.cpu().numpy())

            camera_bev = cam.bevencode(pooled)

            padded, scattered = pillar_inputs(model.lidar_encoder, points)
            write(CALIB_DIR / f"pointnet_{n}.bin", padded.cpu().numpy())
            write(CALIB_DIR / f"pillar_backbone_{n}.bin", scattered.cpu().numpy())

            lidar_bev = model.lidar_encoder.backbone(scattered)
            # bev_encoder takes two inputs; the calibrator reads them in the
            # engine's declared input order.
            write(CALIB_DIR / f"bev_encoder_{n}_0.bin", camera_bev.cpu().numpy())
            write(CALIB_DIR / f"bev_encoder_{n}_1.bin", lidar_bev.cpu().numpy())

            fused = model.bev_encoder(camera_bev, lidar_bev)
            write(CALIB_DIR / f"ssd_{n}.bin", fused.cpu().numpy())

        print(f"  sample {n + 1}/{len(indices)} (dataset index {idx})")

    total_mb = sum(f.stat().st_size for f in CALIB_DIR.glob("*.bin")) / 1e6
    print(f"\nWrote {len(list(CALIB_DIR.glob('*.bin')))} files ({total_mb:.0f} MB) to {CALIB_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
