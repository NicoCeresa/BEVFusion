import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lidar.point_pillars import PointPillars

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def load_lidar(nusc, sample):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    path = Path(nusc.dataroot) / lidar_data['filename']
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return torch.tensor(points[:, :4])  # x, y, z, intensity


X_MIN, X_MAX = -50.0, 50.0
Y_MIN, Y_MAX = -50.0, 50.0
BEV_H, BEV_W = 200, 200
Z_MIN, Z_MAX = -3.0, 5.0


def lidar_input_rgb(points):
    """Height-colored BEV of the raw point cloud (mirrors test.py)."""
    pts = points.numpy()
    mask = (pts[:, 0] >= X_MIN) & (pts[:, 0] < X_MAX) & \
           (pts[:, 1] >= Y_MIN) & (pts[:, 1] < Y_MAX)
    pts = pts[mask]

    xi = ((pts[:, 0] - X_MIN) / (X_MAX - X_MIN) * BEV_W).astype(int).clip(0, BEV_W - 1)
    yi = ((pts[:, 1] - Y_MIN) / (Y_MAX - Y_MIN) * BEV_H).astype(int).clip(0, BEV_H - 1)

    order = np.argsort(pts[:, 2])
    height_map = np.zeros((BEV_H, BEV_W))
    occupied    = np.zeros((BEV_H, BEV_W), dtype=bool)
    height_map[yi[order], xi[order]] = pts[order, 2]
    occupied[yi, xi] = True

    height_norm = np.clip((height_map - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
    rgb = (plt.cm.plasma(height_norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~occupied] = 0
    return rgb


def main():
    nusc = NuScenes(version=cfg["data"]["version"], dataroot=cfg["data"]["root"], verbose=False)

    model = PointPillars()
    model.eval()

    sample = nusc.sample[0]
    points = load_lidar(nusc, sample)
    print(f"LiDAR points: {points.shape}")

    images_dir = Path(__file__).parent.parent / "images"

    # Raw LiDAR input — height-colored BEV
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lidar_input_rgb(points), origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax.set_title("LiDAR Input (height)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(images_dir / "lidar_input.png")
    print(f"Saved lidar_input.png")
    plt.close()

    # PointPillars processed BEV
    with torch.no_grad():
        bev = model(points)
    print(f"bev: {bev.shape}")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(bev[0].max(dim=0).values.numpy(), origin='lower', cmap='inferno')
    ax.set_title("PointPillars BEV Output")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(images_dir / "point_pillars_output.png")
    print(f"Saved point_pillars_output.png")


if __name__ == "__main__":
    main()
