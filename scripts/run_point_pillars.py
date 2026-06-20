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


def main():
    nusc = NuScenes(version=cfg["data"]["version"], dataroot=cfg["data"]["root"], verbose=False)

    model = PointPillars()
    model.eval()

    sample = nusc.sample[0]
    points = load_lidar(nusc, sample)
    print(f"LiDAR points: {points.shape}")

    with torch.no_grad():
        bev = model(points)

    print(f"bev: {bev.shape}")

    out_path = Path(__file__).parent.parent / "images" / "point_pillars_output.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(bev[0].max(dim=0).values.numpy(), origin='lower', cmap='inferno')
    ax.set_title("PointPillars BEV Output")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
