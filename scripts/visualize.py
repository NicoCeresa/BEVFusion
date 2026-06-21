import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backbones.lss_model import compile_model
from lidar.point_pillars import PointPillars
from fusion.bev_pipeline import BEVFusion
from dataloader import NuScenesDataset, CAMERAS

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {
    'xbound': cfg['camera']['xbound'],
    'ybound': cfg['camera']['ybound'],
    'zbound': cfg['camera']['zbound'],
    'dbound': cfg['camera']['dbound'],
}
DATA_AUG_CONF = {'final_dim': (128, 352)}

X_MIN, X_MAX = -50.0, 50.0
Y_MIN, Y_MAX = -50.0, 50.0
BEV_H, BEV_W = 200, 200
Z_MIN, Z_MAX = -3.0, 5.0


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def lidar_height_rgb(points):
    """Height-colored BEV of the raw point cloud. Purple = low, yellow = high."""
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


def save_bev(data, title, path, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, origin='lower', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Per-branch visualizations
# ---------------------------------------------------------------------------

def vis_camera_inputs(nusc, sample, images_dir):
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i, cam in enumerate(CAMERAS):
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = Image.open(Path(nusc.dataroot) / cam_data['filename'])
        axes[i].imshow(img)
        axes[i].set_title(cam.replace('CAM_', ''), fontsize=8)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(images_dir / "camera_inputs.png")
    plt.close()
    print("Saved camera_inputs.png")


def vis_lidar_input(points, images_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lidar_height_rgb(points), origin='lower',
              extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax.set_title("LiDAR Input (height)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(images_dir / "lidar_input.png")
    plt.close()
    print("Saved lidar_input.png")


def vis_lss(cam_inputs, images_dir):
    model = compile_model(GRID_CONF, DATA_AUG_CONF, outC=1)
    model.load_state_dict(torch.load(cfg["weights"]["lss"], map_location="cpu"))
    model.eval()

    with torch.no_grad():
        bev = model(*cam_inputs)

    print(f"LSS output: {bev.shape}")
    save_bev(bev[0, 0].numpy(), "LSS BEV Output", images_dir / "lss_output.png")


def vis_point_pillars(points, images_dir):
    model = PointPillars()
    model.eval()

    with torch.no_grad():
        bev = model(points)

    print(f"PointPillars output: {bev.shape}")
    save_bev(bev[0].max(dim=0).values.numpy(), "PointPillars BEV Output",
             images_dir / "point_pillars_output.png")


def vis_bevfusion(cam_inputs, points, images_dir):
    model = BEVFusion(
        lss_weights   = cfg["weights"]["lss"],
        grid_conf     = GRID_CONF,
        data_aug_conf = DATA_AUG_CONF,
    )
    model.eval()

    with torch.no_grad():
        cls, reg = model(*cam_inputs, points)

    print(f"BEVFusion cls: {cls.shape}  reg: {reg.shape}")
    save_bev(cls[0].max(dim=0).values.numpy(), "Fused BEV Output",
             images_dir / "bevfusion_output.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate BEVFusion pipeline output images.")
    parser.add_argument('--camera',        action='store_true', help='6-camera input grid')
    parser.add_argument('--lidar-input',   action='store_true', dest='lidar_input',
                        help='raw LiDAR point cloud (height-colored BEV)')
    parser.add_argument('--lss',           action='store_true', help='LSS camera BEV output')
    parser.add_argument('--point-pillars', action='store_true', dest='point_pillars',
                        help='PointPillars LiDAR BEV output')
    parser.add_argument('--bevfusion',     action='store_true', help='fused BEV output')
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = not any([args.camera, args.lidar_input, args.lss, args.point_pillars, args.bevfusion])

    nusc   = NuScenes(version=cfg["data"]["version"], dataroot=cfg["data"]["root"], verbose=False)
    sample = nusc.sample[0]

    images_dir = ROOT / "images"
    images_dir.mkdir(exist_ok=True)

    needs_lidar  = run_all or args.lidar_input or args.point_pillars or args.bevfusion
    needs_camera = run_all or args.lss or args.bevfusion

    item = NuScenesDataset(nusc)[0]

    points = item['lidar_points'] if needs_lidar else None
    cam_inputs = (
        item['images'].unsqueeze(0),
        item['rots'].unsqueeze(0),
        item['trans'].unsqueeze(0),
        item['intrins'].unsqueeze(0),
        item['post_rots'].unsqueeze(0),
        item['post_trans'].unsqueeze(0),
    ) if needs_camera else None

    if needs_lidar:
        print(f"LiDAR points: {points.shape}")

    if run_all or args.camera:
        vis_camera_inputs(nusc, sample, images_dir)
    if run_all or args.lidar_input:
        vis_lidar_input(points, images_dir)
    if run_all or args.lss:
        vis_lss(cam_inputs, images_dir)
    if run_all or args.point_pillars:
        vis_point_pillars(points, images_dir)
    if run_all or args.bevfusion:
        vis_bevfusion(cam_inputs, points, images_dir)


if __name__ == "__main__":
    main()
