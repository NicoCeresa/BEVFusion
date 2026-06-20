import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fusion.bev_pipeline import BEVFusion

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {
    'xbound': cfg['camera']['xbound'],
    'ybound': cfg['camera']['ybound'],
    'zbound': cfg['camera']['zbound'],
    'dbound': cfg['camera']['dbound'],
}
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_SIZE = (128, 352)


def get_camera_inputs(nusc, sample):
    images, rots, trans, intrins = [], [], [], []
    for cam in CAMERAS:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = Image.open(Path(nusc.dataroot) / cam_data['filename']).resize((IMG_SIZE[1], IMG_SIZE[0]))
        images.append(torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0)
        cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        rots.append(torch.tensor(Quaternion(cs['rotation']).rotation_matrix, dtype=torch.float))
        trans.append(torch.tensor(cs['translation'], dtype=torch.float))
        K = torch.zeros(3, 3)
        K[0, 0] = cs['camera_intrinsic'][0][0]
        K[1, 1] = cs['camera_intrinsic'][1][1]
        K[0, 2] = cs['camera_intrinsic'][0][2]
        K[1, 2] = cs['camera_intrinsic'][1][2]
        K[2, 2] = 1.0
        intrins.append(K)
    B, N = 1, len(CAMERAS)
    return (
        torch.stack(images).unsqueeze(0),
        torch.stack(rots).unsqueeze(0),
        torch.stack(trans).unsqueeze(0),
        torch.stack(intrins).unsqueeze(0),
        torch.eye(3).view(1, 1, 3, 3).expand(B, N, -1, -1),
        torch.zeros(B, N, 3),
    )


def get_lidar_points(nusc, sample):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    path = Path(nusc.dataroot) / lidar_data['filename']
    return torch.tensor(np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4])


def main():
    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)

    model = BEVFusion(
        lss_weights=cfg['weights']['lss'],
        grid_conf=GRID_CONF,
        data_aug_conf={'final_dim': IMG_SIZE},
    )
    model.eval()

    sample = nusc.sample[0]
    cam_inputs = get_camera_inputs(nusc, sample)
    points = get_lidar_points(nusc, sample)

    with torch.no_grad():
        cls, reg = model(*cam_inputs, points)

    print(f"cls: {cls.shape}")
    print(f"reg: {reg.shape}")

    out_path = Path(__file__).parent.parent / "images" / "bevfusion_output.png"

    fig = plt.figure(figsize=(18, 12))

    for i, cam in enumerate(CAMERAS):
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = Image.open(Path(nusc.dataroot) / cam_data['filename'])
        ax = fig.add_subplot(3, 6, i + 1)
        ax.imshow(img)
        ax.set_title(cam.replace('CAM_', ''), fontsize=7)
        ax.axis('off')

    ax_cls = fig.add_subplot(3, 2, 3)
    ax_cls.imshow(cls[0].max(dim=0).values.numpy(), origin='lower', cmap='inferno')
    ax_cls.set_title("Fused BEV — Classification")
    plt.colorbar(ax_cls.images[0], ax=ax_cls)

    ax_reg = fig.add_subplot(3, 2, 4)
    ax_reg.imshow(reg[0, :7].norm(dim=0).numpy(), origin='lower', cmap='viridis')
    ax_reg.set_title("Fused BEV — Regression")
    plt.colorbar(ax_reg.images[0], ax=ax_reg)

    plt.suptitle("BEVFusion Output", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
