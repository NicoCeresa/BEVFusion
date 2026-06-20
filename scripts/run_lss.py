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
from backbones.lss_model import compile_model

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {
    'xbound': cfg['camera']['xbound'],
    'ybound': cfg['camera']['ybound'],
    'zbound': cfg['camera']['zbound'],
    'dbound': cfg['camera']['dbound'],
}
DATA_AUG_CONF = {'final_dim': (128, 352)}

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
IMG_SIZE = (128, 352)  # H, W


def get_sample_inputs(nusc, sample):
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
        torch.stack(images).unsqueeze(0),          # (1, N, 3, H, W)
        torch.stack(rots).unsqueeze(0),             # (1, N, 3, 3)
        torch.stack(trans).unsqueeze(0),            # (1, N, 3)
        torch.stack(intrins).unsqueeze(0),          # (1, N, 3, 3)
        torch.eye(3).view(1, 1, 3, 3).expand(B, N, -1, -1),  # post_rots: identity
        torch.zeros(B, N, 3),                       # post_trans: none
    )


def main():
    nusc = NuScenes(version=cfg["data"]["version"], dataroot=cfg["data"]["root"], verbose=False)

    model = compile_model(GRID_CONF, DATA_AUG_CONF, outC=1)
    model.load_state_dict(torch.load(cfg["weights"]["lss"], map_location="cpu"))
    model.eval()

    sample = nusc.sample[0]
    inputs = get_sample_inputs(nusc, sample)

    with torch.no_grad():
        bev = model(*inputs)

    print(f"BEV output shape: {bev.shape}")

    images_dir = Path(__file__).parent.parent / "images"

    # save camera inputs once as a shared reference image
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i, cam in enumerate(CAMERAS):
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img = Image.open(Path(nusc.dataroot) / cam_data['filename'])
        axes[i].imshow(img)
        axes[i].set_title(cam.replace('CAM_', ''), fontsize=8)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(images_dir / "camera_inputs.png")
    print(f"Saved camera_inputs.png")
    plt.close()

    # save LSS BEV standalone
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(bev[0, 0].numpy(), origin='lower', cmap='inferno')
    ax.set_title("LSS BEV Output")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(images_dir / "lss_output.png")
    print(f"Saved lss_output.png")


if __name__ == "__main__":
    main()
