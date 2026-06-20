import torch
import numpy as np
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
IMG_SIZE = (128, 352)  # H, W — must match LSS pretrained grid

CLASS_MAP = {
    'vehicle.car':                  0,
    'human.pedestrian.adult':       1,
    'human.pedestrian.child':       1,
    'human.pedestrian.wheelchair':  1,
    'human.pedestrian.stroller':    1,
    'vehicle.bicycle':              2,
}

POINT_CLOUD_RANGE = ((-50.0, 50.0), (-50.0, 50.0), (-5.0, 3.0))


class NuScenesDataset(Dataset):
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx):
        sample = self.nusc.sample[idx]

        images, rots, trans, intrins, post_rots, post_trans = self._load_cameras(sample)
        lidar_points = self._load_lidar(sample)
        gt_boxes, gt_labels = self._load_annotations(sample)

        return {
            'images':       images,        # (N, 3, H, W)
            'rots':         rots,          # (N, 3, 3)
            'trans':        trans,         # (N, 3)
            'intrins':      intrins,       # (N, 3, 3)
            'post_rots':    post_rots,     # (N, 3, 3)
            'post_trans':   post_trans,    # (N, 3)
            'lidar_points': lidar_points,  # (P, 4) — variable length per sample
            'gt_boxes':     gt_boxes,      # (M, 7) — variable length per sample
            'gt_labels':    gt_labels,     # (M,)
        }

    def _load_cameras(self, sample):
        images, rots, trans, intrins = [], [], [], []

        for cam in CAMERAS:
            cam_data = self.nusc.get('sample_data', sample['data'][cam])
            img = Image.open(Path(self.nusc.dataroot) / cam_data['filename']).resize((IMG_SIZE[1], IMG_SIZE[0]))
            images.append(torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0)

            cs = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            rots.append(torch.tensor(Quaternion(cs['rotation']).rotation_matrix, dtype=torch.float))
            trans.append(torch.tensor(cs['translation'], dtype=torch.float))

            K = torch.zeros(3, 3)
            K[0, 0] = cs['camera_intrinsic'][0][0]
            K[1, 1] = cs['camera_intrinsic'][1][1]
            K[0, 2] = cs['camera_intrinsic'][0][2]
            K[1, 2] = cs['camera_intrinsic'][1][2]
            K[2, 2] = 1.0
            intrins.append(K)

        N = len(CAMERAS)
        return (
            torch.stack(images),                                   # (N, 3, H, W)
            torch.stack(rots),                                     # (N, 3, 3)
            torch.stack(trans),                                    # (N, 3)
            torch.stack(intrins),                                  # (N, 3, 3)
            torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone(),  # (N, 3, 3) post_rots: identity
            torch.zeros(N, 3),                                     # (N, 3)    post_trans: none
        )

    def _load_lidar(self, sample):
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        path = Path(self.nusc.dataroot) / lidar_data['filename']
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        return torch.tensor(points[:, :4])  # (P, 4) x,y,z,intensity

    def _load_annotations(self, sample):
        (x_min, x_max), (y_min, y_max), _ = POINT_CLOUD_RANGE
        boxes, labels = [], []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']

            if category not in CLASS_MAP:
                continue

            x, y, z = ann['translation']
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            w, l, h = ann['size']
            yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]

            boxes.append([x, y, z, w, l, h, yaw])
            labels.append(CLASS_MAP[category])

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float),  # (M, 7)
                torch.tensor(labels, dtype=torch.long),  # (M,)
            )
        return torch.zeros(0, 7), torch.zeros(0, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate for variable-length lidar_points, gt_boxes, gt_labels.
    Fixed-size camera tensors are stacked; variable-length fields are kept as lists.
    """
    return {
        'images':       torch.stack([b['images'] for b in batch]),       # (B, N, 3, H, W)
        'rots':         torch.stack([b['rots'] for b in batch]),         # (B, N, 3, 3)
        'trans':        torch.stack([b['trans'] for b in batch]),        # (B, N, 3)
        'intrins':      torch.stack([b['intrins'] for b in batch]),      # (B, N, 3, 3)
        'post_rots':    torch.stack([b['post_rots'] for b in batch]),    # (B, N, 3, 3)
        'post_trans':   torch.stack([b['post_trans'] for b in batch]),   # (B, N, 3)
        'lidar_points': [b['lidar_points'] for b in batch],              # list of (P_i, 4)
        'gt_boxes':     [b['gt_boxes'] for b in batch],                  # list of (M_i, 7)
        'gt_labels':    [b['gt_labels'] for b in batch],                 # list of (M_i,)
    }
