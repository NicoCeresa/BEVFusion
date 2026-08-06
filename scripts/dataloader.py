import torch
import numpy as np
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
IMG_SIZE = (128, 352)  # H, W — must match LSS pretrained grid

# LIDAR_TOP spins at ~20Hz but nuScenes only annotates the 2Hz keyframes, so a
# single keyframe scan is sparse. Aggregating the preceding sweeps (motion-
# compensated into the keyframe's ego frame) densifies the point cloud — the
# same convention used by CenterPoint/BEVFusion's own nuScenes configs.
NUM_SWEEPS = 10

CLASS_MAP = {
    'vehicle.car':                  0,
    'human.pedestrian.adult':       1,
    'human.pedestrian.child':       1,
    'human.pedestrian.wheelchair':  1,
    'human.pedestrian.stroller':    1,
    'vehicle.bicycle':              2,
}

POINT_CLOUD_RANGE = ((-50.0, 50.0), (-50.0, 50.0), (-5.0, 3.0))


def _scene_is_available(nusc, scene):
    sample = nusc.get('sample', scene['first_sample_token'])
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return (Path(nusc.dataroot) / lidar_sd['filename']).exists()


def available_scene_names(nusc):
    """Scene names whose sensor files are actually present on disk — a local
    copy may only have a subset (e.g. a single trainval blob part)."""
    return {scene['name'] for scene in nusc.scene if _scene_is_available(nusc, scene)}


class NuScenesDataset(Dataset):
    def __init__(self, nusc: NuScenes, scene_names: set = None):
        self.nusc = nusc
        # sample_indices maps dataset-local index -> index into nusc.sample.
        # Restricting to scene_names (e.g. nuScenes' own train/val scene split)
        # keeps a scene's samples entirely on one side — a random *sample*-level
        # split would leak correlated frames from the same drive across both.
        # Defaults to every scene actually present on disk if scene_names is None.
        # Callers that need to cross-reference back to nuScenes (e.g. eval.py
        # building a submission) must go through this mapping, not use the
        # dataset index directly as a nusc.sample index.
        if scene_names is None:
            scene_names = available_scene_names(nusc)
        scene_tokens = {s['token'] for s in nusc.scene if s['name'] in scene_names}
        self.sample_indices = [
            i for i, s in enumerate(nusc.sample) if s['scene_token'] in scene_tokens
        ]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample = self.nusc.sample[self.sample_indices[idx]]

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
            'gt_boxes':     gt_boxes,      # (M, 9) — variable length per sample
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
        # Aggregates the keyframe scan with NUM_SWEEPS-1 preceding sweeps, each
        # motion-compensated into the keyframe's ego frame — see NUM_SWEEPS above.
        pc, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=NUM_SWEEPS)
        return torch.tensor(pc.points.T, dtype=torch.float)  # (P, 4) x,y,z,intensity

    def _load_annotations(self, sample):
        (x_min, x_max), (y_min, y_max), _ = POINT_CLOUD_RANGE

        # nuScenes annotations are in global frame — transform to ego frame
        lidar_data  = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose    = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_t       = np.array(ego_pose['translation'])
        ego_r       = Quaternion(ego_pose['rotation'])

        boxes, labels = [], []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']

            if category not in CLASS_MAP:
                continue

            # global → ego frame
            xyz = ego_r.inverse.rotate(np.array(ann['translation']) - ego_t)
            x, y, z = xyz
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            w, l, h = ann['size']
            yaw = (ego_r.inverse * Quaternion(ann['rotation'])).yaw_pitch_roll[0]

            # global → ego frame (rotation only — velocity is a vector, not a position)
            vel_global = self.nusc.box_velocity(ann_token)
            vx, vy, _ = ego_r.inverse.rotate(np.nan_to_num(vel_global, nan=0.0))

            boxes.append([x, y, z, w, l, h, yaw, vx, vy])
            labels.append(CLASS_MAP[category])

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float),  # (M, 9)
                torch.tensor(labels, dtype=torch.long),  # (M,)
            )
        return torch.zeros(0, 9), torch.zeros(0, dtype=torch.long)


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
        'gt_boxes':     [b['gt_boxes'] for b in batch],                  # list of (M_i, 9)
        'gt_labels':    [b['gt_labels'] for b in batch],                 # list of (M_i,)
    }
