import yaml
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

# Single source of truth for input resolution — it also determines the LSS
# frustum shape, so it has to agree across the dataloader, the model, and the
# C++ engine. Keeping separate copies is how they drift out of sync.
with open(Path(__file__).parent.parent / "config.yaml") as _f:
    IMG_SIZE = tuple(yaml.safe_load(_f)['camera']['image_size'])  # H, W

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

# nuScenes ties valid attributes to detection class (see
# nuscenes/eval/detection/algo.py::detection_name_to_rel_attributes) — car's
# category maps only to vehicle.*, pedestrian's only to pedestrian.*,
# bicycle's only to cycle.*, with no overlap. Ordered class-contiguously
# (car=[0:3), pedestrian=[3:6), bicycle=[6:8)) so a class's attribute range is
# a fixed slice — see common.py's ATTR_CLASS_RANGES, which depends on this
# exact ordering.
ATTR_NAMES = [
    'vehicle.moving', 'vehicle.parked', 'vehicle.stopped',
    'pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing',
    'cycle.with_rider', 'cycle.without_rider',
]
ATTR_MAP = {name: i for i, name in enumerate(ATTR_NAMES)}

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
        gt_boxes, gt_labels, gt_attrs = self._load_annotations(sample)

        # Temporal fusion needs one prior keyframe, motion-compensated into
        # the current ego frame. At a scene's first sample there's no prior
        # keyframe (sample['prev'] == '') — reuse the current frame as its
        # own "prev" with a zero transform rather than skipping the sample
        # or feeding zeros: a real (duplicated) BEV keeps the temporal
        # fusion module's BatchNorm statistics in-distribution, and this way
        # every caller (train/eval/sweep/visualize) gets identical, free
        # scene-start behavior instead of reimplementing this per script.
        has_prev = sample['prev'] != ''
        if has_prev:
            prev_sample = self.nusc.get('sample', sample['prev'])
            (prev_images, prev_rots, prev_trans,
             prev_intrins, prev_post_rots, prev_post_trans) = self._load_cameras(prev_sample)
            prev_lidar_points = self._load_lidar(prev_sample)

            ego_t_cur,  ego_r_cur  = self._ego_pose(sample)
            ego_t_prev, ego_r_prev = self._ego_pose(prev_sample)
            # Same idiom _load_annotations uses for global->ego (rotate the
            # translation difference, compose the rotations) — just applied
            # to two ego poses instead of an ego pose and an annotation.
            dx, dy = ego_r_cur.inverse.rotate(ego_t_prev - ego_t_cur)[:2]
            dyaw = (ego_r_cur.inverse * ego_r_prev).yaw_pitch_roll[0]
            ego_transform = torch.tensor([dx, dy, dyaw], dtype=torch.float)
        else:
            (prev_images, prev_rots, prev_trans,
             prev_intrins, prev_post_rots, prev_post_trans) = images, rots, trans, intrins, post_rots, post_trans
            prev_lidar_points = lidar_points
            ego_transform = torch.zeros(3)

        return {
            'images':       images,        # (N, 3, H, W)
            'rots':         rots,          # (N, 3, 3)
            'trans':        trans,         # (N, 3)
            'intrins':      intrins,       # (N, 3, 3)
            'post_rots':    post_rots,     # (N, 3, 3)
            'post_trans':   post_trans,    # (N, 3)
            'lidar_points': lidar_points,  # (P, 4) — variable length per sample
            'prev': {
                'images':       prev_images,
                'rots':         prev_rots,
                'trans':        prev_trans,
                'intrins':      prev_intrins,
                'post_rots':    prev_post_rots,
                'post_trans':   prev_post_trans,
                'lidar_points': prev_lidar_points,
            },
            'ego_transform': ego_transform,   # (3,) — dx, dy, dyaw: prev ego frame relative to current
            'has_prev':      torch.tensor(has_prev),
            'gt_boxes':     gt_boxes,      # (M, 9) — variable length per sample
            'gt_labels':    gt_labels,     # (M,)
            'gt_attrs':     gt_attrs,      # (M,) — -1 where no attribute annotation exists
        }

    def _ego_pose(self, sample):
        """Global-frame ego pose (translation, rotation) at a sample's LIDAR_TOP
        timestamp — used only for the temporal-fusion ego-motion transform above;
        _load_annotations fetches this itself for its own (unrelated) purpose."""
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        return np.array(ego_pose['translation']), Quaternion(ego_pose['rotation'])

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

        boxes, labels, attrs = [], [], []

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

            # ~0.4% of annotations have zero attribute_tokens (nuScenes docs);
            # -1 is an ignore sentinel, matching how the devkit's own attr_acc
            # ignores empty ground-truth attributes rather than scoring them wrong.
            attr_tokens = ann['attribute_tokens']
            if attr_tokens:
                attr_name = self.nusc.get('attribute', attr_tokens[0])['name']
                attrs.append(ATTR_MAP.get(attr_name, -1))
            else:
                attrs.append(-1)

        if boxes:
            return (
                torch.tensor(boxes, dtype=torch.float),  # (M, 9)
                torch.tensor(labels, dtype=torch.long),  # (M,)
                torch.tensor(attrs, dtype=torch.long),   # (M,)
            )
        return torch.zeros(0, 9), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate for variable-length lidar_points, gt_boxes, gt_labels, gt_attrs.
    Fixed-size camera/prev/ego_transform tensors are stacked; variable-length fields are kept as lists.
    """
    return {
        'images':       torch.stack([b['images'] for b in batch]),       # (B, N, 3, H, W)
        'rots':         torch.stack([b['rots'] for b in batch]),         # (B, N, 3, 3)
        'trans':        torch.stack([b['trans'] for b in batch]),        # (B, N, 3)
        'intrins':      torch.stack([b['intrins'] for b in batch]),      # (B, N, 3, 3)
        'post_rots':    torch.stack([b['post_rots'] for b in batch]),    # (B, N, 3, 3)
        'post_trans':   torch.stack([b['post_trans'] for b in batch]),   # (B, N, 3)
        'lidar_points': [b['lidar_points'] for b in batch],              # list of (P_i, 4)
        'prev': {
            'images':       torch.stack([b['prev']['images'] for b in batch]),
            'rots':         torch.stack([b['prev']['rots'] for b in batch]),
            'trans':        torch.stack([b['prev']['trans'] for b in batch]),
            'intrins':      torch.stack([b['prev']['intrins'] for b in batch]),
            'post_rots':    torch.stack([b['prev']['post_rots'] for b in batch]),
            'post_trans':   torch.stack([b['prev']['post_trans'] for b in batch]),
            'lidar_points': [b['prev']['lidar_points'] for b in batch],
        },
        'ego_transform': torch.stack([b['ego_transform'] for b in batch]),  # (B, 3)
        'has_prev':      torch.stack([b['has_prev'] for b in batch]),       # (B,)
        'gt_boxes':     [b['gt_boxes'] for b in batch],                  # list of (M_i, 9)
        'gt_labels':    [b['gt_labels'] for b in batch],                 # list of (M_i,)
        'gt_attrs':     [b['gt_attrs'] for b in batch],                  # list of (M_i,)
    }
