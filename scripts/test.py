import io
import sys
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

from fusion.pipeline import BEVFusion
from dataloader import NuScenesDataset
from train import EPOCHS, NUM_ANCHORS, generate_anchors
from visualize import lidar_height_rgb

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

GRID_CONF = {
    'xbound': cfg['camera']['xbound'],
    'ybound': cfg['camera']['ybound'],
    'zbound': cfg['camera']['zbound'],
    'dbound': cfg['camera']['dbound'],
}
DATA_AUG_CONF = {'final_dim': (128, 352)}

BEV_H, BEV_W = 200, 200
X_MIN, X_MAX  = -50.0, 50.0
Y_MIN, Y_MAX  = -50.0, 50.0
NUM_CLASSES   = 3

SCORE_THRESH   = 0.3
NMS_IOU_THRESH = 0.3
CLASS_NAMES    = ['car', 'pedestrian', 'bicycle']
CLASS_COLORS   = ['#4488ff', '#44ff88', '#ff4444']
NUM_SAMPLES    = 10


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_reg(anchors, reg):
    """Inverse of encode_reg from train.py. anchors, reg: (N, 7) → (N, 7) boxes."""
    diag  = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x     = anchors[:, 0] + reg[:, 0] * diag
    y     = anchors[:, 1] + reg[:, 1] * diag
    z     = anchors[:, 2] + reg[:, 2] * anchors[:, 5]
    w     = anchors[:, 3] * torch.exp(reg[:, 3])
    l     = anchors[:, 4] * torch.exp(reg[:, 4])
    h     = anchors[:, 5] * torch.exp(reg[:, 5])
    theta = anchors[:, 6] + torch.arcsin(reg[:, 6].clamp(-1, 1))
    return torch.stack([x, y, z, w, l, h, theta], dim=1)


def decode_predictions(pred_cls, pred_reg, anchors):
    """
    pred_cls: (1, A*C, H, W)
    pred_reg: (1, A*7, H, W)
    Returns boxes (N, 7), scores (N,), labels (N,) — all above SCORE_THRESH.
    """
    cls_scores = torch.sigmoid(pred_cls[0]).permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, NUM_CLASSES)
    reg_preds  = pred_reg[0].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, 7)

    scores, labels = cls_scores.max(dim=-1)   # (H, W, A)
    keep = scores > SCORE_THRESH

    if not keep.any():
        empty = torch.zeros(0)
        return torch.zeros(0, 7), empty, empty.long()

    anchors_grid  = anchors.view(BEV_H, BEV_W, NUM_ANCHORS, 7)
    kept_anchors  = anchors_grid[keep]
    kept_reg      = reg_preds[keep]
    kept_scores   = scores[keep]
    kept_labels   = labels[keep]

    boxes = decode_reg(kept_anchors, kept_reg)
    return boxes, kept_scores, kept_labels


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def iou_bev_pair(boxes, ref_box):
    """Axis-aligned 2D IoU between (N, 7) boxes and a single (7,) ref_box."""
    ax1 = boxes[:, 0] - boxes[:, 3] / 2;  ax2 = boxes[:, 0] + boxes[:, 3] / 2
    ay1 = boxes[:, 1] - boxes[:, 4] / 2;  ay2 = boxes[:, 1] + boxes[:, 4] / 2
    gx1 = ref_box[0] - ref_box[3] / 2;    gx2 = ref_box[0] + ref_box[3] / 2
    gy1 = ref_box[1] - ref_box[4] / 2;    gy2 = ref_box[1] + ref_box[4] / 2

    inter = (torch.min(ax2, gx2) - torch.max(ax1, gx1)).clamp(0) * \
            (torch.min(ay2, gy2) - torch.max(ay1, gy1)).clamp(0)
    union = boxes[:, 3] * boxes[:, 4] + ref_box[3] * ref_box[4] - inter
    return inter / (union + 1e-6)


def nms(boxes, scores):
    """Greedy NMS. Returns a list of kept indices."""
    if len(boxes) == 0:
        return []

    order = scores.argsort(descending=True)
    kept = []

    while order.numel() > 0:
        i = order[0].item()
        kept.append(i)
        if order.numel() == 1:
            break
        ious  = iou_bev_pair(boxes[order[1:]], boxes[i])
        order = order[1:][ious < NMS_IOU_THRESH]

    return kept

def box_corners(box):
    """4 BEV corners of a box (7,) → (4, 2) in ego metres."""
    x, y, _, w, l, _, theta = box
    c, s = np.cos(theta), np.sin(theta)
    corners = np.array([[-w/2, -l/2], [w/2, -l/2], [w/2, l/2], [-w/2, l/2]])
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def render_frame(lidar_pts, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, sample_idx):
    """Draws one BEV frame and returns it as a PIL Image."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(lidar_height_rgb(lidar_pts), origin='lower',
              extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')

    for box, label in zip(gt_boxes.numpy(), gt_labels.numpy()):
        corners = box_corners(box)
        ax.add_patch(plt.Polygon(corners, fill=False, edgecolor='white',
                                 linestyle='--', linewidth=1.5))

    for box, score, label in zip(pred_boxes.numpy(), pred_scores.numpy(), pred_labels.numpy()):
        corners = box_corners(box)
        color   = CLASS_COLORS[int(label)]
        ax.add_patch(plt.Polygon(corners, fill=False, edgecolor=color, linewidth=2.0))
        ax.text(box[0], box[1], f"{CLASS_NAMES[int(label)]} {score:.2f}",
                color=color, fontsize=6, ha='center', va='center')

    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        ax.plot([], [], color=color, linewidth=2, label=name)
    ax.plot([], [], color='white', linestyle='--', linewidth=1.5, label='GT')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'BEVFusion — sample {sample_idx} | pred (solid) vs GT (dashed)')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = ROOT / "checkpoints"
    ckpts = sorted(ckpt_dir.glob(f"*{EPOCHS}*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir} — run train.py first.")
    ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path.name}")

    model = BEVFusion(
        lss_weights   = cfg['weights']['lss'],
        grid_conf     = GRID_CONF,
        data_aug_conf = DATA_AUG_CONF,
        num_anchors   = NUM_ANCHORS,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    anchors = generate_anchors(device)

    nusc    = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    dataset = NuScenesDataset(nusc)

    images_dir = ROOT / "images"
    images_dir.mkdir(exist_ok=True)

    frames = []

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]

        images     = sample['images'].unsqueeze(0).to(device)
        rots       = sample['rots'].unsqueeze(0).to(device)
        trans      = sample['trans'].unsqueeze(0).to(device)
        intrins    = sample['intrins'].unsqueeze(0).to(device)
        post_rots  = sample['post_rots'].unsqueeze(0).to(device)
        post_trans = sample['post_trans'].unsqueeze(0).to(device)
        points     = sample['lidar_points'].to(device)

        with torch.no_grad():
            pred_cls, pred_reg = model(images, rots, trans, intrins,
                                       post_rots, post_trans, points)

        boxes, scores, labels = decode_predictions(pred_cls, pred_reg, anchors)

        if len(boxes) > 0:
            kept   = nms(boxes, scores)
            boxes  = boxes[kept]
            scores = scores[kept]
            labels = labels[kept]

        print(f"Sample {idx}: {len(boxes)} detections")
        frames.append(render_frame(
            lidar_pts   = sample['lidar_points'].cpu(),
            pred_boxes  = boxes.cpu(),
            pred_scores = scores.cpu(),
            pred_labels = labels.cpu(),
            gt_boxes    = sample['gt_boxes'].cpu(),
            gt_labels   = sample['gt_labels'].cpu(),
            sample_idx  = idx,
        ))


    gif_path = images_dir / f"test_results_{EPOCHS}_epochs.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  
        loop=0,
    )
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    test()
