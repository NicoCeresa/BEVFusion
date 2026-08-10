import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from PIL import Image
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataloader import NuScenesDataset, ATTR_NAMES
from common import (cfg, BEV_H, BEV_W, X_MIN, X_MAX, Y_MIN, Y_MAX, NUM_CLASSES,
                    NUM_ANCHORS, NUM_ATTRS, ATTR_CLASS_RANGES, CLASS_NAMES,
                    anchor_home_classes, generate_anchors, build_model,
                    split_scene_names, default_checkpoint)
from visualize import lidar_height_rgb
from tracker import Tracker

SCORE_THRESH   = 0.3
NMS_IOU_THRESH = 0.3
CLASS_COLORS   = ['#4488ff', '#44ff88', '#ff4444']
NUM_SAMPLES    = 10
TRACK_DT       = 0.5  # nuScenes keyframes are sampled at 2Hz


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_reg(anchors, reg):
    """Inverse of encode_reg from train.py. anchors: (N, 7), reg: (N, 9) → (N, 9) boxes."""
    diag  = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x     = anchors[:, 0] + reg[:, 0] * diag
    y     = anchors[:, 1] + reg[:, 1] * diag
    z     = anchors[:, 2] + reg[:, 2] * anchors[:, 5]
    w     = anchors[:, 3] * torch.exp(reg[:, 3])
    l     = anchors[:, 4] * torch.exp(reg[:, 4])
    h     = anchors[:, 5] * torch.exp(reg[:, 5])
    theta = anchors[:, 6] + torch.arcsin(reg[:, 6].clamp(-1, 1))
    vx    = reg[:, 7]
    vy    = reg[:, 8]
    return torch.stack([x, y, z, w, l, h, theta, vx, vy], dim=1)


def decode_predictions(pred_cls, pred_reg, pred_attr, anchors):
    """
    pred_cls:  (1, A*C, H, W)
    pred_reg:  (1, A*9, H, W)
    pred_attr: (1, A*num_attrs, H, W)
    Returns boxes (N, 9), scores (N,), labels (N,), attrs (N,) — all above SCORE_THRESH.
    """
    cls_scores = torch.sigmoid(pred_cls[0]).permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, NUM_CLASSES)
    reg_preds  = pred_reg[0].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, 9)
    attr_preds = pred_attr[0].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, NUM_ATTRS)

    scores, labels = cls_scores.max(dim=-1)   # (H, W, A)
    keep = scores > SCORE_THRESH

    if not keep.any():
        empty = torch.zeros(0)
        return torch.zeros(0, 9), empty, empty.long(), empty.long()

    anchors_grid  = anchors.view(BEV_H, BEV_W, NUM_ANCHORS, 7)
    kept_anchors  = anchors_grid[keep]
    kept_reg      = reg_preds[keep]
    kept_scores   = scores[keep]
    kept_labels   = labels[keep]
    kept_attr     = attr_preds[keep]  # (K, NUM_ATTRS)

    boxes = decode_reg(kept_anchors, kept_reg)

    # Each anchor's *home* class (its fixed template class, not the predicted
    # label) picks the attribute slice — build_targets only ever supervised
    # the home-class slice for a given anchor slot.
    home_classes = anchor_home_classes(anchors.device).view(BEV_H, BEV_W, NUM_ANCHORS)[keep]
    attrs = torch.zeros(len(kept_scores), dtype=torch.long)
    for cls_label, (start, end) in ATTR_CLASS_RANGES.items():
        mask = home_classes == cls_label
        if mask.any():
            local = kept_attr[mask][:, start:end].argmax(dim=-1)
            attrs[mask] = local + start

    return boxes, kept_scores, kept_labels, attrs


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def iou_bev_pair(boxes, ref_box):
    """Axis-aligned 2D IoU between (N, 7+) boxes and a single (7+,) ref_box (only xy/wl are used)."""
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
    """4 BEV corners of a box (7,) or (9,) → (4, 2) in ego metres."""
    x, y, _, w, l, _, theta = box[:7]
    c, s = np.cos(theta), np.sin(theta)
    corners = np.array([[-w/2, -l/2], [w/2, -l/2], [w/2, l/2], [-w/2, l/2]])
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def render_frame(lidar_pts, pred_boxes, pred_scores, pred_labels, pred_ids, pred_attrs, gt_boxes, gt_labels, sample_idx):
    """Draws one BEV frame and returns it as a PIL Image."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(lidar_height_rgb(lidar_pts), origin='lower',
              extent=(X_MIN, X_MAX, Y_MIN, Y_MAX))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')

    for box, label in zip(gt_boxes.numpy(), gt_labels.numpy()):
        corners = box_corners(box)
        ax.add_patch(Polygon(corners, fill=False, edgecolor='white',
                             linestyle='--', linewidth=1.5))

    for box, score, label, track_id, attr in zip(pred_boxes, pred_scores, pred_labels, pred_ids, pred_attrs):
        corners = box_corners(box)
        color   = CLASS_COLORS[int(label)]
        ax.add_patch(Polygon(corners, fill=False, edgecolor=color, linewidth=2.0))
        ax.text(box[0], box[1], f"{CLASS_NAMES[int(label)]} #{int(track_id)} {score:.2f}\n{ATTR_NAMES[int(attr)]}",
                color=color, fontsize=6, ha='center', va='center')

    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        ax.plot([], [], color=color, linewidth=2, label=name)
    ax.plot([], [], color='white', linestyle='--', linewidth=1.5, label='GT')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'BEVFusion: sample {sample_idx} | pred (solid) vs GT (dashed)')
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = Image.frombuffer('RGBA', canvas.get_width_height(),
                           bytes(canvas.buffer_rgba())).convert('RGB')
    plt.close()
    return img

def test(ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_path is None:
        ckpt_path = default_checkpoint()
    print(f"Checkpoint: {ckpt_path.name}")

    model = build_model(device, ckpt_path)
    anchors = generate_anchors(device)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)

    # Render held-out val scenes — visualizing detections on scenes the model
    # trained on would overstate what it can actually do.
    val_scenes = split_scene_names(nusc, 'val')
    dataset = NuScenesDataset(nusc, scene_names=val_scenes)
    print(f"Rendering from {len(val_scenes)} held-out val scenes ({len(dataset)} samples)")

    images_dir = ROOT / "images"
    images_dir.mkdir(exist_ok=True)

    # Find the first sample that has GT boxes, then run NUM_SAMPLES consecutive
    # frames from there so the ego vehicle moves through the scene sequentially.
    start = next(i for i in range(len(dataset)) if len(dataset[i]['gt_boxes']) > 0)
    indices = range(start, min(start + NUM_SAMPLES, len(dataset)))

    frames = []
    tracker = Tracker(dt=TRACK_DT)

    for idx in indices:
        sample = dataset[idx]

        images        = sample['images'].unsqueeze(0).to(device)
        rots          = sample['rots'].unsqueeze(0).to(device)
        trans         = sample['trans'].unsqueeze(0).to(device)
        intrins       = sample['intrins'].unsqueeze(0).to(device)
        post_rots     = sample['post_rots'].unsqueeze(0).to(device)
        post_trans    = sample['post_trans'].unsqueeze(0).to(device)
        points        = sample['lidar_points'].to(device)
        prev          = {k: (v.unsqueeze(0).to(device) if k != 'lidar_points' else v.to(device))
                         for k, v in sample['prev'].items()}
        ego_transform = sample['ego_transform'].unsqueeze(0).to(device)
        has_prev      = sample['has_prev'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_cls, pred_reg, pred_attr = model(images, rots, trans, intrins, post_rots, post_trans,
                                                   points, prev, ego_transform, has_prev)

        boxes, scores, labels, attrs = decode_predictions(pred_cls, pred_reg, pred_attr, anchors)

        if len(boxes) > 0:
            kept   = nms(boxes, scores)
            boxes  = boxes[kept]
            scores = scores[kept]
            labels = labels[kept]
            attrs  = attrs[kept]

        track_ids, boxes, scores, labels, attrs = tracker.update(
            boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), attrs.cpu().numpy())

        print(f"Sample {idx}: {len(boxes)} tracked detections, {len(sample['gt_boxes'])} GT boxes, max score {scores.max() if len(scores) else 0.0:.3f}")
        frames.append(render_frame(
            lidar_pts   = sample['lidar_points'].cpu(),
            pred_boxes  = boxes,
            pred_scores = scores,
            pred_labels = labels,
            pred_ids    = track_ids,
            pred_attrs  = attrs,
            gt_boxes    = sample['gt_boxes'].cpu(),
            gt_labels   = sample['gt_labels'].cpu(),
            sample_idx  = idx,
        ))


    gif_path = images_dir / f"test_results_{ckpt_path.stem}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  
        loop=0,
    )
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    # Optional second arg overrides SCORE_THRESH — at small training scales the
    # model's confidence on held-out scenes stays well under the 0.3 default,
    # so a lower threshold is needed to visualize what it actually predicts.
    if len(sys.argv) > 2:
        SCORE_THRESH = float(sys.argv[2])
    test(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
