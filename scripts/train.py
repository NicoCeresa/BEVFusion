import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataloader import NuScenesDataset, collate_fn, CLASS_MAP
from common import (cfg, BEV_H, BEV_W, NUM_CLASSES, NUM_ANCHORS, ANCHOR_CLASSES,
                    generate_anchors, build_model, split_scene_names)

EPOCHS = 15
# Epochs of no val-loss improvement before stopping; None disables.
# Off by default: on the 15-epoch trainval01 run, val loss was a poor proxy for
# detection quality — it bottomed at epoch 1 (total) / epoch 4 (cls), while of
# the checkpoints actually scored with eval.py, epoch 10 beat epoch 15
# (partial NDS 0.0517 vs 0.0278). Patience=5 would have stopped at epoch 6 and
# never produced that checkpoint. Enable only if you've confirmed val loss
# tracks mAP on your data.
EARLY_STOP_PATIENCE = None

POS_IOU_THRESH = 0.50
NEG_IOU_THRESH = 0.35

# car/pedestrian are close in annotation count (~51%/46% of train instances);
# bicycle is the real outlier (~3%, ~17x rarer than car). CBGS below groups by
# per-sample class *presence*, matching the original paper's method — but
# bicycle turns out to be instance-sparse rather than frame-sparse here (the
# same few bicycles stay in view across many consecutive frames, so ~37% of
# samples contain one despite there being few distinct bicycle objects
# overall), so CBGS's frame-level oversampling has little effect on it. This
# weight is doing most of the actual correction instead.
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 6.0])  # car, pedestrian, bicycle


# ---------------------------------------------------------------------------
# IoU + regression encoding
# ---------------------------------------------------------------------------

def iou_bev(anchors: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """Axis-aligned 2D IoU between (N, 7) anchors and a single (9,) GT box (only xy/wl are used)."""
    ax1 = anchors[:, 0] - anchors[:, 3] / 2
    ax2 = anchors[:, 0] + anchors[:, 3] / 2
    ay1 = anchors[:, 1] - anchors[:, 4] / 2
    ay2 = anchors[:, 1] + anchors[:, 4] / 2

    gx1 = gt_box[0] - gt_box[3] / 2
    gx2 = gt_box[0] + gt_box[3] / 2
    gy1 = gt_box[1] - gt_box[4] / 2
    gy2 = gt_box[1] + gt_box[4] / 2

    inter = (torch.min(ax2, gx2) - torch.max(ax1, gx1)).clamp(0) * \
            (torch.min(ay2, gy2) - torch.max(ay1, gy1)).clamp(0)
    union = anchors[:, 3] * anchors[:, 4] + gt_box[3] * gt_box[4] - inter
    return inter / (union + 1e-6)


def encode_reg(anchors: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """PointPillars-style regression encoding. anchors: (N, 7), gt_box: (9,) → (N, 9).

    Velocity (gt_box[7:9]) has no anchor prior to encode relative to — anchors
    are static templates — so it's carried through as a direct regression target.
    """
    diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    N = anchors.shape[0]
    return torch.stack([
        (gt_box[0] - anchors[:, 0]) / diag,
        (gt_box[1] - anchors[:, 1]) / diag,
        (gt_box[2] - anchors[:, 2]) / anchors[:, 5],
        torch.log(gt_box[3] / anchors[:, 3]),
        torch.log(gt_box[4] / anchors[:, 4]),
        torch.log(gt_box[5] / anchors[:, 5]),
        torch.sin(gt_box[6] - anchors[:, 6]),
        gt_box[7].expand(N),
        gt_box[8].expand(N),
    ], dim=1)


# ---------------------------------------------------------------------------
# Target assignment
# ---------------------------------------------------------------------------

def build_targets(anchors, gt_boxes, gt_labels, device):
    """
    Match GT boxes to anchors via IoU and build training targets.

    Returns:
        cls_targets (H, W, A, C)  — binary per-class labels
        reg_targets (H, W, A, 9)  — encoded box deltas (only valid at pos anchors)
        pos_mask    (H, W, A)     — True where anchor matched a GT box
        loss_mask   (H, W, A)     — True for pos + neg anchors (ignore ambiguous)
    """
    N = BEV_H * BEV_W * NUM_ANCHORS
    cls_targets = torch.zeros(N, NUM_CLASSES, device=device)
    reg_targets = torch.zeros(N, 9, device=device)
    pos_mask    = torch.zeros(N, dtype=torch.bool, device=device)
    neg_mask    = torch.ones(N, dtype=torch.bool, device=device)

    # (N,) — which class each anchor slot belongs to
    anchor_classes = torch.tensor(ANCHOR_CLASSES * (BEV_H * BEV_W), device=device)

    for box, label in zip(gt_boxes, gt_labels):
        box = box.to(device)
        cls_label = label.item()

        # Only compute IoU against anchors of the matching class
        class_mask = anchor_classes == cls_label
        ious = torch.zeros(N, device=device)
        ious[class_mask] = iou_bev(anchors[class_mask], box)

        pos = ious >= POS_IOU_THRESH
        # Force-assign the best same-class anchor even if below threshold
        best = class_mask.nonzero(as_tuple=True)[0][ious[class_mask].argmax()]
        pos[best] = True

        neg_mask[ious >= NEG_IOU_THRESH] = False

        cls_targets[pos, cls_label] = 1.0
        reg_targets[pos] = encode_reg(anchors[pos], box)
        pos_mask[pos] = True
        neg_mask[pos] = False

    return (
        cls_targets.view(BEV_H, BEV_W, NUM_ANCHORS, NUM_CLASSES),
        reg_targets.view(BEV_H, BEV_W, NUM_ANCHORS, 9),
        pos_mask.view(BEV_H, BEV_W, NUM_ANCHORS),
        (pos_mask | neg_mask).view(BEV_H, BEV_W, NUM_ANCHORS),
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Sigmoid focal loss — implemented manually to avoid torchvision dependency."""
    ce    = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t   = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return alpha_t * (1 - p_t) ** gamma * ce


def compute_loss(pred_cls, pred_reg, gt_boxes_batch, gt_labels_batch, anchors):
    """
    pred_cls: (B, A*C, H, W)
    pred_reg: (B, A*7, H, W)
    gt_boxes_batch, gt_labels_batch: lists of length B
    """
    device = pred_cls.device
    B = pred_cls.shape[0]
    cls_loss_total = torch.tensor(0.0, device=device)
    reg_loss_total = torch.tensor(0.0, device=device)
    num_pos = 0

    for b in range(B):
        cls_targets, reg_targets, pos_mask, loss_mask = build_targets(
            anchors, gt_boxes_batch[b], gt_labels_batch[b], device
        )

        # (A*C, H, W) → (H, W, A, C)
        cls_pred = pred_cls[b].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, NUM_CLASSES)
        reg_pred = pred_reg[b].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, 9)

        cls_loss_total += (focal_loss(cls_pred, cls_targets) * CLASS_WEIGHTS.to(device))[loss_mask].sum()

        if pos_mask.any():
            reg_loss_total += F.smooth_l1_loss(
                reg_pred[pos_mask], reg_targets[pos_mask], reduction='sum'
            )
            num_pos += pos_mask.sum().item()

    norm = max(num_pos, 1)
    return cls_loss_total / norm, reg_loss_total / norm


# ---------------------------------------------------------------------------
# CBGS-style class-balanced sampling
# ---------------------------------------------------------------------------

def compute_cbgs_weights(dataset, nusc, num_classes):
    """Class-Balanced Grouping and Sampling (Zhu et al., 2019) — the scheme
    CenterPoint/BEVFusion's own nuScenes configs use for class imbalance.
    Each sample's repeat factor is driven by the rarest class it contains
    (by per-sample presence, matching the original method), so frames with
    a class present in a smaller fraction of samples get drawn more often.
    On this dataset that ends up mattering little for bicycle specifically —
    see the CLASS_WEIGHTS comment above — but is still a faithful
    reproduction of the paper's approach. Returns per-sample weights for
    WeightedRandomSampler, which achieves the same effect as literally
    duplicating rare-class samples, without materializing a longer index list.
    """
    sample_classes = []
    class_sample_count = [0] * num_classes
    for i in range(len(dataset)):
        sample = nusc.sample[dataset.sample_indices[i]]
        present = {CLASS_MAP[nusc.get('sample_annotation', t)['category_name']]
                   for t in sample['anns']
                   if nusc.get('sample_annotation', t)['category_name'] in CLASS_MAP}
        sample_classes.append(present)
        for c in present:
            class_sample_count[c] += 1

    n = len(dataset)
    frac = [count / n for count in class_sample_count]
    target = 1.0 / num_classes
    ratios = [target / f if f > 0 else 1.0 for f in frac]

    return [max((ratios[c] for c in present), default=1.0) for present in sample_classes]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)

    # Split by scene (nuScenes' own train/val scene assignment, restricted to
    # whatever's actually on disk) rather than by individual sample — a random
    # sample-level split would leak correlated frames from the same drive
    # across both sets.
    train_set = NuScenesDataset(nusc, scene_names=split_scene_names(nusc, 'train'))
    val_set   = NuScenesDataset(nusc, scene_names=split_scene_names(nusc, 'val'))

    # CBGS oversampling on train only — val must stay representative of the
    # true distribution for an honest read on generalization.
    cbgs_weights = compute_cbgs_weights(train_set, nusc, NUM_CLASSES)
    train_sampler = WeightedRandomSampler(cbgs_weights, num_samples=len(train_set), replacement=True)

    num_workers  = os.cpu_count()
    train_loader = DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = build_model(device, eval_mode=False)

    anchors   = generate_anchors(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    epochs_since_improvement = 0

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        t_cls = t_reg = 0.0

        for batch in tqdm(train_loader, desc="train", leave=False):

            images     = batch['images'].to(device)
            rots       = batch['rots'].to(device)
            trans      = batch['trans'].to(device)
            intrins    = batch['intrins'].to(device)
            post_rots  = batch['post_rots'].to(device)
            post_trans = batch['post_trans'].to(device)
            points     = batch['lidar_points'][0].to(device)   # single sample (batch_size=1)
            gt_boxes   = batch['gt_boxes']
            gt_labels  = batch['gt_labels']

            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                pred_cls, pred_reg = model(images, rots, trans, intrins, post_rots, post_trans, points)
                cls_loss, reg_loss = compute_loss(pred_cls, pred_reg, gt_boxes, gt_labels, anchors)
                loss = cls_loss + reg_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_cls += cls_loss.item()
            t_reg += reg_loss.item()

        n = len(train_loader)
        print(f"Epoch {epoch:3d} | train cls {t_cls/n:.4f}  reg {t_reg/n:.4f}", end="")

        model.eval()
        v_cls = v_reg = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images     = batch['images'].to(device)
                rots       = batch['rots'].to(device)
                trans      = batch['trans'].to(device)
                intrins    = batch['intrins'].to(device)
                post_rots  = batch['post_rots'].to(device)
                post_trans = batch['post_trans'].to(device)
                points     = batch['lidar_points'][0].to(device)
                gt_boxes   = batch['gt_boxes']
                gt_labels  = batch['gt_labels']

                pred_cls, pred_reg = model(images, rots, trans, intrins, post_rots, post_trans, points)
                cls_loss, reg_loss = compute_loss(pred_cls, pred_reg, gt_boxes, gt_labels, anchors)
                v_cls += cls_loss.item()
                v_reg += reg_loss.item()

        m = len(val_loader)
        val_total = v_cls / m + v_reg / m
        print(f"  |  val cls {v_cls/m:.4f}  reg {v_reg/m:.4f}  total {val_total:.4f}", end="")

        # Overfitting sets in well before the final epoch on this dataset, so
        # the last checkpoint is not the one you want — track the best.
        if val_total < best_val:
            best_val, best_epoch, epochs_since_improvement = val_total, epoch, 0
            torch.save(model.state_dict(), ckpt_dir / "bevfusion_best.pt")
            print("  * best")
        else:
            epochs_since_improvement += 1
            print("")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"bevfusion_epoch{epoch+1}.pt")

        if EARLY_STOP_PATIENCE and epochs_since_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stop: no val improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    torch.save(model.state_dict(), ckpt_dir / f"bevfusion_{EPOCHS}_epochs.pt")
    print(f"Saved final checkpoint → {ckpt_dir}")
    print(f"Best val loss {best_val:.4f} at epoch {best_epoch} → bevfusion_best.pt")
    print("Note: val loss is a proxy — confirm the pick with scripts/eval.py before shipping one.")


if __name__ == "__main__":
    train()
