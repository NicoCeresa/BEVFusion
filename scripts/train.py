import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fusion.pipeline import BEVFusion
from dataloader import NuScenesDataset, collate_fn

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
EPOCHS = 2

# Per-class anchors: (class_idx, w, l, h, rotation_rad)
# Each class gets anchors sized to match its typical object dimensions.
ANCHORS = [
    (0, 4.73, 2.08, 1.77, 0.0),     # car, 0°
    (0, 4.73, 2.08, 1.77, 1.5708),  # car, 90°
    (1, 0.76, 0.76, 1.73, 0.0),     # pedestrian (symmetric — 1 rotation)
    (2, 1.76, 0.60, 1.73, 0.0),     # bicycle, 0°
    (2, 1.76, 0.60, 1.73, 1.5708),  # bicycle, 90°
]
NUM_ANCHORS    = len(ANCHORS)        # 5
ANCHOR_CLASSES = [a[0] for a in ANCHORS]   # [0, 0, 1, 2, 2]
ANCHOR_Z       = -1.0

POS_IOU_THRESH = 0.50
NEG_IOU_THRESH = 0.35


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def generate_anchors(device: torch.device) -> torch.Tensor:
    """Returns (BEV_H * BEV_W * NUM_ANCHORS, 7). Ordered (H, W, A) so .view works."""
    xs = torch.linspace(X_MIN, X_MAX, BEV_W + 1)[:-1] + (X_MAX - X_MIN) / BEV_W / 2
    ys = torch.linspace(Y_MIN, Y_MAX, BEV_H + 1)[:-1] + (Y_MAX - Y_MIN) / BEV_H / 2
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    per_anchor = []
    for _, w, l, h, rot in ANCHORS:
        per_anchor.append(torch.stack([
            grid_x,
            grid_y,
            torch.full_like(grid_x, ANCHOR_Z),
            torch.full_like(grid_x, w),
            torch.full_like(grid_x, l),
            torch.full_like(grid_x, h),
            torch.full_like(grid_x, rot),
        ], dim=-1))  # (H, W, 7)

    return torch.stack(per_anchor, dim=2).view(-1, 7).to(device)  # (H*W*A, 7)


# ---------------------------------------------------------------------------
# IoU + regression encoding
# ---------------------------------------------------------------------------

def iou_bev(anchors: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """Axis-aligned 2D IoU between (N, 7) anchors and a single (7,) GT box."""
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
    """PointPillars-style regression encoding. anchors: (N, 7), gt_box: (7,) → (N, 7)."""
    diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    return torch.stack([
        (gt_box[0] - anchors[:, 0]) / diag,
        (gt_box[1] - anchors[:, 1]) / diag,
        (gt_box[2] - anchors[:, 2]) / anchors[:, 5],
        torch.log(gt_box[3] / anchors[:, 3]),
        torch.log(gt_box[4] / anchors[:, 4]),
        torch.log(gt_box[5] / anchors[:, 5]),
        torch.sin(gt_box[6] - anchors[:, 6]),
    ], dim=1)


# ---------------------------------------------------------------------------
# Target assignment
# ---------------------------------------------------------------------------

def build_targets(anchors, gt_boxes, gt_labels, device):
    """
    Match GT boxes to anchors via IoU and build training targets.

    Returns:
        cls_targets (H, W, A, C)  — binary per-class labels
        reg_targets (H, W, A, 7)  — encoded box deltas (only valid at pos anchors)
        pos_mask    (H, W, A)     — True where anchor matched a GT box
        loss_mask   (H, W, A)     — True for pos + neg anchors (ignore ambiguous)
    """
    N = BEV_H * BEV_W * NUM_ANCHORS
    cls_targets = torch.zeros(N, NUM_CLASSES, device=device)
    reg_targets = torch.zeros(N, 7, device=device)
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
        reg_targets.view(BEV_H, BEV_W, NUM_ANCHORS, 7),
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
        reg_pred = pred_reg[b].permute(1, 2, 0).view(BEV_H, BEV_W, NUM_ANCHORS, 7)

        cls_loss_total += focal_loss(cls_pred, cls_targets)[loss_mask].sum()

        if pos_mask.any():
            reg_loss_total += F.smooth_l1_loss(
                reg_pred[pos_mask], reg_targets[pos_mask], reduction='sum'
            )
            num_pos += pos_mask.sum().item()

    norm = max(num_pos, 1)
    return cls_loss_total / norm, reg_loss_total / norm


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    nusc    = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    dataset = NuScenesDataset(nusc)

    n_val   = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = BEVFusion(
        lss_weights   = cfg['weights']['lss'],
        grid_conf     = GRID_CONF,
        data_aug_conf = DATA_AUG_CONF,
        num_anchors   = NUM_ANCHORS,
    ).to(device)

    anchors   = generate_anchors(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    

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
        print(f"  |  val cls {v_cls/m:.4f}  reg {v_reg/m:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"bevfusion_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), ckpt_dir / f"bevfusion_{EPOCHS}_epochs.pt")
    print(f"Saved final checkpoint → {ckpt_dir}")


if __name__ == "__main__":
    train()
