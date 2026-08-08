"""
Evaluate a trained checkpoint on nuScenes using the official devkit metrics.

Reports mAP (real, comparable to published baselines) plus the translation/
scale/orientation/velocity TP errors. mAAE is not reported: this model has no
attribute head, so submitting attribute_name="" would pin that error term
near its worst value regardless of detection quality, deflating NDS for
reasons unrelated to the detector.
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import nms as tv_nms
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import test as test_mod  # reuse decode_predictions/nms — must stay in sync with train.py's encoding
from dataloader import NuScenesDataset
from common import cfg, NUM_ANCHORS, generate_anchors, build_model, split_scene_names



OUR_CLASSES = ['car', 'pedestrian', 'bicycle']

# test.py's SCORE_THRESH (0.3) is tuned for a legible visualization, not for
# eval — thresholding predictions before submission truncates the precision-
# recall curve and understates AP. Relax it here; test.py's own file is untouched.
test_mod.SCORE_THRESH = 0.05


def resolve_eval_split(nusc):
    """Picks the val scenes to evaluate on and registers them as a nuScenes
    custom split (a `splits.json` next to the metadata — the devkit's own
    supported mechanism), restricted to scenes actually present on disk.

    A predefined split name (e.g. 'val') can't be used directly here: on a
    partial dataset copy (a single trainval blob part, say), DetectionEval's
    built-in loader would enumerate every officially-designated val scene
    from the metadata regardless of whether its files exist, and then assert
    that GT and prediction sample tokens match exactly — which fails the
    moment any of those scenes are missing locally. Registering our own
    intersected split sidesteps that.
    """
    split_key = 'mini_val' if nusc.version == 'v1.0-mini' else 'val'
    scene_names = sorted(set(create_splits_scenes()[split_key]) & available_scene_names(nusc))

    splits_path = Path(nusc.dataroot) / nusc.version / 'splits.json'
    with open(splits_path, 'w') as f:
        json.dump({'eval_available': scene_names}, f)

    return 'eval_available', scene_names


def fast_nms(boxes, scores):
    """Vectorized CUDA NMS — same axis-aligned BEV IoU as test.py's nms(), but
    without the Python while-loop's per-iteration CPU/GPU sync (test.py's
    version is fine for its 10-sample visualization use, too slow at eval's
    lower score threshold across the full split)."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 3] / 2
    y1 = boxes[:, 1] - boxes[:, 4] / 2
    x2 = boxes[:, 0] + boxes[:, 3] / 2
    y2 = boxes[:, 1] + boxes[:, 4] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    return tv_nms(boxes_xyxy, scores, test_mod.NMS_IOU_THRESH)


def ego_to_global(box, ego_t, ego_r):
    x, y, z, w, l, h, yaw, vx, vy = box
    xyz = ego_r.rotate(np.array([x, y, z])) + ego_t
    rot = ego_r * Quaternion(axis=[0, 0, 1], angle=float(yaw))
    vel = ego_r.rotate(np.array([vx, vy, 0.0]))[:2]  # vector, not a position — rotate only
    return xyz, [float(w), float(l), float(h)], rot, vel


def build_submission(model, dataset, nusc, anchors, device, max_boxes_per_sample):
    results = {}
    for i in range(len(dataset)):
        idx = dataset.sample_indices[i]  # dataset-local index -> nusc.sample index
        sample_token = nusc.sample[idx]['token']
        sample = dataset[i]

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

        boxes, scores, labels = test_mod.decode_predictions(pred_cls, pred_reg, anchors)
        if len(boxes) > 0:
            kept   = fast_nms(boxes, scores)
            boxes  = boxes[kept]
            scores = scores[kept]
            labels = labels[kept]

        if len(boxes) > max_boxes_per_sample:
            top    = scores.argsort(descending=True)[:max_boxes_per_sample]
            boxes  = boxes[top]
            scores = scores[top]
            labels = labels[top]

        lidar_data = nusc.get('sample_data', nusc.sample[idx]['data']['LIDAR_TOP'])
        ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_t      = np.array(ego_pose['translation'])
        ego_r      = Quaternion(ego_pose['rotation'])

        entries = []
        for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
            xyz, size, rot, vel = ego_to_global(box, ego_t, ego_r)
            entries.append({
                'sample_token':    sample_token,
                'translation':     xyz.tolist(),
                'size':            size,
                'rotation':        rot.elements.tolist(),
                'velocity':        vel.tolist(),
                'detection_name':  OUR_CLASSES[int(label)],
                'detection_score': float(score),
                'attribute_name':  '',
            })
        results[sample_token] = entries

    return results


def evaluate(ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if ckpt_path is None:
        ckpt_dir = ROOT / "checkpoints"
        ckpts = sorted(ckpt_dir.glob(f"*{EPOCHS}*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir} — run train.py first.")
        ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path.name}")

    model = build_model(device, ckpt_path)

    anchors = generate_anchors(device)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    EVAL_SPLIT, scene_names = resolve_eval_split(nusc)
    dataset = NuScenesDataset(nusc, scene_names=set(scene_names))
    print(f"Evaluating on {len(dataset)} samples from {len(scene_names)} val scenes")

    det_cfg = config_factory('detection_cvpr_2019')

    results = build_submission(model, dataset, nusc, anchors, device, det_cfg.max_boxes_per_sample)

    output_dir = ROOT / "eval_results"
    output_dir.mkdir(exist_ok=True)
    result_path = output_dir / f"{ckpt_path.stem}_predictions.json"
    with open(result_path, 'w') as f:
        json.dump({
            'meta': {
                'use_camera': True, 'use_lidar': True, 'use_radar': False,
                'use_map': False, 'use_external': False,
            },
            'results': results,
        }, f)
    print(f"Saved submission to {result_path}")

    nusc_eval = DetectionEval(nusc, det_cfg, str(result_path), EVAL_SPLIT,
                              output_dir=str(output_dir), verbose=True)
    metrics, _ = nusc_eval.evaluate()

    print("\n" + "=" * 60)
    print(f"Results — {ckpt_path.name} on {EVAL_SPLIT} ({len(dataset)} samples)")
    print("=" * 60)

    print("\nAP by class (averaged over dist thresholds 0.5/1/2/4m):")
    aps = {c: metrics.mean_dist_aps[c] for c in OUR_CLASSES}
    for c, ap in aps.items():
        print(f"  {c:12s} {ap:.4f}")
    mAP = float(np.mean(list(aps.values())))
    print(f"  {'mAP':12s} {mAP:.4f}")

    print("\nTP errors at dist_th_tp=2m (lower is better):")
    tp_scores = []
    for metric_name, label in [('trans_err', 'ATE (m)'), ('scale_err', 'ASE (1-IOU)'),
                                ('orient_err', 'AOE (rad)'), ('vel_err', 'AVE (m/s)')]:
        per_class = {c: metrics.get_label_tp(c, metric_name) for c in OUR_CLASSES}
        mean_err  = float(np.mean(list(per_class.values())))
        tp_scores.append(max(0.0, 1.0 - mean_err))
        print(f"  {label:14s} " + ", ".join(f"{c}={v:.3f}" for c, v in per_class.items()) + f"  (mean={mean_err:.3f})")
    print("  AAE:           N/A — model has no attribute head")

    partial_nds = (det_cfg.mean_ap_weight * mAP + sum(tp_scores)) / (det_cfg.mean_ap_weight + len(tp_scores))
    print(f"\nPartial NDS (mAP + ATE/ASE/AOE/AVE, excludes AAE): {partial_nds:.4f}")
    print("(Not directly comparable to published full-NDS numbers, which include the attribute term.)")


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
