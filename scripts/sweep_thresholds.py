"""
Sweep score-threshold / NMS-IoU post-processing settings against a fixed
checkpoint to find the best combination without retraining.

nuScenes' calc_ap clips precision below 10% to zero (see algo.py::calc_ap),
and this implementation currently submits ~359k predicted boxes against
~21k GT boxes (~6% precision) — below that floor throughout, which is why
mAP rounds to 0.0000 despite the per-class TP errors showing real matches.
Model inference (the expensive part) runs once per validation sample at the
lowest score threshold in the grid; every combination in the grid then only
re-applies thresholding + NMS + DetectionEval against that cached decode.

Usage: python scripts/sweep_thresholds.py [ckpt]
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import test as test_mod
import eval as eval_mod
from dataloader import NuScenesDataset
from common import cfg, generate_anchors, build_model, default_checkpoint

SCORE_GRID = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
NMS_GRID   = [0.1, 0.2, 0.3, 0.5, 0.7]
SCORE_FLOOR = min(SCORE_GRID)


def cache_candidates(model, dataset, nusc, anchors, device):
    """Runs the model once per sample and decodes candidates at SCORE_FLOOR,
    before NMS — everything each grid combination needs, without touching
    the model again."""
    test_mod.SCORE_THRESH = SCORE_FLOOR
    cache = []
    for i in range(len(dataset)):
        idx = dataset.sample_indices[i]
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

        lidar_data = nusc.get('sample_data', nusc.sample[idx]['data']['LIDAR_TOP'])
        ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        cache.append({
            'sample_token': sample_token,
            'boxes': boxes.cpu(), 'scores': scores.cpu(), 'labels': labels.cpu(),
            'ego_t': np.array(ego_pose['translation']),
            'ego_r': eval_mod.Quaternion(ego_pose['rotation']),
        })

        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1}/{len(dataset)}")
    return cache


def build_submission_from_cache(cache, score_thresh, nms_iou, max_boxes_per_sample):
    test_mod.NMS_IOU_THRESH = nms_iou
    results = {}
    for entry in cache:
        boxes, scores, labels = entry['boxes'], entry['scores'], entry['labels']
        keep = scores > score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) > 0:
            kept = eval_mod.fast_nms(boxes, scores)
            boxes, scores, labels = boxes[kept], scores[kept], labels[kept]

        if len(boxes) > max_boxes_per_sample:
            top = scores.argsort(descending=True)[:max_boxes_per_sample]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        entries = []
        for box, score, label in zip(boxes.numpy(), scores.numpy(), labels.numpy()):
            xyz, size, rot, vel = eval_mod.ego_to_global(box, entry['ego_t'], entry['ego_r'])
            entries.append({
                'sample_token':    entry['sample_token'],
                'translation':     xyz.tolist(),
                'size':            size,
                'rotation':        rot.elements.tolist(),
                'velocity':        vel.tolist(),
                'detection_name':  eval_mod.OUR_CLASSES[int(label)],
                'detection_score': float(score),
                'attribute_name':  '',
            })
        results[entry['sample_token']] = entries
    return results


def sweep(ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ckpt_path is None:
        ckpt_path = default_checkpoint()
    print(f"Checkpoint: {ckpt_path.name}")

    model = build_model(device, ckpt_path)
    anchors = generate_anchors(device)

    nusc = NuScenes(version=cfg['data']['version'], dataroot=cfg['data']['root'], verbose=False)
    EVAL_SPLIT, scene_names = eval_mod.resolve_eval_split(nusc)
    dataset = NuScenesDataset(nusc, scene_names=set(scene_names))
    print(f"Evaluating on {len(dataset)} samples from {len(scene_names)} val scenes")

    det_cfg = config_factory('detection_cvpr_2019')

    print(f"Caching model predictions at score>{SCORE_FLOOR} (single forward pass per sample)...")
    cache = cache_candidates(model, dataset, nusc, anchors, device)

    output_dir = ROOT / "eval_results" / "sweep_tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "predictions.json"

    print(f"\nSweeping {len(SCORE_GRID)}x{len(NMS_GRID)} = {len(SCORE_GRID) * len(NMS_GRID)} combinations:")
    rows = []
    for score_thresh in SCORE_GRID:
        for nms_iou in NMS_GRID:
            results = build_submission_from_cache(cache, score_thresh, nms_iou, det_cfg.max_boxes_per_sample)
            n_boxes = sum(len(v) for v in results.values())

            with open(result_path, 'w') as f:
                json.dump({
                    'meta': {'use_camera': True, 'use_lidar': True, 'use_radar': False,
                              'use_map': False, 'use_external': False},
                    'results': results,
                }, f)

            nusc_eval = DetectionEval(nusc, det_cfg, str(result_path), EVAL_SPLIT,
                                      output_dir=str(output_dir), verbose=False)
            metrics, _ = nusc_eval.evaluate()

            aps = {c: metrics.mean_dist_aps[c] for c in eval_mod.OUR_CLASSES}
            mAP = float(np.mean(list(aps.values())))

            tp_scores = []
            for metric_name in ['trans_err', 'scale_err', 'orient_err', 'vel_err']:
                per_class = {c: metrics.get_label_tp(c, metric_name) for c in eval_mod.OUR_CLASSES}
                mean_err = float(np.mean(list(per_class.values())))
                tp_scores.append(max(0.0, 1.0 - mean_err))
            partial_nds = (det_cfg.mean_ap_weight * mAP + sum(tp_scores)) / (det_cfg.mean_ap_weight + len(tp_scores))

            rows.append((score_thresh, nms_iou, mAP, partial_nds, n_boxes))
            print(f"  score>{score_thresh:.2f} nms={nms_iou:.2f}  mAP={mAP:.4f}  PartialNDS={partial_nds:.4f}  boxes={n_boxes}")

    rows.sort(key=lambda r: r[3], reverse=True)
    print("\n" + "=" * 60)
    print("Top 5 by Partial NDS:")
    print("=" * 60)
    for score_thresh, nms_iou, mAP, partial_nds, n_boxes in rows[:5]:
        print(f"  score>{score_thresh:.2f} nms={nms_iou:.2f}  mAP={mAP:.4f}  PartialNDS={partial_nds:.4f}  boxes={n_boxes}")


if __name__ == "__main__":
    sweep(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
