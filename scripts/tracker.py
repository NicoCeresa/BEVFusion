"""
Lightweight SORT-style multi-object tracker for BEVFusion detections.

Standard SORT fits a Kalman filter per track to estimate velocity from box
history. This model already regresses vx/vy directly per detection, so
track prediction just extrapolates each track's own last known velocity
instead — no filter to fit. IoU-based Hungarian assignment and the
hit/age track lifecycle are otherwise standard SORT.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MATCH_THRESH = 0.1
MAX_AGE  = 3   # frames a track survives with no matching detection
MIN_HITS = 2   # matches needed before a track is shown, to suppress one-frame flicker


def iou_bev_matrix(boxes_a, boxes_b):
    """Axis-aligned BEV IoU between every pair of (N, 7+) and (M, 7+) boxes -> (N, M)."""
    ax1 = boxes_a[:, 0:1] - boxes_a[:, 3:4] / 2;  ax2 = boxes_a[:, 0:1] + boxes_a[:, 3:4] / 2
    ay1 = boxes_a[:, 1:2] - boxes_a[:, 4:5] / 2;  ay2 = boxes_a[:, 1:2] + boxes_a[:, 4:5] / 2
    bx1 = boxes_b[:, 0] - boxes_b[:, 3] / 2;      bx2 = boxes_b[:, 0] + boxes_b[:, 3] / 2
    by1 = boxes_b[:, 1] - boxes_b[:, 4] / 2;      by2 = boxes_b[:, 1] + boxes_b[:, 4] / 2

    inter = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * \
            np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4])[:, None]
    area_b = boxes_b[:, 3] * boxes_b[:, 4]
    return inter / (area_a + area_b - inter + 1e-6)


class Track:
    _next_id = 1

    def __init__(self, box, score, label):
        self.id = Track._next_id
        Track._next_id += 1
        self.box   = box.copy()  # (9,) x,y,z,w,l,h,yaw,vx,vy
        self.score = score
        self.label = label
        self.hits  = 1
        self.time_since_update = 0

    def predict(self, dt):
        """Extrapolate position using the track's own last known velocity."""
        predicted = self.box.copy()
        predicted[0] += self.box[7] * dt
        predicted[1] += self.box[8] * dt
        return predicted

    def update(self, box, score, label):
        self.box, self.score, self.label = box.copy(), score, label
        self.hits += 1
        self.time_since_update = 0


class Tracker:
    """Call update() once per frame, in sample order, with that frame's
    post-NMS detections."""

    def __init__(self, dt):
        self.dt = dt
        self.tracks = []

    def update(self, boxes, scores, labels):
        """
        boxes: (N, 9) numpy array, scores: (N,), labels: (N,) — one frame's
        detections. Returns (track_ids, boxes, scores, labels) for confirmed
        tracks only (hits >= MIN_HITS), all numpy arrays.
        """
        predicted = np.stack([t.predict(self.dt) for t in self.tracks]) if self.tracks else np.zeros((0, 9))

        matched_tracks, matched_dets = set(), set()
        if len(self.tracks) > 0 and len(boxes) > 0:
            iou = iou_bev_matrix(predicted, boxes)
            row_ind, col_ind = linear_sum_assignment(-iou)
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] >= IOU_MATCH_THRESH:
                    self.tracks[r].update(boxes[c], scores[c], labels[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.time_since_update += 1

        for j in range(len(boxes)):
            if j not in matched_dets:
                self.tracks.append(Track(boxes[j], scores[j], labels[j]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE]

        confirmed = [t for t in self.tracks if t.time_since_update == 0 and t.hits >= MIN_HITS]
        if not confirmed:
            return (np.zeros(0, dtype=int), np.zeros((0, 9)), np.zeros(0), np.zeros(0, dtype=int))
        return (
            np.array([t.id for t in confirmed]),
            np.stack([t.box for t in confirmed]),
            np.array([t.score for t in confirmed]),
            np.array([t.label for t in confirmed]),
        )
