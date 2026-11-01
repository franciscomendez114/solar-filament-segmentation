# Label-free calibration of the fused-score threshold for the full-data (train + val) detector ensemble. Why: those
# detectors ran a fully decayed schedule and are 5-12% more confident on test than the split-trained ones, so 0.22 —
# chosen on validation for the split ensemble — would admit more boxes. We pick the threshold whose fused test-box count
# matches the split hv24 ensemble at 0.22, and report how many of the split boxes it still covers (IoU > 0.5).
# Usage (from the repo root): .venv/bin/python scripts/experiments/calibrate_full_threshold.py
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.inference import fuse_boxes

CACHE_DIR = PROJECT_DIR / "data/processed/ensemble_proposals"
SPLIT = ["n", "s", "s_nomosaic", "s1280", "m", "l", "x", "y11m"]
FULL = ["full_yolo26n", "full_yolo26s", "full_yolo26s_nomosaic", "full_yolo26s_1280", "full_yolo26m", "full_yolo26l", "full_yolo26x", "full_yolo11m_flipud"]
views = lambda names: [f"{name}{suffix}" for name in names for suffix in ["", "_hflip", "_vflip"]]


def fused(members):
    caches = [json.loads((CACHE_DIR / f"test_{name}.json").read_text()) for name in members]
    return {stem: fuse_boxes([(np.asarray(c[stem]["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(c[stem]["scores"], dtype=np.float32)) for c in caches],
                             [1.0] * len(members), 0.40, 0.01) for stem in caches[0]}


def coverage(reference, candidate, threshold):
    covered = total = 0
    for stem, (boxes, scores) in reference.items():
        ref = boxes[scores >= 0.22]
        cand = candidate[stem][0][candidate[stem][1] >= threshold]
        total += len(ref)
        if len(ref) and len(cand):
            xywh = lambda b: np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]).astype(np.float64)
            covered += int((mask_utils.iou(xywh(ref), xywh(cand), [0] * len(cand)).max(1) > 0.5).sum())
    return covered / total


split, full = fused(views(SPLIT)), fused(views(FULL))
target = sum(int((s >= 0.22).sum()) for _, s in split.values())
print(f"split hv24 @ 0.22: {target} test boxes")
for threshold in np.arange(0.20, 0.36, 0.01):
    count = sum(int((s >= threshold).sum()) for _, s in full.values())
    print(f"full24 @ {threshold:.2f}: {count} boxes ({count / target - 1:+.1%}), covers {coverage(split, full, threshold):.1%} of split boxes")
