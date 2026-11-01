# Step 1 of the detector-ensemble experiment: cache every Stage 1 detector's low-confidence boxes on the 106 validation
# images, then compare single detectors against fused ensembles at the box level. The box score is only a cheap filter
# for which fusions are worth a full official-PQ run: box mAP climbed with detector size while PQ stayed flat, so a box
# gain here is necessary but not sufficient.
# Round 2 (2026-10-28) adds cheap extra members per detector: mirrored views (flip TTA), a larger inference size, and the
# last-epoch checkpoint, each cached as its own member so any subset can be fused.
# Usage (from the repo root): .venv/bin/python scripts/experiments/detector_ensemble.py
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import ANNOTATIONS_PATH, DEFAULT_PIPELINE_SETTINGS, DETECTOR_MEMBERS, ENSEMBLE_DETECTORS, PROCESSED_DATA_DIR, SPLIT_MANIFEST_PATH, TRAIN_IMAGES_DIR
from src.data import box_iou_with_many, load_records, verify_or_write_split
from src.inference import box_nms, cache_proposals, fuse_boxes

DETECTORS = ENSEMBLE_DETECTORS
MEMBERS = DETECTOR_MEMBERS
CACHE_DIR = PROCESSED_DATA_DIR / "ensemble_proposals"
CACHE_CONFIDENCE = 0.01
THRESHOLDS = np.round(np.arange(0.02, 0.62, 0.02), 2)
OUTPUT = PROJECT_DIR / "experiments/diagnostics/detector_ensemble_boxes.csv"

_, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
records_by_image = defaultdict(list)
for record in validation_records:
    records_by_image[record["physical_id"]].append(record)
paths = sorted(group[0]["image_path"] for group in records_by_image.values())

CACHE_DIR.mkdir(parents=True, exist_ok=True)
for name, (folder, weights, image_size, flip) in MEMBERS.items():
    path = CACHE_DIR / f"validation_{name}.json"
    if path.exists() or not (PROJECT_DIR / f"models/stage1/{folder}/weights/{weights}").exists():
        continue
    print(f"caching {name} ({folder}/{weights} @ {image_size}, flip {flip})", flush=True)
    settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": CACHE_CONFIDENCE, "yolo_image_size": image_size}
    cache = cache_proposals(YOLO(str(PROJECT_DIR / f"models/stage1/{folder}/weights/{weights}")), paths, torch.device("cpu"), settings, flip)
    path.write_text(json.dumps({key: {"boxes": boxes.tolist(), "scores": scores.tolist()} for key, (boxes, scores) in cache.items()}))

proposals = {}
for name in [m for m in MEMBERS if (CACHE_DIR / f"validation_{m}.json").exists()]:
    saved = json.loads((CACHE_DIR / f"validation_{name}.json").read_text())
    proposals[name] = {key: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for key, v in saved.items()}

# The low-confidence cache must reproduce the C0 boxes once filtered at 0.20, otherwise nothing below compares to C0.
reference = json.loads((PROCESSED_DATA_DIR / "validation_proposals_yolo26x.json").read_text())
for key, value in reference.items():
    boxes, scores = proposals["x"][key]
    assert np.allclose(boxes[scores >= 0.20], np.asarray(value["boxes"]).reshape(-1, 4), atol=1e-3), key
print("yolo26x cache filtered at 0.20 matches the C0 proposals", flush=True)

gt_boxes = {record["annotator_image_id"]: np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in record["annotations"]]).reshape(-1, 4)
            for record in validation_records}


def box_curve(image_proposals):
    # Mirrors the official counting with boxes in place of masks: a GT is found when any box has IoU > 0.5 with it.
    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for physical_id, group in records_by_image.items():
            boxes, scores = image_proposals[physical_id]
            boxes = boxes[scores >= threshold]
            for record in group:
                truth = gt_boxes[record["annotator_image_id"]]
                hits = np.array([box_iou_with_many(box, truth) > 0.5 for box in boxes]).reshape(len(boxes), len(truth))
                tp += int(hits.any(0).sum())
                fn += int((~hits.any(0)).sum())
                fp += int((~hits.any(1)).sum())
        rows.append({"threshold": threshold, "box_tp": tp, "box_fp": fp, "box_fn": fn, "box_recall": tp / (tp + fn), "box_f": tp / (tp + 0.5 * fp + 0.5 * fn)})
    return pd.DataFrame(rows)


def fused(members, iou_threshold, post_nms):
    result = {}
    for physical_id in records_by_image:
        boxes, scores = fuse_boxes([proposals[name][physical_id] for name in members], [1.0] * len(members), iou_threshold, CACHE_CONFIDENCE)
        result[physical_id] = box_nms(boxes, scores, DEFAULT_PIPELINE_SETTINGS["yolo_iou"]) if post_nms else (boxes, scores)
    return result


configs = {f"single_{name}": proposals[name] for name in DETECTORS}
for label, members in {"all7": list(DETECTORS), "s1280_m_l_x": ["s1280", "m", "l", "x"], "m_l_x": ["m", "l", "x"], "l_x": ["l", "x"]}.items():
    for iou_threshold in [0.40, 0.55]:
        for post_nms in [True, False]:
            configs[f"wbf_{label}_iou{iou_threshold:.2f}{'_nms' if post_nms else ''}"] = fused(members, iou_threshold, post_nms)

# Round 2: the chosen all7 fusion (IoU 0.40, no NMS) plus each family of extra members, alone and together
base = list(DETECTORS)
families = {variant: [f"{name}_{variant}" for name in DETECTORS if f"{name}_{variant}" in MEMBERS] for variant in ["hflip", "vflip", "up", "last"]}
round2 = {f"r2_all7+{variant}": base + members for variant, members in families.items()}
round2["r2_all7+hflip+vflip"] = base + families["hflip"] + families["vflip"]
round2["r2_all7+hflip+last"] = base + families["hflip"] + families["last"]
round2["r2_all7+hflip+up"] = base + families["hflip"] + families["up"]
round2["r2_everything"] = [m for m in MEMBERS if not m.startswith(("y11m", "y26s_fu"))]
# Round 3: new detectors trained for diversity, added to the 21 plain + flipped views (only those whose caches exist yet)
hv21 = base + families["hflip"] + families["vflip"]
for new in ["y11m", "y26s_fu"]:
    views = [new, f"{new}_hflip", f"{new}_vflip"]
    if all((CACHE_DIR / f"validation_{view}.json").exists() for view in views):
        round2[f"r3_hv21+{new}"] = hv21 + views
        round2[f"r3_all7+{new}"] = base + [new]
# Round 4: high-resolution yolo26s on top of the public-0.39 hv24 set
hv24 = hv21 + ["y11m", "y11m_hflip", "y11m_vflip"]
for new in ["s1536", "s2048"]:
    views = [new, f"{new}_hflip", f"{new}_vflip"]
    if all((CACHE_DIR / f"validation_{view}.json").exists() for view in views):
        round2[f"r4_hv24+{new}"] = hv24 + views
        round2[f"r4_{new}_alone"] = [new]
if "r4_hv24+s1536" in round2 and "r4_hv24+s2048" in round2:
    round2["r4_hv24+both"] = hv24 + [v for new in ["s1536", "s2048"] for v in [new, f"{new}_hflip", f"{new}_vflip"]]
if "r3_hv21+y11m" in round2 and "r3_hv21+y26s_fu" in round2:
    round2["r3_hv21+both"] = hv21 + [v for new in ["y11m", "y26s_fu"] for v in [new, f"{new}_hflip", f"{new}_vflip"]]
for label, members in round2.items():
    for iou_threshold in [0.40, 0.50]:
        configs[f"{label}_iou{iou_threshold:.2f}"] = fused(members, iou_threshold, False)

frames = []
for label, image_proposals in configs.items():
    frames.append(box_curve(image_proposals).assign(config=label))
    print(f"scored {label}", flush=True)
curves = pd.concat(frames, ignore_index=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
curves.to_csv(OUTPUT, index=False)

# Compare each config at its own best threshold, and at the threshold whose false-positive count is closest to C0's
c0 = curves[(curves.config == "single_x") & (curves.threshold == 0.20)].iloc[0]
summary = []
for label, curve in curves.groupby("config", sort=False):
    best = curve.loc[curve.box_f.idxmax()]
    same_fp = curve.loc[(curve.box_fp - c0.box_fp).abs().idxmin()]
    summary.append({"config": label, "best_threshold": best.threshold, "best_box_f": best.box_f, "best_recall": best.box_recall, "best_fp": best.box_fp,
                    "recall_at_c0_fp": same_fp.box_recall, "fp_there": same_fp.box_fp, "threshold_there": same_fp.threshold})
summary = pd.DataFrame(summary).sort_values("best_box_f", ascending=False)
pd.set_option("display.width", 220)
print(f"\nC0 (single_x @ 0.20): box F {c0.box_f:.4f}, recall {c0.box_recall:.4f}, FP {c0.box_fp}")
print(summary.round(4).to_string(index=False))
print(f"\nwrote {OUTPUT.relative_to(PROJECT_DIR)}")
