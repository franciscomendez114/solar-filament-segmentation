# Mask-quality head. Instead of a fixed YOLO confidence cutoff, keep a candidate mask when its expected PQ
# contribution beats what it costs: adding a mask always adds 0.5 to the PQ denominator and on average adds p*q to
# the numerator (p = chance it matches some annotator, q = the IoU it would score), so keep it when p*q > PQ/2.
# Phase 1 "dump" (GPU): every candidate the pipeline produces at confidence 0.05, with features and its true p*q.
# Phase 2 "fit" (CPU): grouped cross-validation inside the validation images, so the reported gain is out-of-fold.
# Usage: python scripts/experiments/quality_head.py dump [--smoke]
#        python scripts/experiments/quality_head.py fit [--smoke]
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import DEFAULT_PIPELINE_SETTINGS
from src.data import load_records, verify_or_write_split
from src.experiments import append_registry
from src.inference import predict_instances
from src.metrics import aggregate_official, annotation_to_rle, bootstrap_pq, official_match_counts, paired_pq_delta

PHASE = sys.argv[1]
SMOKE = "--smoke" in sys.argv
OUTPUT_ROOT = PROJECT_DIR / "colab_transfer/smoke" if SMOKE else PROJECT_DIR
TRAIN_IMAGES_DIR = PROJECT_DIR / "data/raw/train/train_images"
ANNOTATIONS_PATH = PROJECT_DIR / "data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
DETECTOR_PATH = PROJECT_DIR / "models/stage1/yolo26l_filament_detection/weights/best.pt"
UNET_PATH = PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt"
DUMP_PATH = OUTPUT_ROOT / "experiments/diagnostics/quality_candidates.jsonl"
MASK_THRESHOLD = 0.45
DUMP_CONFIDENCE = 0.05  # the head replaces the 0.20 cutoff, so candidates are generated well below it
CONTROL_PQ = 0.4315  # fold 0: saved B5 + yolo26l at confidence 0.20, the rule the head has to beat
FEATURES = ["yolo_confidence", "mean_probability", "instance_score", "area", "probability_p25", "probability_p50", "probability_p75",
            "probability_p90", "crop_side", "box_height", "box_width", "fill_ratio", "elongation", "radius", "neighbours", "rank"]

coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, _, validation_records = verify_or_write_split(records, PROJECT_DIR / "data/processed/split_manifest.json")
groups = defaultdict(list)
for record in validation_records:
    groups[record["physical_id"]].append(record)
if SMOKE:
    groups = dict(list(groups.items())[:6])

if PHASE == "dump":
    from ultralytics import YOLO

    from src.models import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}", flush=True)
    detector, (unet, _) = YOLO(str(DETECTOR_PATH)), load_checkpoint(UNET_PATH, "b5", device)
    settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": DUMP_CONFIDENCE}
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    start, total = time.time(), 0
    with DUMP_PATH.open("w") as file:
        for physical_id, group in groups.items():
            candidates, _ = predict_instances(group[0]["image_path"], detector, unet, device, MASK_THRESHOLD, settings, return_candidates=True)
            targets = {r["annotator_image_id"]: [annotation_to_rle(a, r["height"], r["width"]) for a in r["annotations"]] for r in group}
            rles = [mask_utils.encode(np.asfortranarray(candidate.pop("mask"))) for candidate in candidates]
            for candidate, rle in zip(candidates, rles):
                best = [float(mask_utils.iou([rle], target, [0] * len(target)).max()) if target else 0.0 for target in targets.values()]
                candidate["best_iou_per_record"] = best
                candidate["match_probability"] = float(np.mean([iou > 0.50 for iou in best]))
                candidate["value"] = float(np.mean([iou if iou > 0.50 else 0.0 for iou in best]))  # the p*q this mask is worth
                candidate["rle"] = rle["counts"].decode("ascii")
            file.write(json.dumps({"physical_id": physical_id, "records": list(targets), "ground_truth": {k: [r["counts"].decode("ascii") for r in v] for k, v in targets.items()},
                                   "candidates": candidates}) + "\n")
            total += len(candidates)
    print(f"{total} candidates from {len(groups)} images ({total / max(len(groups), 1):.1f} per image) in {(time.time() - start) / 60:.1f} min -> {DUMP_PATH}", flush=True)
    sys.exit()

# fit: everything below is CPU only and re-scores saved masks, so rules can be compared without re-running the U-Net
entries = [json.loads(line) for line in DUMP_PATH.read_text().splitlines() if line.strip()]
frame = pd.DataFrame([{**{key: candidate[key] for key in FEATURES}, "physical_id": entry["physical_id"], "index": position,
                       "value": candidate["value"], "match_probability": candidate["match_probability"]}
                      for entry in entries for position, candidate in enumerate(entry["candidates"])])
print(f"{len(frame)} candidates from {frame['physical_id'].nunique()} images; {(frame['value'] > 0).mean():.1%} match some annotator")

folds = min(5, frame["physical_id"].nunique())
frame["predicted_value"] = np.nan
for train_index, test_index in GroupKFold(n_splits=folds).split(frame[FEATURES], frame["value"], frame["physical_id"]):
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=42)
    model.fit(frame.iloc[train_index][FEATURES], frame.iloc[train_index]["value"])
    frame.loc[frame.index[test_index], "predicted_value"] = model.predict(frame.iloc[test_index][FEATURES]).clip(0, 1)
print(f"Out-of-fold correlation with the true value: {frame['predicted_value'].corr(frame['value']):.3f}")


def score_rule(keep_column, cutoff):
    # Keep the chosen masks, then resolve overlaps greedily by instance score, exactly as the pipeline ranks them
    rows = []
    for entry in entries:
        selected = [(position, candidate) for position, candidate in enumerate(entry["candidates"])
                    if frame.loc[(frame["physical_id"] == entry["physical_id"]) & (frame["index"] == position), keep_column].item() > cutoff]
        selected.sort(key=lambda item: item[1]["instance_score"], reverse=True)
        claimed, kept = np.zeros((2048, 2048), dtype=bool), []
        for _, candidate in selected:
            mask = mask_utils.decode({"size": [2048, 2048], "counts": candidate["rle"].encode("ascii")}).astype(bool) & ~claimed
            if mask.sum() >= DEFAULT_PIPELINE_SETTINGS["minimum_mask_area"]:
                claimed |= mask
                kept.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
        for record, target_rles in entry["ground_truth"].items():
            targets = [{"size": [2048, 2048], "counts": counts.encode("ascii")} for counts in target_rles]
            rows.append({"physical_id": entry["physical_id"], "annotator_image_id": record, "kept": len(kept), **official_match_counts(kept, targets)})
    table = pd.DataFrame(rows)
    return table, {**aggregate_official(table), **bootstrap_pq(table), "instances_per_image": table.groupby("physical_id")["kept"].first().mean()}


results, comparison = {}, []
rules = [("yolo_confidence", 0.20, "current rule: confidence 0.20")] + [("predicted_value", round(multiplier * CONTROL_PQ / 2, 4), f"head: p*q > {multiplier:g} x PQ/2")
                                                                        for multiplier in ([1.0] if SMOKE else [0.6, 0.8, 1.0, 1.2, 1.5])]
for column, cutoff, label in rules:
    table, summary = score_rule(column, cutoff)
    results[label] = table
    comparison.append({"rule": label, "cutoff": cutoff, **{key: summary[key] for key in ["official_pq", "official_sq", "official_rq", "official_tp", "official_fp", "official_fn", "instances_per_image"]}})
    print(f"{label}: PQ {summary['official_pq']:.4f}, TP {summary['official_tp']}, FP {summary['official_fp']}, FN {summary['official_fn']}", flush=True)

control_label = rules[0][2]
for row in comparison[1:]:
    row.update(paired_pq_delta(results[control_label], results[row["rule"]]))
comparison = pd.DataFrame(comparison)
run_dir = OUTPUT_ROOT / "experiments/pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S_quality_head")
run_dir.mkdir(parents=True, exist_ok=False)
comparison.to_csv(run_dir / "comparison.csv", index=False)
frame.to_csv(run_dir / "oof_predictions.csv", index=False)
(run_dir / "settings.json").write_text(json.dumps({"detector": str(DETECTOR_PATH.relative_to(PROJECT_DIR)), "unet": str(UNET_PATH.relative_to(PROJECT_DIR)),
                                                   "mask_threshold": MASK_THRESHOLD, "dump_confidence": DUMP_CONFIDENCE, "control_pq": CONTROL_PQ,
                                                   "features": FEATURES, "folds": folds, "pipeline": DEFAULT_PIPELINE_SETTINGS}, indent=2) + "\n")
print(comparison.round(4).to_string(index=False))
print(f"Saved to {run_dir}")
