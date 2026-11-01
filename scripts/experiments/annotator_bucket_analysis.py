# Does fixing mask quality help the weak annotators, or only the strong ones?
# Splits every annotator's filaments into matched / near-miss / never-found and asks what lifting only the
# near-miss bucket over the line would do to each annotator's PQ. If the weak annotators' deficit is near-misses
# the fix narrows the spread; if it is filaments the detector never proposed, the fix leaves them behind.
# Usage (from the repo root): .venv/bin/python scripts/experiments/annotator_bucket_analysis.py
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.metrics import official_match_counts

DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"
OUTPUT = PROJECT_DIR / "experiments/diagnostics/annotator_buckets.csv"
KEEP_RULE = 0.20
TARGET_IOU = 0.60  # what a repaired near-miss would score; the measured mean for masks that already pass is 0.68


def as_rle(value):
    return {"counts": value.encode() if isinstance(value, str) else value["counts"], "size": [2048, 2048]}


rows = []
for line in DUMP.read_text().splitlines():
    if not line.strip():
        continue
    entry = json.loads(line)
    kept = [as_rle(c["rle"]) for c in entry["candidates"] if c["instance_score"] >= KEEP_RULE]
    predicted_areas = np.asarray(mask_utils.area(kept), dtype=np.float64) if kept else np.zeros(0)
    for record, targets in entry["ground_truth"].items():
        targets = [as_rle(t) for t in targets]
        counts = official_match_counts(kept, targets)
        target_areas = np.asarray(mask_utils.area(targets), dtype=np.float64)
        ious = mask_utils.iou(kept, targets, [0] * len(targets)) if kept and targets else np.zeros((0, len(targets)))
        for index in range(len(targets)):
            column = ious[:, index] if len(kept) else np.zeros(1)
            winner = int(column.argmax())
            best = float(column[winner]) if len(kept) else 0.0
            rows.append({"annotator": record.split("-", 1)[0], "best_iou": best,
                         "area_ratio": float(predicted_areas[winner] / max(target_areas[index], 1.0)) if len(kept) else np.nan,
                         **counts})

frame = pd.DataFrame(rows)
frame["bucket"] = np.select([frame["best_iou"] > 0.50, frame["best_iou"] > 0.05], ["matched", "near_miss"], "never_found")

counts = pd.read_csv(PROJECT_DIR / "experiments/diagnostics/annotator_scores.csv")
table = []
for annotator, group in frame.groupby("annotator"):
    row = counts[counts["annotator"].astype(str).str.zfill(6) == annotator]
    if row.empty:
        continue
    tp, fp, fn = int(row["official_tp"].iloc[0]), int(row["official_fp"].iloc[0]), int(row["official_fn"].iloc[0])
    iou_sum, denominator = float(row["official_iou_sum"].iloc[0]), tp + 0.5 * fp + 0.5 * fn
    shares = group["bucket"].value_counts(normalize=True)
    near_miss = int((group["bucket"] == "near_miss").sum())
    ratio = group.loc[group["bucket"] == "near_miss", "area_ratio"]
    table.append({
        "annotator": annotator, "filaments": len(group), "pq": iou_sum / denominator,
        "sq": iou_sum / tp if tp else 0.0, "rq": tp / denominator,
        "matched": shares.get("matched", 0.0), "near_miss": shares.get("near_miss", 0.0),
        "never_found": shares.get("never_found", 0.0),
        "near_miss_area_ratio": float(group.loc[group["bucket"] == "near_miss", "area_ratio"].median()),
        "matched_area_ratio": float(group.loc[group["bucket"] == "matched", "area_ratio"].median()),
        # the denominator is invariant: a repaired near-miss removes one FP and one FN (-0.5 each) and adds one TP
        "near_miss_too_big": float((ratio > 1.5).mean()), "near_miss_too_small": float((ratio < 0.67).mean()),
        "pq_if_repaired": (iou_sum + TARGET_IOU * near_miss) / denominator,
    })

table = pd.DataFrame(table).sort_values("pq").reset_index(drop=True)
table["gain"] = table["pq_if_repaired"] - table["pq"]
table.to_csv(OUTPUT, index=False)

pd.set_option("display.width", 200)
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nspread across annotators   now {table['pq'].std():.4f}   repaired {table['pq_if_repaired'].std():.4f}")
print(f"worst annotator            now {table['pq'].min():.4f}   repaired {table['pq_if_repaired'].min():.4f}")
print(f"best annotator             now {table['pq'].max():.4f}   repaired {table['pq_if_repaired'].max():.4f}")
print(f"\ncorrelation of PQ with near-miss share {table['pq'].corr(table['near_miss']):+.3f}, "
      f"with never-found share {table['pq'].corr(table['never_found']):+.3f}")
print(f"gain vs current PQ correlation {table['gain'].corr(table['pq']):+.3f}  (negative = weak annotators gain more)")
print(f"\nwrote {OUTPUT.relative_to(PROJECT_DIR)}")
