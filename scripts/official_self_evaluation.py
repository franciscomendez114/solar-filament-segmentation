# Runs the organizers' own evaluation code (scripts/official_eval.py, vendored from their Self Evaluation
# notebook) on our validation predictions. The rubric scores 70% quantitative on FIVE things, only one of
# which is PQ: mean Dice/PQ, the Dice distribution, the IoU distribution, and the one-to-many / many-to-one
# relation distribution. We have only ever measured PQ, so this produces the other four.
# Usage (from the repo root): .venv/bin/python scripts/official_self_evaluation.py [PREDICTIONS_JSON OUTPUT_DIR]
# With no arguments it scores the C0 prediction set; PREDICTIONS_JSON ({physical_id: [RLE counts]}, written by a
# stage2_gpu_job.py "dump" config) scores another prediction set against the same ground truth, into OUTPUT_DIR.
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from official_eval import get_overlap_df, get_pq_score, plot_distribution, plot_m2n_counts
from src.metrics import aggregate_official, official_match_counts

DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"
OUTPUT_DIR = PROJECT_DIR / (sys.argv[2] if len(sys.argv) > 2 else "experiments/diagnostics/official_self_evaluation")
SAVED = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else None
KEEP_RULE = 0.20  # reproduces the C0 prediction set


def build_frames():
    ground_truth, predictions = [], []
    for line in DUMP.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        kept = SAVED[entry["physical_id"]] if SAVED is not None else [c["rle"] for c in entry["candidates"] if c["instance_score"] >= KEEP_RULE]
        for number, rle in enumerate(kept, 1):
            predictions.append({"filament_id": f"{entry['physical_id']}_{number}", "segmentation_rle": rle})
        for record, rles in entry["ground_truth"].items():
            for number, rle in enumerate(rles, 1):
                ground_truth.append({"filament_id": f"{record}_{number}", "segmentation_rle": rle})
    return pd.DataFrame(ground_truth), pd.DataFrame(predictions)


gt_df, pred_df = build_frames()
print(f"{len(gt_df)} ground-truth filaments, {len(pred_df)} predicted, "
      f"{gt_df['filament_id'].str.split('_').str[0].nunique()} annotator-images")

overlap_df = get_overlap_df(gt_df, pred_df)
official_pq = get_pq_score(overlap_df)

# our own scorer on the same inputs, as a cross-check that src/metrics.py matches theirs
rows = []
for row in overlap_df.itertuples(index=False):
    matrix = np.asarray(row.iou_matrix, dtype=np.float64)
    hits = matrix > 0.50
    rows.append({"official_tp": int(hits.sum()), "official_fp": int((~hits.any(axis=0)).sum()),
                 "official_fn": int((~hits.any(axis=1)).sum()), "official_iou_sum": float(matrix[hits].sum())})
ours = aggregate_official(pd.DataFrame(rows))

print(f"\nofficial notebook PQ : {official_pq:.6f}")
print(f"src/metrics.py PQ    : {ours['official_pq']:.6f}")
print(f"difference           : {abs(official_pq - ours['official_pq']):.2e}")
print(f"TP {ours['official_tp']}, FP {ours['official_fp']}, FN {ours['official_fn']}")

ious = np.concatenate([np.asarray(m, dtype=float).ravel() for m in overlap_df["iou_matrix"]])
dices = np.concatenate([np.asarray(m, dtype=float).ravel() for m in overlap_df["dice_matrix"]])
print(f"\nIoU  over overlapping pairs: mean {ious[ious > 0].mean():.4f}, median {np.median(ious[ious > 0]):.4f}")
print(f"Dice over overlapping pairs: mean {dices[dices > 0].mean():.4f}, median {np.median(dices[dices > 0]):.4f}")

# one-to-many and many-to-one: a GT touched by several predictions is fragmentation, a prediction touching
# several GT filaments is over-merging. The committee looks at these distributions explicitly.
gt_links, pred_links = [], []
for row in overlap_df.itertuples(index=False):
    matrix = np.asarray(row.iou_matrix, dtype=float) > 0
    if matrix.size:
        gt_links.extend(matrix.sum(axis=1).tolist())
        pred_links.extend(matrix.sum(axis=0).tolist())
gt_links, pred_links = np.array(gt_links), np.array(pred_links)
print(f"\nfragmentation  (GT filaments touched by >1 prediction): {np.mean(gt_links > 1):.1%}")
print(f"over-merging   (predictions touching >1 GT filament):   {np.mean(pred_links > 1):.1%}")
print(f"GT filaments touched by nothing:                        {np.mean(gt_links == 0):.1%}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for name, series, title, label in [("iou_distribution", overlap_df["iou_matrix"], "IoU Distribution", "IoU"),
                                   ("dice_distribution", overlap_df["dice_matrix"], "Dice Distribution", "Dice")]:
    plt.figure(figsize=(7, 4))
    plot_distribution(series, title, label)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=110); plt.close()
plt.figure(figsize=(7, 4))
plot_m2n_counts(overlap_df)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "many_to_one.png", dpi=110); plt.close()

summary = {"official_pq": official_pq, "our_pq": ours["official_pq"], **{k: int(v) for k, v in ours.items() if k.startswith("official_") and k != "official_pq" and k != "official_iou_sum"},
           "iou_mean_overlapping": float(ious[ious > 0].mean()), "dice_mean_overlapping": float(dices[dices > 0].mean()),
           "fragmentation_rate": float(np.mean(gt_links > 1)), "over_merge_rate": float(np.mean(pred_links > 1))}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"\nplots and summary.json in {OUTPUT_DIR.relative_to(PROJECT_DIR)}")
