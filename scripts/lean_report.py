# Report numbers for the lean pipeline (l_ps fused >= 0.25 + b5_sym_s2 with 8-way TTA, public 0.39) from a pulled batch-20
# stage2_gpu_job.py run: paired deltas against the C0 shape and the 24-view control, the ingredient ablation, per-annotator PQ,
# the end-to-end reproduction (detector run on the VM vs cached boxes), per-image seconds on each GPU, and a gallery of
# validation predictions at the 10th / 50th / 90th percentile of per-image PQ.
# Usage (from the repo root): .venv/bin/python scripts/lean_report.py BATCH_NAME [TIMING_BATCH_NAME ...]
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import TRAIN_IMAGES_DIR
from src.inference import fuse_boxes
from src.metrics import aggregate_official, bootstrap_pq, bootstrap_pq_by_annotator, official_match_counts, paired_pq_delta, pq_by_annotator

PIPELINE = PROJECT_DIR / "experiments/pipeline"
DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"  # its ground_truth holds every validation record as RLE
ORANGE, GREEN, RED = (1.0, 0.45, 0.0, 0.8), (0.1, 0.9, 0.3, 0.8), (1.0, 0.15, 0.15, 0.85)  # as in scripts/experiments/annotator_analysis.py


def as_rle(counts):
    return {"counts": counts.encode(), "size": [2048, 2048]}


def draw_panel(axis, image, coloured_masks, title):
    # drawn dilated so single-pixel-wide filaments stay visible when shrunk
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*image.shape, 4))
    for mask, colour in coloured_masks:
        overlay[cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = colour
    axis.imshow(overlay)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


batch_dir = PIPELINE / sys.argv[1]
frames = {path.stem: pd.read_csv(path, dtype={"physical_id": str}) for path in sorted(batch_dir.glob("*.csv")) if path.stem not in ["summary"]}
report = {}


def compare(control, label):
    delta = paired_pq_delta(frames[control], frames[label])
    verdict = "WIN" if delta["paired_delta_ci_low"] > 0 else "loss" if delta["paired_delta_ci_high"] < 0 else "tied"
    return {"label": label, "pq": aggregate_official(frames[label])["official_pq"], "against": control, "delta": delta["paired_delta"],
            "ci_low": delta["paired_delta_ci_low"], "ci_high": delta["paired_delta_ci_high"], "verdict": verdict}


# headline: lean against the September baseline shape and against the 24-view ensemble, on identical GPU inputs
headline = pd.DataFrame([compare("c0", "lean"), compare("ctrl", "lean"), compare("c0", "ctrl"), compare("lean", "lean_e2e")])
# ablation: one ingredient changed at a time, everything else as in lean
ablation = pd.DataFrame([compare("lean", label) for label in ["lean_notta", "lean_b5", "det_lps_t020", "det_lpl_t025", "det_lpl_t020", "det_l_t025", "det_l_t020"]])
ablation = pd.concat([ablation, pd.DataFrame([compare("det_lpl_t025", "lean"), compare("det_l_t025", "det_lpl_t025")])])
pd.set_option("display.width", 200)
print("HEADLINE\n" + headline.round(4).to_string(index=False) + "\n\nABLATION (delta = label - against)\n" + ablation.round(4).to_string(index=False))
report["headline"], report["ablation"] = headline.to_dict("records"), ablation.to_dict("records")

# per-annotator PQ and both bootstrap error bars
rows = []
for label in ["c0", "ctrl", "lean"]:
    summary = aggregate_official(frames[label])
    rows.append({"label": label, "pq": summary["official_pq"], "tp": summary["official_tp"], "fp": summary["official_fp"], "fn": summary["official_fn"],
                 "sq": summary["official_sq"], "rq": summary["official_rq"], **bootstrap_pq(frames[label]), **bootstrap_pq_by_annotator(frames[label])})
overall = pd.DataFrame(rows)
print("\nOVERALL\n" + overall.round(4).to_string(index=False))
annotators = pq_by_annotator(frames["lean"])[["team", "annotator", "records", "official_pq", "recall"]]
annotators.to_csv(batch_dir / "lean_pq_by_annotator.csv", index=False)
print(f"\nlean per-annotator PQ: {len(annotators)} annotators with >= 3 records, range {annotators['official_pq'].min():.3f}-{annotators['official_pq'].max():.3f}, "
      f"SD {annotators['official_pq'].std():.3f} (lean_pq_by_annotator.csv)")
report["overall"] = overall.to_dict("records")

# end to end: boxes made on the VM from raw images against the cached lps_t025 boxes the other configs used
cached = json.loads((PROJECT_DIR / "data/processed/ensemble_proposals/validation_l_ps.json").read_text())
fresh = json.loads((batch_dir / "lean_e2e_boxes.json").read_text())
counts, ious = [0, 0], []
for stem, item in fresh.items():
    boxes, scores = fuse_boxes([(np.asarray(cached[stem]["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(cached[stem]["scores"], dtype=np.float32))], [1.0], 0.40, 0.01)
    old, new = boxes[scores >= 0.25], np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 4)
    counts[0], counts[1] = counts[0] + len(old), counts[1] + len(new)
    if len(old) and len(new):
        overlap = mask_utils.iou([list(b[:2]) + list(b[2:] - b[:2]) for b in new], [list(b[:2]) + list(b[2:] - b[:2]) for b in old], [0] * len(old))
        ious.extend(overlap.max(axis=1).tolist())
ious = np.asarray(ious)
report["e2e_boxes"] = {"cached": counts[0], "fresh": counts[1], "fresh_with_cached_twin_iou_gt_0.9": float((ious > 0.9).mean()), "median_best_iou": float(np.median(ious))}
print(f"\nEND TO END: {counts[1]} fresh boxes vs {counts[0]} cached; {report['e2e_boxes']['fresh_with_cached_twin_iou_gt_0.9']:.1%} have a cached twin at IoU > 0.9")

# seconds per image, first image dropped (CUDA warm-up and model load)
report["timing"] = {}
for name in [sys.argv[1]] + sys.argv[2:]:
    timing = json.loads((PIPELINE / name / "lean_e2e_timing.json").read_text())
    stems = list(timing["stage2"])[1:]
    detector, stage2 = np.array([timing["detector"][s] for s in stems]), np.array([timing["stage2"][s] for s in stems])
    total = detector + stage2
    report["timing"][timing["gpu"]] = {"images": len(stems), "detector_mean": detector.mean(), "stage2_mean": stage2.mean(), "total_mean": total.mean(),
                                       "total_median": float(np.median(total)), "total_max": total.max(), "test_set_minutes": total.mean() * 180 / 60}
    print(f"TIMING {timing['gpu']}: detector {detector.mean():.2f} + Stage 2 {stage2.mean():.2f} = {total.mean():.2f} s/image "
          f"(median {np.median(total):.2f}, max {total.max():.2f}); 180 test images ~{total.mean() * 3:.1f} min")

# gallery: per-image PQ of lean pooled over its annotator records, at three percentiles
predictions = json.loads((batch_dir / "lean_predictions.json").read_text())
entries = {entry["physical_id"]: entry for entry in (json.loads(line) for line in DUMP.read_text().splitlines() if line.strip())}
per_image = frames["lean"].groupby("physical_id")[["official_tp", "official_fp", "official_fn", "official_iou_sum"]].sum()
per_image["pq"] = per_image["official_iou_sum"] / (per_image["official_tp"] + 0.5 * per_image["official_fp"] + 0.5 * per_image["official_fn"]).clip(lower=1e-9)
ranked = per_image.sort_values("pq")
picks = [(f"p{q}", ranked.index[int(round(q / 100 * (len(ranked) - 1)))]) for q in [10, 50, 90]]
figure, axes = plt.subplots(len(picks), 2, figsize=(11, 5.6 * len(picks)))
for row, (tag, stem) in zip(axes, picks):
    image = np.asarray(Image.open(TRAIN_IMAGES_DIR / f"{stem}.jpeg").convert("L"), dtype=np.uint8)
    record = max(entries[stem]["ground_truth"], key=lambda r: len(entries[stem]["ground_truth"][r]))
    predicted, targets = [as_rle(p) for p in predictions[stem]], [as_rle(t) for t in entries[stem]["ground_truth"][record]]
    overlap = mask_utils.iou(predicted, targets, [0] * len(targets)) if predicted and targets else np.zeros((len(predicted), len(targets)))
    counts = official_match_counts(predicted, targets)
    pq = counts["official_iou_sum"] / max(counts["official_tp"] + 0.5 * counts["official_fp"] + 0.5 * counts["official_fn"], 1e-9)
    hit_pred = overlap.max(axis=1) > 0.50 if len(targets) else np.zeros(len(predicted), bool)
    hit_gt = overlap.max(axis=0) > 0.50 if len(predicted) else np.zeros(len(targets), bool)
    draw_panel(row[0], image, [(mask_utils.decode(p), GREEN if hit else ORANGE) for p, hit in zip(predicted, hit_pred)],
               f"{stem} ({tag} of images by PQ)\nlean prediction: {len(predicted)} masks — green matched, orange unmatched")
    draw_panel(row[1], image, [(mask_utils.decode(t), GREEN if hit else RED) for t, hit in zip(targets, hit_gt)],
               f"annotator {record.split('-')[0]}: {len(targets)} filaments, PQ {pq:.3f}\ngreen found (IoU > 0.5), red missed")
figure.tight_layout()
figure.savefig(batch_dir / "lean_gallery.png", dpi=80, bbox_inches="tight")
plt.close(figure)
report["gallery"] = dict(picks)
(batch_dir / "lean_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
print(f"\nlean_report.json, lean_gallery.png, lean_pq_by_annotator.csv in {batch_dir.relative_to(PROJECT_DIR)}")
