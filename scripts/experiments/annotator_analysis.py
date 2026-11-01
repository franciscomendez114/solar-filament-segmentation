# Per-annotator scoring and a visual gallery, built from the saved candidate dump.
# Annotators disagree enough that identical predictions on identical images score 0.34 against one and 0.50
# against another, so the weak ones are where generalization has to improve. The gallery renders the same
# image once per annotator of its team, so the disagreement is visible rather than inferred.
# Usage (from the repo root): .venv/bin/python scripts/experiments/annotator_analysis.py
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import TRAIN_IMAGES_DIR
from src.metrics import aggregate_official, bootstrap_pq_by_annotator, official_match_counts, pq_by_annotator

DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"
OUTPUT_DIR = PROJECT_DIR / "experiments/diagnostics"
GALLERY_DIR = OUTPUT_DIR / "annotator_gallery"
KEEP_RULE = 0.20  # the current instance_score threshold
ORANGE, GREEN, RED = (1.0, 0.45, 0.0, 0.8), (0.1, 0.9, 0.3, 0.8), (1.0, 0.15, 0.15, 0.85)
GALLERY_TEAMS = 3  # teams containing the weakest annotators


def as_rle(value):
    return {"counts": value.encode() if isinstance(value, str) else value["counts"], "size": [2048, 2048]}


def load_entries():
    return [json.loads(line) for line in DUMP.read_text().splitlines() if line.strip()]


def score_entries(entries):
    rows = []
    for entry in entries:
        kept = [as_rle(c["rle"]) for c in entry["candidates"] if c["instance_score"] >= KEEP_RULE]
        for record, targets in entry["ground_truth"].items():
            rows.append({"physical_id": entry["physical_id"], "annotator_image_id": record,
                         **official_match_counts(kept, [as_rle(t) for t in targets])})
    return pd.DataFrame(rows)


def draw_panel(axis, image, coloured_masks, title):
    # coloured_masks: (mask, rgba) pairs, drawn dilated so single-pixel-wide filaments stay visible when shrunk
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*image.shape, 4))
    for mask, colour in coloured_masks:
        thick = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        overlay[thick.astype(bool)] = colour
    axis.imshow(overlay)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def render_gallery(entries, scores):
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    weakest_teams = list(dict.fromkeys(scores["team"]))[:GALLERY_TEAMS]
    written = []
    for entry in entries:
        annotators = sorted({r.split("-", 1)[0] for r in entry["ground_truth"]})
        team = annotators[0][:4]
        if team not in weakest_teams or len(annotators) < 2:
            continue
        image = np.asarray(Image.open(TRAIN_IMAGES_DIR / f"{entry['physical_id']}.jpeg").convert("L"), dtype=np.uint8)
        kept = [as_rle(c["rle"]) for c in entry["candidates"] if c["instance_score"] >= KEEP_RULE]
        predicted = [mask_utils.decode(r) for r in kept]

        figure, axes = plt.subplots(1, len(annotators) + 1, figsize=(5 * (len(annotators) + 1), 5.8))
        draw_panel(axes[0], image, [(m, ORANGE) for m in predicted], f"our prediction — {len(predicted)} masks")
        for axis, annotator in zip(axes[1:], annotators):
            record = f"{annotator}-{entry['physical_id']}"
            targets = [as_rle(t) for t in entry["ground_truth"][record]]
            counts = official_match_counts(kept, targets)
            pq = counts["official_iou_sum"] / max(counts["official_tp"] + 0.5 * counts["official_fp"] + 0.5 * counts["official_fn"], 1e-9)
            ious = mask_utils.iou(kept, targets, [0] * len(targets)) if kept and targets else np.zeros((0, len(targets)))
            best = ious.max(axis=0) if len(kept) else np.zeros(len(targets))
            coloured = [(mask_utils.decode(t), GREEN if best[i] > 0.50 else RED) for i, t in enumerate(targets)]
            row = scores[scores["annotator"] == annotator]
            overall = f" | this annotator overall {row['official_pq'].iloc[0]:.3f}" if len(row) else ""
            draw_panel(axis, image, coloured,
                       f"{annotator}: drew {len(targets)}, PQ {pq:.3f}{overall}\n"
                       f"green = we matched it ({counts['official_tp']})   red = we missed it ({counts['official_fn']})   "
                       f"our unmatched masks: {counts['official_fp']}")
        figure.suptitle(f"{entry['physical_id']} — same image, same prediction, different answer key", fontsize=12, y=1.02)
        figure.tight_layout()
        path = GALLERY_DIR / f"{team}_{entry['physical_id']}.png"
        figure.savefig(path, dpi=70, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


entries = load_entries()
frame = score_entries(entries)
scores = pq_by_annotator(frame)
scores.to_csv(OUTPUT_DIR / "annotator_scores.csv", index=False)

pooled = aggregate_official(frame)
spread = bootstrap_pq_by_annotator(frame)
print(f"pooled official PQ {pooled['official_pq']:.4f} over {len(frame)} records")
print(f"annotator-level bootstrap: 95% CI [{spread['annotator_pq_ci_low']:.4f}, {spread['annotator_pq_ci_high']:.4f}], "
      f"sd {spread['annotator_pq_bootstrap_sd']:.4f}")
print(f"spread across {spread['annotators']} annotators: sd {spread['annotator_spread_sd']:.4f}, "
      f"worst {spread['worst_annotator_pq']:.4f}, best {spread['best_annotator_pq']:.4f}\n")
print(scores[["team", "annotator", "records", "official_tp", "official_fp", "official_fn", "recall", "official_pq"]].to_string(index=False))

written = render_gallery(entries, scores)
print(f"\nwrote {len(written)} gallery panels to {GALLERY_DIR.relative_to(PROJECT_DIR)}")
