# Figures for the technical report, written to report/figures/ as vector PDF (for LaTeX) plus a PNG preview.
# pipeline: the final inference path drawn on a real validation image — its l_ps boxes, one real prompted crop with the
# U-Net's 8-way TTA probability (b5_sym_s2, run here on CPU) and the final lean masks from the batch-20 dump.
# development: how the final pipeline was cut out of the baseline -> ensemble teacher -> pseudo-label student path.
# examples: lean predictions against one annotator on the validation images at the 10th / 50th / 90th percentile of per-image PQ.
# annotators: one validation image's lean prediction scored against each of its annotators, to show how much the score
# depends on who drew the labels.
# tuning: PQ against the detector confidence threshold (batch-23 sweep) and the best-match IoU of every validation filament, lean vs C0.
# distributions: the rubric's IoU and Dice distributions (every overlapping prediction-annotation pair, as the organizers' code
# plots them; means match official_eval_*/summary.json) for lean vs C0, and lean's found share by filament size and contrast.
# Usage (from the repo root): .venv/bin/python scripts/report_figures.py
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import TRAIN_IMAGES_DIR
from src.metrics import aggregate_official, official_match_counts, paired_pq_delta
from src.data import build_crop_tensors
from src.models import RotationTTA, load_checkpoint

OUTPUT_DIR = PROJECT_DIR / "report/figures"
BATCH = PROJECT_DIR / "experiments/pipeline/20261029_125607_s2_gpu_report"
SWEEP = PROJECT_DIR / "experiments/pipeline/20261029_192820_s2_gpu_lean_sweep"
DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"  # its ground_truth holds every validation record as RLE
UNET = PROJECT_DIR / "experiments/pipeline/20261028_005250_unet_train/b5_sym_s2.pt"
ANNOTATED = "20150625082034Lh"  # three annotators, same prediction: PQ 0.76 / 0.49 / 0.40 (lean.csv of the batch)
EXAMPLE, CROP_BOX = "20121118092514Th", 3  # the median-PQ validation image of lean_gallery.png and one of its matched boxes
ORANGE, CYAN = "#ff7f0e", "#17becf"
BLUE, RED_ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the dataviz reference palette
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})


def save(figure, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def arrow(figure, start, end):
    figure.add_artist(matplotlib.patches.FancyArrowPatch(start, end, transform=figure.transFigure, arrowstyle="-|>", mutation_scale=8, color="black", lw=0.8))


def pipeline_figure():
    image = cv2.imread(str(TRAIN_IMAGES_DIR / f"{EXAMPLE}.jpeg"), cv2.IMREAD_GRAYSCALE)
    item = json.loads((BATCH / "lean_e2e_boxes.json").read_text())[EXAMPLE]
    boxes = np.asarray(item["boxes"]).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[CROP_BOX]
    inputs, _, _ = build_crop_tensors(image, [x1, y1, x2 - x1, y2 - y1], None, 1.5, 512, 64)
    unet = RotationTTA(load_checkpoint(UNET, "b5", torch.device("cpu"))[0]).eval()
    with torch.inference_mode():
        probability = torch.sigmoid(unet(inputs[None]))[0, 0].numpy()
    masks = [mask_utils.decode({"counts": counts.encode(), "size": [2048, 2048]}) for counts in json.loads((BATCH / "lean_predictions.json").read_text())[EXAMPLE]]

    figure = plt.figure(figsize=(7.0, 1.65))
    width, gap, bottom, height = 0.118, 0.05, 0.27, 0.66
    lefts = [0.0 + i * (width + gap) for i in range(6)]
    axes = [figure.add_axes([left, bottom, width, height]) for left in lefts]
    for axis in axes:
        axis.set_xticks([]), axis.set_yticks([])
    small = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    axes[0].imshow(small, cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(small, cmap="gray", vmin=0, vmax=255)
    for index, (a, b, c, d) in enumerate(boxes / 4):
        chosen = index == CROP_BOX
        axes[1].add_patch(Rectangle((a, b), c - a, d - b, fill=False, lw=1.2 if chosen else 0.6, ec=CYAN if chosen else ORANGE))
    # the prompted crop's three channels, drawn as a stack
    axes[2].set_visible(False)
    for depth, (channel, label) in enumerate(zip(inputs.numpy()[::-1], ["box prompt", "CLAHE", "grayscale"])):
        offset = (2 - depth) * 0.016
        tile = figure.add_axes([lefts[2] + offset, bottom + 0.032 - offset * 2, width * 0.86, height * 0.86])
        tile.imshow(channel, cmap="gray", vmin=0, vmax=1)
        tile.set_xticks([]), tile.set_yticks([])
        for spine in tile.spines.values():
            spine.set_color(CYAN if depth == 2 else "black")
    axes[3].set_visible(False)
    block = figure.add_axes([lefts[3], bottom, width, height])
    block.axis("off")
    block.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84, boxstyle="round,pad=0.02", fc="#e8eef7", ec="black", lw=0.7))
    block.plot([0.2, 0.35, 0.5, 0.65, 0.8], [0.72, 0.52, 0.34, 0.52, 0.72], color="#4a6fa5", lw=2.0, marker="s", ms=3.5)
    block.text(0.5, 0.88, "U-Net", ha="center", va="top", fontsize=7.5, weight="bold")
    block.text(0.5, 0.25, "EfficientNet-B5\n+ scSE, 8-way TTA", ha="center", va="top", fontsize=5.8)
    block.set_xlim(0, 1), block.set_ylim(0, 1)
    axes[4].imshow(probability, cmap="magma", vmin=0, vmax=1)
    overlay = np.zeros((512, 512, 4))
    colours = plt.get_cmap("tab10")
    for index, mask in enumerate(masks):
        thick = cv2.dilate(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA), np.ones((3, 3), np.uint8)) > 0
        overlay[thick] = (*colours(index % 10)[:3], 0.95)
    axes[5].imshow(small, cmap="gray", vmin=0, vmax=255)
    axes[5].imshow(overlay)

    captions = ["Input full disk\n2048 × 2048", "Stage 1: YOLO26l\n@1280, conf ≥ 0.25", "Per box: crop ×1.5 → 512²\ngray, CLAHE, box prompt",
                "Stage 2: prompted\nsegmentation", "Crop probability\n(single filament)", "Paste back, threshold,\ndedup, resolve overlaps"]
    for left, caption in zip(lefts, captions):
        figure.text(left + width / 2, bottom - 0.04, caption, ha="center", va="top", fontsize=6.5)
    for left in lefts[:-1]:
        arrow(figure, (left + width + 0.006, bottom + height / 2), (left + width + gap - 0.006, bottom + height / 2))
    figure.text(lefts[5] + width + 0.004, bottom + height / 2, "→ RLE", ha="left", va="center", fontsize=6.5)
    span = (lefts[2], lefts[4] + width)
    figure.add_artist(matplotlib.lines.Line2D(span, [bottom + height + 0.05] * 2, transform=figure.transFigure, color="#666", lw=0.6))
    figure.text(sum(span) / 2, bottom + height + 0.07, "repeated for every box", ha="center", va="bottom", fontsize=6, style="italic", color="#444")
    save(figure, "pipeline")


def development_figure():
    figure, axis = plt.subplots(figsize=(3.33, 3.95))
    axis.set_xlim(0, 10), axis.set_ylim(0, 13.2)
    axis.axis("off")

    def box(x, y, w, h, title, body, colour="#f2f2f2", dashed=False):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=colour, ec="black", lw=1.1 if dashed else 0.6, ls="--" if dashed else "-"))
        axis.text(x + 0.15, y + h - 0.12, title, ha="left", va="top", fontsize=6.8, weight="bold")
        axis.text(x + 0.15, y + h - 0.58, body, ha="left", va="top", fontsize=5.8, linespacing=1.3)

    def link(start, end, label=None):
        axis.annotate("", end, start, arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "black", "shrinkA": 0, "shrinkB": 0})
        if label:
            axis.text(max(start[0], end[0]) + 0.15, (start[1] + end[1]) / 2, label, fontsize=5.6, style="italic", va="center", color="#333")

    height, step = 1.95, 2.6
    rows = [("1  Baselines", "7 YOLO26 detectors (n to x, 1024-1280 px)\n+ EfficientNet-B5 U-Net with hard negatives.\nBest single pair: public 0.36"),
            ("2  Ensemble teacher", "8 detectors (+ YOLO11m) x 3 flip views, fused by\nweighted box fusion (the teacher). With 3 averaged\nU-Nets + TTA: public 0.39"),
            ("3  Pseudo-labels", "1,000 unlabelled GONG H-\u03b1 frames (>= 1 day from\nany competition image, checked against test)\n-> 4,840 pseudo boxes"),
            ("4  Students", "7 detectors retrained on 984 competition\n+ 1,000 pseudo-labelled images")]
    tops = [13.1 - i * step for i in range(len(rows))]
    for top, (title, body) in zip(tops, rows):
        box(0.1, top - height, 9.8, height, title, body.replace("x 3", "\u00d7 3").replace(">=", "\u2265").replace("->", "\u2192"))
    labels = [None, "teacher labels the pool", None]
    for top, label in zip(tops[:-1], labels):
        link((5.0, top - height - 0.1), (5.0, top - step + 0.1), label)
    box(0.1, 0.0, 4.6, 2.55, "5a  Full ensemble", "17 detectors \u00d7 3 views\n(teachers, students,\n2 union-label) + 3 U-Nets\nPublic 0.40, 383 M params")
    box(5.3, 0.0, 4.6, 2.55, "5b  Final solution", "1 student (l_ps, YOLO26l)\n+ 1 U-Net (b5_sym_s2)\n+ 8-way TTA\nPublic 0.39, 57.5 M params", colour="#fff1dd", dashed=True)
    bottom = tops[-1] - height - 0.1
    link((2.6, bottom), (2.4, 2.73))
    link((7.4, bottom), (7.6, 2.73), "keep one")
    save(figure, "development")


def decode(counts):
    return mask_utils.decode({"counts": counts.encode(), "size": [2048, 2048]})


def ground_truth():
    return {entry["physical_id"]: entry["ground_truth"] for entry in (json.loads(line) for line in DUMP.read_text().splitlines() if line.strip())}


def examples_figure():
    truth, predictions = ground_truth(), json.loads((BATCH / "lean_predictions.json").read_text())
    picks = json.loads((BATCH / "lean_report.json").read_text())["gallery"]
    figure, axes = plt.subplots(1, 6, figsize=(7.0, 1.5))
    for column, (tag, stem) in enumerate(picks.items()):
        image = cv2.imread(str(TRAIN_IMAGES_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        record = max(truth[stem], key=lambda r: len(truth[stem][r]))
        predicted, targets = [{"counts": c.encode(), "size": [2048, 2048]} for c in predictions[stem]], [{"counts": c.encode(), "size": [2048, 2048]} for c in truth[stem][record]]
        overlap = mask_utils.iou(predicted, targets, [0] * len(targets))
        hit_pred, hit_gt = overlap.max(axis=1) > 0.5, overlap.max(axis=0) > 0.5
        counts = official_match_counts(predicted, targets)
        pq = counts["official_iou_sum"] / (counts["official_tp"] + 0.5 * counts["official_fp"] + 0.5 * counts["official_fn"])
        for axis, masks, hits, colours, title in [(axes[2 * column], predicted, hit_pred, ("#1baf7a", ORANGE), f"{tag}: prediction ({len(predicted)})"),
                                                 (axes[2 * column + 1], targets, hit_gt, ("#1baf7a", "#e8344a"), f"annotator ({len(targets)}), PQ {pq:.2f}")]:
            small = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
            overlay = np.zeros((1024, 1024, 4))
            for mask, hit in zip(masks, hits):
                thick = cv2.dilate(cv2.resize(mask_utils.decode(mask), (1024, 1024), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)) > 0
                overlay[thick] = (*matplotlib.colors.to_rgb(colours[0] if hit else colours[1]), 0.9)
            axis.imshow(small, cmap="gray", vmin=0, vmax=255)
            axis.imshow(overlay)
            axis.set_title(title, fontsize=6.2, pad=2)
            axis.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.03)
    save(figure, "examples")


def best_ious(predictions, truth):
    # the best IoU any prediction reaches with each annotated filament, every record counted (1,258 filaments)
    values = []
    for stem, records in truth.items():
        predicted = [{"counts": c.encode(), "size": [2048, 2048]} for c in predictions[stem]]
        for targets in records.values():
            targets = [{"counts": c.encode(), "size": [2048, 2048]} for c in targets]
            values += list(mask_utils.iou(predicted, targets, [0] * len(targets)).max(axis=0)) if predicted else [0.0] * len(targets)
    return np.asarray(values)


def tuning_figure():
    read = lambda label: pd.read_csv(SWEEP / f"{label}.csv", dtype={"physical_id": str})
    lean, thresholds = read("lean"), [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    labels = [f"conf_{round(t * 100):03d}" if t != 0.25 else "lean" for t in thresholds]
    pq, low, high = [], [], []
    for label in labels:
        frame = read(label)
        delta = paired_pq_delta(lean, frame)
        pq.append(aggregate_official(frame)["official_pq"])
        low.append(aggregate_official(lean)["official_pq"] + delta["paired_delta_ci_low"])
        high.append(aggregate_official(lean)["official_pq"] + delta["paired_delta_ci_high"])
    truth = ground_truth()
    lean_iou, c0_iou = best_ious(json.loads((BATCH / "lean_predictions.json").read_text()), truth), best_ious(json.loads((BATCH / "c0_predictions.json").read_text()), truth)
    assert len(lean_iou) == len(c0_iou) == 1258

    figure, (left, right) = plt.subplots(1, 2, figsize=(3.33, 1.45), gridspec_kw={"wspace": 0.42})
    pq, low, high = np.asarray(pq), np.asarray(low), np.asarray(high)
    left.fill_between(thresholds, low, high, color=BLUE, alpha=0.15, lw=0)
    left.plot(thresholds, pq, color=BLUE, lw=1.2, marker="o", ms=2.8)
    left.plot([0.25], [pq[4]], marker="o", ms=4.5, color=BLUE, mec="black", mew=0.6)
    left.set_xlabel("detector confidence threshold", fontsize=6.3)
    left.set_ylabel("validation PQ", fontsize=6.3)
    left.set_xticks([0.05, 0.15, 0.25, 0.35])
    bins = np.linspace(0, 1, 21)
    for values, colour, name in [(c0_iou, RED_ORANGE, "C0"), (lean_iou, BLUE, "final")]:
        right.hist(values, bins=bins, histtype="step", lw=1.2, color=colour, label=name)
    right.axvline(0.5, color="#666", lw=0.6, ls="--")
    right.set_xlabel("best IoU per annotated filament", fontsize=6.3)
    right.set_ylabel("filaments", fontsize=6.3)
    right.legend(fontsize=5.6, frameon=False, loc="center left", bbox_to_anchor=(0.06, 0.62))
    for axis, letter in [(left, "a"), (right, "b")]:
        axis.tick_params(labelsize=5.6, length=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e5e5e5", lw=0.5)
        axis.set_axisbelow(True)
        axis.set_title(f"({letter})", fontsize=6.5, loc="left", pad=2)
    print(f"best IoU > 0.5: lean {(lean_iou > 0.5).mean():.3f}, C0 {(c0_iou > 0.5).mean():.3f}; IoU < 0.05: lean {(lean_iou < 0.05).mean():.3f}, C0 {(c0_iou < 0.05).mean():.3f}; "
          f"median lean {np.median(lean_iou):.3f}, C0 {np.median(c0_iou):.3f}")
    save(figure, "tuning")


def distributions_figure():
    truth = ground_truth()
    predictions = {label: json.loads((BATCH / f"{label}_predictions.json").read_text()) for label in ["lean", "c0"]}
    pairs, rows = {label: [] for label in predictions}, []
    for stem, records in truth.items():
        image = cv2.imread(str(TRAIN_IMAGES_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        for targets in records.values():
            encoded, best = [{"counts": c.encode(), "size": [2048, 2048]} for c in targets], {}
            for label, saved in predictions.items():
                predicted = [{"counts": c.encode(), "size": [2048, 2048]} for c in saved[stem]]
                overlap = mask_utils.iou(predicted, encoded, [0] * len(encoded)) if predicted else np.zeros((0, len(encoded)))
                pairs[label] += overlap[overlap > 0].tolist()
                best[label] = overlap.max(axis=0) if predicted else np.zeros(len(encoded))
            for number, rle in enumerate(encoded):
                # contrast: how much darker the filament is than a 15 px band around it, relative to that band
                mask = mask_utils.decode(rle).astype(bool)
                band = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~mask
                background = np.median(image[band])
                rows.append({"area": mask.sum(), "contrast": (background - image[mask].mean()) / max(background, 1), "lean": best["lean"][number]})
    filaments = pd.DataFrame(rows)
    assert len(filaments) == 1258

    figure, axes = plt.subplots(1, 3, figsize=(3.33, 1.15), gridspec_kw={"wspace": 0.55})
    bins = np.linspace(0, 1, 21)
    for axis, transform, name in [(axes[0], lambda v: v, "IoU"), (axes[1], lambda v: 2 * v / (1 + v), "Dice")]:
        for label, colour, legend in [("c0", RED_ORANGE, "C0"), ("lean", BLUE, "final")]:
            values = transform(np.asarray(pairs[label]))
            axis.hist(values, bins=bins, histtype="step", lw=1.1, color=colour, weights=np.full(len(values), 1 / len(values)), label=legend)
            axis.axvline(values.mean(), color=colour, lw=0.7, ls="--")
            print(f"{name} {label}: {len(values)} overlapping pairs, mean {values.mean():.4f}, median {np.median(values):.4f}")
        axis.set_xlabel(name, fontsize=6.0)
        axis.set_xticks([0, 0.5, 1])
    axes[0].set_ylabel("share of pairs", fontsize=6.0)
    axes[1].legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.2)  # panel (b) is empty at upper left
    right = axes[2]
    for column, marker, name in [("area", "o", "size"), ("contrast", "s", "contrast")]:
        terciles = pd.qcut(filaments[column], 3, labels=False)
        found = filaments.groupby(terciles)["lean"].apply(lambda v: (v > 0.5).mean()).to_numpy()
        never = filaments.groupby(terciles)["lean"].apply(lambda v: (v < 0.05).mean()).to_numpy()
        edges = filaments.groupby(terciles)[column].agg(["min", "max"]).round(3).values.tolist()
        print(f"{name} terciles {edges}: found {found.round(3)}, never found {never.round(3)}")
        right.plot([0, 1, 2], found, color="#444", lw=1.0, marker=marker, ms=2.8)
        right.annotate(name, (2, found[2]), xytext=(3, 0), textcoords="offset points", fontsize=5.4, va="center", color="#222")
    right.set_xticks([0, 1, 2], ["low", "mid", "high"])
    right.set_xlim(-0.3, 2.9)
    right.set_ylim(0.4, 0.9)
    right.set_xlabel("filament tercile", fontsize=6.0)
    right.set_ylabel("found (IoU > 0.5)", fontsize=6.0)
    for axis, letter in zip(axes, "abc"):
        axis.tick_params(labelsize=5.4, length=2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e5e5e5", lw=0.5)
        axis.set_axisbelow(True)
        axis.set_title(f"({letter})", fontsize=6.5, loc="left", pad=2)
    print(f"Spearman(area, contrast) = {filaments[['area', 'contrast']].corr('spearman').iloc[0, 1]:.3f}")
    save(figure, "distributions")


def annotators_figure(stem=ANNOTATED, name="annotators"):
    records, predictions = ground_truth()[stem], json.loads((BATCH / "lean_predictions.json").read_text())[stem]
    image = cv2.resize(cv2.imread(str(TRAIN_IMAGES_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE), (1024, 1024), interpolation=cv2.INTER_AREA)
    predicted = [{"counts": c.encode(), "size": [2048, 2048]} for c in predictions]
    panels = [("final prediction", predicted, [BLUE] * len(predicted))]
    scored = []
    for record in records:
        targets = [{"counts": c.encode(), "size": [2048, 2048]} for c in records[record]]
        counts = official_match_counts(predicted, targets)
        pq = counts["official_iou_sum"] / (counts["official_tp"] + 0.5 * counts["official_fp"] + 0.5 * counts["official_fn"])
        found = mask_utils.iou(predicted, targets, [0] * len(targets)).max(axis=0) > 0.5
        scored.append((pq, f"annotator {record.split('-')[0]}: PQ {pq:.2f}", targets, ["#1baf7a" if hit else "#e8344a" for hit in found]))
    panels += [panel[1:] for panel in sorted(scored, key=lambda panel: -panel[0])]
    figure, axes = plt.subplots(1, len(panels), figsize=(7.0, 7.0 / len(panels) + 0.2))
    for axis, (title, masks, colours) in zip(axes, panels):
        overlay = np.zeros((1024, 1024, 4))
        for mask, colour in zip(masks, colours):
            thick = cv2.dilate(cv2.resize(mask_utils.decode(mask), (1024, 1024), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)) > 0
            overlay[thick] = (*matplotlib.colors.to_rgb(colour), 0.9)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.imshow(overlay)
        axis.set_title(f"{title} ({len(masks)})", fontsize=6.5, pad=2)
        axis.axis("off")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0.03)
    save(figure, name)


if __name__ == "__main__":
    pipeline_figure()
    development_figure()
    examples_figure()
    annotators_figure()
    tuning_figure()
    distributions_figure()
    print(f"wrote {sorted(p.name for p in OUTPUT_DIR.iterdir())}")
