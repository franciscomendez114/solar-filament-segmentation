# How much of each validation filament the lean pipeline's crops can see, per crop expansion and detector confidence:
# a filament is coverable when some crop holds more than half its pixels (a perfect mask restricted to that crop could
# then pass IoU 0.5; nothing outside a crop can be predicted) and fully covered when one crop holds 99% of them. Also the
# old center-inclusion rule (the September sweep), the input scale (crop side / 512) and how many other annotated filaments
# fall inside each crop, which the box prompt has to disambiguate. The U-Net input is 512 px at every expansion, so a wider
# crop costs no compute, only resolution.
# Usage (from the repo root): .venv/bin/python scripts/crop_coverage.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.data import make_crop_geometry

CACHE = PROJECT_DIR / "data/processed/ensemble_proposals/validation_l_ps.json"  # conf 0.01, NMS IoU 0.20, imgsz 1280
DUMP = PROJECT_DIR / "experiments/diagnostics/quality_candidates.jsonl"  # its ground_truth holds every validation record as RLE
OUTPUT = PROJECT_DIR / "experiments/diagnostics/crop_coverage.csv"
EXPANSIONS, CONFIDENCES = [1.00, 1.25, 1.50, 1.75, 2.00], [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

cache = json.loads(CACHE.read_text())
entries = [json.loads(line) for line in DUMP.read_text().splitlines() if line.strip()]
rows = []
for entry in entries:
    item = cache[entry["physical_id"]]
    boxes, scores = np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 4), np.asarray(item["scores"])
    filaments = []
    for record, targets in entry["ground_truth"].items():
        for counts in targets:
            ys, xs = np.nonzero(mask_utils.decode({"counts": counts.encode(), "size": [2048, 2048]}))
            filaments.append((record, xs, ys))
    for expansion in EXPANSIONS:
        crops = [make_crop_geometry([x1, y1, x2 - x1, y2 - y1], expansion, 2048, 2048) for x1, y1, x2, y2 in boxes]
        for confidence in CONFIDENCES:
            kept = [crop for crop, score in zip(crops, scores) if score >= confidence]
            for record, xs, ys in filaments:
                inside = [((xs >= c["x1"]) & (xs < c["x1"] + c["side"]) & (ys >= c["y1"]) & (ys < c["y1"] + c["side"])).mean() for c in kept]
                cx, cy = xs.mean(), ys.mean()
                rows.append({"expansion": expansion, "confidence": confidence, "record": record, "best_fraction": max(inside, default=0.0),
                             "center_inside": any(c["x1"] <= cx < c["x1"] + c["side"] and c["y1"] <= cy < c["y1"] + c["side"] for c in kept)})
            if confidence == 0.25:
                # per crop: input scale and how many of the record's other filaments a crop also contains more than half of
                for crop in kept:
                    for record in entry["ground_truth"]:
                        inside = [((xs >= crop["x1"]) & (xs < crop["x1"] + crop["side"]) & (ys >= crop["y1"]) & (ys < crop["y1"] + crop["side"])).mean() > 0.5
                                  for r, xs, ys in filaments if r == record]
                        rows.append({"expansion": expansion, "confidence": "crop", "record": record, "scale": crop["side"] / 512, "filaments_in_crop": sum(inside)})

frame = pd.DataFrame(rows)
filaments = frame[frame["confidence"] != "crop"].copy()
filaments["confidence"] = filaments["confidence"].astype(float)
assert len(filaments) == 1258 * len(EXPANSIONS) * len(CONFIDENCES)
table = filaments.groupby(["confidence", "expansion"]).agg(coverable=("best_fraction", lambda f: (f > 0.5).mean()), full=("best_fraction", lambda f: (f >= 0.99).mean()),
                                                          center=("center_inside", "mean")).reset_index()
crops = frame[frame["confidence"] == "crop"].groupby("expansion").agg(crops=("scale", "size"), median_scale=("scale", "median"), upsampled=("scale", lambda s: (s < 1).mean()),
                                                                      mean_filaments=("filaments_in_crop", "mean"), multi=("filaments_in_crop", lambda n: (n >= 2).mean()))
table.to_csv(OUTPUT, index=False)
pd.set_option("display.width", 200)
print("FILAMENT COVERAGE (share of 1,258 validation filaments)\n" + table.pivot(index="confidence", columns="expansion", values=["coverable", "full", "center"]).round(4).to_string())
print("\nCROPS AT CONFIDENCE 0.25 (per crop x annotator record)\n" + crops.round(3).to_string())
print(f"\n{OUTPUT.relative_to(PROJECT_DIR)}")
