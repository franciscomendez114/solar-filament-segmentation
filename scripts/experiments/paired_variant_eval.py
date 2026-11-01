# Paired comparison of two Stage 2 checkpoints on the same images with the same cached proposals.
# A difference of 0.01 sits at roughly one annotator-bootstrap SD, so the pooled numbers in each variant's
# summary.json cannot settle which is better; the paired bootstrap can, because it cancels the image draw.
# Usage (from the repo root): .venv/bin/python scripts/experiments/paired_variant_eval.py CONTROL_DIR CANDIDATE_DIR
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import ANNOTATIONS_PATH, DEFAULT_PIPELINE_SETTINGS, PROCESSED_DATA_DIR, SPLIT_MANIFEST_PATH, TRAIN_IMAGES_DIR
from src.data import load_records, verify_or_write_split
from src.inference import predict_instances
from src.metrics import aggregate_official, annotation_to_rle, official_match_counts, paired_pq_delta
from src.models import load_checkpoint

MASK_THRESHOLD = 0.45
device = torch.device("cpu")

_, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
cached = json.loads((PROCESSED_DATA_DIR / "validation_proposals_yolo26x.json").read_text())
proposals = {k: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for k, v in cached.items()}

by_image = defaultdict(list)
for record in validation_records:
    by_image[record["physical_id"]].append(record)


def score(checkpoint_path):
    model, _ = load_checkpoint(checkpoint_path, "b5", device)
    rows = []
    for number, (physical_id, group) in enumerate(by_image.items(), 1):
        masks, _ = predict_instances(group[0]["image_path"], None, model, device, MASK_THRESHOLD,
                                     DEFAULT_PIPELINE_SETTINGS, use_amp=False, proposals=proposals[physical_id])
        predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
        for record in group:
            targets = [annotation_to_rle(a, record["height"], record["width"]) for a in record["annotations"]]
            rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"],
                         **official_match_counts(predicted, targets)})
        if number % 20 == 0:
            print(f"  {number}/{len(by_image)} images", flush=True)
    return pd.DataFrame(rows)


def resolve(argument):
    # accepts a run directory or a checkpoint path directly, so saved champions can be compared to new runs
    path = Path(argument)
    return path if path.suffix == ".pt" else path / "best.pt"


control_path, candidate_path = resolve(sys.argv[1]), resolve(sys.argv[2])
control_dir, candidate_dir = control_path.parent, candidate_path.parent
frames = {}
for name, checkpoint in [("control", control_path), ("candidate", candidate_path)]:
    print(f"{name}: {checkpoint}", flush=True)
    frames[name] = score(checkpoint)
    print(f"  {name} pooled official PQ {aggregate_official(frames[name])['official_pq']:.4f}", flush=True)

delta = paired_pq_delta(frames["control"], frames["candidate"])
print(f"\npaired delta ({candidate_dir.name} - {control_dir.name}): "
      f"{delta['paired_delta']:+.4f} [{delta['paired_delta_ci_low']:+.4f}, {delta['paired_delta_ci_high']:+.4f}]")
print("verdict:", "candidate wins" if delta["paired_delta_ci_low"] > 0 else
      "control wins" if delta["paired_delta_ci_high"] < 0 else "inside noise — tied")
output = PROJECT_DIR / f"experiments/diagnostics/paired_{control_dir.name}_vs_{candidate_dir.name}.json"
output.write_text(json.dumps({"control": control_dir.name, "candidate": candidate_dir.name,
                              "control_pq": aggregate_official(frames["control"])["official_pq"],
                              "candidate_pq": aggregate_official(frames["candidate"])["official_pq"], **delta}, indent=2) + "\n")
print(f"wrote {output.relative_to(PROJECT_DIR)}")
