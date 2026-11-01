# Builds the Kaggle submission for the detector + U-Net ensemble chosen on 2026-10-27 (ENSEMBLE_* in src/config.py):
# WBF of all seven Stage 1 detectors feeding the B5 champion averaged with the symmetric-Tversky B5.
# Validation: official PQ 0.4628 vs C0 0.4303 on the same CPU path, paired delta +0.0325 [+0.0215, +0.0437].
# Runs on CPU fp32. Writes the CSV only; submitting to Kaggle is a separate, manual step.
# Usage (from the repo root): .venv/bin/python scripts/experiments/ensemble_submission.py
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import DEFAULT_PIPELINE_SETTINGS, ENSEMBLE_DETECTORS, ENSEMBLE_SETTINGS, ENSEMBLE_UNETS, PROCESSED_DATA_DIR, SUBMISSIONS_DIR, TEST_IMAGES_DIR
from src.inference import cache_proposals, create_submission, fuse_detector_caches
from src.models import ModelAverage, load_checkpoint

CACHE_DIR = PROCESSED_DATA_DIR / "ensemble_proposals"
device = torch.device("cpu")

test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
assert len(test_paths) == 180

caches = []
for name, (folder, image_size) in ENSEMBLE_DETECTORS.items():
    path = CACHE_DIR / f"test_{name}.json"
    if not path.exists():
        print(f"caching {name} ({folder} @ {image_size})", flush=True)
        settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": ENSEMBLE_SETTINGS["cache_confidence"], "yolo_image_size": image_size}
        cache = cache_proposals(YOLO(str(PROJECT_DIR / f"models/stage1/{folder}/weights/best.pt")), test_paths, device, settings)
        path.write_text(json.dumps({key: {"boxes": boxes.tolist(), "scores": scores.tolist()} for key, (boxes, scores) in cache.items()}))
    saved = json.loads(path.read_text())
    caches.append({key: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for key, v in saved.items()})

unet = ModelAverage([load_checkpoint(PROJECT_DIR / path, "b5", device)[0] for path in ENSEMBLE_UNETS]).eval()
frame, statistics = create_submission(test_paths, None, unet, device, ENSEMBLE_SETTINGS["mask_threshold"], DEFAULT_PIPELINE_SETTINGS, fuse_detector_caches(caches, ENSEMBLE_SETTINGS))

output_dir = SUBMISSIONS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wbf7_unet2"
output_dir.mkdir(parents=True, exist_ok=False)
frame.to_csv(output_dir / "submission.csv", index=False)
statistics.to_csv(output_dir / "image_statistics.csv", index=False)
(output_dir / "settings.json").write_text(json.dumps({"detectors": ENSEMBLE_DETECTORS, "unets": ENSEMBLE_UNETS, **ENSEMBLE_SETTINGS, "pipeline": DEFAULT_PIPELINE_SETTINGS,
    "validation_official_pq_cpu": 0.4628, "validation_c0_pq_cpu": 0.4303, "device": "cpu", "precision": "fp32"}, indent=2) + "\n")
print(f"{len(frame)} masks over {frame.filament_id.map(lambda v: v.rpartition('_')[0]).nunique()} of 180 images "
      f"({len(frame) / 180:.2f} per image); wrote {(output_dir / 'submission.csv').relative_to(PROJECT_DIR)}")
