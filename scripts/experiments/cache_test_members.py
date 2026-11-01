# Caches test-set boxes for the named detector members (DETECTOR_MEMBERS in src/config.py), skipping any already cached.
# Validation caches come from scripts/experiments/detector_ensemble.py; this fills the matching test files a submission needs.
# Usage (from the repo root): .venv/bin/python scripts/experiments/cache_test_members.py MEMBER [MEMBER ...]
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import DEFAULT_PIPELINE_SETTINGS, DETECTOR_MEMBERS, ENSEMBLE_SETTINGS, PROCESSED_DATA_DIR, TEST_IMAGES_DIR
from src.inference import cache_proposals

test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
assert len(test_paths) == 180
for name in sys.argv[1:]:
    path = PROCESSED_DATA_DIR / f"ensemble_proposals/test_{name}.json"
    if path.exists():
        continue
    folder, weights, image_size, flip = DETECTOR_MEMBERS[name]
    print(f"caching test {name} ({folder}/{weights} @ {image_size}, flip {flip})", flush=True)
    settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": ENSEMBLE_SETTINGS["cache_confidence"], "yolo_image_size": image_size}
    cache = cache_proposals(YOLO(str(PROJECT_DIR / f"models/stage1/{folder}/weights/{weights}")), test_paths, torch.device("cpu"), settings, flip)
    path.write_text(json.dumps({key: {"boxes": boxes.tolist(), "scores": scores.tolist()} for key, (boxes, scores) in cache.items()}))
