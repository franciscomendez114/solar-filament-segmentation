# Caches YOLO26x boxes for the 170 validation images so the Tversky sweep VM does not need the 472 MB detector.
# The detector is frozen during Stage 2, so these boxes are identical for every variant; that makes the three
# runs exactly comparable. Boxes come from CPU fp32 here rather than GPU fp16, so absolute PQ from this cache
# can differ from the C0 number in the last decimal — only the between-variant comparison is load-bearing.
# Usage (from the repo root): .venv/bin/python scripts/experiments/build_validation_proposals.py
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import ANNOTATIONS_PATH, DEFAULT_PIPELINE_SETTINGS, PROCESSED_DATA_DIR, SPLIT_MANIFEST_PATH, TRAIN_IMAGES_DIR
from src.data import load_records, verify_or_write_split
from src.inference import cache_proposals

OUTPUT = PROCESSED_DATA_DIR / "validation_proposals_yolo26x.json"
CHECKPOINT = PROJECT_DIR / "models/stage1/yolo26x_filament_detection/weights/best.pt"

_, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
paths = sorted({r["physical_id"]: r["image_path"] for r in validation_records}.values())
print(f"{len(paths)} validation images", flush=True)

cache = cache_proposals(YOLO(str(CHECKPOINT)), paths, torch.device("cpu"), DEFAULT_PIPELINE_SETTINGS)
OUTPUT.write_text(json.dumps({key: {"boxes": boxes.tolist(), "scores": scores.tolist()} for key, (boxes, scores) in cache.items()}))
print(f"wrote {OUTPUT.relative_to(PROJECT_DIR)}: {sum(len(v['scores']) for v in json.loads(OUTPUT.read_text()).values())} proposals")
