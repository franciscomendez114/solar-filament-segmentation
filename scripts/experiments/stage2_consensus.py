# Stage 2 consensus experiment: retrain the B5 crop U-Net on consensus targets (the average of every annotator's
# version of the prompted filament) and compare it with the saved B5 under the official scorer on the frozen split.
# Only the training targets change: same recipe, same hard negatives, same checkpoint selection as the saved B5.
# Usage (from anywhere): python scripts/experiments/stage2_consensus.py [--smoke]
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import DEFAULT_PIPELINE_SETTINGS, TRAINING_SETTINGS
from src.data import FilamentCropDataset, consensus_annotations, load_records, verify_or_write_split
from src.experiments import append_registry, new_run_dir, save_run_files
from src.inference import predict_instances
from src.metrics import aggregate_official, annotation_to_rle, bootstrap_pq, official_match_counts, paired_pq_delta
from src.models import build_model, load_checkpoint, train_crop_unet

SMOKE = "--smoke" in sys.argv  # tiny CPU run that only checks the plumbing; writes under colab_transfer/smoke
OUTPUT_ROOT = PROJECT_DIR / "colab_transfer/smoke" if SMOKE else PROJECT_DIR
TRAIN_IMAGES_DIR = PROJECT_DIR / "data/raw/train/train_images"
ANNOTATIONS_PATH = PROJECT_DIR / "data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
DETECTOR_PATH = PROJECT_DIR / "models/stage1/yolo26l_filament_detection/weights/best.pt"
CONTROL_UNET = PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt"
NEGATIVE_CACHE = PROJECT_DIR / "data/processed/hard_negatives/hard_negatives_yolo26x_conf005.json"  # the saved B5's negatives
MIN_MATCH_IOU = 0.25
MASK_THRESHOLDS = [0.45] if SMOKE else [0.35, 0.45, 0.55]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
start = time.time()
print(f"Device: {device_name}", flush=True)

coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, train_records, validation_records = verify_or_write_split(records, PROJECT_DIR / "data/processed/split_manifest.json")
consensus = consensus_annotations(train_records, MIN_MATCH_IOU)
versions_per_filament = pd.Series([len(value) for value in consensus.values()]).value_counts().sort_index().to_dict()
print(f"Annotator versions averaged per training filament: {versions_per_filament}", flush=True)

# Training: notebook 03 recipe with consensus targets for positives
settings = {**DEFAULT_PIPELINE_SETTINGS, **TRAINING_SETTINGS}
positives = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"], "annotation": a, "consensus": consensus[a["id"]],
              "proposal_box": None} for r in train_records for a in r["annotations"]]
negatives = [{"physical_id": x["physical_id"], "file_name": x["file_name"], "image_path": TRAIN_IMAGES_DIR / x["file_name"], "annotation": None,
              "proposal_box": x["proposal_box"]} for x in json.loads(NEGATIVE_CACHE.read_text()) if x["score"] >= settings["negative_mining_floor"]]
validation_samples = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"], "annotation": a, "proposal_box": None}
                      for r in validation_records for a in r["annotations"]]  # individual targets, so checkpoint selection matches the saved B5
train_dataset = FilamentCropDataset(positives + negatives, settings, augment=True)
validation_dataset = FilamentCropDataset(validation_samples, settings, augment=False)
if SMOKE:
    train_dataset, validation_dataset = Subset(train_dataset, range(2)), Subset(validation_dataset, range(2))
    settings = {**settings, "num_epochs": 3, "batch_size": 2, "num_workers": 0}
print(f"Training crops: {len(positives)} positives, {len(negatives)} hard negatives; validation crops: {len(validation_samples)}", flush=True)

unet_dir = OUTPUT_ROOT / "models/stage2" / datetime.now().strftime("%Y%m%d_%H%M%S_b5_consensus")
unet_dir.mkdir(parents=True, exist_ok=False)
(unet_dir / "consensus_settings.json").write_text(json.dumps({"consensus_targets": True, "min_match_iou": MIN_MATCH_IOU, "negative_cache": NEGATIVE_CACHE.name,
                                                             "versions_per_filament": versions_per_filament, "device": device_name}, indent=2) + "\n")
loader_settings = {"batch_size": settings["batch_size"], "num_workers": settings["num_workers"], "pin_memory": device.type == "cuda"}
tick = time.time()
train_crop_unet(build_model("b5").to(device), DataLoader(train_dataset, shuffle=True, **loader_settings), DataLoader(validation_dataset, shuffle=False, **loader_settings),
                device, settings, unet_dir / "best.pt", unet_dir / "history.csv")
print(f"U-Net training took {(time.time() - tick) / 3600:.2f} h", flush=True)

# Full pipeline under the official rule: saved B5 (control) vs consensus B5, same yolo26l detector, several mask thresholds
detector = YOLO(str(DETECTOR_PATH))
groups = defaultdict(list)
for record in validation_records:
    groups[record["physical_id"]].append(record)
if SMOKE:
    groups = dict(list(groups.items())[:2])
registry_path = OUTPUT_ROOT / "experiments/registry.csv"
results, table = {}, []
for name, unet_path in [("control", CONTROL_UNET), ("consensus", unet_dir / "best.pt")]:
    unet, _ = load_checkpoint(unet_path, "b5", device)
    for threshold in MASK_THRESHOLDS:
        tick = time.time()
        rows, predictions, image_counts = [], [], []
        for physical_id, group in groups.items():
            masks, counts = predict_instances(group[0]["image_path"], detector, unet, device, threshold, DEFAULT_PIPELINE_SETTINGS)
            rles = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
            predictions.append({"physical_id": physical_id, "file_name": group[0]["file_name"], "rles": [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rles]})
            image_counts.append(counts)
            for record in group:
                targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
                rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"], **official_match_counts(rles, targets)})
        by_annotator = pd.DataFrame(rows)
        summary = {**aggregate_official(by_annotator), **bootstrap_pq(by_annotator), "instances_per_image": float(pd.DataFrame(image_counts)["final_instances"].mean())}
        if name == "consensus":
            summary.update(paired_pq_delta(results[("control", threshold)], by_annotator))
        run_id, run_dir = new_run_dir(OUTPUT_ROOT / "experiments", "pipeline", "S2C", f"b5_{name}_yolo26l_mask{threshold:.2f}")
        save_run_files(run_dir, {"detector": str(DETECTOR_PATH.relative_to(PROJECT_DIR)), "unet": str(unet_path.relative_to(OUTPUT_ROOT if name == "consensus" else PROJECT_DIR)),
                                 "mask_threshold": threshold, "pipeline": DEFAULT_PIPELINE_SETTINGS, "consensus_targets": name == "consensus"}, predictions, rows, summary)
        append_registry(registry_path, {"run_id": run_id, "timestamp": datetime.now().isoformat(), "notebook": "scripts/experiments/stage2_consensus.py", "section": "S2C",
                                        "detector": "yolo26l", "detector_imgsz": 1280, "conf": 0.20, "nms_iou": 0.20, "end2end": False,
                                        "unet_checkpoint": str(unet_path.relative_to(OUTPUT_ROOT if name == "consensus" else PROJECT_DIR)), "mask_threshold": threshold, "tta": False,
                                        "official_pq": summary["official_pq"], "sq": summary["official_sq"], "rq": summary["official_rq"], "tp": summary["official_tp"],
                                        "fp": summary["official_fp"], "fn": summary["official_fn"], "instances_per_image": summary["instances_per_image"],
                                        "pq_ci_low": summary["pq_ci_low"], "pq_ci_high": summary["pq_ci_high"], "device": device_name, "precision": "fp16" if device.type == "cuda" else "fp32",
                                        "notes": f"Consensus training targets (min match IoU {MIN_MATCH_IOU})" if name == "consensus" else "Control: saved B5 epoch 13"})
        results[(name, threshold)] = by_annotator
        table.append({"unet": name, "mask_threshold": threshold, **{k: summary[k] for k in ["official_pq", "official_sq", "official_rq", "official_tp", "official_fp", "official_fn",
                      "instances_per_image"]}, **{k: summary.get(k) for k in ["paired_delta", "paired_delta_ci_low", "paired_delta_ci_high"]}, "run_id": run_id})
        print(f"{name} mask {threshold:.2f}: PQ {summary['official_pq']:.4f} ({(time.time() - tick) / 60:.1f} min)", flush=True)

pd.DataFrame(table).to_csv(unet_dir / "comparison.csv", index=False)
print(pd.DataFrame(table).drop(columns="run_id").round(4).to_string(index=False))
print(f"Total {(time.time() - start) / 3600:.2f} h; U-Net and comparison in {unet_dir}", flush=True)
