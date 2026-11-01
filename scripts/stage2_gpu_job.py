# Runs a list of full-pipeline validation configs on a Colab GPU and writes one per-record CSV per config, so all
# scoring and paired deltas happen locally from identical inputs. Detector boxes arrive pre-fused (no detectors on
# the VM); each config names a proposal set, the U-Nets to average, rotation TTA, and the mask threshold. A config with
# "split": "test" instead writes <label>_submission.csv for the 180 test images (its proposal set must be fused test boxes).
# "dump": true also writes <label>_predictions.json ({physical_id: [RLE counts]}) for the organizers' evaluation code
# (scripts/official_self_evaluation.py). "detector": {"weights", "imgsz", "iou", "threshold"} runs end to end from raw images
# instead: the detector in /content/detectors/ makes the boxes here, with the settings the cached boxes used (optional "nms_iou",
# default 0.20, and "end2end", default false, change them), and per-image
# Stage 1 / Stage 2 seconds go to <label>_timing.json, the boxes to <label>_boxes.json (checkable against the cache).
# The VM needs /content/src/, /content/split_manifest.json, /content/proposals/*.json, /content/unets/*.pt,
# /content/configs.json and ~/.kaggle/access_token. Output: /content/results/*.csv, PROGRESS.json, DONE.json, job.log.
import json
import subprocess
import sys
import tarfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/content")
TRAIN_DIR = WORK / "kaggle_data/MAGFiLO_1.0_Kaggle_2026/train"
RESULTS = WORK / "results"


def shell(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def setup():
    if not (TRAIN_DIR / "train_images").exists():
        shell("pip install -q 'kaggle==2.2.4'")
        shell(f"kaggle competitions download -c filament-segmentation-2026 -p {WORK}")
        archive = next(WORK.glob("filament-segmentation-2026*.zip"))
        shell(f"unzip -q -o {archive} -d {WORK / 'kaggle_data'}")
        archive.unlink()
    shell("pip install -q segmentation-models-pytorch")


def detect(spec, image_paths, device, seconds):
    from ultralytics import YOLO

    from src.config import DEFAULT_PIPELINE_SETTINGS
    from src.inference import cache_proposals, fuse_boxes
    detector = YOLO(str(WORK / "detectors" / spec["weights"]))
    settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": 0.01, "yolo_image_size": spec["imgsz"], "yolo_iou": spec.get("nms_iou", 0.20),
                "yolo_end2end": spec.get("end2end", False)}
    proposals = {}
    for path in image_paths:
        started = time.perf_counter()
        boxes, scores = fuse_boxes([cache_proposals(detector, [path], device, settings)[Path(path).stem]], [1.0], spec["iou"], 0.01)
        keep = scores >= spec["threshold"]
        proposals[Path(path).stem] = (np.asarray(boxes[keep], dtype=np.float32).reshape(-1, 4), np.asarray(scores[keep], dtype=np.float32))
        seconds[Path(path).stem] = time.perf_counter() - started
    return proposals


def main():
    finished = []
    try:
        setup()
        sys.path.insert(0, str(WORK))
        import pandas as pd
        import torch
        from pycocotools import mask as mask_utils

        from src.config import DEFAULT_PIPELINE_SETTINGS
        from src.data import load_records, verify_or_write_split
        from src.inference import create_submission, predict_instances
        from src.metrics import aggregate_official, annotation_to_rle, official_match_counts
        from src.models import ModelAverage, RotationTTA, load_checkpoint

        device = torch.device("cuda")
        _, records = load_records(TRAIN_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", TRAIN_DIR / "train_images")
        _, _, validation_records = verify_or_write_split(records, WORK / "split_manifest.json")
        by_image = defaultdict(list)
        for record in validation_records:
            by_image[record["physical_id"]].append(record)
        targets = {r["annotator_image_id"]: [annotation_to_rle(a, r["height"], r["width"]) for a in r["annotations"]] for r in validation_records}

        configs = json.loads((WORK / "configs.json").read_text())
        if any("detector" in c for c in configs):
            shell("pip install -q ultralytics")
        print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
        unets, proposal_sets = {}, {}
        RESULTS.mkdir(exist_ok=True)
        for config in configs:
            if (RESULTS / f"{config['label']}.csv").exists() or (RESULTS / f"{config['label']}_submission.csv").exists():
                finished.append(config["label"])
                continue
            for name, arch in config["unets"]:
                if name not in unets:
                    unets[name] = load_checkpoint(WORK / f"unets/{name}.pt", arch, device)[0]
            members = [RotationTTA(unets[name]).eval() if config["tta"] else unets[name] for name, _ in config["unets"]]
            unet = members[0] if len(members) == 1 else ModelAverage(members).eval()
            detector_seconds, stage2_seconds = {}, {}
            if "detector" in config:
                proposals = detect(config["detector"], [group[0]["image_path"] for group in by_image.values()], device, detector_seconds)
                (RESULTS / f"{config['label']}_boxes.json").write_text(json.dumps({k: {"boxes": b.tolist(), "scores": s.tolist()} for k, (b, s) in proposals.items()}))
            elif config["proposals"] not in proposal_sets:
                saved = json.loads((WORK / f"proposals/{config['proposals']}.json").read_text())
                proposal_sets[config["proposals"]] = {k: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for k, v in saved.items()}
            if "detector" not in config:
                proposals = proposal_sets[config["proposals"]]
            settings = {**DEFAULT_PIPELINE_SETTINGS, **config.get("pipeline", {})}
            if config.get("split") == "test":
                # A submission config: predict the 180 test images and write a validated Kaggle CSV instead of scores
                test_paths = sorted(next((WORK / "kaggle_data").rglob("test_images")).glob("*.jpeg"))
                assert len(test_paths) == 180
                submission, _ = create_submission(test_paths, None, unet, device, config["mask_threshold"], settings, proposals)
                submission.to_csv(RESULTS / f"{config['label']}_submission.csv", index=False)
                finished.append(f"{config['label']}_submission.csv")
                (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished, "total": len(configs)}) + "\n")
                print(f"CONFIG DONE {config['label']}: {len(submission)} test masks", flush=True)
                continue
            rows, dumped, started = [], {}, time.perf_counter()
            for physical_id, group in by_image.items():
                image_started = time.perf_counter()
                masks, _ = predict_instances(group[0]["image_path"], None, unet, device, config["mask_threshold"], settings, proposals=proposals[physical_id])
                stage2_seconds[physical_id] = time.perf_counter() - image_started
                predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
                dumped[physical_id] = [rle["counts"].decode("ascii") for rle in predicted]
                for record in group:
                    rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"], "predicted_instances": len(masks),
                                 **official_match_counts(predicted, targets[record["annotator_image_id"]])})
            frame = pd.DataFrame(rows)
            frame.to_csv(RESULTS / f"{config['label']}.csv", index=False)
            if config.get("dump"):
                (RESULTS / f"{config['label']}_predictions.json").write_text(json.dumps(dumped))
                finished.append(f"{config['label']}_predictions.json")
            if "detector" in config:
                (RESULTS / f"{config['label']}_timing.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "detector": detector_seconds, "stage2": stage2_seconds}))
                finished += [f"{config['label']}_timing.json", f"{config['label']}_boxes.json"]
            finished.append(config["label"])
            (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished, "total": len(configs)}) + "\n")
            # Stage 2 only (boxes are cached): wall seconds per physical image, including image loading and the scorer
            print(f"CONFIG DONE {config['label']}: PQ {aggregate_official(frame)['official_pq']:.4f}, {(time.perf_counter() - started) / len(by_image):.2f} s/image", flush=True)
        with tarfile.open(WORK / "results.tar.gz", "w:gz") as archive:
            archive.add(RESULTS, arcname="results")
        summary = {"status": "ok", "completed": finished}
    except Exception:
        summary = {"status": "failed", "error": traceback.format_exc(), "completed": finished}
        print(summary["error"], flush=True)
    (WORK / "DONE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {summary['status']}", flush=True)


if __name__ == "__main__":
    main()
