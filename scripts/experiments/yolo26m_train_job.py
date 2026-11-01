# Trains YOLO26m on the frozen split inside a Colab L4 VM. It is started detached (nohup) so the Colab kernel
# stays free; everything it prints lands in /content/train.log, which scripts/experiments/watch_yolo26m.py tails once a minute.
# Recipe is the saved YOLO26x one (1280, mosaic 0.5, fliplr 0.5, 300 epochs, patience 30) so the two are comparable.
# The VM needs /content/src/{__init__,config,data}.py, /content/split_manifest.json and ~/.kaggle/access_token first.
import json
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

RUN_NAME = "yolo26m_filament_detection"
# Fixed batch, not the saved runs' batch=0.9: AutoBatch picked 2 on the L4 (and only used 4.8 of 22 GB), and
# BatchNorm statistics over 2 images are too noisy. nbs=64 keeps the effective optimizer batch the same.
BATCH = 8
WORK = Path("/content")
RAW_DIR = WORK / "kaggle_data"
TRAIN_DIR = RAW_DIR / "MAGFiLO_1.0_Kaggle_2026/train"
DATASET_DIR = WORK / "yolo_dataset"
RUNS_DIR = WORK / "runs"
RUN_DIR = RUNS_DIR / RUN_NAME
ARTIFACT = WORK / "yolo26m_artifacts.tar.gz"
DONE_FLAG = WORK / "DONE.json"


def shell(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def download_data():
    if (TRAIN_DIR / "train_images").exists():
        print("Competition data already on the VM", flush=True)
        return
    shell("pip install -q 'kaggle==2.2.4'")
    shell(f"kaggle competitions download -c filament-segmentation-2026 -p {WORK}")
    archive = next(WORK.glob("filament-segmentation-2026*.zip"))
    shell(f"unzip -q -o {archive} -d {RAW_DIR}")
    archive.unlink()
    Path.home().joinpath(".kaggle/access_token").unlink(missing_ok=True)  # only needed for that one download
    print(f"Train images: {len(list((TRAIN_DIR / 'train_images').glob('*.jpeg')))}", flush=True)


def build_dataset():
    sys.path.insert(0, str(WORK))
    from src.data import build_yolo_dataset, load_records, verify_or_write_split

    annotations_path = TRAIN_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco, records = load_records(annotations_path, TRAIN_DIR / "train_images")
    manifest, _, _ = verify_or_write_split(records, WORK / "split_manifest.json")  # asserts the frozen split
    counts = build_yolo_dataset(coco, TRAIN_DIR / "train_images", manifest, DATASET_DIR)
    print(f"YOLO dataset: {counts}", flush=True)


def train():
    shell("pip install -q ultralytics")
    import ultralytics
    from ultralytics import YOLO

    print(f"ultralytics {ultralytics.__version__}", flush=True)
    detector = YOLO("yolo26m.pt")
    detector.train(data=str(DATASET_DIR / "data.yaml"), epochs=300, patience=30, imgsz=1280, batch=BATCH, device=0,
                   workers=8, seed=0, deterministic=True, mosaic=0.5, fliplr=0.5, flipud=0.0, degrees=0.0,
                   close_mosaic=10, project=str(RUNS_DIR), name=RUN_NAME, exist_ok=False, plots=True)


def best_metrics():
    import pandas as pd

    frame = pd.read_csv(RUN_DIR / "results.csv")
    best = frame.loc[frame["metrics/mAP50-95(B)"].idxmax()]
    return {"epochs_run": int(frame["epoch"].max()), "best_epoch": int(best["epoch"]),
            "precision": float(best["metrics/precision(B)"]), "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]), "map50_95": float(best["metrics/mAP50-95(B)"]),
            "train_seconds": float(frame["time"].max())}


def main():
    try:
        download_data()
        build_dataset()
        train()
        with tarfile.open(ARTIFACT, "w:gz") as archive:
            archive.add(RUN_DIR, arcname=RUN_NAME)
        summary = {"status": "ok", "artifact": str(ARTIFACT), "artifact_bytes": ARTIFACT.stat().st_size, **best_metrics()}
    except Exception:
        summary = {"status": "failed", "error": traceback.format_exc()}
        print(summary["error"], flush=True)
    DONE_FLAG.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {json.dumps(summary)[:400]}", flush=True)


if __name__ == "__main__":
    main()
