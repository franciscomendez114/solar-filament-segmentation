import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


REGISTRY_COLUMNS = [
    "run_id", "timestamp", "notebook", "section", "detector", "detector_imgsz", "conf", "nms_iou", "end2end",
    "unet_checkpoint", "mask_threshold", "tta", "other_settings_json", "official_pq", "sq", "rq", "tp", "fp", "fn",
    "instances_per_image", "pq_ci_low", "pq_ci_high", "device", "precision", "kaggle_public", "notes",
]


def new_run_dir(experiments_dir, area, section, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{section.lower().replace('.', '-')}_{name}"
    run_dir = Path(experiments_dir) / area / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def save_run_files(run_dir, settings, predictions, by_annotator, summary):
    run_dir = Path(run_dir)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    with (run_dir / "predictions.jsonl").open("w") as file:
        for entry in predictions:
            file.write(json.dumps(entry) + "\n")
    pd.DataFrame(by_annotator).to_csv(run_dir / "by_annotator.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def append_registry(registry_path, row):
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()
    clean = {column: row.get(column, "") for column in REGISTRY_COLUMNS}
    with registry_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")  # match the existing rows; csv defaults to \r\n
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


def load_registry(registry_path):
    path = Path(registry_path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)

