# The report's hyperparameter table from the batch-23 sweep (scripts/launch_stage2_gpu.py): every config changes one knob of the
# lean pipeline and is paired against lean on identical images. Adds the validation box count per config (from the fused
# proposal set or the VM's boxes) and the detector seconds per image for configs that ran their detector on the VM.
# Writes lean_sweep.csv into the batch folder.
# Usage (from the repo root): .venv/bin/python scripts/lean_sweep_report.py BATCH_NAME
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.metrics import aggregate_official, paired_pq_delta

batch_dir = PROJECT_DIR / "experiments/pipeline" / sys.argv[1]
configs = json.loads((batch_dir / "configs.json").read_text())
cache = json.loads((PROJECT_DIR / "data/processed/ensemble_proposals/validation_l_ps.json").read_text())
read = lambda label: pd.read_csv(batch_dir / f"{label}.csv", dtype={"physical_id": str})
lean = read("lean")
rows = []
for config in configs["configs"]:
    frame, summary = read(config["label"]), aggregate_official(read(config["label"]))
    delta = paired_pq_delta(lean, frame)
    verdict = "WIN" if delta["paired_delta_ci_low"] > 0 else "loss" if delta["paired_delta_ci_high"] < 0 else "tied"
    if "detector" in config:
        boxes = sum(len(item["scores"]) for item in json.loads((batch_dir / f"{config['label']}_boxes.json").read_text()).values())
        timing = json.loads((batch_dir / f"{config['label']}_timing.json").read_text())["detector"]
        seconds = float(np.mean(list(timing.values())[1:]))  # first image dropped (model load, CUDA warm-up)
    else:
        threshold = configs["proposal_sets"][config["proposals"]][2]
        boxes, seconds = sum(int((np.asarray(item["scores"]) >= threshold).sum()) for item in cache.values()), None
    rows.append({"label": config["label"], "pq": summary["official_pq"], "sq": summary["official_sq"], "tp": summary["official_tp"], "fp": summary["official_fp"],
                 "fn": summary["official_fn"], "boxes": boxes, "detector_s": seconds, "delta": delta["paired_delta"], "ci_low": delta["paired_delta_ci_low"],
                 "ci_high": delta["paired_delta_ci_high"], "verdict": verdict})
table = pd.DataFrame(rows)
table.to_csv(batch_dir / "lean_sweep.csv", index=False)
pd.set_option("display.width", 200)
print(table.round(4).to_string(index=False))
