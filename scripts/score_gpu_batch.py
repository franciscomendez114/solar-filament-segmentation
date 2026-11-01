# Scores a pulled stage2_gpu_job.py batch: official PQ per config, paired delta against the batch's own control
# ("ctrl", the current champion on identical GPU inputs, or --control LABEL), and one registry row per config.
# Usage (from the repo root): .venv/bin/python scripts/score_gpu_batch.py BATCH_NAME [--control LABEL] [--no-registry]
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.experiments import append_registry
from src.metrics import aggregate_official, paired_pq_delta

batch = sys.argv[1]
batch_dir = PROJECT_DIR / "experiments/pipeline" / batch
setup = json.loads((batch_dir / "configs.json").read_text())
configs = {c["label"]: c for c in setup["configs"]}
frames = {label: pd.read_csv(batch_dir / f"{label}.csv", dtype={"physical_id": str}) for label, c in configs.items() if c.get("split") != "test" and (batch_dir / f"{label}.csv").exists()}
control_label = sys.argv[sys.argv.index("--control") + 1] if "--control" in sys.argv else "ctrl"
control = frames[control_label]
gpu = next((line.split(" ", 1)[1].replace("NVIDIA ", "").split("-")[0].strip() for line in (batch_dir / "job.log").read_text().splitlines() if line.startswith("GPU ")), "A100") \
    if (batch_dir / "job.log").exists() else "A100"

rows = []
for label, frame in frames.items():
    summary = aggregate_official(frame)
    delta = paired_pq_delta(control, frame) if label != control_label else {"paired_delta": 0.0, "paired_delta_ci_low": 0.0, "paired_delta_ci_high": 0.0}
    verdict = "control" if label == control_label else "WIN" if delta["paired_delta_ci_low"] > 0 else "loss" if delta["paired_delta_ci_high"] < 0 else "tied"
    rows.append({"label": label, "pq": summary["official_pq"], "tp": summary["official_tp"], "fp": summary["official_fp"], "fn": summary["official_fn"],
                 "delta": delta["paired_delta"], "ci_low": delta["paired_delta_ci_low"], "ci_high": delta["paired_delta_ci_high"], "verdict": verdict})
    if "--no-registry" not in sys.argv:
        config = configs[label]
        detectors, wbf_iou, fused_threshold = setup["proposal_sets"][config["proposals"]]
        detector = config.get("detector", {})  # run on the VM instead of the cached proposal set
        append_registry(PROJECT_DIR / "experiments/registry.csv", {
            "run_id": f"{batch}_{label}", "timestamp": datetime.now().isoformat(timespec="seconds"), "notebook": "scripts/stage2_gpu_job.py", "section": "S2G",
            "detector": f"wbf{len(detectors)}", "detector_imgsz": detector.get("imgsz", 1280), "conf": detector.get("threshold", fused_threshold), "end2end": detector.get("end2end", False),
            "unet_checkpoint": "+".join(name for name, _ in config["unets"]), "mask_threshold": config["mask_threshold"], "tta": config["tta"],
            "other_settings_json": json.dumps({"proposals": config["proposals"], "wbf_iou": wbf_iou, **config["pipeline"], **({"detector": detector} if detector else {})}),
            "official_pq": summary["official_pq"], "sq": summary["official_sq"], "rq": summary["official_rq"], "tp": summary["official_tp"], "fp": summary["official_fp"],
            "fn": summary["official_fn"], "instances_per_image": frame.groupby("physical_id")["predicted_instances"].first().mean(), "device": gpu, "precision": "fp16",
            "notes": f"paired delta vs batch {control_label} {delta['paired_delta']:+.4f} [{delta['paired_delta_ci_low']:+.4f}, {delta['paired_delta_ci_high']:+.4f}] ({verdict})",
        })

table = pd.DataFrame(rows).sort_values("delta", ascending=False)
pd.set_option("display.width", 200)
print(table.round(4).to_string(index=False))
table.to_csv(batch_dir / "summary.csv", index=False)
