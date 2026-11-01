# Step 2 of the detector-ensemble experiment: run the full pipeline (saved B5 champion, C0 settings) on fused detector
# boxes and compare each candidate to C0 with the paired bootstrap. Everything runs on CPU fp32 from cached boxes, so the
# control is C0's own boxes through the same CPU path (0.4303, not the 0.4300 GPU fp16 number).
# Usage (from the repo root, after scripts/experiments/detector_ensemble.py): .venv/bin/python scripts/experiments/ensemble_pq_eval.py
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.config import ANNOTATIONS_PATH, DEFAULT_PIPELINE_SETTINGS, EXPERIMENTS_DIR, PROCESSED_DATA_DIR, SPLIT_MANIFEST_PATH, TRAIN_IMAGES_DIR
from src.data import load_records, verify_or_write_split
from src.experiments import append_registry, new_run_dir
from src.inference import box_nms, fuse_boxes, predict_instances
from src.metrics import aggregate_official, annotation_to_rle, official_match_counts, paired_pq_delta
from src.models import ModelAverage, load_checkpoint

UNETS = {"b5": PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt",
         **{variant: PROJECT_DIR / f"experiments/pipeline/20260920_110658_tversky_sweep/{variant}/best.pt" for variant in ["a030_b070", "a050_b050", "a070_b030"]},
         **{name: PROJECT_DIR / f"experiments/pipeline/20261028_005250_unet_train/{name}.pt" for name in ["b5_sym_s1", "b5_sym_s2", "rx50_sym_s0"]}}
MASK_THRESHOLD = 0.45
CONTROL_CACHE = PROJECT_DIR / "experiments/diagnostics/c0_cpu_by_record.csv"
# label: (detectors, WBF cluster IoU, box NMS at 0.20 after fusion, fused-score threshold, U-Nets averaged); detectors ["c0"] means C0's own boxes
ALL_DETECTORS = ["n", "s", "s_nomosaic", "s1280", "m", "l", "x"]
HV24 = ALL_DETECTORS + [f"{n}_hflip" for n in ALL_DETECTORS] + [f"{n}_vflip" for n in ALL_DETECTORS] + ["y11m", "y11m_hflip", "y11m_vflip"]
CANDIDATES = {
    # Round 1 (2026-10-27), all against the C0 control (0.4303): detector fusion +0.0262 / +0.0247 / +0.0190 / +0.0179; two U-Nets
    # +0.0320 / +0.0325; four U-Nets tied two. Round 2-3 moved to the A100 (scripts/launch_stage2_gpu.py).
    # Round 3 CPU check (2026-10-28): does the fourth strong U-Net help on the best boxes? No TTA, relative comparison only.
    # Result: 3net 0.4694, 4net 0.4706, 4net - 3net +0.0012 [-0.0013, +0.0037] tied (run 20261028_015852_de_detector_ensemble)
    # "hv24y11m_t022_3net": (HV24, 0.40, False, 0.22, ["b5", "a050_b050", "b5_sym_s1"]),
    # "hv24y11m_t022_4net": (HV24, 0.40, False, 0.22, ["b5", "a050_b050", "b5_sym_s1", "b5_sym_s2"]),
    "hv24y11m_t022_5net_rx50": (HV24, 0.40, False, 0.22, ["b5", "a050_b050", "b5_sym_s1", "b5_sym_s2", "rx50_sym_s0"]),
}
device = torch.device("cpu")

_, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
_, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
by_image = defaultdict(list)
for record in validation_records:
    by_image[record["physical_id"]].append(record)


def load_boxes(path):
    saved = json.loads(Path(path).read_text())
    return {key: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for key, v in saved.items()}


def score(model, proposals):
    rows = []
    for number, (physical_id, group) in enumerate(by_image.items(), 1):
        masks, _ = predict_instances(group[0]["image_path"], None, model, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS, use_amp=False, proposals=proposals[physical_id])
        predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
        for record in group:
            targets = [annotation_to_rle(a, record["height"], record["width"]) for a in record["annotations"]]
            rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"], "predicted_instances": len(masks), **official_match_counts(predicted, targets)})
        if number % 25 == 0:
            print(f"  {number}/{len(by_image)} images", flush=True)
    return pd.DataFrame(rows)


loaded_unets = {}


def unet_for(names):
    for name in names:
        if name not in loaded_unets:
            loaded_unets[name] = load_checkpoint(UNETS[name], "b5", device)[0]
    return loaded_unets[names[0]] if len(names) == 1 else ModelAverage([loaded_unets[name] for name in names]).eval()


model = unet_for(["b5"])
if CONTROL_CACHE.exists():
    control = pd.read_csv(CONTROL_CACHE, dtype={"physical_id": str})
else:
    print("control: C0 boxes", flush=True)
    control = score(model, load_boxes(PROCESSED_DATA_DIR / "validation_proposals_yolo26x.json"))
    control.to_csv(CONTROL_CACHE, index=False)
control_summary = aggregate_official(control)
print(f"control official PQ {control_summary['official_pq']:.4f} (TP {control_summary['official_tp']}, FP {control_summary['official_fp']}, FN {control_summary['official_fn']})", flush=True)

cached = {name: load_boxes(path) for name in {m for members, *_ in CANDIDATES.values() for m in members if m != "c0"}
          for path in [PROCESSED_DATA_DIR / f"ensemble_proposals/validation_{name}.json"]}
c0_boxes = load_boxes(PROCESSED_DATA_DIR / "validation_proposals_yolo26x.json")
run_dir = None
for label, (members, iou_threshold, post_nms, threshold, unet_names) in CANDIDATES.items():
    if run_dir is None:
        run_id, run_dir = new_run_dir(EXPERIMENTS_DIR, "pipeline", "DE", "detector_ensemble")
    proposals = {}
    for physical_id in by_image:
        if members == ["c0"]:
            proposals[physical_id] = c0_boxes[physical_id]
            continue
        boxes, scores = fuse_boxes([cached[name][physical_id] for name in members], [1.0] * len(members), iou_threshold, 0.01)
        if post_nms:
            boxes, scores = box_nms(boxes, scores, DEFAULT_PIPELINE_SETTINGS["yolo_iou"])
        proposals[physical_id] = (boxes[scores >= threshold], scores[scores >= threshold])
    print(f"{label}: {sum(len(s) for _, s in proposals.values())} boxes, U-Nets {unet_names}", flush=True)
    frame = score(unet_for(unet_names), proposals)
    frame.to_csv(run_dir / f"{label}_by_record.csv", index=False)
    summary, delta = aggregate_official(frame), paired_pq_delta(control, frame)
    verdict = "candidate wins" if delta["paired_delta_ci_low"] > 0 else "control wins" if delta["paired_delta_ci_high"] < 0 else "tied"
    print(f"  PQ {summary['official_pq']:.4f} (TP {summary['official_tp']}, FP {summary['official_fp']}, FN {summary['official_fn']}); "
          f"paired delta {delta['paired_delta']:+.4f} [{delta['paired_delta_ci_low']:+.4f}, {delta['paired_delta_ci_high']:+.4f}] {verdict}", flush=True)
    settings = {"detectors": members, "wbf_iou": iou_threshold, "post_nms": post_nms, "fused_threshold": threshold, "unets": unet_names}
    (run_dir / f"{label}_summary.json").write_text(json.dumps({**settings, **summary, **delta, "control_pq": control_summary["official_pq"]}, indent=2) + "\n")
    append_registry(EXPERIMENTS_DIR / "registry.csv", {
        "run_id": f"{run_id}_{label}", "timestamp": datetime.now().isoformat(timespec="seconds"), "notebook": "scripts/experiments/ensemble_pq_eval.py", "section": "DE",
        "detector": "+".join(members), "detector_imgsz": 1280, "conf": threshold, "nms_iou": DEFAULT_PIPELINE_SETTINGS["yolo_iou"] if post_nms else "", "end2end": False,
        "unet_checkpoint": "+".join(str(UNETS[name].relative_to(PROJECT_DIR)) for name in unet_names), "mask_threshold": MASK_THRESHOLD, "tta": False, "other_settings_json": json.dumps(settings),
        "official_pq": summary["official_pq"], "sq": summary["official_sq"], "rq": summary["official_rq"], "tp": summary["official_tp"], "fp": summary["official_fp"],
        "fn": summary["official_fn"], "instances_per_image": frame.groupby("physical_id")["predicted_instances"].first().mean(), "device": "cpu", "precision": "fp32",
        "notes": f"WBF detector ensemble on CPU-cached boxes; paired delta vs C0 boxes on the same CPU path {delta['paired_delta']:+.4f} "
                 f"[{delta['paired_delta_ci_low']:+.4f}, {delta['paired_delta_ci_high']:+.4f}] ({verdict})",
    })
