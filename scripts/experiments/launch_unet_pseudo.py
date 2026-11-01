# Starts a round-7 unet_train_job.py run that trains U-Net students on train + pseudo-labelled masks of the verified GONG pool.
# Without --masks the VM first draws the masks with the teacher U-Nets (b5 + a050 + b5_sym_s1 with TTA, the public-0.39 average)
# inside the round-5 teacher boxes (pseudo_boxes.json, hv24 fused >= 0.30); with --masks it reuses a pulled pseudo_masks.json.
# Usage (from the repo root): .venv/bin/python scripts/experiments/launch_unet_pseudo.py SESSION RUNS_JSON [--masks PSEUDO_MASKS_JSON]
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from colab_launch import start_job

POOL = PROJECT_DIR / "colab_transfer/round5/gong_pool.tar"  # built by scripts/launch_pseudo.py
BOXES = PROJECT_DIR / "experiments/stage1/20261028_073435_pseudo_pl1/pseudo_boxes.json"
TEACHERS = {"b5": "models/stage2/crop_unet_b5/best_crop_unet_b5.pt", "a050": "experiments/pipeline/20260920_110658_tversky_sweep/a050_b050/best.pt",
            "b5_sym_s1": "experiments/pipeline/20261028_005250_unet_train/b5_sym_s1.pt"}

session, runs = sys.argv[1], Path(sys.argv[2])
processed = PROJECT_DIR / "data/processed"
uploads = [(runs, "/content/unet_runs.json"), (POOL, "/content/gong_pool.tar"),
           (processed / "hard_negatives/hard_negatives_yolo26x_conf005.json", "/content/hard_negatives_yolo26x_conf005.json"),
           (processed / "validation_proposals_yolo26x.json", "/content/validation_proposals_yolo26x.json")]
if "--masks" in sys.argv:
    uploads.append((Path(sys.argv[sys.argv.index("--masks") + 1]), "/content/pseudo_masks.json"))
else:
    uploads.append((BOXES, "/content/pseudo_boxes.json"))
    uploads += [(PROJECT_DIR / path, f"/content/teacher_unets/{name}.pt") for name, path in TEACHERS.items()]
start_job(session, "A100", PROJECT_DIR / "scripts/unet_train_job.py", uploads)
