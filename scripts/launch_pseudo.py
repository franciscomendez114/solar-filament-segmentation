# Starts a round-5 detector_train_job.py run that trains on train + pseudo-labeled GONG pool images (scripts/gong_pool.py).
# Without --labels the VM first labels the pool with the split-trained hv24 teachers (the public-0.39 fusion, boxes with
# fused score >= 0.30: box precision 0.75 / recall 0.61 on validation); with --labels it reuses pulled pseudo_boxes.json.
# Teacher checkpoints are uploaded with the optimizer stripped (yolo26x is 450 MB with it), staged in colab_transfer/.
# The pool goes up as one tar of the grayscale JPEGs from scripts/gong_pool.py --download.
# Usage (from the repo root): .venv/bin/python scripts/launch_pseudo.py SESSION RUNS_JSON [--labels PSEUDO_BOXES_JSON] [--gpu L4]
import json
import sys
import tarfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from colab_launch import start_job
from src.config import DETECTOR_MEMBERS

THRESHOLD = 0.30
TEACHERS = [f"{name}{suffix}" for name in ["n", "s", "s_nomosaic", "s1280", "m", "l", "x", "y11m"] for suffix in ["", "_hflip", "_vflip"]]
STAGE = PROJECT_DIR / "colab_transfer/round5"

session, runs = sys.argv[1], Path(sys.argv[2])
STAGE.mkdir(parents=True, exist_ok=True)
if not (STAGE / "gong_pool.tar").exists():
    with tarfile.open(STAGE / "gong_pool.tar", "w") as archive:
        for path in sorted((PROJECT_DIR / "data/raw/gong_pool").glob("*.jpeg")):
            archive.add(path, arcname=path.name)
uploads = [(runs, "/content/detector_runs.json"), (STAGE / "gong_pool.tar", "/content/gong_pool.tar")]
if "--labels" in sys.argv:
    uploads.append((Path(sys.argv[sys.argv.index("--labels") + 1]), "/content/pseudo_boxes.json"))
else:
    from ultralytics.utils.torch_utils import strip_optimizer
    (STAGE / "teachers").mkdir(parents=True, exist_ok=True)
    members = []
    for name in TEACHERS:
        folder, weights, image_size, flip = DETECTOR_MEMBERS[name]
        stripped = STAGE / "teachers" / f"{folder}.pt"
        if not stripped.exists():
            strip_optimizer(PROJECT_DIR / f"models/stage1/{folder}/weights/{weights}", str(stripped))
        members.append([name, stripped.name, image_size, flip])
    (STAGE / "teachers.json").write_text(json.dumps({"threshold": THRESHOLD, "members": members}, indent=1))
    uploads.append((STAGE / "teachers.json", "/content/teachers.json"))
    uploads += [(path, f"/content/teachers/{path.name}") for path in sorted((STAGE / "teachers").glob("*.pt"))]
start_job(session, sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "A100", PROJECT_DIR / "scripts/detector_train_job.py", uploads)
