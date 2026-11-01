# Writes the two final checkpoints to weights/ for the public repo. GitHub rejects files over 100 MB, so the U-Net goes to fp16:
# inference already runs under fp16 autocast, and load_state_dict copies the values back into the model's fp32 parameters.
# l_ps is already fp16 with its optimizer stripped, so it is copied as is.
# Usage (from the repo root): .venv/bin/python scripts/export_final_weights.py
import shutil
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
UNET = PROJECT_DIR / "experiments/pipeline/20261028_005250_unet_train/b5_sym_s2.pt"
DETECTOR = PROJECT_DIR / "models/stage1/l_ps/weights/best.pt"
OUTPUT = PROJECT_DIR / "weights"

OUTPUT.mkdir(exist_ok=True)
shutil.copy2(DETECTOR, OUTPUT / "l_ps.pt")
checkpoint = torch.load(UNET, map_location="cpu", weights_only=False)
checkpoint["model_state_dict"] = {name: value.half() if value.is_floating_point() else value for name, value in checkpoint["model_state_dict"].items()}
torch.save(checkpoint, OUTPUT / "b5_sym_s2.pt")
for path in sorted(OUTPUT.glob("*.pt")):
    print(f"{path.name}: {path.stat().st_size / 2 ** 20:.1f} MB")
