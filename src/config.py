import sys
from pathlib import Path


SEED = 42
PROJECT_NAME = "Solar Filament Segmentation Challenge"
IS_COLAB = "google.colab" in sys.modules
PROJECT_DIR = Path("/content/drive/MyDrive/Google Colab") / PROJECT_NAME if IS_COLAB else Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train/train_images"
TEST_IMAGES_DIR = RAW_DATA_DIR / "test/test_images"
ANNOTATIONS_PATH = RAW_DATA_DIR / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "split_manifest.json"
MODELS_DIR = PROJECT_DIR / "models"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"

DEFAULT_PIPELINE_SETTINGS = {
    "crop_size": 512, "min_crop_side": 64, "crop_expansion": 1.50,
    "minimum_mask_area": 20, "dedup_iou": 0.50, "dedup_containment": 0.80,
    "yolo_confidence": 0.20, "yolo_iou": 0.20, "yolo_image_size": 1280, "yolo_end2end": False,
    "max_yolo_detections": 300, "max_crops_per_image": 100, "inference_batch_size": 24,
}

TRAINING_SETTINGS = {
    "validation_fraction": 0.15, "train_expansion_min": 1.25, "train_expansion_max": 1.75, "center_jitter": 0.20,
    "hard_negative_ratio": 0.25, "max_negatives_per_image": 4, "negative_iou_limit": 0.05, "negative_mining_floor": 0.05,
    "batch_size": 16, "accumulation_steps": 2, "num_epochs": 30, "warmup_epochs": 2, "early_stopping_patience": 8,
    "learning_rate": 3e-4, "min_learning_rate": 1e-6, "weight_decay": 1e-4, "num_workers": 6,
    "bce_weight": 0.25, "tversky_weight": 0.75, "tversky_alpha": 0.30, "tversky_beta": 0.70, "tversky_gamma": 0.75,
}


# Chosen 2026-10-27 (local official PQ 0.4628 vs C0 0.4303): every Stage 1 detector at its training image size, fused by WBF, feeding two averaged B5 U-Nets
ENSEMBLE_DETECTORS = {"n": ("yolo26n_filament_detection", 1024), "s": ("yolo26s_filament_detection", 1024), "s_nomosaic": ("yolo26s_1024_nomosaic", 1024),
                      "s1280": ("yolo26s_1280_mosaic", 1280), "m": ("yolo26m_filament_detection", 1280), "l": ("yolo26l_filament_detection", 1280),
                      "x": ("yolo26x_filament_detection", 1280)}
ENSEMBLE_UNETS = ["models/stage2/crop_unet_b5/best_crop_unet_b5.pt", "experiments/pipeline/20260920_110658_tversky_sweep/a050_b050/best.pt"]
ENSEMBLE_SETTINGS = {"cache_confidence": 0.01, "wbf_iou": 0.40, "fused_threshold": 0.22, "mask_threshold": 0.45}
# Every cached Stage 1 view: member name -> (folder, weights file, image size, flip). Plain names are the round-1 detectors;
# _hflip/_vflip mirror the image (TTA), _up runs 256 px larger, _last uses the final-epoch checkpoint (round 2, 2026-10-28).
DETECTOR_MEMBERS = {name: (folder, "best.pt", size, None) for name, (folder, size) in ENSEMBLE_DETECTORS.items()}
for _name, (_folder, _size) in ENSEMBLE_DETECTORS.items():
    DETECTOR_MEMBERS.update({f"{_name}_hflip": (_folder, "best.pt", _size, "h"), f"{_name}_vflip": (_folder, "best.pt", _size, "v"),
                             f"{_name}_up": (_folder, "best.pt", _size + 256, None)})
    if _name != "x":  # yolo26x last.pt was never synced from Drive
        DETECTOR_MEMBERS[f"{_name}_last"] = (_folder, "last.pt", _size, None)
# Round 3 (2026-10-28): detectors trained for ensemble diversity by scripts/detector_train_job.py (A100); their caches come from the VM
# Round 4: yolo26s trained and run above 1280 (flipud 0.5)
for _name, _folder, _size in [("y11m", "yolo11m_flipud", 1280), ("y26s_fu", "yolo26s_flipud", 1280), ("s1536", "yolo26s_1536_fu", 1536), ("s2048", "yolo26s_2048_fu", 2048)]:
    DETECTOR_MEMBERS.update({_name: (_folder, "best.pt", _size, None), f"{_name}_hflip": (_folder, "best.pt", _size, "h"), f"{_name}_vflip": (_folder, "best.pt", _size, "v")})
