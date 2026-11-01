# Tversky alpha/beta sweep for Stage 2, run detached inside a Colab L4 VM.
# Why: 19.7% of validation filaments are found but drawn with median predicted/true area 1.77x, and lifting only
# that bucket past IoU 0.50 is worth PQ 0.43 -> 0.555. alpha 0.30 / beta 0.70 penalises a missed pixel 2.3x more
# than an included background pixel, which is the obvious way to teach a model to bleed outward. Three runs,
# identical except the asymmetry, selected on official PQ. Global thresholds cannot fix this (masks that already
# pass have area ratio 1.09), so the fix has to be in the loss.
# The VM needs /content/src/, split_manifest.json, the yolo26x hard-negative cache, the validation proposal cache,
# and ~/.kaggle/access_token. Everything it prints lands in /content/sweep.log.
import json
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

import numpy as np

VARIANTS = [("a030_b070", 0.30, 0.70), ("a050_b050", 0.50, 0.50), ("a070_b030", 0.70, 0.30)]
MASK_THRESHOLD = 0.45  # what inference deploys; the validator scores at the deployed threshold
WORK = Path("/content")
RAW_DIR = WORK / "kaggle_data"
TRAIN_DIR = RAW_DIR / "MAGFiLO_1.0_Kaggle_2026/train"
SWEEP_DIR = WORK / "sweep"
PROGRESS = WORK / "PROGRESS.json"
DONE_FLAG = WORK / "DONE.json"


def shell(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def setup():
    if not (TRAIN_DIR / "train_images").exists():
        shell("pip install -q 'kaggle==2.2.4'")
        shell(f"kaggle competitions download -c filament-segmentation-2026 -p {WORK}")
        archive = next(WORK.glob("filament-segmentation-2026*.zip"))
        shell(f"unzip -q -o {archive} -d {RAW_DIR}")
        archive.unlink()
        Path.home().joinpath(".kaggle/access_token").unlink(missing_ok=True)  # only needed for that one download
    shell("pip install -q segmentation-models-pytorch")
    print(f"Train images: {len(list((TRAIN_DIR / 'train_images').glob('*.jpeg')))}", flush=True)


def build_everything():
    sys.path.insert(0, str(WORK))
    import torch
    from torch.utils.data import DataLoader

    from src.config import DEFAULT_PIPELINE_SETTINGS, TRAINING_SETTINGS
    from src.data import FilamentCropDataset, load_records, verify_or_write_split

    images_dir = TRAIN_DIR / "train_images"
    _, records = load_records(TRAIN_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", images_dir)
    _, train_records, validation_records = verify_or_write_split(records, WORK / "split_manifest.json")  # asserts the frozen split

    settings = {**DEFAULT_PIPELINE_SETTINGS, **TRAINING_SETTINGS}
    positives = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"],
                  "annotation": annotation, "proposal_box": None} for r in train_records for annotation in r["annotations"]]
    hard_negatives = json.loads((WORK / "hard_negatives_yolo26x_conf005.json").read_text())
    negatives = [{"physical_id": x["physical_id"], "file_name": x["file_name"], "image_path": images_dir / x["file_name"],
                  "annotation": None, "proposal_box": x["proposal_box"]} for x in hard_negatives if x["score"] >= settings["negative_mining_floor"]]
    validation_samples = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"],
                           "annotation": annotation, "proposal_box": None} for r in validation_records for annotation in r["annotations"]]
    print(f"{len(positives)} positive crops, {len(negatives)} hard negatives, {len(validation_samples)} validation crops", flush=True)

    train_loader = DataLoader(FilamentCropDataset(positives + negatives, settings, augment=True), batch_size=settings["batch_size"],
                              shuffle=True, num_workers=settings["num_workers"], pin_memory=True, drop_last=True)
    validation_loader = DataLoader(FilamentCropDataset(validation_samples, settings, augment=False), batch_size=settings["batch_size"],
                                   shuffle=False, num_workers=settings["num_workers"], pin_memory=True)

    cached = json.loads((WORK / "validation_proposals_yolo26x.json").read_text())
    proposals = {key: (np.asarray(value["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(value["scores"], dtype=np.float32)) for key, value in cached.items()}
    return torch.device("cuda"), settings, train_loader, validation_loader, validation_records, proposals


def decompose(unet, validation_records, proposals, device, settings):
    # The sweep's second signal: where the missed filaments sit, and how oversized the salvageable ones are.
    from collections import defaultdict

    from pycocotools import mask as mask_utils

    from src.inference import predict_instances
    from src.metrics import annotation_to_rle

    by_image = defaultdict(list)
    for record in validation_records:
        by_image[record["physical_id"]].append(record)
    best_ious, area_ratios = [], []
    for physical_id, group in by_image.items():
        masks, _ = predict_instances(group[0]["image_path"], None, unet, device, MASK_THRESHOLD, settings, proposals=proposals[physical_id])
        predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
        predicted_areas = np.asarray(mask_utils.area(predicted), dtype=np.float64) if predicted else np.zeros(0)
        for record in group:
            targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
            target_areas = np.asarray(mask_utils.area(targets), dtype=np.float64)
            if not predicted or not targets:
                best_ious.extend([0.0] * len(targets))
                area_ratios.extend([np.nan] * len(targets))
                continue
            ious = mask_utils.iou(predicted, targets, [0] * len(targets))
            for index in range(len(targets)):
                column = ious[:, index]
                winner = int(column.argmax())
                best_ious.append(float(column[winner]))
                area_ratios.append(float(predicted_areas[winner] / max(target_areas[index], 1.0)))
    best_ious, area_ratios = np.asarray(best_ious), np.asarray(area_ratios)
    matched, middle, missed = best_ious > 0.50, (best_ious > 0.05) & (best_ious <= 0.50), best_ious <= 0.05
    return {"filaments": int(len(best_ious)),
            "matched_share": float(matched.mean()), "matched_mean_iou": float(best_ious[matched].mean()) if matched.any() else 0.0,
            "matched_median_area_ratio": float(np.nanmedian(area_ratios[matched])) if matched.any() else 0.0,
            "middle_share": float(middle.mean()), "middle_mean_iou": float(best_ious[middle].mean()) if middle.any() else 0.0,
            "middle_median_area_ratio": float(np.nanmedian(area_ratios[middle])) if middle.any() else 0.0,
            "missed_share": float(missed.mean())}


def run_variant(name, alpha, beta, device, settings, train_loader, validation_loader, validation_records, proposals):
    import torch

    from src.inference import official_pq_for_model
    from src.models import build_model, load_checkpoint, train_crop_unet

    variant_dir = SWEEP_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_settings = {**settings, "tversky_alpha": alpha, "tversky_beta": beta}
    torch.manual_seed(0)  # same initialisation and augmentation stream for all three, so only the loss differs
    model = build_model("b5").to(device)
    validator = lambda unet: official_pq_for_model(unet, validation_records, proposals, device, MASK_THRESHOLD, settings)
    history = train_crop_unet(model, train_loader, validation_loader, device, variant_settings,
                              variant_dir / "best.pt", variant_dir / "history.csv", validator)

    best_model, checkpoint = load_checkpoint(variant_dir / "best.pt", "b5", device)
    summary = {"variant": name, "tversky_alpha": alpha, "tversky_beta": beta, "epochs_run": int(len(history)),
               "best_epoch": int(checkpoint["epoch"]), "official_pq": float(checkpoint["official_pq"]),
               "official_tp": int(checkpoint["official_tp"]), "official_fp": int(checkpoint["official_fp"]),
               "official_fn": int(checkpoint["official_fn"]), "validation_dice": float(checkpoint["validation_dice"]),
               **decompose(best_model, validation_records, proposals, device, settings)}
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with tarfile.open(WORK / f"{name}.tar.gz", "w:gz") as archive:
        archive.add(variant_dir, arcname=name)
    del model, best_model
    torch.cuda.empty_cache()
    return summary


def main():
    results = []
    try:
        setup()
        context = build_everything()
        for name, alpha, beta in VARIANTS:
            print(f"\n===== variant {name}: alpha {alpha}, beta {beta} =====", flush=True)
            results.append(run_variant(name, alpha, beta, *context))
            PROGRESS.write_text(json.dumps({"completed": results}, indent=2) + "\n")
            print(f"VARIANT DONE {json.dumps(results[-1])}", flush=True)
        summary = {"status": "ok", "results": results}
    except Exception:
        summary = {"status": "failed", "error": traceback.format_exc(), "results": results}
        print(summary["error"], flush=True)
    DONE_FLAG.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {json.dumps(summary)[:600]}", flush=True)


if __name__ == "__main__":
    main()
