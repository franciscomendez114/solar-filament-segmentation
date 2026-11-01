# Trains extra Stage 2 U-Nets for the averaged ensemble inside a Colab GPU VM. Why: averaging a second B5 gave
# +0.0077 [+0.0030, +0.0134] while every post-processing knob tied, so Stage 2 variance is the lever; the four-U-Net
# average tied two only because the extra members were the weaker asymmetric-Tversky runs. New members are strong
# (symmetric Tversky, which peaks around epoch 3) and differ in seed or encoder. Selection is official PQ on the
# cached C0 boxes, as in the September sweep.
# The VM needs /content/src/, split_manifest.json, the yolo26x hard-negative cache, the validation proposal cache and
# ~/.kaggle/access_token. Output: /content/results/<name>.pt + <name>_history.csv, PROGRESS.json, DONE.json.
import json
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

# The run list comes from /content/unet_runs.json: {name: {"encoder": ..., "seed": ..., "full_data": bool, **settings overrides}}.
# Round 3 (2026-10-28) ran b5_sym_s1 / b5_sym_s2 (efficientnet-b5, seeds 1/2) and rx50_sym_s0 (resnext50_32x4d), symmetric
# Tversky, selected on official PQ. A full_data run also trains on the validation filaments and needs "stop_after_epoch"
# (the epoch its split run chose), since nothing inside training can select once validation is in it.
# Round 7 pseudo-labels masks on the verified GONG pool (scripts/gong_pool.py, scripts/check_gong_pool.py) when /content/gong_pool.tar
# is present: the teacher U-Nets in /content/teacher_unets/ (b5 + a050 + b5_sym_s1, 8-way TTA, the public-0.39 average) draw masks
# inside the teacher detector boxes /content/pseudo_boxes.json, and each final mask becomes a polygon annotation in
# results/pseudo_masks.json; a VM given /content/pseudo_masks.json skips that step. A run with "pseudo": true adds the pool crops.
MASK_THRESHOLD = 0.45
WORK = Path("/content")
RAW_DIR = WORK / "kaggle_data"
RESULTS = WORK / "results"


def shell(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def main():
    finished = []
    try:
        if not RAW_DIR.exists():
            shell("pip install -q 'kaggle==2.2.4'")
            shell(f"kaggle competitions download -c filament-segmentation-2026 -p {WORK}")
            archive = next(WORK.glob("filament-segmentation-2026*.zip"))
            shell(f"unzip -q -o {archive} -d {RAW_DIR}")
            archive.unlink()
        shell("pip install -q segmentation-models-pytorch")
        sys.path.insert(0, str(WORK))
        import torch
        from torch.utils.data import DataLoader

        from src.config import DEFAULT_PIPELINE_SETTINGS, TRAINING_SETTINGS
        from src.data import FilamentCropDataset, load_records, verify_or_write_split
        from src.inference import official_pq_for_model
        from src.models import build_model, train_crop_unet

        device = torch.device("cuda")
        train_dir = next(RAW_DIR.rglob("train_images")).parent
        images_dir = train_dir / "train_images"
        _, records = load_records(train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", images_dir)
        _, train_records, validation_records = verify_or_write_split(records, WORK / "split_manifest.json")
        settings = {**DEFAULT_PIPELINE_SETTINGS, **TRAINING_SETTINGS, "tversky_alpha": 0.5, "tversky_beta": 0.5, "early_stopping_patience": 4}
        crops = lambda chosen: [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"], "annotation": a, "proposal_box": None}
                                for r in chosen for a in r["annotations"]]
        hard_negatives = json.loads((WORK / "hard_negatives_yolo26x_conf005.json").read_text())
        negatives = [{"physical_id": x["physical_id"], "file_name": x["file_name"], "image_path": images_dir / x["file_name"], "annotation": None,
                      "proposal_box": x["proposal_box"]} for x in hard_negatives if x["score"] >= settings["negative_mining_floor"]]
        validation_samples = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"], "annotation": a, "proposal_box": None}
                              for r in validation_records for a in r["annotations"]]
        print(f"{len(crops(train_records))} train-split positive crops, {len(negatives)} hard negatives", flush=True)
        cached = json.loads((WORK / "validation_proposals_yolo26x.json").read_text())
        proposals = {k: (np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(v["scores"], dtype=np.float32)) for k, v in cached.items()}
        validator = lambda unet: official_pq_for_model(unet, validation_records, proposals, device, MASK_THRESHOLD, settings)
        RESULTS.mkdir(exist_ok=True)

        pool = []
        if (WORK / "gong_pool.tar").exists():
            import cv2
            from src.inference import predict_instances
            from src.models import ModelAverage, RotationTTA, load_checkpoint
            gong_dir, masks_path = WORK / "gong", RESULTS / "pseudo_masks.json"
            if not gong_dir.exists():
                gong_dir.mkdir()
                shell(f"tar -xf {WORK / 'gong_pool.tar'} -C {gong_dir}")
            if not masks_path.exists() and (WORK / "pseudo_masks.json").exists():
                shutil.copy2(WORK / "pseudo_masks.json", masks_path)
            if not masks_path.exists():
                teacher = ModelAverage([RotationTTA(load_checkpoint(path, "b5", device)[0]).eval() for path in sorted((WORK / "teacher_unets").glob("*.pt"))]).eval()
                pseudo = {}
                for stem, item in json.loads((WORK / "pseudo_boxes.json").read_text()).items():
                    boxes, scores = np.asarray(item["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(item["scores"], dtype=np.float32)
                    masks = predict_instances(gong_dir / f"{stem}.jpeg", None, teacher, device, MASK_THRESHOLD, settings, proposals=(boxes, scores))[0] if len(boxes) else []
                    annotations = []
                    for mask in masks:
                        contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        polygons = [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]
                        if polygons:
                            x, y, width, height = cv2.boundingRect(np.asarray(mask, dtype=np.uint8))
                            annotations.append({"segmentation": polygons, "bbox": [x, y, width, height]})
                    pseudo[stem] = annotations
                masks_path.write_text(json.dumps(pseudo))
                finished.append("pseudo_masks.json")
                (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished}) + "\n")
                del teacher
                torch.cuda.empty_cache()
            pool = [{"physical_id": stem, "file_name": f"{stem}.jpeg", "image_path": gong_dir / f"{stem}.jpeg", "annotation": a, "proposal_box": None}
                    for stem, annotations in json.loads(masks_path.read_text()).items() for a in annotations]
            print(f"{len(pool)} pseudo-labelled pool crops", flush=True)

        for name, spec in json.loads((WORK / "unet_runs.json").read_text()).items():
            if (RESULTS / f"{name}.pt").exists():
                continue
            spec = dict(spec)
            encoder, seed, full_data, use_pseudo = spec.pop("encoder"), spec.pop("seed"), spec.pop("full_data", False), spec.pop("pseudo", False)
            positives = crops(train_records + validation_records if full_data else train_records)
            if use_pseudo:
                assert pool, "pseudo run without a pseudo-labelled pool"
                positives = positives + pool
            print(f"\n===== {name}: encoder {encoder}, seed {seed}, full data {full_data}, {len(positives)} positives, overrides {spec} =====", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            variant_settings = {**settings, **spec, "encoder_name": encoder}
            loaders = [DataLoader(FilamentCropDataset(positives + negatives, variant_settings, augment=True), batch_size=settings["batch_size"], shuffle=True,
                                  num_workers=settings["num_workers"], pin_memory=True, drop_last=True),
                       DataLoader(FilamentCropDataset(validation_samples, variant_settings, augment=False), batch_size=settings["batch_size"], shuffle=False,
                                  num_workers=settings["num_workers"], pin_memory=True)]
            model = build_model("b5", {"encoder_name": encoder}).to(device)
            train_crop_unet(model, *loaders, device, variant_settings, WORK / f"{name}.pt", RESULTS / f"{name}_history.csv", None if full_data else validator)
            shutil.move(WORK / f"{name}.pt", RESULTS / f"{name}.pt")
            finished += [f"{name}_history.csv", f"{name}.pt"]
            (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished}) + "\n")
            print(f"VARIANT DONE {name}", flush=True)
            del model
            torch.cuda.empty_cache()
        summary = {"status": "ok", "completed": finished}
    except Exception:
        summary = {"status": "failed", "error": traceback.format_exc(), "completed": finished}
        print(summary["error"], flush=True)
    (WORK / "DONE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {summary['status']}", flush=True)


if __name__ == "__main__":
    main()
