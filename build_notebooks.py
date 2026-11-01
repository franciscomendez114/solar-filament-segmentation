import json
import textwrap
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "notebooks"


def lines(text):
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def common_setup(title, purpose):
    return [
        markdown(f"# {title}\n\n**One job:** {purpose}\n\nRun from top to bottom. Slow actions are off by default."),
        code("%pip install -q ultralytics pycocotools segmentation-models-pytorch"),
        code('''
            from google.colab import drive
            drive.mount("/content/drive")

            import json
            import sys
            from collections import defaultdict
            from datetime import datetime
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch

            PROJECT_DIR = Path("/content/drive/MyDrive/Google Colab/Solar Filament Segmentation Challenge")
            sys.path.insert(0, str(PROJECT_DIR))
            %load_ext autoreload
            %autoreload 2

            from src.config import DEFAULT_PIPELINE_SETTINGS, TRAINING_SETTINGS
            from src.data import copy_raw_data_to_local, load_records, make_split, verify_or_write_split

            LOCAL_DATA_DIR = copy_raw_data_to_local(PROJECT_DIR / "data/raw")
            TRAIN_IMAGES_DIR = LOCAL_DATA_DIR / "train/train_images"
            TEST_IMAGES_DIR = LOCAL_DATA_DIR / "test/test_images"
            ANNOTATIONS_PATH = LOCAL_DATA_DIR / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
            SPLIT_MANIFEST_PATH = PROJECT_DIR / "data/processed/split_manifest.json"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")
            if device.type == "cuda":
                print(torch.cuda.get_device_name(0))
        ''')
    ]


def write_notebook(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(json.dumps(notebook, indent=1) + "\n")


cells = common_setup("01 — Data, EDA, and frozen split", "inspect the competition data and verify the one shared split")
cells += [
    code("RUN_HUMAN_AGREEMENT = False  # Slow COCO-RLE comparison; the saved result is shown below"),
    markdown("## 1. Data inventory and integrity\n\nFacts here are computed from the raw files, not copied from the handoff."),
    code('''
        from PIL import Image

        coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
        train_images = sorted(TRAIN_IMAGES_DIR.glob("*.jpeg"))
        test_images = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
        sizes = {Image.open(path).size for path in train_images[:20] + test_images[:20]}
        assert (len(train_images), len(test_images), len(coco["images"]), len(coco["annotations"])) == (707, 180, 1154, 8199)
        assert sizes == {(2048, 2048)}

        def image_metadata(paths, split):
            return pd.DataFrame({"split": split, "file": [path.name for path in paths],
                                 "year": [int(path.stem[:4]) for path in paths], "station": [path.stem[-2] for path in paths]})

        image_frame = pd.concat([image_metadata(train_images, "train"), image_metadata(test_images, "test")])
        display(image_frame.groupby(["split", "station"]).size().unstack(fill_value=0))
        display(image_frame.groupby(["split", "year"]).size().unstack(fill_value=0))
        print(f"Train images: {len(train_images)}, test images: {len(test_images)}, annotator records: {len(records)}, filaments: {len(coco['annotations'])}")
    '''),
    markdown("## 2. Annotator structure and disagreement"),
    code('''
        annotator_codes = [record["annotator_image_id"].split("-", 1)[0] for record in records]
        records_by_image = pd.Series([record["physical_id"] for record in records]).value_counts()
        print(f"Annotator codes: {len(set(annotator_codes))}")
        print("Physical images by annotator count:")
        display(records_by_image.value_counts().sort_index().rename("images").to_frame())

        counts = pd.DataFrame({"physical_id": [record["physical_id"] for record in records],
                               "filaments": [len(record["annotations"]) for record in records]})
        disagreement = counts.groupby("physical_id")["filaments"].agg(["count", "min", "max"])
        shared = disagreement.query("count > 1")
        print(f"Shared images with a >=3-filament count gap: {((shared['max'] - shared['min']) >= 3).mean():.1%}")
    '''),
    markdown("## 3. Label-noise ceiling\n\nThe saved human-vs-human result is PQ 0.34, SQ 0.634, RQ 0.536 over 296 multi-annotator images. Recompute only when auditing the scorer."),
    code('''
        from src.metrics import human_agreement

        if RUN_HUMAN_AGREEMENT:
            agreement_rows, agreement = human_agreement(records)
            display(pd.Series(agreement))
        else:
            display(pd.Series({"official_pq": 0.34, "official_sq": 0.634, "official_rq": 0.536, "status": "saved verified result; recomputation skipped"}))
    '''),
    markdown("## 4. Frozen split verification\n\nThis is the only cell allowed to create `split_manifest.json`, and only when it is missing."),
    code('''
        manifest, train_records, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH, seed=42, validation_fraction=0.15)
        print(f"Validation: {len(manifest['validation_physical_ids'])} physical images / {len(validation_records)} records / {sum(len(r['annotations']) for r in validation_records)} filaments")
        print(f"Training: {len(manifest['train_physical_ids'])} physical images / {len(train_records)} records / {sum(len(r['annotations']) for r in train_records)} filaments")

        legacy_splits = [
            PROJECT_DIR / "archive/outputs_legacy/outputs/stage2_crop_unet/split_manifest.json",
            PROJECT_DIR / "archive/outputs_legacy/outputs/stage2_crop_unet_b5/split_manifest.json",
        ]
        for path in legacy_splits:
            if path.exists():
                assert json.loads(path.read_text()) == manifest
                print(f"Exact match: {path}")
    '''),
]
write_notebook("01_data_eda_and_split.ipynb", cells)


cells = common_setup("02 — Stage 1 YOLO training", "build the detector dataset, record recipes, and optionally train one detector")
cells += [
    code('''
        BUILD_DATASET = False  # Copies 1,154 annotator-specific samples into /content
        TRAIN = False
        MODEL_NAME = "yolo26x.pt"
        IMAGE_SIZE = 1280
        FLIP_UP_DOWN = 0.5  # The next planned generalization experiment
    '''),
    markdown("## 1. Build the YOLO dataset from the frozen split"),
    code('''
        from src.data import build_yolo_dataset

        coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
        manifest, _, _ = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
        YOLO_DATASET_DIR = Path("/content/yolo_dataset")
        if BUILD_DATASET:
            counts = build_yolo_dataset(coco, TRAIN_IMAGES_DIR, manifest, YOLO_DATASET_DIR)
            print(counts)
        else:
            print("Dataset build skipped. Set BUILD_DATASET = True for training.")
    '''),
    markdown("## 2. Existing detector recipes\n\n| Run | Image size | Mosaic | Best epoch | Precision | Recall | mAP50 | mAP50-95 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| YOLO26s | 1024 | 1.0 | — | 0.584 | 0.603 | 0.628 | 0.370 |\n| YOLO26l | 1280 | 1.0 | 92 | 0.612 | 0.637 | 0.648 | 0.386 |\n| YOLO26x | 1280 | 0.5 | 82 | 0.609 | 0.649 | 0.659 | 0.388 |\n\nAll existing runs used `fliplr=0.5`, `flipud=0.0`, and `degrees=0`. Their complete `args.yaml` files remain beside their weights."),
    markdown("### 2.1 Flip-up/down retrain\n\n**Question:** does adding the missing vertical symmetry improve generalization?\n\n**Varies:** `flipud=0.5`.\n\n**Fixed:** YOLO26x, 1280, mosaic 0.5, all other saved recipe settings.\n\n**Cost:** about 1.7 hours on A100.\n\n**Result:** untested.\n\n**Decision:** pending."),
    code('''
        if TRAIN:
            from ultralytics import YOLO

            assert (YOLO_DATASET_DIR / "data.yaml").exists(), "Set BUILD_DATASET = True first"
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S_yolo26x_1280_flipud")
            detector = YOLO(MODEL_NAME)
            detector.train(data=str(YOLO_DATASET_DIR / "data.yaml"), epochs=300, patience=30, imgsz=IMAGE_SIZE,
                           mosaic=0.5, fliplr=0.5, flipud=FLIP_UP_DOWN, degrees=0, batch=-1, device=0,
                           project=str(PROJECT_DIR / "models/stage1"), name=run_name)
        else:
            print("Training skipped.")
    '''),
    markdown("## 3. Training-curve comparison"),
    code('''
        result_files = sorted((PROJECT_DIR / "models/stage1").glob("*/results.csv"))
        for path in result_files:
            frame = pd.read_csv(path)
            metric = next((column for column in frame if "mAP50-95" in column), None)
            if metric:
                plt.plot(frame[metric], label=path.parent.name)
        if result_files:
            plt.xlabel("Epoch"); plt.ylabel("Box mAP50-95"); plt.legend(); plt.show()
        else:
            print("No migrated results.csv files found yet.")
    '''),
    markdown("## 4. Detector validation summary"),
    code('''
        detector_summary = pd.DataFrame([
            {"detector": "yolo26s", "imgsz": 1024, "precision": 0.584, "recall": 0.603, "map50": 0.628, "map50_95": 0.370},
            {"detector": "yolo26l", "imgsz": 1280, "precision": 0.612, "recall": 0.637, "map50": 0.648, "map50_95": 0.386},
            {"detector": "yolo26x", "imgsz": 1280, "precision": 0.609, "recall": 0.649, "map50": 0.659, "map50_95": 0.388},
        ])
        display(detector_summary)
    '''),
]
write_notebook("02_stage1_yolo_train.ipynb", cells)


cells = common_setup("03 — Stage 2 U-Net training", "mine detector-specific negatives and optionally train one crop model")
cells += [
    code('''
        MINE_HARD_NEGATIVES = False
        TRAIN = False
        MODEL_NAME = "b5"  # "crop_unet" or "b5"
        DETECTOR_NAME = "yolo26x"
        YOLO_CHECKPOINT = PROJECT_DIR / "models/stage1/yolo26x_filament_detection/weights/best.pt"
        HARD_NEGATIVE_CACHE = PROJECT_DIR / "data/processed/hard_negatives/hard_negatives_yolo26x_conf005.json"
        MASK_THRESHOLD = 0.45  # what inference uses; checkpoint selection now scores at the deployed threshold
    '''),
    markdown("## 1. Hard-negative mining\n\nCaches are detector-specific. Mining uses confidence 0.05; the optional score floor is applied only when building samples."),
    code('''
        from src.data import mine_hard_negatives

        coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
        _, train_records, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
        train_positive_samples = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"],
                                   "annotation": annotation, "proposal_box": None} for r in train_records for annotation in r["annotations"]]
        if HARD_NEGATIVE_CACHE.exists():
            hard_negatives = json.loads(HARD_NEGATIVE_CACHE.read_text())
        elif MINE_HARD_NEGATIVES:
            from ultralytics import YOLO

            settings = {**DEFAULT_PIPELINE_SETTINGS, **TRAINING_SETTINGS, "yolo_mining_confidence": 0.05}
            hard_negatives = mine_hard_negatives(train_records, YOLO(str(YOLO_CHECKPOINT)), settings, device,
                                                 round(len(train_positive_samples) * settings["hard_negative_ratio"]))
            HARD_NEGATIVE_CACHE.write_text(json.dumps(hard_negatives))
        else:
            hard_negatives = []
            print("No cache found; mining skipped.")
        print(f"Positive crops: {len(train_positive_samples)}, cached hard negatives: {len(hard_negatives)}")
    '''),
    markdown("## 2. Datasets, loaders, and crop sanity check"),
    code('''
        from torch.utils.data import DataLoader
        from src.data import FilamentCropDataset
        from src.viz import show_crop

        settings = {**DEFAULT_PIPELINE_SETTINGS, **TRAINING_SETTINGS}
        negative_samples = [{"physical_id": x["physical_id"], "file_name": x["file_name"], "image_path": TRAIN_IMAGES_DIR / x["file_name"],
                             "annotation": None, "proposal_box": x["proposal_box"]} for x in hard_negatives if x["score"] >= settings["negative_mining_floor"]]
        train_dataset = FilamentCropDataset(train_positive_samples + negative_samples, settings, augment=True)
        validation_samples = [{"physical_id": r["physical_id"], "file_name": r["file_name"], "image_path": r["image_path"],
                               "annotation": annotation, "proposal_box": None} for r in validation_records for annotation in r["annotations"]]
        validation_dataset = FilamentCropDataset(validation_samples, settings, augment=False)
        inputs, target, _ = validation_dataset[0]
        show_crop(inputs, target)
    '''),
    markdown("## 3. Model recipes\n\n| Recipe | Selection metric | Saved result |\n|---|---|---|\n| Plain CropUNet, GroupNorm, from scratch | Positive-crop oracle Dice | 0.8183 at epoch 16 |\n| EfficientNet-B5 U-Net, ImageNet, scse | Positive-crop oracle Dice | 0.8283 at epoch 13 |\n| Proposal-fine-tuned CropUNet + 8-way TTA | Official full-pipeline PQ | 0.4381 |\n\nSince 2026-09-19 training selects on **official PQ**, not oracle Dice: the Dice validation crops are all positives with oracle prompts, so it cannot see the false positives that dominate the score. `center_jitter` also widened from 0.10 to 0.20 to match measured proposal offsets (median 0.030, p90 0.229)."),
    markdown("## 4. Training\n\nThe validator runs the whole pipeline each epoch. Detector boxes are cached once, so only Stage 2 is recomputed."),
    code('''
        from src.inference import cache_proposals, official_pq_for_model
        from src.models import build_model, train_crop_unet

        if TRAIN:
            from ultralytics import YOLO

            run_dir = PROJECT_DIR / "models/stage2" / datetime.now().strftime(f"%Y%m%d_%H%M%S_{MODEL_NAME}")
            run_dir.mkdir(parents=True, exist_ok=False)
            validation_paths = sorted({r["physical_id"]: r["image_path"] for r in validation_records}.values())
            proposals = cache_proposals(YOLO(str(YOLO_CHECKPOINT)), validation_paths, device, DEFAULT_PIPELINE_SETTINGS)
            pq_validator = lambda unet: official_pq_for_model(unet, validation_records, proposals, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS)
            train_loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=settings["num_workers"], pin_memory=True)
            validation_loader = DataLoader(validation_dataset, batch_size=settings["batch_size"], shuffle=False, num_workers=settings["num_workers"], pin_memory=True)
            model = build_model(MODEL_NAME).to(device)
            history = train_crop_unet(model, train_loader, validation_loader, device, settings, run_dir / "best.pt", run_dir / "history.csv", pq_validator)
        else:
            print("Training skipped.")
    '''),
    markdown("## 5. Oracle evaluation\n\n> Warning: oracle crop Dice and IoU do not predict the official end-to-end score. Use notebook 04 for model decisions."),
    code('''
        oracle_results = pd.DataFrame([
            {"checkpoint": "CropUNet", "oracle_dice": 0.8183, "oracle_iou": 0.6986},
            {"checkpoint": "EfficientNet-B5 U-Net", "oracle_dice": 0.8283, "oracle_iou": 0.7122},
        ])
        display(oracle_results)
    '''),
]
write_notebook("03_stage2_unet_train.ipynb", cells)


cells = common_setup("04 — Inference and official evaluation", "run all detector, crop-model, and full-pipeline comparisons against one frozen C0")
cells += [
    code('''
        RUN_C0 = False
        RUN_SWEEPS = False
        RUN_TTA = False
        RUN_ENSEMBLE = False
        YOLO_CHECKPOINT = PROJECT_DIR / "models/stage1/yolo26x_filament_detection/weights/best.pt"
        UNET_CHECKPOINT = PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt"
    '''),
    markdown("## Current best configuration"),
    code('''
        from src.experiments import append_registry, load_registry, new_run_dir, save_run_files

        registry_path = PROJECT_DIR / "experiments/registry.csv"
        registry = load_registry(registry_path)
        current_best = registry.sort_values(["kaggle_public", "official_pq"], ascending=False, na_position="last").head(1)
        display(current_best[["run_id", "detector", "detector_imgsz", "conf", "nms_iou", "unet_checkpoint", "mask_threshold", "tta", "official_pq", "kaggle_public"]])
    '''),
    markdown("## A. Stage 1 in isolation\n\nA1 compares n/s/l/x and image size. A2 sweeps confidence. A3 sweeps NMS IoU and compares end-to-end vs NMS. A4 measures expansion versus >50% crop coverage. A5 tests detector TTA. A6, detector ensembling, is measured end to end in C4. Detector-only cells must report proposals/image, precision, recall, and crop coverage before involving a U-Net."),
    markdown("### A1–A6 experiment record\n\nUse the same question/varies/fixed/cost/result/decision block as Section C. Append detector-only runs to `experiments/stage1/` and the registry."),
    markdown("## B. Stage 2 in isolation\n\nB1 compares oracle Dice/IoU per checkpoint. B2 sweeps the oracle mask threshold. B3 tests 8-way TTA. B4 measures negative-crop rejection at the real 20-pixel bar. B5, U-Net averaging, is measured end to end in C4. Oracle results are diagnostic only."),
    markdown("## C. Full pipeline under the official rule"),
    code('''
        from pycocotools import mask as mask_utils
        from tqdm.auto import tqdm
        from src.inference import predict_instances
        from src.metrics import aggregate_official, annotation_to_rle, bootstrap_pq, official_match_counts
        from src.models import RotationTTA, load_checkpoint

        coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
        _, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
        validation_groups = defaultdict(list)
        for record in validation_records:
            validation_groups[record["physical_id"]].append(record)

        def evaluate_pipeline(detector, unet, settings, mask_threshold, run_dir, proposals=None):
            rows, predictions, image_counts = [], [], []
            for physical_id, group in tqdm(validation_groups.items()):
                image_proposals = None if proposals is None else proposals[physical_id]
                masks, counts = predict_instances(group[0]["image_path"], detector, unet, device, mask_threshold, settings, proposals=image_proposals)
                rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks]
                saved_rles = [{"size": rle["size"], "counts": rle["counts"].decode("ascii")} for rle in rles]
                predictions.append({"physical_id": physical_id, "file_name": group[0]["file_name"], "rles": saved_rles})
                image_counts.append(counts)
                for record in group:
                    targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
                    rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"], **official_match_counts(rles, targets)})
            by_annotator = pd.DataFrame(rows)
            summary = {**aggregate_official(by_annotator), **bootstrap_pq(by_annotator),
                       "instances_per_image": float(pd.DataFrame(image_counts)["final_instances"].mean())}
            save_run_files(run_dir, {"pipeline": settings, "mask_threshold": mask_threshold}, predictions, rows, summary)
            return by_annotator, summary
    '''),
    markdown("### C0 Reproduce the current best\n\n**Question:** does the reorganized path reproduce the saved current best?\n\n**Varies:** nothing.\n\n**Fixed:** YOLO26x @1280, conf 0.20, NMS IoU 0.20, B5 epoch 13, mask 0.45, crop 1.5, dedup 0.50/0.80, minimum area 20.\n\n**Cost:** ~1 minute on A100 or ~15 minutes on CPU.\n\n**Expected:** PQ 0.4300; TP 786, FP 451, FN 472 in the saved fp16 run.\n\n**Decision:** this remains the control unless a candidate clears uncertainty or earns a leaderboard test."),
    code('''
        if RUN_C0:
            from ultralytics import YOLO

            settings = DEFAULT_PIPELINE_SETTINGS.copy()
            detector = YOLO(str(YOLO_CHECKPOINT))
            unet, checkpoint = load_checkpoint(UNET_CHECKPOINT, "b5", device)
            run_id, run_dir = new_run_dir(PROJECT_DIR / "experiments", "pipeline", "C0", "b5_yolo26x")
            by_annotator_c0, summary_c0 = evaluate_pipeline(detector, unet, settings, 0.45, run_dir)
            append_registry(registry_path, {"run_id": run_id, "timestamp": datetime.now().isoformat(), "notebook": "04_inference_and_evaluation.ipynb",
                                           "section": "C0", "detector": "yolo26x", "detector_imgsz": 1280, "conf": 0.20, "nms_iou": 0.20,
                                           "end2end": False, "unet_checkpoint": UNET_CHECKPOINT.name, "mask_threshold": 0.45, "tta": False,
                                           "official_pq": summary_c0["official_pq"], "sq": summary_c0["official_sq"], "rq": summary_c0["official_rq"],
                                           "tp": summary_c0["official_tp"], "fp": summary_c0["official_fp"], "fn": summary_c0["official_fn"],
                                           "instances_per_image": summary_c0["instances_per_image"], "pq_ci_low": summary_c0["pq_ci_low"],
                                           "pq_ci_high": summary_c0["pq_ci_high"], "device": str(device), "precision": "fp16" if device.type == "cuda" else "fp32"})
            display(pd.Series(summary_c0))
        else:
            print("C0 inference skipped. Saved C0: PQ 0.4300496381, SQ 0.6825533378, RQ 0.6300601202, TP 786, FP 451, FN 472.")
    '''),
    markdown("### C1.2 NMS IoU sweep\n\n**Question:** does NMS at IoU 0.20 suppress real neighbouring filaments?\n\n**Varies:** `yolo_iou` in {0.20, 0.35, 0.50, 0.70}.\n\n**Fixed:** YOLO26x @1280, conf 0.20, B5 epoch 13, mask 0.45, everything else as C0.\n\n**Cost:** ~1 minute per pass on A100.\n\n**Result:** untested on real data.\n\n**Decision:** pending."),
    code('''
        if RUN_SWEEPS:
            from ultralytics import YOLO

            detector = YOLO(str(YOLO_CHECKPOINT))
            unet, checkpoint = load_checkpoint(UNET_CHECKPOINT, "b5", device)
            sweep_rows = []
            for nms_iou in [0.20, 0.35, 0.50, 0.70]:
                settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_iou": nms_iou}
                run_id, run_dir = new_run_dir(PROJECT_DIR / "experiments", "pipeline", "C1-2", f"nms_iou_{nms_iou:.2f}")
                by_annotator, summary = evaluate_pipeline(detector, unet, settings, 0.45, run_dir)
                append_registry(registry_path, {"run_id": run_id, "timestamp": datetime.now().isoformat(), "notebook": "04_inference_and_evaluation.ipynb",
                                               "section": "C1.2", "detector": "yolo26x", "detector_imgsz": 1280, "conf": 0.20, "nms_iou": nms_iou,
                                               "end2end": False, "unet_checkpoint": UNET_CHECKPOINT.name, "mask_threshold": 0.45, "tta": False,
                                               "official_pq": summary["official_pq"], "sq": summary["official_sq"], "rq": summary["official_rq"],
                                               "tp": summary["official_tp"], "fp": summary["official_fp"], "fn": summary["official_fn"],
                                               "instances_per_image": summary["instances_per_image"], "pq_ci_low": summary["pq_ci_low"],
                                               "pq_ci_high": summary["pq_ci_high"], "device": str(device), "precision": "fp16" if device.type == "cuda" else "fp32"})
                sweep_rows.append({"run_id": run_id, "nms_iou": nms_iou, **summary})
            display(pd.DataFrame(sweep_rows))
    '''),
    markdown("### C1 remaining one-factor sweeps\n\nC1.1 confidence; C1.3 mask threshold; C1.4 dedup; C1.5 minimum mask area; C1.6 crop expansion. Change one key in a copy of `DEFAULT_PIPELINE_SETTINGS`, save every run, and compare it with the same C0 predictions."),
    markdown("### C2 Combine winners\n\nOnly combine changes that were useful in the one-factor runs."),
    markdown("### C3 Eight-way U-Net TTA\n\n**Question:** does symmetry averaging improve the B5 stack's generalization?\n\n**Varies:** wrap the loaded U-Net in `RotationTTA`.\n\n**Fixed:** all C0 detector and mask settings.\n\n**Result:** untested on B5."),
    code('''
        if RUN_TTA:
            from ultralytics import YOLO

            detector = YOLO(str(YOLO_CHECKPOINT))
            unet, checkpoint = load_checkpoint(UNET_CHECKPOINT, "b5", device)
            run_id, run_dir = new_run_dir(PROJECT_DIR / "experiments", "pipeline", "C3", "b5_tta")
            by_annotator_tta, summary_tta = evaluate_pipeline(detector, RotationTTA(unet).to(device).eval(), DEFAULT_PIPELINE_SETTINGS.copy(), 0.45, run_dir)
            append_registry(registry_path, {"run_id": run_id, "timestamp": datetime.now().isoformat(), "notebook": "04_inference_and_evaluation.ipynb",
                                           "section": "C3", "detector": "yolo26x", "detector_imgsz": 1280, "conf": 0.20, "nms_iou": 0.20,
                                           "end2end": False, "unet_checkpoint": UNET_CHECKPOINT.name, "mask_threshold": 0.45, "tta": True,
                                           "official_pq": summary_tta["official_pq"], "sq": summary_tta["official_sq"], "rq": summary_tta["official_rq"],
                                           "tp": summary_tta["official_tp"], "fp": summary_tta["official_fp"], "fn": summary_tta["official_fn"],
                                           "instances_per_image": summary_tta["instances_per_image"], "pq_ci_low": summary_tta["pq_ci_low"],
                                           "pq_ci_high": summary_tta["pq_ci_high"], "device": str(device), "precision": "fp16" if device.type == "cuda" else "fp32"})
            display(pd.Series(summary_tta))
    '''),
    markdown("### C4 Detector and U-Net ensemble — the current best\n\n**Question:** does agreement across detectors separate real low-confidence filaments from false positives, and does averaging U-Nets reduce Stage 2 variance?\n\n**Varies:** proposals come from weighted box fusion of all seven Stage 1 detectors, each at its training image size and confidence 0.01 (cluster IoU 0.40; a cluster scores the mean of each detector's best box, so missing detectors count as zero; kept at fused score ≥ 0.22). Masks come from the B5 champion averaged with the symmetric-Tversky B5 (`ModelAverage`). Settings live in `ENSEMBLE_*` in `src/config.py`.\n\n**Fixed:** every C0 crop, mask and dedup setting.\n\n**Cost:** ~3 minutes on A100; ~60 minutes on CPU.\n\n**Result (2026-10-27, CPU fp32):** official PQ **0.4628** vs C0 0.4303 on the same path, TP 826 / FP 349 / FN 432 against 789 / 458 / 469, paired delta **+0.0325 [+0.0215, +0.0437]**. Detector fusion alone gave +0.025 and every fused threshold 0.18–0.28 cleared zero; averaging a second U-Net added +0.0077 [+0.0030, +0.0134]; four U-Nets tied two. Kaggle public: see `submissions/leaderboard_log.csv`.\n\n**Decision:** adopted as the submission pipeline in notebook 05."),
    code('''
        if RUN_ENSEMBLE:
            from src.config import ENSEMBLE_SETTINGS, ENSEMBLE_UNETS
            from src.inference import ensemble_proposals
            from src.models import ModelAverage

            validation_paths = [group[0]["image_path"] for group in validation_groups.values()]
            proposals = ensemble_proposals(PROJECT_DIR / "models", validation_paths, device, DEFAULT_PIPELINE_SETTINGS)
            unet = ModelAverage([load_checkpoint(PROJECT_DIR / path, "b5", device)[0] for path in ENSEMBLE_UNETS]).to(device).eval()
            run_id, run_dir = new_run_dir(PROJECT_DIR / "experiments", "pipeline", "C4", "wbf7_unet2")
            by_annotator_ensemble, summary_ensemble = evaluate_pipeline(None, unet, DEFAULT_PIPELINE_SETTINGS.copy(), ENSEMBLE_SETTINGS["mask_threshold"], run_dir, proposals)
            append_registry(registry_path, {"run_id": run_id, "timestamp": datetime.now().isoformat(), "notebook": "04_inference_and_evaluation.ipynb",
                                           "section": "C4", "detector": "wbf7", "detector_imgsz": 1280, "conf": ENSEMBLE_SETTINGS["fused_threshold"], "nms_iou": "",
                                           "end2end": False, "unet_checkpoint": "+".join(ENSEMBLE_UNETS), "mask_threshold": ENSEMBLE_SETTINGS["mask_threshold"], "tta": False,
                                           "other_settings_json": json.dumps(ENSEMBLE_SETTINGS),
                                           "official_pq": summary_ensemble["official_pq"], "sq": summary_ensemble["official_sq"], "rq": summary_ensemble["official_rq"],
                                           "tp": summary_ensemble["official_tp"], "fp": summary_ensemble["official_fp"], "fn": summary_ensemble["official_fn"],
                                           "instances_per_image": summary_ensemble["instances_per_image"], "pq_ci_low": summary_ensemble["pq_ci_low"],
                                           "pq_ci_high": summary_ensemble["pq_ci_high"], "device": str(device), "precision": "fp16" if device.type == "cuda" else "fp32"})
            display(pd.Series(summary_ensemble))
        else:
            print("C4 inference skipped. Saved C4 (CPU fp32): PQ 0.4628, TP 826, FP 349, FN 432; paired delta vs C0 +0.0325 [+0.0215, +0.0437].")
    '''),
    markdown("### C5 Later experiment\n\nC5 tests a confidence-0.05 fallback only on images where C0 produces zero masks."),
    markdown("### C6 Summary against C0"),
    code('''
        from src.metrics import paired_pq_delta

        registry = load_registry(registry_path)
        display(registry.sort_values("official_pq", ascending=False)[["run_id", "section", "official_pq", "sq", "rq", "tp", "fp", "fn", "kaggle_public", "notes"]])
        print("Use paired_pq_delta(control_by_annotator, candidate_by_annotator) when both per-record files are available.")
    '''),
]
write_notebook("04_inference_and_evaluation.ipynb", cells)


cells = common_setup("05 — Submission", "select one frozen run, predict the test set, validate the CSV, and log it")
cells += [
    code('''
        RUN_SUBMISSION = False
        USE_ENSEMBLE = True  # seven-detector WBF + two averaged U-Nets (notebook 04 C4); False reproduces the single-model C0 submission
        if USE_ENSEMBLE:
            from src.config import ENSEMBLE_SETTINGS
            RUN_ID = "20261027_212727_de_detector_ensemble_all7_t022_unet2"
            SUBMISSION_NAME = "wbf7_unet2"
            MASK_THRESHOLD = ENSEMBLE_SETTINGS["mask_threshold"]
        else:
            RUN_ID = "20260909_230758_b5_c0"
            SUBMISSION_NAME = "b5_yolo26x"
            MASK_THRESHOLD = 0.45
        YOLO_CHECKPOINT = PROJECT_DIR / "models/stage1/yolo26x_filament_detection/weights/best.pt"
        UNET_CHECKPOINT = PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt"
    '''),
    markdown("## 1. Choose and inspect the run"),
    code('''
        from src.experiments import load_registry

        registry = load_registry(PROJECT_DIR / "experiments/registry.csv")
        selected = registry.loc[registry["run_id"] == RUN_ID]
        assert len(selected) == 1, f"Expected one registry row for {RUN_ID}"
        display(selected)
    '''),
    markdown("## 2. Predict all 180 test images"),
    code('''
        from src.inference import create_submission, validate_submission
        from src.models import load_checkpoint

        if RUN_SUBMISSION:
            from ultralytics import YOLO

            test_paths = sorted(TEST_IMAGES_DIR.glob("*.jpeg"))
            assert len(test_paths) == 180
            if USE_ENSEMBLE:
                from src.config import ENSEMBLE_UNETS
                from src.inference import ensemble_proposals
                from src.models import ModelAverage

                proposals = ensemble_proposals(PROJECT_DIR / "models", test_paths, device, DEFAULT_PIPELINE_SETTINGS)
                unet = ModelAverage([load_checkpoint(PROJECT_DIR / path, "b5", device)[0] for path in ENSEMBLE_UNETS]).to(device).eval()
                submission, image_statistics = create_submission(test_paths, None, unet, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS.copy(), proposals)
            else:
                detector = YOLO(str(YOLO_CHECKPOINT))
                unet, checkpoint = load_checkpoint(UNET_CHECKPOINT, "b5", device)
                submission, image_statistics = create_submission(test_paths, detector, unet, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS.copy())
        else:
            print("Submission inference skipped.")
    '''),
    markdown("## 3. Validate and list zero-mask images\n\nValidation checks unique ids, exact 2048×2048 decoding, and no overlap within an image."),
    code('''
        if RUN_SUBMISSION:
            assert validate_submission(submission)
            zero_mask_images = image_statistics.query("final_instances == 0")["image"].tolist()
            print(f"Rows: {len(submission)}, zero-mask images: {zero_mask_images}")
    '''),
    markdown("## 4. Save without overwriting and append the log"),
    code('''
        if RUN_SUBMISSION:
            output_dir = PROJECT_DIR / "submissions" / datetime.now().strftime(f"%Y%m%d_%H%M%S_{SUBMISSION_NAME}")
            output_dir.mkdir(parents=True, exist_ok=False)
            submission.to_csv(output_dir / "submission.csv", index=False)
            image_statistics.to_csv(output_dir / "image_statistics.csv", index=False)
            (output_dir / "settings.json").write_text(json.dumps({"run_id": RUN_ID, "mask_threshold": MASK_THRESHOLD, "pipeline": DEFAULT_PIPELINE_SETTINGS,
                                                                   "ensemble": ENSEMBLE_SETTINGS if USE_ENSEMBLE else None}, indent=2))
            log_path = PROJECT_DIR / "submissions/leaderboard_log.csv"
            row = pd.DataFrame([{"date": datetime.now().date(), "run_id": RUN_ID, "local_pq": float(selected.iloc[0]["official_pq"]),
                                 "kaggle_public": "", "submission_path": str(output_dir.relative_to(PROJECT_DIR)), "notes": "Kaggle score pending"}])
            row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            print(output_dir)
    '''),
]
write_notebook("05_submission.ipynb", cells)


cells = common_setup("06 — Diagnostics", "audit failures, uncertainty, label style, train-test behavior, and empty images")
cells += [
    code('''
        RUN_HUMAN_AGREEMENT = False
        RUN_EMPTY_IMAGE_DIAGNOSTIC = False
        YOLO_CHECKPOINT = PROJECT_DIR / "models/stage1/yolo26x_filament_detection/weights/best.pt"
        UNET_CHECKPOINT = PROJECT_DIR / "models/stage2/crop_unet_b5/best_crop_unet_b5.pt"
    '''),
    markdown("## 1. Failure audit\n\nThe saved old-pipeline audit found 468 validation FNs: 219 crop-coverage failures (192 with no proposal) and 230 raw-mask failures. Read the migrated `annotation_details.csv`, `traces.jsonl`, and `coverage_headroom.csv` here before changing the architecture."),
    code('''
        audit_files = sorted((PROJECT_DIR / "experiments/diagnostics").rglob("annotation_details.csv"))
        if audit_files:
            audit = pd.read_csv(audit_files[-1])
            display(audit.head())
        else:
            print("No migrated stage-audit table found.")
    '''),
    markdown("## 2. Confidence break-even"),
    code('''
        pq, sq = 0.43004963808881497, 0.6825533378063571
        break_even = 2 * sq / pq - 1
        print(f"Removing one FP is worthwhile while it costs fewer than {break_even:.2f} TPs.")
    '''),
    markdown("## 3. Bootstrap and paired comparisons"),
    code('''
        from src.metrics import bootstrap_pq, paired_pq_delta, score_prediction_file

        print("Load two migrated by_annotator.csv files and use paired_pq_delta(control, candidate).")
        print("The saved marginal bootstrap SD is 0.0149, roughly a ±0.03 95% interval.")
    '''),
    markdown("## 4. Gap checks: annotator count, aggregation, style, and firing rate\n\nSaved checks: PQ for 1/2/3-annotator images = 0.4398/0.4350/0.4252; pooled/mean-record/mean-image/first-record = 0.4331/0.4144/0.4214/0.4297; final masks per image = 7.28 test vs 7.26 validation. Known annotator style does not explain the leaderboard gap."),
    code('''
        coco, records = load_records(ANNOTATIONS_PATH, TRAIN_IMAGES_DIR)
        _, _, validation_records = verify_or_write_split(records, SPLIT_MANIFEST_PATH)
        records_per_image = pd.Series([record["physical_id"] for record in validation_records]).value_counts()
        display(records_per_image.value_counts().sort_index().rename("validation_images").to_frame())
    '''),
    markdown("## 5. Empty test images\n\nThe saved C0 submission had four zero-mask images. This cell rechecks only those images at detector confidence 0.01 and pipeline confidence 0.05; it is off by default."),
    code('''
        if RUN_EMPTY_IMAGE_DIAGNOSTIC:
            from PIL import Image
            from ultralytics import YOLO
            from src.inference import predict_instances
            from src.models import load_checkpoint

            statistics_files = sorted((PROJECT_DIR / "submissions").rglob("image_statistics.csv"))
            assert statistics_files, "No submission image_statistics.csv found"
            empty = pd.read_csv(statistics_files[-1]).query("final_instances == 0")
            detector = YOLO(str(YOLO_CHECKPOINT))
            unet, checkpoint = load_checkpoint(UNET_CHECKPOINT, "b5", device)
            low_settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": 0.05}
            for row in empty.itertuples():
                image_path = TEST_IMAGES_DIR / row.image
                result = detector.predict(str(image_path), imgsz=1280, conf=0.01, iou=0.20, max_det=300, end2end=False,
                                          device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
                masks, counts = predict_instances(image_path, detector, unet, device, 0.45, low_settings)
                scores = result.boxes.conf.detach().cpu().numpy()
                print(row.image, "boxes >=0.05:", int((scores >= 0.05).sum()), "fallback masks:", len(masks), counts)
    '''),
    markdown("## 6. False-positive anatomy\n\nUse the official per-record files and saved masks to inspect full-size blobs, duplicate elongated structures, and unmatched low-confidence proposals. Keep this descriptive; do not tune on hand-picked examples."),
]
write_notebook("06_diagnostics.ipynb", cells)

cells = [
    markdown('''
        # Final pipeline, end to end

        **One job:** reproduce the final solution from a clean Colab T4 runtime: a YOLO26l detector (`l_ps`) proposes boxes and one box-prompted U-Net (`b5_sym_s2`) segments each filament. Both checkpoints are in `weights/`.

        **Data:** put the competition download at `/content/kaggle_data`, or leave it to the next cell, which needs a Kaggle token at `~/.kaggle/access_token` from an account that joined the competition. The images are not in this repo because they cannot be redistributed.

        The demo takes under a minute once the data is in place. Validation (~3 min) and test (~5 min) are behind flags that default to `False`. Training is summarized at the end.
    '''),
    code('''
        !git clone -q https://github.com/franciscomendez114/solar-filament-segmentation.git /content/solar-filament-segmentation
        %cd /content/solar-filament-segmentation
        %pip install -q ultralytics==8.4.191 pycocotools segmentation-models-pytorch==0.5.0 kaggle==2.2.4
        !test -d /content/kaggle_data || (kaggle competitions download -c filament-segmentation-2026 -p /content && unzip -q /content/filament-segmentation-2026.zip -d /content/kaggle_data)
    '''),
    code('''
        import sys
        from collections import defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from pycocotools import mask as mask_utils
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        LOGGER.setLevel("ERROR")  # two deprecation warnings per image otherwise
        sys.path.insert(0, "/content/solar-filament-segmentation")
        from src.config import DEFAULT_PIPELINE_SETTINGS
        from src.data import load_records, verify_or_write_split
        from src.inference import cache_proposals, create_submission, fuse_boxes, predict_instances
        from src.metrics import aggregate_official, annotation_to_rle, official_match_counts
        from src.models import RotationTTA, load_checkpoint

        RUN_VALIDATION = False  # reproduces official PQ 0.4666 on the frozen 106-image split
        RUN_TEST = False  # writes submission.csv for the 180 test images
        DETECTOR_CONFIDENCE, MASK_THRESHOLD = 0.25, 0.45

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector = YOLO("weights/l_ps.pt")
        unet = RotationTTA(load_checkpoint("weights/b5_sym_s2.pt", "b5", device)[0]).eval()  # 8-way rotation and flip TTA
        settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": 0.01, "yolo_image_size": 1280}

        train_dir = next(Path("/content/kaggle_data").rglob("train_images")).parent
        test_paths = sorted(next(Path("/content/kaggle_data").rglob("test_images")).glob("*.jpeg"))
        _, records = load_records(train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", train_dir / "train_images")
        _, _, validation_records = verify_or_write_split(records, Path("data/processed/split_manifest.json"))
        print(device, len(validation_records), "validation records,", len(test_paths), "test images")


        def final_boxes(image_paths):
            # Stage 1: confidence >= 0.25, after the fusion the ensemble used, which with one detector only merges near-identical boxes
            proposals = {}
            for stem, candidates in cache_proposals(detector, image_paths, device, settings).items():
                boxes, scores = fuse_boxes([candidates], [1.0], 0.40, 0.01)
                keep = scores >= DETECTOR_CONFIDENCE
                proposals[stem] = (np.asarray(boxes[keep], dtype=np.float32).reshape(-1, 4), np.asarray(scores[keep], dtype=np.float32))
            return proposals
    '''),
    markdown("## Demo: one validation image\n\nLeft: the detector's boxes. Right: the final non-overlapping masks, drawn dilated so thin filaments stay visible."),
    code('''
        import cv2

        record = validation_records[0]
        path = record["image_path"]
        boxes, scores = final_boxes([path])[Path(path).stem]
        masks, _ = predict_instances(path, None, unet, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS, proposals=(boxes, scores))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        overlay = np.zeros((*image.shape, 4))
        for mask in masks:
            overlay[cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = (0.1, 0.9, 0.4, 0.9)
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        for axis in axes:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
        for x1, y1, x2, y2 in boxes:
            axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="orange", linewidth=1))
        axes[0].set_title(f"{len(boxes)} detector boxes")
        axes[1].imshow(overlay)
        axes[1].set_title(f"{len(masks)} final masks")
        plt.show()
    '''),
    markdown("## Validation: official PQ\n\nScores each annotator record separately and pools the counts, as the organizers' code does. Expected: PQ 0.4666, TP 828, FP 360, FN 430."),
    code('''
        if RUN_VALIDATION:
            by_image = defaultdict(list)
            for record in validation_records:
                by_image[record["physical_id"]].append(record)
            proposals = final_boxes([group[0]["image_path"] for group in by_image.values()])
            rows = []
            for physical_id, group in by_image.items():
                masks, _ = predict_instances(group[0]["image_path"], None, unet, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS, proposals=proposals[physical_id])
                predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
                for record in group:
                    targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
                    rows.append({"physical_id": physical_id, "annotator_image_id": record["annotator_image_id"], "predicted_instances": len(masks), **official_match_counts(predicted, targets)})
            print(aggregate_official(pd.DataFrame(rows)))
    '''),
    markdown("## Test: Kaggle submission\n\nWrites `submission.csv` after checking for duplicate ids and overlapping masks. This pipeline scored 0.39 on the public leaderboard."),
    code('''
        if RUN_TEST:
            submission, _ = create_submission(test_paths, None, unet, device, MASK_THRESHOLD, DEFAULT_PIPELINE_SETTINGS, final_boxes(test_paths))
            submission.to_csv("submission.csv", index=False)
            print(len(submission), "masks in submission.csv")
    '''),
    markdown('''
        ## Training

        Training ran on Colab GPUs through the scripts below, not in this notebook: the final detector took about two hours on an A100.

        1. **Baseline detectors and U-Net:** notebooks `02` and `03`, then `scripts/detector_train_job.py` for the YOLO11m member.
        2. **Unlabeled pool:** `scripts/gong_pool.py` picks and downloads 1,000 GONG frames (`data/processed/gong_pool.json`), and `scripts/check_gong_pool.py` proves none is a test image.
        3. **Teacher labels:** `scripts/launch_pseudo.py` fuses the 24 teacher views at score >= 0.30. The teacher list is `configs/teachers.json` and the labels used are in `data/processed/pseudo_boxes.json`.
        4. **Final detector:** `scripts/detector_train_job.py` with `configs/final_detector.json`.
        5. **Final U-Net:** `scripts/unet_train_job.py` with `configs/final_unet.json`, selected on official PQ.
    '''),
]
write_notebook("07_final_pipeline.ipynb", cells)

print(f"Wrote seven notebooks to {OUTPUT_DIR}")
