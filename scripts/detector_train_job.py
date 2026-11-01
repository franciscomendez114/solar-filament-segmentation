# Trains extra Stage 1 detectors for the WBF ensemble inside a Colab GPU VM, then caches each one's low-confidence
# boxes (plain, h-flip, v-flip) on the 106 validation and 180 test images so they drop straight into the fusion.
# Why: diversity across detectors is what moved PQ (WBF of 7: +0.025), and v-flip TTA helped although no saved detector
# trained with flipud, so new members get a different family (YOLO11) or different augmentation (flipud 0.5).
# Recipe otherwise follows yolo26m: 1280, mosaic 0.5, pinned batch (AutoBatch is unreliable here), patience 25.
# The VM needs /content/src/, /content/split_manifest.json and ~/.kaggle/access_token. Output in /content/results/.
import json
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

# The run list comes from /content/detector_runs.json: {run name: {"weights": ..., "full_data": bool, **train argument overrides}}.
# Round 3 (2026-10-28) ran yolo11m_flipud {"weights": "yolo11m.pt", "batch": 16, "flipud": 0.5} and yolo26s_flipud
# {"weights": "yolo26s.pt", "batch": 16, "flipud": 0.5, "seed": 1}. A full_data run trains on train + validation for a fixed
# schedule (no early stopping, last.pt kept, since validation is then inside training) and caches test boxes only.
# Round 4 trains above 1280 (1024 -> 1280 was the largest detector gain on record): yolo26s_1536_fu {"imgsz": 1536, "batch": 16,
# "flipud": 0.5, "seed": 2} and yolo26s_2048_fu {"imgsz": 2048, "batch": 8, "flipud": 0.5, "seed": 3}. 1536 is the best detector yet
# (mAP50-95 0.4048 @ e94; alone +0.0064 PQ over yolo26x, tied) but adds nothing to the hv24 fusion; 2048 was stopped at e95 (mAP 0.394).
# Round 6 (2026-10-28) trains on the union of annotators ("labels": "union", src/data.py build_union_yolo_dataset): one sample per
# physical image instead of one per record. On multi-annotator train images an annotator marks 6.7 filaments but the union 10.2,
# so the per-record copies teach ~1300 filaments as background.
# Round 5 pseudo-labels unlabeled GONG H-alpha (scripts/gong_pool.py) when /content/gong_pool.tar is present: the split-trained
# hv24 teachers (/content/teachers.json: {"threshold": t, "members": [[name, weights file, imgsz, flip], ...]}, weights in
# /content/teachers/) are fused exactly as in the 0.39 submission and boxes >= t become YOLO labels; results/pseudo_boxes.json
# is pulled, and a VM given /content/pseudo_boxes.json skips the labeling. A run with "pseudo": true trains on train + pool.
# Round 8 (2026-10-31): "contrast": true trains and infers on src/data.py contrast_detector_image (gray + two CLAHE channels) to test
# whether Stage 1 recall on faint filaments moves. l_cl is the l_ps recipe plus hsv_h/hsv_s 0, since hue and saturation jitter would mix the channels.
WORK = Path("/content")
RAW_DIR = WORK / "kaggle_data"
DATASET_DIR = WORK / "yolo_dataset"
RUNS_DIR = WORK / "runs"
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
        shell("pip install -q 'ultralytics==8.4.191' pycocotools")  # the version l_ps trained on
        sys.path.insert(0, str(WORK))
        import torch
        from ultralytics import YOLO

        from src.config import DEFAULT_PIPELINE_SETTINGS
        from src.data import build_union_yolo_dataset, build_yolo_dataset, load_records, verify_or_write_split, write_contrast_images
        from src.inference import cache_proposals

        train_dir = next(RAW_DIR.rglob("train_images")).parent
        test_paths = sorted(next(RAW_DIR.rglob("test_images")).glob("*.jpeg"))
        assert len(test_paths) == 180
        coco, records = load_records(train_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json", train_dir / "train_images")
        manifest, _, validation_records = verify_or_write_split(records, WORK / "split_manifest.json")
        validation_paths = sorted({r["physical_id"]: r["image_path"] for r in validation_records}.values())
        if not (DATASET_DIR / "data.yaml").exists():
            print(f"YOLO dataset: {build_yolo_dataset(coco, train_dir / 'train_images', manifest, DATASET_DIR)}", flush=True)
        full_dir = WORK / "yolo_dataset_full"
        if not full_dir.exists():
            # train + validation in the training folders; the val folder stays only because Ultralytics requires one
            shutil.copytree(DATASET_DIR, full_dir)
            for kind in ["images", "labels"]:
                for item in (full_dir / f"{kind}/val").iterdir():
                    shutil.copy2(item, full_dir / f"{kind}/train" / item.name)
            (full_dir / "data.yaml").write_text((DATASET_DIR / "data.yaml").read_text().replace(str(DATASET_DIR), str(full_dir)))
        union_dir = WORK / "yolo_dataset_union"
        if not union_dir.exists():
            print(f"union YOLO dataset: {build_union_yolo_dataset(coco, manifest, DATASET_DIR, union_dir)}", flush=True)
        RESULTS.mkdir(exist_ok=True)
        pool_path, pseudo_path = WORK / "gong_pool.tar", RESULTS / "pseudo_boxes.json"
        if pool_path.exists():
            from src.inference import fuse_boxes
            gong_dir = WORK / "gong"
            if not gong_dir.exists():
                # The pool is fetched locally by scripts/gong_pool.py --download (Colab VMs time out on the NSO archive)
                with tarfile.open(WORK / "gong_pool.tar") as archive:
                    archive.extractall(gong_dir, filter="data")
            gong_paths = sorted(gong_dir.glob("*.jpeg"))
            print(f"{len(gong_paths)} GONG pool images", flush=True)
            if not pseudo_path.exists() and (WORK / "pseudo_boxes.json").exists():
                shutil.copy2(WORK / "pseudo_boxes.json", pseudo_path)
            if not pseudo_path.exists():
                teachers = json.loads((WORK / "teachers.json").read_text())
                caches = []
                for name, weights_file, image_size, flip in teachers["members"]:
                    print(f"teacher {name}", flush=True)
                    settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": 0.01, "yolo_image_size": image_size}
                    caches.append(cache_proposals(YOLO(str(WORK / "teachers" / weights_file)), gong_paths, torch.device("cuda"), settings, flip))
                pseudo = {}
                for stem in caches[0]:
                    boxes, scores = fuse_boxes([cache[stem] for cache in caches], [1.0] * len(caches), 0.40, 0.01)
                    keep = scores >= teachers["threshold"]
                    pseudo[stem] = {"boxes": boxes[keep].tolist(), "scores": scores[keep].tolist()}
                pseudo_path.write_text(json.dumps(pseudo))
                finished.append("pseudo_boxes.json")
                (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished}) + "\n")
            pseudo_dir = WORK / "yolo_dataset_pseudo"
            if not pseudo_dir.exists():
                shutil.copytree(DATASET_DIR, pseudo_dir)
                for stem, item in json.loads(pseudo_path.read_text()).items():
                    shutil.copy2(gong_dir / f"{stem}.jpeg", pseudo_dir / f"images/train/{stem}.jpeg")
                    rows = [f"0 {(x1 + x2) / 4096:.6f} {(y1 + y2) / 4096:.6f} {(x2 - x1) / 2048:.6f} {(y2 - y1) / 2048:.6f}" for x1, y1, x2, y2 in item["boxes"]]
                    (pseudo_dir / f"labels/train/{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                (pseudo_dir / "data.yaml").write_text((DATASET_DIR / "data.yaml").read_text().replace(str(DATASET_DIR), str(pseudo_dir)))

        for run_name, spec in json.loads((WORK / "detector_runs.json").read_text()).items():
            spec = dict(spec)
            weights, full_data, use_pseudo, labels = spec.pop("weights"), spec.pop("full_data", False), spec.pop("pseudo", False), spec.pop("labels", "records")
            contrast = spec.pop("contrast", False)
            data_dir = full_dir if full_data else WORK / "yolo_dataset_pseudo" if use_pseudo else union_dir if labels == "union" else DATASET_DIR
            if contrast:
                source, data_dir = data_dir, data_dir.with_name(f"{data_dir.name}_contrast")
                if not data_dir.exists():
                    shutil.copytree(source / "labels", data_dir / "labels", ignore=shutil.ignore_patterns("*.cache"))
                    for kind in ["train", "val"]:
                        write_contrast_images(sorted((source / f"images/{kind}").iterdir()), data_dir / f"images/{kind}")
                    (data_dir / "data.yaml").write_text((source / "data.yaml").read_text().replace(str(source), str(data_dir)))
            run_dir = RUNS_DIR / run_name
            kept = run_dir / ("weights/last.pt" if full_data else "weights/best.pt")
            if not kept.exists():
                arguments = {"epochs": 200, "patience": 25, "imgsz": 1280, "batch": 16, "device": 0, "workers": 8, "seed": 0, "deterministic": True,
                             "mosaic": 0.5, "fliplr": 0.5, "flipud": 0.0, "degrees": 0.0, "close_mosaic": 10, "project": str(RUNS_DIR), "name": run_name,
                             "exist_ok": True, "plots": True, **spec}
                if full_data:
                    arguments["patience"] = arguments["epochs"]
                print(f"===== training {run_name} from {weights} (full data {full_data}): {arguments}", flush=True)
                YOLO(weights).train(data=str(data_dir / "data.yaml"), **arguments)
            detector = YOLO(str(kept))
            settings = {**DEFAULT_PIPELINE_SETTINGS, "yolo_confidence": 0.01, "yolo_image_size": spec.get("imgsz", 1280)}
            for flip, suffix in [(None, ""), ("h", "_hflip"), ("v", "_vflip")]:
                for split, paths in [("test", test_paths)] if full_data else [("validation", validation_paths), ("test", test_paths)]:
                    paths = write_contrast_images(paths, WORK / f"contrast_{split}") if contrast and flip is None else sorted((WORK / f"contrast_{split}").iterdir()) if contrast else paths
                    cache = cache_proposals(detector, paths, torch.device("cuda"), settings, flip)
                    name = f"{split}_{run_name}{suffix}.json"
                    (RESULTS / name).write_text(json.dumps({k: {"boxes": b.tolist(), "scores": s.tolist()} for k, (b, s) in cache.items()}))
                    finished.append(name)
            with tarfile.open(RESULTS / f"{run_name}.tar.gz", "w:gz") as archive:
                for item in [str(kept.relative_to(run_dir)), "results.csv", "args.yaml"]:
                    archive.add(run_dir / item, arcname=f"{run_name}/{item}")
            finished.append(f"{run_name}.tar.gz")
            (WORK / "PROGRESS.json").write_text(json.dumps({"completed": finished}) + "\n")
            print(f"RUN DONE {run_name}", flush=True)
        summary = {"status": "ok", "completed": finished}
    except Exception:
        summary = {"status": "failed", "error": traceback.format_exc(), "completed": finished}
        print(summary["error"], flush=True)
    (WORK / "DONE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {summary['status']}", flush=True)


if __name__ == "__main__":
    main()
