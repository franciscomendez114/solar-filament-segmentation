import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


def copy_raw_data_to_local(raw_data_dir, local_data_dir=Path("/content/solar_filament_data")):
    raw_data_dir, local_data_dir = Path(raw_data_dir), Path(local_data_dir)
    source_train, source_test = raw_data_dir / "train/train_images", raw_data_dir / "test/test_images"
    local_train, local_test = local_data_dir / "train/train_images", local_data_dir / "test/test_images"
    if len(list(local_train.glob("*.jpeg"))) != len(list(source_train.glob("*.jpeg"))):
        shutil.copytree(source_train, local_train, dirs_exist_ok=True)
    if len(list(local_test.glob("*.jpeg"))) != len(list(source_test.glob("*.jpeg"))):
        shutil.copytree(source_test, local_test, dirs_exist_ok=True)
    local_annotations = local_data_dir / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    local_annotations.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_data_dir / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json", local_annotations)
    return local_data_dir


def load_records(annotations_path, train_images_dir):
    coco = json.loads(Path(annotations_path).read_text())
    annotations_by_id = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_id[annotation["image_id"]].append(annotation)

    records = []
    for image_info in coco["images"]:
        physical_id = image_info["id"].split("-", 1)[1]
        assert physical_id == Path(image_info["file_name"]).stem
        records.append({
            "annotator_image_id": image_info["id"], "physical_id": physical_id, "file_name": image_info["file_name"],
            "image_path": Path(train_images_dir) / image_info["file_name"], "height": image_info["height"], "width": image_info["width"],
            "annotations": annotations_by_id[image_info["id"]],
        })
    return coco, records


def make_split(records, seed=42, validation_fraction=0.15):
    physical_ids = sorted({record["physical_id"] for record in records})
    random.Random(seed).shuffle(physical_ids)
    validation_ids = set(physical_ids[:round(len(physical_ids) * validation_fraction)])
    train_records = [record for record in records if record["physical_id"] not in validation_ids]
    validation_records = [record for record in records if record["physical_id"] in validation_ids]
    manifest = {
        "seed": seed, "validation_fraction": validation_fraction,
        "train_physical_ids": sorted({record["physical_id"] for record in train_records}),
        "validation_physical_ids": sorted(validation_ids),
    }
    return manifest, train_records, validation_records


def verify_or_write_split(records, manifest_path, seed=42, validation_fraction=0.15):
    expected, train_records, validation_records = make_split(records, seed, validation_fraction)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        assert saved == expected, f"Saved split does not match seed {seed} and validation fraction {validation_fraction}"
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
        saved = expected

    train_filaments = sum(len(record["annotations"]) for record in train_records)
    validation_filaments = sum(len(record["annotations"]) for record in validation_records)
    assert (len(saved["train_physical_ids"]), len(train_records), train_filaments) == (601, 984, 6941)
    assert (len(saved["validation_physical_ids"]), len(validation_records), validation_filaments) == (106, 170, 1258)
    return saved, train_records, validation_records


def contrast_detector_image(gray):
    # Detector input for the contrast experiment: raw grayscale plus CLAHE at two clip limits, since found rate climbs from 47% to 83% across contrast terciles
    return cv2.merge([gray, *(cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray) for clip in [1.5, 3.0])])


def write_contrast_images(paths, folder):
    # 4:4:4 JPEG: default chroma subsampling would blur the channels this input exists to separate
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for path in paths:
        written.append(folder / Path(path).name)
        cv2.imwrite(str(written[-1]), contrast_detector_image(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)), [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    return written


def build_yolo_dataset(coco, source_images_dir, manifest, dataset_dir):
    source_images_dir, dataset_dir = Path(source_images_dir), Path(dataset_dir)
    annotations_by_id = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_id[annotation["image_id"]].append(annotation)
    validation_ids = set(manifest["validation_physical_ids"])
    counts = {"train_samples": 0, "validation_samples": 0, "boxes": 0}

    for image_info in coco["images"]:
        physical_id = image_info["id"].split("-", 1)[1]
        split = "val" if physical_id in validation_ids else "train"
        image_dir, label_dir = dataset_dir / f"images/{split}", dataset_dir / f"labels/{split}"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source = source_images_dir / image_info["file_name"]
        destination = image_dir / f"{image_info['id']}{source.suffix.lower()}"
        shutil.copy2(source, destination)

        rows = []
        for annotation in annotations_by_id[image_info["id"]]:
            x, y, width, height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(image_info["width"]), x + width), min(float(image_info["height"]), y + height)
            clipped_width, clipped_height = x2 - x1, y2 - y1
            if clipped_width <= 0 or clipped_height <= 0:
                continue
            values = [((x1 + x2) / 2) / image_info["width"], ((y1 + y2) / 2) / image_info["height"],
                      clipped_width / image_info["width"], clipped_height / image_info["height"]]
            assert all(0.0 <= value <= 1.0 for value in values)
            rows.append("0 " + " ".join(f"{value:.6f}" for value in values))
            counts["boxes"] += 1
        (label_dir / f"{image_info['id']}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        counts["validation_samples" if split == "val" else "train_samples"] += 1

    (dataset_dir / "data.yaml").write_text(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\nnames:\n  0: filament\n")
    assert counts == {"train_samples": 984, "validation_samples": 170, "boxes": 8199}
    return counts


def union_boxes(record_boxes, iou_threshold=0.50):
    # record_boxes: one list of xyxy boxes per annotator of the same image. Greedy clustering, largest first, against each
    # cluster's mean box; one mean box per cluster, so a filament any annotator marked appears once.
    clusters = []
    for box in sorted([box for boxes in record_boxes for box in boxes], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if clusters:
            ious = box_iou_with_many(np.asarray(box, dtype=np.float32), np.asarray([np.mean(c, 0) for c in clusters], dtype=np.float32))
            if ious.max() > iou_threshold:
                clusters[int(ious.argmax())].append(box)
                continue
        clusters.append([box])
    return [np.mean(c, 0).tolist() for c in clusters]


def build_union_yolo_dataset(coco, manifest, standard_dir, dataset_dir):
    # Detector training on the union of annotators: one training sample per physical image instead of one per annotator
    # record, so a filament only some annotators marked is never taught as background. Validation stays per record.
    standard_dir, dataset_dir = Path(standard_dir), Path(dataset_dir)
    shutil.copytree(standard_dir / "images/val", dataset_dir / "images/val")
    shutil.copytree(standard_dir / "labels/val", dataset_dir / "labels/val")
    (dataset_dir / "images/train").mkdir(parents=True)
    (dataset_dir / "labels/train").mkdir(parents=True)
    annotations_by_id = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_id[annotation["image_id"]].append(annotation)
    validation_ids = set(manifest["validation_physical_ids"])
    groups = defaultdict(list)
    for image_info in coco["images"]:
        if image_info["id"].split("-", 1)[1] not in validation_ids:
            groups[image_info["id"].split("-", 1)[1]].append(image_info)
    counts = {"train_samples": 0, "boxes": 0}
    for physical_id, infos in groups.items():
        record_boxes = [[[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in annotations_by_id[info["id"]]] for info in infos]
        width, height = infos[0]["width"], infos[0]["height"]
        rows = []
        for x1, y1, x2, y2 in union_boxes(record_boxes):
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
            if x2 > x1 and y2 > y1:
                rows.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        source = next((standard_dir / "images/train").glob(f"{infos[0]['id']}.*"))
        shutil.copy2(source, dataset_dir / f"images/train/{physical_id}{source.suffix}")
        (dataset_dir / f"labels/train/{physical_id}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        counts["train_samples"] += 1
        counts["boxes"] += len(rows)
    (dataset_dir / "data.yaml").write_text(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\nnames:\n  0: filament\n")
    assert counts["train_samples"] == 601
    return counts


def consensus_annotations(records, min_iou=0.25):
    # For each annotation: itself plus the best-matching filament from every other annotator of the same image
    # (mask IoU >= min_iou). Annotators who did not draw the filament are left out, so the average is a consensus
    # on its outline, not on whether it exists. Used only for training targets; evaluation stays per annotator.
    groups = defaultdict(list)
    for record in records:
        groups[record["physical_id"]].append(record)
    consensus = {}
    for group in groups.values():
        rles = [[mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], r["height"], r["width"])) for a in r["annotations"]] for r in group]
        for i, record in enumerate(group):
            for k, annotation in enumerate(record["annotations"]):
                matched = [annotation]
                for j, other in enumerate(group):
                    if j != i and rles[j]:
                        ious = mask_utils.iou([rles[i][k]], rles[j], [0] * len(rles[j]))[0]
                        if ious.max() >= min_iou:
                            matched.append(other["annotations"][int(ious.argmax())])
                consensus[annotation["id"]] = matched
    return consensus


def box_iou_with_many(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    intersection_x1, intersection_y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    intersection_x2, intersection_y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(0, intersection_y2 - intersection_y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-6)


def mine_hard_negatives(train_records, detector, settings, device, desired_count, seed=42):
    image_paths, boxes_by_physical = {}, defaultdict(list)
    for record in train_records:
        image_paths.setdefault(record["physical_id"], record["image_path"])
        for annotation in record["annotations"]:
            x, y, width, height = map(float, annotation["bbox"])
            boxes_by_physical[record["physical_id"]].append([x, y, x + width, y + height])
    boxes_by_physical = {key: np.asarray(value, dtype=np.float32) for key, value in boxes_by_physical.items()}

    candidates = []
    for physical_id, image_path in image_paths.items():
        result = detector.predict(
            source=str(image_path), batch=1, imgsz=settings["yolo_image_size"], conf=settings["yolo_mining_confidence"],
            iou=settings["yolo_iou"], max_det=settings["max_yolo_detections"], end2end=settings["yolo_end2end"],
            device=0 if device.type == "cuda" else "cpu", verbose=False,
        )[0]
        ground_truth = boxes_by_physical[physical_id]
        gt_center_x, gt_center_y = (ground_truth[:, 0] + ground_truth[:, 2]) / 2, (ground_truth[:, 1] + ground_truth[:, 3]) / 2
        accepted = 0
        for box, score in zip(result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()):
            if accepted >= settings["max_negatives_per_image"]:
                break
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            proposal_box = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            geometry = make_crop_geometry(proposal_box, settings["train_expansion_max"], result.orig_shape[1], result.orig_shape[0])
            crop_x2, crop_y2 = geometry["x1"] + geometry["side"], geometry["y1"] + geometry["side"]
            gt_center_in_crop = ((gt_center_x >= geometry["x1"]) & (gt_center_x <= crop_x2) & (gt_center_y >= geometry["y1"]) & (gt_center_y <= crop_y2)).any()
            proposal_center_in_gt = ((center_x >= ground_truth[:, 0]) & (center_x <= ground_truth[:, 2]) & (center_y >= ground_truth[:, 1]) & (center_y <= ground_truth[:, 3])).any()
            if box_iou_with_many(box, ground_truth).max(initial=0.0) < settings["negative_iou_limit"] and not gt_center_in_crop and not proposal_center_in_gt:
                candidates.append({"physical_id": physical_id, "file_name": Path(result.path).name, "proposal_box": proposal_box, "score": float(score)})
                accepted += 1

    random.Random(seed).shuffle(candidates)
    return candidates[:desired_count]


def jitter_box(box, amount):
    x, y, width, height = map(float, box)
    x += random.uniform(-amount, amount) * width
    y += random.uniform(-amount, amount) * height
    return [x, y, width, height]


def make_crop_geometry(box, expansion, image_width, image_height, min_crop_side=64):
    x, y, width, height = map(float, box)
    center_x, center_y = x + width / 2, y + height / 2
    side = int(math.ceil(max(min_crop_side, max(width, height) * expansion)))
    side = min(side, max(image_width, image_height))
    return {"x1": int(math.floor(center_x - side / 2)), "y1": int(math.floor(center_y - side / 2)), "side": side,
            "image_width": image_width, "image_height": image_height}


def extract_padded_square(image, geometry):
    x1, y1, side = geometry["x1"], geometry["y1"], geometry["side"]
    image_height, image_width = image.shape
    source_x1, source_y1 = max(0, x1), max(0, y1)
    source_x2, source_y2 = min(image_width, x1 + side), min(image_height, y1 + side)
    destination_x1, destination_y1 = source_x1 - x1, source_y1 - y1
    crop = np.zeros((side, side), dtype=image.dtype)
    crop[destination_y1:destination_y1 + source_y2 - source_y1, destination_x1:destination_x1 + source_x2 - source_x1] = image[source_y1:source_y2, source_x1:source_x2]
    return crop


def build_crop_tensors(image, proposal_box, annotation, expansion, crop_size=512, min_crop_side=64, consensus=None):
    image_height, image_width = image.shape
    geometry = make_crop_geometry(proposal_box, expansion, image_width, image_height, min_crop_side)
    crop = extract_padded_square(image, geometry)
    interpolation = cv2.INTER_AREA if geometry["side"] > crop_size else cv2.INTER_LINEAR
    resized_original = cv2.resize(crop, (crop_size, crop_size), interpolation=interpolation)
    resized_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(resized_original)

    scale = crop_size / geometry["side"]
    x, y, width, height = map(float, proposal_box)
    x1 = int(np.clip(np.floor((x - geometry["x1"]) * scale), 0, crop_size))
    y1 = int(np.clip(np.floor((y - geometry["y1"]) * scale), 0, crop_size))
    x2 = int(np.clip(np.ceil((x + width - geometry["x1"]) * scale), 0, crop_size))
    y2 = int(np.clip(np.ceil((y + height - geometry["y1"]) * scale), 0, crop_size))
    prompt = np.zeros((crop_size, crop_size), dtype=np.float32)
    prompt[y1:y2, x1:x2] = 1.0

    # With consensus, the target is the average of every annotator's version of this filament (a soft mask)
    versions = consensus or ([annotation] if annotation is not None else [])
    target = np.zeros((crop_size, crop_size), dtype=np.float32)
    for version in versions:
        layer = np.zeros((crop_size, crop_size), dtype=np.uint8)
        for polygon in version["segmentation"]:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = (points[:, 0] - geometry["x1"]) * scale
            points[:, 1] = (points[:, 1] - geometry["y1"]) * scale
            cv2.fillPoly(layer, [np.rint(points).astype(np.int32)], 1)
        target += layer
    if versions:
        target /= len(versions)

    inputs = torch.from_numpy(np.stack((resized_original / 255.0, resized_clahe / 255.0, prompt), axis=0)).float()
    return inputs, torch.from_numpy(target).unsqueeze(0), geometry


class FilamentCropDataset(Dataset):
    def __init__(self, samples, crop_settings, augment=False):
        self.samples = samples
        self.settings = crop_settings
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = np.asarray(Image.open(sample["image_path"]).convert("L"), dtype=np.uint8)
        is_positive = sample["annotation"] is not None
        if is_positive:
            box = jitter_box(sample["annotation"]["bbox"], self.settings["center_jitter"]) if self.augment else sample["annotation"]["bbox"]
        else:
            box = sample["proposal_box"]
        expansion = random.uniform(self.settings["train_expansion_min"], self.settings["train_expansion_max"]) if self.augment else self.settings["crop_expansion"]
        inputs, target, _ = build_crop_tensors(image, box, sample["annotation"], expansion, self.settings["crop_size"], self.settings["min_crop_side"], sample.get("consensus"))
        if self.augment:
            if random.random() < 0.5:
                inputs, target = inputs.flip(-1), target.flip(-1)
            if random.random() < 0.5:
                inputs, target = inputs.flip(-2), target.flip(-2)
            rotation = random.randrange(4)
            inputs, target = torch.rot90(inputs, rotation, (-2, -1)), torch.rot90(target, rotation, (-2, -1))
        return inputs, target, torch.tensor(is_positive, dtype=torch.bool)
