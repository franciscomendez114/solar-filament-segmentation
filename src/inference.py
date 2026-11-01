from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

from .config import ENSEMBLE_DETECTORS, ENSEMBLE_SETTINGS
from .data import box_iou_with_many, build_crop_tensors
from .metrics import aggregate_official, annotation_to_rle, official_match_counts


def predict_instances(image_path, yolo_model, unet_model, device, mask_threshold, settings, use_amp=None, candidate_filter=None, return_candidates=False, proposals=None):
    use_amp = device.type == "cuda" if use_amp is None else use_amp
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    image_height, image_width = image.shape
    if proposals is None:
        result = yolo_model.predict(
            str(image_path), imgsz=settings["yolo_image_size"], conf=settings["yolo_confidence"], iou=settings["yolo_iou"],
            max_det=settings["max_yolo_detections"], end2end=settings["yolo_end2end"],
            device=0 if device.type == "cuda" else "cpu", half=device.type == "cuda", verbose=False,
        )[0]
        boxes, scores = result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()
    else:
        boxes, scores = proposals  # the detector is frozen during Stage 2, so its boxes are cached once and reused
    raw_count = len(boxes)
    empty = {"raw_candidates": raw_count, "used_candidates": 0, "nonempty_before_dedup": 0, "after_dedup": 0, "final_instances": 0}
    if raw_count == 0:
        return [], empty

    order = np.argsort(scores)[::-1][:settings["max_crops_per_image"]]
    boxes, scores = boxes[order], scores[order]
    # Optional mask TTA over crop views: each [crop expansion, prompt-box scale] pair is predicted and the pasted
    # probabilities are averaged per box. The default single view reproduces the original path exactly.
    views = settings.get("crop_views") or [[settings["crop_expansion"], 1.0]]
    crop_inputs, geometries = [], []
    for box in boxes:
        for expansion, box_scale in views:
            proposal_box = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            if box_scale != 1.0:
                width, height = proposal_box[2] * box_scale, proposal_box[3] * box_scale
                proposal_box = [proposal_box[0] + (proposal_box[2] - width) / 2, proposal_box[1] + (proposal_box[3] - height) / 2, width, height]
            inputs, _, geometry = build_crop_tensors(image, proposal_box, None, expansion, settings["crop_size"], settings["min_crop_side"])
            crop_inputs.append(inputs)
            geometries.append(geometry)

    probability_maps, instance_scores, candidates = [], [], []
    summed = [None] * len(boxes)
    unet_model.eval()
    with torch.inference_mode():
        for start in range(0, len(crop_inputs), settings["inference_batch_size"]):
            batch = torch.stack(crop_inputs[start:start + settings["inference_batch_size"]]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                probabilities = torch.sigmoid(unet_model(batch))[:, 0]
            for offset, crop_probability in enumerate(probabilities):
                geometry = geometries[start + offset]
                side, crop_x1, crop_y1 = geometry["side"], geometry["x1"], geometry["y1"]
                resized = F.interpolate(crop_probability[None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
                full = torch.zeros((image_height, image_width), dtype=resized.dtype, device=device)
                full_x1, full_y1 = max(0, crop_x1), max(0, crop_y1)
                full_x2, full_y2 = min(image_width, crop_x1 + side), min(image_height, crop_y1 + side)
                source_x1, source_y1 = full_x1 - crop_x1, full_y1 - crop_y1
                full[full_y1:full_y2, full_x1:full_x2] = resized[source_y1:source_y1 + full_y2 - full_y1, source_x1:source_x1 + full_x2 - full_x1]
                box_index = (start + offset) // len(views)
                summed[box_index] = full if summed[box_index] is None else summed[box_index] + full
        for index, total in enumerate(summed):
            full = total if len(views) == 1 else total / len(views)
            geometry = geometries[index * len(views)]
            binary = full >= mask_threshold
            if binary.sum().item() < settings["minimum_mask_area"]:
                continue
            values = full[binary].float()
            probability_maps.append(full)
            instance_scores.append(float(scores[index]) * values.mean().item())
            rows, columns = binary.any(1).nonzero(), binary.any(0).nonzero()
            y1, y2, x1, x2 = rows.min().item(), rows.max().item(), columns.min().item(), columns.max().item()
            box_height, box_width = y2 - y1 + 1, x2 - x1 + 1
            quantiles = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75, 0.90], device=values.device)).tolist()
            candidates.append({
                "yolo_confidence": float(scores[index]), "mean_probability": values.mean().item(), "instance_score": instance_scores[-1],
                "area": int(binary.sum().item()), "probability_p25": quantiles[0], "probability_p50": quantiles[1],
                "probability_p75": quantiles[2], "probability_p90": quantiles[3], "crop_side": geometry["side"],
                "box_height": box_height, "box_width": box_width, "fill_ratio": int(binary.sum().item()) / (box_height * box_width),
                "elongation": max(box_height, box_width) / min(box_height, box_width),
                "radius": float(np.hypot((x1 + x2) / 2 - image_width / 2, (y1 + y2) / 2 - image_height / 2)),
            })

    if not probability_maps:
        empty["used_candidates"] = len(boxes)
        return [], empty

    binary_masks = [probability >= mask_threshold for probability in probability_maps]
    kept_indices = []
    for candidate_index in sorted(range(len(binary_masks)), key=lambda i: instance_scores[i], reverse=True):
        candidate = binary_masks[candidate_index]
        candidate_area = candidate.sum().item()
        duplicate = False
        for kept_index in kept_indices:
            kept = binary_masks[kept_index]
            intersection = (candidate & kept).sum().item()
            if intersection == 0:
                continue
            union = candidate_area + kept.sum().item() - intersection
            containment = intersection / min(candidate_area, kept.sum().item())
            if intersection / union >= settings["dedup_iou"] or containment >= settings["dedup_containment"]:
                duplicate = True
                break
        if not duplicate:
            kept_indices.append(candidate_index)

    # Neighbour count is only known once deduplication has settled, so it is added here rather than in the crop loop
    for position, index in enumerate(kept_indices):
        candidates[index]["neighbours"] = sum(1 for other in kept_indices if other != index and (binary_masks[index] & binary_masks[other]).any().item())
        candidates[index]["rank"] = position
    kept_candidates = [candidates[index] for index in kept_indices]
    if candidate_filter is not None:
        keep = candidate_filter(kept_candidates)
        assert len(keep) == len(kept_candidates), "candidate_filter must return one decision per candidate"
        kept_indices = [index for index, decision in zip(kept_indices, keep) if decision]
        kept_candidates = [candidate for candidate, decision in zip(kept_candidates, keep) if decision]
    if return_candidates:
        metadata = {"raw_candidates": raw_count, "used_candidates": len(boxes), "nonempty_before_dedup": len(probability_maps),
                    "after_dedup": len(kept_indices), "final_instances": len(kept_indices)}
        return [{**candidate, "mask": binary_masks[index].cpu().numpy().astype(np.uint8)} for candidate, index in zip(kept_candidates, kept_indices)], metadata
    if not kept_indices:
        empty["used_candidates"] = len(boxes)
        empty["nonempty_before_dedup"] = len(probability_maps)
        return [], empty

    kept_probabilities = torch.stack([probability_maps[index] for index in kept_indices])
    kept_binary = kept_probabilities >= mask_threshold
    kept_scores = torch.tensor([instance_scores[index] for index in kept_indices], dtype=kept_probabilities.dtype, device=device)
    pixel_scores = kept_probabilities * kept_scores[:, None, None]
    pixel_scores[~kept_binary] = -1
    winning_scores, winning_indices = pixel_scores.max(dim=0)
    any_prediction = winning_scores >= 0
    final_masks = []
    for index in range(len(kept_indices)):
        mask = (winning_indices == index) & any_prediction
        if mask.sum().item() >= settings["minimum_mask_area"]:
            final_masks.append(mask.cpu().numpy().astype(np.uint8))

    metadata = {
        "raw_candidates": raw_count, "used_candidates": len(boxes), "nonempty_before_dedup": len(probability_maps),
        "after_dedup": len(kept_indices), "final_instances": len(final_masks),
    }
    return final_masks, metadata


def cache_proposals(detector, image_paths, device, settings, flip=None):
    # One image per predict call; batching all paths previously caused CUDA OOM.
    # flip "h" or "v" runs the detector on the mirrored image and mirrors its boxes back (test-time augmentation).
    cache = {}
    for image_path in image_paths:
        source = str(image_path)
        if flip:
            image = cv2.imread(source)
            source = np.ascontiguousarray(image[:, ::-1] if flip == "h" else image[::-1])
        result = detector.predict(
            source, imgsz=settings["yolo_image_size"], conf=settings["yolo_confidence"], iou=settings["yolo_iou"],
            max_det=settings["max_yolo_detections"], end2end=settings["yolo_end2end"],
            device=0 if device.type == "cuda" else "cpu", half=device.type == "cuda", verbose=False,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        height, width = result.orig_shape
        if flip == "h":
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        elif flip == "v":
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        cache[Path(image_path).stem] = (boxes, result.boxes.conf.detach().cpu().numpy())
    return cache


def fuse_boxes(model_proposals, weights, iou_threshold=0.55, skip_score=0.01):
    # Weighted box fusion across detectors: boxes cluster greedily by descending score, coordinates are score-weighted
    # averages, and a cluster scores by weighted agreement (each detector's best box counts once), so a filament every
    # detector finds outranks one only a single detector finds at the same confidence.
    entries = sorted(((score, model, box) for model, (boxes, scores) in enumerate(model_proposals) for box, score in zip(boxes, scores) if score >= skip_score), key=lambda entry: -entry[0])
    clusters, fused = [], np.zeros((0, 4), dtype=np.float64)
    for score, model, box in entries:
        ious = box_iou_with_many(box, fused)
        if len(ious) and ious.max() > iou_threshold:
            best = int(ious.argmax())
            clusters[best].append((score, model, box))
            member_scores, member_boxes = np.array([m[0] for m in clusters[best]]), np.array([m[2] for m in clusters[best]])
            fused[best] = (member_scores[:, None] * member_boxes).sum(0) / member_scores.sum()
        else:
            clusters.append([(score, model, box)])
            fused = np.vstack([fused, box])
    fused_scores = []
    for members in clusters:
        best_per_model = {}
        for score, model, _ in members:
            best_per_model[model] = max(score, best_per_model.get(model, 0.0))
        fused_scores.append(sum(weights[model] * score for model, score in best_per_model.items()) / sum(weights))
    order = np.argsort(fused_scores)[::-1]
    return fused[order].astype(np.float32), np.asarray(fused_scores, dtype=np.float32)[order]


def box_nms(boxes, scores, iou_threshold):
    keep = []
    for index in np.argsort(scores)[::-1]:
        if not keep or box_iou_with_many(boxes[index], boxes[keep]).max() <= iou_threshold:
            keep.append(index)
    return boxes[keep], scores[keep]


def fuse_detector_caches(caches, ensemble_settings):
    # caches: one {image stem: (boxes, scores)} per detector, each cached at ensemble_settings["cache_confidence"]
    proposals = {}
    for stem in caches[0]:
        boxes, scores = fuse_boxes([cache[stem] for cache in caches], [1.0] * len(caches), ensemble_settings["wbf_iou"], ensemble_settings["cache_confidence"])
        keep = scores >= ensemble_settings["fused_threshold"]
        proposals[stem] = (boxes[keep], scores[keep])
    return proposals



def ensemble_proposals(models_dir, image_paths, device, settings):
    from ultralytics import YOLO
    caches = []
    for name, (folder, image_size) in ENSEMBLE_DETECTORS.items():
        print(f"Detector {name}: {folder} @ {image_size}")
        detector_settings = {**settings, "yolo_confidence": ENSEMBLE_SETTINGS["cache_confidence"], "yolo_image_size": image_size}
        caches.append(cache_proposals(YOLO(str(Path(models_dir) / "stage1" / folder / "weights/best.pt")), image_paths, device, detector_settings))
    return fuse_detector_caches(caches, ENSEMBLE_SETTINGS)

def official_pq_for_model(unet_model, validation_records, proposals, device, mask_threshold, settings, use_amp=None):
    # Full pipeline under the official rule: one prediction per physical image, scored against every annotator record.
    records_by_image = defaultdict(list)
    for record in validation_records:
        records_by_image[record["physical_id"]].append(record)
    rows = []
    for physical_id, records in records_by_image.items():
        masks, _ = predict_instances(records[0]["image_path"], None, unet_model, device, mask_threshold, settings,
                                     use_amp=use_amp, proposals=proposals[physical_id])
        predicted = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
        for record in records:
            targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
            rows.append(official_match_counts(predicted, targets))
    return aggregate_official(pd.DataFrame(rows))


def masks_to_submission_rows(image_path, masks):
    rows, claimed = [], np.zeros((2048, 2048), dtype=bool)
    for number, mask in enumerate(masks, 1):
        mask = mask.astype(bool)
        if mask.shape != (2048, 2048):
            raise ValueError(f"Wrong mask shape for {image_path}: {mask.shape}")
        if np.any(mask & claimed):
            raise ValueError(f"Overlapping masks for {image_path}")
        claimed |= mask
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rows.append({"filament_id": f"{Path(image_path).stem}_{number}", "segmentation_rle": rle["counts"].decode("ascii")})
    return rows


def validate_submission(frame):
    assert list(frame.columns) == ["filament_id", "segmentation_rle"]
    assert not frame["filament_id"].duplicated().any()
    assert not frame.isna().any().any()
    parsed = frame.copy()
    parsed["image_id"] = parsed["filament_id"].map(lambda value: value.rpartition("_")[0])
    assert parsed["filament_id"].map(lambda value: bool(value.rpartition("_")[0]) and value.rpartition("_")[2].isdigit()).all()
    for image_id, group in parsed.groupby("image_id", sort=False):
        claimed = np.zeros((2048, 2048), dtype=bool)
        for row in group.itertuples(index=False):
            decoded = mask_utils.decode({"size": [2048, 2048], "counts": row.segmentation_rle.encode("ascii")})
            assert decoded.shape == (2048, 2048)
            decoded = decoded.astype(bool)
            assert not np.any(claimed & decoded), f"Overlapping masks for {image_id}"
            claimed |= decoded
    return True


def create_submission(test_image_paths, yolo_model, unet_model, device, mask_threshold, settings, proposals=None):
    rows, statistics = [], []
    for image_path in test_image_paths:
        image_proposals = None if proposals is None else proposals[Path(image_path).stem]
        masks, metadata = predict_instances(image_path, yolo_model, unet_model, device, mask_threshold, settings, proposals=image_proposals)
        rows.extend(masks_to_submission_rows(image_path, masks))
        statistics.append({"image": Path(image_path).name, **metadata})
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    validate_submission(frame)
    return frame, pd.DataFrame(statistics)
