import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def annotation_to_rle(annotation, height=2048, width=2048):
    return mask_utils.merge(mask_utils.frPyObjects(annotation["segmentation"], height, width))


def normalize_saved_rles(rles):
    return [{"size": rle["size"], "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]} for rle in rles]


def official_match_counts(predicted_rles, target_rles):
    if not predicted_rles or not target_rles:
        return {"official_tp": 0, "official_fp": len(predicted_rles), "official_fn": len(target_rles), "official_iou_sum": 0.0}
    ious = mask_utils.iou(predicted_rles, target_rles, [0] * len(target_rles)).astype(np.float32)
    hits = ious > 0.50
    return {
        "official_tp": int(hits.sum()), "official_fp": int((~hits.any(axis=1)).sum()),
        "official_fn": int((~hits.any(axis=0)).sum()), "official_iou_sum": float(ious[hits].astype(np.float64).sum()),
    }


def aggregate_official(frame):
    tp, fp, fn = (int(frame[column].sum()) for column in ["official_tp", "official_fp", "official_fn"])
    iou_sum = float(frame["official_iou_sum"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "official_pq": iou_sum / denominator if denominator else 0.0,
        "official_sq": iou_sum / tp if tp else 0.0,
        "official_rq": tp / denominator if denominator else 0.0,
        "official_tp": tp, "official_fp": fp, "official_fn": fn, "official_iou_sum": iou_sum,
    }


def score_prediction_file(predictions_path, validation_records):
    entries = [json.loads(line) for line in Path(predictions_path).read_text().splitlines() if line.strip()]
    predictions = {}
    for entry in entries:
        key = entry.get("physical_id") or Path(entry["file_name"]).stem
        predictions[key] = normalize_saved_rles(entry["rles"])

    rows = []
    for record in validation_records:
        predicted = predictions[record["physical_id"]]
        targets = [annotation_to_rle(annotation, record["height"], record["width"]) for annotation in record["annotations"]]
        rows.append({"physical_id": record["physical_id"], "annotator_image_id": record["annotator_image_id"],
                     "ground_truth_instances": len(targets), "predicted_instances": len(predicted), **official_match_counts(predicted, targets)})
    frame = pd.DataFrame(rows)
    return frame, aggregate_official(frame)


def bootstrap_pq(frame, samples=2000, seed=0):
    columns = ["official_tp", "official_fp", "official_fn", "official_iou_sum"]
    by_image = frame.groupby("physical_id")[columns].sum().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    totals = by_image[rng.integers(0, len(by_image), (samples, len(by_image)))].sum(axis=1)
    pq = totals[:, 3] / np.maximum(totals[:, 0] + 0.5 * totals[:, 1] + 0.5 * totals[:, 2], 1e-9)
    return {"pq_ci_low": float(np.percentile(pq, 2.5)), "pq_ci_high": float(np.percentile(pq, 97.5)), "pq_bootstrap_sd": float(pq.std())}


def pq_by_annotator(frame, minimum_records=3):
    # Each record's annotator is the prefix of its annotator_image_id. Annotators disagree enough that the
    # same predictions score 0.34 against one and 0.50 against another on identical images, so the pooled
    # number is an average over whichever annotators the split happened to contain.
    columns = ["official_tp", "official_fp", "official_fn", "official_iou_sum"]
    table = frame.copy()
    table["annotator"] = table["annotator_image_id"].str.split("-").str[0]
    table["team"] = table["annotator"].str[:4]
    grouped = table.groupby(["team", "annotator"]).agg(records=("annotator_image_id", "size"), **{c: (c, "sum") for c in columns})
    grouped["official_pq"] = grouped["official_iou_sum"] / (grouped["official_tp"] + 0.5 * grouped["official_fp"] + 0.5 * grouped["official_fn"]).clip(lower=1e-9)
    grouped["recall"] = grouped["official_tp"] / (grouped["official_tp"] + grouped["official_fn"]).clip(lower=1)
    return grouped[grouped["records"] >= minimum_records].sort_values("official_pq").reset_index()


def bootstrap_pq_by_annotator(frame, samples=2000, seed=0):
    # Resampling records treats two tracings of one image as independent draws; they are not. The annotator is
    # the unit of variation, so resample annotators. This gives the honest error bar on any PQ comparison.
    columns = ["official_tp", "official_fp", "official_fn", "official_iou_sum"]
    table = frame.copy()
    table["annotator"] = table["annotator_image_id"].str.split("-").str[0]
    by_annotator = table.groupby("annotator")[columns].sum().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    totals = by_annotator[rng.integers(0, len(by_annotator), (samples, len(by_annotator)))].sum(axis=1)
    pq = totals[:, 3] / np.maximum(totals[:, 0] + 0.5 * totals[:, 1] + 0.5 * totals[:, 2], 1e-9)
    per_annotator = by_annotator[:, 3] / np.maximum(by_annotator[:, 0] + 0.5 * by_annotator[:, 1] + 0.5 * by_annotator[:, 2], 1e-9)
    return {"annotator_pq_ci_low": float(np.percentile(pq, 2.5)), "annotator_pq_ci_high": float(np.percentile(pq, 97.5)),
            "annotator_pq_bootstrap_sd": float(pq.std()), "annotator_spread_sd": float(per_annotator.std()),
            "worst_annotator_pq": float(per_annotator.min()), "best_annotator_pq": float(per_annotator.max()),
            "annotators": int(len(by_annotator))}


def paired_pq_delta(control, candidate, samples=2000, seed=0):
    columns = ["official_tp", "official_fp", "official_fn", "official_iou_sum"]
    left = control.groupby("physical_id")[columns].sum().sort_index()
    right = candidate.groupby("physical_id")[columns].sum().sort_index()
    assert left.index.equals(right.index)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), (samples, len(left)))
    deltas = []
    for table in [right.to_numpy(), left.to_numpy()]:
        totals = table[indices].sum(axis=1)
        deltas.append(totals[:, 3] / np.maximum(totals[:, 0] + 0.5 * totals[:, 1] + 0.5 * totals[:, 2], 1e-9))
    difference = deltas[0] - deltas[1]
    return {"paired_delta": float(difference.mean()), "paired_delta_ci_low": float(np.percentile(difference, 2.5)),
            "paired_delta_ci_high": float(np.percentile(difference, 97.5))}


def human_agreement(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["physical_id"]].append(record)
    rows = []
    for physical_id, group in grouped.items():
        for first, second in combinations(group, 2):
            predicted = [annotation_to_rle(annotation, first["height"], first["width"]) for annotation in first["annotations"]]
            targets = [annotation_to_rle(annotation, second["height"], second["width"]) for annotation in second["annotations"]]
            rows.append({"physical_id": physical_id, **official_match_counts(predicted, targets)})
    frame = pd.DataFrame(rows)
    return frame, aggregate_official(frame)

