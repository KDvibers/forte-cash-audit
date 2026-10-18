"""Local, single-class box evaluation without model calls or hidden negatives.

All boxes use full-image XYXY coordinates, including predictions made on crops.
This evaluates annotated objects, not actions, authenticity or violations.
"""

from collections.abc import Mapping
import math
from numbers import Real


_CONTEXTS = ("in_hand", "in_tray", "on_table", "unclear")
_COUNTS = ("frames", "ground_truth", "predictions", "below_threshold_predictions",
           "ignored_predictions", "tp", "fp", "fn")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return value


def _box(value, name):
    values = _sequence(value, name)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    result = tuple(_number(v, name) for v in values)
    if not (0 <= result[0] < result[2] and 0 <= result[1] < result[3]):
        raise ValueError(f"{name} must be a nonempty rectangle with nonnegative coordinates")
    return result


def _inside(box, bounds):
    return (bounds[0] <= box[0] < box[2] <= bounds[2]
            and bounds[1] <= box[1] < box[3] <= bounds[3])


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    intersection = _intersection(a, b)
    return intersection / (_area(a) + _area(b) - intersection)


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every row and prediction requires a nonempty string id")
    return value


def _index(items, name):
    indexed = {}
    for item in _sequence(items, name):
        if not isinstance(item, Mapping):
            raise ValueError(f"Every {name} item must be an object")
        identifier = _identifier(item.get("id"))
        if identifier in indexed:
            raise ValueError(f"Duplicate {name} id: {identifier}")
        indexed[identifier] = item
    return indexed


def _empty_counts():
    return dict.fromkeys(_COUNTS, 0)


def _ratios(counts):
    return {**counts,
            "precision": counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None,
            "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None}


def _context_counts():
    return {name: {"tp": 0, "fn": 0} for name in _CONTEXTS}


def _context_ratios(contexts):
    return {name: {**counts, "ground_truth": counts["tp"] + counts["fn"],
                   "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None}
            for name, counts in contexts.items()}


def evaluate_detections(rows, predictions, score_threshold=.5, iou_threshold=.5, ignore_overlap=.5):
    """Match confident predictions greedily by descending score, once per GT.

    ``rows`` are export-manifest rows. ``predictions`` contains exactly one
    ``{id, boxes_xyxy, scores}`` object per row, even for successful empty output.
    A missing/failed result is an error, never an inferred negative. An unmatched
    prediction can be ignored when intersection / prediction area reaches the
    ignore threshold. A duplicate of a matched GT remains a false positive.
    Context recall comes from GT labels; false positives have no invented context.
    The function neither mutates inputs nor reads any image or file.
    """
    thresholds = {"score_threshold": score_threshold, "iou_threshold": iou_threshold,
                  "ignore_overlap": ignore_overlap}
    for name, value in thresholds.items():
        number = _number(value, name)
        if not (0 <= number <= 1) or (name != "score_threshold" and number == 0):
            raise ValueError(f"{name} is outside its allowed probability range")
        thresholds[name] = number
    score_threshold, iou_threshold, ignore_overlap = thresholds.values()
    indexed_rows = _index(rows, "rows")
    indexed_predictions = _index(predictions, "predictions")
    if not indexed_rows:
        raise ValueError("Evaluation requires at least one annotated row")
    if indexed_rows.keys() != indexed_predictions.keys():
        missing = sorted(indexed_rows.keys() - indexed_predictions.keys())
        extra = sorted(indexed_predictions.keys() - indexed_rows.keys())
        raise ValueError(f"Prediction ids do not match rows: missing={missing}, extra={extra}")

    totals = _empty_counts()
    contexts = _context_counts()
    videos = {}
    details = []
    for identifier, row in indexed_rows.items():
        prediction = indexed_predictions[identifier]
        if prediction.get("error") is not None or prediction.get("errors") is not None:
            raise ValueError(f"Prediction {identifier} reports an error")
        if "status" in prediction and prediction["status"] not in {"ok", "done", "success", "succeeded"}:
            raise ValueError(f"Prediction {identifier} was not successful")
        if "success" in prediction and prediction["success"] is not True:
            raise ValueError(f"Prediction {identifier} was not successful")
        if row.get("fully_annotated") is not True:
            raise ValueError(f"Row {identifier} must be fully annotated inside its ROI")
        video_id = row.get("video_id")
        if isinstance(video_id, bool) or not isinstance(video_id, (str, int)) or not str(video_id).strip():
            raise ValueError(f"Row {identifier} requires a video_id")
        video_id = str(video_id)
        roi = _box(row.get("roi_xyxy"), "roi_xyxy")
        ground_truth = [_box(b, "ground truth box") for b in _sequence(row.get("boxes_xyxy"), "boxes_xyxy")]
        ignored = [_box(b, "ignore box") for b in _sequence(row.get("ignore_boxes_xyxy"), "ignore_boxes_xyxy")]
        if any(not _inside(box, roi) for box in ground_truth + ignored):
            raise ValueError(f"Annotation box is outside the ROI in {identifier}")
        labels = _sequence(row.get("box_contexts"), "box_contexts")
        if len(labels) != len(ground_truth) or any(not isinstance(c, str) or c not in _CONTEXTS for c in labels):
            raise ValueError("Each ground truth box requires one allowed context")
        boxes = [_box(b, "prediction box") for b in _sequence(prediction.get("boxes_xyxy"), "prediction boxes_xyxy")]
        scores = [_number(s, "score") for s in _sequence(prediction.get("scores"), "scores")]
        if len(boxes) != len(scores) or any(not 0 <= score <= 1 for score in scores):
            raise ValueError("Every prediction box requires one score between zero and one")

        unmatched_gt = set(range(len(ground_truth)))
        matches, false_positive, ignored_predictions = [], [], []
        selected = sorted((i for i, s in enumerate(scores) if s >= score_threshold), key=lambda i: (-scores[i], i))
        for p in selected:
            overlaps = [_iou(boxes[p], box) for box in ground_truth]
            candidates = [g for g in sorted(unmatched_gt) if overlaps[g] >= iou_threshold]
            if candidates:
                g = max(candidates, key=lambda index: overlaps[index])
                unmatched_gt.remove(g)
                matches.append({"prediction_index": p, "ground_truth_index": g,
                                "score": scores[p], "iou": overlaps[g], "context": labels[g]})
            elif any(overlap >= iou_threshold for overlap in overlaps):
                # Existing real objects take precedence over ambiguous regions.
                false_positive.append(p)
            elif any(_intersection(boxes[p], box) / _area(boxes[p]) >= ignore_overlap for box in ignored):
                ignored_predictions.append(p)
            else:
                false_positive.append(p)

        counts = {"frames": 1, "ground_truth": len(ground_truth), "predictions": len(selected),
                  "below_threshold_predictions": len(boxes) - len(selected),
                  "ignored_predictions": len(ignored_predictions), "tp": len(matches),
                  "fp": len(false_positive), "fn": len(unmatched_gt)}
        video = videos.setdefault(video_id, {"counts": _empty_counts(), "contexts": _context_counts()})
        for key in _COUNTS:
            totals[key] += counts[key]
            video["counts"][key] += counts[key]
        for g, context in enumerate(labels):
            outcome = "fn" if g in unmatched_gt else "tp"
            contexts[context][outcome] += 1
            video["contexts"][context][outcome] += 1
        details.append({"id": identifier, "video_id": video_id, **_ratios(counts), "matches": matches,
                        "false_positive_indices": false_positive, "ignored_prediction_indices": ignored_predictions,
                        "false_negative_indices": sorted(unmatched_gt)})
    return {"schema_version": 1, "kind": "local_single_class_box_metrics", **thresholds,
            "coordinate_system": "full_image_xyxy", "matching": "greedy_descending_score_one_to_one",
            "totals": _ratios(totals), "by_context": _context_ratios(contexts),
            "by_video": {key: {**_ratios(value["counts"]), "by_context": _context_ratios(value["contexts"])}
                         for key, value in videos.items()}, "rows": details}
