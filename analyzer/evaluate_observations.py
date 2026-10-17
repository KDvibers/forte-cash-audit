"""Evaluate local visual observations against fixed labels, without label leakage."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import time

from PIL import Image

from .features import CASH_FEATURES, FEATURE_STATES
from .local_client import LocalModelClient
from .observations import (
    DOOR_PROMPT, DOOR_SCHEMA, FEATURE_SUPPORT, ImageObserver, ObservationDeadline,
    policy_fingerprint, read_enum,
)
from .video import VideoFrame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = {feature: (feature, allowed) for feature, allowed in FEATURE_STATES.items()}
# Existing fixed annotations use metal_cash_cabinet. Accept it without requiring
# old labels to be rewritten, while new annotations may use the catalog name.
TRUTH_KEYS = {feature: (feature,) for feature in FEATURES}
TRUTH_KEYS["cabinet_door"] = ("cabinet_door", "metal_cash_cabinet")
SPLITS = ("develop", "reserve")


def feature_metrics(rows, feature):
    """Errors are separate from abstention; undefined fractions are JSON null.

    not_visible is a definite cabinet-visibility class, not a closed-door claim.
    Known GT includes all valid labels except uncertain. Coverage uses all known
    GT, including technical failures, so failures cannot improve coverage.
    """
    if feature not in FEATURES:
        raise ValueError(f"Unsupported feature: {feature!r}")
    allowed = FEATURES[feature][1]
    counts = Counter()
    truth_counts = Counter()
    confusion = {}
    for row in rows:
        # An unlabelled feature is not negative ground truth or an abstention.
        if feature not in row["truth"]:
            continue
        truth = row["truth"][feature]
        if truth not in allowed:
            raise ValueError(f"Invalid ground truth for {feature}: {truth!r}")
        pred = row.get("pred", {}).get(feature)
        failed = any(error.get("feature") in (feature, "all") for error in row.get("errors", []))
        unsupported = row.get("feature_support", {}).get(feature, {}).get("status") == "not_validated"
        failed = failed or (not unsupported and pred not in allowed)
        prediction = "error" if failed else "not_validated" if unsupported else pred
        truth_counts[truth] += 1
        confusion.setdefault(truth, {})[prediction] = confusion.get(truth, {}).get(prediction, 0) + 1
        counts["total"] += 1
        if truth == "uncertain":
            counts["uncertain_gt"] += 1
            if failed:
                counts["technical_errors_on_uncertain_gt"] += 1
            elif unsupported:
                counts["not_validated_on_uncertain_gt"] += 1
            elif pred == "uncertain":
                counts["abstained_on_uncertain_gt"] += 1
            else:
                counts["definite_predictions_on_uncertain_gt"] += 1
        else:
            counts["known_gt"] += 1
            if failed:
                counts["technical_errors_on_known"] += 1
            elif unsupported:
                counts["not_validated_on_known"] += 1
            elif pred == "uncertain":
                counts["abstained_on_known"] += 1
            elif pred == truth:
                counts["correct_known"] += 1
            else:
                counts["incorrect_definite"] += 1
    result = {key: counts[key] for key in (
        "total", "known_gt", "correct_known", "incorrect_definite", "abstained_on_known",
        "technical_errors_on_known", "uncertain_gt", "definite_predictions_on_uncertain_gt",
        "abstained_on_uncertain_gt", "technical_errors_on_uncertain_gt",
        "not_validated_on_known", "not_validated_on_uncertain_gt",
    )}
    definite = counts["correct_known"] + counts["incorrect_definite"]
    result["definite_coverage"] = definite / counts["known_gt"] if counts["known_gt"] else None
    result["accuracy_on_definite_known"] = counts["correct_known"] / definite if definite else None
    result["truth_counts"] = dict(truth_counts)
    result["confusion"] = confusion
    return result


def aggregate_metrics(rows, features=None):
    features = (tuple(feature for feature in FEATURES
                      if any(feature in row["truth"] for row in rows))
                if features is None else tuple(features))
    return {
        "overall": {feature: feature_metrics(rows, feature) for feature in features},
        "by_split": {
            split: {feature: feature_metrics([row for row in rows if row["split"] == split], feature)
                    for feature in features}
            for split in SPLITS
        },
    }


def _truth_values(label):
    truth = {}
    for feature, keys in TRUTH_KEYS.items():
        present = [key for key in keys if key in label]
        if not present:
            continue
        for key in present:
            if not isinstance(label[key], str) or label[key] not in FEATURE_STATES[feature]:
                raise ValueError(f"Invalid {key} ground truth: {label[key]!r}")
        if any(label[key] != label[present[0]] for key in present[1:]):
            raise ValueError(f"Conflicting ground truth aliases for {feature}")
        truth[feature] = label[present[0]]
    if not truth:
        raise ValueError("At least one supported ground truth field is required per frame")
    return truth


def _validate_labels(document):
    if not isinstance(document, dict) or not isinstance(document.get("frames"), list):
        raise ValueError("Labels must be an object with a frames array")
    videos, hashes = {}, {}
    for index, row in enumerate(document["frames"]):
        if not isinstance(row, dict) or row.get("split") not in SPLITS:
            raise ValueError(f"Frame {index}: split must be develop or reserve")
        if not isinstance(row.get("image_path"), str) or not row["image_path"].strip():
            raise ValueError(f"Frame {index}: image_path is required")
        timestamp = row.get("timestamp_sec")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError(f"Frame {index}: invalid timestamp_sec")
        try:
            _truth_values(row)
        except ValueError as error:
            raise ValueError(f"Frame {index}: {error}") from error
        video_id = row.get("video_id")
        if isinstance(video_id, bool) or not isinstance(video_id, (int, str)):
            raise ValueError(f"Frame {index}: video_id must be a string or integer")
        video_key = str(video_id)
        if video_key in videos and videos[video_key] != row["split"]:
            raise ValueError("One video cannot occur in both evaluation splits")
        videos[video_key] = row["split"]
        digest = row.get("image_sha256")
        if digest is not None:
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest.lower())):
                raise ValueError(f"Frame {index}: invalid image_sha256")
            digest = digest.lower()
            if digest in hashes and hashes[digest] != row["split"]:
                raise ValueError("An identical image cannot occur in both evaluation splits")
            hashes[digest] = row["split"]


def _load_frame(label, root):
    path = (root / label["image_path"]).resolve()
    image_bytes = path.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    if label.get("image_sha256") and digest != label["image_sha256"].lower():
        raise ValueError("Image SHA256 does not match the fixed annotation")
    with Image.open(BytesIO(image_bytes)) as image:
        if image.format != "JPEG":
            raise ValueError("Evaluation requires the actual annotated JPEG")
        image.load()
        width, height = image.size
    return VideoFrame(label["timestamp_sec"], image_bytes, width, height), digest


def _error(feature, error):
    return {"feature": feature, "type": type(error).__name__, "message": str(error)}


def evaluate(labels_path, out, *, root=PROJECT_ROOT, base_url="http://127.0.0.1:8080",
             split="all", model_label, compare_full_frame=False, request_timeout_sec=30.0,
             cabinet_mode="context_crops", activity_base_url=None):
    """Write a new, locally ignored report; perform model calls only when invoked."""
    if split not in ("all", *SPLITS):
        raise ValueError("split must be all, develop, or reserve")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("An explicit nonempty model_label is required")
    if (isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float))
            or not math.isfinite(request_timeout_sec) or request_timeout_sec <= 0):
        raise ValueError("request_timeout_sec must be finite and positive")
    # Constructor enforces localhost before any frame is read or output created.
    client = LocalModelClient(base_url, timeout=request_timeout_sec)
    activity_client = (None if activity_base_url is None else
                       LocalModelClient(activity_base_url, timeout=request_timeout_sec))
    observer = ImageObserver(client, cabinet_mode=cabinet_mode, activity_client=activity_client)
    root = Path(root).expanduser().resolve()
    labels_path = Path(labels_path).expanduser().resolve()
    label_bytes = labels_path.read_bytes()
    document = json.loads(label_bytes)
    _validate_labels(document)
    selected = [label for label in document["frames"] if split == "all" or label["split"] == split]
    if not selected:
        raise ValueError("No labelled frames in the requested split")
    selected_truths = [_truth_values(label) for label in selected]
    evaluated_features = tuple(feature for feature in FEATURES
                               if any(feature in truth for truth in selected_truths))
    if activity_client is None and any(feature in CASH_FEATURES for feature in evaluated_features):
        raise ValueError("Cash ground truth requires activity_base_url (--activity-base-url)")
    feature_support = {feature: dict(support) for feature, support in FEATURE_SUPPORT.items()
                       if activity_client is not None or feature not in CASH_FEATURES}
    destination = Path(out).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    # Reports contain dataset-derived labels and must stay out of Git even when
    # the caller chooses a directory outside the repository's ignored runs/.
    (destination / ".gitignore").write_text("*\n", encoding="utf-8")
    started = time.perf_counter()
    report = {
        "schema_version": 1, "kind": "visual_component_evaluation_not_violation_accuracy",
        "status": "running", "model_label": model_label,
        "policy_sha256": policy_fingerprint(cabinet_mode, cash_enabled=activity_client is not None),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "labels_path": str(labels_path), "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "source_root": str(root), "base_url": base_url, "activity_base_url": activity_base_url, "split": split,
        "request_timeout_sec": request_timeout_sec, "compare_full_frame": bool(compare_full_frame),
        "settings": {"cabinet_mode": cabinet_mode, "cash_enabled": activity_client is not None},
        "feature_support": feature_support,
        "all_videos_previously_reviewed": document.get("all_videos_previously_reviewed") is True,
        "independent_test": document.get("independent_test") is True,
        "selected_frame_count": len(selected), "evaluated_features": list(evaluated_features), "rows": [],
        "metric_definitions": {
            "truth_keys": {feature: feature for feature in evaluated_features},
            "accepted_truth_keys": {feature: list(TRUTH_KEYS[feature]) for feature in evaluated_features},
            "missing_truth": "Unlabelled fields are excluded per feature, never inferred as no or uncertain.",
            "not_validated": "Disabled, unvalidated features are counted separately from model abstention and technical errors.",
            "known_gt": "Valid truth labels other than uncertain; not_visible is a cabinet visibility class.",
            "definite_coverage": "(correct_known + incorrect_definite) / known_gt; errors remain in denominator.",
            "accuracy_on_definite_known": "correct_known / (correct_known + incorrect_definite).",
            "undefined_fraction": "null when denominator is zero",
            "uncertain_gt": "Reported separately; definite predictions on uncertain truth are not automatically recognition errors.",
        },
        "limitations": [
            "Model label is supplied by the caller, not verification of loaded weights or server settings.",
            "Review and independence flags are copied from labels, not independently verified.",
            "Frame observations do not measure violation detection or temporal event accuracy.",
            "Frames from one video are dependent; class counts and split results must be reported.",
            "Two contextual crops do not constitute independent evidence or calibrated confidence.",
        ],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["processed_frame_count"] = len(report["rows"])
        report["metrics"] = aggregate_metrics(report["rows"], evaluated_features)
        report["request_count"] = sum(len((row["response"] or {}).get("calls", []))
                                      + int(row.get("full_frame_cabinet", {}).get("attempted", False))
                                      for row in report["rows"])
        if compare_full_frame:
            baseline_rows = [{"truth": row["truth"], "split": row["split"],
                              "pred": {"cabinet_door": row["full_frame_cabinet"]["pred"]},
                              "errors": row["full_frame_cabinet"]["errors"]}
                             for row in report["rows"]]
            report["full_frame_cabinet_metrics"] = aggregate_metrics(baseline_rows, ("cabinet_door",))
        temp = destination / "report.json.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(destination / "report.json")

    save()
    try:
        for index, (label, truth) in enumerate(zip(selected, selected_truths)):
            row = {
                "frame_id": f"eval_{index:04d}", "image_path": label["image_path"],
                "video_id": label["video_id"], "timestamp_sec": label["timestamp_sec"], "split": label["split"],
                "truth": truth,
                "feature_support": feature_support,
                "pred": {feature: None for feature in evaluated_features}, "response": None, "errors": [],
            }
            frame = None
            try:
                frame, row["image_sha256"] = _load_frame(label, root)
                # Identifiers here are generated counters; label text, paths,
                # rationales and video IDs never enter the observer's prompts.
                row["response"] = observer.observe(frame, row["frame_id"])
            except ObservationDeadline as error:
                row["response"] = error.partial_record
                if row["response"] is None:
                    row["errors"].append(_error("all", error))
            except Exception as error:
                row["errors"].append(_error("all", error))
            if row["response"] is not None:
                row["pred"].update(row["response"]["states"])
                row["errors"].extend(row["response"].get("errors", []))
                questions = {call.get("question") for call in row["response"].get("calls", [])}
                cabinet_question = "cabinet_door_full_frame" if cabinet_mode == "full_frame" else "cabinet_location"
                feature_questions = {
                    "person_in_work_area": "person" if activity_client is None else "person_position",
                    "cabinet_door": cabinet_question,
                    "cash_on_worktop": "desk_location",
                }
                for feature, question in feature_questions.items():
                    if feature not in row["pred"]:
                        continue
                    if question not in questions:
                        # A timeout before a question is not a model abstention.
                        row["pred"][feature] = None
            if compare_full_frame:
                baseline = {"pred": None, "response": None, "errors": [], "attempted": False}
                row["full_frame_cabinet"] = baseline
                if frame is None:
                    baseline["errors"] = list(row["errors"])
                elif "cabinet_door" in truth:
                    try:
                        baseline["attempted"] = True
                        baseline["response"] = client.ask_image(frame, DOOR_PROMPT, DOOR_SCHEMA)
                        baseline["pred"] = read_enum(baseline["response"], "door_state", FEATURES["cabinet_door"][1])
                    except Exception as error:
                        baseline["errors"].append(_error("cabinet_door", error))
            errors = row["errors"] or row.get("full_frame_cabinet", {}).get("errors", [])
            row["status"] = "partial" if errors and row["response"] is not None else "error" if errors else "ok"
            report["rows"].append(row)
            save()
        report["status"] = "partial" if any(row["status"] != "ok" for row in report["rows"]) else "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Base directory for image_path values")
    parser.add_argument("--out", type=Path, required=True, help="New report directory; files are locally Git-ignored")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--activity-base-url", help="Optional localhost model for person position and cash observations")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--model-label", required=True, help="Explicit model/runtime configuration label for this run")
    parser.add_argument("--compare-full-frame", action="store_true")
    parser.add_argument("--cabinet-mode", choices=("context_crops", "full_frame"), default="context_crops")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()
    report = evaluate(args.labels, args.out, root=args.root, base_url=args.base_url, split=args.split,
                      model_label=args.model_label, compare_full_frame=args.compare_full_frame,
                      request_timeout_sec=args.request_timeout, cabinet_mode=args.cabinet_mode,
                      activity_base_url=args.activity_base_url)
    print(json.dumps({"report": str(args.out.resolve() / "report.json"), "status": report["status"],
                      "processed_frames": report["processed_frame_count"], "metrics": report["metrics"]},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
