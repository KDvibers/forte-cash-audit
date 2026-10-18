"""Compare two frozen local detector evaluations without selecting a winner.

Recompute metrics from predictions against one shared test manifest. This reads
JSON only: no model, weight, image, video, or network calls are made. Source-video
split checks use the hashes recorded in the independently verified manifests.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re

from analyzer.bundle_metrics import evaluate_detections


_HASH = re.compile(r"[0-9a-f]{64}")
_SELECTION = "best validation COCO mAP; test is not run during training"
_THRESHOLDS = {"score_threshold": .5, "iou_threshold": .5, "ignore_overlap": .5}
_FIXED_SETTINGS = {"epochs": 30, "batch_size": 1, "grad_accum_steps": 2, "num_workers": 0,
                   "multi_scale": False, "expanded_scales": False, "use_ema": False,
                   "early_stopping": False, "run_test": False, "tensorboard": False,
                   "wandb": False, "lr": 1e-4, "lr_encoder": 1.5e-4}
_DELTAS = ("tp", "fp", "fn", "precision", "recall")
_EXPANDED_PROTOCOL = "expanded-validation-resolution-v3"
_PROTOCOLS = ("strict", _EXPANDED_PROTOCOL)


def _hash(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Reports cannot contain nonfinite numbers")
    if isinstance(value, Mapping):
        for member in value.values():
            _finite(member)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _finite(member)


def _read(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    value = json.loads(data)
    _finite(value)
    return value, hashlib.sha256(data).hexdigest()


def _manifest_rows(manifest):
    if not isinstance(manifest, Mapping) or manifest.get("kind") != "local_cash_detector_experiment":
        raise ValueError("Expected a bundle_dataset manifest")
    rows = manifest.get("rows")
    splits = manifest.get("split_by_video")
    if not isinstance(rows, list) or not rows or not isinstance(splits, dict):
        raise ValueError("Manifest requires nonempty rows and a split plan")
    seen, video_splits, video_ids, image_splits = set(), {}, {}, {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Manifest row must be an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Manifest row ids must be unique nonempty strings")
        seen.add(identifier)
        split = row.get("split")
        if split not in {"train", "valid", "test"} or splits.get(str(row.get("video_id"))) != split:
            raise ValueError("Row split differs from the declared video split")
        sha = _hash(row.get("source_video_sha256"), "source_video_sha256")
        if video_splits.setdefault(sha, split) != split:
            raise ValueError("Duplicate source video crosses splits")
        if video_ids.setdefault(str(row.get("video_id")), sha) != sha:
            raise ValueError("A video id refers to different source hashes")
        for key in ("image_sha256", "export_image_sha256"):
            digest = _hash(row.get(key), key)
            if image_splits.setdefault((key, digest), split) != split:
                raise ValueError("Duplicate image crosses splits")
        tags = row.get("frame_tags", [])
        if (not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags)
                or len(set(tags)) != len(tags)):
            raise ValueError("frame_tags must be a list of unique nonempty strings")
    return rows


def _split_provenance(rows, split):
    """Fingerprint the images, crops and labels that affected training/selection."""
    fields = ("source_video_sha256", "image_sha256", "export_image_sha256", "roi_xyxy",
              "boxes_xyxy", "ignore_boxes_xyxy", "box_contexts")
    selected = [row for row in rows if row["split"] == split]
    signatures = sorted(json.dumps({key: row[key] for key in fields}, sort_keys=True)
                        for row in selected)
    digest = hashlib.sha256(json.dumps(signatures).encode("utf-8")).hexdigest()
    return {"annotations_sha256": digest, "frames": len(selected),
            "ground_truth": sum(len(row["boxes_xyxy"]) for row in selected),
            "source_video_sha256": sorted({row["source_video_sha256"] for row in selected})}


def _experiment(path, metrics, test_rows, *, protocol="strict"):
    experiment, digest = _read(path)
    if (experiment.get("status") != "completed"
            or experiment.get("checkpoint_selection") != _SELECTION
            or experiment.get("selected_checkpoint") != "checkpoint_best_total.pth"):
        raise ValueError("Require a completed training run with the predeclared validation checkpoint")
    if _hash(experiment.get("selected_checkpoint_sha256"), "checkpoint SHA256") != metrics["weights_sha256"]:
        raise ValueError("Evaluation weights differ from the validation-selected checkpoint")
    settings = experiment.get("settings", {})
    for name, expected in _FIXED_SETTINGS.items():
        if name not in settings or type(settings[name]) is not type(expected) or settings[name] != expected:
            raise ValueError(f"Training setting {name} differs from the fixed experiment")
    resolution = experiment.get("resolution")
    allowed_resolutions = {384, 512} if protocol == _EXPANDED_PROTOCOL else {384}
    if (experiment.get("model") != "RFDETRNano" or type(resolution) is not int
            or resolution not in allowed_resolutions or experiment.get("seed") != 42):
        raise ValueError("Model, resolution or seed differs from the fixed experiment")
    if protocol == _EXPANDED_PROTOCOL and (
            type(metrics.get("resolution")) is not int or metrics["resolution"] != resolution):
        raise ValueError("Evaluation resolution must be recorded and match its training run")
    for key in ("score_threshold", "iou_threshold"):
        if experiment.get(key) != _THRESHOLDS[key]:
            raise ValueError("Training report thresholds differ from the fixed experiment")
    _hash(experiment.get("initial_weights_sha256"), "initial_weights_sha256")
    dataset_dir = settings.get("dataset_dir")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise ValueError("Training experiment requires dataset_dir")
    training_manifest, training_digest = _read(Path(dataset_dir) / "manifest.json")
    if experiment.get("dataset_manifest_sha256") != training_digest:
        raise ValueError("Training dataset manifest hash mismatch")
    training_rows = _manifest_rows(training_manifest)
    if experiment.get("split_by_video") != training_manifest["split_by_video"]:
        raise ValueError("Training report split plan differs from its manifest")
    exposed = [row for row in training_rows if row["split"] in {"train", "valid"}]
    if not any(row["split"] == "train" for row in exposed) or not any(row["split"] == "valid" for row in exposed):
        raise ValueError("Training provenance requires both train and validation rows")
    for key in ("source_video_sha256", "image_sha256", "export_image_sha256"):
        if {row[key] for row in exposed} & {row[key] for row in test_rows}:
            raise ValueError(f"Current test overlaps training or validation by {key}")
    provenance = {split: _split_provenance(exposed, split) for split in ("train", "valid")}
    return experiment, digest, provenance


def _validated_metrics(directory, manifest_digest, rows):
    directory = Path(directory)
    report, report_digest = _read(directory / "metrics.json")
    predictions, predictions_digest = _read(directory / "predictions.json")
    if report.get("split") != "test":
        raise ValueError("Only the frozen test split may be compared")
    if report.get("dataset_manifest_sha256") != manifest_digest:
        raise ValueError("Evaluation dataset manifest hash mismatch")
    _hash(report.get("weights_sha256"), "weights_sha256")
    for key, expected in _THRESHOLDS.items():
        if isinstance(report.get(key), bool) or report.get(key) != expected:
            raise ValueError("Evaluation thresholds differ from the fixed experiment")
    computed = evaluate_detections(rows, predictions, **_THRESHOLDS)
    for key, value in computed.items():
        declared = report.get(key)
        if key == "rows" and isinstance(declared, list):
            declared = sorted(declared, key=lambda row: row.get("id", ""))
            value = sorted(value, key=lambda row: row["id"])
        if declared != value:
            raise ValueError(f"Declared metrics disagree with recomputation: {key}")
    return report, {"metrics_sha256": report_digest, "predictions_sha256": predictions_digest}


def _delta(old, new, keys=_DELTAS):
    return {key: new[key] - old[key] if new.get(key) is not None and old.get(key) is not None else None
            for key in keys}


def _comparison(old, new, keys=_DELTAS):
    return {"baseline": old, "candidate": new, "delta_candidate_minus_baseline": _delta(old, new, keys)}


def _frame_group(ids, old_rows, new_rows):
    def counts(rows):
        selected = [rows[identifier] for identifier in ids]
        return {"frames": len(selected), "tp": sum(row["tp"] for row in selected),
                "fp": sum(row["fp"] for row in selected), "fn": sum(row["fn"] for row in selected),
                "frames_with_false_positives": sum(row["fp"] > 0 for row in selected)}
    return {"frame_ids": sorted(ids), **_comparison(counts(old_rows), counts(new_rows),
            ("tp", "fp", "fn", "frames_with_false_positives"))}


def compare_evaluations(manifest_path, baseline_dir, candidate_dir, baseline_experiment, candidate_experiment,
                        *, protocol="strict"):
    """Read paired evaluations, verify their provenance, and report diagnostic deltas.

    Checkpoint choice must already have been made by validation. A passed check
    here does not select a model, approve deployment or validate action direction.
    The input JSONs and model weights are never modified. The explicit expanded
    protocol permits different validation labels and 384/512 input resolution;
    it compares complete experimental protocols, not the isolated effect of
    either change. All other fixed settings and test-leak checks still apply.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError("Unknown comparison protocol")
    manifest, manifest_digest = _read(manifest_path)
    rows = [row for row in _manifest_rows(manifest) if row["split"] == "test"]
    if not rows:
        raise ValueError("The common manifest has no test rows")
    old, old_hashes = _validated_metrics(baseline_dir, manifest_digest, rows)
    new, new_hashes = _validated_metrics(candidate_dir, manifest_digest, rows)
    old_experiment, old_experiment_hash, old_provenance = _experiment(
        baseline_experiment, old, rows, protocol=protocol)
    new_experiment, new_experiment_hash, new_provenance = _experiment(
        candidate_experiment, new, rows, protocol=protocol)
    same_validation = old_provenance["valid"] == new_provenance["valid"]
    if not same_validation and protocol == "strict":
        raise ValueError("Validation images, ROI or annotations differ between training runs")
    if old["weights_sha256"] == new["weights_sha256"]:
        raise ValueError("Baseline and candidate refer to the same checkpoint")
    for key in ("initial_weights_sha256", "model", "resolution", "seed", "packages", "torch_version"):
        if key == "resolution" and protocol == _EXPANDED_PROTOCOL:
            continue
        if old_experiment.get(key) != new_experiment.get(key):
            raise ValueError(f"Training runs differ in {key}")
    def comparable_settings(experiment):
        return {key: value for key, value in experiment["settings"].items() if key not in {"dataset_dir", "output_dir"}}
    if comparable_settings(old_experiment) != comparable_settings(new_experiment):
        raise ValueError("Training hyperparameters differ")
    differences, confounds = {}, []
    for name, split, explanation in (
        ("training_annotations", "train", "Different training images or labels can change detection quality."),
        ("validation_annotations", "valid", "Different validation images or labels can change checkpoint selection."),
    ):
        if old_provenance[split] != new_provenance[split]:
            differences[name] = {"baseline": old_provenance[split], "candidate": new_provenance[split]}
            confounds.append(explanation)
    if old_experiment["resolution"] != new_experiment["resolution"]:
        differences["resolution"] = {"baseline": old_experiment["resolution"],
                                     "candidate": new_experiment["resolution"]}
        confounds.append("Different model input resolution can affect detections, memory and runtime.")
    old_rows = {row["id"]: row for row in old["rows"]}
    new_rows = {row["id"]: row for row in new["rows"]}
    tags = sorted({tag for row in rows for tag in row.get("frame_tags", [])})
    by_tag = {tag: _frame_group([row["id"] for row in rows if tag in row.get("frame_tags", [])], old_rows, new_rows)
              for tag in tags}
    negatives = _frame_group([row["id"] for row in rows if not row["boxes_xyxy"] and not row["ignore_boxes_xyxy"]],
                             old_rows, new_rows)
    cylinder = by_tag.get("cylinder_visible")
    checks = {
        "candidate_fp_not_greater": new["totals"]["fp"] <= old["totals"]["fp"],
        "candidate_fn_not_greater": new["totals"]["fn"] <= old["totals"]["fn"],
        "candidate_no_fp_on_clean_negative_frames": negatives["candidate"]["fp"] == 0 if negatives["frame_ids"] else None,
        "candidate_no_fp_on_cylinder_visible_frames": cylinder["candidate"]["fp"] == 0 if cylinder else None,
        "candidate_all_hand_objects_found": new["by_context"]["in_hand"]["fn"] == 0 if new["by_context"]["in_hand"]["ground_truth"] else None,
        "candidate_all_tray_objects_found": new["by_context"]["in_tray"]["fn"] == 0 if new["by_context"]["in_tray"]["ground_truth"] else None,
    }
    preregistered = {
        "candidate_hand_recall_not_lower": new["by_context"]["in_hand"]["tp"] >= old["by_context"]["in_hand"]["tp"]
            if old["by_context"]["in_hand"]["ground_truth"] else None,
        "recovered_missed_hand_object_if_any": new["by_context"]["in_hand"]["tp"] > old["by_context"]["in_hand"]["tp"]
            if old["by_context"]["in_hand"]["fn"] else None,
        "fewer_fp_on_cylinder_frames_if_any": cylinder["candidate"]["fp"] < cylinder["baseline"]["fp"]
            if cylinder and cylinder["baseline"]["fp"] else None,
        "candidate_tp_not_lower": new["totals"]["tp"] >= old["totals"]["tp"],
        "candidate_fp_not_greater": checks["candidate_fp_not_greater"],
    }
    return {"schema_version": 1, "kind": "paired_local_detector_diagnostics", "split": "test", **_THRESHOLDS,
            "dataset_manifest_sha256": manifest_digest,
            "baseline": {"weights_sha256": old["weights_sha256"], "experiment_sha256": old_experiment_hash, **old_hashes},
            "candidate": {"weights_sha256": new["weights_sha256"], "experiment_sha256": new_experiment_hash, **new_hashes},
            "checkpoint_selection": _SELECTION, "same_validation_annotations": same_validation,
            "protocol": protocol,
            "allowed_protocol_differences": ["validation_annotations", "resolution"]
                if protocol == _EXPANDED_PROTOCOL else [],
            "training_annotations_may_differ_in_both_protocols": True,
            "observed_differences": differences, "confounds": confounds,
            "training_provenance": {
                "baseline": {**old_provenance, "resolution": old_experiment["resolution"]},
                "candidate": {**new_provenance, "resolution": new_experiment["resolution"]}},
            "initial_weights_sha256": old_experiment["initial_weights_sha256"],
            "totals": _comparison(old["totals"], new["totals"]),
            "by_context": {context: _comparison(old["by_context"][context], new["by_context"][context], ("tp", "fn", "recall"))
                           for context in old["by_context"]},
            "by_video": {video: _comparison(old["by_video"][video], new["by_video"][video]) for video in old["by_video"]},
            "by_frame_tag": by_tag, "clean_negative_frames": negatives,
            "preregistered_diagnostic_checks": preregistered, "diagnostic_checks": checks,
            "winner_selected": False, "eligible_for_production": False, "independent_jury_test": False,
            "limits": ["Source overlap checks use recorded hashes from the verified manifests; source bytes are not reread here.",
                       "False positives on cylinder-visible frames are not automatically false positives on cylinders.",
                       "When training data, validation data or resolution differ, deltas do not isolate the cause of a change.",
                       "Sparse manual-ROI evaluation does not establish dense-video, action or violation accuracy.",
                       "These checks do not select a checkpoint or authorize production promotion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-experiment", type=Path, required=True)
    parser.add_argument("--candidate-experiment", type=Path, required=True)
    parser.add_argument("--protocol", choices=_PROTOCOLS, default="strict")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare_evaluations(args.manifest, args.baseline, args.candidate,
                                 args.baseline_experiment, args.candidate_experiment, protocol=args.protocol)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"totals": report["totals"], "preregistered_diagnostic_checks": report["preregistered_diagnostic_checks"],
                      "additional_diagnostic_checks": report["diagnostic_checks"]}, allow_nan=False))


if __name__ == "__main__":
    main()
