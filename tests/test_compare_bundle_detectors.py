"""Paired comparisons reject stale labels, incomplete inference and split leaks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyzer.bundle_metrics import evaluate_detections
from analyzer.compare_bundle_detectors import compare_evaluations


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def row(identifier, video, split="test", boxes=None, tags=None):
    boxes = [[10, 10, 20, 20]] if boxes is None else boxes
    return {"id": identifier, "video_id": video, "split": split, "roi_xyxy": [0, 0, 100, 100],
            "boxes_xyxy": boxes, "ignore_boxes_xyxy": [], "fully_annotated": True,
            "box_contexts": ["in_hand"] * len(boxes), "frame_tags": [] if tags is None else tags,
            "source_video_sha256": digest(f"video{video}"), "image_sha256": digest(f"image{identifier}"),
            "export_image_sha256": digest(f"crop{identifier}")}


def manifest(rows):
    return {"kind": "local_cash_detector_experiment", "rows": rows,
            "split_by_video": {str(r["video_id"]): r["split"] for r in rows}}


class CompareBundleDetectorsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.test = manifest([row("a", 101), row("b", 102, boxes=[], tags=["cylinder_visible"])])
        self.manifest_path = self.root / "manifest.json"
        self.write(self.manifest_path, self.test)
        self.dirs = [self.root / "old", self.root / "new"]
        self.exps = []
        self.predictions = [
            [{"id": "a", "boxes_xyxy": [], "scores": []}, {"id": "b", "boxes_xyxy": [[30, 30, 40, 40]], "scores": [.8]}],
            [{"id": "a", "boxes_xyxy": [[10, 10, 20, 20]], "scores": [.9]}, {"id": "b", "boxes_xyxy": [], "scores": []}],
        ]
        for i, directory in enumerate(self.dirs):
            directory.mkdir()
            dataset = directory / "dataset"
            dataset.mkdir()
            training = manifest([row(f"train{i}", i + 1, "train"), row("valid", 50, "valid")])
            self.write(dataset / "manifest.json", training)
            experiment = {"status": "completed", "model": "RFDETRNano", "resolution": 384, "seed": 42,
                          "checkpoint_selection": "best validation COCO mAP; test is not run during training",
                          "selected_checkpoint": "checkpoint_best_total.pth", "selected_checkpoint_sha256": digest(f"weights{i}"),
                          "initial_weights_sha256": digest("official initial"), "score_threshold": .5, "iou_threshold": .5,
                          "dataset_manifest_sha256": self.sha(dataset / "manifest.json"), "split_by_video": training["split_by_video"],
                          "settings": {"dataset_dir": str(dataset), "output_dir": str(directory), "epochs": 30,
                              "batch_size": 1, "grad_accum_steps": 2, "num_workers": 0, "multi_scale": False,
                              "expanded_scales": False, "use_ema": False, "early_stopping": False, "run_test": False,
                              "tensorboard": False, "wandb": False, "lr": 1e-4, "lr_encoder": 1.5e-4}}
            path = directory / "experiment.json"
            self.write(path, experiment)
            self.exps.append(path)
        self.evaluations()

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def evaluations(self):
        for i, directory in enumerate(self.dirs):
            report = evaluate_detections(self.test["rows"], self.predictions[i])
            report.update(split="test", weights_sha256=digest(f"weights{i}"),
                          dataset_manifest_sha256=self.sha(self.manifest_path))
            self.write(directory / "predictions.json", self.predictions[i])
            self.write(directory / "metrics.json", report)

    def compare(self, **kwargs):
        return compare_evaluations(self.manifest_path, *self.dirs, *self.exps, **kwargs)

    def expanded_protocol_fixture(self):
        """New candidate selection data and explicit actual inference sizes."""
        experiment = self.read(self.exps[1])
        path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
        training = self.read(path)
        training["rows"].append(row("fresh-valid", 51, "valid"))
        training["split_by_video"]["51"] = "valid"
        self.write(path, training)
        experiment.update(resolution=512, dataset_manifest_sha256=self.sha(path),
                          split_by_video=training["split_by_video"])
        self.write(self.exps[1], experiment)
        for directory, resolution in zip(self.dirs, (384, 512)):
            metrics_path = directory / "metrics.json"
            metrics = self.read(metrics_path)
            metrics["resolution"] = resolution
            self.write(metrics_path, metrics)

    def test_same_frozen_set_reports_deltas_without_selecting_winner(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        result = self.compare()
        self.assertEqual(result["totals"]["delta_candidate_minus_baseline"]["tp"], 1)
        self.assertEqual(result["totals"]["delta_candidate_minus_baseline"]["fp"], -1)
        self.assertEqual(result["totals"]["delta_candidate_minus_baseline"]["fn"], -1)
        self.assertEqual(result["by_context"]["in_hand"]["delta_candidate_minus_baseline"]["recall"], 1)
        self.assertTrue(result["diagnostic_checks"]["candidate_no_fp_on_cylinder_visible_frames"])
        self.assertTrue(all(result["preregistered_diagnostic_checks"].values()))
        self.assertIsNone(result["diagnostic_checks"]["candidate_all_tray_objects_found"])
        self.assertFalse(result["winner_selected"])
        self.assertFalse(result["eligible_for_production"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.json")})

    def test_notes_do_not_create_cylinder_tags(self):
        self.test["rows"][1].update(frame_tags=[], notes="A cylinder is visible")
        self.write(self.manifest_path, self.test)
        self.evaluations()
        result = self.compare()
        self.assertIsNone(result["diagnostic_checks"]["candidate_no_fp_on_cylinder_visible_frames"])
        self.assertEqual(result["by_frame_tag"], {})

    def test_candidate_false_positive_on_tagged_frame_fails_limited_check(self):
        self.predictions[1][1] = {"id": "b", "boxes_xyxy": [[70, 70, 80, 80]], "scores": [.9]}
        self.evaluations()
        result = self.compare()
        self.assertFalse(result["diagnostic_checks"]["candidate_no_fp_on_cylinder_visible_frames"])
        self.assertEqual(result["by_frame_tag"]["cylinder_visible"]["candidate"]["frames_with_false_positives"], 1)

    def test_report_manifest_sha_split_and_thresholds_are_required(self):
        path = self.dirs[1] / "metrics.json"
        original = self.read(path)
        for key, value in (("dataset_manifest_sha256", digest("other")), ("split", "valid"),
                           ("score_threshold", .3), ("iou_threshold", .4), ("ignore_overlap", .2)):
            with self.subTest(key=key):
                self.write(path, {**original, key: value})
                with self.assertRaises(ValueError):
                    self.compare()
        self.write(path, original)

    def test_stale_gt_contexts_detected_even_when_report_sha_is_rewritten(self):
        self.test["rows"][0]["box_contexts"] = ["in_tray"]
        self.write(self.manifest_path, self.test)
        for directory in self.dirs:
            path = directory / "metrics.json"
            report = self.read(path)
            report["dataset_manifest_sha256"] = self.sha(self.manifest_path)
            self.write(path, report)
        with self.assertRaisesRegex(ValueError, "recomputation"):
            self.compare()

    def test_fabricated_metrics_and_nonfinite_metrics_rejected(self):
        path = self.dirs[0] / "metrics.json"
        original = self.read(path)
        for value in (999, float("nan"), float("inf")):
            report = deepcopy(original)
            report["totals"]["tp"] = value
            self.write(path, report)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare()

    def test_missing_prediction_and_failed_empty_prediction_rejected(self):
        path = self.dirs[0] / "predictions.json"
        for predictions in ([self.predictions[0][0]],
                            [{**self.predictions[0][0], "error": "OOM"}, self.predictions[0][1]]):
            self.write(path, predictions)
            with self.assertRaises(ValueError):
                self.compare()

    def test_wrong_checkpoint_or_test_selected_checkpoint_rejected(self):
        path = self.exps[1]
        original = self.read(path)
        for change in ({"selected_checkpoint_sha256": digest("wrong")},
                       {"checkpoint_selection": "best test precision"}, {"status": "failed"}):
            self.write(path, {**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare()

    def test_changed_initial_weights_or_training_settings_rejected(self):
        path = self.exps[1]
        original = self.read(path)
        variants = [dict(original, initial_weights_sha256=digest("different initial"))]
        changed = deepcopy(original)
        changed["settings"]["epochs"] = 31
        variants.append(changed)
        changed = deepcopy(original)
        changed["settings"]["run_test"] = True
        variants.append(changed)
        for experiment in variants:
            self.write(path, experiment)
            with self.assertRaises(ValueError):
                self.compare()

    def test_training_manifest_cannot_change_unnoticed(self):
        experiment = self.read(self.exps[1])
        path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
        training = self.read(path)
        training["rows"][0]["boxes_xyxy"] = []
        self.write(path, training)
        with self.assertRaisesRegex(ValueError, "Training dataset manifest hash"):
            self.compare()

    def test_exact_duplicate_of_new_test_in_either_train_or_valid_rejected(self):
        for model_index in (0, 1):
            for split_index in (0, 1):
                with self.subTest(model=model_index, split=split_index):
                    experiment = self.read(self.exps[model_index])
                    path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
                    original = self.read(path)
                    training = deepcopy(original)
                    training["rows"][split_index]["source_video_sha256"] = self.test["rows"][0]["source_video_sha256"]
                    self.write(path, training)
                    experiment["dataset_manifest_sha256"] = self.sha(path)
                    self.write(self.exps[model_index], experiment)
                    with self.assertRaisesRegex(ValueError, "test overlaps"):
                        self.compare()
                    self.write(path, original)
                    experiment["dataset_manifest_sha256"] = self.sha(path)
                    self.write(self.exps[model_index], experiment)

    def test_duplicate_image_with_distinct_video_hash_is_also_leak(self):
        experiment = self.read(self.exps[0])
        path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
        training = self.read(path)
        training["rows"][0]["image_sha256"] = self.test["rows"][0]["image_sha256"]
        self.write(path, training)
        experiment["dataset_manifest_sha256"] = self.sha(path)
        self.write(self.exps[0], experiment)
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            self.compare()

    def test_validation_gt_cannot_change_between_checkpoint_selections(self):
        experiment = self.read(self.exps[1])
        path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
        training = self.read(path)
        training["rows"][1]["boxes_xyxy"] = [[10, 10, 30, 30]]
        self.write(path, training)
        experiment["dataset_manifest_sha256"] = self.sha(path)
        self.write(self.exps[1], experiment)
        with self.assertRaisesRegex(ValueError, "Validation images"):
            self.compare()

    def test_paired_predictions_can_arrive_in_different_frame_order(self):
        self.predictions[1].reverse()
        self.evaluations()
        self.assertEqual(self.compare()["totals"]["candidate"]["tp"], 1)

    def test_recovery_and_cylinder_improvement_are_null_when_baseline_has_no_errors(self):
        self.predictions[0] = deepcopy(self.predictions[1])
        self.evaluations()
        checks = self.compare()["preregistered_diagnostic_checks"]
        self.assertTrue(checks["candidate_hand_recall_not_lower"])
        self.assertIsNone(checks["recovered_missed_hand_object_if_any"])
        self.assertIsNone(checks["fewer_fp_on_cylinder_frames_if_any"])

    def test_reducing_but_not_eliminating_cylinder_fp_meets_only_preregistered_improvement(self):
        self.predictions[0][1]["boxes_xyxy"].append([60, 60, 70, 70])
        self.predictions[0][1]["scores"].append(.8)
        self.predictions[1][1] = {"id": "b", "boxes_xyxy": [[30, 30, 40, 40]], "scores": [.8]}
        self.evaluations()
        result = self.compare()
        self.assertTrue(result["preregistered_diagnostic_checks"]["fewer_fp_on_cylinder_frames_if_any"])
        self.assertFalse(result["diagnostic_checks"]["candidate_no_fp_on_cylinder_visible_frames"])

    def test_expanded_protocol_reports_actual_differences_without_changing_inputs(self):
        self.expanded_protocol_fixture()
        before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        result = self.compare(protocol="expanded-validation-resolution-v3")
        self.assertFalse(result["same_validation_annotations"])
        self.assertEqual(result["allowed_protocol_differences"], ["validation_annotations", "resolution"])
        self.assertEqual(set(result["observed_differences"]),
                         {"training_annotations", "validation_annotations", "resolution"})
        validation = result["observed_differences"]["validation_annotations"]
        self.assertEqual(validation["baseline"]["frames"], 1)
        self.assertEqual(validation["candidate"]["frames"], 2)
        self.assertNotEqual(validation["baseline"]["annotations_sha256"],
                            validation["candidate"]["annotations_sha256"])
        self.assertEqual(result["observed_differences"]["resolution"], {"baseline": 384, "candidate": 512})
        self.assertEqual(len(result["confounds"]), 3)
        self.assertEqual(result["totals"]["candidate"]["tp"], 1)
        self.assertFalse(result["winner_selected"])
        self.assertFalse(result["eligible_for_production"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.json")})
        with self.assertRaises(ValueError):
            self.compare()  # Expanded behavior cannot silently become default.

    def test_expanded_protocol_does_not_invent_validation_or_resolution_changes(self):
        for directory in self.dirs:
            path = directory / "metrics.json"
            self.write(path, {**self.read(path), "resolution": 384})
        result = self.compare(protocol="expanded-validation-resolution-v3")
        self.assertTrue(result["same_validation_annotations"])
        self.assertEqual(set(result["observed_differences"]), {"training_annotations"})

    def test_expanded_protocol_requires_recorded_matching_bounded_resolution(self):
        self.expanded_protocol_fixture()
        path = self.dirs[1] / "metrics.json"
        original = self.read(path)
        for value in (None, 384, 512.0, True, 1024):
            with self.subTest(value=value):
                metrics = dict(original)
                if value is None:
                    del metrics["resolution"]
                else:
                    metrics["resolution"] = value
                self.write(path, metrics)
                with self.assertRaisesRegex(ValueError, "Evaluation resolution"):
                    self.compare(protocol="expanded-validation-resolution-v3")
        self.write(path, {**original, "resolution": 1024})
        experiment = self.read(self.exps[1])
        self.write(self.exps[1], {**experiment, "resolution": 1024})
        with self.assertRaisesRegex(ValueError, "Model, resolution or seed"):
            self.compare(protocol="expanded-validation-resolution-v3")

    def test_expanded_protocol_keeps_other_training_and_checkpoint_constraints(self):
        self.expanded_protocol_fixture()
        original = self.read(self.exps[1])
        variants = [{**original, key: value} for key, value in (
            ("initial_weights_sha256", digest("different initial")), ("seed", 43),
            ("model", "RFDETRSmall"), ("packages", {"rfdetr": "different"}),
            ("torch_version", "different"), ("selected_checkpoint_sha256", digest("other checkpoint")),
            ("checkpoint_selection", "best test precision"), ("status", "failed"))]
        for name, value in (("epochs", 31), ("lr", 2e-4), ("run_test", True), ("new_setting", True)):
            variant = deepcopy(original)
            variant["settings"][name] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(variant=variant):
                self.write(self.exps[1], variant)
                with self.assertRaises(ValueError):
                    self.compare(protocol="expanded-validation-resolution-v3")

    def test_expanded_validation_cannot_contain_fresh_test_video_or_image(self):
        self.expanded_protocol_fixture()
        for model_index in (0, 1):
            experiment = self.read(self.exps[model_index])
            path = Path(experiment["settings"]["dataset_dir"]) / "manifest.json"
            original = self.read(path)
            for key in ("source_video_sha256", "image_sha256", "export_image_sha256"):
                with self.subTest(model=model_index, key=key):
                    training = deepcopy(original)
                    training["rows"][-1][key] = self.test["rows"][0][key]
                    self.write(path, training)
                    self.write(self.exps[model_index], {**experiment, "dataset_manifest_sha256": self.sha(path)})
                    with self.assertRaisesRegex(ValueError, "Current test overlaps"):
                        self.compare(protocol="expanded-validation-resolution-v3")
            self.write(path, original)
            self.write(self.exps[model_index], experiment)

    def test_expanded_protocol_still_rejects_incomplete_predictions_and_changed_threshold(self):
        self.expanded_protocol_fixture()
        path = self.dirs[1] / "predictions.json"
        original = self.read(path)
        self.write(path, original[:-1])
        with self.assertRaises(ValueError):
            self.compare(protocol="expanded-validation-resolution-v3")
        self.write(path, original)
        path = self.dirs[1] / "metrics.json"
        self.write(path, {**self.read(path), "score_threshold": .4})
        with self.assertRaisesRegex(ValueError, "thresholds"):
            self.compare(protocol="expanded-validation-resolution-v3")

    def test_unknown_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown comparison protocol"):
            self.compare(protocol="allow-any-difference")


if __name__ == "__main__":
    unittest.main()
