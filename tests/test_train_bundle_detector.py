"""Fail-closed detector runner checks with local fixtures and mocked models."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from analyzer.bundle_dataset import export_dataset, file_sha
from analyzer.train_bundle_detector import evaluate, train, verified_dataset


class DetectorRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        rows = []
        for i in range(1, 4):
            video = self.root / f"video{i}.mov"
            video.write_bytes(f"local-fixture-video-{i}".encode())
            image = self.root / f"frame{i}.jpg"
            Image.new("RGB", (60, 40), (40*i, 20, 40)).save(image)
            rows.append({"id": f"frame{i}", "video_id": i, "timestamp_sec": 0,
                         "source_video_path": video.name, "source_video_sha256": file_sha(video),
                         "image_path": image.name, "image_sha256": file_sha(image),
                         "width": 60, "height": 40, "roi_xyxy": [10, 5, 55, 35],
                         "boxes_xyxy": [[20, 10, 30, 20]], "box_contexts": ["in_hand"],
                         "ignore_boxes_xyxy": [], "fully_annotated": True})
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"schema_version": 1, "rows": rows}), encoding="utf-8")
        export_dataset([labels], self.root, {1: "train", 2: "valid", 3: "test"}, self.dataset)
        self.weights = self.root / "local-checkpoint.pth"
        self.weights.write_bytes(b"fake weights: never passed to torch")
        self.args = SimpleNamespace(dataset=self.dataset, weights=self.weights,
                                    out=self.root / "run", split="test", epochs=1)
        self.torch = SimpleNamespace(
            __version__="test", version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                                 get_device_name=Mock(return_value="mock GPU"),
                                 max_memory_allocated=Mock(return_value=0),
                                 max_memory_reserved=Mock(return_value=0)))

    def fake_model(self, result=None, *, class_names=None):
        model = SimpleNamespace(class_names={1: "banknote_bundle"} if class_names is None else class_names,
                                predict=Mock(return_value=result), train=Mock(),
                                model=SimpleNamespace(resolution=384), model_config=SimpleNamespace(resolution=384))
        return model, Mock(return_value=model)

    def run_evaluate(self, constructor):
        with patch("analyzer.train_bundle_detector.offline_runtime", return_value=(self.torch, constructor)), \
                redirect_stdout(StringIO()):
            evaluate(self.args)

    def test_mutated_export_pixels_rejected_before_model_initialization(self):
        (self.dataset / "test/frame3.jpg").write_bytes(b"changed")
        with patch("analyzer.train_bundle_detector.offline_runtime") as runtime:
            with self.assertRaises(ValueError):
                evaluate(self.args)
        runtime.assert_not_called()
        self.assertFalse(self.args.out.exists())

    def test_mutated_coco_training_labels_rejected_despite_unchanged_images(self):
        path = self.dataset / "train/_annotations.coco.json"
        before = path.read_bytes()
        data = json.loads(before)
        data["annotations"][0]["bbox"] = [0, 0, 30, 20]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            verified_dataset(self.dataset)
        path.write_bytes(before)
        self.assertEqual(verified_dataset(self.dataset)[0], self.dataset.resolve())

    def test_wrong_checkpoint_class_cannot_write_metrics_or_call_predict(self):
        model, constructor = self.fake_model(class_names={0: "person", 1: "bicycle"})
        with self.assertRaises(ValueError):
            self.run_evaluate(constructor)
        model.predict.assert_not_called()
        self.assertFalse((self.args.out / "metrics.json").exists())
        self.assertFalse((self.args.out / "predictions.json").exists())

    def test_failed_prediction_cannot_be_replaced_with_successful_empty_output(self):
        model, constructor = self.fake_model()
        model.predict.side_effect = RuntimeError("local inference failed")
        with self.assertRaisesRegex(RuntimeError, "local inference failed"):
            self.run_evaluate(constructor)
        self.assertFalse((self.args.out / "metrics.json").exists())
        self.assertFalse((self.args.out / "predictions.json").exists())

    def test_misaligned_prediction_arrays_are_rejected_instead_of_zip_truncation(self):
        for number, result in enumerate((
                SimpleNamespace(xyxy=[[0, 0, 10, 10]], confidence=[], class_id=[1]),
                SimpleNamespace(xyxy=[], confidence=[.9], class_id=[1]),
                SimpleNamespace(xyxy=[[0, 0, 10, 10]], confidence=[.9], class_id=[]))):
            with self.subTest(result=result):
                self.args.out = self.root / f"mismatch{number}"
                _, constructor = self.fake_model(result)
                with self.assertRaises(ValueError):
                    self.run_evaluate(constructor)
                self.assertFalse((self.args.out / "metrics.json").exists())

    def test_invalid_raw_coordinates_scores_and_classes_raise_before_clipping(self):
        cases = [([float("nan"), 0, 10, 10], .9, 1),
                 ([0, 0, float("inf"), 10], .9, 1),
                 ([10, 0, 0, 10], .9, 1),
                 ([0, 0, 10, 10], float("nan"), 1),
                 ([0, 0, 10, 10], 1.1, 1),
                 ([0, 0, 10, 10], .9, 1.5)]
        for number, (box, score, category) in enumerate(cases):
            with self.subTest(box=box, score=score, category=category):
                self.args.out = self.root / f"invalid{number}"
                _, constructor = self.fake_model(SimpleNamespace(xyxy=[box], confidence=[score], class_id=[category]))
                with self.assertRaises(ValueError):
                    self.run_evaluate(constructor)
                self.assertFalse((self.args.out / "metrics.json").exists())

    def test_valid_crop_predictions_restore_original_coordinates_and_ignore_outside_boxes(self):
        _, constructor = self.fake_model(SimpleNamespace(
            xyxy=[[10, 5, 20, 15], [-20, 0, -10, 5]], confidence=[.9, .9], class_id=[1, 1]))
        self.run_evaluate(constructor)
        predictions = json.loads((self.args.out / "predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(predictions[0]["boxes_xyxy"], [[20, 10, 30, 20]])
        metrics = json.loads((self.args.out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["totals"]["tp"], 1)
        self.assertEqual(metrics["totals"]["fp"], 0)
        self.assertEqual(metrics["weights_sha256"], file_sha(self.weights))

    def test_training_without_selected_checkpoint_records_failure_and_does_not_run_test(self):
        model, constructor = self.fake_model()
        with patch("analyzer.train_bundle_detector.offline_runtime", return_value=(self.torch, constructor)), \
                patch("analyzer.train_bundle_detector.importlib.metadata.version", return_value="test"):
            with self.assertRaisesRegex(RuntimeError, "selected checkpoint"):
                train(self.args)
        self.assertFalse(model.train.call_args.kwargs["run_test"])
        report = json.loads((self.args.out / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("selected_checkpoint_sha256", report)
        self.assertIn("RuntimeError", report["error"])

    def test_evaluation_uses_and_records_requested_resolution(self):
        self.args.resolution = 512
        model, constructor = self.fake_model(SimpleNamespace(xyxy=[], confidence=[], class_id=[]))
        model.model.resolution = model.model_config.resolution = 512
        self.run_evaluate(constructor)
        self.assertEqual(constructor.call_args.kwargs["resolution"], 512)
        metrics = json.loads((self.args.out / "metrics.json").read_text())
        self.assertEqual(metrics["resolution"], 512)

    def test_silently_ignored_resolution_cannot_generate_metrics(self):
        self.args.resolution = 512
        model, constructor = self.fake_model()
        with self.assertRaisesRegex(ValueError, "resolution differs"):
            self.run_evaluate(constructor)
        model.predict.assert_not_called()
        self.assertFalse((self.args.out / "metrics.json").exists())

    def test_unsupported_resolution_rejected_before_dataset_or_model(self):
        self.args.resolution = 513
        with patch("analyzer.train_bundle_detector.offline_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "384 or 512"):
                evaluate(self.args)
        runtime.assert_not_called()
        self.assertFalse(self.args.out.exists())


if __name__ == "__main__":
    unittest.main()
