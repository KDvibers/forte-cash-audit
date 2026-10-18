"""CPU contract and evidence-boundary tests for the real rule service."""

import base64
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer import service
from backend.validation import validate_result


RULE_ID = "cash_visible_after_person_disappears"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def image_bytes(index, format="JPEG"):
    stream = BytesIO()
    Image.new("RGB", (80, 60), (20 + index * 30, 100, 180 - index * 20)).save(stream, format)
    return stream.getvalue()


def synthetic_report(directory, mode="positive", status="completed"):
    """A visible person followed by cash on the desk; no dataset or model data."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frames").mkdir(exist_ok=True)
    if mode == "positive":
        scene = [("yes", "uncertain"), ("yes", "uncertain"),
                 ("no", "yes"), ("no", "yes"), ("no", "yes")]
    elif mode == "negative":
        # Either known operand can disprove the conjunction on a sampled frame.
        scene = [("yes", "uncertain"), ("yes", "yes"), ("uncertain", "no")]
    elif mode == "unknown":
        scene = [("yes", "uncertain"), ("yes", "uncertain"), ("uncertain", "yes")]
    else:
        raise AssertionError("Unknown synthetic scene")
    times = [.0333, 1.2, 2.4667, 3.9667, 5.1]
    rows = []
    for index, (person, cash) in enumerate(scene):
        raw = image_bytes(index)
        path = f"frames/source_{index}.jpg"
        (directory / path).write_bytes(raw)
        rows.append({"frame_id": f"source_{index}", "timestamp_sec": times[index],
                     "states": {"person_in_work_area": person, "cash_on_worktop": cash},
                     "status": "ok", "errors": [], "calls": [], "image_path": path,
                     "image_sha256": hashlib.sha256(raw).hexdigest()})
    report = {"observations": rows, "status": status, "errors": [], "budget_exceeded": False,
              "selected_frame_count": len(rows), "processed_frame_count": len(rows),
              "all_selected_frames_processed": status == "completed",
              "selected_timestamps": times[:len(rows)]}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class RuleServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.report_directory = self.directory / "review"
        self.out = self.directory / "job"
        self.video = self.directory / "input.mov"
        self.video.write_bytes(b"synthetic input; decoding is replaced in these CPU tests")
        self.source_sha = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.clock = Clock()
        self.lock = threading.Lock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("analyzer.service._LOCK", self.lock))
        self.stack.enter_context(patch("analyzer.service.time.perf_counter", self.clock))
        # Fail closed on any accidental real readiness/model invocation.
        self.ready = self.stack.enter_context(patch("analyzer.service.check_ready", side_effect=self.fake_ready))
        self.observe = self.stack.enter_context(patch("analyzer.service.observe_video", side_effect=self.fake_observe))
        self.mode = "positive"
        self.report_status = "completed"
        self.observe_seconds = 5.0
        self.mutate_source = False

    def fake_ready(self, **kwargs):
        self.clock.now += 2
        return [{"url": "http://127.0.0.1:8080", "runtime": {"model_path": "synthetic-2b"}},
                {"url": "http://127.0.0.1:8081", "runtime": {"model_path": "synthetic-4b"}}]

    def fake_observe(self, source, destination, **kwargs):
        report = synthetic_report(Path(destination), self.mode, self.report_status)
        self.clock.now += self.observe_seconds
        if self.mutate_source:
            Path(source).write_bytes(b"source replaced while the observation pass was running")
        return report

    def job(self):
        return json.loads((self.out / "job.json").read_text(encoding="utf-8"))

    def assert_failed_job(self, error_type):
        metadata = self.job()
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], error_type)
        self.assertFalse((self.out / "result.json").exists())
        self.assertFalse(self.lock.locked())
        self.assertFalse((self.out / "job.json.tmp").exists())

    def test_actual_jpeg_bytes_and_pts_survive_export_and_backend_validation(self):
        report = synthetic_report(self.report_directory)
        original = deepcopy(report)
        result = service.result_from_report(report, self.report_directory)
        self.assertEqual(validate_result(result), result)
        self.assertEqual(result["verdict"], "violation")
        self.assertEqual(len(result["violations"]), 1)
        event = result["violations"][0]
        self.assertEqual(event["timecode_sec"], 2.4667)
        # Last visible person, first match and the actual confirmation frame.
        expected = [report["observations"][i] for i in (1, 2, 3)]
        self.assertEqual(len(event["evidence_frames"]), 3)
        self.assertEqual(event["evidence_frames"][1]["timestamp_sec"], event["timecode_sec"])
        for row, frame in zip(expected, event["evidence_frames"]):
            raw = (self.report_directory / row["image_path"]).read_bytes()
            self.assertEqual(frame["timestamp_sec"], row["timestamp_sec"])
            self.assertEqual(base64.b64decode(frame["image_base64"], validate=True), raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["image_sha256"])
        self.assertIn("ранее принятым", event["description"])
        self.assertIn("не установлены", event["description"])
        self.assertIn("подтверждённое следующим кадром", event["description"])
        self.assertIn("1.20 с", event["description"])
        self.assertIn("3.97 с", event["description"])
        self.assertIn("https://adilet.zan.kz/rus/docs/V1900019680#z112", event["description"])
        self.assertEqual(report, original)
        self.ready.assert_not_called()
        self.observe.assert_not_called()

    def test_partial_positive_is_exported_with_explicit_coverage_limitation(self):
        report = synthetic_report(self.report_directory, status="partial")
        report["errors"] = [{"type": "ObservationDeadline", "message": "later frame not processed"}]
        report["selected_frame_count"] += 1
        report["selected_timestamps"].append(6.5)
        result = service.result_from_report(report, self.report_directory)
        self.assertEqual(validate_result(result), result)
        self.assertEqual(result["verdict"], "violation")
        self.assertIn("Полнота остальных проверок ограничена", result["violations"][0]["description"])
        self.assertEqual(len(result["violations"][0]["evidence_frames"]), 3)

    def test_complete_scoped_negative_is_exact_api_dict_but_unknown_or_incomplete_is_not(self):
        for mode, status, raises in (("negative", "completed", False),
                                     ("negative", "partial", True),
                                     ("unknown", "completed", True)):
            with self.subTest(mode=mode, status=status):
                report = synthetic_report(self.report_directory, mode, status)
                if raises:
                    with self.assertRaises(service.IncompleteAnalysisError):
                        service.result_from_report(report, self.report_directory)
                else:
                    result = service.result_from_report(report, self.report_directory)
                    self.assertEqual(result, {"verdict": "no_violation", "violations": []})
                    self.assertEqual(validate_result(result), result)

    def test_unfinished_or_failed_report_cannot_publish_even_a_positive(self):
        for status in ("running", "failed"):
            with self.subTest(status=status):
                report = synthetic_report(self.report_directory, status=status)
                with self.assertRaises(service.IncompleteAnalysisError):
                    service.result_from_report(report, self.report_directory)

    def test_missing_changed_or_unhashed_evidence_fails_without_substitution(self):
        for problem in ("missing", "changed", "missing_hash", "invalid_hash"):
            with self.subTest(problem=problem):
                report = synthetic_report(self.report_directory)
                row = report["observations"][2]
                path = self.report_directory / row["image_path"]
                if problem == "missing":
                    path.unlink()
                elif problem == "changed":
                    path.write_bytes(image_bytes(7))
                elif problem == "missing_hash":
                    row.pop("image_sha256")
                else:
                    row["image_sha256"] = "not-a-sha256"
                with self.assertRaises(service.EvidenceError):
                    service.result_from_report(report, self.report_directory)

    def test_correct_hash_does_not_make_png_or_truncated_jpeg_acceptable(self):
        for raw in (image_bytes(0, "PNG"), image_bytes(0)[:80]):
            with self.subTest(header=raw[:8]):
                report = synthetic_report(self.report_directory)
                row = report["observations"][2]
                (self.report_directory / row["image_path"]).write_bytes(raw)
                row["image_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises(service.EvidenceError):
                    service.result_from_report(report, self.report_directory)

    def test_traversal_and_absolute_paths_are_rejected_even_with_matching_external_jpeg(self):
        outside = self.directory / "outside.jpg"
        outside.write_bytes(image_bytes(5))
        for path in ("../outside.jpg", str(outside.resolve()), "frames/../../outside.jpg"):
            with self.subTest(path=path):
                report = synthetic_report(self.report_directory)
                row = report["observations"][2]
                row.update(image_path=path, image_sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
                with self.assertRaises(service.EvidenceError):
                    service.result_from_report(report, self.report_directory)

    def test_hardlinked_evidence_is_rejected_even_when_inside_run_and_hash_matches(self):
        report = synthetic_report(self.report_directory)
        row = report["observations"][2]
        original = self.report_directory / row["image_path"]
        alias = self.report_directory / "frames" / "alias.jpg"
        try:
            os.link(original, alias)
        except OSError as error:
            self.skipTest(f"Hard links unavailable: {error}")
        row["image_path"] = "frames/alias.jpg"
        self.assertGreater(alias.stat().st_nlink, 1)
        with self.assertRaises(service.EvidenceError):
            service.result_from_report(report, self.report_directory)

    def test_symlink_evidence_is_rejected_even_when_target_is_inside_run(self):
        report = synthetic_report(self.report_directory)
        row = report["observations"][2]
        target = self.report_directory / row["image_path"]
        alias = self.report_directory / "frames" / "linked.jpg"
        try:
            alias.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink creation unavailable without Windows privilege: {error}")
        row["image_path"] = "frames/linked.jpg"
        with self.assertRaises(service.EvidenceError):
            service.result_from_report(report, self.report_directory)

    def test_disabled_missing_or_legal_finding_catalog_entry_cannot_be_emitted(self):
        report = synthetic_report(self.report_directory)
        specification = deepcopy(service.RULES[RULE_ID])
        policies = ({}, {RULE_ID: {**specification, "enabled": False}},
                    {RULE_ID: {**specification, "direct_legal_finding": True}},
                    {RULE_ID: {**specification, "requires_human_review": False}})
        for catalog in policies:
            with self.subTest(catalog=list(catalog)), patch("analyzer.service.RULES", catalog):
                with self.assertRaises(service.IncompleteAnalysisError):
                    service.result_from_report(report, self.report_directory)

    def test_export_refuses_zero_duplicate_or_over_limit_evidence_ids(self):
        report = synthetic_report(self.report_directory)
        original = service.evaluate_rules(report)
        for ids in ([], ["source_1", "source_1"], [f"source_{i}" for i in range(4)]):
            evaluation = deepcopy(original)
            evaluation["candidates"][0]["evidence_ids"] = ids
            with self.subTest(ids=ids), patch("analyzer.service.evaluate_rules", return_value=evaluation):
                with self.assertRaises(service.EvidenceError):
                    service.result_from_report(report, self.report_directory)

    def test_job_success_records_source_hash_nested_observations_and_default_budget(self):
        result = service.analyze_to_directory(self.video, self.out)
        self.assertEqual(validate_result(result), result)
        metadata = self.job()
        self.assertEqual(metadata["status"], "done")
        self.assertEqual(metadata["budget_sec"], 260.0)
        self.assertEqual(metadata["source_video_sha256"], self.source_sha)
        self.assertEqual(metadata["rule_scope"], [RULE_ID])
        self.assertEqual(metadata["verdict_semantics"], "case_review_signal_not_established_legal_breach")
        self.assertEqual(metadata["total_seconds"], 7)
        self.assertEqual(self.observe.call_args.args, (self.video.resolve(), self.out.resolve() / "observations"))
        arguments = self.observe.call_args.kwargs
        self.assertGreater(arguments["budget_sec"], 0)
        self.assertLess(arguments["budget_sec"], metadata["budget_sec"] - 2)
        self.assertFalse(arguments["enable_actions"])
        self.assertEqual(arguments["cabinet_mode"], "disabled")
        self.assertEqual(arguments["max_frames"], 80)
        self.assertEqual(arguments["max_refinement_frames"], 0)
        self.assertTrue(arguments["skip_cash_when_person_present"])
        self.assertEqual(arguments["interval_sec"], 2.0)
        saved = json.loads((self.out / "observations" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["source_video_sha256"], self.source_sha)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(json.loads((self.out / "result.json").read_text(encoding="utf-8")), result)
        evaluation = json.loads((self.out / "rule_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(len(evaluation["candidates"]), 1)
        self.assertFalse(self.lock.locked())

    def test_existing_job_directory_is_never_overwritten_and_lock_is_released(self):
        self.out.mkdir()
        sentinel = self.out / "job.json"
        sentinel.write_bytes(b"existing report must remain byte-identical")
        with self.assertRaises(FileExistsError):
            service.analyze_to_directory(self.video, self.out)
        self.assertEqual(sentinel.read_bytes(), b"existing report must remain byte-identical")
        self.assertEqual(list(self.out.iterdir()), [sentinel])
        self.ready.assert_not_called()
        self.observe.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_source_mutation_fails_job_before_any_result_is_published(self):
        self.mutate_source = True
        with self.assertRaises(service.EvidenceError):
            service.analyze_to_directory(self.video, self.out)
        self.assert_failed_job("EvidenceError")
        self.assertEqual(self.job()["source_video_sha256"], self.source_sha)
        self.assertNotEqual(hashlib.sha256(self.video.read_bytes()).hexdigest(), self.source_sha)

    def test_incomplete_negative_job_fails_instead_of_returning_a_mock_or_normal_verdict(self):
        self.mode = "negative"
        self.report_status = "partial"
        with self.assertRaises(service.IncompleteAnalysisError):
            service.analyze_to_directory(self.video, self.out)
        self.assert_failed_job("IncompleteAnalysisError")
        evaluation = json.loads((self.out / "rule_evaluation.json").read_text(encoding="utf-8"))
        self.assertFalse(evaluation["candidates"])
        self.assertFalse(evaluation["can_conclude_no_candidate"])
        self.assertIn("report_not_completed", evaluation["coverage"]["negative_blockers"])

    def test_live_partial_positive_preserves_candidates_without_admitting_next_gpu_job(self):
        self.report_status = "partial"
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.analyze_to_directory(self.video, self.out)
        self.assertFalse(caught.exception.inference_completed)
        self.assert_failed_job("IncompleteAnalysisError")
        evaluation = json.loads((self.out / "rule_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(len(evaluation["candidates"]), 1)
        self.assertTrue((self.out / "observations" / "report.json").exists())

    def test_positive_with_completed_summary_but_timeout_or_missing_samples_is_not_published(self):
        for i, change in enumerate(({"errors": [{"type": "ObservationDeadline"}]},
                                    {"budget_exceeded": True}, {"all_selected_frames_processed": False})):
            with self.subTest(change=change):
                self.out = self.directory / f"incomplete-positive-{i}"
                def observe(source, destination, **kwargs):
                    report = self.fake_observe(source, destination, **kwargs)
                    report.update(change)
                    return report
                self.observe.side_effect = observe
                with self.assertRaises(service.IncompleteAnalysisError) as caught:
                    service.analyze_to_directory(self.video, self.out)
                self.assertFalse(caught.exception.inference_completed)
                self.assert_failed_job("IncompleteAnalysisError")

    def test_ready_failure_releases_worker_lock_and_a_subsequent_job_can_finish(self):
        self.ready.side_effect = RuntimeError("local model unavailable")
        with self.assertRaisesRegex(RuntimeError, "local model unavailable"):
            service.analyze_to_directory(self.video, self.out)
        self.assert_failed_job("RuntimeError")
        self.observe.assert_not_called()
        self.ready.side_effect = self.fake_ready
        self.out = self.directory / "next-job"
        self.mode = "negative"
        result = service.analyze_to_directory(self.video, self.out)
        self.assertEqual(result, {"verdict": "no_violation", "violations": []})
        self.assertEqual(self.job()["status"], "done")
        self.assertFalse(self.lock.locked())

    def test_result_at_job_deadline_is_not_published_even_if_signal_is_positive(self):
        self.observe_seconds = 18  # Readiness takes two; local job budget is twenty.
        with self.assertRaises(TimeoutError):
            service.analyze_to_directory(self.video, self.out, budget_sec=20)
        self.assert_failed_job("TimeoutError")
        self.assertEqual(self.job()["total_seconds"], 20)
        self.assertFalse((self.out / "rule_evaluation.json").exists())

    def test_backend_entrypoint_respects_reduced_budget_after_queue_wait(self):
        self.observe_seconds = 18
        with patch("analyzer.service._ROOT", self.directory):
            with self.assertRaises(TimeoutError):
                service.analyze_video(str(self.video), budget_sec=20)
        jobs = list((self.directory / "runs" / "stage3").glob("service-*"))
        self.assertEqual(len(jobs), 1)
        self.out = jobs[0]
        self.assert_failed_job("TimeoutError")
        self.assertEqual(self.job()["budget_sec"], 20)

    def test_late_result_file_write_does_not_publish_or_return_success(self):
        self.observe_seconds = 17.5
        original = Path.write_text

        def slow_write(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path.name == "result.json.tmp":
                self.clock.now += 1
            return value

        with patch.object(Path, "write_text", slow_write):
            with self.assertRaises(TimeoutError):
                service.analyze_to_directory(self.video, self.out, budget_sec=20)
        self.assert_failed_job("TimeoutError")
        self.assertFalse((self.out / "result.json.tmp").exists())

    def test_late_final_metadata_revokes_the_result_and_releases_worker(self):
        self.observe_seconds = 17.5
        original = service._save

        def slow_save(path, value, **kwargs):
            original(path, value, **kwargs)
            if path.name == "job.json" and value.get("status") == "done":
                self.clock.now += 1

        with patch("analyzer.service._save", side_effect=slow_save):
            with self.assertRaises(TimeoutError):
                service.analyze_to_directory(self.video, self.out, budget_sec=20)
        self.assert_failed_job("TimeoutError")
        self.assertEqual(self.job()["total_seconds"], 20.5)

    def test_single_argument_entrypoint_retains_default_budget_and_valid_result(self):
        with patch("analyzer.service._ROOT", self.directory):
            result = service.analyze_video(str(self.video))
        self.assertEqual(validate_result(result), result)
        jobs = list((self.directory / "runs" / "stage3").glob("service-*"))
        self.assertEqual(len(jobs), 1)
        self.out = jobs[0]
        self.assertEqual(self.job()["budget_sec"], 260)
        self.assertEqual(self.job()["status"], "done")

    def test_public_scope_lists_only_enabled_rules_without_model_requests(self):
        scope = service.describe_scope()
        self.assertEqual([rule["id"] for rule in scope["rules"]], [RULE_ID])
        self.assertFalse(scope["complete_case_coverage"])
        self.assertTrue(scope["notice"])
        self.assertEqual(set(scope["rules"][0]), {"id", "title"})
        self.ready.assert_not_called()
        self.observe.assert_not_called()

    def test_invalid_budget_and_busy_worker_do_not_create_jobs_or_call_models(self):
        for budget in (True, -1, 0, 10, float("nan"), float("inf"), "260"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                service.analyze_to_directory(self.video, self.out, budget_sec=budget)
            self.assertFalse(self.out.exists())
        self.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                service.analyze_to_directory(self.video, self.out)
            self.assertTrue(self.lock.locked())
            self.assertFalse(self.out.exists())
        finally:
            self.lock.release()
        self.ready.assert_not_called()
        self.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
