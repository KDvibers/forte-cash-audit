"""Returned semantic failures may release the worker, never publish a verdict."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyzer import service
from analyzer.outerwear import OuterwearObserver, outerwear_policy_fingerprint
from analyzer.outerwear_pass import observe_outerwear_video
from tests.test_outerwear import Client, answer, primary
from tests.test_rule_service import image_bytes, synthetic_report


OUTERWEAR = "removed_outerwear_in_cash_workplace"


class OuterwearRefusalRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.counter = 0
        self.rules = deepcopy(service.RULES)
        self.rules[OUTERWEAR]["enabled"] = True
        context = patch("analyzer.service.RULES", self.rules)
        context.start()
        self.addCleanup(context.stop)
        # Only these generated solid-color JPEGs are used; no videos or model.
        self.frames = [SimpleNamespace(timestamp_sec=float(i*4), jpeg=image_bytes(i)) for i in range(3)]
        context = patch("analyzer.outerwear_pass.sample_video", return_value={"frames": self.frames})
        context.start()
        self.addCleanup(context.stop)

    def cash(self, directory, mode="negative"):
        report = synthetic_report(directory, mode)
        report["settings"] = {"enable_actions": False, "max_motion_windows": 0, "max_refinement_frames": 0}
        for row in report["observations"]:
            row["calls"] = [{"completed": True, "finish_reason": "stop", "valid_json": True,
                             "request_seconds": .01}]
        report["request_count"] = len(report["observations"])
        return report

    def report(self, *, responses=None, mode="negative"):
        self.counter += 1
        directory = self.directory / str(self.counter)
        report = self.cash(directory, mode)
        if responses is None:
            responses = [primary("no"), primary("yes"),
                         answer({"outerwear": "resting_in_workplace", "garment_feature": "buttons"})]
        client = Client(responses)
        report.update(observe_outerwear_video(directory / "stub.mov", directory,
                      client=client, deadline=time.perf_counter()+60))
        report["combined_status"] = "partial"
        return report, directory, client

    def completed(self, report, evaluation=None):
        return service._refusal_inference_completed(report, evaluation or service.evaluate_report(report))

    def test_actual_stopped_pass_prefix_with_bad_inventory_is_safe_but_not_publishable(self):
        report, directory, client = self.report()
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(report["outerwear_checks"]), 2)
        self.assertEqual(len(report["selected_outerwear_frame_ids"]), 3)
        self.assertTrue(self.completed(report))
        self.assertFalse(service._inference_completed(report, service.evaluate_report(report)))
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.result_from_report(report, directory)
        self.assertTrue(caught.exception.inference_completed)
        self.assertEqual(report["outerwear_checks"][-1]["state"], "uncertain")

    def test_primary_schema_failure_and_invalid_json_are_returned_refusals(self):
        for response in (primary("wrong"), answer({}, valid_json=False)):
            with self.subTest(response=response):
                report, _, client = self.report(responses=[response])
                self.assertEqual(len(client.calls), 1)
                self.assertTrue(self.completed(report))

    def test_failure_on_last_selected_frame_still_cannot_be_a_negative(self):
        report, directory, _ = self.report(responses=[primary("no"), primary("no"), primary("wrong")])
        self.assertEqual(len(report["outerwear_checks"]), len(report["selected_outerwear_frame_ids"]))
        self.assertTrue(self.completed(report))
        with self.assertRaises(service.IncompleteAnalysisError):
            service.result_from_report(report, directory)

    def test_uncertain_previous_answer_is_completed_but_is_not_changed_to_no(self):
        report, _, _ = self.report(responses=[primary("uncertain"), primary("wrong")])
        self.assertTrue(self.completed(report))
        self.assertEqual(report["outerwear_checks"][0]["state"], "uncertain")

    def test_real_timeout_or_io_error_must_still_quarantine(self):
        for error in (TimeoutError("synthetic timeout"), OSError("synthetic transport failure")):
            with self.subTest(error=type(error).__name__):
                report, _, _ = self.report(responses=[primary("no"), error])
                self.assertFalse(self.completed(report))

    def test_missing_pending_or_unproven_call_still_quarantines(self):
        changes = [lambda row: row.pop("calls"), lambda row: row.update(calls=[]),
                   lambda row: row.update(calls=None),
                   lambda row: row["calls"][-1].update(completed=False),
                   lambda row: row["calls"][-1].update(completed=1),
                   lambda row: row["calls"][-1].update(error_type="TimeoutError"),
                   lambda row: row["calls"][-1].update(error="failed read"),
                   lambda row: row["calls"][-1].update(request_seconds=float("nan")),
                   lambda row: row["calls"][-1].update(request_seconds=-1),
                   lambda row: row["calls"][-1]["response"].update(finish_reason="length"),
                   lambda row: row["calls"][-1]["response"].update(valid_json=None),
                   lambda row: row["calls"][-1].update(question="unknown_extra_head")]
        original, _, _ = self.report()
        evaluation = service.evaluate_report(original)
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = deepcopy(original)
                change(report["outerwear_checks"][-1])
                self.assertFalse(self.completed(report, evaluation))

    def test_selection_must_be_an_exact_ordered_prefix(self):
        original, _, _ = self.report()
        evaluation = service.evaluate_report(original)
        changes = [lambda r: r.update(selected_outerwear_frame_ids=[]),
                   lambda r: r.update(selected_outerwear_timestamps=[]),
                   lambda r: r["selected_outerwear_frame_ids"].reverse(),
                   lambda r: r["selected_outerwear_frame_ids"].append("extra"),
                   lambda r: r["selected_outerwear_frame_ids"].__setitem__(2, r["selected_outerwear_frame_ids"][1]),
                   lambda r: r["selected_outerwear_timestamps"].__setitem__(0, .1),
                   lambda r: r["selected_outerwear_timestamps"].__setitem__(2, float("nan")),
                   lambda r: r["outerwear_checks"].pop(0)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = deepcopy(original)
                change(report)
                self.assertFalse(self.completed(report, evaluation))

    def test_only_last_row_may_have_the_known_semantic_error(self):
        original, _, _ = self.report()
        evaluation = service.evaluate_report(original)
        changes = [lambda r: r["outerwear_checks"][0].update(status="partial"),
                   lambda r: r["outerwear_checks"][-1].update(status="ok"),
                   lambda r: r["outerwear_checks"][-1].update(state="yes"),
                   lambda r: r["outerwear_checks"][-1].update(errors=[]),
                   lambda r: r["outerwear_checks"][-1]["errors"][0].update(type="RuntimeError"),
                   lambda r: r["outerwear_checks"][-1]["errors"][0].update(feature="unknown"),
                   lambda r: r["outerwear_checks"][0].update(policy_sha256="unknown"),
                   lambda r: r.update(outerwear_policy_sha256="unknown"),
                   lambda r: r.update(outerwear_status="failed"),
                   lambda r: r.update(outerwear_budget_exceeded=True),
                   lambda r: r.update(all_selected_outerwear_frames_processed=True),
                   lambda r: r.update(outerwear_errors=[]),
                   lambda r: r["outerwear_errors"][0].update(type="ObservationDeadline"),
                   lambda r: r["outerwear_errors"].append({"type": "ObservationError"})]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = deepcopy(original)
                change(report)
                self.assertFalse(self.completed(report, evaluation))

    def test_cash_and_other_inference_cannot_be_hidden_by_the_outerwear_failure(self):
        original, _, _ = self.report()
        evaluation = service.evaluate_report(original)
        changes = [lambda r: r.update(status="partial"),
                   lambda r: r.update(budget_exceeded=True),
                   lambda r: r.update(request_count=0),
                   lambda r: r.update(processed_frame_count=0),
                   lambda r: r.update(selected_timestamps=[]),
                   lambda r: r["observations"][0].update(calls=[]),
                   lambda r: r["observations"][0]["calls"][0].update(completed=False),
                   lambda r: r["settings"].update(enable_actions=True),
                   lambda r: r["settings"].update(max_refinement_frames=2),
                   lambda r: r.update(action_checks=[{"calls": []}]),
                   lambda r: r.update(counter_discovery={"calls": []})]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = deepcopy(original)
                change(report)
                self.assertFalse(self.completed(report, evaluation))

    def test_live_cash_positive_is_kept_locally_but_failed_then_next_job_can_finish(self):
        source = self.directory / "input.mov"
        source.write_bytes(b"Synthetic input; sample_video is stubbed, no real decode or inference")
        first = self.directory / "failed"
        client = Client([primary("no"), primary("wrong")])
        with patch("analyzer.service.check_ready", return_value=[]), patch(
                "analyzer.service.observe_video", side_effect=lambda source, out, **kw: self.cash(out, "positive")), patch(
                "analyzer.service.LocalModelClient", return_value=client):
            with self.assertRaises(service.IncompleteAnalysisError) as caught:
                service.analyze_to_directory(source, first)
        self.assertTrue(caught.exception.inference_completed)
        self.assertFalse((first / "result.json").exists())
        self.assertEqual(json.loads((first / "job.json").read_text())["status"], "failed")
        saved = json.loads((first / "rule_evaluation.json").read_text())
        self.assertGreater(len(saved["candidates"]), 0)
        client = Client([primary("no")] * 3)
        with patch("analyzer.service.check_ready", return_value=[]), patch(
                "analyzer.service.observe_video", side_effect=lambda source, out, **kw: self.cash(out)), patch(
                "analyzer.service.LocalModelClient", return_value=client):
            result = service.analyze_to_directory(source, self.directory / "next")
        self.assertEqual(result, {"verdict": "no_violation", "violations": []})

    def test_transport_journal_change_does_not_change_frozen_visual_policy(self):
        self.assertEqual(outerwear_policy_fingerprint(),
                         "7c4ebb8b8b842238750322b4c1e4c2121090e1a9ad3707ae299101852c35c5e1")


if __name__ == "__main__":
    unittest.main()
