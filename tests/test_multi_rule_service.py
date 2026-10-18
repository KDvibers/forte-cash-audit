"""Export independent visual checks with correct sources and original evidence."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyzer import service
from backend.validation import validate_result
from tests.test_rule_service import synthetic_report, image_bytes, Clock

OUTERWEAR = "removed_outerwear_in_cash_workplace"


def add_outerwear(report, directory, states, *, status="completed"):
    rows = []
    for index, state in enumerate(states):
        raw = image_bytes(index)
        name = f"frames/outerwear_{index}.jpg"
        (directory / name).write_bytes(raw)
        rows.append({"frame_id": f"outerwear_{index}", "timestamp_sec": float(index * 4),
                     "state": state, "status": "ok", "errors": [], "image_path": name,
                     "image_sha256": hashlib.sha256(raw).hexdigest()})
    report.update(outerwear_checks=rows, outerwear_status=status, outerwear_errors=[],
                  selected_outerwear_frame_ids=[r["frame_id"] for r in rows],
                  selected_outerwear_timestamps=[r["timestamp_sec"] for r in rows],
                  all_selected_outerwear_frames_processed=status == "completed")
    return report


class MultiRuleExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rules = deepcopy(service.RULES)
        rules[OUTERWEAR]["enabled"] = True
        patched = patch("analyzer.service.RULES", rules)
        patched.start()
        self.addCleanup(patched.stop)

    def report(self, cash="negative", outerwear=("no", "no", "no")):
        return add_outerwear(synthetic_report(self.directory, cash), self.directory, outerwear)

    def test_outerwear_uses_its_own_source_and_two_matching_original_jpegs(self):
        report = self.report(outerwear=("no", "yes", "yes"))
        result = service.result_from_report(report, self.directory)
        self.assertEqual(validate_result(result), result)
        event = result["violations"][0]
        self.assertEqual(len(result["violations"]), 1)
        self.assertEqual(event["timecode_sec"], 4)
        self.assertIn("снятая верхняя одежда", event["description"])
        self.assertIn("#z81", event["description"])
        self.assertNotIn("#z112", event["description"])
        self.assertNotIn("Человек ранее виден", event["description"])
        self.assertNotIn("Полнота остальных проверок ограничена", event["description"])
        for frame, row in zip(event["evidence_frames"], report["outerwear_checks"][1:]):
            self.assertEqual(frame["timestamp_sec"], row["timestamp_sec"])
            self.assertEqual(base64.b64decode(frame["image_base64"]), (self.directory / row["image_path"]).read_bytes())

    def test_both_rule_types_remain_distinct_and_chronological(self):
        result = service.result_from_report(self.report("positive", ("no", "yes", "yes")), self.directory)
        self.assertEqual(validate_result(result), result)
        self.assertEqual([e["timecode_sec"] for e in result["violations"]], [2.4667, 4])
        self.assertIn("#z112", result["violations"][0]["description"])
        self.assertIn("#z81", result["violations"][1]["description"])
        self.assertEqual([len(e["evidence_frames"]) for e in result["violations"]], [3, 2])

    def test_negative_requires_both_checks_complete_and_known(self):
        self.assertEqual(service.result_from_report(self.report(), self.directory),
                         {"verdict": "no_violation", "violations": []})
        for states in (("no", "uncertain", "no"), ("no", "yes", "no")):
            with self.subTest(states=states), self.assertRaises(service.IncompleteAnalysisError):
                service.result_from_report(self.report(outerwear=states), self.directory)
        report = self.report()
        report["all_selected_outerwear_frames_processed"] = False
        with self.assertRaises(service.IncompleteAnalysisError):
            service.result_from_report(report, self.directory)

    def test_missing_enabled_pass_is_not_a_successful_negative(self):
        report = synthetic_report(self.directory, "negative")
        with self.assertRaises(service.IncompleteAnalysisError):
            service.result_from_report(report, self.directory)

    def test_supported_cash_survives_unknown_outerwear_with_coverage_notice(self):
        result = service.result_from_report(self.report("positive", ("no", "uncertain", "no")), self.directory)
        self.assertEqual(result["verdict"], "violation")
        self.assertEqual(len(result["violations"]), 1)
        self.assertIn("Полнота остальных проверок ограничена", result["violations"][0]["description"])

    def test_supported_outerwear_survives_unknown_cash_with_coverage_notice(self):
        result = service.result_from_report(self.report("unknown", ("no", "yes", "yes")), self.directory)
        self.assertEqual(result["verdict"], "violation")
        self.assertIn("Полнота остальных проверок ограничена", result["violations"][0]["description"])

    def test_failed_cash_does_not_discard_completed_outerwear_or_publish_cash_rows(self):
        report = self.report("positive", ("no", "yes", "yes"))
        report["status"] = "failed"
        result = service.result_from_report(report, self.directory)
        self.assertEqual(len(result["violations"]), 1)
        self.assertIn("#z81", result["violations"][0]["description"])
        self.assertIn("Полнота остальных проверок ограничена", result["violations"][0]["description"])

    def test_failed_outerwear_does_not_discard_completed_cash_or_publish_its_rows(self):
        report = self.report("positive", ("no", "yes", "yes"))
        report["outerwear_status"] = "failed"
        result = service.result_from_report(report, self.directory)
        self.assertEqual(len(result["violations"]), 1)
        self.assertIn("#z112", result["violations"][0]["description"])
        check = service.evaluate_report(report)["checks"]["cash_visible_after_person_disappears"]
        self.assertNotIn("report_not_completed", check["coverage"]["negative_blockers"])

    def test_failure_of_either_check_blocks_negative(self):
        for field in ("status", "outerwear_status"):
            with self.subTest(field=field):
                report = self.report()
                report[field] = "failed"
                with self.assertRaises(service.IncompleteAnalysisError):
                    service.result_from_report(report, self.directory)

    def test_changed_outerwear_evidence_is_not_replaced_by_cash_frames(self):
        report = self.report(outerwear=("no", "yes", "yes"))
        (self.directory / report["outerwear_checks"][1]["image_path"]).write_bytes(image_bytes(4))
        with self.assertRaises(service.EvidenceError):
            service.result_from_report(report, self.directory)

    def test_cross_pass_id_collision_is_rejected(self):
        report = self.report(outerwear=("no", "yes", "yes"))
        report["outerwear_checks"][0]["frame_id"] = report["observations"][0]["frame_id"]
        report["selected_outerwear_frame_ids"][0] = report["observations"][0]["frame_id"]
        with self.assertRaises(service.EvidenceError):
            service.result_from_report(report, self.directory)


class OptionalPassCompletionGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.video = self.directory / 'input.mov'
        self.video.write_bytes(b'synthetic source; model calls are replaced in this test')
        self.out = self.directory / 'job'
        self.clock = Clock()
        self.cash_mode = 'negative'
        self.outerwear_states = ['no', 'no', 'no']
        self.cash_mutation = lambda report: None
        self.cash_seconds = 5
        rules = deepcopy(service.RULES)
        rules[OUTERWEAR]['enabled'] = True
        for target, options in (
                ('analyzer.service.RULES', {'new': rules}),
                ('analyzer.service.time.perf_counter', {'new': self.clock}),
                ('analyzer.service.check_ready', {'return_value': []}),
                ('analyzer.service.LocalModelClient', {'return_value': object()})):
            context = patch(target, **options)
            context.start()
            self.addCleanup(context.stop)
        context = patch('analyzer.service.observe_video', side_effect=self.cash)
        self.cash_mock = context.start()
        self.addCleanup(context.stop)
        context = patch('analyzer.outerwear_pass.observe_outerwear_video', side_effect=self.outerwear)
        self.outerwear_mock = context.start()
        self.addCleanup(context.stop)

    def cash(self, source, directory, **kwargs):
        report = synthetic_report(directory, self.cash_mode)
        self.cash_mutation(report)
        self.clock.now += self.cash_seconds
        return report

    def outerwear(self, source, directory, **kwargs):
        self.assertLess(self.clock.now, kwargs['deadline'])
        return add_outerwear({}, directory, self.outerwear_states)

    def test_partial_timeout_or_pending_call_never_starts_optional_inference(self):
        mutations = (
            lambda report: report.update(status='partial', all_selected_frames_processed=False),
            lambda report: report.update(status='partial', errors=[{'type': 'ObservationDeadline'}]),
            lambda report: report.update(budget_exceeded=True),
            lambda report: report['observations'][0].update(calls=[{'completed': False}]),
            lambda report: report['observations'][0].update(status='partial', errors=[{'type': 'OSError'}]),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.cash_mutation = mutation
                self.out = self.directory / f'failed-{index}'
                with self.assertRaises(service.IncompleteAnalysisError) as caught:
                    service.analyze_to_directory(self.video, self.out)
                self.assertFalse(caught.exception.inference_completed)
                self.outerwear_mock.assert_not_called()
                self.assertFalse((self.out / 'result.json').exists())
                report = json.loads((self.out / 'observations/report.json').read_text())
                evaluation = json.loads((self.out / 'rule_evaluation.json').read_text())
                self.assertNotIn('outerwear_checks', report)
                self.assertNotIn('checks', evaluation, 'The saved evaluation describes only the attempted cash pass')
                job = json.loads((self.out / 'job.json').read_text())
                self.assertEqual(job['status'], 'failed')
                self.assertEqual(job['skipped_rule_passes'][0]['rule_id'], OUTERWEAR)

    def test_schema_failure_is_safe_refusal_but_still_skips_optional_pass(self):
        def malformed(report):
            report.update(status='partial', all_selected_frames_processed=False,
                          settings={'enable_actions': False, 'max_motion_windows': 0, 'max_refinement_frames': 0})
            for row in report['observations']:
                row['calls'] = [{'completed': True, 'finish_reason': 'stop', 'valid_json': True, 'request_seconds': .2}]
            report['request_count'] = len(report['observations'])
            report['observations'][-1].update(status='partial', errors=[{'type': 'ObservationError'}])
        self.cash_mutation = malformed
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.analyze_to_directory(self.video, self.out)
        self.assertTrue(caught.exception.inference_completed)
        self.outerwear_mock.assert_not_called()
        self.assertFalse((self.out / 'result.json').exists())

    def test_completed_visual_uncertainty_allows_optional_pass_without_claiming_negative(self):
        self.cash_mode = 'unknown'
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.analyze_to_directory(self.video, self.out)
        self.assertTrue(caught.exception.inference_completed)
        self.outerwear_mock.assert_called_once()
        report = json.loads((self.out / 'observations/report.json').read_text())
        self.assertEqual(report['outerwear_status'], 'completed')
        self.assertFalse((self.out / 'result.json').exists())

    def test_completed_two_pass_profile_preserves_both_candidates_and_evidence(self):
        self.cash_mode = 'positive'
        self.outerwear_states = ['no', 'yes', 'yes']
        result = service.analyze_to_directory(self.video, self.out)
        self.outerwear_mock.assert_called_once()
        self.assertEqual(validate_result(result), result)
        self.assertEqual(result['verdict'], 'violation')
        self.assertEqual(len(result['violations']), 2)
        self.assertEqual([event['timecode_sec'] for event in result['violations']], [2.4667, 4.])

    def test_optional_pass_cannot_start_after_shared_budget_reserve_has_expired(self):
        self.cash_seconds = 18
        with self.assertRaises(TimeoutError):
            service.analyze_to_directory(self.video, self.out, budget_sec=20)
        self.outerwear_mock.assert_not_called()
        self.assertFalse((self.out / 'result.json').exists())


if __name__ == "__main__":
    unittest.main()
