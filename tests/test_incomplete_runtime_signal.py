"""Only a completed visual refusal may safely admit the next GPU job."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyzer import service
from tests.test_rule_service import synthetic_report
from tests.test_multi_rule_service import add_outerwear, OUTERWEAR


class CompletedInferenceSignalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.rules = deepcopy(service.RULES)
        self.rules[OUTERWEAR]['enabled'] = False
        context = patch('analyzer.service.RULES', self.rules)
        context.start()
        self.addCleanup(context.stop)

    def refusal(self, report):
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.result_from_report(report, self.directory)
        return caught.exception

    def test_default_exception_never_claims_completed_inference(self):
        self.assertFalse(service.IncompleteAnalysisError('unknown').inference_completed)

    def test_clean_completed_unknown_is_recoverable_without_negative_verdict(self):
        report = synthetic_report(self.directory, 'unknown')
        self.assertTrue(self.refusal(report).inference_completed)

    def test_technical_or_unfinished_report_never_requests_recovery(self):
        for change in ({'status': 'partial'}, {'status': 'failed'}, {'budget_exceeded': True},
                       {'errors': [{'type': 'TimeoutError'}]}, {'all_selected_frames_processed': False}):
            with self.subTest(change=change):
                report = synthetic_report(self.directory, 'unknown')
                report.update(change)
                self.assertFalse(self.refusal(report).inference_completed)
        report = synthetic_report(self.directory, 'unknown')
        report['observations'][1]['errors'] = [{'type': 'TimeoutError'}]
        self.assertFalse(self.refusal(report).inference_completed)

    def test_all_enabled_passes_must_finish_before_recovery(self):
        self.rules[OUTERWEAR]['enabled'] = True
        report = add_outerwear(synthetic_report(self.directory, 'negative'), self.directory,
                              ['no', 'uncertain', 'no'])
        self.assertTrue(self.refusal(report).inference_completed)
        report['outerwear_budget_exceeded'] = True
        self.assertFalse(self.refusal(report).inference_completed)
        report['outerwear_budget_exceeded'] = False
        report['outerwear_status'] = 'partial'
        self.assertFalse(self.refusal(report).inference_completed)


class CompletedSchemaFailureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.rules = deepcopy(service.RULES)
        self.rules[OUTERWEAR]['enabled'] = False
        context = patch('analyzer.service.RULES', self.rules)
        context.start()
        self.addCleanup(context.stop)

    def report(self, directory=None, mode='unknown'):
        report = synthetic_report(directory or self.directory, mode)
        report.update(status='partial', all_selected_frames_processed=False,
                      settings={'enable_actions': False, 'max_motion_windows': 0, 'max_refinement_frames': 0})
        for row in report['observations']:
            row['calls'] = [{'completed': True, 'finish_reason': 'stop',
                             'valid_json': True, 'request_seconds': .25}]
        row = report['observations'][-1]
        row.update(status='partial', errors=[{'type': 'ObservationError', 'feature': 'cash_on_worktop',
                                             'message': 'Invalid bounding box after complete response'}])
        report['request_count'] = len(report['observations'])
        return report

    def completed(self, report):
        return service._refusal_inference_completed(report, service.evaluate_report(report))

    def test_schema_refusal_is_recoverable_but_never_a_negative_result(self):
        report = self.report()
        self.assertFalse(service._inference_completed(report, service.evaluate_report(report)))
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.result_from_report(report, self.directory)
        self.assertTrue(caught.exception.inference_completed)

    def test_completed_invalid_json_also_refuses_without_pending_transport(self):
        report = self.report()
        report['observations'][-1]['calls'][0]['valid_json'] = False
        self.assertTrue(self.completed(report))

    def test_missing_or_ambiguous_call_evidence_still_quarantines(self):
        for change in ({'completed': False}, {'completed': 1}, {'error_type': 'TimeoutError'},
                       {'error': 'read failed'}, {'finish_reason': None}, {'finish_reason': 'length'},
                       {'valid_json': None}, {'request_seconds': float('nan')}, {'request_seconds': -1}):
            with self.subTest(change=change):
                report = self.report()
                report['observations'][0]['calls'][0].update(change)
                self.assertFalse(self.completed(report))
        for calls in ([], None):
            report = self.report()
            report['observations'][0]['calls'] = calls
            self.assertFalse(self.completed(report))
        report = self.report()
        del report['observations'][0]['calls']
        self.assertFalse(self.completed(report))

    def test_global_timeout_or_unknown_error_never_uses_schema_recovery(self):
        for change in ({'budget_exceeded': True}, {'status': 'failed'},
                       {'errors': [{'type': 'ObservationDeadline'}]}, {'errors': [{'type': 'OSError'}]}):
            with self.subTest(change=change):
                report = self.report()
                report.update(change)
                self.assertFalse(self.completed(report))
        for error in ('TimeoutError', 'ObservationDeadline', 'OSError', 'RuntimeError', 'JSONDecodeError'):
            report = self.report()
            report['observations'][-1]['errors'][0]['type'] = error
            self.assertFalse(self.completed(report))

    def test_counts_timestamps_and_other_passes_are_not_silently_ignored(self):
        for change in ({'selected_frame_count': 4}, {'processed_frame_count': 2}, {'request_count': 0},
                       {'selected_timestamps': [.0333, 1.2, 999.]}, {'action_checks': [{'calls': []}]},
                       {'counter_discovery': {'calls': []}}, {'settings': {'enable_actions': True}}):
            with self.subTest(change=change):
                report = self.report()
                report.update(change)
                self.assertFalse(self.completed(report))
        report = self.report()
        evaluation = service.evaluate_report(report)
        report['observations'][-1]['frame_id'] = report['observations'][0]['frame_id']
        self.assertFalse(service._refusal_inference_completed(report, evaluation))
        self.rules[OUTERWEAR]['enabled'] = True
        report = add_outerwear(self.report(), self.directory, ['no', 'no', 'no'])
        self.assertFalse(self.completed(report))

    def test_clean_summary_cannot_override_an_explicit_incomplete_request(self):
        report = synthetic_report(self.directory, 'unknown')
        report['observations'][0]['calls'] = [{'completed': False}]
        with self.assertRaises(service.IncompleteAnalysisError) as caught:
            service.result_from_report(report, self.directory)
        self.assertFalse(caught.exception.inference_completed)

    def test_cash_only_recovery_requires_no_optional_pass_to_have_started(self):
        self.rules[OUTERWEAR]['enabled'] = True
        report = self.report()
        evaluation = service.evaluate_rules(report)
        self.assertTrue(service._refusal_inference_completed(report, evaluation, cash_pass_only=True))
        for key, value in (('outerwear_checks', []), ('outerwear_status', 'running'),
                           ('selected_outerwear_frame_ids', []), ('all_selected_outerwear_frames_processed', False)):
            with self.subTest(key=key):
                changed = deepcopy(report)
                changed[key] = value
                self.assertFalse(service._refusal_inference_completed(changed, evaluation, cash_pass_only=True))

    def test_live_positive_with_schema_failure_is_still_not_published_and_next_job_can_run(self):
        source = self.directory / 'input.mov'
        source.write_bytes(b'synthetic input; observation is stubbed')
        first = self.directory / 'failed'
        with patch('analyzer.service.check_ready', return_value=[]), patch(
                'analyzer.service.observe_video', side_effect=lambda path, out, **kw: self.report(out, 'positive')):
            with self.assertRaises(service.IncompleteAnalysisError) as caught:
                service.analyze_to_directory(source, first)
        self.assertTrue(caught.exception.inference_completed)
        self.assertFalse((first / 'result.json').exists())
        self.assertEqual(json.loads((first / 'job.json').read_text())['status'], 'failed')
        evaluation = json.loads((first / 'rule_evaluation.json').read_text())
        self.assertGreater(len(evaluation['candidates']), 0, 'Supported earlier evidence stays reviewable locally')
        with patch('analyzer.service.check_ready', return_value=[]), patch(
                'analyzer.service.observe_video', side_effect=lambda path, out, **kw: synthetic_report(out, 'negative')):
            result = service.analyze_to_directory(source, self.directory / 'next')
        self.assertEqual(result, {'verdict': 'no_violation', 'violations': []})


if __name__ == '__main__':
    unittest.main()
