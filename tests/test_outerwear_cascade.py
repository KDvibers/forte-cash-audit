"""Inactive cascade tests: generated JPEGs, fake clients/clocks, no inference/network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyzer import service
from analyzer.observations import ObservationDeadline
from analyzer.outerwear import (PRIMARY_PROMPT, VERIFICATION_PROMPT, evaluate_outerwear,
                                outerwear_policy_fingerprint)
from analyzer.outerwear_cascade import (CascadeOuterwearObserver, CascadeValidationError,
                                       canonical_runtime, cascade_policy_fingerprint, validate_cascade_record)
from tests.test_outerwear import Client, Clock, answer, inventory, primary
from tests.test_rule_service import image_bytes, synthetic_report


RULE = "removed_outerwear_in_cash_workplace"


def identity(size):
    return {"model_path": f"local/models/Qwen3VL-{size}B-Instruct-Q4_K_M.gguf", "build_info": "frozen-test-build",
            "model_alias": "forte-qwen", "model_ftype": "Q4_K - Medium",
            "modalities": {"vision": True, "audio": False}, "total_slots": 1,
            "context_size": 4096 if size == 4 else 2048, "weights_sha256": str(size)*64}


class RuntimeClient(Client):
    def __init__(self, size, responses=(), **kwargs):
        super().__init__(responses, **kwargs)
        self.runtime = identity(size)
        self.info_calls = 0
        self.info_error = None
        self.info_duration = 0

    def server_info(self):
        self.info_calls += 1
        if self.clock:
            self.clock.now += self.info_duration
        if self.info_error:
            raise self.info_error
        return deepcopy(self.runtime)


class ObserverWorker:
    """Narrow worker protocol deliberately has no ask_image method."""
    timeout = 20.0

    def __init__(self, responses):
        self.local_client = RuntimeClient(8, responses)
        self.calls = []
        self.mutate = lambda record: record

    def server_info(self):
        result = self.local_client.server_info()
        result["model_path"] = result["model_path"].rsplit("/", 1)[-1]
        return result

    def observe(self, frame, frame_id, *, deadline):
        from analyzer.outerwear import OuterwearObserver
        self.calls.append((frame_id, deadline))
        return self.mutate(OuterwearObserver(self.local_client).observe(frame, frame_id, deadline=deadline))


def pass_report(rows):
    return {"outerwear_checks": rows, "outerwear_policy_sha256": cascade_policy_fingerprint(),
            "outerwear_status": "completed", "outerwear_errors": [], "outerwear_budget_exceeded": False,
            "selected_outerwear_frame_ids": [r["frame_id"] for r in rows],
            "selected_outerwear_timestamps": [r["timestamp_sec"] for r in rows],
            "all_selected_outerwear_frames_processed": True}


class CascadeTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(timestamp_sec=4.0, jpeg=image_bytes(1))
        self.expected = {"filter4b": identity(4), "review8b": identity(8)}

    def pair(self, first, second=(), **kwargs):
        a, b = RuntimeClient(4, first, **kwargs), RuntimeClient(8, second, **kwargs)
        return CascadeOuterwearObserver(a, b, expected_runtimes=self.expected), a, b

    def observe(self, first, second=()):
        observer, a, b = self.pair(first, second)
        import time
        row = observer.observe(self.frame, "f1", deadline=time.perf_counter()+60)
        return row, a, b

    def validate(self, row, frame=None, frame_id="f1"):
        return validate_cascade_record(row, frame or self.frame, frame_id, expected_runtimes=self.expected)

    def test_clean_no_never_contacts_review_and_does_not_mutate_original(self):
        row, a, b = self.observe([primary("no")])
        self.assertEqual((row["state"], row["status"], row["selected_stage"]), ("no", "ok", "filter4b"))
        self.assertEqual((len(a.calls), len(b.calls), b.info_calls), (1, 0, 0))
        self.assertEqual(a.calls[0]["prompt"], PRIMARY_PROMPT)
        self.assertEqual(a.calls[0]["max_tokens"], 48)
        self.assertEqual(row["stages"][0]["record"]["calls"][0]["model_role"], "local_activity_4b")
        copied = self.validate(row)
        copied["calls"].clear()
        self.assertEqual(len(row["calls"]), 1)

    def test_clean_uncertain_and_full_yes_dispatch_to_original_8b(self):
        for first in ([primary("uncertain")], [primary("yes"), inventory()],
                      [primary("yes"), inventory("worn_or_carried", "none")]):
            for final in ("no", "uncertain", "yes"):
                with self.subTest(first=first, final=final):
                    second = [primary(final)] + ([inventory()] if final == "yes" else [])
                    row, a, b = self.observe(first, second)
                    self.assertEqual(row["state"], final)
                    self.assertEqual(row["selected_stage"], "review8b")
                    self.assertEqual(len(row["calls"]), len(a.calls)+len(b.calls))
                    self.assertEqual([c["prompt"] for c in b.calls], [PRIMARY_PROMPT]+([VERIFICATION_PROMPT] if final == "yes" else []))
                    self.assertEqual([c["max_tokens"] for c in b.calls], [48]+([96] if final == "yes" else []))
                    self.assertEqual(row["stages"][1]["record"]["calls"][0]["model_role"], "local_activity_4b")
                    self.assertEqual(row["calls"][-1]["model_role"], "team_activity_8b")
                    self.validate(row)

    def test_8b_verification_mapping_is_unchanged_without_4b_fallback(self):
        for location in ("resting_in_workplace", "worn_or_carried", "absent", "uncertain"):
            for feature in ("sleeve", "collar", "garment_opening", "none", "uncertain"):
                with self.subTest(location=location, feature=feature):
                    row, _, _ = self.observe([primary("yes"), inventory()], [primary("yes"), inventory(location, feature)])
                    expected = "yes" if location == "resting_in_workplace" and feature in {"sleeve", "collar", "garment_opening"} else "uncertain"
                    self.assertEqual(row["state"], expected)
                    self.validate(row)

    def test_primary_semantic_failures_never_dispatch_review(self):
        invalid = [None, {}, answer({}), primary("wrong"), answer({"removed_outerwear": "no", "extra": 1}),
                   answer({}, valid_json=False), primary("no") | {"finish_reason": "length"}]
        for response in invalid:
            with self.subTest(response=response):
                row, a, b = self.observe([response])
                self.assertEqual((row["status"], row["state"]), ("partial", "uncertain"))
                self.assertEqual((len(a.calls), len(b.calls), b.info_calls), (1, 0, 0))
                self.assertEqual(len(row["calls"]), 1)
                with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_bad_4b_verification_stops_with_both_original_calls(self):
        row, _, b = self.observe([primary("yes"), inventory(feature="buttons")])
        self.assertEqual((row["status"], len(row["calls"]), b.info_calls), ("partial", 2, 0))
        self.assertEqual(row["stages"][0]["record"]["errors"][0]["type"], "ObservationError")

    def test_8b_semantic_failure_keeps_all_stages_and_never_falls_back(self):
        row, _, b = self.observe([primary("yes"), inventory()], [primary("yes"), inventory(feature="buttons")])
        self.assertEqual((row["status"], row["state"], len(row["calls"])), ("partial", "uncertain", 4))
        self.assertEqual(len(b.calls), 2)
        with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_transport_errors_at_every_call_are_preserved_and_stop(self):
        for index in range(4):
            for error_type in (TimeoutError, OSError):
                with self.subTest(index=index, error_type=error_type):
                    replies = [primary("yes"), inventory(), primary("yes"), inventory()]
                    replies[index] = error_type("synthetic transport error")
                    observer, a, b = self.pair(replies[:2], replies[2:])
                    import time
                    if error_type is TimeoutError:
                        with self.assertRaises(ObservationDeadline) as caught:
                            observer.observe(self.frame, "f1", deadline=time.perf_counter()+60)
                        row = caught.exception.partial_record
                    else:
                        row = observer.observe(self.frame, "f1", deadline=time.perf_counter()+60)
                    self.assertEqual(len(a.calls)+len(b.calls), index+1)
                    self.assertEqual(len(row["calls"]), index+1)
                    self.assertFalse(row["calls"][-1]["completed"])
                    self.assertEqual(row["calls"][-1]["error_type"], error_type.__name__)
                    with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_deadlines_before_dispatch_between_stages_and_before_verification(self):
        cases = ("before_all", "4b_response_late", "review_runtime_late", "8b_primary_late")
        for case in cases:
            with self.subTest(case=case):
                clock = Clock()
                observer, a, b = self.pair([primary("uncertain")], [primary("yes"), inventory()], clock=clock)
                deadline = 100.0 if case == "before_all" else 101.0
                if case == "4b_response_late": a.durations = [1.0]
                if case == "review_runtime_late": b.info_duration = 1.0
                if case == "8b_primary_late": b.durations = [1.0]
                with patch("analyzer.outerwear_cascade.time.perf_counter", clock):
                    with self.assertRaises(ObservationDeadline) as caught:
                        observer.observe(self.frame, "f1", deadline=deadline)
                self.assertEqual(len(b.calls), 1 if case == "8b_primary_late" else 0)
                self.assertEqual(len(a.calls), 0 if case == "before_all" else 1)
                self.assertEqual(caught.exception.partial_record["state"], "uncertain")

    def test_late_final_validation_cannot_return_usable_record(self):
        clock = Clock()
        observer, a, b = self.pair([primary("no")], clock=clock)
        def late(*args, **kwargs):
            result = validate_cascade_record(*args, **kwargs)
            clock.now = 101
            return result
        with patch("analyzer.outerwear_cascade.time.perf_counter", clock), patch(
                "analyzer.outerwear_cascade.validate_cascade_record", side_effect=late):
            with self.assertRaises(ObservationDeadline) as caught:
                observer.observe(self.frame, "f1", deadline=101)
        self.assertEqual((caught.exception.partial_record["status"], b.info_calls), ("partial", 0))

    def test_deadline_expiry_while_preparing_runtime_call_does_not_dispatch(self):
        observer,a,b=self.pair([primary("no")])
        with patch("analyzer.outerwear_cascade.time.perf_counter",side_effect=[100.0,100.0,101.0,101.0]):
            with self.assertRaises(ObservationDeadline): observer.observe(self.frame,"f1",deadline=101.0)
        self.assertEqual((a.info_calls,b.info_calls,len(a.calls)),(0,0,0))

    def test_runtime_mismatch_prevents_that_stages_inference(self):
        for name in ("filter4b", "review8b"):
            for key, value in (("model_path", "wrong.gguf"), ("build_info", "changed"), ("total_slots", True),
                               ("context_size", 8192), ("weights_sha256", "x"*64)):
                with self.subTest(name=name, key=key):
                    observer, a, b = self.pair([primary("uncertain")], [primary("no")])
                    target = a if name == "filter4b" else b
                    target.runtime[key] = value
                    import time
                    row = observer.observe(self.frame, "f1", deadline=time.perf_counter()+60)
                    self.assertEqual((row["status"], len(target.calls)), ("partial", 0))
                    with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_runtime_exception_and_timeout_restoration(self):
        observer, a, b = self.pair([primary("no")])
        a.info_error = OSError("synthetic runtime failure")
        import time
        row = observer.observe(self.frame, "f1", deadline=time.perf_counter()+60)
        self.assertEqual((row["status"], len(a.calls), b.info_calls, a.timeout), ("partial", 0, 0, 20.0))

    def test_narrow_observer_worker_returns_original_record_without_synthesized_calls(self):
        import time
        a = RuntimeClient(4, [primary("uncertain")])
        worker = ObserverWorker([primary("yes"), inventory()])
        observer = CascadeOuterwearObserver(a, review_worker=worker, expected_runtimes=self.expected)
        deadline = time.perf_counter()+60
        row = observer.observe(self.frame, "f1", deadline=deadline)
        self.assertEqual(row["state"], "yes")
        self.assertEqual(worker.calls, [("f1", deadline)])
        self.assertEqual(len(row["stages"][1]["record"]["calls"]), 2)
        self.assertEqual([c["question"] for c in row["calls"]], ["removed_outerwear", "removed_outerwear", "outerwear_structure"])
        self.assertEqual(row["stages"][1]["runtime"]["model_path"], "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
        self.validate(row)

    def test_narrow_worker_not_contacted_for_no_or_filter_error(self):
        import time
        for response in (primary("no"), primary("wrong")):
            a, worker = RuntimeClient(4,[response]), ObserverWorker([])
            observer = CascadeOuterwearObserver(a,review_worker=worker,expected_runtimes=self.expected)
            observer.observe(self.frame,"f1",deadline=time.perf_counter()+60)
            self.assertEqual((worker.calls,worker.local_client.info_calls),([],0))

    def test_worker_identity_timeout_and_frame_record_mismatch(self):
        import time
        for case in ("identity", "timeout", "frame_id", "timestamp"):
            with self.subTest(case=case):
                a=RuntimeClient(4,[primary("uncertain")]); worker=ObserverWorker([primary("no")])
                if case=="identity": worker.local_client.runtime["model_path"]="other.gguf"
                if case=="timeout": worker.local_client.responses=[TimeoutError("synthetic worker timeout")]
                if case=="frame_id": worker.mutate=lambda r: dict(r,frame_id="another")
                if case=="timestamp": worker.mutate=lambda r: dict(r,timestamp_sec=99)
                observer=CascadeOuterwearObserver(a,review_worker=worker,expected_runtimes=self.expected)
                if case=="timeout":
                    with self.assertRaises(ObservationDeadline) as caught: observer.observe(self.frame,"f1",deadline=time.perf_counter()+60)
                    row=caught.exception.partial_record
                    self.assertEqual(len(row["calls"]),2)
                else: row=observer.observe(self.frame,"f1",deadline=time.perf_counter()+60)
                self.assertEqual(row["status"],"partial")
                self.assertEqual(len(worker.calls),0 if case=="identity" else 1)
                with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_worker_and_client_are_mutually_exclusive_and_runtime_is_canonical(self):
        a=RuntimeClient(4,[]); b=RuntimeClient(8,[]); worker=ObserverWorker([])
        with self.assertRaises(CascadeValidationError): CascadeOuterwearObserver(a,expected_runtimes=self.expected)
        with self.assertRaises(CascadeValidationError): CascadeOuterwearObserver(a,b,review_worker=worker,expected_runtimes=self.expected)
        self.assertEqual(canonical_runtime(identity(8))["model_path"],"Qwen3VL-8B-Instruct-Q4_K_M.gguf")

    def test_strict_ledger_rejects_missing_extra_reordered_or_forged_calls(self):
        original, _, _ = self.observe([primary("yes"), inventory()], [primary("yes"), inventory()])
        changes = [lambda r: r.update(calls=tuple(r["calls"])), lambda r: r["calls"].clear(), lambda r: r["calls"].pop(),
                   lambda r: r["calls"].append(deepcopy(r["calls"][-1])), lambda r: r["calls"].reverse(),
                   lambda r: r["calls"][0].update(completed=1), lambda r: r["calls"][0].update(model_role="team_activity_8b"),
                   lambda r: r["calls"][0].update(stage_call_index=3), lambda r: r["calls"][0].update(request_seconds=float("nan")),
                   lambda r: r["calls"][0].update(request_seconds=-1), lambda r: r["calls"][0].update(error_type="unknown"),
                   lambda r: r["stages"].reverse(), lambda r: r["stages"].pop(),
                   lambda r: r["stages"].append(deepcopy(r["stages"][-1])),
                   lambda r: r["stages"][0]["record"]["calls"].clear(),
                   lambda r: r["stages"][0]["record"]["calls"][0].update(completed=False),
                   lambda r: r["stages"][0]["record"]["calls"][0].update(question="unknown"),
                   lambda r: r["stages"][1]["record"]["calls"][1]["response"].update(response_text='{"forged":true}'),
                   lambda r: r["stages"][1]["record"].update(state="no"),
                   lambda r: r.update(state="no"), lambda r: r.update(selected_stage="filter4b"),
                   lambda r: r.update(policy_sha256=outerwear_policy_fingerprint()), lambda r: r.update(extra="hidden"),
                   lambda r: r["stages"][0]["record"].update(policy_sha256="unknown"),
                   lambda r: r["stages"][0]["record"].update(extra_call="hidden"),
                   lambda r: r["stages"][1]["runtime"].update(total_slots=True)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                row = deepcopy(original); change(row)
                with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_self_consistent_but_forged_original_reduction_is_rejected(self):
        from analyzer.outerwear_cascade import _flatten
        original, _, _ = self.observe([primary("yes"), inventory()], [primary("no")])
        for change in (lambda r: r["stages"][0]["record"].update(primary_state="no"),
                       lambda r: r["stages"][0]["record"].update(verification={"outerwear":"absent","garment_feature":"none"}),
                       lambda r: r["stages"][1]["record"].update(state="yes"),
                       lambda r: r["stages"][1]["record"]["calls"][0].update(completed=1)):
            row = deepcopy(original); change(row); row["calls"] = _flatten(row["stages"])
            with self.assertRaises(CascadeValidationError): self.validate(row)

    def test_frame_pts_bytes_and_stage_identity_are_bound(self):
        original, _, _ = self.observe([primary("uncertain")], [primary("no")])
        for change in (lambda r:r.update(frame_id="other"), lambda r:r.update(timestamp_sec=True),
                       lambda r:r.update(image_sha256="0"*64), lambda r:r["stages"][1].update(frame_sha256="0"*64),
                       lambda r:r["stages"][1]["record"].update(frame_id="other"),
                       lambda r:r["stages"][1]["record"].update(timestamp_sec=4.1)):
            row=deepcopy(original); change(row)
            with self.assertRaises(CascadeValidationError): self.validate(row)
        with self.assertRaises(CascadeValidationError):
            self.validate(original, SimpleNamespace(timestamp_sec=4.0,jpeg=image_bytes(2)))

    def test_forbidden_review_after_no_and_omitted_required_review_rejected(self):
        negative, _, _ = self.observe([primary("no")])
        reviewed, _, _ = self.observe([primary("uncertain")], [primary("no")])
        negative["stages"].append(deepcopy(reviewed["stages"][1]))
        with self.assertRaises(CascadeValidationError): self.validate(negative)
        reviewed["stages"].pop()
        with self.assertRaises(CascadeValidationError): self.validate(reviewed)

    def test_temporal_evaluator_receives_only_validated_states(self):
        def rows(states):
            result=[]
            for i,state in enumerate(states):
                frame=SimpleNamespace(timestamp_sec=float(i*4),jpeg=image_bytes(i))
                observer,_,_=self.pair([primary("uncertain")],[primary(state)]+([inventory()] if state=="yes" else []))
                import time
                row=observer.observe(frame,f"f{i}",deadline=time.perf_counter()+60)
                result.append(validate_cascade_record(row,frame,f"f{i}",expected_runtimes=self.expected))
            return result
        self.assertEqual(len(evaluate_outerwear(pass_report(rows(["yes","yes"])))['candidates']),1)
        for states in (["yes"], ["yes","uncertain","yes"]):
            evaluation=evaluate_outerwear(pass_report(rows(states)))
            self.assertEqual(evaluation['candidates'],[])
            self.assertFalse(evaluation['can_conclude_no_candidate'])
        self.assertTrue(evaluate_outerwear(pass_report(rows(["no","no"])))['can_conclude_no_candidate'])

    def test_existing_cash_guard_still_blocks_optional_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'synthetic.mov'; source.write_bytes(b'synthetic, decoder stubbed')
            rules=deepcopy(service.RULES); rules[RULE]['enabled']=True
            def incomplete(source,out,**kwargs):
                report=synthetic_report(out,'negative'); report['status']='partial'; return report
            with patch('analyzer.service.RULES',rules),patch('analyzer.service.check_ready',return_value=[]),patch(
                    'analyzer.service.observe_video',side_effect=incomplete),patch('analyzer.outerwear_pass.observe_outerwear_video') as optional:
                with self.assertRaises(service.IncompleteAnalysisError): service.analyze_to_directory(source,root/'out')
                optional.assert_not_called()
                self.assertFalse((root/'out/result.json').exists())

    def test_cash_positive_with_failed_cascade_is_not_published_or_released(self):
        bad,_,_=self.observe([primary('uncertain')],[primary('wrong')])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'synthetic.mov'; source.write_bytes(b'synthetic, decoder stubbed')
            rules=deepcopy(service.RULES); rules[RULE]['enabled']=True
            def cash(source,out,**kwargs):
                report=synthetic_report(out,'positive')
                report['settings']={'enable_actions':False,'max_motion_windows':0,'max_refinement_frames':0}
                for row in report['observations']:
                    row['calls']=[{'completed':True,'finish_reason':'stop','valid_json':True,'request_seconds':.01}]
                report['request_count']=len(report['observations'])
                return report
            failed=pass_report([bad]); failed.update(outerwear_status='partial',all_selected_outerwear_frames_processed=False,
                                                    outerwear_errors=[{'type':'ObservationError'}])
            with patch('analyzer.service.RULES',rules),patch('analyzer.service.check_ready',return_value=[]),patch(
                    'analyzer.service.observe_video',side_effect=cash),patch('analyzer.outerwear_pass.observe_outerwear_video',return_value=failed):
                with self.assertRaises(service.IncompleteAnalysisError) as caught:
                    service.analyze_to_directory(source,root/'out')
                self.assertFalse(caught.exception.inference_completed)
                self.assertFalse((root/'out/result.json').exists())
                saved=json.loads((root/'out/rule_evaluation.json').read_text(encoding='utf-8'))
                self.assertTrue(saved['checks']['cash_visible_after_person_disappears']['candidates'])

    def test_policy_is_distinct_and_current_production_profile_is_unchanged(self):
        self.assertEqual(outerwear_policy_fingerprint(),'7c4ebb8b8b842238750322b4c1e4c2121090e1a9ad3707ae299101852c35c5e1')
        self.assertNotEqual(cascade_policy_fingerprint(),outerwear_policy_fingerprint())
        self.assertFalse(service.RULES[RULE]['enabled'])
        self.assertEqual([url for url,_ in service._SERVERS],['http://127.0.0.1:8080','http://127.0.0.1:8081'])


if __name__ == '__main__':
    unittest.main()
