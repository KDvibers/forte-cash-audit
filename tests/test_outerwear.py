"""CPU-only checks for strict local responses and bounded temporal evidence."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyzer import outerwear
from analyzer.observations import ObservationDeadline
from analyzer.outerwear import OuterwearObserver, evaluate_outerwear, outerwear_policy_fingerprint


def answer(value, **overrides):
    result = {"observation": value, "finish_reason": "stop", "valid_json": True}
    result.update(overrides)
    return result


def primary(state):
    return answer({"removed_outerwear": state})


def inventory(state="resting_in_workplace", feature="sleeve"):
    return answer({"outerwear": state, "garment_feature": feature})


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Client:
    timeout = 20.0

    def __init__(self, responses=(), *, clock=None, durations=()):
        self.responses = list(responses)
        self.clock = clock
        self.durations = list(durations)
        self.calls = []

    def ask_image(self, frame, prompt, schema, **kwargs):
        self.calls.append({"frame": frame, "prompt": prompt, "schema": schema, **kwargs})
        if self.clock is not None:
            self.clock.now += self.durations.pop(0) if self.durations else 0
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def report(states, *, times=None):
    times = list(range(len(states))) if times is None else times
    rows = [{"frame_id": f"f{i}", "timestamp_sec": timestamp, "state": state,
             "status": "ok", "errors": []} for i, (state, timestamp) in enumerate(zip(states, times))]
    return {"outerwear_checks": rows, "outerwear_status": "completed", "outerwear_errors": [],
            "selected_outerwear_frame_ids": [row["frame_id"] for row in rows],
            "selected_outerwear_timestamps": list(times), "all_selected_outerwear_frames_processed": True}


class OuterwearObserverTests(unittest.TestCase):
    def observe(self, responses, **kwargs):
        client = Client(responses)
        result = OuterwearObserver(client).observe(SimpleNamespace(timestamp_sec=3.25), "real-frame", **kwargs)
        return result, client

    def test_negative_and_unknown_primary_do_not_issue_verification(self):
        for state in ("no", "uncertain"):
            with self.subTest(state=state):
                row, client = self.observe([primary(state)])
                self.assertEqual(row["state"], state)
                self.assertEqual((row["status"], row["errors"]), ("ok", []))
                self.assertEqual((row["frame_id"], row["timestamp_sec"]), ("real-frame", 3.25))
                self.assertEqual([call["max_tokens"] for call in client.calls], [48])
                self.assertNotIn("production_enabled", row)

    def test_yes_requires_inventory_and_recognizable_structure(self):
        for feature in ("sleeve", "collar", "garment_opening"):
            with self.subTest(feature=feature):
                row, client = self.observe([primary("yes"), inventory(feature=feature)])
                self.assertEqual(row["state"], "yes")
                self.assertEqual([call["max_tokens"] for call in client.calls], [48, 96])
                self.assertTrue(all(call["completed"] for call in row["calls"]))
                self.assertEqual(row["policy_sha256"], outerwear_policy_fingerprint())

    def test_disagreement_never_turns_into_yes_or_certain_no(self):
        for state in ("resting_in_workplace", "worn_or_carried", "absent", "uncertain"):
            for feature in ("sleeve", "collar", "garment_opening", "none", "uncertain"):
                if state == "resting_in_workplace" and feature in {"sleeve", "collar", "garment_opening"}:
                    continue
                with self.subTest(state=state, feature=feature):
                    row, client = self.observe([primary("yes"), inventory(state, feature)])
                    self.assertEqual(row["state"], "uncertain")
                    self.assertEqual(row["status"], "ok")
                    self.assertEqual(len(client.calls), 2)

    def test_invalid_primary_response_is_partial_unknown_and_stops(self):
        invalid = [None, {}, answer({}), answer({"removed_outerwear": "yes", "extra": "x"}),
                   primary("YES"), primary(True), primary(["yes"]),
                   answer({"removed_outerwear": "yes"}, finish_reason="length"),
                   answer({"removed_outerwear": "yes"}, valid_json=False),
                   answer({"removed_outerwear": "yes"}, valid_json=1)]
        for response in invalid:
            with self.subTest(response=response):
                row, client = self.observe([response])
                self.assertEqual((row["state"], row["status"]), ("uncertain", "partial"))
                self.assertEqual(len(row["errors"]), 1)
                self.assertEqual(len(client.calls), 1)
                self.assertTrue(row["calls"][0]["completed"], "The response returned; its meaning is still invalid")

    def test_invalid_inventory_cannot_confirm_primary_yes(self):
        for response in (answer({"outerwear": "resting_in_workplace"}),
                         answer({"outerwear": "resting_in_workplace", "garment_feature": "buttons"}),
                         answer({"outerwear": "resting_in_workplace", "garment_feature": "sleeve", "extra": 1}),
                         answer({"outerwear": "resting_in_workplace", "garment_feature": "sleeve"}, finish_reason="length")):
            with self.subTest(response=response):
                row, _ = self.observe([primary("yes"), response])
                self.assertEqual((row["state"], row["status"]), ("uncertain", "partial"))
                self.assertTrue(row["calls"][1]["completed"], "Returned transport is separate from a valid answer")

    def test_transport_error_keeps_unknown_and_actual_call_error(self):
        row, _ = self.observe([OSError("synthetic transport failure")])
        self.assertEqual((row["state"], row["status"]), ("uncertain", "partial"))
        self.assertEqual(row["calls"][0]["error_type"], "OSError")
        self.assertEqual(row["errors"][0]["feature"], "removed_outerwear")

    def test_expired_deadline_makes_no_request_and_retains_current_frame(self):
        clock, client = Clock(), Client()
        with patch("analyzer.outerwear.time.perf_counter", clock), self.assertRaises(ObservationDeadline) as caught:
            OuterwearObserver(client).observe(SimpleNamespace(timestamp_sec=3.25), "current", deadline=100)
        row = caught.exception.partial_record
        self.assertEqual((row["frame_id"], row["timestamp_sec"]), ("current", 3.25))
        self.assertEqual((row["state"], row["status"]), ("uncertain", "partial"))
        self.assertEqual(row["calls"], [])
        self.assertEqual(client.calls, [])

    def test_late_primary_yes_never_requests_inventory(self):
        clock = Clock()
        client = Client([primary("yes")], clock=clock, durations=[2])
        with patch("analyzer.outerwear.time.perf_counter", clock), self.assertRaises(ObservationDeadline) as caught:
            OuterwearObserver(client).observe(SimpleNamespace(timestamp_sec=0), "f", deadline=102)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["timeout"], 2)
        row = caught.exception.partial_record
        self.assertEqual(row["state"], "uncertain")
        self.assertFalse(row["calls"][0]["completed"])
        self.assertEqual(row["calls"][0]["response"], primary("yes"))

    def test_late_inventory_yes_cannot_be_accepted_and_remaining_budget_is_clamped(self):
        clock = Clock()
        client = Client([primary("yes"), inventory()], clock=clock, durations=[1, 2])
        with patch("analyzer.outerwear.time.perf_counter", clock), self.assertRaises(ObservationDeadline) as caught:
            OuterwearObserver(client).observe(SimpleNamespace(timestamp_sec=0), "f", deadline=103)
        row = caught.exception.partial_record
        self.assertEqual([call["timeout"] for call in client.calls], [3, 2])
        self.assertEqual((row["state"], row["status"]), ("uncertain", "partial"))
        self.assertEqual([call["completed"] for call in row["calls"]], [True, False])

    def test_client_timeout_preserves_partial_evidence_and_raises_deadline(self):
        client = Client([primary("yes"), TimeoutError("synthetic timeout")])
        with self.assertRaises(ObservationDeadline) as caught:
            OuterwearObserver(client).observe(SimpleNamespace(timestamp_sec=0), "f")
        self.assertEqual(caught.exception.partial_record["state"], "uncertain")
        self.assertEqual(len(caught.exception.partial_record["calls"]), 2)

    def test_invalid_input_rejected_before_calls(self):
        client = Client()
        observer = OuterwearObserver(client)
        for timestamp in (None, -1, True, float("nan"), float("inf"), "0"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                observer.observe(SimpleNamespace(timestamp_sec=timestamp), "f")
        for frame_id in (None, "", " ", 1):
            with self.subTest(frame_id=frame_id), self.assertRaises(ValueError):
                observer.observe(SimpleNamespace(timestamp_sec=0), frame_id)
        for deadline in (True, "102", float("inf"), float("nan")):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                observer.observe(SimpleNamespace(timestamp_sec=0), "f", deadline=deadline)
        self.assertEqual(client.calls, [])

    def test_policy_hash_tracks_both_questions_and_schemas(self):
        original = outerwear_policy_fingerprint()
        self.assertEqual(len(original), 64)
        for name in ("PRIMARY_PROMPT", "VERIFICATION_PROMPT"):
            with self.subTest(name=name), patch.object(outerwear, name, "different prompt"):
                self.assertNotEqual(outerwear_policy_fingerprint(), original)
        modified = deepcopy(outerwear.VERIFICATION_SCHEMA)
        modified["properties"]["garment_feature"]["enum"].append("other")
        with patch.object(outerwear, "VERIFICATION_SCHEMA", modified):
            self.assertNotEqual(outerwear_policy_fingerprint(), original)
        self.assertEqual(outerwear_policy_fingerprint(), original)


class OuterwearEvaluationTests(unittest.TestCase):
    def test_first_matching_timestamp_and_two_actual_support_ids(self):
        result = evaluate_outerwear(report(["no", "yes", "yes", "yes"], times=[0, 4, 8, 12]))
        self.assertEqual(len(result["candidates"]), 1)
        event = result["candidates"][0]
        self.assertEqual(event["source_rule_id"], "removed_outerwear_in_cash_workplace")
        self.assertEqual((event["timecode_sec"], event["confirmed_at_sec"], event["end_sec"]), (4, 8, 12))
        self.assertEqual(event["evidence_ids"], ["f1", "f2"])
        self.assertEqual(event["matching_frame_ids"], ["f1", "f2", "f3"])
        self.assertFalse(event["identity_established"])
        self.assertNotIn("verdict", result)
        self.assertFalse(result["can_conclude_no_candidate"])

    def test_unknown_breaks_confirmation_without_or_of_isolated_positives(self):
        result = evaluate_outerwear(report(["yes", "uncertain", "yes"]))
        self.assertEqual(result["candidates"], [])
        self.assertEqual([row["frame_id"] for row in result["blocked_candidates"]], ["f0", "f2"])
        self.assertFalse(result["can_conclude_no_candidate"])

    def test_unknown_and_one_negative_merge_confirmed_fragments_with_interruptions(self):
        for interruption in (["uncertain"], ["no"], ["no", "uncertain", "no"]):
            with self.subTest(interruption=interruption):
                result = evaluate_outerwear(report(["yes", "yes"] + interruption + ["yes", "yes"]))
                self.assertEqual(len(result["candidates"]), 1)
                event = result["candidates"][0]
                self.assertEqual(len(event["matching_segments"]), 2)
                self.assertEqual(event["evidence_ids"], ["f0", "f1"])
                self.assertEqual([row["state"] for row in event["interruptions"][0]["observations"]], interruption)

    def test_two_consecutive_no_separate_episodes(self):
        result = evaluate_outerwear(report(["yes", "yes", "no", "no", "yes", "yes"]))
        self.assertEqual([event["timecode_sec"] for event in result["candidates"]], [0, 4])
        self.assertEqual([event["evidence_ids"] for event in result["candidates"]], [["f0", "f1"], ["f4", "f5"]])

    def test_exact_gap_limit_can_confirm_but_larger_gap_cannot(self):
        self.assertEqual(len(evaluate_outerwear(report(["yes", "yes"], times=[0, 6]))["candidates"]), 1)
        result = evaluate_outerwear(report(["yes", "yes"], times=[0, 6.001]))
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["blocked_candidates"]), 2)
        self.assertIn("excessive_frame_gaps", result["coverage"]["negative_blockers"])

    def test_gap_separates_confirmed_fragments_instead_of_merging(self):
        result = evaluate_outerwear(report(["yes"] * 4, times=[0, 4, 11, 15]))
        self.assertEqual([event["timecode_sec"] for event in result["candidates"]], [0, 11])

    def test_technical_yes_is_unknown_and_technical_no_cannot_end_episode(self):
        data = report(["yes", "yes"])
        data["outerwear_checks"][1]["errors"] = [{"message": "failed verification"}]
        result = evaluate_outerwear(data)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["coverage"]["technical_partial_frame_ids"], ["f1"])
        data = report(["yes", "yes", "no", "no", "yes", "yes"])
        data["outerwear_checks"][3]["status"] = "partial"
        result = evaluate_outerwear(data)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["interruptions"][0]["observations"][1]["state"], "uncertain")

    def test_partial_report_preserves_supported_positive_but_never_negative(self):
        for states in (["yes", "yes"], ["no", "no"]):
            with self.subTest(states=states):
                data = report(states)
                data.update(outerwear_status="partial", outerwear_errors=[{"message": "deadline"}],
                            all_selected_outerwear_frames_processed=False)
                result = evaluate_outerwear(data)
                self.assertEqual(len(result["candidates"]), int(states[0] == "yes"))
                self.assertFalse(result["can_conclude_no_candidate"])

    def test_complete_clean_negative_is_scoped_to_observed_outerwear(self):
        result = evaluate_outerwear(report(["no", "no", "no"]))
        self.assertTrue(result["can_conclude_no_candidate"])
        self.assertEqual(result["coverage"]["negative_blockers"], [])
        self.assertEqual(result["kind"], "visual_outerwear_candidates_not_legal_verdict")

    def test_exact_selection_and_status_required_for_negative(self):
        changes = [
            {"selected_outerwear_frame_ids": ["wrong", "f1"]},
            {"selected_outerwear_timestamps": [0, 1.1]},
            {"all_selected_outerwear_frames_processed": False},
            {"all_selected_outerwear_frames_processed": 1},
            {"outerwear_status": "partial"}, {"outerwear_errors": ["technical failure"]},
            {"outerwear_budget_exceeded": True},
        ]
        for change in changes:
            with self.subTest(change=change):
                data = report(["no", "no"])
                data.update(change)
                self.assertFalse(evaluate_outerwear(data)["can_conclude_no_candidate"])
        for missing in ("selected_outerwear_frame_ids", "selected_outerwear_timestamps",
                        "all_selected_outerwear_frames_processed", "outerwear_status"):
            with self.subTest(missing=missing):
                data = report(["no", "no"])
                data.pop(missing)
                self.assertFalse(evaluate_outerwear(data)["can_conclude_no_candidate"])

    def test_common_cash_metadata_cannot_overwrite_outerwear_status(self):
        data = report(["no", "no"])
        data.update(status="partial", errors=["cash failure"], budget_exceeded=True)
        self.assertTrue(evaluate_outerwear(data)["can_conclude_no_candidate"])
        data.pop("outerwear_status")
        data.update(status="completed", errors=[], budget_exceeded=False)
        self.assertFalse(evaluate_outerwear(data)["can_conclude_no_candidate"])

    def test_empty_unknown_and_isolated_positive_cannot_be_negative(self):
        for states in ([], ["uncertain"], ["yes"], ["no", "yes", "no"]):
            with self.subTest(states=states):
                self.assertFalse(evaluate_outerwear(report(states))["can_conclude_no_candidate"])

    def test_invalid_rows_rejected_instead_of_treated_as_empty_negative(self):
        invalid = [("frame_id", "f0"), ("frame_id", None), ("timestamp_sec", 0),
                   ("timestamp_sec", float("nan")), ("timestamp_sec", True),
                   ("state", "YES"), ("state", []), ("status", []), ("status", "done"), ("errors", {})]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                data = report(["no", "no"])
                data["outerwear_checks"][1][key] = value
                with self.assertRaises(ValueError):
                    evaluate_outerwear(data)
        for change in ({"outerwear_checks": None}, {"outerwear_status": []}, {"outerwear_errors": {}},
                       {"selected_outerwear_frame_ids": ["f0", "f0"]},
                       {"selected_outerwear_timestamps": [0, float("inf")]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_outerwear({**report(["no", "no"]), **change})

    def test_evaluation_is_pure_and_cannot_accept_a_supplied_verdict(self):
        data = report(["yes", "yes", "uncertain", "yes", "yes"])
        data["verdict"] = "no_violation"
        before = deepcopy(data)
        result = evaluate_outerwear(data)
        self.assertEqual(data, before)
        self.assertEqual(len(result["candidates"]), 1)
        result["policy"]["confirmation_frames"] = 1
        self.assertEqual(evaluate_outerwear(data)["policy"]["confirmation_frames"], 2)


if __name__ == "__main__":
    unittest.main()
