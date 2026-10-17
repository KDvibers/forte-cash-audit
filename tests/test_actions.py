"""Action safety boundaries with synthetic images and no model/server access."""

from io import BytesIO
import json
import math
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.actions import ActionObserver
from analyzer.video import VideoFrame


def answer(value, *, finish="stop", valid=True):
    return {"observation": value, "finish_reason": finish, "valid_json": valid,
            "response_text": json.dumps(value), "usage": {"completion_tokens": 12}}


def frames_at(times=(1.0, 1.5, 2.5)):
    frames = []
    for index, timestamp in enumerate(times):
        image = Image.new("RGB", (160, 120), (30 + index * 50, 80, 140))
        stream = BytesIO()
        image.save(stream, format="JPEG")
        frames.append(VideoFrame(timestamp, stream.getvalue(), 160, 120))
    return frames


def found():
    return answer({"visible": "yes", "bbox": [250, 250, 750, 750]})


class FakeClock:
    def __init__(self, now=100):
        self.now = now

    def __call__(self):
        return self.now


class FakeClient:
    timeout = 30

    def __init__(self, *responses, clock=None, durations=()):
        self.responses = list(responses)
        self.calls = []
        self.clock = clock
        self.durations = iter(durations)

    def _ask(self, method, images, prompt, schema, **kwargs):
        self.calls.append({"method": method, "images": images, "prompt": prompt,
                           "schema": schema, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected extra model request")
        response = self.responses.pop(0)
        if self.clock is not None:
            self.clock.now += next(self.durations)
        if isinstance(response, BaseException):
            raise response
        return response

    def ask_image(self, *args, **kwargs):
        return self._ask("image", *args, **kwargs)

    def ask_sequence(self, *args, **kwargs):
        return self._ask("sequence", *args, **kwargs)


class ActionSafetyTests(unittest.TestCase):
    def assert_unknown(self, record):
        self.assertEqual(set(record["result"].values()), {"uncertain"})

    def test_raw_entry_claim_never_establishes_cash_transfer(self):
        for near in ("yes", "no", "uncertain"):
            for entry in ("yes", "no", "uncertain"):
                with self.subTest(near=near, entry=entry):
                    raw = answer({"paper_near_open_cabinet": near, "entry_visible": entry})
                    grounding, activity = FakeClient(found()), FakeClient(raw)
                    record = ActionObserver(grounding, activity).check(
                        frames_at(), region="cabinet", window_id="window")
                    self.assertEqual(record["status"], "ok")
                    self.assertEqual(record["result"]["cash_near_cabinet"], near)
                    self.assertEqual(record["raw_entry_visible"], entry)
                    self.assertEqual(record["result"]["cash_into_cabinet"], "uncertain")
                    self.assertEqual(record["result"]["counter_direction"], "uncertain")
                    self.assertEqual(record["kind"], "action_candidate_for_review")
                    self.assertEqual(record["feature_support"]["cash_into_cabinet"]["status"],
                                     "not_validated")
                    self.assertIs(record["calls"][-1]["response"], raw)

    def test_counter_interaction_requires_agreement_and_never_predicts_direction(self):
        for interaction in ("yes", "no", "uncertain"):
            for hand in ("top", "front", "away", "uncertain"):
                with self.subTest(interaction=interaction, hand=hand):
                    grounding = FakeClient(found())
                    activity = FakeClient(answer({"interaction": interaction}),
                                          answer({"hand_position": hand}),
                                          *(answer({"hand_position": "away"}) for _ in range(2)
                                            if interaction == "yes" and hand in ("away", "uncertain")))
                    record = ActionObserver(grounding, activity).check(
                        frames_at(), region="desk", window_id="window")
                    contact = "yes" if hand in ("top", "front") else "no" if hand == "away" else "uncertain"
                    expected = interaction if interaction == contact else "uncertain"
                    self.assertEqual(record["status"], "ok")
                    self.assertEqual(record["result"]["interaction"], expected)
                    self.assertEqual(record["raw_hand_position"], hand)
                    self.assertEqual(record["result"]["counter_direction"], "uncertain")
                    self.assertEqual(record["result"]["cash_into_cabinet"], "uncertain")
                    self.assertEqual(record["feature_support"]["counter_direction"]["status"],
                                     "not_validated")
                    self.assertNotIn("load", record["result"].values())
                    self.assertNotIn("remove", record["result"].values())
                    expected_calls = 4 if interaction == "yes" and hand in ("away", "uncertain") else 2
                    self.assertEqual(len(activity.calls), expected_calls)

    def test_missing_box_keeps_every_action_unknown_without_activity_request(self):
        for region in ("cabinet", "desk"):
            for visible in ("no", "uncertain"):
                with self.subTest(region=region, visible=visible):
                    grounding = FakeClient(answer({"visible": visible, "bbox": []}))
                    activity = FakeClient()
                    record = ActionObserver(grounding, activity).check(
                        frames_at(), region=region, window_id="window")
                    self.assertEqual(record["status"], "ok")
                    self.assert_unknown(record)
                    self.assertIsNone(record["object_bbox_normalized"])
                    self.assertEqual(activity.calls, [])

    def test_crops_keep_source_order_times_and_ground_correct_anchor(self):
        frames = frames_at()
        for region in ("cabinet", "desk"):
            with self.subTest(region=region):
                responses = ([answer({"paper_near_open_cabinet": "uncertain", "entry_visible": "uncertain"})]
                             if region == "cabinet" else
                             [answer({"interaction": "yes"}), answer({"hand_position": "top"})])
                grounding, activity = FakeClient(found()), FakeClient(*responses)
                record = ActionObserver(grounding, activity).check(frames, region=region, window_id="window")
                self.assertIs(grounding.calls[0]["images"], frames[0 if region == "cabinet" else 1])
                sequence = activity.calls[0]
                self.assertEqual(sequence["method"], "sequence")
                self.assertEqual([frame.timestamp_sec for frame in sequence["images"]], [1.0, 1.5, 2.5])
                self.assertEqual(record["start_sec"], 1.0)
                self.assertEqual(record["end_sec"], 2.5)
                self.assertEqual(record["max_anchor_gap_sec"], 1.0)
                self.assertEqual(len(record["crop_pixel_boxes"]), 3)
                if region == "desk":
                    self.assertIs(activity.calls[1]["images"], sequence["images"][1])

    def test_invalid_inputs_do_not_make_requests(self):
        for region, times in (("unknown", (1, 2, 3)), ("desk", ()), ("desk", (1, 2)),
                              ("cabinet", (1, 2, 3, 4)), ("desk", (1, 1, 2)),
                              ("cabinet", (2, 1, 3))):
            grounding, activity = FakeClient(), FakeClient()
            with self.subTest(region=region, times=times), self.assertRaises(ValueError):
                ActionObserver(grounding, activity).check(frames_at(times), region=region, window_id="bad")
            self.assertEqual(grounding.calls + activity.calls, [])

    def test_nonfinite_or_negative_times_are_rejected_before_grounding(self):
        for times in ((math.nan, 2, 3), (1, math.nan, 3), (1, 2, math.inf),
                      (-math.inf, 2, 3), (-1, 0, 1), (False, 1, 2),
                      (0, True, 2), ("0", 1, 2), (None, 1, 2)):
            grounding, activity = FakeClient(), FakeClient()
            with self.subTest(times=times), self.assertRaises(ValueError):
                ActionObserver(grounding, activity).check(frames_at(times), region="cabinet", window_id="bad")
            self.assertEqual(grounding.calls + activity.calls, [])


class ActionFailureTests(unittest.TestCase):
    assert_unknown = ActionSafetyTests.assert_unknown

    def test_invalid_cabinet_answers_are_partial_and_keep_raw_response(self):
        valid = {"paper_near_open_cabinet": "yes", "entry_visible": "yes"}
        responses = [answer(valid, valid=False), answer(valid, finish="length"),
                     answer(valid, finish=None), answer([]), answer(None),
                     answer({"paper_near_open_cabinet": "yes"}),
                     answer({**valid, "confidence": 1}),
                     answer({**valid, "entry_visible": True}),
                     answer({**valid, "paper_near_open_cabinet": "yes|no|uncertain"})]
        for raw in responses:
            with self.subTest(response=raw):
                record = ActionObserver(FakeClient(found()), FakeClient(raw)).check(
                    frames_at(), region="cabinet", window_id="window")
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual(record["errors"][0]["type"], "ObservationError")
                self.assertIs(record["calls"][-1]["response"], raw)
                self.assertNotIn("raw_entry_visible", record)

    def test_invalid_grounding_stops_activity_and_preserves_partial(self):
        for raw in (answer({"visible": "yes", "bbox": [900, 0, 100, 900]}),
                    answer({"visible": "yes", "bbox": []}),
                    answer({"visible": "uncertain", "bbox": [0, 0, 900, 900]}),
                    answer({"visible": "yes", "bbox": [0, 0, 900, 900]}, finish="length")):
            activity = FakeClient()
            with self.subTest(response=raw):
                record = ActionObserver(FakeClient(raw), activity).check(
                    frames_at(), region="desk", window_id="window")
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertIs(record["calls"][0]["response"], raw)
                self.assertEqual(activity.calls, [])

    def test_bad_counter_sequence_never_reaches_confirmation(self):
        for raw in (answer({"interaction": "yes"}, valid=False),
                    answer({"interaction": "yes"}, finish="length"),
                    answer({"interaction": "yes", "direction": "load"}),
                    answer({"interaction": True})):
            activity = FakeClient(raw)
            with self.subTest(response=raw):
                record = ActionObserver(FakeClient(found()), activity).check(
                    frames_at(), region="desk", window_id="window")
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual(len(activity.calls), 1)
                self.assertIs(record["calls"][-1]["response"], raw)

    def test_failed_hand_confirmation_cannot_promote_valid_interaction(self):
        for raw in (answer({"hand_position": "top"}, finish="length"),
                    answer({"hand_position": "front", "direction": "remove"}),
                    answer({"hand_position": "load"}), TimeoutError("hand timed out")):
            activity = FakeClient(answer({"interaction": "yes"}), raw)
            with self.subTest(response=raw):
                record = ActionObserver(FakeClient(found()), activity).check(
                    frames_at(), region="desk", window_id="window")
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual(record["calls"][1]["response"]["observation"], {"interaction": "yes"})
                self.assertEqual(len(record["calls"]), 3)
                if isinstance(raw, TimeoutError):
                    self.assertFalse(record["calls"][-1]["completed"])
                    self.assertEqual(record["calls"][-1]["error"]["type"], "TimeoutError")
                else:
                    self.assertIs(record["calls"][-1]["response"], raw)

    def test_transport_failure_keeps_attempt_and_completed_grounding(self):
        for error in (TimeoutError("sequence timed out"), RuntimeError("HTTP 500")):
            with self.subTest(error=error):
                record = ActionObserver(FakeClient(found()), FakeClient(error)).check(
                    frames_at(), region="cabinet", window_id="window")
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual([call["completed"] for call in record["calls"]], [True, False])
                self.assertEqual(record["calls"][-1]["error"]["type"], type(error).__name__)
                self.assertEqual(record["errors"][0]["message"], str(error))

    def test_expired_deadline_makes_no_request(self):
        for deadline in (99, 100):
            grounding, activity = FakeClient(), FakeClient()
            with self.subTest(deadline=deadline), patch("analyzer.actions.time.perf_counter", return_value=100):
                record = ActionObserver(grounding, activity).check(
                    frames_at(), region="desk", window_id="window", deadline=deadline)
            self.assertEqual(record["status"], "partial")
            self.assert_unknown(record)
            self.assertEqual(record["calls"], [])
            self.assertEqual(grounding.calls + activity.calls, [])
            self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")

    def test_future_deadline_passes_remaining_timeout_and_allows_timely_answer(self):
        clock = FakeClock()
        grounding = FakeClient(found(), clock=clock, durations=[1])
        activity = FakeClient(answer({"interaction": "yes"}), answer({"hand_position": "front"}),
                              clock=clock, durations=[1, 1])
        with patch("analyzer.actions.time.perf_counter", clock):
            record = ActionObserver(grounding, activity).check(
                frames_at(), region="desk", window_id="window", deadline=105)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["result"]["interaction"], "yes")
        self.assertEqual([call["timeout"] for call in grounding.calls + activity.calls], [5, 4, 3])

    def test_deadline_never_extends_client_timeout(self):
        grounding, activity = FakeClient(found()), FakeClient(answer({"interaction": "no"}),
                                                             answer({"hand_position": "away"}))
        with patch("analyzer.actions.time.perf_counter", return_value=100):
            record = ActionObserver(grounding, activity).check(
                frames_at(), region="desk", window_id="window", deadline=1000)
        self.assertEqual(record["status"], "ok")
        self.assertEqual([call["timeout"] for call in grounding.calls + activity.calls], [30, 30, 30])

    def test_late_grounding_sequence_or_confirmation_never_becomes_fact(self):
        for late_stage in ("ground", "sequence", "hand"):
            with self.subTest(late_stage=late_stage):
                clock = FakeClock()
                grounding = FakeClient(found(), clock=clock, durations=[6 if late_stage == "ground" else 1])
                activity = FakeClient(answer({"interaction": "yes"}), answer({"hand_position": "top"}),
                                      clock=clock, durations=[5 if late_stage == "sequence" else 1, 4])
                with patch("analyzer.actions.time.perf_counter", clock):
                    record = ActionObserver(grounding, activity).check(
                        frames_at(), region="desk", window_id="window", deadline=105)
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
                self.assertFalse(record["calls"][-1]["completed"])
                self.assertIn("response", record["calls"][-1])
                self.assertEqual(len(record["calls"]), {"ground": 1, "sequence": 2, "hand": 3}[late_stage])

    def test_late_cabinet_yes_stays_only_in_raw_call(self):
        clock = FakeClock()
        grounding = FakeClient(found(), clock=clock, durations=[1])
        raw = answer({"paper_near_open_cabinet": "yes", "entry_visible": "yes"})
        activity = FakeClient(raw, clock=clock, durations=[5])
        with patch("analyzer.actions.time.perf_counter", clock):
            record = ActionObserver(grounding, activity).check(
                frames_at(), region="cabinet", window_id="window", deadline=105)
        self.assertEqual(record["status"], "partial")
        self.assert_unknown(record)
        self.assertIs(record["calls"][-1]["response"], raw)
        self.assertNotIn("raw_entry_visible", record)


class CounterBoxActionTests(unittest.TestCase):
    assert_unknown = ActionSafetyTests.assert_unknown

    def test_stable_counter_box_skips_grounding_and_uses_same_crop_geometry(self):
        frames = frames_at()
        box = [250, 250, 750, 750]
        grounding = FakeClient()
        activity = FakeClient(answer({"interaction": "yes"}), answer({"hand_position": "top"}))
        record = ActionObserver(grounding, activity).check(
            frames, region="desk", window_id="counter", counter_box=box)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(grounding.calls, [])
        self.assertEqual(record["grounding_source"], "counter_discovery_cluster")
        self.assertEqual(record["object_bbox_normalized"], box)
        self.assertEqual(box, [250, 250, 750, 750])
        self.assertEqual(record["crop_pixel_boxes"], [[8, 6, 152, 114]] * 3)
        self.assertEqual([call["question"] for call in record["calls"]], ["counter_sequence", "counter_hand"])
        self.assertEqual([f.timestamp_sec for f in activity.calls[0]["images"]], [1.0, 1.5, 2.5])
        self.assertEqual(record["result"]["interaction"], "yes")
        self.assertEqual(record["result"]["counter_direction"], "uncertain")

    def test_valid_geometry_does_not_replace_action_agreement(self):
        activity = FakeClient(answer({"interaction": "yes"}),
                              *(answer({"hand_position": "away"}) for _ in range(3)))
        record = ActionObserver(FakeClient(), activity).check(
            frames_at(), region="desk", window_id="counter", counter_box=[250, 250, 750, 750])
        self.assertEqual(record["status"], "ok")
        self.assert_unknown(record)

    def test_invalid_counter_box_is_partial_without_any_model_call(self):
        for box in ([], [900, 100, 200, 800], [0, 0, 1001, 900], [True, 0, 900, 900],
                    [0.0, 0, 900, 900], [0, 0, 900], "0,0,900,900", [100, 100, 101, 101]):
            grounding, activity = FakeClient(), FakeClient()
            with self.subTest(box=box):
                record = ActionObserver(grounding, activity).check(
                    frames_at(), region="desk", window_id="counter", counter_box=box)
                self.assertEqual(record["status"], "partial")
                self.assert_unknown(record)
                self.assertEqual(grounding.calls + activity.calls, [])
                self.assertEqual(record["calls"], [])
                self.assertEqual(record["errors"][0]["type"], "ObservationError")

    def test_counter_box_cannot_replace_cabinet_grounding(self):
        grounding = FakeClient(answer({"visible": "no", "bbox": []}))
        activity = FakeClient()
        record = ActionObserver(grounding, activity).check(
            frames_at(), region="cabinet", window_id="cabinet", counter_box=[250, 250, 750, 750])
        self.assertEqual(record["status"], "ok")
        self.assertEqual(len(grounding.calls), 1)
        self.assertEqual(record["grounding_source"], "sequence_anchor")
        self.assertIsNone(record["object_bbox_normalized"])
        self.assert_unknown(record)
        self.assertEqual(activity.calls, [])

    def test_missing_counter_box_uses_existing_grounding_path(self):
        grounding = FakeClient(found())
        activity = FakeClient(answer({"interaction": "no"}), answer({"hand_position": "away"}))
        record = ActionObserver(grounding, activity).check(
            frames_at(), region="desk", window_id="counter", counter_box=None)
        self.assertEqual(record["grounding_source"], "sequence_anchor")
        self.assertEqual(len(grounding.calls), 1)
        self.assertEqual(record["result"]["interaction"], "no")

    def test_preexpired_or_late_response_remains_unknown_with_stable_geometry(self):
        for initial, duration in ((105, 0), (100, 6)):
            clock = FakeClock(initial)
            grounding = FakeClient()
            activity = FakeClient(answer({"interaction": "yes"}), clock=clock, durations=[duration])
            with self.subTest(initial=initial), patch("analyzer.actions.time.perf_counter", clock):
                record = ActionObserver(grounding, activity).check(
                    frames_at(), region="desk", window_id="counter", deadline=105,
                    counter_box=[250, 250, 750, 750])
            self.assertEqual(record["status"], "partial")
            self.assert_unknown(record)
            self.assertEqual(grounding.calls, [])
            self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
            if initial == 105:
                self.assertEqual(activity.calls, [])
            else:
                self.assertEqual(len(activity.calls), 1)
                self.assertFalse(record["calls"][0]["completed"])
                self.assertEqual(record["calls"][0]["response"]["observation"], {"interaction": "yes"})


class CounterEdgeContactTests(unittest.TestCase):
    assert_unknown = ActionSafetyTests.assert_unknown

    def check(self, client, *, frames=None, deadline=None):
        return ActionObserver(FakeClient(), client).check(
            frames_at((2.0, 2.4, 3.2)) if frames is None else frames,
            region="desk", window_id="edge-contact", deadline=deadline,
            counter_box=[250, 250, 750, 750])

    def test_first_anchor_contact_recovers_sequence_and_stops_before_last(self):
        for middle in ("away", "uncertain"):
            for edge in ("top", "front"):
                with self.subTest(middle=middle, edge=edge):
                    client = FakeClient(answer({"interaction": "yes"}),
                                        answer({"hand_position": middle}), answer({"hand_position": edge}))
                    record = self.check(client)
                    self.assertEqual(record["status"], "ok")
                    self.assertEqual(record["result"]["interaction"], "yes")
                    self.assertEqual(record["result"]["counter_direction"], "uncertain")
                    self.assertEqual(record["result"]["cash_into_cabinet"], "uncertain")
                    self.assertEqual(record["contact_checks"], [
                        {"timestamp_sec": 2.4, "hand_position": middle},
                        {"timestamp_sec": 2.0, "hand_position": edge}])
                    self.assertEqual(record["contact_evidence_sec"], [2.0])
                    self.assertEqual(record["raw_hand_position"], middle)
                    self.assertEqual([call["images"].timestamp_sec for call in client.calls[1:]], [2.4, 2.0])
                    self.assertEqual(len(client.calls), 3)

    def test_last_anchor_contact_is_checked_only_after_unconfirmed_first(self):
        for first in ("away", "uncertain"):
            for edge in ("top", "front"):
                with self.subTest(first=first, edge=edge):
                    client = FakeClient(answer({"interaction": "yes"}), answer({"hand_position": "away"}),
                                        answer({"hand_position": first}), answer({"hand_position": edge}))
                    record = self.check(client)
                    self.assertEqual(record["status"], "ok")
                    self.assertEqual(record["result"]["interaction"], "yes")
                    self.assertEqual(record["result"]["counter_direction"], "uncertain")
                    self.assertEqual(record["contact_evidence_sec"], [3.2])
                    self.assertEqual([row["timestamp_sec"] for row in record["contact_checks"]], [2.4, 2.0, 3.2])
                    self.assertEqual([call["images"].timestamp_sec for call in client.calls[1:]], [2.4, 2.0, 3.2])
                    self.assertEqual(len(client.calls), 4)

    def test_three_unconfirmed_anchors_never_promote_or_request_more_images(self):
        for hands in (("away", "away", "away"), ("uncertain", "away", "uncertain"),
                      ("away", "uncertain", "away")):
            with self.subTest(hands=hands):
                client = FakeClient(answer({"interaction": "yes"}),
                                    *(answer({"hand_position": hand}) for hand in hands))
                record = self.check(client)
                self.assertEqual(record["status"], "ok")
                self.assert_unknown(record)
                self.assertEqual(record["contact_evidence_sec"], [])
                self.assertEqual(len(record["contact_checks"]), 3)
                self.assertEqual(len(client.calls), 4)

    def test_only_yes_sequence_with_unconfirmed_middle_can_query_edges(self):
        for sequence, hand in (("no", "away"), ("no", "top"), ("no", "uncertain"),
                               ("uncertain", "away"), ("uncertain", "front"),
                               ("uncertain", "uncertain"), ("yes", "top"), ("yes", "front")):
            with self.subTest(sequence=sequence, hand=hand):
                client = FakeClient(answer({"interaction": sequence}), answer({"hand_position": hand}))
                record = self.check(client)
                self.assertEqual(record["status"], "ok")
                self.assertEqual(len(client.calls), 2)
                self.assertEqual(len(record["contact_checks"]), 1)
                self.assertEqual(record["result"]["counter_direction"], "uncertain")
                if sequence != "yes":
                    self.assertNotEqual(record["result"]["interaction"], "yes")
                else:
                    self.assertEqual(record["contact_evidence_sec"], [2.4])

    def test_invalid_or_timed_out_edge_cannot_promote_sequence_yes(self):
        for edge_index in (0, 2):
            for bad in (answer({"hand_position": "top"}, finish="length"),
                        answer({"hand_position": "front"}, valid=False),
                        answer({"hand_position": "top", "direction": "load"}),
                        answer({"hand_position": "load"}), TimeoutError("edge timed out")):
                with self.subTest(edge_index=edge_index, bad=bad):
                    before = [answer({"interaction": "yes"}), answer({"hand_position": "away"})]
                    if edge_index == 2:
                        before.append(answer({"hand_position": "uncertain"}))
                    client = FakeClient(*before, bad)
                    record = self.check(client)
                    self.assertEqual(record["status"], "partial")
                    self.assert_unknown(record)
                    self.assertEqual(len(client.calls), 3 if edge_index == 0 else 4)
                    self.assertEqual(len(record["contact_checks"]), 1 if edge_index == 0 else 2)
                    self.assertFalse(record.get("contact_evidence_sec"))
                    self.assertEqual(record["calls"][-1]["question"], "counter_hand_edge")
                    if isinstance(bad, TimeoutError):
                        self.assertFalse(record["calls"][-1]["completed"])
                        self.assertEqual(record["errors"][0]["type"], "TimeoutError")
                    else:
                        self.assertIs(record["calls"][-1]["response"], bad)
                        self.assertEqual(record["errors"][0]["type"], "ObservationError")

    def test_late_first_or_last_contact_keeps_raw_response_without_accepting_evidence(self):
        for edge_index in (0, 2):
            clock = FakeClock()
            responses = [answer({"interaction": "yes"}), answer({"hand_position": "away"})]
            durations = [1, 1]
            if edge_index == 2:
                responses.append(answer({"hand_position": "away"}))
                durations.append(1)
            raw = answer({"hand_position": "top"})
            client = FakeClient(*responses, raw, clock=clock, durations=durations + [4])
            with self.subTest(edge_index=edge_index), patch("analyzer.actions.time.perf_counter", clock):
                record = self.check(client, deadline=105)
            self.assertEqual(record["status"], "partial")
            self.assert_unknown(record)
            self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
            self.assertIs(record["calls"][-1]["response"], raw)
            self.assertFalse(record["calls"][-1]["completed"])
            self.assertFalse(record.get("contact_evidence_sec"))
            self.assertEqual(len(record["contact_checks"]), 1 if edge_index == 0 else 2)
            self.assertEqual([call["timeout"] for call in client.calls], [5, 4, 3] if edge_index == 0 else [5, 4, 3, 2])

    def test_middle_answer_at_deadline_is_not_accepted_and_sends_no_edge_request(self):
        clock = FakeClock()
        client = FakeClient(answer({"interaction": "yes"}), answer({"hand_position": "away"}),
                            clock=clock, durations=[1, 4])
        with patch("analyzer.actions.time.perf_counter", clock):
            record = self.check(client, deadline=105)
        self.assertEqual(record["status"], "partial")
        self.assert_unknown(record)
        self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("contact_checks", record)
        self.assertEqual(record["calls"][-1]["response"]["observation"], {"hand_position": "away"})
        self.assertFalse(record["calls"][-1]["completed"])


if __name__ == "__main__":
    unittest.main()
