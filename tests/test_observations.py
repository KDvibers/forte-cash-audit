"""Failure-boundary tests using in-memory images and no model/server access."""

from io import BytesIO
import base64
import json
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from analyzer.features import CASH_FEATURES
from analyzer.local_client import LocalModelClient
from analyzer.observations import (
    ImageObserver, ObservationDeadline, ObservationError,
    BOX_SCHEMA, CONTACT_PROMPT, CONTACT_SCHEMA, DESK_PROMPT, DOOR_PROMPT, DOOR_SCHEMA, FEATURE_SUPPORT,
    DOORWAY_PROMPT, DOORWAY_SCHEMA,
    TABLE_VIEW_PROMPT, TABLE_VIEW_SCHEMA, CASH_RECHECK_PROMPT,
    PERSON_SCHEMA, POSITION_SCHEMA, WORKTOP_PROMPT, WORKTOP_SCHEMA,
    crop_object, policy_fingerprint, read_box, read_enum,
)
from analyzer.video import VideoFrame


def answer(value, *, finish="stop", valid=True):
    return {"observation": value, "finish_reason": finish, "valid_json": valid}


def synthetic_frame():
    image = Image.new("RGB", (160, 120), "red")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 0, 159, 59), fill=(0, 255, 0))
    draw.rectangle((0, 60, 79, 119), fill=(0, 0, 255))
    draw.rectangle((80, 60, 159, 119), fill=(255, 255, 0))
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=100, subsampling=0)
    return VideoFrame(12345.678, stream.getvalue(), 160, 120)


class FakeClient:
    timeout = 30

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def ask_image(self, picture, prompt, schema, **kwargs):
        self.calls.append((picture, prompt, schema, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected extra model request")
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeClock:
    def __init__(self, now=100):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TimedFakeClient(FakeClient):
    def __init__(self, clock, durations, *responses):
        super().__init__(*responses)
        self.clock = clock
        self.durations = iter(durations)

    def ask_image(self, *args, **kwargs):
        duration = next(self.durations)
        try:
            return super().ask_image(*args, **kwargs)
        finally:
            self.clock.advance(duration)


class StrictResponseTests(unittest.TestCase):
    def test_complete_looking_json_with_non_stop_finish_is_rejected(self):
        for finish in (None, "length", "error", "tool_calls"):
            response = answer({"person_in_work_area": "yes"}, finish=finish)
            with self.subTest(finish=finish), self.assertRaises(ObservationError):
                read_enum(response, "person_in_work_area", ("yes", "no", "uncertain"))

    def test_enum_cannot_copy_alternatives_add_keys_or_coerce_values(self):
        for value in ({}, {"person_in_work_area": "yes|no|uncertain"},
                      {"person_in_work_area": True}, {"person_in_work_area": ["yes"]},
                      {"person_in_work_area": "yes", "verdict": "violation"}):
            with self.subTest(value=value), self.assertRaises(ObservationError):
                read_enum(answer(value), "person_in_work_area", ("yes", "no", "uncertain"))

    def test_invalid_json_flag_and_non_objects_are_not_valid_observations(self):
        for response in (answer({"door_state": "open"}, valid=False),
                         answer(None), answer([]), answer("open")):
            with self.subTest(response=response), self.assertRaises(ObservationError):
                read_enum(response, "door_state", ("open", "closed", "uncertain"))

    def test_bbox_rejects_bools_floats_range_length_and_orientation_errors(self):
        for box in ([False, 0, 900, 900], [0.0, 0, 900, 900],
                    [-1, 0, 900, 900], [0, 0, 1001, 900],
                    [], [0, 0, 900], [0, 0, 900, 900, 1],
                    [900, 0, 0, 900], [0, 900, 900, 0], [100, 0, 100, 900],
                    "0,0,900,900", None):
            with self.subTest(box=box), self.assertRaises(ObservationError):
                read_box(answer({"visible": "yes", "bbox": box}))

    def test_unseen_or_uncertain_cabinet_cannot_carry_a_box(self):
        for visible in ("no", "uncertain"):
            with self.subTest(visible=visible), self.assertRaises(ObservationError):
                read_box(answer({"visible": visible, "bbox": [0, 0, 900, 900]}))
            self.assertEqual(read_box(answer({"visible": visible, "bbox": []})),
                             (visible, None))

    def test_bbox_rejects_unknown_visibility_unexpected_keys_and_truncation(self):
        for response in (
            answer({"visible": "probably", "bbox": []}),
            answer({"visible": "yes", "bbox": [0, 0, 900, 900], "confidence": 1}),
            answer({"visible": "yes", "bbox": [0, 0, 900, 900]}, finish="length"),
        ):
            with self.subTest(response=response), self.assertRaises(ObservationError):
                read_box(response)


class CropTests(unittest.TestCase):
    def test_crop_preserves_time_source_content_orientation_and_pixel_scale(self):
        original = synthetic_frame()
        original_bytes = original.jpeg
        crop, box = crop_object(original, [250, 250, 750, 750], 0)
        self.assertEqual(box, [40, 30, 120, 90])
        self.assertEqual((crop.width, crop.height), (80, 60))
        self.assertEqual(crop.timestamp_sec, original.timestamp_sec)
        self.assertEqual(original.jpeg, original_bytes)
        with Image.open(BytesIO(crop.jpeg)) as decoded:
            for point, expected in (((10, 10), (255, 0, 0)),
                                    ((70, 10), (0, 255, 0)),
                                    ((10, 50), (0, 0, 255)),
                                    ((70, 50), (255, 255, 0))):
                with self.subTest(point=point):
                    actual = decoded.getpixel(point)
                    self.assertTrue(all(abs(a-b) < 20 for a, b in zip(actual, expected)),
                                    (actual, expected))

    def test_padding_clips_to_source_without_inventing_border_pixels(self):
        frame = synthetic_frame()
        crop, box = crop_object(frame, [0, 0, 1000, 1000], 0.4)
        self.assertEqual(box, [0, 0, 160, 120])
        self.assertEqual((crop.width, crop.height), (160, 120))

    def test_dimension_mismatch_and_tiny_detection_are_rejected(self):
        frame = synthetic_frame()
        wrong = VideoFrame(frame.timestamp_sec, frame.jpeg, 161, 120)
        with self.assertRaises(ValueError):
            crop_object(wrong, [0, 0, 1000, 1000], 0)
        with self.assertRaises(ObservationError):
            crop_object(frame, [100, 100, 101, 101], 0.4)


class ObserverFailureTests(unittest.TestCase):
    def test_full_frame_mode_asks_two_questions_without_grounding_or_cropping(self):
        for state in ("open", "closed", "uncertain", "not_visible"):
            client = FakeClient(answer({"person_in_work_area": "yes"}), answer({"door_state": state}))
            frame = synthetic_frame()
            result = ImageObserver(client, cabinet_mode="full_frame").observe(frame, "f0")
            with self.subTest(state=state):
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["states"]["cabinet_door"], state)
                self.assertEqual(len(client.calls), 2)
                self.assertTrue(all(call[0] is frame for call in client.calls))
                self.assertEqual(client.calls[1][1:3], (DOOR_PROMPT, DOOR_SCHEMA))
                self.assertEqual([call["question"] for call in result["calls"]],
                                 ["person", "cabinet_door_full_frame"])
                self.assertIsNone(result["cabinet_bbox_normalized"])
                self.assertEqual(result["crop_checks"], [])
                self.assertEqual(result["cabinet_consistency"], "single_full_frame")

    def test_full_frame_mode_rejects_invalid_extra_keys_and_truncated_answers(self):
        for bad in (answer({"door_state": "open|closed"}),
                    answer({"door_state": "closed", "confidence": 1}),
                    answer({"door_state": "closed"}, finish="length")):
            client = FakeClient(answer({"person_in_work_area": "yes"}), bad)
            result = ImageObserver(client, cabinet_mode="full_frame").observe(synthetic_frame(), "f0")
            with self.subTest(response=bad):
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["states"]["person_in_work_area"], "yes")
                self.assertEqual(result["states"]["cabinet_door"], "uncertain")
                self.assertEqual(result["errors"][0]["feature"], "cabinet_door")
                self.assertEqual(len(client.calls), 2)

    def test_bad_person_answer_does_not_erase_valid_cabinet_observation(self):
        client = FakeClient(answer({"person_in_work_area": "yes|no"}),
                            answer({"visible": "no", "bbox": []}))
        result = ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["states"], {
            "person_in_work_area": "uncertain", "cabinet_door": "not_visible"})
        self.assertEqual(result["errors"][0]["feature"], "person_in_work_area")

    def test_transport_error_does_not_turn_into_person_absence(self):
        client = FakeClient(RuntimeError("HTTP 500"), answer({"visible": "uncertain", "bbox": []}))
        result = ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["states"]["person_in_work_area"], "uncertain")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result["calls"]), 2)
        self.assertEqual([call["completed"] for call in result["calls"]], [False, True])
        self.assertEqual(result["calls"][0]["error_type"], "RuntimeError")
        self.assertIn("HTTP 500", result["calls"][0]["error"])

    def test_invalid_localization_never_reaches_crop_questions(self):
        client = FakeClient(answer({"person_in_work_area": "yes"}),
                            answer({"visible": "yes", "bbox": [800, 0, 200, 900]}))
        result = ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["states"]["person_in_work_area"], "yes")
        self.assertEqual(result["states"]["cabinet_door"], "uncertain")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["crop_checks"], [])

    def test_disagreeing_or_ambiguous_crops_do_not_produce_a_known_door_state(self):
        for pair in (("open", "closed"), ("open", "uncertain"),
                     ("not_visible", "not_visible"), ("uncertain", "uncertain")):
            client = FakeClient(answer({"person_in_work_area": "no"}),
                                answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                                *(answer({"door_state": state}) for state in pair))
            result = ImageObserver(client).observe(synthetic_frame(), "f0")
            with self.subTest(pair=pair):
                self.assertEqual(result["states"]["cabinet_door"], "uncertain")
                self.assertEqual(len(result["crop_checks"]), 2)

    def test_failed_second_crop_cannot_promote_first_crop_to_known_state(self):
        client = FakeClient(answer({"person_in_work_area": "no"}),
                            answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                            answer({"door_state": "open"}),
                            answer({"door_state": "open"}, finish="length"))
        result = ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["states"]["cabinet_door"], "uncertain")
        self.assertEqual(len(result["crop_checks"]), 1)

    def test_identical_clamped_crops_are_not_two_confirming_observations(self):
        client = FakeClient(answer({"person_in_work_area": "yes"}),
                            answer({"visible": "yes", "bbox": [0, 0, 1000, 1000]}),
                            answer({"door_state": "open"}))
        result = ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(result["states"]["cabinet_door"], "uncertain")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(result["crop_checks"]), 1)
        self.assertEqual(result["errors"], [])

    def test_timeout_preserves_partial_record_and_failed_attempt(self):
        client = FakeClient(answer({"person_in_work_area": "yes"}), TimeoutError("timed out"))
        with self.assertRaises(ObservationDeadline) as caught:
            ImageObserver(client).observe(synthetic_frame(), "f0")
        self.assertEqual(len(client.calls), 2)
        partial = caught.exception.partial_record
        self.assertEqual(partial["frame_id"], "f0")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["states"], {
            "person_in_work_area": "yes", "cabinet_door": "uncertain"})
        self.assertEqual([call["completed"] for call in partial["calls"]], [True, False])
        self.assertEqual(partial["calls"][1]["error_type"], "TimeoutError")
        self.assertEqual(partial["errors"][0]["feature"], "cabinet_door")

    def test_exhausted_deadline_makes_no_model_request(self):
        client = FakeClient()
        with patch("analyzer.observations.time.perf_counter", return_value=100):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(client).observe(synthetic_frame(), "f0", deadline=99)
        self.assertEqual(client.calls, [])
        partial = caught.exception.partial_record
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["calls"], [])
        self.assertEqual(set(partial["states"].values()), {"uncertain"})

    def test_remaining_deadline_limits_each_request_timeout(self):
        clock = FakeClock()
        client = TimedFakeClient(clock, [2, 3], answer({"person_in_work_area": "no"}),
                                answer({"visible": "no", "bbox": []}))
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            result = ImageObserver(client).observe(synthetic_frame(), "f0", deadline=110)
        self.assertEqual([call[3]["timeout"] for call in client.calls], [10, 8])
        self.assertEqual(result["elapsed_seconds"], 5)

    def test_late_last_answer_retains_raw_response_but_not_late_derived_state(self):
        clock = FakeClock()
        late_response = answer({"visible": "no", "bbox": []})
        client = TimedFakeClient(clock, [1, 15], answer({"person_in_work_area": "yes"}),
                                late_response)
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(client).observe(synthetic_frame(), "late-frame", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["elapsed_seconds"], 16)
        self.assertEqual(partial["states"], {
            "person_in_work_area": "yes", "cabinet_door": "uncertain"})
        self.assertEqual(len(partial["calls"]), 2)
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["observation"], late_response["observation"])
        self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")

    def test_first_late_answer_stops_followup_requests_and_preserves_unknown_states(self):
        clock = FakeClock()
        client = TimedFakeClient(clock, [11], answer({"person_in_work_area": "no"}))
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(client).observe(synthetic_frame(), "f0", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(partial["calls"]), 1)
        self.assertEqual(set(partial["states"].values()), {"uncertain"})
        self.assertEqual(partial["status"], "partial")

    def test_frame_identifier_and_dataset_label_never_enter_model_prompts(self):
        identifier = "clip_123_label_violation_at_99_sec"
        client = FakeClient(answer({"person_in_work_area": "yes"}),
                            answer({"visible": "no", "bbox": []}))
        frame = synthetic_frame()
        result = ImageObserver(client).observe(frame, identifier)
        self.assertEqual(result["frame_id"], identifier)
        for picture, prompt, schema, kwargs in client.calls:
            self.assertIs(picture, frame)
            self.assertNotIn(identifier, prompt)
            self.assertNotIn(str(frame.timestamp_sec), prompt)
            self.assertNotIn("clip_123", json.dumps(schema))


class ActivityObserverTests(unittest.TestCase):
    def test_person_behind_glass_requires_agreement_and_uses_separate_client(self):
        cabinet = FakeClient(answer({"door_state": "open"}),
                      answer({"cash_on_worktop": "yes"}))
        activity = FakeClient(answer({"position": "behind"}),
                       answer({"person_in_work_area": "no"}),
                       answer({"doorway_activity": "no"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                       answer({"resting_on_desk": "yes"}))
        frame = synthetic_frame()
        result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(frame, "f0")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["states"], {
            "person_in_work_area": "no", "cabinet_door": "open", "cash_on_worktop": "yes",
            "cash_in_hands": "uncertain", "notes_in_machine": "uncertain"})
        self.assertEqual(result["person_position"], "behind")
        self.assertEqual(result["person_side"], "no")
        self.assertEqual(result["person_doorway_check"], "no")
        self.assertEqual(result["person_consistency"], "absence_and_doorway_agreed")
        self.assertEqual(result["cash_support"], "worktop_only")
        self.assertEqual(result["desk_localization_model_role"], "activity")
        self.assertEqual([call[2] for call in activity.calls],
                         [POSITION_SCHEMA, PERSON_SCHEMA, DOORWAY_SCHEMA, BOX_SCHEMA, CONTACT_SCHEMA])
        self.assertEqual(activity.calls[2][1], DOORWAY_PROMPT)
        self.assertEqual(activity.calls[3][1], DESK_PROMPT)
        self.assertIs(activity.calls[3][0], frame)
        self.assertEqual(activity.calls[-1][1], CONTACT_PROMPT)
        self.assertEqual([call[1] for call in cabinet.calls], [DOOR_PROMPT, WORKTOP_PROMPT])
        self.assertEqual([call[2] for call in cabinet.calls], [DOOR_SCHEMA, WORKTOP_SCHEMA])
        self.assertTrue(all(call[0] is frame for call in activity.calls[:4] + cabinet.calls[:1]))
        crop = cabinet.calls[-1][0]
        self.assertIsNot(crop, frame)
        self.assertEqual((crop.width, crop.height), (96, 72))
        self.assertEqual(crop.timestamp_sec, frame.timestamp_sec)
        self.assertEqual(result["desk_crop_pixel_box"], [32, 24, 128, 96])
        contact_crop = activity.calls[-1][0]
        self.assertEqual(result["cash_contact_pixel_box"], [32, 0, 128, 99])
        self.assertEqual((contact_crop.width, contact_crop.height), (96, 99))
        self.assertEqual(contact_crop.timestamp_sec, frame.timestamp_sec)
        self.assertEqual(result["cash_banknotes_check"], "yes")
        self.assertEqual(result["cash_contact_check"], "yes")
        self.assertEqual(result["cash_consistency"], "agreed")
        self.assertEqual([call["model_role"] for call in result["calls"]],
                         ["activity", "activity", "activity", "cabinet", "activity", "cabinet", "activity"])
        desk_call = next(call for call in result["calls"] if call["question"] == "desk_location")
        self.assertEqual(desk_call["observation"], {"visible": "yes", "bbox": [250, 250, 750, 750]})
        self.assertTrue(desk_call["completed"])
        cash_call = next(call for call in result["calls"] if call["question"] == "cash_on_worktop")
        self.assertEqual(cash_call["model_role"], "cabinet")
        self.assertTrue(all(call[3]["max_tokens"] == 96 for call in activity.calls + cabinet.calls))

    def test_positive_person_and_worktop_are_retained_but_unused_cash_heads_stay_unknown(self):
        cabinet = FakeClient(answer({"door_state": "closed"}),
                      answer({"cash_on_worktop": "yes"}))
        activity = FakeClient(answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                       answer({"resting_on_desk": "yes"}))
        result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
        self.assertEqual(result["states"]["person_in_work_area"], "yes")
        self.assertEqual(result["states"]["cash_on_worktop"], "yes")
        self.assertEqual(result["person_consistency"], "agreed")
        self.assertNotIn("person_doorway_check", result)
        self.assertEqual([call[2] for call in activity.calls], [POSITION_SCHEMA, PERSON_SCHEMA, BOX_SCHEMA, CONTACT_SCHEMA])
        for feature in ("cash_in_hands", "notes_in_machine"):
            self.assertEqual(result["states"][feature], "uncertain")
            self.assertEqual(FEATURE_SUPPORT[feature]["status"], "not_validated")
        self.assertEqual(result["status"], "ok")

    def test_disagreement_or_uncertain_side_cannot_become_a_known_person_state(self):
        for position, side in (("front", "no"), ("behind", "yes"), ("absent", "yes"),
                               ("front", "uncertain"), ("behind", "uncertain")):
            cabinet = FakeClient(answer({"door_state": "closed"}))
            activity = FakeClient(answer({"position": position}),
                           answer({"person_in_work_area": side}),
                           answer({"visible": "no", "bbox": []}))
            with self.subTest(position=position, side=side):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["states"]["person_in_work_area"], "uncertain")
                self.assertEqual(result["person_consistency"], "uncertain_or_disagreed")
                self.assertEqual(result["states"]["cabinet_door"], "closed")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["errors"], [])
                self.assertNotIn("person_doorway_check", result)
                self.assertEqual([call[2] for call in activity.calls], [POSITION_SCHEMA, PERSON_SCHEMA, BOX_SCHEMA])

    def test_agreed_absence_requires_a_negative_doorway_check(self):
        for position in ("behind", "absent"):
            for doorway, expected in (("no", "no"), ("yes", "uncertain"), ("uncertain", "uncertain")):
                with self.subTest(position=position, doorway=doorway):
                    cabinet = FakeClient(answer({"door_state": "closed"}))
                    activity = FakeClient(answer({"position": position}),
                                   answer({"person_in_work_area": "no"}),
                                   answer({"doorway_activity": doorway}),
                                   answer({"visible": "no", "bbox": []}))
                    frame = synthetic_frame()
                    result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(frame, "boundary")
                    self.assertEqual(result["states"]["person_in_work_area"], expected)
                    self.assertEqual(result["person_doorway_check"], doorway)
                    self.assertEqual(result["person_consistency"], "absence_and_doorway_agreed" if doorway == "no"
                                     else "doorway_or_uncertain")
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["errors"], [])
                    self.assertEqual(result["states"]["cabinet_door"], "closed")
                    self.assertEqual([call[2] for call in activity.calls],
                                     [POSITION_SCHEMA, PERSON_SCHEMA, DOORWAY_SCHEMA, BOX_SCHEMA])
                    self.assertIs(activity.calls[-1][0], frame)

    def test_invalid_doorway_check_keeps_person_unknown_and_independent_fields(self):
        for bad in (answer({"doorway_activity": False}), answer({"person_in_work_area": "no"}),
                    answer({"doorway_activity": "no", "confidence": 1}),
                    answer({"doorway_activity": "no"}, finish="length"), RuntimeError("doorway unavailable")):
            with self.subTest(response=bad):
                cabinet = FakeClient(answer({"door_state": "closed"}),
                              answer({"cash_on_worktop": "yes"}))
                activity = FakeClient(answer({"position": "behind"}),
                               answer({"person_in_work_area": "no"}),
                               bad,
                               answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                               answer({"resting_on_desk": "yes"}))
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "bad-gate")
                self.assertEqual(result["states"]["person_in_work_area"], "uncertain")
                self.assertNotIn("person_doorway_check", result)
                self.assertEqual(result["person_position"], "behind")
                self.assertEqual(result["person_side"], "no")
                self.assertEqual(result["states"]["cabinet_door"], "closed")
                self.assertEqual(result["states"]["cash_on_worktop"], "yes")
                self.assertEqual(result["status"], "partial")
                self.assertEqual([error["feature"] for error in result["errors"]], ["person_in_work_area"])
                self.assertNotIn(result["person_consistency"], ("agreed", "absence_and_doorway_agreed"))

    def test_doorway_timeout_does_not_leave_a_provisional_no(self):
        cabinet = FakeClient()
        activity = FakeClient(answer({"position": "absent"}), answer({"person_in_work_area": "no"}),
                              TimeoutError("doorway timed out"))
        with self.assertRaises(ObservationDeadline) as caught:
            ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "gate-timeout")
        partial = caught.exception.partial_record
        self.assertEqual(partial["states"]["person_in_work_area"], "uncertain")
        self.assertNotIn("person_doorway_check", partial)
        self.assertEqual(partial["person_position"], "absent")
        self.assertEqual(partial["person_side"], "no")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["person_in_work_area"])
        self.assertEqual(partial["calls"][-1]["question"], "person_doorway_activity")
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["error_type"], "TimeoutError")
        self.assertEqual(cabinet.calls, [])

    def test_doorway_answer_at_or_after_deadline_is_raw_only(self):
        for gate_seconds in (8, 8.1):
            with self.subTest(gate_seconds=gate_seconds):
                clock = FakeClock()
                cabinet = FakeClient()
                activity = TimedFakeClient(clock, [1, 1, gate_seconds], answer({"position": "behind"}),
                                           answer({"person_in_work_area": "no"}), answer({"doorway_activity": "no"}))
                with patch("analyzer.observations.time.perf_counter", side_effect=clock):
                    with self.assertRaises(ObservationDeadline) as caught:
                        ImageObserver(cabinet, "full_frame", activity_client=activity).observe(
                            synthetic_frame(), "late-gate", deadline=110)
                partial = caught.exception.partial_record
                self.assertEqual(partial["states"]["person_in_work_area"], "uncertain")
                self.assertNotIn("person_doorway_check", partial)
                self.assertEqual(partial["person_position"], "behind")
                self.assertEqual(partial["person_side"], "no")
                self.assertEqual(partial["status"], "partial")
                self.assertEqual([error["feature"] for error in partial["errors"]], ["person_in_work_area"])
                self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")
                self.assertEqual(partial["calls"][-1]["observation"], {"doorway_activity": "no"})
                self.assertFalse(partial["calls"][-1]["completed"])
                self.assertEqual(activity.calls[-1][3]["timeout"], 8)
                self.assertEqual(cabinet.calls, [])

    def test_exhausted_budget_before_doorway_dispatch_cannot_confirm_absence(self):
        clock = FakeClock()
        activity = TimedFakeClient(clock, [1, 1], answer({"position": "behind"}),
                                   answer({"person_in_work_area": "no"}))
        cabinet = FakeClient()

        def validate_then_exhaust(response, key, allowed):
            value = read_enum(response, key, allowed)
            if key == "person_in_work_area":
                clock.advance(8)
            return value

        with patch("analyzer.observations.time.perf_counter", side_effect=clock), \
                patch("analyzer.observations.read_enum", side_effect=validate_then_exhaust):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(cabinet, "full_frame", activity_client=activity).observe(
                    synthetic_frame(), "before-gate", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["states"]["person_in_work_area"], "uncertain")
        self.assertNotIn("person_doorway_check", partial)
        self.assertEqual(partial["person_side"], "no")
        self.assertEqual([call["question"] for call in partial["calls"]], ["person_position", "person_side"])
        self.assertTrue(all(call["completed"] for call in partial["calls"]))
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["person_in_work_area"])
        self.assertEqual(len(activity.calls), 2)
        self.assertEqual(cabinet.calls, [])

    def test_doorway_and_uncertain_position_skip_the_side_question(self):
        for position in ("doorway", "uncertain"):
            cabinet = FakeClient(answer({"door_state": "closed"}))
            activity = FakeClient(answer({"position": position}),
                           answer({"visible": "no", "bbox": []}))
            with self.subTest(position=position):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["states"]["person_in_work_area"], "uncertain")
                self.assertEqual(result["person_consistency"], "uncertain_position")
                self.assertNotIn("person_side", result)
                self.assertEqual([call[2] for call in activity.calls], [POSITION_SCHEMA, BOX_SCHEMA])
                self.assertEqual([call["question"] for call in result["calls"]],
                                 ["person_position", "cabinet_door_full_frame", "desk_location"])
                self.assertEqual(result["status"], "ok")

    def test_unseen_or_ambiguous_desk_cannot_establish_absence_of_banknotes(self):
        for visible in ("no", "uncertain"):
            cabinet = FakeClient(answer({"door_state": "closed"}))
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           answer({"visible": visible, "bbox": []}))
            with self.subTest(visible=visible):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                self.assertEqual(result["states"]["person_in_work_area"], "yes")
                self.assertEqual(result["states"]["cabinet_door"], "closed")
                self.assertEqual(len(cabinet.calls), 1)
                self.assertIsNone(result["desk_crop_pixel_box"])
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["desk_localization_model_role"], "activity")
                self.assertEqual(activity.calls[-1][1], DESK_PROMPT)
                self.assertEqual(result["calls"][-1]["observation"], {"visible": visible, "bbox": []})
                self.assertNotIn("cash_banknotes_check", result)

    def test_invalid_desk_box_or_failed_location_never_reaches_worktop_question(self):
        for bad in (answer({"visible": "yes", "bbox": [800, 0, 200, 900]}),
                    answer({"visible": "yes", "bbox": [100, 100, 101, 101]}),
                    answer({"visible": "uncertain", "bbox": [250, 250, 750, 750]}),
                    RuntimeError("desk transport failed")):
            cabinet = FakeClient(answer({"door_state": "closed"}))
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           bad)
            with self.subTest(response=bad):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["states"]["person_in_work_area"], "yes")
                self.assertEqual(result["states"]["cabinet_door"], "closed")
                self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                self.assertEqual([error["feature"] for error in result["errors"]], ["cash_on_worktop"])
                self.assertEqual(len(cabinet.calls), 1)
                self.assertIsNone(result["desk_crop_pixel_box"])
                self.assertEqual(activity.calls[-1][1], DESK_PROMPT)
                self.assertEqual(result["calls"][-1]["model_role"], "activity")
                self.assertNotIn("cash_banknotes_check", result)

    def test_activity_desk_timeout_keeps_confirmed_person_and_cabinet_but_no_cash_answer(self):
        primary = FakeClient(answer({"door_state": "closed"}))
        activity = FakeClient(answer({"position": "front"}), answer({"person_in_work_area": "yes"}),
                              TimeoutError("desk localization timed out"))
        with self.assertRaises(ObservationDeadline) as caught:
            ImageObserver(primary, "full_frame", activity_client=activity).observe(synthetic_frame(), "desk-timeout")
        partial = caught.exception.partial_record
        self.assertEqual(partial["states"], {"person_in_work_area": "yes", "cabinet_door": "closed",
                                             **dict.fromkeys(CASH_FEATURES, "uncertain")})
        self.assertIsNone(partial["desk_bbox_normalized"])
        self.assertIsNone(partial["desk_crop_pixel_box"])
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual(partial["calls"][-1]["question"], "desk_location")
        self.assertEqual(partial["calls"][-1]["model_role"], "activity")
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["error_type"], "TimeoutError")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(activity.calls), 3)

    def test_activity_desk_response_at_or_after_deadline_is_raw_not_a_valid_region(self):
        for raw in ({"visible": "yes", "bbox": [250, 250, 750, 750]}, {"visible": "no", "bbox": []}):
            for duration in (7, 7.1):
                with self.subTest(raw=raw, duration=duration):
                    clock = FakeClock()
                    primary = TimedFakeClient(clock, [1], answer({"door_state": "closed"}))
                    activity = TimedFakeClient(clock, [1, 1, duration], answer({"position": "front"}),
                                               answer({"person_in_work_area": "yes"}), answer(raw))
                    with patch("analyzer.observations.time.perf_counter", side_effect=clock):
                        with self.assertRaises(ObservationDeadline) as caught:
                            ImageObserver(primary, "full_frame", activity_client=activity).observe(
                                synthetic_frame(), "late-desk", deadline=110)
                    partial = caught.exception.partial_record
                    self.assertEqual(partial["states"], {"person_in_work_area": "yes", "cabinet_door": "closed",
                                                         **dict.fromkeys(CASH_FEATURES, "uncertain")})
                    self.assertIsNone(partial["desk_bbox_normalized"])
                    self.assertIsNone(partial["desk_crop_pixel_box"])
                    self.assertNotIn("cash_banknotes_check", partial)
                    self.assertEqual(partial["status"], "partial")
                    self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
                    self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")
                    self.assertEqual(partial["calls"][-1]["question"], "desk_location")
                    self.assertEqual(partial["calls"][-1]["model_role"], "activity")
                    self.assertEqual(partial["calls"][-1]["observation"], raw)
                    self.assertFalse(partial["calls"][-1]["completed"])
                    self.assertEqual(activity.calls[-1][3]["timeout"], 7)
                    self.assertEqual(len(primary.calls), 1)
                    self.assertEqual(len(activity.calls), 3)

    def test_exhaustion_before_activity_desk_dispatch_preserves_independent_states(self):
        clock = FakeClock()
        primary = TimedFakeClient(clock, [1], answer({"door_state": "closed"}))
        activity = TimedFakeClient(clock, [1, 1], answer({"position": "front"}), answer({"person_in_work_area": "yes"}))

        def validate_then_exhaust(response, key, allowed):
            value = read_enum(response, key, allowed)
            if key == "door_state":
                clock.advance(7)
            return value

        with patch("analyzer.observations.time.perf_counter", side_effect=clock), \
                patch("analyzer.observations.read_enum", side_effect=validate_then_exhaust):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(primary, "full_frame", activity_client=activity).observe(synthetic_frame(), "before-desk", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["states"], {"person_in_work_area": "yes", "cabinet_door": "closed",
                                             **dict.fromkeys(CASH_FEATURES, "uncertain")})
        self.assertIsNone(partial["desk_bbox_normalized"])
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual([call["question"] for call in partial["calls"]],
                         ["person_position", "person_side", "cabinet_door_full_frame"])
        self.assertTrue(all(call["completed"] for call in partial["calls"]))
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(activity.calls), 2)

    def test_invalid_worktop_answer_does_not_erase_known_person_or_cabinet(self):
        for bad in (answer({}), answer({"cash_on_worktop": "possibly"}),
                    answer({"cash_on_worktop": False}), answer({"cash_on_worktop": ["yes"]}),
                    answer({"cash_on_worktop": "yes", "cash_in_hands": "no"}),
                    answer({"cash_on_worktop": "yes"}, finish="length"),
                    answer({"cash_on_worktop": "yes"}, valid=False), answer(None)):
            cabinet = FakeClient(answer({"door_state": "closed"}),
                          bad)
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
            with self.subTest(response=bad):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["states"], {
                    "person_in_work_area": "yes", "cabinet_door": "closed",
                    **dict.fromkeys(CASH_FEATURES, "uncertain")})
                self.assertEqual([error["feature"] for error in result["errors"]], ["cash_on_worktop"])
                self.assertEqual(result["errors"][0]["type"], "ObservationError")
                self.assertEqual(result["person_consistency"], "agreed")
                self.assertEqual(len(result["calls"]), 5)

    def test_invalid_position_does_not_prevent_independent_worktop_or_cabinet_observations(self):
        cabinet = FakeClient(answer({"door_state": "open"}),
                      answer({"cash_on_worktop": "yes"}))
        activity = FakeClient(answer({"position": "front", "confidence": 1}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                       answer({"resting_on_desk": "yes"}))
        result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["states"]["person_in_work_area"], "uncertain")
        self.assertEqual(result["states"]["cabinet_door"], "open")
        self.assertEqual(result["states"]["cash_on_worktop"], "yes")
        self.assertEqual([error["feature"] for error in result["errors"]], ["person_in_work_area"])
        self.assertEqual([call[2] for call in activity.calls], [POSITION_SCHEMA, BOX_SCHEMA, CONTACT_SCHEMA])

    def test_worktop_timeout_preserves_prior_fields_and_names_only_failed_feature(self):
        cabinet = FakeClient(answer({"door_state": "open"}),
                      TimeoutError("cash timed out"))
        activity = FakeClient(answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
        with self.assertRaises(ObservationDeadline) as caught:
            ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "cash-timeout")
        partial = caught.exception.partial_record
        self.assertEqual(partial["frame_id"], "cash-timeout")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["states"], {
            "person_in_work_area": "yes", "cabinet_door": "open",
            **dict.fromkeys(CASH_FEATURES, "uncertain")})
        self.assertEqual(partial["person_consistency"], "agreed")
        self.assertEqual(partial["cabinet_consistency"], "single_full_frame")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual(partial["errors"][0]["type"], "TimeoutError")
        self.assertEqual([call["completed"] for call in partial["calls"]], [True, True, True, True, False])
        self.assertEqual(partial["calls"][-1]["question"], "cash_on_worktop")
        self.assertEqual(partial["calls"][-1]["error_type"], "TimeoutError")

    def test_late_worktop_response_keeps_raw_answer_without_promoting_late_state(self):
        clock = FakeClock()
        late_cash = {"cash_on_worktop": "yes"}
        cabinet = TimedFakeClient(clock,
                      [1, 7],
                      answer({"door_state": "closed"}),
                      answer(late_cash))
        activity = TimedFakeClient(clock,
                       [1, 1, 1],
                       answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "late-cash", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["elapsed_seconds"], 11)
        self.assertEqual(partial["states"], {
            "person_in_work_area": "yes", "cabinet_door": "closed",
            **dict.fromkeys(CASH_FEATURES, "uncertain")})
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["observation"], late_cash)
        self.assertEqual(cabinet.calls[-1][3]["timeout"], 6)

    def test_worktop_requires_agreement_including_a_negative_result(self):
        for banknotes, contact, expected in (("no", "no", "no"), ("yes", "no", "uncertain"),
                                              ("no", "yes", "uncertain"), ("yes", "uncertain", "uncertain"),
                                              ("no", "uncertain", "uncertain")):
            cabinet = FakeClient(answer({"door_state": "closed"}),
                          answer({"cash_on_worktop": banknotes}))
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                           answer({"resting_on_desk": contact}),
                           *([answer({"table_view": "person_over_table"})]
                                    if (banknotes, contact) == ("no", "yes") else []))
            with self.subTest(banknotes=banknotes, contact=contact):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["states"]["cash_on_worktop"], expected)
                self.assertEqual(result["cash_banknotes_check"], banknotes)
                self.assertEqual(result["cash_contact_check"], contact)
                self.assertEqual(result["cash_consistency"],
                                 "agreed" if banknotes == contact else "uncertain_or_disagreed")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["errors"], [])
                self.assertNotIn("cash_negative_recheck", result)
                self.assertEqual([call["question"] for call in result["calls"]].count("cash_table_view"),
                                 int((banknotes, contact) == ("no", "yes")))

    def test_uncertain_banknotes_skip_contact_instead_of_asking_for_confirmation(self):
        cabinet = FakeClient(answer({"door_state": "closed"}),
                      answer({"cash_on_worktop": "uncertain"}))
        activity = FakeClient(answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
        result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
        self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
        self.assertEqual(result["cash_consistency"], "uncertain_banknotes")
        self.assertEqual(result["cash_banknotes_check"], "uncertain")
        self.assertNotIn("cash_contact_check", result)
        self.assertNotIn("cash_contact_pixel_box", result)
        self.assertEqual(len(activity.calls), 3)
        self.assertEqual(result["status"], "ok")

    def test_invalid_contact_answer_cannot_promote_an_earlier_banknotes_answer(self):
        for bad in (answer({"resting_on_desk": False}), answer({"cash_on_worktop": "yes"}),
                    answer({"resting_on_desk": "yes", "confidence": 1}),
                    answer({"resting_on_desk": "yes"}, finish="length")):
            cabinet = FakeClient(answer({"door_state": "closed"}),
                          answer({"cash_on_worktop": "yes"}))
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                           bad)
            with self.subTest(response=bad):
                result = ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
                self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                self.assertEqual(result["cash_banknotes_check"], "yes")
                self.assertNotIn("cash_contact_check", result)
                self.assertEqual(result["status"], "partial")
                self.assertEqual([error["feature"] for error in result["errors"]], ["cash_on_worktop"])
                self.assertEqual(result["states"]["person_in_work_area"], "yes")
                self.assertEqual(result["states"]["cabinet_door"], "closed")

    def test_contact_timeout_retains_raw_banknotes_but_never_a_finished_cash_state(self):
        for banknotes in ("yes", "no"):
            cabinet = FakeClient(answer({"door_state": "open"}),
                          answer({"cash_on_worktop": banknotes}))
            activity = FakeClient(answer({"position": "front"}),
                           answer({"person_in_work_area": "yes"}),
                           answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                           TimeoutError("contact timed out"))
            with self.subTest(banknotes=banknotes), self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0")
            partial = caught.exception.partial_record
            self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
            self.assertEqual(partial["cash_banknotes_check"], banknotes)
            self.assertNotIn("cash_contact_check", partial)
            self.assertEqual(partial["states"]["person_in_work_area"], "yes")
            self.assertEqual(partial["states"]["cabinet_door"], "open")
            self.assertEqual(partial["status"], "partial")
            self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
            self.assertEqual(partial["calls"][-1]["question"], "cash_surface_contact")
            self.assertFalse(partial["calls"][-1]["completed"])

    def test_late_contact_answer_cannot_finalize_agreement_after_deadline(self):
        clock = FakeClock()
        cabinet = TimedFakeClient(clock,
                      [1, 1],
                      answer({"door_state": "closed"}),
                      answer({"cash_on_worktop": "yes"}))
        activity = TimedFakeClient(clock,
                       [1, 1, 1, 7],
                       answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                       answer({"resting_on_desk": "yes"}))
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["elapsed_seconds"], 12)
        self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
        self.assertEqual(partial["cash_banknotes_check"], "yes")
        self.assertNotIn("cash_contact_check", partial)
        self.assertEqual(partial["states"]["person_in_work_area"], "yes")
        self.assertEqual(partial["states"]["cabinet_door"], "closed")
        self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["observation"], {"resting_on_desk": "yes"})
        self.assertEqual(activity.calls[-1][3]["timeout"], 5)

    def test_worktop_answer_at_deadline_is_raw_only_and_makes_no_contact_request(self):
        clock = FakeClock()
        cabinet = TimedFakeClient(clock,
                      [1, 6],
                      answer({"door_state": "closed"}),
                      answer({"cash_on_worktop": "yes"}))
        activity = TimedFakeClient(clock,
                       [1, 1, 1],
                       answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
        with patch("analyzer.observations.time.perf_counter", side_effect=clock):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(cabinet, "full_frame", activity_client=activity).observe(synthetic_frame(), "f0", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(partial["elapsed_seconds"], 10)
        self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
        self.assertNotIn("cash_banknotes_check", partial)
        self.assertNotIn("cash_contact_check", partial)
        self.assertEqual(partial["states"]["person_in_work_area"], "yes")
        self.assertEqual(partial["states"]["cabinet_door"], "closed")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(len(activity.calls), 3)
        self.assertEqual(len(partial["calls"]), 5)
        self.assertEqual(partial["calls"][-1]["question"], "cash_on_worktop")
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(partial["calls"][-1]["observation"], {"cash_on_worktop": "yes"})
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")

    def test_disabled_activity_mode_preserves_the_original_two_feature_contract(self):
        client = FakeClient(answer({"person_in_work_area": "yes"}), answer({"door_state": "closed"}))
        result = ImageObserver(client, "full_frame", activity_client=None).observe(synthetic_frame(), "f0")
        self.assertEqual(result["states"], {"person_in_work_area": "yes", "cabinet_door": "closed"})
        self.assertNotIn("person_consistency", result)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([call[3]["max_tokens"] for call in client.calls], [96, 96])
        self.assertEqual(result["status"], "ok")

    def test_fingerprint_separates_optional_activity_policy_from_legacy_reports(self):
        for mode in ("context_crops", "full_frame"):
            with self.subTest(mode=mode):
                self.assertEqual(policy_fingerprint(mode), policy_fingerprint(mode, cash_enabled=False))
                self.assertNotEqual(policy_fingerprint(mode), policy_fingerprint(mode, cash_enabled=True))
        self.assertNotEqual(policy_fingerprint("context_crops", cash_enabled=True),
                            policy_fingerprint("full_frame", cash_enabled=True))

    def test_boundary_prompt_is_part_of_activity_policy_fingerprint_only(self):
        activity = policy_fingerprint("full_frame", cash_enabled=True)
        legacy = policy_fingerprint("full_frame", cash_enabled=False)
        for name in ("DOORWAY_PROMPT", "TABLE_VIEW_PROMPT", "CASH_RECHECK_PROMPT", "DESK_PROMPT"):
            with self.subTest(prompt=name), patch("analyzer.observations." + name, "Changed visual definition."):
                self.assertNotEqual(activity, policy_fingerprint("full_frame", cash_enabled=True))
                self.assertEqual(legacy, policy_fingerprint("full_frame", cash_enabled=False))


class CashNegativeRecheckTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()

    def clients(self, *extra_responses):
        primary = FakeClient(answer({"door_state": "closed"}),
                      answer({"cash_on_worktop": "no"}))
        activity = FakeClient(answer({"position": "front"}),
                       answer({"person_in_work_area": "yes"}),
                       answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                       answer({"resting_on_desk": "yes"}),
                       *extra_responses)
        return primary, activity

    def assert_prior_results_retained(self, result):
        self.assertEqual(result["cash_banknotes_check"], "no")
        self.assertEqual(result["cash_contact_check"], "yes")
        self.assertEqual(result["states"]["person_in_work_area"], "yes")
        self.assertEqual(result["states"]["cabinet_door"], "closed")
        self.assertEqual(result["states"]["cash_in_hands"], "uncertain")
        self.assertEqual(result["states"]["notes_in_machine"], "uncertain")

    def test_negative_recheck_requires_clear_view_and_reuses_the_expanded_context(self):
        primary, activity = self.clients(answer({"table_view": "clear"}), answer({"cash_on_worktop": "no"}))
        result = ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "not-a-prompt")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["errors"], [])
        self.assert_prior_results_retained(result)
        self.assertEqual(result["states"]["cash_on_worktop"], "no")
        self.assertEqual(result["cash_table_view"], "clear")
        self.assertEqual(result["cash_negative_recheck"], "no")
        self.assertEqual(result["cash_consistency"], "negative_rechecked_unobstructed_worktop")
        self.assertEqual([call["question"] for call in result["calls"]][-3:],
                         ["cash_surface_contact", "cash_table_view", "cash_negative_recheck"])
        self.assertEqual([call[1] for call in activity.calls][-3:],
                         [CONTACT_PROMPT, TABLE_VIEW_PROMPT, CASH_RECHECK_PROMPT])
        self.assertEqual([call[2] for call in activity.calls][-3:],
                         [CONTACT_SCHEMA, TABLE_VIEW_SCHEMA, WORKTOP_SCHEMA])
        context = activity.calls[3][0]
        self.assertEqual((context.width, context.height), (96, 99))
        self.assertEqual(result["cash_contact_pixel_box"], [32, 0, 128, 99])
        self.assertIs(activity.calls[4][0], context)
        self.assertIs(activity.calls[5][0], context)
        self.assertEqual(context.timestamp_sec, self.frame.timestamp_sec)
        self.assertNotEqual(context.jpeg, primary.calls[-1][0].jpeg)
        for _, prompt, schema, _ in activity.calls:
            self.assertNotIn("not-a-prompt", prompt)
            self.assertNotIn(str(self.frame.timestamp_sec), prompt)
            self.assertNotIn("not-a-prompt", json.dumps(schema))

    def test_occluded_or_undetermined_table_does_not_reach_negative_recheck(self):
        for view in ("person_over_table", "uncertain"):
            with self.subTest(view=view):
                primary, activity = self.clients(answer({"table_view": view}))
                result = ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "occluded")
                self.assert_prior_results_retained(result)
                self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                self.assertEqual(result["cash_table_view"], view)
                self.assertEqual(result["cash_consistency"], "uncertain_or_disagreed")
                self.assertNotIn("cash_negative_recheck", result)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["errors"], [])
                self.assertEqual(len(activity.calls), 5)
                self.assertEqual(result["calls"][-1]["question"], "cash_table_view")

    def test_nonnegative_recheck_cannot_establish_either_cash_presence_or_absence(self):
        for state in ("yes", "uncertain"):
            with self.subTest(recheck=state):
                primary, activity = self.clients(answer({"table_view": "clear"}), answer({"cash_on_worktop": state}))
                result = ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "conflict")
                self.assert_prior_results_retained(result)
                self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                self.assertEqual(result["cash_table_view"], "clear")
                self.assertEqual(result["cash_negative_recheck"], state)
                self.assertEqual(result["cash_consistency"], "uncertain_or_disagreed")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["errors"], [])

    def test_invalid_or_failed_reconciliation_cannot_turn_original_no_into_final_no(self):
        stages = {
            "cash_table_view": (answer({"table_view": "no"}), answer({"table_view": False}),
                                answer({"visible": "yes"}), answer({"table_view": "clear", "bbox": []}),
                                answer({"table_view": "clear"}, finish="length"), RuntimeError("view failed")),
            "cash_negative_recheck": (answer({"resting_on_desk": "no"}), answer({"cash_on_worktop": False}),
                                      answer({"cash_on_worktop": "no", "confidence": 1}),
                                      answer({"cash_on_worktop": "no"}, finish="length"), RuntimeError("recheck failed")),
        }
        for stage, bad_responses in stages.items():
            for bad in bad_responses:
                with self.subTest(stage=stage, response=bad):
                    extras = [answer({"table_view": "clear"})] if stage == "cash_negative_recheck" else []
                    primary, activity = self.clients(*extras, bad)
                    result = ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "bad-check")
                    self.assert_prior_results_retained(result)
                    self.assertEqual(result["states"]["cash_on_worktop"], "uncertain")
                    self.assertNotIn(stage, result)
                    self.assertEqual(result["cash_consistency"], "uncertain_or_disagreed")
                    self.assertEqual(result["status"], "partial")
                    self.assertEqual([error["feature"] for error in result["errors"]], ["cash_on_worktop"])
                    self.assertEqual(result["calls"][-1]["question"], stage)
                    self.assertEqual(len(activity.calls), 6 if extras else 5)
                    if extras:
                        self.assertEqual(result["cash_table_view"], "clear")

    def test_timeout_at_either_new_question_keeps_raw_prior_checks_and_unknown_cash(self):
        for stage in ("cash_table_view", "cash_negative_recheck"):
            with self.subTest(stage=stage):
                extras = [answer({"table_view": "clear"})] if stage == "cash_negative_recheck" else []
                primary, activity = self.clients(*extras, TimeoutError("reconciliation timeout"))
                with self.assertRaises(ObservationDeadline) as caught:
                    ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "timeout")
                partial = caught.exception.partial_record
                self.assert_prior_results_retained(partial)
                self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
                self.assertNotIn(stage, partial)
                self.assertEqual(partial["status"], "partial")
                self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
                self.assertEqual(partial["calls"][-1]["question"], stage)
                self.assertFalse(partial["calls"][-1]["completed"])
                self.assertEqual(partial["calls"][-1]["error_type"], "TimeoutError")
                if extras:
                    self.assertEqual(partial["cash_table_view"], "clear")

    def test_response_at_or_after_deadline_cannot_complete_either_new_question(self):
        for stage in ("cash_table_view", "cash_negative_recheck"):
            for late_by in (0, 0.1):
                with self.subTest(stage=stage, late_by=late_by):
                    clock = FakeClock()
                    responses = [answer({"position": "front"}), answer({"person_in_work_area": "yes"}),
                                 answer({"visible": "yes", "bbox": [250, 250, 750, 750]}),
                                 answer({"resting_on_desk": "yes"}), answer({"table_view": "clear"})]
                    durations = [1, 1, 1, 1, 4 + late_by]
                    if stage == "cash_negative_recheck":
                        responses.append(answer({"cash_on_worktop": "no"}))
                        durations = [1, 1, 1, 1, 1, 3 + late_by]
                    primary = TimedFakeClient(clock, [1, 1], answer({"door_state": "closed"}),
                                              answer({"cash_on_worktop": "no"}))
                    activity = TimedFakeClient(clock, durations, *responses)
                    with patch("analyzer.observations.time.perf_counter", side_effect=clock):
                        with self.assertRaises(ObservationDeadline) as caught:
                            ImageObserver(primary, "full_frame", activity_client=activity).observe(
                                self.frame, "late-check", deadline=110)
                    partial = caught.exception.partial_record
                    self.assert_prior_results_retained(partial)
                    self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
                    self.assertNotIn(stage, partial)
                    self.assertEqual(partial["status"], "partial")
                    self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
                    self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")
                    self.assertEqual(partial["calls"][-1]["question"], stage)
                    self.assertFalse(partial["calls"][-1]["completed"])
                    self.assertEqual(partial["calls"][-1]["observation"], responses[-1]["observation"])
                    self.assertEqual(activity.calls[-1][3]["timeout"], 4 if stage == "cash_table_view" else 3)
                    self.assertEqual(len(activity.calls), len(responses))
                    if stage == "cash_negative_recheck":
                        self.assertEqual(partial["cash_table_view"], "clear")

    def test_budget_exhausted_after_clear_view_makes_no_recheck_request(self):
        clock = FakeClock()
        primary, activity = self.clients(answer({"table_view": "clear"}))

        def validate_then_exhaust(response, key, allowed):
            value = read_enum(response, key, allowed)
            if key == "table_view":
                clock.advance(10)
            return value

        with patch("analyzer.observations.time.perf_counter", side_effect=clock), \
                patch("analyzer.observations.read_enum", side_effect=validate_then_exhaust):
            with self.assertRaises(ObservationDeadline) as caught:
                ImageObserver(primary, "full_frame", activity_client=activity).observe(self.frame, "before-recheck", deadline=110)
        partial = caught.exception.partial_record
        self.assert_prior_results_retained(partial)
        self.assertEqual(partial["states"]["cash_on_worktop"], "uncertain")
        self.assertEqual(partial["cash_table_view"], "clear")
        self.assertNotIn("cash_negative_recheck", partial)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cash_on_worktop"])
        self.assertEqual(partial["calls"][-1]["question"], "cash_table_view")
        self.assertTrue(partial["calls"][-1]["completed"])
        self.assertEqual(len(activity.calls), 5)


class RequestBoundaryTests(unittest.TestCase):
    def test_wire_payload_uses_exact_image_bytes_without_frame_metadata(self):
        frame = synthetic_frame()
        client = LocalModelClient()
        captured = []

        class FakeOpener:
            def open(self, request, **kwargs):
                captured.append((json.loads(request.data), kwargs))
                return BytesIO(json.dumps({"choices": [{
                    "message": {"content": '{"person_in_work_area":"no"}'},
                    "finish_reason": "stop",
                }]}).encode())

        client.opener = FakeOpener()
        result = client.ask_image(frame, "Is a person visible?", PERSON_SCHEMA, timeout=7)
        payload, kwargs = captured[0]
        content = payload["messages"][0]["content"]
        sent_bytes = base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(sent_bytes, frame.jpeg)
        self.assertEqual(content[1], {"type": "text", "text": "Is a person visible?"})
        self.assertNotIn(str(frame.timestamp_sec), json.dumps(payload))
        self.assertEqual(kwargs["timeout"], 7)
        self.assertEqual(read_enum(result, "person_in_work_area", ("yes", "no", "uncertain")), "no")

    def test_generation_limit_rejects_bool_out_of_range_and_non_integer_values(self):
        client = LocalModelClient()
        for value in (True, 0, -1, 257, 96.0, "96", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client.ask_image(synthetic_frame(), "Question", PERSON_SCHEMA, max_tokens=value)


if __name__ == "__main__":
    unittest.main()
