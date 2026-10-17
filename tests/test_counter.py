"""Counter discovery tests use synthetic JPEGs and bounded fake clients only."""

from io import BytesIO
import math
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.counter import COUNTER_IDENTITY_PROMPT, discover_counter
from analyzer.video import VideoFrame


def response(value, *, finish="stop"):
    return {"observation": value, "finish_reason": finish, "valid_json": True}


def found(box=None):
    return response({"visible": "yes", "bbox": [250, 250, 750, 750] if box is None else box})


def identity(value="yes"):
    return response({"banknote_counter": value})


def frames_at(times):
    frames = []
    for timestamp in times:
        stream = BytesIO()
        Image.new("RGB", (160, 120), "navy").save(stream, "JPEG")
        frames.append(VideoFrame(timestamp, stream.getvalue(), 160, 120))
    return frames


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


class Client:
    timeout = 30

    def __init__(self, *answers, clock=None, durations=()):
        self.answers = iter(answers)
        self.calls = []
        self.clock = clock
        self.durations = iter(durations)

    def ask_image(self, frame, prompt, schema, **kwargs):
        self.calls.append({"frame": frame, "prompt": prompt, "schema": schema, **kwargs})
        if self.clock is not None:
            self.clock.now += next(self.durations)
        value = next(self.answers)
        if isinstance(value, Exception):
            raise value
        return value


class CounterTests(unittest.TestCase):
    def test_first_middle_last_verified_counter_boxes_are_motion_observations(self):
        frames = frames_at([0, .5, 2, 3.5, 5, 8, 10])
        grounding = Client(found(), found(), found())
        activity = Client(identity(), identity(), identity())
        report = discover_counter(frames, [0, 2, 4, 6], grounding, activity)
        self.assertEqual([call["frame"].timestamp_sec for call in grounding.calls], [0, 5, 10])
        self.assertEqual([row["timestamp_sec"] for row in report["observations"]], [0, 5, 10])
        self.assertTrue(all(row["desk_bbox_normalized"] == [250, 250, 750, 750]
                            and row["status"] == "ok" for row in report["observations"]))
        self.assertEqual(report["coverage"]["validated_frame_count"], 3)
        self.assertEqual(len(report["calls"]), 6)
        self.assertTrue(all(call["prompt"] == COUNTER_IDENTITY_PROMPT for call in activity.calls))
        # The identity model sees a crop, not an unrelated full-frame object.
        self.assertEqual([(c["frame"].width, c["frame"].height) for c in activity.calls], [(144, 108)] * 3)
        self.assertEqual([c["frame"].timestamp_sec for c in activity.calls], [0, 5, 10])

    def test_duplicate_indices_and_pts_do_not_duplicate_model_calls(self):
        frames = frames_at([0, 0, 2, 4])
        grounding, activity = Client(found(), found(), found()), Client(identity(), identity(), identity())
        report = discover_counter(frames, [0, 0, 1, 2, 3, 3], grounding, activity)
        self.assertEqual(report["coverage"]["selected_frame_indices"], [0, 2, 3])
        self.assertEqual(len(grounding.calls), 3)
        self.assertEqual(len(activity.calls), 3)

    def test_one_two_and_empty_inputs_never_repeat_or_invent_frames(self):
        for times in ([], [0], [0, 2]):
            with self.subTest(times=times):
                grounding = Client(*(found() for _ in times))
                activity = Client(*(identity() for _ in times))
                report = discover_counter(frames_at(times), list(range(len(times))), grounding, activity)
                self.assertEqual(len(report["observations"]), len(times))
                self.assertEqual(report["coverage"]["selected_frame_count"], len(times))
                self.assertEqual(report["status"], "ok")

    def test_no_or_unknown_identity_rejects_roi_without_claiming_no_action(self):
        for state in ("no", "uncertain"):
            with self.subTest(state=state):
                report = discover_counter(frames_at([1]), [0], Client(found()), Client(identity(state)))
                self.assertEqual(report["observations"], [])
                self.assertEqual(report["checks"][0]["banknote_counter"], state)
                self.assertEqual(report["status"], "ok")
                self.assertNotIn("result", report)
                self.assertEqual(report["coverage"]["unvalidated_frame_count"], 1)

    def test_missing_object_does_not_call_identity_or_infer_no_action(self):
        for visible in ("no", "uncertain"):
            activity = Client()
            report = discover_counter(frames_at([1]), [0], Client(response({"visible": visible, "bbox": []})), activity)
            self.assertEqual(activity.calls, [])
            self.assertEqual(report["observations"], [])
            self.assertEqual(report["checks"][0]["banknote_counter"], "uncertain")

    def test_invalid_grounding_is_partial_and_later_valid_frame_is_retained(self):
        grounding = Client(found([700, 0, 200, 900]), found())
        report = discover_counter(frames_at([0, 2]), [0, 1], grounding, Client(identity()))
        self.assertEqual(report["status"], "partial")
        self.assertEqual([row["timestamp_sec"] for row in report["observations"]], [2])
        self.assertEqual(report["errors"][0]["question"], "ground_counter")
        self.assertEqual(len(report["calls"]), 3)

    def test_invalid_identity_and_transport_failures_never_validate_roi(self):
        answers = [identity("printer"), identity(True), response({"banknote_counter": "yes", "extra": 1}),
                   response({"banknote_counter": "yes"}, finish="length"), TimeoutError("local timeout")]
        for raw in answers:
            with self.subTest(raw=raw):
                report = discover_counter(frames_at([0]), [0], Client(found()), Client(raw))
                self.assertEqual(report["observations"], [])
                self.assertEqual(report["status"], "partial")
                self.assertEqual(report["errors"][0]["question"], "counter_identity")
                self.assertEqual(len(report["calls"]), 2)

    def test_expired_budget_makes_no_requests(self):
        grounding, activity = Client(), Client()
        with patch("analyzer.counter.time.perf_counter", return_value=10):
            report = discover_counter(frames_at([0, 2]), [0, 1], grounding, activity, deadline=10)
        self.assertEqual(grounding.calls + activity.calls, [])
        self.assertEqual(report["observations"], [])
        self.assertTrue(report["coverage"]["budget_exhausted"])
        self.assertEqual(report["coverage"]["skipped_frame_count"], 2)

    def test_late_grounding_never_reaches_identity(self):
        clock = Clock()
        grounding, activity = Client(found(), clock=clock, durations=[2]), Client()
        with patch("analyzer.counter.time.perf_counter", clock):
            report = discover_counter(frames_at([0, 2]), [0, 1], grounding, activity, deadline=11)
        self.assertEqual(activity.calls, [])
        self.assertEqual(report["observations"], [])
        self.assertIn("response", report["calls"][0])
        self.assertFalse(report["calls"][0]["completed"])
        self.assertEqual(grounding.calls[0]["timeout"], 1)

    def test_late_identity_preserves_previous_roi_but_not_late_yes(self):
        clock = Clock()
        grounding = Client(found(), found(), clock=clock, durations=[.2, .2])
        activity = Client(identity(), identity(), clock=clock, durations=[.2, 2])
        with patch("analyzer.counter.time.perf_counter", clock):
            report = discover_counter(frames_at([0, 2, 4]), [0, 1, 2], grounding, activity, deadline=11)
        self.assertEqual([row["timestamp_sec"] for row in report["observations"]], [0])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(len(report["calls"]), 4)
        self.assertIn("response", report["calls"][-1])
        self.assertFalse(report["calls"][-1]["completed"])
        self.assertAlmostEqual(activity.calls[-1]["timeout"], .4)
        self.assertEqual(report["coverage"]["skipped_frame_count"], 1)

    def test_deadline_expiring_during_crop_prevents_identity_request(self):
        clock = Clock()
        grounding, activity = Client(found()), Client()
        from analyzer.observations import crop_object
        def slow_crop(*args):
            result = crop_object(*args)
            clock.now = 12
            return result
        with patch("analyzer.counter.time.perf_counter", clock), patch("analyzer.counter.crop_object", slow_crop):
            report = discover_counter(frames_at([0]), [0], grounding, activity, deadline=11)
        self.assertEqual(activity.calls, [])
        self.assertEqual(report["observations"], [])
        self.assertTrue(report["coverage"]["budget_exhausted"])

    def test_invalid_indices_times_or_deadline_fail_before_any_request(self):
        cases = [([0, 1], [True], None), ([0, 1], [-1], None), ([0, 1], [2], None),
                 ([0, 1], [1, 0], None), ([0, 1], [0.0], None), ([1, 0], [0, 1], None),
                 ([0, math.nan], [0, 1], None), ([False], [0], None), ([-1], [0], None),
                 ([0], [0], math.inf), ([0], [0], True)]
        for times, indices, deadline in cases:
            grounding, activity = Client(), Client()
            with self.subTest(times=times, indices=indices, deadline=deadline), self.assertRaises(ValueError):
                discover_counter(frames_at(times), indices, grounding, activity, deadline)
            self.assertEqual(grounding.calls + activity.calls, [])


if __name__ == "__main__":
    unittest.main()
