"""Review-only direction safety with actual synthetic JPEGs and fake inference."""

from io import BytesIO
import hashlib
import json
import math
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from analyzer.direction import DirectionObserver, direction_policy_fingerprint
from analyzer.observations import crop_object
from analyzer.video import VideoFrame


BOX = [250, 200, 500, 800]


def frames_at(times=(.125, 1.375, 2.9)):
    frames = []
    for index, timestamp in enumerate(times):
        image = Image.new("RGB", (200, 100), (30 + index * 25, 80, 140))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 49, 99), fill=(190, 20, 20))
        draw.rectangle((150, 0, 199, 99), fill=(20, 190, 20))
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=95)
        frames.append(VideoFrame(timestamp, stream.getvalue(), 200, 100))
    return frames


def answer(motion="hand_to_input", evidence="A visible bundle changes position.", **updates):
    value = {"banknote_motion": motion, "evidence": evidence}
    result = {"observation": value, "finish_reason": "stop", "valid_json": True,
              "response_text": json.dumps(value), "usage": {"completion_tokens": 18}}
    result.update(updates)
    return result


class Clock:
    def __init__(self, now=100):
        self.now = now

    def __call__(self):
        return self.now


class FakeClient:
    timeout = 30

    def __init__(self, response=None, *, clock=None, duration=0):
        self.response = answer() if response is None else response
        self.calls = []
        self.clock, self.duration = clock, duration

    def ask_sequence(self, frames, prompt, schema, **kwargs):
        self.calls.append({"frames": list(frames), "prompt": prompt, "schema": schema, **kwargs})
        if self.clock is not None:
            self.clock.now += self.duration
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class DirectionSafetyTests(unittest.TestCase):
    def check(self, client=None, frames=None, **kwargs):
        return DirectionObserver(client or FakeClient()).check(
            frames_at() if frames is None else frames,
            window_id="synthetic_window", counter_box=list(BOX), **kwargs)

    def assert_not_promoted(self, record):
        self.assertEqual(record["result"], {"counter_direction": "uncertain"})
        self.assertEqual(record["feature_support"]["status"], "not_validated")
        self.assertNotIn("verdict", record)

    def test_valid_raw_answers_remain_review_only_even_when_confident(self):
        for motion in ("hand_to_input", "output_to_hand", "no_transfer", "uncertain"):
            with self.subTest(motion=motion):
                raw = answer(motion, "The transfer is definitely visible.")
                record = self.check(FakeClient(raw))
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["raw_banknote_motion"], motion)
                self.assertEqual(record["model_evidence_description"], raw["observation"]["evidence"])
                self.assertTrue(record["calls"][0]["completed"])
                self.assertIs(record["calls"][0]["response"], raw)
                self.assert_not_promoted(record)

    def test_crops_are_from_same_source_geometry_with_real_pts_and_matching_hashes(self):
        frames = frames_at()
        original = [(frame.timestamp_sec, frame.jpeg) for frame in frames]
        client = FakeClient()
        record = self.check(client, frames)
        sent = client.calls[0]["frames"]
        self.assertEqual([f.timestamp_sec for f in sent], [.125, 1.375, 2.9])
        self.assertEqual(record["calls"][0]["timestamps_sec"], [.125, 1.375, 2.9])
        self.assertEqual(record["crop_pixel_boxes"], [[15, 0, 135, 100]] * 3)
        self.assertEqual([(f.width, f.height) for f in sent], [(120, 100)] * 3)
        self.assertEqual(record["crop_sha256"], [hashlib.sha256(f.jpeg).hexdigest() for f in sent])
        self.assertEqual(len(set(record["crop_sha256"])), 3)
        for frame in sent:
            with Image.open(BytesIO(frame.jpeg)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (120, 100))
                # Left red stripe remains; the far-right green stripe is excluded.
                self.assertGreater(image.getpixel((10, 50))[0], 150)
                self.assertLess(image.getpixel((110, 50))[1], 120)
        self.assertEqual([(f.timestamp_sec, f.jpeg) for f in frames], original)
        self.assertAlmostEqual(record["max_anchor_gap_sec"], 1.525)
        self.assertEqual(client.calls[0]["max_tokens"], 192)
        self.assertFalse(client.calls[0]["schema"]["additionalProperties"])
        self.assertNotIn("synthetic_window", client.calls[0]["prompt"])

    def test_two_and_five_frames_work_but_invalid_counts_never_call_model(self):
        for count in (2, 5):
            with self.subTest(count=count):
                client = FakeClient()
                record = self.check(client, frames_at(range(count)))
                self.assertEqual(record["status"], "ok")
                self.assertEqual(len(client.calls[0]["frames"]), count)
        for count in (0, 1, 6):
            with self.subTest(count=count):
                client = FakeClient()
                with self.assertRaises(ValueError):
                    self.check(client, frames_at(range(count)))
                self.assertEqual(client.calls, [])

    def test_invalid_pts_boxes_and_deadlines_fail_before_request(self):
        for times in ((1, 1), (2, 1), (-1, 0), (False, 1), (0, math.nan), (0, math.inf), (0, "1")):
            with self.subTest(times=times):
                client = FakeClient()
                with self.assertRaises(ValueError):
                    self.check(client, frames_at(times))
                self.assertFalse(client.calls)
        for box in ([], None, [0, 0, 0, 100], [True, 0, 100, 100], [0, 0, 1001, 100]):
            with self.subTest(box=box):
                client = FakeClient()
                with self.assertRaises(ValueError):
                    DirectionObserver(client).check(frames_at(), window_id="w", counter_box=box)
                self.assertFalse(client.calls)
        for deadline in (True, math.nan, math.inf, "100"):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                self.check(deadline=deadline)

    def test_schema_and_truncation_fail_closed_while_preserving_raw_response(self):
        invalid = [answer(finish_reason="length"), answer(valid_json=False),
                   answer(observation=[]), answer(observation={"banknote_motion": "hand_to_input"}),
                   answer(observation={"banknote_motion": "hand_to_input", "evidence": "Seen", "verdict": "violation"}),
                   answer("load"), answer(["hand_to_input"]), answer({"value": "hand_to_input"}),
                   answer(True), answer(evidence=None), answer(evidence="  "), answer(evidence="x" * 601)]
        for raw in invalid:
            with self.subTest(raw=raw):
                record = self.check(FakeClient(raw))
                self.assertEqual(record["status"], "partial")
                self.assertIsNone(record["raw_banknote_motion"])
                self.assertNotIn("model_evidence_description", record)
                self.assertFalse(record["calls"][0]["completed"])
                self.assertIs(record["calls"][0]["response"], raw)
                self.assertTrue(record["errors"])
                self.assert_not_promoted(record)

    def test_unreadable_or_mismatched_source_never_reaches_model(self):
        good = frames_at((0, 1))[0]
        for broken in (VideoFrame(1, b"not-a-jpeg", 200, 100),
                       VideoFrame(1, good.jpeg, 100, 200)):
            with self.subTest(broken=broken.width):
                client = FakeClient()
                record = self.check(client, [good, broken])
                self.assertEqual(record["status"], "partial")
                self.assertFalse(client.calls)
                self.assertTrue(record["errors"])
                self.assert_not_promoted(record)

    def test_deadline_before_crops_does_no_image_work_or_request(self):
        client = FakeClient()
        with patch("analyzer.direction.time.perf_counter", return_value=100), \
                patch("analyzer.direction.crop_object") as crop:
            record = self.check(client, deadline=100)
        crop.assert_not_called()
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["calls"], [])
        self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
        self.assert_not_promoted(record)

    def test_budget_spent_on_crops_keeps_hashes_but_skips_request(self):
        clock, client = Clock(), FakeClient()

        def slow_crop(*args):
            result = crop_object(*args)
            clock.now += 2
            return result

        with patch("analyzer.direction.time.perf_counter", clock), \
                patch("analyzer.direction.crop_object", side_effect=slow_crop):
            record = self.check(client, frames_at((0, 1)), deadline=103)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(len(record["crop_sha256"]), 2)
        self.assertFalse(client.calls)
        self.assert_not_promoted(record)

    def test_budget_expiring_while_timeout_is_computed_must_not_send_request(self):
        client = FakeClient()
        # The process may be preempted after the initial pre-request check.
        readings = iter((100, 100.9, 101.1))

        def advancing_clock():
            return next(readings, 101.1)

        with patch("analyzer.direction.time.perf_counter", advancing_clock):
            record = self.check(client, deadline=101)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(client.calls, [])
        self.assert_not_promoted(record)

    def test_request_timeout_is_bounded_by_remaining_budget_and_client_limit(self):
        for deadline, expected in ((109, 9), (200, 30), (None, 30)):
            with self.subTest(deadline=deadline):
                clock = Clock()
                client = FakeClient(clock=clock, duration=.5)
                with patch("analyzer.direction.time.perf_counter", clock):
                    record = self.check(client, deadline=deadline)
                self.assertEqual(record["status"], "ok")
                self.assertEqual(client.calls[0]["timeout"], expected)
                self.assertEqual(record["calls"][0]["timeout_sec"], expected)

    def test_answer_at_or_after_deadline_retains_raw_but_never_completes(self):
        for duration in (2, 2.1):
            with self.subTest(duration=duration):
                clock, raw = Clock(), answer("output_to_hand")
                client = FakeClient(raw, clock=clock, duration=duration)
                with patch("analyzer.direction.time.perf_counter", clock):
                    record = self.check(client, deadline=102)
                self.assertEqual(record["status"], "partial")
                self.assertIs(record["calls"][0]["response"], raw)
                self.assertFalse(record["calls"][0]["completed"])
                self.assertIsNone(record["raw_banknote_motion"])
                self.assertEqual(record["errors"][0]["type"], "ObservationDeadline")
                self.assert_not_promoted(record)

    def test_transport_timeout_preserves_attempt_and_crops_without_inventing_answer(self):
        client = FakeClient(TimeoutError("synthetic timeout"))
        record = self.check(client)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(len(record["crop_sha256"]), 3)
        self.assertEqual(len(record["calls"]), 1)
        self.assertFalse(record["calls"][0]["completed"])
        self.assertNotIn("response", record["calls"][0])
        self.assertEqual(record["calls"][0]["error"]["type"], "TimeoutError")
        self.assertIsNone(record["raw_banknote_motion"])
        self.assert_not_promoted(record)

    def test_policy_fingerprint_tracks_meaningful_policy_changes(self):
        original = direction_policy_fingerprint()
        self.assertEqual(original, direction_policy_fingerprint())
        with patch("analyzer.direction.DIRECTION_PROMPT", "Changed interpretation"):
            self.assertNotEqual(original, direction_policy_fingerprint())
        self.assertEqual(len(original), 64)


if __name__ == "__main__":
    unittest.main()
