"""Scoped cash omission and orchestration deadlines, without models or a GPU."""

from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer import observations
from analyzer.features import CASH_FEATURES
from analyzer.observe import observe_video
from analyzer.observations import ImageObserver, ObservationDeadline, policy_fingerprint
from analyzer.rules import evaluate_rules
from analyzer.video import VideoFrame


def answer(value):
    return {"observation": value, "finish_reason": "stop", "valid_json": True}


def frame(timestamp=0):
    image = BytesIO()
    Image.new("RGB", (100, 80), (80, 120, 160)).save(image, "JPEG")
    return VideoFrame(timestamp, image.getvalue(), 100, 80)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Client:
    timeout = 20.0

    def __init__(self, responses=(), *, clock=None, image_seconds=0,
                 metadata_seconds=0, metadata_error=None):
        self.responses = list(responses)
        self.calls = []
        self.clock = clock
        self.image_seconds = image_seconds
        self.metadata_seconds = metadata_seconds
        self.metadata_error = metadata_error
        self.metadata_timeouts = []

    def server_info(self):
        self.metadata_timeouts.append(self.timeout)
        if self.clock:
            self.clock.now += self.metadata_seconds
        if self.metadata_error:
            raise self.metadata_error
        return {"model_path": "synthetic.gguf"}

    def ask_image(self, picture, prompt, schema, **kwargs):
        self.calls.append({"timestamp": picture.timestamp_sec, "prompt": prompt, **kwargs})
        if self.clock:
            self.clock.now += self.image_seconds
        if not self.responses:
            raise AssertionError("Unexpected request: " + prompt)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return answer(value)


PRESENT = [{"position": "front"}, {"person_in_work_area": "yes"}]
DESK = {"visible": "yes", "bbox": [200, 200, 800, 800]}


class CashShortCircuitTests(unittest.TestCase):
    def observe(self, person=PRESENT, *, skip=True, cabinet_mode="disabled",
                cabinet_responses=(), cash="yes", omit_option=False):
        primary = Client([*cabinet_responses, {"cash_on_worktop": cash}])
        activity = Client([*person, DESK, {"resting_on_desk": cash}])
        options = {} if omit_option else {"skip_cash_when_person_present": skip}
        with patch("analyzer.observations.time.perf_counter", return_value=100):
            row = ImageObserver(primary, cabinet_mode, activity_client=activity,
                                **options).observe(frame(), "f0", deadline=200)
        return row, primary, activity

    def test_confirmed_presence_skips_all_desk_and_cash_queries_without_claiming_no_cash(self):
        row, primary, activity = self.observe()
        self.assertEqual(row["states"]["person_in_work_area"], "yes")
        self.assertEqual(row["cash_consistency"], "not_needed_current_rule")
        self.assertEqual(row["cash_skip_rule_id"], "cash_visible_after_person_disappears")
        self.assertEqual([call["question"] for call in row["calls"]], ["person_position", "person_side"])
        self.assertEqual(primary.calls, [])
        self.assertEqual(len(activity.calls), 2)
        self.assertTrue(all(row["states"][key] == "uncertain" for key in CASH_FEATURES))
        self.assertIsNone(row["desk_bbox_normalized"])
        self.assertIsNone(row["desk_crop_pixel_box"])
        self.assertEqual((row["status"], row["errors"]), ("ok", []))
        report = {"observations": [row], "status": "completed", "errors": [],
                  "selected_frame_count": 1, "processed_frame_count": 1,
                  "all_selected_frames_processed": True, "selected_timestamps": [0]}
        evaluation = evaluate_rules(report)
        self.assertTrue(evaluation["can_conclude_no_candidate"])
        self.assertEqual(evaluation["candidates"], [])

    def test_disabled_and_omitted_option_keep_original_queries_and_results(self):
        implicit, first_primary, first_activity = self.observe(omit_option=True)
        explicit, second_primary, second_activity = self.observe(skip=False)
        self.assertEqual(implicit, explicit)
        self.assertEqual(first_primary.calls, second_primary.calls)
        self.assertEqual(first_activity.calls, second_activity.calls)
        self.assertEqual(explicit["states"]["cash_on_worktop"], "yes")
        self.assertEqual(explicit["cash_consistency"], "agreed")
        self.assertNotIn("cash_skip_rule_id", explicit)
        self.assertIn("desk_location", [call["question"] for call in explicit["calls"]])

    def test_absence_ambiguity_and_disagreement_keep_cash_queries(self):
        situations = [
            ([{"position": "absent"}, {"person_in_work_area": "no"}, {"doorway_activity": "no"}], "no"),
            ([{"position": "uncertain"}], "uncertain"),
            ([{"position": "doorway"}], "uncertain"),
            ([{"position": "front"}, {"person_in_work_area": "no"}], "uncertain"),
            ([{"position": "front"}, {"person_in_work_area": "uncertain"}], "uncertain"),
            ([{"position": "behind"}, {"person_in_work_area": "no"}, {"doorway_activity": "yes"}], "uncertain"),
        ]
        for person, expected in situations:
            with self.subTest(person=person):
                row, primary, _ = self.observe(person)
                self.assertEqual(row["states"]["person_in_work_area"], expected)
                self.assertEqual(row["states"]["cash_on_worktop"], "yes")
                self.assertEqual(len(primary.calls), 1)
                self.assertNotIn("cash_skip_rule_id", row)
                self.assertEqual(row["cash_consistency"], "agreed")

    def test_person_transport_or_parse_error_cannot_skip_cash_or_create_false_negative(self):
        for person in ([OSError("synthetic transport error")],
                       [{"position": "front"}, ValueError("invalid side answer")],
                       [{"position": "front", "unexpected": "yes"}]):
            with self.subTest(person=person):
                row, primary, _ = self.observe(person)
                self.assertEqual(row["states"]["person_in_work_area"], "uncertain")
                self.assertEqual(row["states"]["cash_on_worktop"], "yes")
                self.assertEqual(row["status"], "partial")
                self.assertEqual(row["errors"][0]["feature"], "person_in_work_area")
                self.assertEqual(len(primary.calls), 1)
                self.assertNotIn("cash_skip_rule_id", row)

    def test_cash_omission_does_not_skip_full_frame_or_context_cabinet_checks(self):
        for mode, responses, questions in (
            ("full_frame", [{"door_state": "open"}], ["cabinet_door_full_frame"]),
            ("context_crops", [DESK, {"door_state": "closed"}, {"door_state": "closed"}],
             ["cabinet_location", "cabinet_door_pad_0.15", "cabinet_door_pad_0.4"]),
        ):
            with self.subTest(mode=mode):
                row, primary, _ = self.observe(cabinet_mode=mode, cabinet_responses=responses)
                self.assertEqual([call["question"] for call in row["calls"]][2:], questions)
                self.assertEqual(len(primary.calls), len(responses))
                self.assertEqual(row["states"]["cabinet_door"], responses[-1]["door_state"])
                self.assertEqual(row["cash_consistency"], "not_needed_current_rule")

    def test_cabinet_error_is_preserved_and_never_disguised_by_cash_omission(self):
        row, primary, _ = self.observe(cabinet_mode="full_frame", cabinet_responses=[ValueError("bad door")])
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["errors"][0]["feature"], "cabinet_door")
        self.assertEqual(row["states"]["cabinet_door"], "uncertain")

    def test_late_positive_presence_is_not_accepted_or_marked_as_planned_cash_omission(self):
        clock = Clock()
        primary = Client()
        activity = Client(PRESENT, clock=clock, image_seconds=1)
        with patch("analyzer.observations.time.perf_counter", clock):
            observer = ImageObserver(primary, "disabled", activity_client=activity,
                                     skip_cash_when_person_present=True)
            with self.assertRaises(ObservationDeadline) as caught:
                observer.observe(frame(), "f0", deadline=102)
        row = caught.exception.partial_record
        self.assertEqual(row["states"]["person_in_work_area"], "uncertain")
        self.assertEqual(row["cash_consistency"], "not_checked")
        self.assertNotIn("cash_skip_rule_id", row)
        self.assertFalse(row["calls"][-1]["completed"])
        self.assertEqual(primary.calls, [])

    def test_invalid_flag_or_missing_activity_rejected_before_inference(self):
        for flag in (None, 0, 1, "yes", [], {}):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ImageObserver(Client(), activity_client=Client(), skip_cash_when_person_present=flag)
        with self.assertRaises(ValueError):
            ImageObserver(Client(), skip_cash_when_person_present=True)

    def test_previous_policy_hashes_remain_exactly_unchanged_when_disabled(self):
        # Frozen revision-6 values captured from the pre-change observer.
        baselines = {
            ("context_crops", False): "a93d21892306d5d2809d137c3be1e20e97e24b918ced0d6912ef37763a744202",
            ("context_crops", True): "6aee565e35221e588615ea4b8c28b988192958db9c183b8e82c481d5b380b6c2",
            ("full_frame", False): "fff41df09ab7053e7473819532230a6098e1c6209bbb3e0befe6fc2bfba164a9",
            ("full_frame", True): "dfcf92e23559a9e51267b744b02f1c2e47574c0b473c721849838261bc710e88",
            ("disabled", False): "4f0423372228679452bc80ee2948a024469366abcf536234cac9680536a89d3b",
            ("disabled", True): "ab75bc2087c3f0c54ce62107d6e1e1666ac1f5afc025f557f19ab582c51b3cd1",
        }
        for (mode, cash), expected in baselines.items():
            with self.subTest(mode=mode, cash=cash):
                self.assertEqual(policy_fingerprint(mode, cash_enabled=cash), expected)
                self.assertEqual(policy_fingerprint(mode, cash_enabled=cash,
                                                    skip_cash_when_person_present=False), expected)

    def test_enabled_policy_hash_includes_scope_without_affecting_legacy_policy(self):
        previous = policy_fingerprint("disabled", cash_enabled=True)
        enabled = policy_fingerprint("disabled", cash_enabled=True, skip_cash_when_person_present=True)
        self.assertNotEqual(previous, enabled)
        with patch.object(observations, "CASH_SHORT_CIRCUIT_RULE", "different_scope"):
            self.assertEqual(previous, policy_fingerprint("disabled", cash_enabled=True))
            self.assertNotEqual(enabled, policy_fingerprint("disabled", cash_enabled=True,
                                                           skip_cash_when_person_present=True))
        with self.assertRaises(ValueError):
            policy_fingerprint(skip_cash_when_person_present=True)


class NoGpuMonitor:
    samples = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class ObserveShortCircuitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.out = self.directory / "run"
        self.clock = Clock()
        self.primary = Client(clock=self.clock)
        self.activity = Client(PRESENT * 2, clock=self.clock)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("analyzer.observe.time.perf_counter", self.clock))
        self.stack.enter_context(patch("analyzer.observe.GpuMonitor", NoGpuMonitor))
        self.clients = self.stack.enter_context(patch("analyzer.observe.LocalModelClient",
                                                    side_effect=[self.primary, self.activity]))
        sampled = {"frames": [frame(0), frame(2)], "duration_sec": 2.1,
                   "sampling_capped": False, "extraction_seconds": 0, "max_sample_gap_sec": 2}
        self.sample = self.stack.enter_context(patch("analyzer.observe.sample_video", return_value=sampled))

    def run_video(self, **overrides):
        options = {"activity_base_url": "http://127.0.0.1:8081", "cabinet_mode": "disabled",
                   "max_refinement_frames": 0, "max_motion_windows": 0, "budget_sec": 10,
                   "skip_cash_when_person_present": True}
        options.update(overrides)
        return observe_video(self.directory / "input.mov", self.out, **options)

    def test_report_forwards_flag_records_scope_and_keeps_real_frames(self):
        report = self.run_video()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["settings"]["skip_cash_when_person_present"])
        self.assertEqual(report["cash_observation_scope"]["rule_id"], "cash_visible_after_person_disappears")
        self.assertEqual(report["policy_sha256"], policy_fingerprint(
            "disabled", cash_enabled=True, skip_cash_when_person_present=True))
        self.assertEqual(report["request_count"], 4)
        self.assertEqual(self.primary.calls, [])
        self.assertEqual(self.sample.call_args.kwargs["deadline"], 110)
        self.assertEqual(report["coarse_max_sample_gap_sec"], 2)
        self.assertEqual(report["coarse_coverage_blockers"], [])
        for row, original in zip(report["observations"], self.sample.return_value["frames"]):
            self.assertEqual(row["cash_consistency"], "not_needed_current_rule")
            self.assertEqual(row["timestamp_sec"], original.timestamp_sec)
            self.assertEqual((self.out / row["image_path"]).read_bytes(), original.jpeg)
        self.assertEqual(json.loads((self.out / "report.json").read_text(encoding="utf-8")), report)

    def test_incompatible_flags_rejected_before_any_clients_output_or_decoding(self):
        for override in ({"activity_base_url": None}, {"enable_actions": True},
                         {"skip_cash_when_person_present": 1}, {"skip_cash_when_person_present": None}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_video(**override)
        self.clients.assert_not_called()
        self.sample.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_runtime_timeout_uses_remaining_budget_and_is_restored(self):
        self.primary.metadata_seconds = 2
        report = self.run_video(budget_sec=5)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(self.primary.metadata_timeouts, [5])
        self.assertEqual(self.activity.metadata_timeouts, [3])
        self.assertEqual((self.primary.timeout, self.activity.timeout), (20, 20))
        self.assertEqual(self.sample.call_args.kwargs["deadline"], 105)
        self.assertTrue(all(call["timeout"] <= 3 for call in self.activity.calls))

    def test_late_runtime_metadata_cannot_continue_to_another_server_or_decoder(self):
        self.primary.metadata_seconds = 10
        report = self.run_video()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["errors"][0]["type"], "ObservationDeadline")
        self.assertNotIn("runtime", report)
        self.assertEqual(report["request_count"], 0)
        self.assertEqual(self.clients.call_count, 1)
        self.assertEqual(self.activity.metadata_timeouts, [])
        self.sample.assert_not_called()

    def test_runtime_error_reaching_deadline_is_not_swallowed_as_optional_metadata(self):
        self.primary.metadata_seconds = 10
        self.primary.metadata_error = OSError("synthetic late metadata failure")
        report = self.run_video()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["errors"][0]["type"], "ObservationDeadline")
        self.sample.assert_not_called()

    def test_second_metadata_call_reaching_deadline_cannot_start_decoding(self):
        self.primary.metadata_seconds = 2
        self.activity.metadata_seconds = 3
        report = self.run_video(budget_sec=5)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(self.activity.metadata_timeouts, [3])
        self.assertNotIn("runtime_activity", report)
        self.assertEqual(report["errors"][0]["type"], "ObservationDeadline")
        self.assertEqual(self.activity.calls, [])
        self.sample.assert_not_called()

    def test_expired_budget_before_metadata_makes_no_metadata_request(self):
        def delayed_client(*args, **kwargs):
            self.clock.now += 10
            return self.primary

        self.clients.side_effect = delayed_client
        report = self.run_video()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(self.primary.metadata_timeouts, [])
        self.sample.assert_not_called()

    def test_decode_deadline_is_preserved_as_partial_without_inference_or_negative(self):
        self.sample.side_effect = TimeoutError("synthetic decode deadline")
        report = self.run_video()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["errors"][0]["type"], "TimeoutError")
        self.assertEqual(report["request_count"], 0)
        self.assertEqual(self.sample.call_args.kwargs["deadline"], 110)
        self.assertFalse(evaluate_rules(report)["can_conclude_no_candidate"])

    def test_explicit_metadata_deadline_is_never_swallowed_even_before_clock_limit(self):
        self.primary.metadata_error = ObservationDeadline("synthetic deadline")
        report = self.run_video()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["errors"][0]["type"], "ObservationDeadline")
        self.sample.assert_not_called()

    def test_optional_metadata_endpoint_error_does_not_disable_valid_observation(self):
        self.primary.metadata_error = OSError("optional props unavailable")
        report = self.run_video()
        self.assertEqual(report["status"], "completed")
        self.assertIn("optional props unavailable", report["runtime_metadata_error"])
        self.assertEqual(report["errors"], [])

    def test_coarse_gap_diagnostic_is_recorded_without_relaxing_rule_limit(self):
        self.sample.return_value.update(frames=[frame(0), frame(6)], duration_sec=6.1,
                                        max_sample_gap_sec=6)
        report = self.run_video(max_frames=2)
        self.assertEqual(report["coarse_max_sample_gap_sec"], 6)
        self.assertEqual(report["coarse_rule_max_gap_sec"], 4)
        self.assertEqual(report["coarse_coverage_blockers"], ["coarse_gap_exceeds_rule_max_gap"])
        self.assertEqual(report["transitions"], [])
        self.assertFalse(evaluate_rules(report)["can_conclude_no_candidate"])


if __name__ == "__main__":
    unittest.main()
