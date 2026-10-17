"""Exercise optional action reservation with synthetic time, frames and answers.

The orchestration, frame selection, action checks, evidence writes and report
finalization are real. No local model server, video dataset or GPU is used.
"""

from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.observe import observe_video
from analyzer.observations import ImageObserver, ObservationDeadline
from analyzer.video import VideoFrame


# Keep the real method available for its deadline-boundary check; the broader
# orchestration fixtures replace per-frame inference with a deterministic scene.
_observe_image = ImageObserver.observe


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class NoGpuMonitor:
    samples = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class SyntheticClient:
    timeout = 20.0

    def __init__(self, clock):
        self.clock = clock
        self.requests = []
        self.sequence_seconds = .25
        self.image_seconds = .25

    def server_info(self):
        return {"model_path": "synthetic-model.gguf"}

    def _answer(self, frames, schema, kwargs, seconds):
        self.requests.append({"started": self.clock.now, "kwargs": kwargs,
                              "timestamps": [frame.timestamp_sec for frame in frames]})
        self.clock.now += seconds
        if set(schema["properties"]) == {"interaction"}:
            value = {"interaction": "yes"}
        elif set(schema["properties"]) == {"hand_position"}:
            value = {"hand_position": "front"}
        else:
            raise AssertionError("Unexpected real model request in budget fixture")
        return {"observation": value, "valid_json": True, "finish_reason": "stop"}

    def ask_sequence(self, frames, prompt, schema, **kwargs):
        return self._answer(frames, schema, kwargs, self.sequence_seconds)

    def ask_image(self, frame, prompt, schema, **kwargs):
        return self._answer([frame], schema, kwargs, self.image_seconds)


class ObserveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.out = self.directory / "report"
        self.video = self.directory / "synthetic.mov"
        self.clock = Clock()
        self.client = SyntheticClient(self.clock)
        self.frame_requests = []
        self.frame_seconds = .5
        self.discovery_seconds = .5
        self.windows = [[10, 12, 14]]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        frames = []
        for index in range(17):
            image = BytesIO()
            Image.new("RGB", (64, 48), (index * 10, 100, 140)).save(image, "JPEG")
            frames.append(VideoFrame(index / 2, image.getvalue(), 64, 48))
        self.frames = frames
        sampled = {"frames": frames, "duration_sec": 8.5, "extraction_seconds": 0,
                   "sampling_capped": False, "max_sample_gap_sec": .5}
        self.monitor = self.stack.enter_context(patch("analyzer.observe.GpuMonitor", return_value=NoGpuMonitor()))
        self.clients = self.stack.enter_context(patch("analyzer.observe.LocalModelClient", return_value=self.client))
        self.sample = self.stack.enter_context(patch("analyzer.observe.sample_video", return_value=sampled))
        self.stack.enter_context(patch("analyzer.observe.time.perf_counter", self.clock))
        self.stack.enter_context(patch("analyzer.observe.ImageObserver.observe", side_effect=self.observe_frame))
        self.discovery = self.stack.enter_context(patch("analyzer.observe.discover_counter", side_effect=self.discover))
        self.stack.enter_context(patch("analyzer.observe.propose_motion_windows", side_effect=self.motion))

    def observe_frame(self, frame, frame_id, *, deadline):
        self.frame_requests.append((frame_id, frame.timestamp_sec, self.clock.now, deadline))
        self.clock.now += self.frame_seconds
        return {"frame_id": frame_id, "timestamp_sec": frame.timestamp_sec, "status": "ok",
                "states": {"person_in_work_area": "yes",
                           "cabinet_door": "open" if 3.5 <= frame.timestamp_sec <= 4.5 else "closed",
                           "cash_on_worktop": "uncertain", "cash_in_hands": "uncertain",
                           "notes_in_machine": "uncertain"}, "calls": [], "errors": []}

    def discover(self, *args, **kwargs):
        self.clock.now += self.discovery_seconds
        return {"status": "ok", "observations": [], "calls": [], "errors": []}

    def motion(self, frames, observations, **kwargs):
        if "roi_padding" not in kwargs:
            return {"windows": [], "dropped_windows": [], "refinement": {"selection_order": []}}
        windows = []
        for number, indices in enumerate(self.windows, 1):
            times = [frames[index].timestamp_sec for index in indices]
            windows.append({"roi": "desk", "group_id": f"counter-{number}", "priority": number,
                            "frame_indices": indices, "frame_timestamps": times,
                            "start_sec": times[0], "end_sec": times[-1]})
        return {"windows": windows, "dropped_windows": [],
                "rois": {"desk": {"union_bbox_normalized": [250, 250, 750, 750]}}}

    def run_video(self, **overrides):
        options = {"interval_sec": 2, "budget_sec": 20, "enable_actions": True,
                   "activity_base_url": "http://127.0.0.1:8081"}
        options.update(overrides)
        return observe_video(self.video, self.out, **options)

    def assert_saved(self, report):
        self.assertEqual(json.loads((self.out / "report.json").read_text(encoding="utf-8")), report)
        self.assertNotIn("verdict", report)
        self.assertNotIn("violations", report)
        times = [row["timestamp_sec"] for row in report["observations"]]
        self.assertEqual(times, sorted(set(times)))
        for row in report["observations"]:
            index = int(row["frame_id"].split("_")[1])
            self.assertEqual(row["timestamp_sec"], self.frames[index].timestamp_sec)
            self.assertEqual((self.out / row["image_path"]).read_bytes(), self.frames[index].jpeg)

    def test_invalid_reserve_is_rejected_before_output_clients_or_monitor(self):
        invalid = [{"action_reserve_sec": value}
                   for value in (-1, True, float("nan"), float("inf"), "5", None, 20, 21)]
        invalid += [{"action_reserve_sec": 1, "enable_actions": False},
                    {"action_reserve_sec": 1, "max_action_windows": 0}]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_video(**options)
            self.assertFalse(self.out.exists())
        self.clients.assert_not_called()
        self.sample.assert_not_called()
        self.monitor.assert_not_called()
        self.assertEqual(self.frame_requests, [])

    def test_default_and_zero_reserve_keep_chronological_legacy_processing(self):
        reports, orders = [], []
        for number, options in enumerate(({}, {"action_reserve_sec": 0})):
            self.out = self.directory / str(number)
            self.frame_requests.clear()
            self.clock.now = 100
            reports.append(self.run_video(**options))
            orders.append([request[1] for request in self.frame_requests])
            self.assert_saved(reports[-1])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0][:5], [0, 2, 4, 6, 8])
        for report in reports:
            self.assertFalse(report["budget_control"]["enabled"])
            self.assertEqual(report["budget_control"]["skipped_refinement_indices"], [])
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["all_selected_frames_processed"])
        for field in ("selected_timestamps", "transitions", "observed_events", "coarse_processing_order"):
            self.assertEqual(reports[0][field], reports[1][field])

    def test_ample_reserve_covers_both_ends_first_and_retains_every_planned_refinement(self):
        report = self.run_video(action_reserve_sec=5)
        self.assertEqual([request[1] for request in self.frame_requests[:2]], [0, 8])
        self.assertEqual(set(request[1] for request in self.frame_requests[:5]), {0, 2, 4, 6, 8})
        self.assertTrue(report["refinement"]["indices"])
        self.assertEqual(report["budget_control"]["skipped_refinement_indices"], [])
        self.assertGreater(report["processed_frame_count"], report["coarse_frame_count"])
        self.assertTrue(report["all_selected_frames_processed"])
        self.assertTrue(report["all_selected_action_windows_processed"])
        self.assertEqual(report["status"], "completed")
        self.assertEqual([change["to_state"] for change in report["transitions"]], ["open", "closed"])
        self.assertEqual(self.discovery.call_args.kwargs["deadline"], 120)
        self.assertTrue(all(request[3] == 120 for request in self.frame_requests))
        self.assert_saved(report)

    def test_small_remaining_budget_skips_refinements_but_still_checks_actual_action_frames(self):
        report = self.run_video(action_reserve_sec=9, budget_sec=12)
        skipped = report["budget_control"]["skipped_refinement_indices"]
        self.assertTrue(skipped)
        self.assertEqual(set(skipped), set(report["combined_refinement"]["indices"]))
        self.assertEqual(report["processed_frame_count"], report["coarse_frame_count"])
        self.assertGreater(report["selected_frame_count"], report["processed_frame_count"])
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertTrue(report["all_selected_action_windows_processed"])
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["budget_exceeded"])
        self.assertIn("RefinementBudgetReserved", [error["type"] for error in report["errors"]])
        self.assertEqual(report["transitions"], [])  # One coarse open frame is not confirmation.
        self.assertEqual(report["observed_events"], [])
        self.assertTrue(all(row["states"]["cash_on_worktop"] == "uncertain" for row in report["observations"]))
        self.assertTrue(all(row["states"]["person_in_work_area"] == "yes" for row in report["observations"]))
        action = report["action_checks"][0]
        self.assertEqual(action["result"]["interaction"], "yes")
        for field in ("cash_near_cabinet", "cash_into_cabinet", "counter_direction"):
            self.assertEqual(action["result"][field], "uncertain")
        self.assertEqual([item["timestamp_sec"] for item in action["evidence"]], [5, 6, 7])
        observed = {row["frame_id"] for row in report["observations"]}
        self.assertFalse(any(f"frame_{index:04d}" in observed for index in skipped))
        self.assertEqual(len(self.client.requests), 2)
        self.assert_saved(report)

    def test_reserve_does_not_extend_deadline_if_discovery_consumes_remaining_budget(self):
        self.discovery_seconds = 9.5  # Five coarse frames cost 2.5; total reaches exactly 12.
        report = self.run_video(action_reserve_sec=9, budget_sec=12)
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["all_selected_action_windows_processed"])
        self.assertEqual(report["action_selection"]["selected_count"], 1)
        self.assertEqual(report["action_checks"], [])
        self.assertEqual(self.client.requests, [])
        self.assertIn("ObservationDeadline", [error["type"] for error in report["errors"]])
        self.assert_saved(report)

    def test_late_action_response_stays_uncertain_preserves_raw_and_blocks_next_window(self):
        self.windows = [[0, 1, 2], [10, 12, 14]]
        self.client.sequence_seconds = 10
        report = self.run_video(action_reserve_sec=9, budget_sec=12)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertFalse(report["all_selected_action_windows_processed"])
        self.assertEqual(report["action_selection"]["selected_count"], 2)
        self.assertEqual(len(report["action_checks"]), 1)
        self.assertEqual(len(self.client.requests), 1)
        request = self.client.requests[0]
        self.assertGreater(request["kwargs"]["timeout"], 0)
        self.assertLessEqual(request["kwargs"]["timeout"], 112 - request["started"])
        action = report["action_checks"][0]
        self.assertEqual(action["status"], "partial")
        self.assertTrue(all(value == "uncertain" for value in action["result"].values()))
        self.assertFalse(action["calls"][0]["completed"])
        self.assertEqual(action["calls"][0]["response"]["observation"], {"interaction": "yes"})
        self.assertEqual(len(action["evidence"]), 3)
        self.assert_saved(report)

    def test_action_answer_at_deadline_is_not_a_completed_observation(self):
        self.client.sequence_seconds = 1
        self.client.image_seconds = 8  # Coarse + discovery = 3; requests finish at 12 exactly.
        report = self.run_video(action_reserve_sec=9, budget_sec=12)
        self.assertEqual(self.clock.now, 112)
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["all_selected_action_windows_processed"])
        action = report["action_checks"][0]
        self.assertEqual(action["status"], "partial")
        self.assertEqual(action["result"]["interaction"], "uncertain")
        self.assertEqual(action["calls"][-1]["response"]["observation"], {"hand_position": "front"})
        self.assertFalse(action["calls"][-1]["completed"])
        self.assert_saved(report)

    def test_image_answer_at_deadline_preserves_prior_state_and_raw_but_not_late_state(self):
        responses = iter(((1, {"person_in_work_area": "yes"}), (9, {"door_state": "closed"})))
        timeouts = []

        def respond(frame, prompt, schema, **kwargs):
            seconds, observation = next(responses)
            timeouts.append(kwargs["timeout"])
            self.clock.now += seconds
            return {"observation": observation, "valid_json": True, "finish_reason": "stop"}

        with patch.object(self.client, "ask_image", side_effect=respond):
            with self.assertRaises(ObservationDeadline) as caught:
                _observe_image(ImageObserver(self.client, "full_frame"), self.frames[0],
                               "frame_0000", deadline=110)
        partial = caught.exception.partial_record
        self.assertEqual(self.clock.now, 110)
        self.assertEqual(timeouts, [10, 9])
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["states"], {"person_in_work_area": "yes", "cabinet_door": "uncertain"})
        self.assertEqual([call["completed"] for call in partial["calls"]], [True, False])
        self.assertEqual(partial["calls"][-1]["observation"], {"door_state": "closed"})
        self.assertEqual([error["feature"] for error in partial["errors"]], ["cabinet_door"])
        self.assertEqual(partial["errors"][0]["type"], "ObservationDeadline")


if __name__ == "__main__":
    unittest.main()
