"""Replay integrity, isolation and shared deadline; no network, GPU or dataset."""

from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.inspect_short_actions import inspect_short_actions
from analyzer.local_client import LocalModelClient as RealLocalModelClient
from analyzer.video import VideoFrame


def sha(value):
    return hashlib.sha256(value).hexdigest()


def answer(value):
    return {"response_text": json.dumps(value), "observation": value,
            "valid_json": True, "finish_reason": "stop"}


class FakeClient:
    timeout = 20.

    def __init__(self):
        self.requests = []
        self.metadata_calls = []
        self.on_request = None
        self.on_metadata = None
        self.responses = []

    def server_info(self):
        self.metadata_calls.append(self.timeout)
        if self.on_metadata:
            self.on_metadata(len(self.metadata_calls))
        return {"model_path": "synthetic-local-model.gguf"}

    def _answer(self, images, prompt, schema, kwargs):
        self.requests.append((images, prompt, schema, kwargs))
        if self.on_request:
            self.on_request(len(self.requests))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if "interaction" in schema["properties"]:
            return answer({"interaction": "yes"})
        return answer({"hand_position": "top"})

    def ask_sequence(self, images, prompt, schema, **kwargs):
        return self._answer(images, prompt, schema, kwargs)

    def ask_image(self, image, prompt, schema, **kwargs):
        return self._answer(image, prompt, schema, kwargs)


class InspectShortActionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "original"
        (self.old / "frames").mkdir(parents=True)
        self.source_path = self.old / "report.json"
        self.out = self.root / "short-review"
        self.video = self.root / "arbitrary-synthetic-video.mov"
        self.video.write_bytes(b"bytes consumed by the fake decoder")
        self.frames = []
        for index in range(31):
            encoded = BytesIO()
            Image.new("RGB", (64, 48), (index*7 % 256, 80, 140)).save(encoded, format="JPEG")
            self.frames.append(VideoFrame(index*.5, encoded.getvalue(), 64, 48))
        times = [frame.timestamp_sec for frame in self.frames]
        self.sampled = {"frames": self.frames, "duration_sec": 15.1, "extraction_seconds": .01,
                        "sampling_capped": False, "max_sample_gap_sec": .5}
        self.source = {"status": "completed", "video_path": str(self.video), "duration_sec": 15.1,
            "available_frame_count": len(times), "dense_max_sample_gap_sec": .5,
            "observations": [self.write_frame(0)],
            "counter_motion": {"scores": {"desk": [
                {"frame_index": i, "start_sec": times[i-1], "end_sec": times[i]}
                for i in range(1, len(times))]}},
            "action_checks": [self.action("first", [0, 3, 6]), self.action("second", [14, 17, 20])]}
        self.write_frame(12)  # Stored but unreferenced: verification must still cover it.
        self.client = FakeClient()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.sampler = self.stack.enter_context(patch("analyzer.inspect_short_actions.sample_video", return_value=self.sampled))
        self.factory = self.stack.enter_context(patch("analyzer.inspect_short_actions.LocalModelClient", return_value=self.client))

    def write_frame(self, index):
        frame = self.frames[index]
        relative = f"frames/frame_{index:04d}.jpg"
        (self.old / relative).write_bytes(frame.jpeg)
        return {"timestamp_sec": frame.timestamp_sec, "image_path": relative, "image_sha256": sha(frame.jpeg)}

    def action(self, name, indices):
        return {"window_id": name, "region": "desk", "status": "ok", "errors": [],
            "object_visibility": "yes", "object_bbox_normalized": [250, 250, 750, 750],
            "start_sec": self.frames[indices[0]].timestamp_sec, "end_sec": self.frames[indices[-1]].timestamp_sec,
            "result": {"interaction": "uncertain", "counter_direction": "UNTRUSTED_OLD_DIRECTION"},
            "calls": [{"response": {"response_text": "UNTRUSTED_OLD_RESPONSE"}}],
            "evidence": [self.write_frame(i) for i in indices]}

    def run_report(self, **kwargs):
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")
        self.before = self.source_path.read_bytes()
        return inspect_short_actions(self.source_path, self.out, **kwargs)

    def assert_saved(self, report, unchanged=True):
        self.assertEqual(json.loads((self.out / "report.json").read_text(encoding="utf-8")), report)
        self.assertEqual((self.out / "source_report.json").read_bytes(), self.before)
        self.assertEqual(report["source_report_sha256"], sha(self.before))
        if unchanged:
            self.assertEqual(self.source_path.read_bytes(), self.before)
            self.assertTrue(report["source_report_unchanged"])
        self.assertFalse((self.out / "report.json.tmp").exists())
        self.assertNotIn("verdict", report)
        self.assertNotIn("violations", report)

    def assert_no_model(self, report):
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.client.metadata_calls, [])
        self.assert_saved(report)

    def test_success_preserves_source_and_saves_actual_original_jpegs(self):
        report = self.run_report(max_windows=2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["selected_window_count"], 2)
        self.assertEqual(report["processed_window_count"], 2)
        self.assertEqual(report["request_count"], 4)
        self.sampler.assert_called_once_with(self.video.resolve(), max_frames=600, max_side=1280, interval_sec=.5)
        for row in report["action_checks"]:
            self.assertEqual(row["result"]["interaction"], "yes")
            self.assertEqual(row["result"]["counter_direction"], "uncertain")
            self.assertEqual(row["result"]["cash_into_cabinet"], "uncertain")
            for evidence in row["evidence"]:
                frame = next(f for f in self.frames if f.timestamp_sec == evidence["timestamp_sec"])
                self.assertEqual((self.out / evidence["image_path"]).read_bytes(), frame.jpeg)
                self.assertEqual(evidence["image_sha256"], sha(frame.jpeg))
        for _, prompt, schema, _ in self.client.requests:
            self.assertNotIn("UNTRUSTED", prompt+json.dumps(schema))
        self.assert_saved(report)

    def test_new_negative_result_is_not_or_merged_with_positive_source(self):
        for row in self.source["action_checks"]:
            row["result"]["interaction"] = "yes"
        self.client.responses = [answer({"interaction": "no"}), answer({"hand_position": "away"})]
        report = self.run_report(max_windows=1)
        self.assertEqual(report["action_checks"][0]["result"]["interaction"], "no")
        self.assertTrue(all(row["result"]["interaction"] == "yes" for row in self.source["action_checks"]))
        self.assert_saved(report)

    def test_saved_evidence_hash_mismatch_blocks_all_model_calls(self):
        self.source["action_checks"][1]["evidence"][0]["image_sha256"] = "0"*64
        self.assert_no_model(self.run_report(max_windows=1))

    def test_missing_evidence_hash_blocks_model_calls(self):
        self.source["action_checks"][0]["evidence"][0].pop("image_sha256")
        self.assert_no_model(self.run_report())

    def test_unreferenced_stored_jpeg_also_must_match(self):
        (self.old / "frames/frame_0012.jpg").write_bytes(self.frames[13].jpeg)
        self.assert_no_model(self.run_report())

    def test_external_image_path_does_not_get_loaded_or_queried(self):
        self.source["action_checks"][0]["evidence"][0]["image_path"] = "../private.jpg"
        self.assert_no_model(self.run_report())

    def test_recorded_video_hash_mismatch_stops_before_decode(self):
        self.source["video_sha256"] = "0"*64
        self.assert_no_model(self.run_report())
        self.sampler.assert_not_called()

    def test_timestamp_conflict_cannot_silently_select_other_frames(self):
        self.source["counter_motion"]["scores"]["desk"][1]["end_sec"] += .01
        self.assert_no_model(self.run_report())

    def test_source_change_during_decode_is_detected_before_inference(self):
        def mutate(*args, **kwargs):
            self.source_path.write_text('{"status":"running"}', encoding="utf-8")
            return self.sampled
        self.sampler.side_effect = mutate
        report = self.run_report()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_report_unchanged"])
        self.assertEqual(self.client.requests, [])
        self.assert_saved(report, unchanged=False)

    def test_source_deletion_during_decode_still_saves_failure_and_snapshot(self):
        def remove(*args, **kwargs):
            self.source_path.unlink()
            return self.sampled
        self.sampler.side_effect = remove
        report = self.run_report()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_report_unchanged"])
        self.assertEqual(self.client.requests, [])
        self.assert_saved(report, unchanged=False)

    def test_zero_windows_performs_no_model_metadata_or_inference(self):
        report = self.run_report(max_windows=0)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["short_action_selection"]["capped"])
        self.assertEqual(report["action_checks"], [])
        self.assertEqual(report["request_count"], 0)
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.client.metadata_calls, [])
        self.assert_saved(report)

    def test_decode_past_deadline_stops_before_metadata_and_inference(self):
        now = [100.]
        self.stack.enter_context(patch("analyzer.inspect_short_actions.time.perf_counter", side_effect=lambda: now[0]))
        def slow_decode(*args, **kwargs):
            now[0] = 111.
            return self.sampled
        self.sampler.side_effect = slow_decode
        report = self.run_report(budget_sec=10)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.client.metadata_calls, [])
        self.assertEqual(report["processed_window_count"], 0)
        self.assert_saved(report)

    def test_late_metadata_does_not_start_second_metadata_or_inference(self):
        now = [100.]
        self.stack.enter_context(patch("analyzer.inspect_short_actions.time.perf_counter", side_effect=lambda: now[0]))
        self.client.on_metadata = lambda count: now.__setitem__(0, 111.)
        report = self.run_report(budget_sec=10)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(len(self.client.metadata_calls), 1)
        self.assertLessEqual(self.client.metadata_calls[0], 10)
        self.assertEqual(self.client.requests, [])
        self.assert_saved(report)

    def test_late_positive_is_retained_raw_not_accepted_and_later_windows_stop(self):
        self.source["action_checks"].append(self.action("third", [24, 27, 30]))
        now = [100.]
        self.stack.enter_context(patch("analyzer.inspect_short_actions.time.perf_counter", side_effect=lambda: now[0]))
        self.client.on_request = lambda count: now.__setitem__(0, 111. if count == 3 else 101.)
        report = self.run_report(budget_sec=10, max_windows=3)
        self.assertEqual(report["status"], "partial")
        self.assertEqual([r["status"] for r in report["action_checks"]], ["ok", "partial", "not_started"])
        late = report["action_checks"][1]
        self.assertEqual(late["result"]["interaction"], "uncertain")
        self.assertEqual(late["calls"][0]["response"]["observation"], {"interaction": "yes"})
        self.assertFalse(late["calls"][0]["completed"])
        self.assertEqual(len(self.client.requests), 3)
        self.assertEqual(report["processed_window_count"], 2)
        self.assert_saved(report)

    def test_response_at_exact_deadline_is_not_accepted(self):
        now = [100.]
        self.stack.enter_context(patch("analyzer.inspect_short_actions.time.perf_counter", side_effect=lambda: now[0]))
        self.client.on_request = lambda count: now.__setitem__(0, 110.)
        report = self.run_report(budget_sec=10, max_windows=1)
        self.assertEqual(report["action_checks"][0]["result"]["interaction"], "uncertain")
        self.assertEqual(report["status"], "partial")
        self.assertEqual(len(self.client.requests), 1)
        self.assert_saved(report)

    def test_exception_from_decoder_saves_failed_report_and_leaves_source(self):
        self.sampler.side_effect = RuntimeError("synthetic decode failure")
        report = self.run_report()
        self.assert_no_model(report)
        self.assertEqual(report["errors"][0]["type"], "RuntimeError")

    def test_model_exception_does_not_turn_into_negative_result(self):
        self.client.responses = [TimeoutError("synthetic inference timeout")]
        report = self.run_report(max_windows=1)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["action_checks"][0]["result"]["interaction"], "uncertain")
        self.assert_saved(report)

    def test_partial_source_keeps_coverage_limit_after_successful_replay(self):
        self.source["status"] = "partial"
        report = self.run_report(max_windows=1)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["action_checks"][0]["status"], "ok")
        self.assert_saved(report)

    def focused_sample(self, *args, **kwargs):
        # Synthetic decoder result at finer real-PTS values; no generated times
        # are substituted by the runner after this fixture returns its pool.
        frames = [VideoFrame(t, self.frames[i].jpeg, 64, 48)
                  for i, t in enumerate((0., .15, .3))]
        return {"frames": frames, "windows": [{"window_id": "short_action_0000", "frame_indices": [0, 1, 2]}],
                "sampling_interval_sec": .15}

    def test_focus_evidence_uses_finer_source_frames_and_keeps_original_selection(self):
        with patch("analyzer.focus_video.sample_video_intervals", side_effect=self.focused_sample), \
             patch("analyzer.focus_video.motion_peak_triple", return_value={"selected": True, "frame_indices": [0, 1, 2]}):
            report = self.run_report(max_windows=1, focus=True)
        row = report["action_checks"][0]
        self.assertEqual([e["timestamp_sec"] for e in row["evidence"]], [0., .15, .3])
        self.assertEqual(row["selection"]["frame_timestamps"], [0., .5, 1.])
        self.assertEqual(row["end_sec"], .3)
        self.assertEqual([f.timestamp_sec for f in self.client.requests[0][0]], [0., .15, .3])
        for index, evidence in enumerate(row["evidence"]):
            self.assertEqual((self.out / evidence["image_path"]).read_bytes(), self.frames[index].jpeg)
        self.assert_saved(report)

    def test_source_video_change_during_focus_blocks_inference(self):
        def mutate(*args, **kwargs):
            self.video.write_bytes(b"changed during finer extraction")
            return self.focused_sample()
        with patch("analyzer.focus_video.sample_video_intervals", side_effect=mutate), \
             patch("analyzer.focus_video.motion_peak_triple", return_value={"selected": False}):
            self.assert_no_model(self.run_report(max_windows=1, focus=True))

    def test_focus_decode_budget_exhaustion_stops_before_model(self):
        now = [100.]
        self.stack.enter_context(patch("analyzer.inspect_short_actions.time.perf_counter", side_effect=lambda: now[0]))
        def slow(*args, **kwargs):
            now[0] = 111.
            return self.focused_sample()
        with patch("analyzer.focus_video.sample_video_intervals", side_effect=slow), \
             patch("analyzer.focus_video.motion_peak_triple", return_value={"selected": False}):
            report = self.run_report(max_windows=1, focus=True, budget_sec=10)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.client.metadata_calls, [])
        self.assert_saved(report)

    def test_nonlocal_urls_are_rejected_before_source_or_video_reads(self):
        with patch("analyzer.inspect_short_actions.LocalModelClient", RealLocalModelClient):
            for field in ("grounding_url", "activity_url"):
                for url in ("https://api.example.org", "http://192.168.1.5:8080", "http://127.0.0.1@example.org:8080"):
                    with self.subTest(field=field, url=url), self.assertRaises(ValueError):
                        inspect_short_actions(self.root / "missing-report.json", self.out, **{field: url})
        self.sampler.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_existing_destination_is_not_overwritten(self):
        self.out.mkdir()
        marker = self.out / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_report()
        self.assertEqual(marker.read_text(), "preserve")
        self.sampler.assert_not_called()
        self.assertEqual(self.client.requests, [])


if __name__ == "__main__":
    unittest.main()
