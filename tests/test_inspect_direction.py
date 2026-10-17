"""Exercise replay integrity and partial reports with no GPU or network."""

from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.inspect_direction import inspect_direction
from analyzer.video import VideoFrame


def sha(data):
    return hashlib.sha256(data).hexdigest()


def answer(motion="output_to_hand", **overrides):
    text = json.dumps({"banknote_motion": motion, "evidence": "A visible bundle moves away."})
    return {"response_text": text, "observation": json.loads(text), "valid_json": True,
            "finish_reason": "stop", "usage": {"prompt_tokens": 123}, **overrides}


class FakeClient:
    timeout = 20.0

    def __init__(self):
        self.responses = []
        self.requests = []
        self.on_request = None

    def server_info(self):
        return {"model_path": "synthetic-direction.gguf"}

    def ask_sequence(self, frames, prompt, schema, **kwargs):
        self.requests.append((frames, prompt, schema, kwargs))
        if self.on_request:
            self.on_request(len(self.requests))
        result = self.responses.pop(0) if self.responses else answer()
        if isinstance(result, Exception):
            raise result
        return result


class NoGpu:
    instances = []

    def __init__(self):
        self.samples = [{"memory_mib": 123, "utilization_percent": 0}]
        self.exited = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.exited = True


class InspectDirectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old"
        self.old.mkdir()
        (self.old / "frames").mkdir()
        self.source_path = self.old / "report.json"
        self.out = self.root / "direction-new"
        self.video = self.root / "synthetic.mov"
        self.video.write_bytes(b"mock decoder source bytes")
        times = [i*.5 for i in range(31)]
        times[1:3] = [.47, 1.03]
        self.frames = []
        for i, timestamp in enumerate(times):
            encoded = BytesIO()
            Image.new("RGB", (64, 48), (i*7 % 256, 80, 140)).save(encoded, format="JPEG")
            self.frames.append(VideoFrame(timestamp, encoded.getvalue(), 64, 48))
        self.sampled = {"frames": self.frames, "duration_sec": 15.1, "extraction_seconds": .01,
                        "sampling_capped": False,
                        "max_sample_gap_sec": max(b-a for a, b in zip(times, times[1:]))}
        self.source = {
            "status": "completed", "video_path": str(self.video), "duration_sec": 15.1,
            "available_frame_count": len(times), "dense_max_sample_gap_sec": self.sampled["max_sample_gap_sec"],
            "policy_sha256": "source-policy", "action_policy_sha256": "source-action-policy",
            "observations": [{"timestamp_sec": times[0], "image_path": "frames/frame_0000.jpg"}],
            "counter_motion": {"scores": {"desk": [
                {"frame_index": i, "start_sec": times[i-1], "end_sec": times[i]}
                for i in range(1, len(times))]}},
            "action_checks": [self.action("a", [2, 4, 6]), self.action("b", [14, 16, 18])],
        }
        # Include an unreferenced stored image: it must also be verified.
        self.write_frame(0)
        self.write_frame(10)
        self.client = FakeClient()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.sampler = self.stack.enter_context(patch("analyzer.inspect_direction.sample_video", return_value=self.sampled))
        self.client_factory = self.stack.enter_context(patch("analyzer.inspect_direction.LocalModelClient", return_value=self.client))
        NoGpu.instances = []
        self.stack.enter_context(patch("analyzer.inspect_direction.GpuMonitor", NoGpu))

    def write_frame(self, index):
        relative = f"frames/frame_{index:04d}.jpg"
        frame = self.frames[index]
        (self.old / relative).write_bytes(frame.jpeg)
        return {"timestamp_sec": frame.timestamp_sec, "image_path": relative, "image_sha256": sha(frame.jpeg)}

    def action(self, name, indices):
        return {"window_id": name, "region": "desk", "status": "ok", "errors": [],
                "object_visibility": "yes", "object_bbox_normalized": [250, 250, 750, 750],
                "start_sec": self.frames[indices[0]].timestamp_sec,
                "end_sec": self.frames[indices[-1]].timestamp_sec,
                "result": {"interaction": "yes", "counter_direction": "UNTRUSTED_OLD_DIRECTION"},
                "calls": [{"response": {"response_text": "UNTRUSTED_OLD_RESPONSE"}}],
                "evidence": [self.write_frame(i) for i in indices]}

    def run_report(self, **kwargs):
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")
        self.before = self.source_path.read_bytes()
        return inspect_direction(self.source_path, self.out, **kwargs)

    def assert_saved(self, report):
        self.assertEqual(json.loads((self.out / "report.json").read_text(encoding="utf-8")), report)
        self.assertEqual((self.out / "source_report.json").read_bytes(), self.before)
        self.assertEqual(self.source_path.read_bytes(), self.before)
        self.assertEqual(report["source_report_sha256"], sha(self.before))
        self.assertFalse((self.out / "report.json.tmp").exists())
        self.assertNotIn("verdict", report)
        self.assertNotIn("violations", report)

    def assert_no_gpu(self, report):
        self.assertEqual(report["status"], "failed")
        self.client_factory.assert_not_called()
        self.assertEqual(self.client.requests, [])
        self.assertEqual(NoGpu.instances, [])
        self.assert_saved(report)

    def test_success_preserves_source_and_exact_five_actual_full_frames(self):
        report = self.run_report()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["selected_window_count"], 2)
        self.assertTrue(report["all_selected_windows_processed"])
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(report["model_label"], "synthetic-direction.gguf")
        self.sampler.assert_called_once_with(self.video.resolve(), max_frames=600, max_side=1280, interval_sec=.5)
        self.assertEqual(report["source_verification"]["verified_source_image_count"], 8)
        self.assertFalse(report["source_verification"]["source_video_hash_previously_recorded"])
        self.assertEqual(report["video_sha256"], sha(self.video.read_bytes()))
        for row in report["direction_checks"]:
            self.assertEqual(len(row["evidence"]), 5)
            self.assertEqual(row["result"], {"counter_direction": "uncertain"})
            self.assertEqual(row["raw_banknote_motion"], "output_to_hand")
            self.assertTrue(row["attempted"])
            for evidence in row["evidence"]:
                frame = self.frames[evidence["pool_index"]]
                self.assertEqual((self.out / evidence["image_path"]).read_bytes(), frame.jpeg)
                self.assertEqual(evidence["timestamp_sec"], frame.timestamp_sec)
                self.assertEqual(evidence["image_sha256"], sha(frame.jpeg))
        for _, prompt, schema, _ in self.client.requests:
            self.assertNotIn("UNTRUSTED", prompt + json.dumps(schema))
        self.assertTrue(NoGpu.instances[0].exited)
        self.assertEqual(report["peak_observed_device_memory_mib"], 123)
        self.assert_saved(report)

    def test_existing_destination_cannot_overwrite_old_files(self):
        self.out.mkdir()
        (self.out / "keep").write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_report()
        self.assertEqual((self.out / "keep").read_text(), "keep")
        self.client_factory.assert_not_called()

    def test_dense_timestamp_conflict_is_rejected_before_model_or_monitor(self):
        self.source["counter_motion"]["scores"]["desk"][1]["end_sec"] += .01
        self.assert_no_gpu(self.run_report())

    def test_incomplete_dense_pool_cannot_be_replaced_by_sparse_observations(self):
        self.source["counter_motion"]["scores"]["desk"] = []
        self.assert_no_gpu(self.run_report())

    def test_reconstructed_count_and_pts_must_match_not_just_duration(self):
        self.sampled["frames"] = self.frames[:-1]
        self.assert_no_gpu(self.run_report())

    def test_saved_evidence_hash_mismatch_even_in_nonselected_action_is_rejected(self):
        bad = self.action("irrelevant", [20, 22, 24])
        bad["result"]["interaction"] = "no"
        bad["evidence"][0]["image_sha256"] = "0" * 64
        self.source["action_checks"].append(bad)
        self.assert_no_gpu(self.run_report())

    def test_missing_action_evidence_hash_is_not_silently_trusted(self):
        self.source["action_checks"][0]["evidence"][0].pop("image_sha256")
        self.assert_no_gpu(self.run_report())

    def test_unreferenced_stored_jpeg_must_match_reconstructed_bytes(self):
        (self.old / "frames/frame_0010.jpg").write_bytes(self.frames[11].jpeg)
        self.assert_no_gpu(self.run_report())

    def test_external_evidence_path_is_rejected_before_model(self):
        self.source["action_checks"][0]["evidence"][0]["image_path"] = "../outside.jpg"
        self.assert_no_gpu(self.run_report())

    def test_recorded_video_hash_and_running_source_are_checked(self):
        self.source["video_sha256"] = "0" * 64
        report = self.run_report()
        self.assert_no_gpu(report)
        self.sampler.assert_not_called()

    def test_running_source_gets_snapshot_and_failed_report_without_decoding(self):
        self.source["status"] = "running"
        self.assert_no_gpu(self.run_report())
        self.sampler.assert_not_called()

    def test_timeout_second_window_keeps_first_result_and_actual_evidence(self):
        self.client.responses = [answer(), TimeoutError("synthetic timeout")]
        report = self.run_report()
        self.assertEqual(report["status"], "partial")
        self.assertEqual([row["status"] for row in report["direction_checks"]], ["ok", "partial"])
        self.assertEqual(report["direction_checks"][0]["raw_banknote_motion"], "output_to_hand")
        partial = report["direction_checks"][1]
        self.assertEqual(partial["result"]["counter_direction"], "uncertain")
        self.assertEqual(len(partial["evidence"]), 5)
        self.assertFalse(partial["calls"][0]["completed"])
        self.assertEqual(partial["calls"][0]["error"]["type"], "TimeoutError")
        self.assertFalse(report["all_selected_windows_processed"])
        self.assertTrue(NoGpu.instances[0].exited)
        self.assert_saved(report)

    def test_truncated_answer_retains_raw_response_without_promoting_direction(self):
        self.client.responses = [answer(finish_reason="length"), answer()]
        report = self.run_report()
        first = report["direction_checks"][0]
        self.assertEqual(report["status"], "partial")
        self.assertEqual(first["result"]["counter_direction"], "uncertain")
        self.assertEqual(first["calls"][0]["response"]["finish_reason"], "length")
        self.assert_saved(report)

    def test_decode_over_budget_stops_before_client_and_preserves_plan(self):
        now = [100.0]
        self.stack.enter_context(patch("analyzer.inspect_direction.time.perf_counter", side_effect=lambda: now[0]))
        def slow_decode(*args, **kwargs):
            now[0] = 111.0
            return self.sampled
        self.sampler.side_effect = slow_decode
        report = self.run_report(budget_sec=10)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertEqual(report["processed_window_count"], 0)
        self.assertEqual([x["status"] for x in report["direction_checks"]], ["not_started", "not_started"])
        self.client_factory.assert_not_called()
        self.assert_saved(report)

    def test_late_second_response_remains_raw_and_third_window_is_not_started(self):
        self.source["action_checks"].append(self.action("c", [26, 28, 30]))
        now = [100.0]
        self.stack.enter_context(patch("analyzer.inspect_direction.time.perf_counter", side_effect=lambda: now[0]))
        self.client.on_request = lambda count: now.__setitem__(0, 111.0 if count == 2 else 101.0)
        report = self.run_report(budget_sec=10)
        self.assertEqual(report["status"], "partial")
        self.assertEqual([x["status"] for x in report["direction_checks"]], ["ok", "partial", "not_started"])
        second = report["direction_checks"][1]
        self.assertIn("response_text", second["calls"][0]["response"])
        self.assertEqual(second["result"]["counter_direction"], "uncertain")
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(report["processed_window_count"], 2)
        self.assert_saved(report)

    def test_partial_source_keeps_coverage_limit_even_when_new_calls_succeed(self):
        self.source["status"] = "partial"
        report = self.run_report()
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["all_selected_windows_processed"])
        self.assertEqual(report["source_status"], "partial")
        self.assert_saved(report)

    def test_zero_cap_records_dropped_windows_without_creating_client(self):
        report = self.run_report(max_windows=0)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["direction_selection"]["capped"])
        self.assertEqual(len(report["direction_selection"]["dropped_windows"]), 2)
        self.client_factory.assert_not_called()
        self.assertEqual(report["request_count"], 0)
        self.assert_saved(report)

    def test_source_change_during_decode_is_detected_and_original_snapshot_remains(self):
        def mutate(*args, **kwargs):
            self.source_path.write_text('{"status":"running"}', encoding="utf-8")
            return self.sampled
        self.sampler.side_effect = mutate
        report = self.run_report()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_report_unchanged"])
        self.assertEqual((self.out / "source_report.json").read_bytes(), self.before)
        self.client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
