"""Outerwear pass integration boundaries, actual source PTS, and no GPU calls."""

from copy import deepcopy
from fractions import Fraction
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import av
from PIL import Image

from analyzer.observations import ObservationDeadline
from analyzer.outerwear import OuterwearObserver, evaluate_outerwear, outerwear_policy_fingerprint
from analyzer.outerwear_cascade import (CascadeOuterwearObserver,
                                       cascade_policy_fingerprint, validate_cascade_record)
from analyzer.outerwear_pass import observe_outerwear_video
from analyzer.video import VideoFrame
from tests.test_outerwear import Client, Clock, primary, inventory


class FakeClient:
    timeout = 20.0

    def __init__(self, on_request=None):
        self.requests = []
        self.on_request = on_request

    def ask_image(self, frame, prompt, schema, **kwargs):
        self.requests.append((frame, kwargs))
        if self.on_request:
            self.on_request()
        return {"observation": {"removed_outerwear": "no"}, "finish_reason": "stop", "valid_json": True}


class CascadeClient(Client):
    def __init__(self, size, responses=(), **kwargs):
        super().__init__(responses, **kwargs)
        self.runtime = {"model_path": f"Qwen3VL-{size}B-Instruct-Q4_K_M.gguf",
                        "model_alias": "forte-qwen", "model_ftype": "Q4_K - Medium",
                        "build_info": "frozen-build" if size == 4 else "sha256:" + "a"*64,
                        "modalities": {"vision": True, "audio": False}, "total_slots": 1}
        self.info_calls = 0

    def server_info(self):
        self.info_calls += 1
        return deepcopy(self.runtime)


def cascade(first, second=(), **kwargs):
    a, b = CascadeClient(4, first, **kwargs), CascadeClient(8, second, **kwargs)
    observer = CascadeOuterwearObserver(a, b, expected_runtimes={"filter4b": a.runtime, "review8b": b.runtime})
    return observer, a, b


def samples():
    frames = []
    for index, timestamp in enumerate([0.0, 2.1, 5.0]):
        data = BytesIO()
        Image.new("RGB", (64, 48), (40*index, 20, 180)).save(data, format="JPEG", quality=90)
        frames.append(VideoFrame(timestamp, data.getvalue(), 64, 48))
    return {"frames": frames, "duration_sec": 5.001, "max_sample_gap_sec": 2.9,
            "sampling_capped": False, "extraction_seconds": .01}


def check(frame, frame_id, state="no", status="ok", errors=None):
    return {"frame_id": frame_id, "timestamp_sec": frame.timestamp_sec, "state": state,
            "status": status, "errors": [] if errors is None else errors, "calls": []}


class OuterwearPassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "observations"
        self.directory.mkdir()
        (self.directory / "frames").mkdir()
        self.cash_report = self.directory / "report.json"
        self.cash_report.write_bytes(b'{"status":"completed","cash":"keep"}')
        self.cash_image = self.directory / "frames/frame_0000.jpg"
        self.cash_image.write_bytes(b"existing cash evidence")
        self.video = self.root / "video.mkv"

    def assert_saved(self, report):
        saved = json.loads((self.directory / "outerwear-report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertEqual(self.cash_report.read_bytes(), b'{"status":"completed","cash":"keep"}')
        self.assertEqual(self.cash_image.read_bytes(), b"existing cash evidence")
        self.assertFalse(list(self.directory.glob(".outerwear-*.tmp")))
        self.assertNotIn("status", report)
        self.assertNotIn("errors", report)
        self.assertNotIn("verdict", report)
        for row in report["outerwear_checks"]:
            payload = (self.directory / row["image_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["image_sha256"])
            with Image.open(BytesIO(payload)) as picture:
                self.assertEqual(picture.format, "JPEG")

    def test_actual_variable_pts_and_jpeg_bytes_are_preserved_in_existing_observation_root(self):
        with av.open(str(self.video), mode="w") as output:
            stream = output.add_stream("ffv1", rate=1000)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate([5000, 5200, 5900, 7100, 9500, 10000]):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (index*30, 20, 180)))
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        client = FakeClient()
        report = observe_outerwear_video(self.video, self.directory, client=client, deadline=time.perf_counter()+30)
        self.assertEqual(report["outerwear_status"], "completed")
        self.assertEqual(report["selected_outerwear_timestamps"], [0, 2.1, 5])
        self.assertEqual(report["selected_outerwear_frame_ids"], [f"outerwear_frame_{i:04d}" for i in range(3)])
        self.assertTrue(report["all_selected_outerwear_frames_processed"])
        self.assertTrue(evaluate_outerwear(report)["can_conclude_no_candidate"])
        reference = {}
        with av.open(str(self.video)) as source:
            origin = None
            for frame in source.decode(video=0):
                stamp = frame.pts * Fraction(frame.time_base)
                if origin is None:
                    origin = stamp
                encoded = BytesIO()
                frame.to_image().convert("RGB").save(encoded, format="JPEG", quality=90)
                reference[float(stamp-origin)] = encoded.getvalue()
        for row, (frame, _) in zip(report["outerwear_checks"], client.requests):
            self.assertEqual((self.directory/row["image_path"]).read_bytes(), reference[row["timestamp_sec"]])
            self.assertEqual(frame.jpeg, reference[row["timestamp_sec"]])
        self.assert_saved(report)

    def test_shared_deadline_and_fixed_sampling_settings_are_forwarded(self):
        client = FakeClient()
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()) as sample, \
                patch("analyzer.outerwear_pass.time.perf_counter", return_value=10):
            report = observe_outerwear_video(self.video, self.directory, client=client, deadline=17)
        sample.assert_called_once_with(self.video, max_frames=40, interval_sec=4.0, max_side=1280, deadline=17)
        self.assertEqual([request[1]["timeout"] for request in client.requests], [7, 7, 7])
        self.assert_saved(report)

    def test_expired_budget_stops_before_decode_or_model_request(self):
        client = FakeClient()
        with patch("analyzer.outerwear_pass.sample_video") as sample, \
                patch("analyzer.outerwear_pass.time.perf_counter", return_value=10):
            report = observe_outerwear_video(self.video, self.directory, client=client, deadline=10)
        sample.assert_not_called()
        self.assertEqual(client.requests, [])
        self.assertEqual(report["outerwear_status"], "failed")
        self.assertTrue(report["outerwear_budget_exceeded"])
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_decode_that_finishes_late_cannot_start_inference(self):
        now = [10]
        client = FakeClient()
        def decode(*args, **kwargs):
            now[0] = 20
            return samples()
        with patch("analyzer.outerwear_pass.sample_video", side_effect=decode), \
                patch("analyzer.outerwear_pass.time.perf_counter", side_effect=lambda: now[0]):
            report = observe_outerwear_video(self.video, self.directory, client=client, deadline=20)
        self.assertEqual(client.requests, [])
        self.assertEqual(report["outerwear_checks"], [])
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assert_saved(report)

    def test_real_observer_late_negative_remains_uncertain_and_stops_following_frames(self):
        now = [10]
        client = FakeClient(on_request=lambda: now.__setitem__(0, 20))
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()), \
                patch("analyzer.outerwear_pass.time.perf_counter", side_effect=lambda: now[0]):
            report = observe_outerwear_video(self.video, self.directory, client=client, deadline=20)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(report["outerwear_status"], "partial")
        row, = report["outerwear_checks"]
        self.assertEqual((row["state"], row["status"], row["frame_id"]), ("uncertain", "partial", "outerwear_frame_0000"))
        self.assertTrue(report["outerwear_budget_exceeded"])
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_wrapper_rejects_late_answer_even_if_observer_forgets_its_deadline(self):
        for state in ("yes", "no"):
            with self.subTest(state=state):
                out = self.root/state
                out.mkdir()
                now = [10]
                def observe(frame, frame_id, **kwargs):
                    now[0] = 20
                    return check(frame, frame_id, state=state)
                with patch("analyzer.outerwear_pass.sample_video", return_value=samples()), \
                        patch("analyzer.outerwear_pass.OuterwearObserver") as observer, \
                        patch("analyzer.outerwear_pass.time.perf_counter", side_effect=lambda: now[0]):
                    observer.return_value.observe.side_effect = observe
                    report = observe_outerwear_video(self.video, out, client=FakeClient(), deadline=20)
                self.assertEqual(report["outerwear_checks"][0]["state"], "uncertain")
                self.assertEqual(observer.return_value.observe.call_count, 1)
                self.assertFalse(report["all_selected_outerwear_frames_processed"])

    def test_timeout_record_cannot_borrow_the_previous_frames_identity_or_evidence(self):
        frames = samples()
        original = check(frames["frames"][0], "outerwear_frame_0000")
        with patch("analyzer.outerwear_pass.sample_video", return_value=frames), \
                patch("analyzer.outerwear_pass.OuterwearObserver") as observer:
            observer.return_value.observe.side_effect = [deepcopy(original), ObservationDeadline("timeout", partial_record=original)]
            report = observe_outerwear_video(self.video, self.directory, client=FakeClient(), deadline=time.perf_counter()+30)
        self.assertEqual(len(report["outerwear_checks"]), 1)
        self.assertEqual(report["outerwear_checks"][0]["frame_id"], "outerwear_frame_0000")
        self.assertTrue(any("current saved frame" in error["message"] for error in report["outerwear_errors"]))
        self.assertNotIn("image_path", original)
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_existing_outerwear_image_is_preserved_and_preflight_prevents_any_new_calls(self):
        collision = self.directory/"frames/outerwear_frame_0001.jpg"
        collision.write_bytes(b"old evidence")
        client = FakeClient()
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()):
            report = observe_outerwear_video(self.video, self.directory, client=client, deadline=time.perf_counter()+30)
        self.assertEqual(collision.read_bytes(), b"old evidence")
        self.assertFalse((self.directory/"frames/outerwear_frame_0000.jpg").exists())
        self.assertEqual(client.requests, [])
        self.assertEqual(report["outerwear_status"], "failed")
        self.assert_saved(report)

    def test_previous_outerwear_report_is_never_overwritten(self):
        path = self.directory/"outerwear-report.json"
        path.write_bytes(b"previous report")
        with patch("analyzer.outerwear_pass.sample_video") as sample, self.assertRaises(FileExistsError):
            observe_outerwear_video(self.video, self.directory, client=FakeClient(), deadline=time.perf_counter()+30)
        sample.assert_not_called()
        self.assertEqual(path.read_bytes(), b"previous report")

    def test_returned_technical_negative_stops_pass_and_never_becomes_clean_negative(self):
        data = samples()
        bad = check(data["frames"][0], "outerwear_frame_0000", status="partial", errors=[{"message":"decode response error"}])
        with patch("analyzer.outerwear_pass.sample_video", return_value=data), \
                patch("analyzer.outerwear_pass.OuterwearObserver") as observer:
            observer.return_value.observe.return_value = bad
            report = observe_outerwear_video(self.video, self.directory, client=FakeClient(), deadline=time.perf_counter()+30)
        self.assertEqual(observer.return_value.observe.call_count, 1)
        self.assertEqual(report["outerwear_checks"][0]["state"], "uncertain")
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assertEqual(bad["state"], "no")
        self.assert_saved(report)

    def test_non_video_input_produces_failed_report_without_fake_checks(self):
        self.video.write_bytes(b"not a video")
        client = FakeClient()
        report = observe_outerwear_video(self.video, self.directory, client=client, deadline=time.perf_counter()+30)
        self.assertEqual(report["outerwear_status"], "failed")
        self.assertEqual(report["outerwear_checks"], [])
        self.assertTrue(report["outerwear_errors"])
        self.assertEqual(client.requests, [])
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_empty_sample_pool_cannot_claim_completion(self):
        with patch("analyzer.outerwear_pass.sample_video", return_value={"frames":[]}):
            report = observe_outerwear_video(self.video, self.directory, client=FakeClient(), deadline=time.perf_counter()+30)
        self.assertEqual(report["outerwear_status"], "failed")
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assert_saved(report)

    def test_slow_atomic_report_replacement_is_marked_partial_before_return(self):
        now = [10]
        original_replace = Path.replace
        def slow_replace(path, target):
            result = original_replace(path, target)
            now[0] = 20
            return result
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()), \
                patch("analyzer.outerwear_pass.time.perf_counter", side_effect=lambda: now[0]), \
                patch.object(Path, "replace", slow_replace):
            report = observe_outerwear_video(self.video, self.directory, client=FakeClient(), deadline=20)
        self.assertEqual(len(report["outerwear_checks"]), 3)
        self.assertEqual(report["outerwear_status"], "partial")
        self.assertTrue(report["outerwear_budget_exceeded"])
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_explicit_original_observer_keeps_the_default_report_byte_equivalent(self):
        outputs = []
        for name, explicit in (("default", False), ("explicit", True)):
            directory = self.root/name
            directory.mkdir()
            client = FakeClient()
            with (patch("analyzer.outerwear_pass.sample_video", return_value=samples()),
                  patch("analyzer.outerwear_pass.time.perf_counter", return_value=10)):
                report = observe_outerwear_video(self.video, directory, client=client, deadline=20,
                    observer=OuterwearObserver(client) if explicit else None)
            outputs.append(report)
            self.assertNotIn("outerwear_observer_kind", report)
            self.assertNotIn("outerwear_expected_runtimes", report)
            self.assertEqual(report["outerwear_policy_sha256"], outerwear_policy_fingerprint())
        self.assertEqual(outputs[0], outputs[1])

    def test_unknown_observer_and_subclasses_rejected_before_sampling(self):
        class Unsupported(OuterwearObserver):
            pass
        for observer in (object(), Unsupported(FakeClient())):
            with (self.subTest(observer=type(observer).__name__),
                  patch("analyzer.outerwear_pass.sample_video") as sample,
                  self.assertRaises(TypeError)):
                observe_outerwear_video(self.video, self.directory, client=FakeClient(),
                                        deadline=time.perf_counter()+30, observer=observer)
            sample.assert_not_called()
        self.assertFalse((self.directory/"outerwear-report.json").exists())

    def test_clean_cascade_records_have_exact_runtime_policy_and_source_evidence(self):
        observer, a, b = cascade([primary("no")]*3)
        expected = deepcopy(observer.expected_runtimes)
        data = samples()
        unused_client = FakeClient()
        with patch("analyzer.outerwear_pass.sample_video", return_value=data):
            report = observe_outerwear_video(self.video, self.directory, client=unused_client,
                                            deadline=time.perf_counter()+30, observer=observer)
        self.assertEqual(report["outerwear_status"], "completed")
        self.assertEqual(report["outerwear_observer_kind"], "inactive_outerwear_cascade_v1")
        self.assertEqual(report["outerwear_expected_runtimes"], expected)
        self.assertEqual(report["outerwear_policy_sha256"], cascade_policy_fingerprint())
        self.assertEqual((len(a.calls), b.info_calls, unused_client.requests), (3, 0, []))
        for frame, row in zip(data["frames"], report["outerwear_checks"]):
            original = deepcopy(row)
            original.pop("image_path")
            self.assertEqual(validate_cascade_record(original, frame, row["frame_id"],
                expected_runtimes=expected), original)
        observer.expected_runtimes["review8b"]["build_info"] = "mutated-after-return"
        self.assertEqual(report["outerwear_expected_runtimes"], expected)
        self.assert_saved(report)

    def test_cascade_review_positive_preserves_both_original_stages(self):
        observer, a, b = cascade([primary("uncertain")], [primary("yes"), inventory()])
        data = samples(); data["frames"] = data["frames"][:1]
        with patch("analyzer.outerwear_pass.sample_video", return_value=data):
            report = observe_outerwear_video(self.video, self.directory, client=a,
                                            deadline=time.perf_counter()+30, observer=observer)
        row, = report["outerwear_checks"]
        self.assertEqual((row["state"], row["selected_stage"]), ("yes", "review8b"))
        self.assertEqual([stage["name"] for stage in row["stages"]], ["filter4b", "review8b"])
        self.assertEqual((len(a.calls),len(b.calls),len(row["calls"])), (1,2,3))
        self.assertNotIn("verdict", report)
        self.assert_saved(report)

    def test_invalid_expected_runtime_rejects_before_decode_and_wrong_actual_runtime_is_partial(self):
        observer, a, b = cascade([primary("no")]*3)
        observer.expected_runtimes["review8b"]["model_path"] = "wrong.gguf"
        with patch("analyzer.outerwear_pass.sample_video") as sample, self.assertRaises(ValueError):
            observe_outerwear_video(self.video, self.directory, client=a,
                                    deadline=time.perf_counter()+30, observer=observer)
        sample.assert_not_called()
        self.assertFalse((self.directory/"outerwear-report.json").exists())
        observer, a, b = cascade([primary("no")]*3)
        a.runtime["build_info"] = "changed"
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()):
            report = observe_outerwear_video(self.video, self.directory, client=a,
                                            deadline=time.perf_counter()+30, observer=observer)
        self.assertEqual(report["outerwear_status"], "partial")
        self.assertEqual((len(a.calls),len(b.calls)), (0,0))
        self.assertEqual(report["outerwear_checks"][0]["state"], "uncertain")
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assert_saved(report)

    def test_forged_clean_cascade_never_becomes_a_saved_usable_observation(self):
        def change_runtime(row):
            row["stages"][0]["runtime"]["build_info"] = "changed"
        def change_stage_source(row):
            row["stages"][0]["frame_sha256"] = "0"*64
        mutations = [lambda r: r.update(image_sha256="0"*64), lambda r: r.update(frame_id="other"),
                     lambda r: r.update(timestamp_sec=123), lambda r: r.update(policy_sha256="0"*64),
                     lambda r: r.update(kind="other"), change_runtime, change_stage_source,
                     lambda r: r["calls"].clear(), lambda r: r.update(state="yes")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                output = self.root/f"forged-{index}"; output.mkdir()
                observer, a, b = cascade([primary("no")]*3)
                original = observer.observe
                def corrupt(frame, frame_id, **kwargs):
                    row = original(frame, frame_id, **kwargs)
                    mutate(row)
                    return row
                with (patch("analyzer.outerwear_pass.sample_video", return_value=samples()),
                      patch.object(observer,"observe",side_effect=corrupt)):
                    report = observe_outerwear_video(self.video, output, client=a,
                                                    deadline=time.perf_counter()+30, observer=observer)
                self.assertEqual(report["outerwear_status"], "failed")
                self.assertEqual(report["outerwear_checks"], [])
                self.assertFalse(report["all_selected_outerwear_frames_processed"])
                self.assertEqual(len(a.calls),1)

    def test_runtime_snapshot_cannot_be_rewritten_by_observer_during_the_pass(self):
        observer, a, b = cascade([primary("no")]*3)
        expected = deepcopy(observer.expected_runtimes)
        original = observer.observe
        def rewrite(frame, frame_id, **kwargs):
            row = original(frame, frame_id, **kwargs)
            observer.expected_runtimes["filter4b"]["build_info"] = "changed"
            row["stages"][0]["runtime"]["build_info"] = "changed"
            return row
        with (patch("analyzer.outerwear_pass.sample_video", return_value=samples()),
              patch.object(observer,"observe",side_effect=rewrite)):
            report = observe_outerwear_video(self.video, self.directory, client=a,
                                            deadline=time.perf_counter()+30, observer=observer)
        self.assertEqual(report["outerwear_status"], "failed")
        self.assertEqual(report["outerwear_expected_runtimes"], expected)
        self.assertEqual(report["outerwear_checks"], [])
        self.assert_saved(report)

    def test_cascade_semantic_failure_preserves_only_partial_diagnostics_and_stops(self):
        observer, a, b = cascade([primary("uncertain")]*3, [primary("invalid enum")])
        with patch("analyzer.outerwear_pass.sample_video", return_value=samples()):
            report = observe_outerwear_video(self.video, self.directory, client=a,
                                            deadline=time.perf_counter()+30, observer=observer)
        row, = report["outerwear_checks"]
        self.assertEqual((row["status"],row["state"],report["outerwear_status"]), ("partial","uncertain","partial"))
        self.assertTrue(row["errors"])
        self.assertEqual((len(a.calls),len(b.calls)), (1,1))
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assertFalse(evaluate_outerwear(report)["can_conclude_no_candidate"])
        self.assert_saved(report)

    def test_cascade_late_model_response_keeps_current_frame_partial_and_no_next_call(self):
        clock = Clock()
        a = CascadeClient(4,[primary("uncertain")]*3,clock=clock)
        b = CascadeClient(8,[primary("yes"),inventory()],clock=clock,durations=[20])
        observer = CascadeOuterwearObserver(a,b,expected_runtimes={"filter4b":a.runtime,"review8b":b.runtime})
        with (patch("analyzer.outerwear_pass.sample_video",return_value=samples()),
              patch("analyzer.outerwear_pass.time.perf_counter",side_effect=clock)):
            report = observe_outerwear_video(self.video,self.directory,client=a,deadline=120,observer=observer)
        row, = report["outerwear_checks"]
        self.assertEqual((row["frame_id"],row["state"],row["status"]), ("outerwear_frame_0000","uncertain","partial"))
        self.assertTrue(report["outerwear_budget_exceeded"])
        self.assertEqual((len(a.calls),len(b.calls)),(1,1))
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assert_saved(report)

    def test_wrapper_downgrades_fully_valid_cascade_positive_returned_after_deadline(self):
        clock = Clock()
        observer, a, b = cascade([primary("uncertain")], [primary("yes"), inventory()])
        original = observer.observe
        raw = []
        def late(frame, frame_id, **kwargs):
            row = original(frame, frame_id, **kwargs)
            raw.append(deepcopy(row))
            clock.now = 120
            return row
        with (patch("analyzer.outerwear_pass.sample_video",return_value=samples()),
              patch("analyzer.outerwear_pass.time.perf_counter",side_effect=clock),
              patch.object(observer,"observe",side_effect=late)):
            report = observe_outerwear_video(self.video,self.directory,client=a,deadline=120,observer=observer)
        row, = report["outerwear_checks"]
        self.assertEqual((raw[0]["state"],raw[0]["status"]),("yes","ok"))
        self.assertEqual((row["state"],row["status"]),("uncertain","partial"))
        self.assertTrue(report["outerwear_budget_exceeded"])
        self.assertFalse(report["all_selected_outerwear_frames_processed"])
        self.assertEqual((len(a.calls),len(b.calls)),(1,2))
        self.assert_saved(report)


if __name__ == "__main__":
    unittest.main()
