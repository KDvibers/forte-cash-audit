"""Exercise report boundaries with a fake model and no GPU or network access."""

from contextlib import ExitStack
from io import BytesIO
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from analyzer.observe import observe_video
from analyzer.timeline import build_transitions
from analyzer.video import VideoFrame


def answer(value):
    return {"observation": value, "finish_reason": "stop", "valid_json": True}


class FakeModel:
    timeout = 20.0

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []

    def server_info(self):
        return {"model_path": "synthetic-model.gguf"}

    def ask_image(self, frame, prompt, schema, **kwargs):
        self.requests.append((frame, prompt, schema, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected model request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class SceneModel(FakeModel):
    """Deterministic scene answers, independent of coarse/refinement call order."""

    def __init__(self, door_at, on_request=None):
        super().__init__()
        self.door_at = door_at
        self.on_request = on_request

    def ask_image(self, frame, prompt, schema, **kwargs):
        self.requests.append((frame, prompt, schema, kwargs))
        keys = set(schema["properties"])
        if keys == {"person_in_work_area"}:
            question, result = "person", {"person_in_work_area": "yes"}
        elif keys == {"door_state"}:
            question, result = "door", {"door_state": self.door_at(frame.timestamp_sec)}
        elif keys == {"visible", "bbox"}:
            question, result = "location", {"visible": "yes", "bbox": [250, 250, 750, 750]}
        else:
            raise AssertionError(f"Unexpected model schema: {keys}")
        if self.on_request is not None:
            self.on_request(frame.timestamp_sec, question)
        return answer(result)


class NoGpuMonitor:
    def __init__(self):
        self.samples = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def sample_fixture(count, step=1.0):
    encoded = BytesIO()
    Image.new("RGB", (64, 48), (80, 120, 160)).save(encoded, format="JPEG")
    return {
        "frames": [VideoFrame(index * step, encoded.getvalue(), 64, 48) for index in range(count)],
        "duration_sec": count * step, "extraction_seconds": 0.01,
        "sampling_capped": False, "max_sample_gap_sec": step if count > 1 else 0.0,
    }


class ObserveReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.out = self.directory / "new-run"
        self.video = self.directory / "synthetic.mov"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("analyzer.observe.GpuMonitor", NoGpuMonitor))

    def model(self, *responses):
        client = FakeModel(responses)
        return self.use_model(client)

    def use_model(self, client):
        self.stack.enter_context(patch("analyzer.observe.LocalModelClient", return_value=client))
        return client

    def sample(self, count, step=1.0):
        result = sample_fixture(count, step)
        self.sampler = self.stack.enter_context(patch("analyzer.observe.sample_video", return_value=result))
        return result

    def assert_saved(self, report):
        self.assertEqual(json.loads((self.out / "report.json").read_text(encoding="utf-8")), report)
        self.assertFalse((self.out / "report.json.tmp").exists())
        self.assertNotIn("verdict", report)
        self.assertNotIn("violations", report)
        for row in report["observations"]:
            with Image.open(self.out / row["image_path"]) as image:
                self.assertEqual(image.format, "JPEG")

    def test_existing_output_directory_is_rejected_without_overwriting_files(self):
        client = self.model()
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("previous run", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            observe_video(self.video, self.out)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous run")
        self.assertFalse((self.out / "report.json").exists())
        self.assertEqual(client.requests, [])

    def test_undecodable_video_preserves_failed_report_without_model_calls(self):
        client = self.model()
        self.video.write_bytes(b"not a video")
        report = observe_video(self.video, self.out)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_frame_count"], 0)
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertEqual(report["transitions"], [])
        self.assertTrue(report["errors"])
        self.assertEqual(client.requests, [])
        self.assert_saved(report)

    def test_timeout_retains_previous_frame_and_partially_answered_current_frame(self):
        self.sample(3)
        client = self.model(
            answer({"person_in_work_area": "yes"}), answer({"visible": "no", "bbox": []}),
            answer({"person_in_work_area": "no"}), TimeoutError("synthetic request timeout"),
        )
        report = observe_video(self.video, self.out, interval_sec=1, max_refinement_frames=0,
                               cabinet_mode="context_crops")
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["selected_frame_count"], 3)
        self.assertEqual(report["processed_frame_count"], 2)
        first, partial = report["observations"]
        self.assertEqual(first["status"], "ok")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["frame_id"], "frame_0001")
        self.assertEqual(partial["timestamp_sec"], 1.0)
        self.assertEqual(partial["states"], {"person_in_work_area": "no", "cabinet_door": "uncertain"})
        self.assertEqual(partial["errors"][0]["feature"], "cabinet_door")
        self.assertFalse(partial["calls"][-1]["completed"])
        self.assertEqual(report["request_count"], 4)
        self.assertEqual(len(client.requests), 4)
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertEqual(report["transitions"], [])
        self.assert_saved(report)

    def test_partial_field_error_is_not_all_frames_success_even_at_final_frame(self):
        self.sample(1)
        self.model(answer({"person_in_work_area": "wrong"}), answer({"visible": "no", "bbox": []}))
        report = observe_video(self.video, self.out, cabinet_mode="context_crops")
        self.assertEqual(report["processed_frame_count"], report["selected_frame_count"])
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertEqual(report["observations"][0]["states"]["person_in_work_area"], "uncertain")
        self.assert_saved(report)

    def test_budget_exhausted_by_decoding_stops_before_first_model_request(self):
        client = self.model()
        now = [100.0]
        self.stack.enter_context(patch("analyzer.observe.time.perf_counter", side_effect=lambda: now[0]))

        def slow_decode(*args, **kwargs):
            now[0] = 111.0
            return sample_fixture(2)

        self.stack.enter_context(patch("analyzer.observe.sample_video", side_effect=slow_decode))
        report = observe_video(self.video, self.out, budget_sec=10.0)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertEqual(report["processed_frame_count"], 0)
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertEqual(client.requests, [])
        self.assert_saved(report)

    def test_valid_observations_save_frames_and_confirm_only_state_change(self):
        self.sample(4)
        responses = []
        for state in ("yes", "yes", "no", "no"):
            responses.extend((answer({"person_in_work_area": state}), answer({"visible": "no", "bbox": []})))
        client = self.model(*responses)
        report = observe_video(self.video, self.out, interval_sec=1, max_refinement_frames=0,
                               cabinet_mode="context_crops")
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["all_selected_frames_processed"])
        self.assertEqual(report["request_count"], 8)
        self.assertEqual(len(client.requests), 8)
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["transitions"]), 1)
        change = report["transitions"][0]
        self.assertEqual((change["feature"], change["from_state"], change["to_state"]),
                         ("person_in_work_area", "yes", "no"))
        self.assertEqual(change["evidence_frame_ids"], ["frame_0001", "frame_0002", "frame_0003"])
        self.assert_saved(report)

    def test_dense_refinement_confirms_brief_opening_and_preserves_frame_identity(self):
        source = self.sample(17, step=0.5)
        client = self.use_model(SceneModel(lambda timestamp: "open" if 3.5 <= timestamp <= 4.5 else "closed"))
        report = observe_video(self.video, self.out, cabinet_mode="full_frame")
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["all_selected_frames_processed"])
        self.sampler.assert_called_once()
        self.assertEqual(report["available_frame_count"], 17)
        self.assertEqual(report["coarse_frame_count"], 5)
        self.assertEqual(report["selected_frame_count"], 13)
        self.assertEqual(report["refinement"]["candidate_count"], 8)
        self.assertFalse(report["refinement"]["capped"])
        self.assertEqual(report["settings"]["cabinet_mode"], "full_frame")
        coarse_ids = {"frame_0000", "frame_0004", "frame_0008", "frame_0012", "frame_0016"}
        coarse_rows = [row for row in report["observations"] if row["frame_id"] in coarse_ids]
        self.assertEqual(sum(row["states"]["cabinet_door"] == "open" for row in coarse_rows), 1)
        self.assertEqual(build_transitions(coarse_rows), [])
        self.assertEqual([(row["from_state"], row["to_state"], row["interval_start_sec"],
                           row["interval_end_sec"], row["confirmed_at_sec"]) for row in report["transitions"]],
                         [("closed", "open", 3.0, 3.5, 4.0), ("open", "closed", 4.5, 5.0, 5.5)])
        timestamps = [row["timestamp_sec"] for row in report["observations"]]
        self.assertEqual(timestamps, report["selected_timestamps"])
        self.assertEqual(timestamps, sorted(set(timestamps)))
        ids = [row["frame_id"] for row in report["observations"]]
        self.assertEqual(len(ids), len(set(ids)))
        for row in report["observations"]:
            original_index = int(row["timestamp_sec"] * 2)
            self.assertEqual(row["frame_id"], f"frame_{original_index:04d}")
            self.assertEqual(row["image_path"], f"frames/frame_{original_index:04d}.jpg")
            self.assertEqual((self.out / row["image_path"]).read_bytes(), source["frames"][original_index].jpeg)
            self.assertEqual(row["cabinet_consistency"], "single_full_frame")
            self.assertEqual(row["crop_checks"], [])
            self.assertEqual([call["question"] for call in row["calls"]], ["person", "cabinet_door_full_frame"])
        self.assertEqual(len(client.requests), 26)
        self.assert_saved(report)

    def test_refinement_limit_keeps_coverage_of_both_change_windows(self):
        self.sample(17, step=0.5)
        client = self.use_model(SceneModel(lambda timestamp: "open" if 3.5 <= timestamp <= 4.5 else "closed"))
        report = observe_video(self.video, self.out, cabinet_mode="full_frame", max_refinement_frames=2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["refinement"], {"indices": [6, 10], "candidate_count": 8, "capped": True})
        self.assertEqual(report["selected_timestamps"], [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
        self.assertEqual(report["selected_frame_count"], 7)
        self.assertEqual(report["processed_frame_count"], 7)
        self.assertTrue(report["all_selected_frames_processed"])
        self.assertEqual(len(client.requests), 14)
        self.assert_saved(report)

    def test_motion_finds_cycle_between_identical_coarse_endpoints(self):
        source = self.sample(17, step=0.5)
        for index in (4, 5, 6):
            image = Image.new("RGB", (64, 48), (80, 120, 160))
            ImageDraw.Draw(image).rectangle((20, 16, 31, 30), fill="white")
            encoded = BytesIO()
            image.save(encoded, format="JPEG")
            source["frames"][index] = VideoFrame(index*.5, encoded.getvalue(), 64, 48)
        self.use_model(SceneModel(lambda timestamp: "open" if 2 <= timestamp <= 3 else "closed"))
        report = observe_video(self.video, self.out, interval_sec=4, max_refinement_frames=24)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["refinement"]["candidate_count"], 0)
        self.assertTrue(report["motion"]["windows"])
        self.assertEqual([change["to_state"] for change in report["transitions"]], ["open", "closed"])
        self.assertLessEqual(report["processed_frame_count"]-report["coarse_frame_count"], 24)
        self.assertEqual(len({row["frame_id"] for row in report["observations"]}), report["processed_frame_count"])
        self.assert_saved(report)

    def test_motion_and_state_proposals_share_one_refinement_cap(self):
        self.sample(17, step=.5)
        self.use_model(SceneModel(lambda t: "open" if 3.5 <= t <= 4.5 else "closed"))
        motion = {"refinement_indices": [1, 15], "refinement": {"selection_order": [15, 1]}, "windows": []}
        self.stack.enter_context(patch("analyzer.observe.propose_motion_windows", return_value=motion))
        report = observe_video(self.video, self.out, interval_sec=2, max_refinement_frames=2)
        self.assertEqual(report["processed_frame_count"], report["coarse_frame_count"]+2)
        self.assertEqual(report["combined_refinement"]["indices"], [1, 15])
        self.assertTrue(report["combined_refinement"]["capped"])
        self.assertTrue(report["combined_refinement"]["state_proposals_dropped"])
        self.assert_saved(report)

    def test_action_checks_use_counter_pool_and_save_actual_evidence(self):
        source = self.sample(7, step=.5)
        self.use_model(FakeModel())
        self.stack.enter_context(patch("analyzer.observe.ImageObserver.observe", side_effect=lambda frame, frame_id, **kw: {
            "frame_id": frame_id, "timestamp_sec": frame.timestamp_sec,
            "states": {"cabinet_door": "closed", "person_in_work_area": "yes"},
            "status": "ok", "calls": [], "errors": []}))
        primary = {"windows": [], "dropped_windows": [], "refinement": {"selection_order": []}}
        window = {"roi": "desk", "group_id": "desk-0", "priority": 1,
                  "frame_indices": [0, 1, 2], "frame_timestamps": [0, .5, 1], "start_sec": 0, "end_sec": 1}
        counter = {"windows": [window], "dropped_windows": [],
                   "rois": {"desk": {"union_bbox_normalized": [250, 250, 750, 750]}}}
        self.stack.enter_context(patch("analyzer.observe.propose_motion_windows", side_effect=[primary, counter]))
        self.stack.enter_context(patch("analyzer.observe.discover_counter", return_value={
            "status": "ok", "observations": [], "calls": [{}], "errors": []}))
        check = self.stack.enter_context(patch("analyzer.observe.ActionObserver.check", return_value={
            "window_id": "action_0000", "region": "desk", "start_sec": 0, "end_sec": 1,
            "result": {"interaction": "uncertain"}, "status": "ok", "calls": [{}], "errors": []}))
        report = observe_video(self.video, self.out, enable_actions=True, activity_base_url="http://127.0.0.1:8081")
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["all_selected_action_windows_processed"])
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(check.call_args.kwargs["counter_box"], [250, 250, 750, 750])
        self.assertEqual(report["action_selection"]["selected_count"], 1)
        self.assertEqual(report["motion"]["windows"], [])
        for i, evidence in enumerate(report["action_checks"][0]["evidence"]):
            expected = source["frames"][i]
            self.assertEqual(evidence["timestamp_sec"], expected.timestamp_sec)
            self.assertEqual((self.out/evidence["image_path"]).read_bytes(), expected.jpeg)
            self.assertEqual(evidence["image_sha256"], hashlib.sha256(expected.jpeg).hexdigest())
        self.assert_saved(report)

    def test_saved_cabinet_claim_requires_open_state_and_retains_raw_answer_and_evidence(self):
        source = self.sample(7, step=.5)
        client = self.model(answer({"visible": "yes", "bbox": [250, 250, 750, 750]}))
        raw_answer = answer({"paper_near_open_cabinet": "yes", "entry_visible": "yes"})
        raw_answer["response_text"] = '{"paper_near_open_cabinet":"yes","entry_visible":"yes"}'
        sequence = self.stack.enter_context(patch.object(client, "ask_sequence", return_value=raw_answer, create=True))
        self.stack.enter_context(patch("analyzer.observe.ImageObserver.observe", side_effect=lambda frame, frame_id, **kw: {
            "frame_id": frame_id, "timestamp_sec": frame.timestamp_sec,
            "states": {"cabinet_door": "closed", "person_in_work_area": "yes"},
            "status": "ok", "calls": [], "errors": []}))
        indices = [0, 3, 6]
        window = {"roi": "cabinet", "group_id": "cabinet-0", "priority": 1,
                  "frame_indices": indices, "frame_timestamps": [0, 1.5, 3], "start_sec": 0, "end_sec": 3}
        primary = {"windows": [window], "dropped_windows": [], "refinement": {"selection_order": []}}
        counter = {"windows": [], "dropped_windows": [], "rois": {"desk": {}}}
        self.stack.enter_context(patch("analyzer.observe.propose_motion_windows", side_effect=[primary, counter]))
        self.stack.enter_context(patch("analyzer.observe.discover_counter", return_value={
            "status": "ok", "observations": [], "calls": [], "errors": []}))

        # Keep the real ActionObserver, combiner, selector, gate and disk save.
        report = observe_video(self.video, self.out, interval_sec=.5, max_refinement_frames=0,
                               enable_actions=True, activity_base_url="http://127.0.0.1:8081")
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["all_selected_action_windows_processed"])
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(len(client.requests), 1)
        sequence.assert_called_once()
        self.assertEqual([frame.timestamp_sec for frame in sequence.call_args.args[0]], [0, 1.5, 3])
        self.assertTrue(all(row["states"]["cabinet_door"] == "closed" for row in report["observations"]))

        saved = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["action_checks"]), 1)
        check = saved["action_checks"][0]
        self.assertEqual(check["result"]["cash_near_cabinet"], "uncertain")
        self.assertEqual(check["cash_near_cabinet_before_gate"], "yes")
        self.assertEqual(check["cabinet_state_gate"], "no_observed_open_in_window")
        self.assertEqual(check["cabinet_state_gate_open_times"], [])
        self.assertEqual(check["result"]["cash_into_cabinet"], "uncertain")
        self.assertEqual(check["raw_entry_visible"], "yes")
        self.assertEqual([call["question"] for call in check["calls"]], ["ground_object", "cabinet_sequence"])
        self.assertTrue(all(call["completed"] for call in check["calls"]))
        self.assertEqual(check["calls"][1]["response"], raw_answer)
        self.assertEqual(raw_answer["observation"]["paper_near_open_cabinet"], "yes")
        self.assertIn("cabinet_action_gate", saved["settings"])
        self.assertEqual(len(check["evidence"]), 3)
        for index, evidence in zip(indices, check["evidence"]):
            frame = source["frames"][index]
            self.assertEqual(evidence["timestamp_sec"], frame.timestamp_sec)
            self.assertEqual(evidence["image_path"], f"frames/frame_{index:04d}.jpg")
            self.assertEqual((self.out / evidence["image_path"]).read_bytes(), frame.jpeg)
            self.assertEqual(evidence["image_sha256"], hashlib.sha256(frame.jpeg).hexdigest())
        self.assert_saved(report)

    def test_deadline_during_refinement_preserves_sorted_coarse_and_partial_records(self):
        self.sample(17, step=0.5)
        now = [100.0]
        self.stack.enter_context(patch("analyzer.observe.time.perf_counter", side_effect=lambda: now[0]))

        def expire_during_first_refinement(timestamp, question):
            if timestamp == 1.5:
                now[0] = 109.0 if question == "person" else 111.0

        client = self.use_model(SceneModel(
            lambda timestamp: "open" if 3.5 <= timestamp <= 4.5 else "closed",
            on_request=expire_during_first_refinement,
        ))
        report = observe_video(self.video, self.out, cabinet_mode="full_frame", budget_sec=10.0)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["budget_exceeded"])
        self.assertFalse(report["all_selected_frames_processed"])
        self.assertEqual(report["selected_frame_count"], 13)
        self.assertEqual(report["processed_frame_count"], 6)
        rows = report["observations"]
        self.assertEqual([row["timestamp_sec"] for row in rows], [0.0, 1.5, 2.0, 4.0, 6.0, 8.0])
        partial = rows[1]
        self.assertEqual(partial["frame_id"], "frame_0003")
        self.assertEqual(partial["image_path"], "frames/frame_0003.jpg")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["states"], {"person_in_work_area": "yes", "cabinet_door": "uncertain"})
        self.assertTrue(all(row["status"] == "ok" for row in rows if row is not partial))
        self.assertEqual(report["errors"][0]["type"], "ObservationDeadline")
        self.assertEqual(len(client.requests), 12)
        self.assert_saved(report)


if __name__ == "__main__":
    unittest.main()
