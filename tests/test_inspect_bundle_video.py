"""Replay regressions using a generated local video and a stub detector."""

from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
from PIL import Image

from analyzer.bundle_dataset import file_sha
from analyzer.inspect_bundle_video import replay, validate_plan


class BundleReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/"fixture.mkv"
        with av.open(str(self.video), mode="w") as output:
            stream = output.add_stream("ffv1", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
            for index in range(5):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64,48), (index*40,20,10)))
                frame.pts, frame.time_base = index+10, Fraction(1,10)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        self.plan = {"videos": [{"video_id": 1, "source_video_path": str(self.video),
                                  "source_video_sha256": file_sha(self.video), "roi_xyxy": [0,0,64,48]}]}
        self.plan_path = self.root/"plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.weights = self.root/"fixture.pth"
        self.weights.write_bytes(b"test only, never loaded")
        self.args = SimpleNamespace(plan=self.plan_path,weights=self.weights,out=self.root/"result",interval=.1)
        self.model = Mock()
        self.model.class_names = {1: "banknote_bundle"}
        self.model.model.resolution = 384
        self.model.model_config.resolution = 384
        self.model.predict.return_value = SimpleNamespace(xyxy=[],confidence=[],class_id=[])
        self.constructor = Mock(return_value=self.model)

    def run_replay(self):
        with patch("analyzer.inspect_bundle_video.offline_runtime", return_value=(Mock(),self.constructor)):
            replay(self.args)

    def test_dense_replay_retains_actual_times_and_original_jpeg_hashes(self):
        self.run_replay()
        report=json.loads((self.args.out/"report.json").read_text())
        self.assertEqual(report["status"], "completed")
        row=report["videos"][0]
        self.assertEqual(row["frame_count"],5)
        self.assertFalse(row["sampling_capped"])
        self.assertAlmostEqual(row["max_sample_gap_sec"],.1)
        frames=json.loads((self.args.out/"1/predictions.json").read_text())["frames"]
        self.assertEqual([f["timestamp_sec"] for f in frames],[0,.1,.2,.3,.4])
        for f in frames:
            self.assertEqual(f["jpeg_sha256"],file_sha(self.args.out/"1"/f["jpeg"]))
        with av.open(str(self.args.out/"1/preview.mp4")) as preview:
            self.assertEqual(len(preview.streams.audio),0)
            self.assertEqual(len(list(preview.decode(preview.streams.video[0]))),5)
        self.assertTrue((self.args.out/"index.html").is_file())

    def test_changed_source_or_duplicate_id_rejected_before_inference(self):
        self.plan["videos"].append(dict(self.plan["videos"][0]))
        with self.assertRaises(ValueError):
            validate_plan(self.plan)
        self.plan["videos"].pop()
        self.video.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError,"hash differs"):
            validate_plan(self.plan)

    def test_invalid_roi_records_failure_without_empty_success(self):
        self.plan["videos"][0]["roi_xyxy"]=[0,0,65,48]
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,"ROI or source geometry"):
            self.run_replay()
        self.model.predict.assert_not_called()
        report=json.loads((self.args.out/"report.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertFalse((self.args.out/"index.html").exists())

    def test_model_failure_keeps_report_failed_not_completed(self):
        self.model.predict.side_effect=RuntimeError("GPU failure")
        with self.assertRaisesRegex(RuntimeError,"GPU failure"):
            self.run_replay()
        report=json.loads((self.args.out/"report.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["videos"],[])
        self.assertFalse((self.args.out/"index.html").exists())

    def test_dense_replay_records_verified_resolution(self):
        self.args.resolution = 512
        self.model.model.resolution = self.model.model_config.resolution = 512
        self.run_replay()
        self.assertEqual(self.constructor.call_args.kwargs["resolution"],512)
        self.assertEqual(json.loads((self.args.out/"report.json").read_text())["resolution"],512)


if __name__ == "__main__":
    unittest.main()
