"""Deadlines reject late decoding, hashing and model readiness without GPU calls."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyzer import service, video
from tests.test_video_sampling import write_vfr


class IODeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "short.mkv"
        write_vfr(self.path, [0, 200, 500])

    def test_expired_decode_never_opens_file(self):
        with patch("analyzer.video.perf_counter", return_value=10), patch("analyzer.video.av.open") as opened:
            with self.assertRaises(TimeoutError):
                video.sample_video(self.path, deadline=10)
            opened.assert_not_called()

    def test_scan_stops_during_real_decode_and_no_second_pass_starts(self):
        original = video._timestamp
        now = [0]
        calls = []

        def timestamp(frame):
            calls.append(frame.pts)
            now[0] = 10
            return original(frame)

        with patch("analyzer.video.perf_counter", side_effect=lambda: now[0]), patch("analyzer.video._timestamp", side_effect=timestamp):
            with self.assertRaises(TimeoutError):
                video.sample_video(self.path, deadline=10)
        self.assertEqual(len(calls), 1)

    def test_second_pass_rejects_a_frame_finishing_after_deadline(self):
        original = video._timestamp
        now, calls = [0], []

        def timestamp(frame):
            calls.append(frame.pts)
            if len(calls) == 4:  # Three real frames scanned; first selected frame decoded.
                now[0] = 10
            return original(frame)

        with patch("analyzer.video.perf_counter", side_effect=lambda: now[0]), patch("analyzer.video._timestamp", side_effect=timestamp):
            with self.assertRaises(TimeoutError):
                video.sample_video(self.path, max_frames=2, deadline=10)
        self.assertEqual(len(calls), 4)

    def test_hash_has_same_bytes_in_budget_and_rejects_late_digest(self):
        with patch("analyzer.service.time.perf_counter", return_value=0):
            self.assertEqual(service._sha256(self.path, deadline=10), hashlib.sha256(self.path.read_bytes()).hexdigest())
        with patch("analyzer.service.time.perf_counter", side_effect=[0, 10]):
            with self.assertRaises(TimeoutError):
                service._sha256(self.path, deadline=10)

    def test_readiness_clamps_timeout_and_does_not_accept_late_identity(self):
        now = [7]

        def late_info():
            now[0] = 10
            return {"model_path": service._SERVERS[0][1]}

        with patch("analyzer.service.time.perf_counter", side_effect=lambda: now[0]), patch("analyzer.service.LocalModelClient") as client:
            client.return_value.server_info.side_effect = late_info
            with self.assertRaises(TimeoutError):
                service.check_ready(deadline=10)
            self.assertEqual(client.call_count, 1)
            self.assertEqual(client.call_args.kwargs["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
