"""Exercise the sampler against real, synthetic variable-timestamp video files."""

from fractions import Fraction
from io import BytesIO
import math
from pathlib import Path
import tempfile
import unittest

import av
from PIL import Image

from analyzer.video import sample_video


def write_vfr(path, timestamps_ms, size=(160, 90)):
    """Use lossless coloured frames so decoded pixels identify the source frame."""
    with av.open(str(path), "w", format="matroska") as output:
        stream = output.add_stream("ffv1", rate=1000)
        stream.width, stream.height = size
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, timestamp in enumerate(timestamps_ms):
            colour = (index * 40, 20, 200 - index * 25)
            frame = av.VideoFrame.from_image(Image.new("RGB", size, colour))
            frame.pts = timestamp
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def decoded_reference(path):
    """Read actual presentation times and pixels independently of sampler helpers."""
    with av.open(str(path)) as source:
        return [
            (float(frame.pts * frame.time_base), frame.to_image().convert("RGB").getpixel((40, 20)))
            for frame in source.decode(video=0)
        ]


class VideoSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.video = cls.directory / "synthetic-vfr.mkv"
        write_vfr(cls.video, [5000, 5200, 5900, 7100, 9500, 10000])
        cls.reference = decoded_reference(cls.video)

    def assert_actual_frames(self, result):
        origin = self.reference[0][0]
        seen = set()
        previous = -1
        for frame in result["frames"]:
            matches = [
                colour for timestamp, colour in self.reference
                if abs(timestamp - origin - frame.timestamp_sec) < 1e-8
            ]
            self.assertEqual(len(matches), 1, "Every selected time must identify a decoded source frame")
            self.assertNotIn(frame.timestamp_sec, seen)
            self.assertGreater(frame.timestamp_sec, previous)
            seen.add(frame.timestamp_sec)
            previous = frame.timestamp_sec
            with Image.open(BytesIO(frame.jpeg)) as picture:
                self.assertEqual(picture.format, "JPEG")
                self.assertEqual(picture.size, (frame.width, frame.height))
                actual = picture.convert("RGB").getpixel((frame.width // 2, frame.height // 2))
            for channel, expected in zip(actual, matches[0]):
                self.assertLessEqual(abs(channel - expected), 4, "JPEG pixels must match the frame at that PTS")
        self.assertEqual(result["frames"][0].timestamp_sec, 0.0)
        self.assertAlmostEqual(result["frames"][-1].timestamp_sec, 5.0)

    def test_fixture_really_has_nonzero_start_and_nonuniform_decoded_pts(self):
        self.assertEqual([time for time, _ in self.reference], [5.0, 5.2, 5.9, 7.1, 9.5, 10.0])
        self.assertEqual(len({right[0] - left[0] for left, right in zip(self.reference, self.reference[1:])}), 5)

    def test_original_sampler_uses_pts_and_keeps_matching_pixels(self):
        result = sample_video(self.video, max_frames=4)
        self.assert_actual_frames(result)
        # Four equal-count frame indices would select a different sequence.
        self.assertEqual([frame.timestamp_sec for frame in result["frames"]], [0.0, 2.1, 4.5, 5.0])
        self.assertFalse(result["sampling_capped"])
        self.assertAlmostEqual(result["max_sample_gap_sec"], 2.4)

    def test_interval_mode_uses_real_frames_not_interpolated_target_times(self):
        result = sample_video(self.video, max_frames=90, interval_sec=2.0)
        self.assert_actual_frames(result)
        self.assertEqual([frame.timestamp_sec for frame in result["frames"]], [0.0, 2.1, 4.5, 5.0])
        self.assertFalse(result["sampling_capped"])
        # A requested spacing is not a promise about gaps in the source video.
        self.assertGreater(result["max_sample_gap_sec"], 2.0)
        self.assertAlmostEqual(result["max_sample_gap_sec"], 2.4)

    def test_interval_mode_reports_cap_and_actual_larger_gap(self):
        result = sample_video(self.video, max_frames=2, interval_sec=1.0)
        self.assert_actual_frames(result)
        self.assertEqual([frame.timestamp_sec for frame in result["frames"]], [0.0, 5.0])
        self.assertTrue(result["sampling_capped"])
        self.assertEqual(result["max_sample_gap_sec"], 5.0)

    def test_more_requested_frames_than_source_returns_each_source_once(self):
        result = sample_video(self.video, max_frames=90, interval_sec=0.1)
        self.assert_actual_frames(result)
        self.assertEqual([frame.timestamp_sec for frame in result["frames"]], [0.0, 0.2, 0.9, 2.1, 4.5, 5.0])
        self.assertFalse(result["sampling_capped"])
        self.assertAlmostEqual(result["max_sample_gap_sec"], 2.4)

    def test_nearest_targets_do_not_duplicate_frames_across_large_source_gap(self):
        path = self.directory / "large-gap.mkv"
        write_vfr(path, [5000, 5100, 5200, 5300, 5400, 15000])
        result = sample_video(path, max_frames=5)
        times = [frame.timestamp_sec for frame in result["frames"]]
        self.assertEqual(times, [0.0, 0.4, 10.0])
        self.assertEqual(len(times), len(set(times)))
        self.assertLess(len(times), 5)
        self.assertAlmostEqual(result["max_sample_gap_sec"], 9.6)

    def test_long_interval_still_includes_first_and_last(self):
        result = sample_video(self.video, max_frames=90, interval_sec=100.0)
        self.assert_actual_frames(result)
        self.assertEqual(len(result["frames"]), 2)
        self.assertFalse(result["sampling_capped"])

    def test_resize_preserves_aspect_ratio_and_jpeg_dimensions(self):
        result = sample_video(self.video, max_frames=2, max_side=80)
        self.assert_actual_frames(result)
        self.assertEqual({(frame.width, frame.height) for frame in result["frames"]}, {(80, 45)})

    def test_small_original_frames_are_not_enlarged(self):
        result = sample_video(self.video, max_frames=2, max_side=1280)
        self.assert_actual_frames(result)
        self.assertEqual({(frame.width, frame.height) for frame in result["frames"]}, {(160, 90)})

    def test_duration_excludes_nonzero_start_and_covers_last_frame(self):
        result = sample_video(self.video, max_frames=2)
        self.assertGreater(result["duration_sec"], 5.0)
        self.assertLess(result["duration_sec"], 5.05)
        self.assertGreaterEqual(result["extraction_seconds"], 0.0)
        self.assertTrue(math.isfinite(result["extraction_seconds"]))

    def test_single_source_frame_has_zero_relative_time_and_gap(self):
        path = self.directory / "single.mkv"
        write_vfr(path, [7000])
        result = sample_video(path, max_frames=12, interval_sec=2.0)
        self.assertEqual(len(result["frames"]), 1)
        self.assertEqual(result["frames"][0].timestamp_sec, 0.0)
        self.assertEqual(result["max_sample_gap_sec"], 0.0)
        self.assertFalse(result["sampling_capped"])
        self.assertGreater(result["duration_sec"], 0.0)

    def test_invalid_interval_values_are_rejected(self):
        for value in (0, -1, True, False, "2", math.nan, math.inf, -math.inf):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                sample_video(self.video, interval_sec=value)

    def test_invalid_size_and_frame_limit_are_rejected(self):
        for value in (0, 1, -1, True, 3.0, math.nan):
            with self.subTest(max_frames=value), self.assertRaises(ValueError):
                sample_video(self.video, max_frames=value)
        for value in (0, -1, True, 80.0, math.nan):
            with self.subTest(max_side=value), self.assertRaises(ValueError):
                sample_video(self.video, max_side=value)

    def test_invalid_or_undecodable_input_never_returns_empty_success(self):
        for value in ("", "   ", None, True):
            with self.subTest(path=value), self.assertRaises(ValueError):
                sample_video(value)
        with self.assertRaises(FileNotFoundError):
            sample_video(self.directory / "missing.mkv")
        with self.assertRaises(ValueError):
            sample_video(self.directory)
        malformed = self.directory / "malformed.mov"
        malformed.write_bytes(b"This is not a video.")
        with self.assertRaises(av.error.FFmpegError):
            sample_video(malformed)


if __name__ == "__main__":
    unittest.main()
