"""Actual VFR video, source pixels, ROI isolation and continuity checks."""

from copy import deepcopy
from fractions import Fraction
from io import BytesIO
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
from PIL import Image, ImageDraw

from analyzer.focus_video import motion_peak_triple, sample_video_intervals
from analyzer.video import VideoFrame
from tests.test_video_sampling import decoded_reference, write_vfr


def window(name="a", start=0, end=2.2):
    return {"window_id": name, "start_sec": start, "end_sec": end}


def picture(time, left=0, right=0, size=(40, 20)):
    image = Image.new("RGB", size, (right, right, right))
    ImageDraw.Draw(image).rectangle((0, 0, size[0]//2-1, size[1]-1), fill=(left, left, left))
    data = BytesIO()
    image.save(data, format="JPEG", quality=100, subsampling=0)
    return VideoFrame(time, data.getvalue(), *size)


class FocusSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.video = cls.directory / "focus-vfr.mkv"
        write_vfr(cls.video, [5000, 5200, 5900, 7100, 9500, 10000])
        cls.reference = decoded_reference(cls.video)

    def assert_source_pixels(self, frames):
        origin = self.reference[0][0]
        for frame in frames:
            reference = [colour for timestamp, colour in self.reference
                         if abs(timestamp-origin-frame.timestamp_sec) < 1e-9]
            self.assertEqual(len(reference), 1)
            with Image.open(BytesIO(frame.jpeg)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (frame.width, frame.height))
                pixel = image.getpixel((frame.width//2, frame.height//2))
            for actual, expected in zip(pixel, reference[0]):
                self.assertLessEqual(abs(actual-expected), 4)

    def test_bounds_select_actual_vfr_times_without_nearest_outside_substitutes(self):
        request = [window(start=.1, end=1.0)]
        snapshot = deepcopy(request)
        result = sample_video_intervals(self.video, request)
        self.assertEqual([f.timestamp_sec for f in result["frames"]], [.2, .9])
        self.assert_source_pixels(result["frames"])
        row = result["windows"][0]
        self.assertEqual(row["frame_indices"], [0, 1])
        self.assertEqual(row["requested_start_sec"], .1)
        self.assertEqual(row["requested_end_sec"], 1.0)
        self.assertAlmostEqual(row["left_boundary_gap_sec"], .1)
        self.assertAlmostEqual(row["right_boundary_gap_sec"], .1)
        self.assertAlmostEqual(row["max_sample_gap_sec"], .7)
        self.assertGreater(row["max_sample_gap_sec"], result["settings"]["interval_sec"])
        self.assertEqual(request, snapshot)

    def test_overlapping_windows_share_sorted_unique_pool_even_for_reversed_requests(self):
        result = sample_video_intervals(self.video, [window("later", .9, 2.2), window("earlier", 0, 1)])
        self.assertEqual([f.timestamp_sec for f in result["frames"]], [0, .2, .9, 2.1])
        self.assertEqual([r["frame_indices"] for r in result["windows"]], [[2, 3], [0, 1, 2]])
        for row in result["windows"]:
            self.assertEqual(row["frame_timestamps"], [result["frames"][i].timestamp_sec for i in row["frame_indices"]])
        self.assert_source_pixels(result["frames"])

    def test_cap_reports_larger_actual_gap_and_keeps_real_endpoints(self):
        result = sample_video_intervals(self.video, [window()], max_frames_per_window=2)
        row = result["windows"][0]
        self.assertEqual(row["frame_timestamps"], [0, 2.1])
        self.assertTrue(row["sampling_capped"])
        self.assertTrue(result["sampling_capped"])
        self.assertEqual(row["available_distinct_frame_count"], 4)
        self.assertEqual(row["max_unobserved_interval_sec"], 2.1)

    def test_duplicate_pts_keep_first_decoded_pixels_and_never_duplicate_pool(self):
        path = self.directory / "focus-duplicates.mkv"
        write_vfr(path, [5000, 5100, 5100, 5200])
        result = sample_video_intervals(path, [window("a", 0, .2), window("b", .1, .2)])
        self.assertEqual(result["duplicate_source_pts_removed"], 1)
        self.assertEqual([f.timestamp_sec for f in result["frames"]], [0, .1, .2])
        with Image.open(BytesIO(result["frames"][1].jpeg)) as image:
            pixel = image.getpixel((40, 20))
        for actual, expected in zip(pixel, (40, 20, 175)):
            self.assertLessEqual(abs(actual-expected), 4)

    def test_resize_never_enlarges_and_retains_original_aspect(self):
        full = sample_video_intervals(self.video, [window()], max_side=1280)
        small = sample_video_intervals(self.video, [window()], max_side=80)
        self.assertEqual({(f.width, f.height) for f in full["frames"]}, {(160, 90)})
        self.assertEqual({(f.width, f.height) for f in small["frames"]}, {(80, 45)})
        self.assert_source_pixels(small["frames"])

    def test_empty_request_and_exact_point_are_explicit(self):
        empty = sample_video_intervals(self.video, [], max_windows=0)
        self.assertEqual(empty["frames"], [])
        self.assertEqual(empty["windows"], [])
        point = sample_video_intervals(self.video, [window(start=.9, end=.9)])
        self.assertEqual(point["windows"][0]["frame_timestamps"], [.9])
        self.assertEqual(point["windows"][0]["max_sample_gap_sec"], 0)

    def test_window_limits_invalid_boundaries_and_empty_gaps_raise(self):
        invalid_requests = [None, [None], [window(), window()], [window(start=-1)],
                            [window(start=2, end=1)], [window(end=3.01)],
                            [window(start=4, end=6)], [window(start=.3, end=.8)],
                            [window(start=True)], [window(end=math.inf)], [window(name=" ")]]
        for rows in invalid_requests:
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                sample_video_intervals(self.video, rows)
        with self.assertRaises(ValueError):
            sample_video_intervals(self.video, [window("a"), window("b")], max_windows=1)
        for key, values in {"interval_sec": [0, -1, True, math.nan],
                            "max_frames_per_window": [1, True, 2.0],
                            "max_windows": [-1, True, 1.0], "max_side": [0, True, 2.0]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    sample_video_intervals(self.video, [window()], **{key: value})

    def test_changed_pts_or_invalid_source_never_returns_success(self):
        with patch("analyzer.focus_video._timestamp", return_value=Fraction(900)), self.assertRaises(RuntimeError):
            sample_video_intervals(self.video, [window()])
        malformed = self.directory / "invalid.mov"
        malformed.write_bytes(b"not a video")
        with self.assertRaises(av.error.FFmpegError):
            sample_video_intervals(malformed, [window()])
        for path in ("", None, self.directory):
            with self.subTest(path=path), self.assertRaises(ValueError):
                sample_video_intervals(path, [window()])

    def test_both_passes_explicitly_decode_only_video_and_incomplete_second_pass_raises(self):
        real_open = av.open
        for truncate in (False, True):
            decoded = []
            opened = []

            class Container:
                def __init__(self, *args, **kwargs):
                    self.source = real_open(*args, **kwargs)
                    self.number = len(opened)
                    opened.append(self)
                    self.streams = self.source.streams

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.source.close()

                def decode(self, stream):
                    self_type.assertIs(stream, self.streams.video[0])
                    decoded.append(stream.type)
                    for index, frame in enumerate(self.source.decode(stream)):
                        if truncate and self.number == 1 and index > 0:
                            break
                        yield frame

            self_type = self
            with patch("analyzer.focus_video.av.open", side_effect=Container):
                if truncate:
                    with self.assertRaises(RuntimeError):
                        sample_video_intervals(self.video, [window()])
                else:
                    sample_video_intervals(self.video, [window()])
            self.assertEqual(decoded, ["video", "video"])


class MotionPeakTripleTests(unittest.TestCase):
    def test_largest_pair_selects_real_adjacent_samples_and_no_action_claim(self):
        frames = [picture(t, level) for t, level in zip([0, .07, .18, .31, .44], [0, 0, 220, 220, 220])]
        result = motion_peak_triple(frames, [0, 0, 400, 1000])
        self.assertTrue(result["selected"])
        self.assertEqual(result["peak_pair_indices"], [1, 2])
        self.assertEqual(result["frame_indices"], [1, 2, 3])
        self.assertEqual(result["frame_timestamps"], [.07, .18, .31])
        self.assertGreater(result["score"], 200)
        self.assertNotIn("action", result)
        self.assertNotIn("verdict", result)

    def test_changes_outside_roi_do_not_create_motion_proposal(self):
        frames = [picture(t, left=80, right=r) for t, r in zip([0, .1, .2], [0, 255, 0])]
        result = motion_peak_triple(frames, [0, 0, 250, 1000])
        self.assertFalse(result["selected"])
        self.assertEqual(result["reason"], "no_pixel_change")

    def test_first_and_last_peak_use_three_real_neighbors(self):
        for levels, peak in (([0, 255, 255], [0, 1]), ([0, 0, 255], [1, 2])):
            with self.subTest(levels=levels):
                result = motion_peak_triple([picture(i*.1, level) for i, level in enumerate(levels)],
                                            [0, 0, 400, 1000])
                self.assertEqual(result["frame_indices"], [0, 1, 2])
                self.assertEqual(result["peak_pair_indices"], peak)

    def test_large_gap_breaks_continuity_and_cannot_win_with_huge_change(self):
        frames = [picture(t, level) for t, level in zip([0, .1, 2, 2.1, 2.2], [0, 0, 255, 230, 230])]
        result = motion_peak_triple(frames, [0, 0, 400, 1000])
        self.assertEqual(result["frame_indices"], [2, 3, 4])
        self.assertEqual(result["scores"][1]["skipped_reason"], "frame_gap")
        no_context = motion_peak_triple(frames[:3], [0, 0, 400, 1000])
        self.assertFalse(no_context["selected"])
        self.assertEqual(no_context["reason"], "no_three_frame_continuous_context")

    def test_geometry_change_breaks_continuity_and_bad_jpeg_metadata_raises(self):
        frames = [picture(0), picture(.1, 100), picture(.2, 255, size=(80, 40))]
        result = motion_peak_triple(frames, [0, 0, 400, 1000])
        self.assertFalse(result["selected"])
        self.assertEqual(result["scores"][1]["skipped_reason"], "frame_geometry_changed")
        wrong = VideoFrame(.2, frames[1].jpeg, 400, 20)
        with self.assertRaises(ValueError):
            motion_peak_triple([*frames[:2], wrong], [0, 0, 400, 1000])
        with self.assertRaises(OSError):
            motion_peak_triple([*frames[:2], VideoFrame(.2, b"bad jpeg", 40, 20)], [0, 0, 400, 1000])

    def test_ties_are_stable_inputs_unchanged_and_invalid_parameters_raise(self):
        frames = [picture(i*.1, level) for i, level in enumerate([0, 255, 0, 0])]
        original = list(frames)
        box = [0, 0, 400, 1000]
        first = motion_peak_triple(frames, box)
        self.assertEqual(first["peak_pair_indices"], [0, 1])
        self.assertEqual(first, motion_peak_triple(frames, box))
        self.assertEqual(frames, original)
        for invalid in ([0, 0, 0, 100], [0, 0, 1001, 1000], [True, 0, 100, 100], None):
            with self.subTest(box=invalid), self.assertRaises(ValueError):
                motion_peak_triple(frames, invalid)
        for invalid_frames in (frames[:2], [frames[0], frames[0], frames[1]], list(reversed(frames))):
            with self.subTest(frames=invalid_frames), self.assertRaises(ValueError):
                motion_peak_triple(invalid_frames, box)
        for gap in (0, -1, True, math.inf):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                motion_peak_triple(frames, box, max_pair_gap_sec=gap)


if __name__ == "__main__":
    unittest.main()
