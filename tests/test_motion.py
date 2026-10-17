"""CPU motion planning failure cases on synthetic images, without model calls."""

import copy
from io import BytesIO
import math
from types import SimpleNamespace
import unittest

from PIL import Image, ImageDraw

from analyzer.motion import propose_motion_windows


def frame(timestamp, rectangle=None, *, brightness=64, width=100, height=100):
    picture = Image.new("RGB", (width, height), (brightness,)*3)
    if rectangle is not None:
        ImageDraw.Draw(picture).rectangle(rectangle, fill=(220,)*3)
    output = BytesIO()
    picture.save(output, "JPEG", quality=95)
    return SimpleNamespace(timestamp_sec=timestamp, jpeg=output.getvalue(), width=width, height=height)


def coarse(*, desk=False):
    result = [{"cabinet_bbox_normalized": [200, 200, 600, 600],
               "states": {"cabinet_door": "closed"}} for _ in range(3)]
    if desk:
        for row in result:
            row["desk_bbox_normalized"] = [200, 200, 600, 600]
    return result


def pulse():
    return [frame(index*.5, (30, 30, 45, 45) if index in (2, 3) else None) for index in range(7)]


class MotionTests(unittest.TestCase):
    def test_hidden_cycle_is_proposed_even_when_coarse_states_are_identical(self):
        frames = pulse()
        report = propose_motion_windows(frames, coarse())
        self.assertTrue(report["windows"])
        selected = report["windows"][0]
        self.assertEqual(selected["roi"], "cabinet")
        self.assertLess(selected["start_sec"], 1)
        self.assertGreater(selected["end_sec"], 2)
        indices = selected["frame_indices"]
        self.assertEqual(len(set(indices)), 3)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(selected["frame_timestamps"], [frames[index].timestamp_sec for index in indices])
        self.assertEqual(report["kind"], "motion_proposals_not_actions")
        self.assertNotIn("verdict", report)

    def test_static_image_and_uniform_brightness_shift_do_not_create_motion(self):
        for frames in ([frame(index*.5) for index in range(6)],
                       [frame(index*.5, brightness=60+index*8) for index in range(6)]):
            with self.subTest(brightness=frames[-1].jpeg[-12:]):
                result = propose_motion_windows(frames, coarse())
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["coverage"]["compared_pairs_by_roi"]["cabinet"], 5)

    def test_motion_outside_detected_region_does_not_expand_to_full_frame(self):
        frames = [frame(index*.5, (80, 80, 95, 95) if index in (2, 3) else None) for index in range(6)]
        self.assertEqual(propose_motion_windows(frames, coarse())["windows"], [])

    def test_isolated_large_roi_outlier_is_excluded_from_union(self):
        observations = coarse()
        observations.append({"cabinet_bbox_normalized": [0, 0, 1000, 1000]})
        result = propose_motion_windows(pulse(), observations)
        roi = result["rois"]["cabinet"]
        self.assertEqual(roi["accepted_box_count"], 3)
        self.assertEqual(roi["outlier_box_count"], 1)
        self.assertEqual(roi["bbox_normalized"], [160, 160, 640, 640])

    def test_missing_or_single_box_produces_explicit_no_roi_coverage(self):
        for observations in ([], coarse()[:1], [{"cabinet_bbox_normalized": [True, 0, 500, 500]}]):
            with self.subTest(observations=observations):
                result = propose_motion_windows(pulse(), observations)
                self.assertEqual(result["windows"], [])
                self.assertIsNone(result["rois"]["cabinet"]["bbox_normalized"])
                self.assertTrue(result["coverage"]["image_decode_skipped_no_roi"])

    def test_corrupt_frame_breaks_continuity_and_is_not_a_negative_observation(self):
        frames = pulse()
        frames[3].jpeg = b"not a jpeg"
        result = propose_motion_windows(frames, coarse())
        self.assertEqual(result["frame_errors"][0]["frame_index"], 3)
        self.assertEqual(result["coverage"]["decoded_frame_count"], len(frames)-1)
        self.assertEqual(result["coverage"]["skipped_pairs_by_roi"]["cabinet"], 2)
        for window in result["windows"]:
            self.assertFalse(window["frame_indices"][0] < 3 < window["frame_indices"][-1])

    def test_actual_large_pts_gap_is_reported_and_never_bridged(self):
        frames = pulse()
        for item in frames[3:]:
            item.timestamp_sec += 10
        result = propose_motion_windows(frames, coarse())
        self.assertEqual(result["coverage"]["actual_max_frame_gap_sec"], 10.5)
        self.assertTrue(result["coverage"]["blind_shorter_actions"])
        self.assertEqual(result["coverage"]["skipped_pairs_by_roi"]["cabinet"], 1)
        for window in result["windows"]:
            self.assertFalse(window["start_sec"] <= 1 and window["end_sec"] >= 11.5)

    def test_geometry_change_breaks_pair_even_when_metadata_is_valid(self):
        frames = pulse()
        frames[3] = frame(1.5, (30, 30, 45, 45), width=120)
        result = propose_motion_windows(frames, coarse())
        skipped = [pair for pair in result["scores"]["cabinet"] if pair.get("skipped_reason") == "frame_geometry_changed"]
        self.assertEqual(len(skipped), 2)

    def test_window_cap_prioritizes_distinct_temporal_groups_over_one_long_group(self):
        frames = [frame(index*.5, (30, 30, 45, 45) if (1 <= index <= 21 and index % 2) or index == 36 else None)
                  for index in range(44)]
        result = propose_motion_windows(frames, coarse(), max_windows=2)
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(len({window["group_id"] for window in result["windows"]}), 2)
        self.assertTrue(result["coverage"]["capped"])
        self.assertTrue(any(window["reason"] == "window_cap" for window in result["dropped_windows"]))
        for window in result["windows"]:
            self.assertLessEqual(window["end_sec"]-window["start_sec"], 6+1e-9)

    def test_regions_share_window_budget_without_one_consuming_both_slots(self):
        result = propose_motion_windows(pulse(), coarse(desk=True), max_windows=2)
        self.assertEqual({window["roi"] for window in result["windows"]}, {"cabinet", "desk"})

    def test_dense_refinement_excludes_observed_frames_and_exposes_cap_omissions(self):
        result = propose_motion_windows(pulse(), coarse(), observed_indices=[0, 2, 6], max_refinement_frames=2)
        self.assertEqual(len(result["refinement_indices"]), 2)
        self.assertFalse(set(result["refinement_indices"]) & {0, 2, 6})
        self.assertTrue(result["refinement"]["capped"])
        self.assertTrue(result["refinement"]["dropped_indices"])
        window = result["windows"][0]
        self.assertFalse(window["refinement_complete"])
        self.assertEqual(set(window["eligible_refinement_indices"]),
                         set(window["refinement_indices"]) | set(window["refinement_dropped_indices"]))

    def test_refinement_round_robin_gives_each_separate_window_a_turn(self):
        frames = [frame(index*.5, (30, 30, 45, 45) if index in (3, 18) else None) for index in range(24)]
        result = propose_motion_windows(frames, coarse(), max_refinement_frames=2)
        self.assertEqual(len(result["windows"]), 2)
        self.assertTrue(all(len(window["refinement_indices"]) == 1 for window in result["windows"]))

    def test_no_refinement_cap_keeps_every_dense_frame_in_selected_window(self):
        result = propose_motion_windows(pulse(), coarse(), max_refinement_frames=100)
        window = result["windows"][0]
        self.assertTrue(window["refinement_complete"])
        self.assertEqual(window["refinement_indices"], list(range(window["frame_indices"][0], window["frame_indices"][-1]+1)))

    def test_zero_window_cap_records_all_eligible_windows_as_dropped(self):
        result = propose_motion_windows(pulse(), coarse(), max_windows=0)
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["refinement_indices"], [])
        self.assertTrue(result["coverage"]["capped"])
        self.assertEqual(result["coverage"]["candidate_window_count"], len(result["dropped_windows"]))

    def test_two_frames_cannot_fabricate_three_frame_context(self):
        result = propose_motion_windows([frame(0), frame(.5, (30, 30, 45, 45))], coarse())
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["dropped_windows"][0]["reason"], "insufficient_three_frame_context")

    def test_determinism_and_no_mutation_of_frames_or_coarse_observations(self):
        frames, observations = pulse(), coarse()
        original = copy.deepcopy((frames, observations))
        result = propose_motion_windows(frames, observations)
        self.assertEqual(result, propose_motion_windows(frames, observations))
        self.assertEqual((frames, observations), original)

    def test_invalid_configurations_and_frame_metadata_are_rejected(self):
        for kwargs in ({"max_windows": True}, {"max_refinement_frames": -1}, {"pixel_threshold": 256},
                       {"min_changed_fraction": 0}, {"min_changed_fraction": math.nan},
                       {"context_sec": 3, "max_window_sec": 6}, {"analysis_side": 0},
                       {"max_pair_gap_sec": 0}, {"roi_padding": 2}, {"observed_indices": [99]}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                propose_motion_windows(pulse(), coarse(), **kwargs)
        for timestamp in (-1, True, math.inf, "0"):
            frames = pulse()
            frames[0].timestamp_sec = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises((TypeError, ValueError)):
                propose_motion_windows(frames, coarse())
        frames = pulse()
        frames[1].timestamp_sec = frames[0].timestamp_sec
        with self.assertRaises(ValueError):
            propose_motion_windows(frames, coarse())


if __name__ == "__main__":
    unittest.main()
