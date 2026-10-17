"""State-driven sampling plans on synthetic timestamps, without model or video I/O."""

import copy
import math
from types import SimpleNamespace
import unittest

from analyzer.refinement import coarse_indices, refinement_indices


def frames(times):
    return [SimpleNamespace(timestamp_sec=timestamp) for timestamp in times]


def dense(end):
    return frames([index / 2 for index in range(int(end * 2) + 1)])


def observations(times, values, feature="person_in_work_area"):
    return [{"frame_id": f"f{index}", "timestamp_sec": timestamp, "states": {feature: value}}
            for index, (timestamp, value) in enumerate(zip(times, values))]


def selected_times(source, result):
    return [source[index].timestamp_sec for index in result["indices"]]


class CoarseSelectionTests(unittest.TestCase):
    def test_uses_variable_timestamps_and_preserves_endpoints(self):
        source = frames([0.0, 0.2, 0.9, 2.1, 4.5, 5.0])
        self.assertEqual(coarse_indices(source, interval_sec=2), [0, 3, 4, 5])

    def test_cap_spans_entire_clip_instead_of_truncating_end(self):
        source = dense(20)
        indices = coarse_indices(source, interval_sec=0.5, max_frames=3)
        self.assertEqual([source[index].timestamp_sec for index in indices], [0, 10, 20])

    def test_nearest_source_frame_is_not_duplicated(self):
        source = frames([0, 0.1, 0.2, 0.3, 0.4, 10])
        self.assertEqual(coarse_indices(source, interval_sec=2.5, max_frames=5), [0, 4, 5])

    def test_empty_single_and_long_requested_interval(self):
        self.assertEqual(coarse_indices([]), [])
        self.assertEqual(coarse_indices(frames([3.0])), [0])
        self.assertEqual(coarse_indices(dense(3), interval_sec=100), [0, 6])

    def test_tiny_positive_interval_is_capped_before_integer_conversion(self):
        self.assertEqual(coarse_indices(dense(3), interval_sec=5e-324, max_frames=2), [0, 6])

    def test_duplicate_dense_pts_keep_first_original_index_per_time(self):
        source = frames([0, 0, 0.5, 0.5, 1, 1, 1.5, 2, 2])
        self.assertEqual(coarse_indices(source, interval_sec=1), [0, 4, 7])
        self.assertEqual(coarse_indices(frames([0, 0, 0])), [0])

    def test_invalid_parameters_and_frame_timestamps(self):
        for value in (True, False, 0, -1, "2", None, math.nan, math.inf):
            with self.subTest(interval=value), self.assertRaises((ValueError, TypeError)):
                coarse_indices([], interval_sec=value)
        for value in (True, 0, 1, -1, 2.0, "2", math.nan):
            with self.subTest(max_frames=value), self.assertRaises((ValueError, TypeError)):
                coarse_indices([], max_frames=value)
        for source in (None, (), [object()], frames([1, 0]), frames([-1]),
                       frames([math.nan]), frames([math.inf]), frames([True]), frames(["0"])):
            with self.subTest(source=source), self.assertRaises((ValueError, TypeError)):
                coarse_indices(source)


class RefinementSelectionTests(unittest.TestCase):
    def test_brief_one_frame_change_refines_both_sides_without_duplicates(self):
        source = dense(6)
        rows = observations([0, 2, 4, 6], ["closed", "closed", "open", "closed"], "cabinet_door")
        result = refinement_indices(source, rows)
        self.assertEqual(selected_times(source, result), [1.5, 2.5, 3, 3.5, 4.5, 5, 5.5])
        self.assertEqual(result["candidate_count"], 7)
        self.assertFalse(result["capped"])
        self.assertEqual(len(result["indices"]), len(set(result["indices"])))

    def test_start_and_end_windows_do_not_require_frames_outside_video(self):
        source = dense(4)
        rows = observations([0, 2, 4], ["yes", "no", "yes"])
        result = refinement_indices(source, rows)
        self.assertEqual(selected_times(source, result), [0.5, 1, 1.5, 2.5, 3, 3.5])

    def test_last_single_new_state_gets_nearby_earlier_frames(self):
        source = dense(10)
        rows = observations([0, 2, 4, 6, 8, 10], ["yes", "yes", "yes", "yes", "yes", "no"])
        result = refinement_indices(source, rows)
        self.assertEqual(selected_times(source, result), [7.5, 8.5, 9, 9.5])

    def test_distant_gaps_are_not_refined_or_bridged(self):
        source = dense(10)
        self.assertEqual(refinement_indices(source, observations([0, 10], ["yes", "no"]))["indices"], [])
        self.assertEqual(refinement_indices(source, observations([0, 4.001], ["yes", "no"]))["indices"], [])

    def test_actual_gap_2_point_4_and_exactly_four_are_eligible(self):
        source = frames([0, 0.5, 1, 1.5, 2, 2.4, 2.9, 4])
        result = refinement_indices(source, observations([0, 2.4], ["yes", "no"]))
        self.assertEqual(selected_times(source, result), [0.5, 1, 1.5, 2, 2.9])
        self.assertTrue(refinement_indices(source, observations([0, 4], ["yes", "no"]))["indices"])

    def test_cap_spreads_across_separate_windows_nearest_centres_first(self):
        source = dense(20)
        rows = observations([0, 2, 10, 12, 20], ["yes", "no", "no", "yes", "yes"])
        result = refinement_indices(source, rows, max_new_frames=2)
        self.assertEqual(selected_times(source, result), [1, 11])
        self.assertEqual(result["candidate_count"], 9)
        self.assertTrue(result["capped"])
        extended = refinement_indices(source, rows, max_new_frames=4)
        self.assertEqual(selected_times(source, extended), [0.5, 1, 10.5, 11])

    def test_zero_budget_reports_unselected_candidates(self):
        result = refinement_indices(dense(2), observations([0, 2], ["yes", "no"]), max_new_frames=0)
        self.assertEqual(result, {"indices": [], "candidate_count": 3, "capped": True})

    def test_uncertain_next_to_known_triggers_but_unseen_and_missing_do_not(self):
        source = dense(2)
        for values in (["yes", "uncertain"], ["uncertain", "no"]):
            with self.subTest(values=values):
                self.assertEqual(refinement_indices(source, observations([0, 2], values))["candidate_count"], 3)
        for values in (["uncertain", "uncertain"], ["open", "not_visible"], ["not_visible", "closed"]):
            with self.subTest(values=values):
                self.assertEqual(refinement_indices(source, observations([0, 2], values, "cabinet_door"))["indices"], [])
        rows = observations([0, 2], ["yes", "no"])
        rows[1]["states"] = {}
        self.assertEqual(refinement_indices(source, rows)["indices"], [])

    def test_technical_error_is_not_visual_uncertainty_and_is_not_bridged(self):
        rows = observations([0, 2, 4], ["yes", "uncertain", "no"])
        rows[1].update(status="partial", errors=[{"feature": "person_in_work_area", "message": "timeout"}])
        self.assertEqual(refinement_indices(dense(4), rows)["indices"], [])
        rows[1].update(status="error", errors=[])
        self.assertEqual(refinement_indices(dense(4), rows)["indices"], [])

    def test_feature_specific_error_does_not_hide_other_valid_change(self):
        rows = observations([0, 2], ["yes", "uncertain"])
        rows[0]["states"]["cabinet_door"] = "closed"
        rows[1]["states"]["cabinet_door"] = "open"
        rows[1].update(status="partial", errors=[{"feature": "person_in_work_area", "message": "timeout"}])
        self.assertEqual(refinement_indices(dense(2), rows)["candidate_count"], 3)
        rows[1]["errors"] = ["general failure"]
        self.assertEqual(refinement_indices(dense(2), rows)["indices"], [])

    def test_already_observed_timestamps_are_excluded_with_epsilon(self):
        source = dense(2.5)
        rows = observations([0.0000004, 2.0000004], ["yes", "no"])
        result = refinement_indices(source, rows)
        self.assertEqual(selected_times(source, result), [0.5, 1, 1.5, 2.5])

    def test_duplicate_dense_pts_are_not_selected_or_counted_twice(self):
        source = frames([0, 0, 0.5, 0.5, 1, 1, 1.5, 2, 2])
        result = refinement_indices(source, observations([0, 2], ["yes", "no"]))
        self.assertEqual(result, {"indices": [2, 4, 6], "candidate_count": 3, "capped": False})

    def test_unchanged_empty_and_fully_observed_source_have_no_candidates(self):
        self.assertEqual(refinement_indices([], []), {"indices": [], "candidate_count": 0, "capped": False})
        self.assertEqual(refinement_indices(dense(2), observations([0, 2], ["yes", "yes"]))["indices"], [])
        self.assertEqual(refinement_indices(frames([0, 2]), observations([0, 2], ["yes", "no"]))["candidate_count"], 0)

    def test_inputs_are_not_changed_and_irrelevant_feature_is_ignored(self):
        source = dense(2)
        rows = observations([0, 2], ["yes", "no"])
        rows[0]["states"]["other"] = {"arbitrary": "value"}
        before = copy.deepcopy((source, rows))
        refinement_indices(source, rows)
        self.assertEqual((source, rows), before)

    def test_invalid_limits_and_observation_structure_are_rejected(self):
        for value in (True, -1, 2.0, "2", None, math.nan):
            with self.subTest(limit=value), self.assertRaises((ValueError, TypeError)):
                refinement_indices([], [], max_new_frames=value)
        for value in (None, {}, (), [None]):
            with self.subTest(observations=value), self.assertRaises((ValueError, TypeError)):
                refinement_indices([], value)
        for key, value in (("frame_id", ""), ("frame_id", 1), ("timestamp_sec", -1),
                           ("timestamp_sec", math.nan), ("timestamp_sec", True), ("states", []),
                           ("states", {"person_in_work_area": "present"}), ("status", "failed"),
                           ("errors", "timeout")):
            rows = observations([0], ["yes"])
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                refinement_indices([], rows)
        for times in ([1, 0], [1, 1]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                refinement_indices(dense(2), observations(times, ["yes", "no"]))
        rows = observations([0, 2], ["yes", "no"])
        rows[1]["frame_id"] = rows[0]["frame_id"]
        with self.assertRaises(ValueError):
            refinement_indices(dense(2), rows)


class CashRefinementTests(unittest.TestCase):
    def test_each_cash_feature_can_trigger_short_window_refinement(self):
        for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine"):
            rows = observations([0, 2], ["yes", "no"], feature)
            with self.subTest(feature=feature):
                self.assertEqual(selected_times(dense(2), refinement_indices(dense(2), rows)), [0.5, 1, 1.5])
                rows[1]["states"][feature] = "uncertain"
                self.assertEqual(refinement_indices(dense(2), rows)["candidate_count"], 3)

    def test_error_in_one_cash_feature_blocks_it_but_preserves_other_valid_signals(self):
        rows = observations([0, 2], ["yes", "no"], "cash_on_worktop")
        rows[1].update(status="partial", errors=[{"feature": "cash_on_worktop", "message": "invalid answer"}])
        self.assertEqual(refinement_indices(dense(2), rows)["indices"], [])
        for feature, before, after in (("person_in_work_area", "yes", "no"),
                                        ("cabinet_door", "open", "closed"),
                                        ("cash_in_hands", "no", "yes"),
                                        ("notes_in_machine", "no", "yes")):
            changed = copy.deepcopy(rows)
            changed[0]["states"][feature] = before
            changed[1]["states"][feature] = after
            with self.subTest(feature=feature):
                self.assertEqual(refinement_indices(dense(2), changed)["candidate_count"], 3)

    def test_many_cash_changes_in_same_window_share_one_sampling_budget(self):
        rows = observations([0, 2], ["yes", "yes"])
        for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine"):
            rows[0]["states"][feature] = "no"
            rows[1]["states"][feature] = "yes"
        result = refinement_indices(dense(2), rows, max_new_frames=1)
        self.assertEqual(result, {"indices": [2], "candidate_count": 3, "capped": True})

    def test_cash_window_shares_cap_with_separate_person_window(self):
        rows = observations([0, 2, 10, 12], ["yes", "no", "no", "no"])
        rows[2]["states"]["notes_in_machine"] = "no"
        rows[3]["states"]["notes_in_machine"] = "yes"
        source = dense(12.5)
        result = refinement_indices(source, rows, max_new_frames=2)
        self.assertEqual(selected_times(source, result), [1, 11])
        self.assertTrue(result["capped"])

    def test_legacy_report_and_unobserved_cash_fields_preserve_original_selection(self):
        rows = observations([0, 2], ["open", "closed"], "cabinet_door")
        expected = refinement_indices(dense(2), rows)
        for row in rows:
            row["states"].update({"cash_on_worktop": "uncertain", "cash_in_hands": "uncertain",
                                  "notes_in_machine": "uncertain"})
        self.assertEqual(refinement_indices(dense(2), rows), expected)
        rows[0]["states"].pop("cash_on_worktop")
        rows[1]["states"]["cash_on_worktop"] = "yes"
        self.assertEqual(refinement_indices(dense(2), rows), expected)

    def test_technical_cash_uncertainty_does_not_trigger_refinement(self):
        rows = observations([0, 2], ["yes", "uncertain"], "notes_in_machine")
        self.assertEqual(refinement_indices(dense(2), rows)["candidate_count"], 3)
        rows[1].update(status="partial", errors=[{"feature": "notes_in_machine", "message": "timeout"}])
        self.assertEqual(refinement_indices(dense(2), rows)["indices"], [])

    def test_cash_rejects_values_specific_to_cabinet_or_invalid_types(self):
        for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine"):
            for value in ("closed", "not_visible", True, ["no"]):
                rows = observations([0, 2], ["yes", value], feature)
                with self.subTest(feature=feature, value=value), self.assertRaises(ValueError):
                    refinement_indices(dense(2), rows)


if __name__ == "__main__":
    unittest.main()
