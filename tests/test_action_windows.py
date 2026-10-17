"""Synthetic selection tests; no dataset, model, or video decoding required."""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

from analyzer.action_windows import combine_action_motion, gate_cabinet_claim, select_action_windows


def window(priority, roi="cabinet", group=None, start=None, reason=None):
    start = priority * 10 if start is None else start
    row = {"roi": roi, "group_id": group or f"{roi}-{priority}", "priority": priority,
           "start_sec": start, "end_sec": start + 2,
           "frame_indices": [priority * 3, priority * 3 + 1, priority * 3 + 2],
           "frame_timestamps": [start, start + 1, start + 2]}
    if reason:
        row["reason"] = reason
    return row


def observation(time, state="open", status="ok", errors=None):
    return {"timestamp_sec": time, "states": {"cabinet_door": state},
            "status": status, "errors": [] if errors is None else errors}


class ActionWindowTests(unittest.TestCase):
    def test_open_cabinet_recovers_cap_dropped_group_and_preserves_both_rois(self):
        motion = {"windows": [window(1), window(2, "desk"), window(3), window(4, "desk")],
                  "dropped_windows": [window(17, start=115.5, reason="window_cap"),
                                      window(13, start=122.5, reason="window_cap")]}
        result = select_action_windows(motion, [observation(117.5), observation(123)], 4)
        self.assertEqual([w["priority"] for w in result["windows"]], [13, 2, 17, 4])
        self.assertEqual([w["action_priority"] for w in result["windows"]], [1, 2, 3, 4])
        self.assertEqual(result["windows"][0]["reason"], "observed_open_cabinet")
        self.assertEqual(result["windows"][0]["selection_source"], "motion_dropped")
        self.assertTrue(result["capped"])
        self.assertEqual(result["selected_count"], 4)

    def test_no_open_preserves_original_priority_including_cap_recovery(self):
        motion = {"windows": [window(3), window(1)],
                  "dropped_windows": [window(2, "desk", reason="window_cap")]}
        for rows in ([], [observation(31, "closed")], [observation(1000)], [observation(21)]):
            with self.subTest(rows=rows):
                result = select_action_windows(motion, rows, 2)
                self.assertEqual([w["priority"] for w in result["windows"]], [1, 2])
                self.assertFalse(result["state_conditioned"])

    def test_distinct_groups_precede_repeated_chunks_even_if_all_chunks_overlap_open(self):
        motion = {"windows": [window(1, group="long", start=10), window(2, "desk"),
                              window(3, group="long", start=11), window(4, start=30),
                              window(5, "desk"), window(6, group="long", start=12)]}
        result = select_action_windows(motion, [observation(12)], 4)
        self.assertEqual([w["priority"] for w in result["windows"]], [1, 2, 4, 5])
        self.assertEqual(len({(w["roi"], w["group_id"]) for w in result["windows"]}), 4)

    def test_matching_chunk_is_group_representative(self):
        motion = {"windows": [window(1, group="long"), window(2, "desk")],
                  "dropped_windows": [window(9, group="long", start=90, reason="window_cap")]}
        result = select_action_windows(motion, [observation(91)], 2)
        self.assertEqual([w["priority"] for w in result["windows"]], [9, 2])

    def test_failed_or_unreliable_open_cannot_change_selection(self):
        motion = {"windows": [window(1), window(2, "desk")],
                  "dropped_windows": [window(9, reason="window_cap")]}
        rows = [observation(91, status="error"), observation(91, status="partial"),
                observation(91, errors=[{"feature": "cabinet_door"}]),
                observation(91, errors=[{"message": "timeout"}]),
                observation(91, errors=["failure"]), observation(91, errors=[{"feature": "unknown"}]),
                observation(math.nan), observation(True)]
        for row in rows:
            with self.subTest(row=row):
                result = select_action_windows(motion, [row], 2)
                self.assertEqual([w["priority"] for w in result["windows"]], [1, 2])
                self.assertFalse(result["state_conditioned"])

    def test_unrelated_feature_error_does_not_discard_valid_cabinet_state(self):
        motion = {"windows": [window(1), window(2, "desk"), window(9)]}
        rows = [observation(91, status="partial", errors=[{"feature": "cash_on_worktop"}])]
        result = select_action_windows(motion, rows, 2)
        self.assertEqual([w["priority"] for w in result["windows"]], [9, 2])

    def test_invalid_triples_and_noncap_drops_are_never_selected(self):
        for field, value in (("frame_indices", [1, 1, 2]), ("frame_indices", [1, 2]),
                             ("frame_indices", [True, 2, 3]), ("frame_timestamps", [10, math.nan, 12]),
                             ("frame_timestamps", [10, 12, 11]), ("end_sec", 50),
                             ("roi", "other")):
            bad = window(1)
            bad[field] = value
            with self.subTest(field=field, value=value):
                result = select_action_windows({"windows": [bad]}, [], 8)
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["dropped_windows"][0]["reason"], "invalid_action_window")
        result = select_action_windows({"dropped_windows": [window(1, reason="insufficient_three_frame_context")]},
                                       [observation(11)])
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["dropped_windows"][0]["reason"], "insufficient_three_frame_context")

    def test_duplicate_window_is_not_queried_twice(self):
        row = window(1)
        result = select_action_windows({"windows": [row], "dropped_windows": [{**row, "reason": "window_cap"}]}, [])
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["dropped_windows"][0]["reason"], "duplicate_action_window")

    def test_zero_empty_caps_and_input_validation(self):
        self.assertEqual(select_action_windows({}, [])["selected_count"], 0)
        result = select_action_windows({"windows": [window(1)]}, [], 0)
        self.assertEqual(result["windows"], [])
        self.assertTrue(result["capped"])
        self.assertEqual(result["dropped_windows"][0]["reason"], "action_window_cap")
        for cap in (-1, True, 1.0, math.inf):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select_action_windows({}, [], cap)

    def test_deterministic_no_mutation_or_motion_refinement_changes(self):
        motion = {"windows": [window(1), window(2, "desk")], "refinement_indices": [7, 8],
                  "refinement": {"selection_order": [8, 7]}}
        motion["windows"][0]["refinement_indices"] = [7]
        rows = [observation(11)]
        original = deepcopy((motion, rows))
        first = select_action_windows(motion, rows)
        self.assertEqual(first, select_action_windows(motion, rows))
        self.assertEqual((motion, rows), original)
        first["windows"][0]["refinement_indices"].append(99)
        self.assertEqual((motion, rows), original)
        self.assertNotEqual(first["policy_sha256"], select_action_windows(motion, rows, 7)["policy_sha256"])
        self.assertEqual(len(first["policy_sha256"]), 64)


class CombineActionMotionTests(unittest.TestCase):
    def setUp(self):
        # Nonuniform PTS: the temporal midpoint is index 2, not index 3
        # (the array midpoint), and not index 4 (the strongest motion).
        self.frames = [SimpleNamespace(timestamp_sec=t) for t in (0, .2, 1.4, 2.2, 2.7, 3)]

    def proposal(self, roi="desk", priority=1, *, reason=None):
        row = window(priority, roi, start=0, reason=reason)
        row.update(end_sec=3, frame_indices=[0, 4, 5], frame_timestamps=[0, 2.7, 3])
        return row

    def test_only_cabinet_from_general_motion_and_desk_from_counter_motion(self):
        general = {"windows": [self.proposal("cabinet"), self.proposal("desk", 2)],
                   "dropped_windows": [self.proposal("desk", 3, reason="window_cap")]}
        counter = {"windows": [self.proposal("desk"), self.proposal("cabinet", 2)]}
        result = combine_action_motion(general, counter, self.frames)
        self.assertEqual([(w["roi"], w["priority"]) for w in result["windows"]],
                         [("cabinet", 1), ("desk", 2)])
        self.assertEqual(result["dropped_windows"], [])
        self.assertEqual(result["windows"][1]["group_id"], "counter-desk-1")

    def test_missing_counter_does_not_fall_back_to_whole_desk(self):
        general = {"windows": [self.proposal("desk"), self.proposal("cabinet", 2)]}
        result = combine_action_motion(general, {}, self.frames)
        self.assertEqual([w["roi"] for w in result["windows"]], ["cabinet"])

    def test_temporal_midpoint_is_actual_source_frame_not_peak_or_array_middle(self):
        result = combine_action_motion({}, {"windows": [self.proposal()]}, self.frames)
        selected = result["windows"][0]
        self.assertEqual(selected["frame_indices"], [0, 2, 5])
        self.assertEqual(selected["frame_timestamps"], [0, 1.4, 3])
        self.assertEqual(selected["motion_peak_frame_index"], 4)
        self.assertEqual(selected["anchor_policy"], "actual_temporal_midpoint")
        self.assertNotIn(1.5, selected["frame_timestamps"])

    def test_midpoint_tie_uses_earlier_actual_frame_and_keeps_endpoints(self):
        frames = [SimpleNamespace(timestamp_sec=t) for t in (0, 1, 2, 3)]
        row = self.proposal()
        row.update(frame_indices=[0, 2, 3], frame_timestamps=[0, 2, 3])
        selected = combine_action_motion({}, {"windows": [row]}, frames)["windows"][0]
        self.assertEqual(selected["frame_indices"], [0, 1, 3])
        self.assertEqual(selected["frame_timestamps"], [0, 1, 3])

    def test_out_of_range_counter_indices_are_rejected(self):
        for indices in ([0, 4, 6], [5, 6, 7]):
            row = self.proposal()
            row["frame_indices"] = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                combine_action_motion({}, {"windows": [row]}, self.frames)

    def test_out_of_range_cabinet_indices_are_also_rejected(self):
        row = self.proposal("cabinet")
        row["frame_indices"] = [0, 4, 6]
        with self.assertRaises(ValueError):
            combine_action_motion({"windows": [row]}, {}, self.frames)

    def test_every_anchor_timestamp_must_match_source_before_midpoint_replacement(self):
        for roi in ("cabinet", "desk"):
            for position in range(3):
                row = self.proposal(roi)
                row["frame_timestamps"][position] += .1
                row["start_sec"] = row["frame_timestamps"][0]
                row["end_sec"] = row["frame_timestamps"][-1]
                general = {"windows": [row]} if roi == "cabinet" else {}
                counter = {"windows": [row]} if roi == "desk" else {}
                with self.subTest(roi=roi, position=position), self.assertRaises(ValueError):
                    combine_action_motion(general, counter, self.frames)

    def test_counter_window_cannot_exceed_three_seconds_but_cabinet_can(self):
        frames = [SimpleNamespace(timestamp_sec=t) for t in (0, .2, 1.4, 2.2, 2.7, 3.5)]
        for roi in ("cabinet", "desk"):
            row = self.proposal(roi)
            row.update(end_sec=3.5, frame_timestamps=[0, 2.7, 3.5])
            with self.subTest(roi=roi):
                if roi == "desk":
                    with self.assertRaises(ValueError):
                        combine_action_motion({}, {"windows": [row]}, frames)
                else:
                    combined = combine_action_motion({"windows": [row]}, {}, frames)
                    self.assertEqual(combined["windows"][0]["frame_timestamps"], [0, 2.7, 3.5])

    def test_invalid_triples_remain_ineligible_after_combining(self):
        for indices in ([-1, 2, 5], [0, 0, 5], [0, 5], [0, True, 5]):
            row = self.proposal()
            row["frame_indices"] = indices
            with self.subTest(indices=indices):
                combined = combine_action_motion({}, {"windows": [row]}, self.frames)
                selection = select_action_windows(combined, [])
                self.assertEqual(selection["windows"], [])
                self.assertEqual(selection["dropped_windows"][0]["reason"], "invalid_action_window")

    def test_cap_dropped_counter_windows_get_real_midpoint_and_stay_recoverable(self):
        counter = {"dropped_windows": [self.proposal(reason="window_cap")]}
        combined = combine_action_motion({}, counter, self.frames)
        row = combined["dropped_windows"][0]
        self.assertEqual(row["frame_indices"], [0, 2, 5])
        self.assertEqual(row["reason"], "window_cap")
        selection = select_action_windows(combined, [])
        self.assertEqual(selection["selected_count"], 1)
        self.assertEqual(selection["windows"][0]["selection_source"], "motion_dropped")

    def test_no_mutation_and_cabinet_anchor_remains_motion_peak(self):
        general = {"windows": [self.proposal("cabinet")], "refinement_indices": [1, 2]}
        counter = {"windows": [self.proposal()], "refinement_indices": []}
        original = deepcopy((general, counter, self.frames))
        combined = combine_action_motion(general, counter, self.frames)
        self.assertEqual((general, counter, self.frames), original)
        cabinet = combined["windows"][0]
        self.assertEqual(cabinet["frame_indices"], [0, 4, 5])
        self.assertEqual(cabinet["frame_timestamps"], [0, 2.7, 3])
        combined["windows"][0]["frame_indices"][0] = 99
        combined["windows"][1]["frame_timestamps"][0] = 99
        self.assertEqual((general, counter, self.frames), original)


class CabinetClaimGateTests(unittest.TestCase):
    def claim(self, near="yes", region="cabinet"):
        return {"region": region, "start_sec": 10.0, "end_sec": 13.0, "status": "ok",
                "result": {"cash_near_cabinet": near, "cash_into_cabinet": "uncertain",
                           "interaction": "uncertain", "counter_direction": "uncertain"},
                "calls": [{"question": "cabinet_sequence", "completed": True,
                           "response": {"observation": {"paper_near_open_cabinet": near,
                                                        "entry_visible": "yes"}}}],
                "raw_entry_visible": "yes", "errors": []}

    def assert_unsupported(self, result):
        self.assertEqual(result["result"]["cash_near_cabinet"], "uncertain")
        self.assertEqual(result["cash_near_cabinet_before_gate"], "yes")
        self.assertEqual(result["cabinet_state_gate"], "no_observed_open_in_window")
        self.assertEqual(result["cabinet_state_gate_open_times"], [])

    def test_closed_uncertain_or_missing_door_support_changes_yes_to_uncertain_not_no(self):
        for rows in ([], [observation(11, "closed")], [observation(11, "uncertain")],
                     [observation(11, "not_visible")]):
            with self.subTest(rows=rows):
                self.assert_unsupported(gate_cabinet_claim(self.claim(), rows))

    def test_errored_open_or_related_partial_error_cannot_support_positive_claim(self):
        for row in (observation(11, status="error"), observation(11, status="partial"),
                    observation(11, status="partial", errors=[{"feature": "cabinet_door"}]),
                    observation(11, errors=[{"feature": "cabinet_door"}]),
                    observation(11, errors=[{"message": "transport failed"}])):
            with self.subTest(row=row):
                self.assert_unsupported(gate_cabinet_claim(self.claim(), [row]))

    def test_open_outside_window_is_not_interpolated_across_gap(self):
        self.assert_unsupported(gate_cabinet_claim(self.claim(), [observation(9.9), observation(13.1)]))

    def test_trusted_open_inside_window_preserves_yes_and_records_only_supporting_times(self):
        rows = [observation(9), observation(10), observation(11, "closed"),
                observation(12), observation(13), observation(14)]
        result = gate_cabinet_claim(self.claim(), rows)
        self.assertEqual(result["result"]["cash_near_cabinet"], "yes")
        self.assertEqual(result["cash_near_cabinet_before_gate"], "yes")
        self.assertEqual(result["cabinet_state_gate"], "observed_open_in_window")
        self.assertEqual(result["cabinet_state_gate_open_times"], [10, 12, 13])
        self.assertEqual(result["result"]["cash_into_cabinet"], "uncertain")

    def test_partial_unrelated_error_keeps_valid_cabinet_support(self):
        result = gate_cabinet_claim(self.claim(), [observation(
            11, status="partial", errors=[{"feature": "person_in_work_area"}])])
        self.assertEqual(result["result"]["cash_near_cabinet"], "yes")
        self.assertEqual(result["cabinet_state_gate_open_times"], [11])

    def test_no_uncertain_and_other_regions_remain_unchanged(self):
        for near, region in (("no", "cabinet"), ("uncertain", "cabinet"), ("yes", "desk")):
            original = self.claim(near, region)
            with self.subTest(near=near, region=region):
                result = gate_cabinet_claim(original, [observation(11)])
                self.assertEqual(result, original)
                self.assertIsNot(result, original)

    def test_deepcopy_preserves_calls_all_other_heads_and_inputs(self):
        original = self.claim()
        rows = [observation(11, "closed")]
        snapshot = deepcopy((original, rows))
        result = gate_cabinet_claim(original, rows)
        self.assertEqual((original, rows), snapshot)
        self.assertEqual(result["calls"], original["calls"])
        self.assertEqual(result["raw_entry_visible"], "yes")
        for key in ("cash_into_cabinet", "interaction", "counter_direction"):
            self.assertEqual(result["result"][key], original["result"][key])
        result["calls"][0]["response"]["observation"]["entry_visible"] = "no"
        self.assertEqual((original, rows), snapshot)

    def test_invalid_window_bounds_cannot_accept_open_support(self):
        for start, end in ((math.nan, 13), (10, math.inf), (14, 13), (True, 13), ("10", 13)):
            original = self.claim()
            original.update(start_sec=start, end_sec=end)
            with self.subTest(start=start, end=end):
                self.assert_unsupported(gate_cabinet_claim(original, [observation(11)]))


if __name__ == "__main__":
    unittest.main()
