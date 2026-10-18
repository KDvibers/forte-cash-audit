"""Selection failure cases on synthetic PTS; no image decoding or model calls."""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

from analyzer.direction_windows import select_direction_windows


def pool(times):
    # Deliberately invalid JPEG bytes: the selector must never inspect the image.
    return [SimpleNamespace(timestamp_sec=t, jpeg=b"not-an-image") for t in times]


def regular(end=10):
    return pool([index / 2 for index in range(2*end+1)])


def action(name="a", start=1, end=3, box=None, **overrides):
    return {"window_id": name, "status": "ok", "region": "desk", "errors": [],
            "result": {"interaction": "yes"}, "object_visibility": "yes",
            "object_bbox_normalized": [100, 100, 300, 300] if box is None else box,
            "start_sec": start, "end_sec": end, **overrides}


class DirectionWindowTests(unittest.TestCase):
    def test_neighboring_positive_windows_merge_and_keep_actual_order(self):
        frames = regular()
        result = select_direction_windows(frames, [action("b", 3.5, 5.5), action("a", 1, 3)])
        self.assertEqual(result["selected_count"], 1)
        window = result["windows"][0]
        self.assertEqual(window["source_window_ids"], ["a", "b"])
        self.assertEqual(window["frame_timestamps"], [1, 2, 3, 4.5, 5.5])
        self.assertEqual(window["frame_timestamps"], [frames[i].timestamp_sec for i in window["frame_indices"]])
        self.assertNotIn("verdict", window)
        self.assertNotIn("direction", window)

    def test_gap_and_duration_bound_prevent_overmerging(self):
        separated = select_direction_windows(regular(), [action("a", 1, 2), action("b", 3.5, 4.5)])
        self.assertEqual(separated["coverage"]["temporal_episode_count"], 2)
        bounded = select_direction_windows(regular(), [action("a", 0, 4), action("b", 4.5, 7)])
        self.assertEqual(bounded["coverage"]["temporal_episode_count"], 1)
        self.assertEqual(bounded["candidate_count"], 2)
        self.assertTrue(all(w["end_sec"]-w["start_sec"] <= 6 for w in bounded["windows"]))

    def test_different_counters_do_not_merge_even_when_times_overlap(self):
        rows = [action("left", 1, 3), action("right", 1.5, 3.5, [600, 100, 800, 300])]
        result = select_direction_windows(regular(), rows)
        self.assertEqual(result["selected_count"], 2)
        self.assertEqual([w["source_window_ids"] for w in result["windows"]], [["left"], ["right"]])

    def test_interleaved_other_counter_does_not_hide_matching_neighbor(self):
        rows = [action("left1", 1, 2), action("right", 1.5, 2.5, [600, 100, 800, 300]),
                action("left2", 2.5, 3.5)]
        result = select_direction_windows(regular(), rows)
        self.assertEqual([w["source_window_ids"] for w in result["windows"]], [["left1", "left2"], ["right"]])

    def test_pairwise_iou_chain_cannot_drift_between_objects(self):
        rows = [action("a", 1, 2, [0, 0, 300, 300]),
                action("b", 2, 3, [75, 0, 375, 300]),
                action("c", 3, 4, [150, 0, 450, 300])]
        result = select_direction_windows(regular(), rows)
        self.assertEqual([w["source_window_ids"] for w in result["windows"]], [["a", "b"], ["c"]])
        self.assertGreaterEqual(result["windows"][0]["bbox_agreement_min_iou"], .5)

    def test_stable_geometry_is_an_observed_representative_not_a_union(self):
        boxes = [[100, 100, 300, 300], [110, 100, 310, 300], [120, 100, 320, 300]]
        rows = [action(str(i), 1+i*.5, 2+i*.5, box) for i, box in enumerate(boxes)]
        first = select_direction_windows(regular(), rows)
        self.assertEqual(first["windows"][0]["object_bbox_normalized"], boxes[1])
        self.assertEqual(first, select_direction_windows(regular(), list(reversed(rows))))

    def test_failed_partial_hidden_or_nonpositive_checks_cannot_qualify(self):
        cases = [({"status": "partial"}, "action_not_ok"), ({"status": "error"}, "action_not_ok"),
                 ({"status": None}, "action_not_ok"), ({"region": "cabinet"}, "region_not_desk"),
                 ({"errors": [{"message": "timeout"}]}, "action_errors"), ({"errors": ""}, "action_errors"),
                 ({"result": {"interaction": "uncertain"}}, "interaction_not_yes"),
                 ({"result": {"interaction": "no"}}, "interaction_not_yes"),
                 ({"result": []}, "interaction_not_yes"), ({"object_visibility": "uncertain"}, "object_not_visible")]
        for override, reason in cases:
            with self.subTest(override=override):
                result = select_direction_windows(regular(), [action(**override)])
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["dropped_windows"][0]["reason"], reason)

    def test_invalid_geometry_is_not_replaced_with_a_full_frame_box(self):
        for box in (None, [], [0, 0, 1001, 1000], [100, 100, 100, 200], [300, 100, 100, 200],
                    [0.0, 0, 100, 100], [False, 0, 100, 100], [0, 0, math.inf, 100]):
            with self.subTest(box=box):
                row = action()
                row["object_bbox_normalized"] = box
                result = select_direction_windows(regular(), [row])
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["dropped_windows"][0]["reason"], "invalid_bbox")

    def test_interval_must_be_valid_and_within_actual_pool(self):
        for start, end in ((True, 2), (math.nan, 2), (1, math.inf), (-1, 2), (3, 2), (0, 11)):
            with self.subTest(start=start, end=end):
                result = select_direction_windows(regular(), [action(start=start, end=end)])
                self.assertFalse(result["windows"])
                self.assertIn(result["dropped_windows"][0]["reason"], ("invalid_interval", "interval_outside_frame_pool"))

    def test_evidence_timestamps_cannot_point_to_unsampled_or_wrong_frames(self):
        for evidence in ([], [{"timestamp_sec": 1}, {"timestamp_sec": 2.1}, {"timestamp_sec": 3}],
                         [{"timestamp_sec": 1}, {"timestamp_sec": 1}, {"timestamp_sec": 3}],
                         [{"timestamp_sec": 0}, {"timestamp_sec": 3}], [{"timestamp_sec": math.nan}]):
            with self.subTest(evidence=evidence):
                result = select_direction_windows(regular(), [action(evidence=evidence)])
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["dropped_windows"][0]["reason"], "invalid_source_evidence")
        valid = action(evidence=[{"timestamp_sec": t} for t in (1, 2, 3)])
        self.assertEqual(select_direction_windows(regular(), [valid])["selected_count"], 1)

    def test_real_pts_selection_does_not_round_to_nominal_frame_times(self):
        frames = pool([0, .12, .46, 1.03, 1.98, 2.04, 2.51, 3.07])
        result = select_direction_windows(frames, [action(start=.12, end=2.51)])
        window = result["windows"][0]
        self.assertEqual(window["frame_indices"], [1, 2, 3, 4, 6])
        self.assertEqual(window["frame_timestamps"], [.12, .46, 1.03, 1.98, 2.51])
        self.assertEqual(len(set(window["frame_indices"])), len(window["frame_indices"]))
        self.assertAlmostEqual(result["coverage"]["actual_max_frame_gap_sec"], .95)
        self.assertTrue(result["coverage"]["blind_shorter_actions"])

    def test_short_window_adds_context_without_losing_contact_anchor(self):
        rows = [action(start=.5, end=1.5, contact_evidence_sec=[999])]
        result = select_direction_windows(regular(2), rows)
        window = result["windows"][0]
        self.assertEqual(window["frame_timestamps"], [0, .5, 1, 1.5, 2])
        self.assertEqual(window["context_indices_added"], [0, 4])
        rows[0]["contact_evidence_sec"] = [math.nan, "not a time"]
        self.assertEqual(result, select_direction_windows(regular(2), rows))

    def test_full_five_interior_anchors_cannot_be_evicted_for_context(self):
        window = select_direction_windows(regular(), [action(start=2, end=4)])["windows"][0]
        self.assertEqual(window["frame_timestamps"], [2, 2.5, 3, 3.5, 4])
        self.assertEqual(window["context_indices_added"], [])
        self.assertEqual([r["reason"] for r in window["context_candidates_dropped"]], ["no_spare_frame_slot"]*2)

    def test_opt_in_reserves_before_and_after_instead_of_five_inside_frames(self):
        frames = regular()
        rows = [action(start=2, end=4)]
        legacy = select_direction_windows(frames, rows)
        selected = select_direction_windows(frames, rows, reserve_context=True)
        window = selected["windows"][0]
        self.assertEqual(window["frame_timestamps"], [1.5, 2, 3, 4, 4.5])
        self.assertEqual(window["context_indices_added"], [3, 9])
        self.assertEqual(window["context_candidates_dropped"], [])
        self.assertEqual(window["source_window_ids"], legacy["windows"][0]["source_window_ids"])
        self.assertEqual(selected["coverage"]["selected_image_reference_count"], 5)
        self.assertFalse(selected["coverage"]["upstream_missed_or_uncertain_interactions_recovered"])

    def test_context_mode_preserves_irregular_actual_pts_and_input_bytes(self):
        times = [.97, 1.03, 1.54, 1.98, 2.04, 2.51, 2.98]
        frames = pool(times)
        row = action(start=1.03, end=2.51)
        snapshot = deepcopy(row)
        window = select_direction_windows(frames, [row], reserve_context=True)["windows"][0]
        self.assertEqual(window["frame_timestamps"], [.97, 1.03, 1.98, 2.51, 2.98])
        self.assertEqual(window["frame_timestamps"], [frames[i].timestamp_sec for i in window["frame_indices"]])
        self.assertEqual(row, snapshot)
        self.assertEqual([f.timestamp_sec for f in frames], times)
        self.assertEqual([f.jpeg for f in frames], [b"not-an-image"]*len(times))

    def test_context_mode_edges_report_missing_side_without_filling_with_inside(self):
        for start, end, expected, missing in (
                (0, 2, [0, 1, 2, 2.5], "before"),
                (2, 4, [1.5, 2, 3, 4], "after")):
            with self.subTest(side=missing):
                window = select_direction_windows(regular(4), [action(start=start, end=end)],
                                                  reserve_context=True)["windows"][0]
                self.assertEqual(window["frame_timestamps"], expected)
                self.assertEqual(window["context_candidates_dropped"], [{
                    "side": missing, "frame_index": None, "reason": "video_edge_or_no_frame_in_context"}])
        one = select_direction_windows(pool([0]), [action(start=0, end=0)], reserve_context=True)
        self.assertEqual(one["windows"], [])
        self.assertEqual(one["dropped_windows"][0]["reason"], "insufficient_actual_frames")

    def test_context_mode_never_trades_away_endpoints_to_exceed_six_seconds(self):
        window = select_direction_windows(regular(8), [action(start=1, end=6.5)],
                                          reserve_context=True)["windows"][0]
        self.assertEqual(window["frame_timestamps"], [.5, 1, 3.5, 6.5])
        self.assertEqual(window["end_sec"]-window["start_sec"], 6)
        self.assertEqual(window["context_candidates_dropped"], [{
            "side": "after", "frame_index": 14, "reason": "duration_cap"}])
        full = select_direction_windows(regular(8), [action(start=1, end=7)],
                                        reserve_context=True)["windows"][0]
        self.assertEqual(full["frame_timestamps"], [1, 4, 7])
        self.assertEqual([r["reason"] for r in full["context_candidates_dropped"]], ["duration_cap"]*2)

    def test_context_mode_distinct_middle_keeps_sparse_interior_samples(self):
        # The nominal midpoint is past every interior sample. It must not
        # collapse the middle choice onto the already selected endpoint.
        frames = pool([0, 1, 1.01, 1.02, 3.1, 4])
        window = select_direction_windows(frames, [action(start=.5, end=3)],
                                          reserve_context=True)["windows"][0]
        self.assertEqual(window["frame_timestamps"], [0, 1, 1.01, 1.02, 3.1])
        self.assertEqual(len(set(window["frame_indices"])), 5)

    def test_context_mode_is_explicit_reproducible_and_legacy_sha_unchanged(self):
        frames = regular()
        rows = [action("b", 3.5, 5), action("a", 1, 3)]
        legacy = select_direction_windows(frames, rows)
        self.assertEqual(legacy, select_direction_windows(frames, rows, reserve_context=False))
        self.assertEqual(legacy["policy_sha256"],
                         "adf7ed909f85e6e53ced6137fafca08eee1e21f3a15d8d00b645940e167098c1")
        current = select_direction_windows(frames, rows, reserve_context=True)
        self.assertNotEqual(current["policy_sha256"], legacy["policy_sha256"])
        self.assertEqual(current, select_direction_windows(frames, list(reversed(rows)), reserve_context=True))
        rows[0].update(truth="remove", video_id=35, contact_evidence_sec=[999])
        self.assertEqual(current, select_direction_windows(frames, rows, reserve_context=True))
        with self.assertRaises(TypeError):
            select_direction_windows(frames, rows, 4, True)
        for invalid in (None, 0, 1, "true", []):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                select_direction_windows(frames, rows, reserve_context=invalid)

    def test_edges_single_frame_and_empty_pool_never_fabricate_context(self):
        window = select_direction_windows(regular(2), [action(start=0, end=1)])["windows"][0]
        self.assertEqual(window["frame_timestamps"], [0, .5, 1, 1.5])
        self.assertEqual(window["context_candidates_dropped"][0]["reason"], "video_edge_or_no_frame_in_context")
        point = select_direction_windows(pool([0, .5, 1]), [action(start=0, end=0)])
        self.assertEqual(point["windows"][0]["frame_timestamps"], [0, .5])
        one = select_direction_windows(pool([0]), [action(start=0, end=0)])
        self.assertEqual(one["windows"], [])
        self.assertEqual(one["dropped_windows"][0]["reason"], "insufficient_actual_frames")
        self.assertEqual(select_direction_windows([], [action()])["dropped_windows"][0]["reason"], "no_frame_pool")
        self.assertEqual(select_direction_windows([], [])["selected_count"], 0)

    def test_context_cannot_exceed_duration_cap_even_with_spare_slots(self):
        result = select_direction_windows(pool([0, .1, 5.9, 6, 6.4]), [action(start=0, end=6)])
        window = result["windows"][0]
        self.assertEqual(window["frame_timestamps"], [0, .1, 5.9, 6])
        self.assertIn("duration_cap", [r["reason"] for r in window["context_candidates_dropped"]])
        self.assertAlmostEqual(result["coverage"]["selected_max_anchor_gap_sec"], 5.8)

    def test_huge_finite_pts_gap_does_not_enumerate_empty_time_intervals(self):
        result = select_direction_windows(pool([0, 1e12]), [action(start=0, end=1e12)])
        self.assertEqual(result["windows"], [])
        self.assertGreater(result["coverage"]["empty_split_windows_skipped"], 1e10)
        self.assertEqual(result["coverage"]["actual_max_frame_gap_sec"], 1e12)
        self.assertTrue(result["coverage"]["blind_shorter_actions"])

    def test_cap_visits_different_episodes_before_more_chunks_of_long_episode(self):
        rows = [action("long", 0, 18), action("later", 25, 27), action("last", 35, 37)]
        result = select_direction_windows(regular(40), rows, max_windows=3)
        self.assertEqual([w["source_window_ids"] for w in result["windows"]], [["long"], ["later"], ["last"]])
        self.assertEqual(len({w["episode_id"] for w in result["windows"]}), 3)
        self.assertTrue(result["capped"])
        self.assertEqual(result["coverage"]["partially_cap_dropped_source_window_ids"], ["long"])
        self.assertTrue(all(w["end_sec"]-w["start_sec"] <= 6 for w in result["windows"]))
        zero = select_direction_windows(regular(40), rows, max_windows=0)
        self.assertEqual(zero["windows"], [])
        self.assertEqual(zero["coverage"]["fully_cap_dropped_source_window_ids"], ["last", "later", "long"])

    def test_duplicate_checks_do_not_consume_query_slots(self):
        repeated = select_direction_windows(regular(), [action(), action()])
        self.assertEqual(repeated["selected_count"], 1)
        self.assertEqual(repeated["coverage"]["drop_counts_by_reason"]["duplicate_window_id"], 1)
        aliases = select_direction_windows(regular(), [action("b"), action("a")])
        self.assertEqual(aliases["selected_count"], 1)
        self.assertEqual(aliases["windows"][0]["source_window_ids"], ["a", "b"])
        self.assertEqual(aliases["coverage"]["drop_counts_by_reason"]["duplicate_action_window"], 1)

    def test_conflicting_duplicate_id_rejects_both_instead_of_using_input_order(self):
        for other in (action(start=4, end=5), action(status="partial")):
            rows = [action(), other]
            for ordered in (rows, list(reversed(rows))):
                result = select_direction_windows(regular(), ordered)
                self.assertEqual(result["windows"], [])
                self.assertEqual(result["coverage"]["drop_counts_by_reason"]["conflicting_window_id"], 2)

    def test_inputs_results_and_policy_are_reproducible_without_label_influence(self):
        frames = regular()
        rows = [action("b", 3.5, 5), action("a", 1, 3)]
        snapshot = deepcopy(rows)
        original_times = [f.timestamp_sec for f in frames]
        first = select_direction_windows(frames, rows)
        self.assertEqual(first, select_direction_windows(frames, list(reversed(rows))))
        rows[0].update(video_id=999, truth="remove", rationale="pretend this is loading")
        self.assertEqual(first, select_direction_windows(frames, rows))
        self.assertEqual([f.timestamp_sec for f in frames], original_times)
        self.assertEqual([f.jpeg for f in frames], [b"not-an-image"]*len(frames))
        first["windows"][0]["object_bbox_normalized"][0] = 999
        first["windows"][0]["source_window_ids"].append("invented")
        self.assertEqual(rows[0]["object_bbox_normalized"], snapshot[0]["object_bbox_normalized"])
        clean = select_direction_windows(frames, snapshot)
        self.assertNotEqual(clean["policy_sha256"], select_direction_windows(frames, snapshot, 3)["policy_sha256"])
        self.assertEqual(len(clean["policy_sha256"]), 64)

    def test_strict_parameters_timestamps_and_container_validation(self):
        for cap in (-1, True, 1.0, math.inf, "4"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select_direction_windows(regular(), [], cap)
        for times in ([0, 0], [1, 0], [0, math.nan], [0, math.inf], [-1, 0], [False, .5], ["0", .5]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                select_direction_windows(pool(times), [])
        for frames, checks in ((None, []), ([], None), ([], [{} , "bad"])):
            with self.subTest(frames=frames, checks=checks), self.assertRaises(TypeError):
                select_direction_windows(frames, checks)


if __name__ == "__main__":
    unittest.main()
