"""Short-query coverage and failure cases; no dataset or model required."""

from copy import deepcopy
from types import SimpleNamespace
import math
import unittest

from analyzer.short_action_windows import select_short_action_windows


def pool(times=None):
    # Invalid JPEG proves selection cannot require decoding or model calls.
    return [SimpleNamespace(timestamp_sec=t, jpeg=b"not-an-image")
            for t in (times if times is not None else [i/2 for i in range(25)])]


def source(name="a", times=(0, 1.5, 3), state="uncertain", roi="desk", **overrides):
    return {"window_id": name, "status": "ok", "errors": [], "region": roi,
            "object_visibility": "yes", "object_bbox_normalized": [100, 100, 300, 300],
            "result": {"interaction" if roi == "desk" else "cash_near_cabinet": state},
            "start_sec": times[0], "end_sec": times[-1],
            "evidence": [{"timestamp_sec": t} for t in times], **overrides}


class ShortActionWindowTests(unittest.TestCase):
    def test_sparse_three_frame_query_gets_real_omitted_frames(self):
        frames = pool()
        result = select_short_action_windows(frames, [source()], max_windows=1)
        window = result["windows"][0]
        self.assertEqual(window["frame_timestamps"], [0, .5, 1])
        self.assertEqual(window["new_frame_timestamps"], [.5, 1])
        self.assertEqual(window["frame_timestamps"], [frames[i].timestamp_sec for i in window["frame_indices"]])
        self.assertTrue(result["capped"])
        self.assertEqual(result["coverage"]["selected_new_frame_references"], 2)
        self.assertEqual(result["coverage"]["remaining_eligible_frame_references"], 2)
        self.assertNotIn("result", window)
        self.assertNotIn("verdict", result)

    def test_nonuniform_pts_are_never_replaced_by_synthetic_times(self):
        frames = pool([0, .13, .48, 1.02, 1.96, 2.03, 2.49])
        result = select_short_action_windows(frames, [source(times=(0, 1.02, 2.49))])
        self.assertTrue(result["windows"])
        for window in result["windows"]:
            self.assertEqual(window["frame_timestamps"], [frames[i].timestamp_sec for i in window["frame_indices"]])
            self.assertLessEqual(window["end_sec"]-window["start_sec"], 1.5)

    def test_uncertain_then_negative_then_positive_priority_without_truth_changes(self):
        rows = [source("yes", (0, 1.5, 3), "yes"), source("no", (4, 5.5, 7), "no"),
                source("uncertain", (8, 9.5, 11))]
        selected = select_short_action_windows(pool(), rows, 1)["windows"]
        self.assertEqual(selected[0]["source_window_ids"], ["uncertain"])
        self.assertEqual(selected[0]["source_priority"], "uncertain")
        remaining = select_short_action_windows(pool(), rows[:2], 1)["windows"]
        self.assertEqual(remaining[0]["source_window_ids"], ["no"])
        self.assertEqual(rows[0]["result"], {"interaction": "yes"})

    def test_distinct_regions_and_episodes_precede_second_query(self):
        rows = [source("d1"), source("c", roi="cabinet"), source("d2", (6, 7.5, 9))]
        windows = select_short_action_windows(pool(), rows, 3)["windows"]
        self.assertEqual([w["source_window_ids"] for w in windows], [["d1"], ["c"], ["d2"]])
        self.assertEqual(len({w["episode_id"] for w in windows}), 3)

    def test_neighboring_source_anchors_count_as_already_shown(self):
        rows = [source("a"), source("b", (.5, 1, 2)), source("c", (1, 2, 2.5))]
        result = select_short_action_windows(pool(), rows)
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["coverage"]["unshown_inside_frame_references"], 0)

    def test_another_object_or_roi_does_not_hide_omitted_crop_frames(self):
        rows = [source(), source("b", (.5, 1, 2), object_bbox_normalized=[600, 100, 900, 300]),
                source("c", (1, 2, 2.5), roi="cabinet")]
        result = select_short_action_windows(pool(), rows)
        self.assertTrue(any(w["source_window_ids"] == ["a"] for w in result["windows"]))
        self.assertEqual(result["coverage"]["temporal_episode_count"], 3)

    def test_pairwise_geometry_agreement_prevents_drift(self):
        rows = [source("a", object_bbox_normalized=[0, 0, 300, 300]),
                source("b", object_bbox_normalized=[75, 0, 375, 300]),
                source("c", object_bbox_normalized=[150, 0, 450, 300])]
        self.assertEqual(select_short_action_windows(pool(), rows)["coverage"]["temporal_episode_count"], 2)

    def test_each_extra_window_adds_new_coverage_and_does_not_requery_same_triple(self):
        result = select_short_action_windows(pool(), [source()], 20)
        seen, triples = set(), set()
        for window in result["windows"]:
            keys = {(window["episode_id"], i) for i in window["new_frame_indices"]}
            self.assertTrue(keys-seen)
            seen.update(keys)
            triple = tuple(window["frame_indices"])
            self.assertNotIn(triple, triples)
            triples.add(triple)
        self.assertEqual(result["coverage"]["remaining_eligible_frame_references"], 0)
        self.assertFalse(result["capped"])

    def test_candidate_stays_inside_source_and_span_cap_breaks_large_gaps(self):
        frames = pool([0, .1, 1, 4, 10])
        rows = [source(times=(0, 1, 10))]
        result = select_short_action_windows(frames, rows)
        self.assertEqual([w["frame_timestamps"] for w in result["windows"]], [[0, .1, 1]])
        self.assertEqual(result["coverage"]["unshown_inside_frame_references"], 2)
        self.assertEqual(result["coverage"]["eligible_new_frame_references"], 1)

    def test_empty_and_dense_already_shown_inputs_create_no_proposals(self):
        self.assertEqual(select_short_action_windows([], [])["windows"], [])
        self.assertEqual(select_short_action_windows(pool(), [])["windows"], [])
        self.assertEqual(select_short_action_windows(pool(), [source(times=(0, .5, 1))])["windows"], [])

    def test_zero_cap_records_dropped_queries(self):
        result = select_short_action_windows(pool(), [source()], 0)
        self.assertEqual(result["windows"], [])
        self.assertTrue(result["capped"])
        self.assertGreater(result["coverage"]["drop_counts_by_reason"]["short_action_window_cap"], 0)

    def test_failed_partial_or_hidden_source_is_not_reinterpreted(self):
        cases = [{"status": "partial"}, {"status": "error"}, {"errors": [{}]}, {"errors": "timeout"},
                 {"object_visibility": "uncertain"}, {"object_visibility": "no"},
                 {"result": {"interaction": []}}, {"result": {}}, {"region": "elsewhere"}]
        for override in cases:
            with self.subTest(override=override):
                result = select_short_action_windows(pool(), [source(**override)])
                self.assertEqual(result["windows"], [])
                self.assertTrue(result["dropped_windows"])

    def test_invalid_boxes_cannot_fall_back_to_full_frame(self):
        for box in (None, [], [0, 0, 1001, 1000], [100, 100, 50, 200], [0, 0, True, 300],
                    [0, 0, 0, 300], [0.0, 0, 100, 300]):
            with self.subTest(box=box):
                result = select_short_action_windows(pool(), [source(object_bbox_normalized=box)])
                self.assertEqual(result["windows"], [])

    def test_unavailable_evidence_and_invalid_intervals_are_rejected(self):
        cases = [{"evidence": []}, {"evidence": [{"timestamp_sec": t} for t in (0, .7, 3)]},
                 {"evidence": [{"timestamp_sec": t} for t in (0, 0, 3)]},
                 {"evidence": [{"timestamp_sec": t} for t in (0, math.nan, 3)]},
                 {"start_sec": .5}, {"end_sec": 4}, {"window_id": ""}]
        for override in cases:
            with self.subTest(override=override):
                self.assertEqual(select_short_action_windows(pool(), [source(**override)])["windows"], [])
        self.assertEqual(select_short_action_windows([], [source()])["windows"], [])

    def test_duplicate_source_does_not_duplicate_queries_and_conflict_is_rejected(self):
        one = select_short_action_windows(pool(), [source()])
        duplicate = select_short_action_windows(pool(), [source(), source()])
        self.assertEqual(one["windows"], duplicate["windows"])
        conflicting = select_short_action_windows(pool(), [source(), source(state="yes")])
        self.assertEqual(conflicting["windows"], [])
        self.assertEqual(conflicting["dropped_windows"][0]["reason"], "conflicting_window_id")

    def test_deterministic_permutation_and_no_input_mutation(self):
        rows = [source("a"), source("b", (1, 2.5, 4))]
        original = deepcopy(rows)
        first = select_short_action_windows(pool(), rows)
        self.assertEqual(first, select_short_action_windows(pool(), rows[::-1]))
        self.assertEqual(rows, original)
        first["windows"][0]["object_bbox_normalized"][0] = 99
        self.assertEqual(rows, original)

    def test_input_validation(self):
        for cap in (-1, True, 1.0, math.inf):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select_short_action_windows(pool(), [], cap)
        for span in (0, -1, True, math.inf, math.nan):
            with self.subTest(span=span), self.assertRaises(ValueError):
                select_short_action_windows(pool(), [], max_span_sec=span)
        for times in ([0, 0], [1, 0], [0, math.nan], [0, True]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                select_short_action_windows(pool(times), [])
        with self.assertRaises(TypeError):
            select_short_action_windows(pool(), [None])
        with self.assertRaises(TypeError):
            select_short_action_windows(None, [])

    def test_policy_records_scope_and_parameter_changes(self):
        first = select_short_action_windows(pool(), [source()])
        other = select_short_action_windows(pool(), [source()], max_span_sec=2)
        self.assertNotEqual(first["policy_sha256"], other["policy_sha256"])
        self.assertFalse(first["coverage"]["upstream_dropped_windows_recovered"])
        self.assertTrue(first["coverage"]["blind_subsample_actions"])


if __name__ == "__main__":
    unittest.main()
