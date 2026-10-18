"""Bounded condition-specific frame selection; no images, models or GPU."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyzer.refinement import refinement_indices, rule_refinement_indices
from tests import test_cash_short_circuit as fixture

PRESENT, DESK = fixture.PRESENT, fixture.DESK


def frames(end):
    return [SimpleNamespace(timestamp_sec=i / 2) for i in range(int(end * 2) + 1)]


def row(time, person, cash, **extra):
    return {"frame_id": f"f{time}", "timestamp_sec": time,
            "states": {"person_in_work_area": person, "cash_on_worktop": cash}, **extra}


def times(source, result, key="indices"):
    return [source[index].timestamp_sec for index in result[key]]


class RuleRefinementTests(unittest.TestCase):
    def test_equal_false_conditions_do_not_spend_cap_on_planned_cash_unknown(self):
        source = frames(6)
        rows = [row(0, "yes", "uncertain"), row(2, "yes", "yes"),
                row(4, "no", "no"), row(6, "uncertain", "no")]
        self.assertTrue(refinement_indices(source, rows)["indices"])
        result = rule_refinement_indices(source, rows)
        self.assertEqual(result, {"indices": [], "selection_order": [], "candidate_count": 0, "capped": False})

    def test_known_boundary_has_priority_over_earlier_visual_uncertainty(self):
        source = frames(14)
        rows = [row(0, "yes", "uncertain"), row(2, "uncertain", "yes"), row(4, "uncertain", "yes"),
                row(10, "no", "no"), row(12, "no", "yes"), row(14, "no", "yes")]
        one = rule_refinement_indices(source, rows, max_new_frames=1)
        self.assertEqual(times(source, one), [11])
        self.assertEqual(times(source, one, "selection_order"), [11])
        full = rule_refinement_indices(source, rows, max_new_frames=100)
        self.assertEqual(times(source, full, "selection_order")[:5], [11, 10.5, 11.5, 9.5, 12.5])
        self.assertEqual(times(source, full, "selection_order")[5], 1)
        self.assertEqual(full["indices"], sorted(set(full["selection_order"])))
        self.assertFalse(full["capped"])

    def test_round_robin_spreads_cap_across_separate_known_boundaries(self):
        source = frames(14)
        rows = [row(0, "yes", "uncertain"), row(2, "no", "yes"), row(4, "no", "yes"),
                row(10, "no", "no"), row(12, "no", "yes"), row(14, "no", "yes")]
        result = rule_refinement_indices(source, rows, max_new_frames=4)
        self.assertEqual(times(source, result, "selection_order"), [1, 11, .5, 10.5])
        self.assertEqual(times(source, result), [.5, 1, 10.5, 11])
        self.assertTrue(result["capped"])

    def test_true_to_false_and_false_to_true_are_both_boundaries(self):
        source = frames(4)
        rows = [row(0, "yes", "uncertain"), row(2, "no", "yes"), row(4, "no", "no")]
        result = rule_refinement_indices(source, rows, max_new_frames=2)
        self.assertEqual(times(source, result, "selection_order"), [1, 3])

    def test_unknown_known_can_refine_but_same_unknown_or_same_true_cannot(self):
        source = frames(2)
        for rows in ([row(0, "uncertain", "yes"), row(2, "no", "yes")],
                     [row(0, "no", "uncertain"), row(2, "no", "no")]):
            with self.subTest(rows=rows):
                self.assertEqual(times(source, rule_refinement_indices(source, rows, 1)), [1])
        for rows in ([row(0, "uncertain", "yes"), row(2, "no", "uncertain")],
                     [row(0, "no", "yes"), row(2, "no", "yes")]):
            with self.subTest(rows=rows):
                self.assertEqual(rule_refinement_indices(source, rows)["indices"], [])

    def test_unrelated_cabinet_change_cannot_trigger_current_rule_refinement(self):
        source = frames(2)
        rows = [row(0, "yes", "uncertain"), row(2, "yes", "uncertain")]
        rows[0]["states"]["cabinet_door"] = "closed"
        rows[1]["states"]["cabinet_door"] = "open"
        self.assertTrue(refinement_indices(source, rows)["indices"])
        self.assertEqual(rule_refinement_indices(source, rows)["indices"], [])

    def test_relevant_errors_cannot_create_a_known_boundary_or_be_reobserved(self):
        source = frames(12)
        # The first apparent true has a failed person head: unknown/false is
        # secondary. The later valid true/false pair must win the one-frame cap.
        rows = [row(0, "no", "yes", status="partial", errors=[{"feature": "person_in_work_area"}]),
                row(2, "no", "no"), row(10, "no", "yes"), row(12, "no", "no")]
        result = rule_refinement_indices(source, rows, 1)
        self.assertEqual(times(source, result), [11])
        complete = rule_refinement_indices(source, rows, 100)
        self.assertTrue(all(t not in (0, 2, 10, 12) for t in times(source, complete)))

    def test_known_false_operand_remains_false_when_other_operand_has_error(self):
        for bad_feature in ("cash_on_worktop", "person_in_work_area"):
            if bad_feature == "cash_on_worktop":
                a = row(0, "yes", "yes", status="partial", errors=[{"feature": bad_feature}])
            else:
                a = row(0, "no", "no", status="partial", errors=[{"feature": bad_feature}])
            with self.subTest(feature=bad_feature):
                self.assertEqual(rule_refinement_indices(frames(2), [a, row(2, "yes", "uncertain")])["indices"], [])

    def test_unknown_error_scope_blocks_both_operands_instead_of_using_raw_states(self):
        for errors in ([{"feature": "person_typo"}], ["transport error"], [{}]):
            rows = [row(0, "no", "yes", status="partial", errors=errors),
                    row(2, "no", "no", status="partial", errors=errors)]
            with self.subTest(errors=errors):
                self.assertEqual(rule_refinement_indices(frames(2), rows)["indices"], [])
        rows = [row(0, "no", "yes", status="error", errors=[]),
                row(2, "no", "no", status="partial", errors=[])]
        self.assertEqual(rule_refinement_indices(frames(2), rows)["indices"], [])

    def test_unrelated_scoped_error_does_not_hide_valid_condition_boundary(self):
        rows = [row(0, "yes", "uncertain"), row(2, "no", "yes", status="partial",
                                                     errors=[{"feature": "cabinet_door"}])]
        self.assertEqual(times(frames(2), rule_refinement_indices(frames(2), rows, 1)), [1])

    def test_large_gap_is_not_bridged_but_exact_four_second_gap_is_eligible(self):
        for gap in (4, 4.0001):
            result = rule_refinement_indices(frames(5), [row(0, "yes", "uncertain"), row(gap, "no", "yes")])
            self.assertEqual(bool(result["indices"]), gap == 4)

    def test_actual_fractional_pts_and_first_duplicate_source_index_are_preserved(self):
        source = [SimpleNamespace(timestamp_sec=t) for t in (.033, .033, .51, .51, 1.017, 1.533, 2.033)]
        rows = [row(.0330004, "yes", "uncertain"), row(2.0330004, "no", "yes")]
        result = rule_refinement_indices(source, rows, 100)
        self.assertEqual(result["indices"], [2, 4, 5])
        self.assertEqual(times(source, result), [.51, 1.017, 1.533])
        self.assertEqual(result["candidate_count"], 3)

    def test_zero_cap_still_reports_unselected_candidates_and_never_mutates_inputs(self):
        source = frames(2)
        rows = [row(0, "yes", "uncertain"), row(2, "no", "yes")]
        before = deepcopy(rows)
        result = rule_refinement_indices(source, rows, 0)
        self.assertEqual(result, {"indices": [], "selection_order": [], "candidate_count": 3, "capped": True})
        self.assertEqual(rows, before)
        self.assertEqual(rule_refinement_indices([], [])["selection_order"], [])

    def test_invalid_input_cannot_silently_become_an_empty_plan(self):
        for cap in (True, -1, 1.0, "8"):
            with self.subTest(cap=cap), self.assertRaises((ValueError, TypeError)):
                rule_refinement_indices([], [], cap)
        rows = [row(0, "yes", "uncertain"), row(0, "no", "yes")]
        with self.assertRaises(ValueError):
            rule_refinement_indices(frames(2), rows)


class RuleRefinementOrchestrationTests(unittest.TestCase):
    # Reuse setup helpers without rediscovering the other module's test cases.
    setUp = fixture.ObserveShortCircuitTests.setUp
    run_video = fixture.ObserveShortCircuitTests.run_video

    def test_scoped_selector_is_used_when_flag_enabled_and_generic_is_not_called(self):
        with patch("analyzer.observe.rule_refinement_indices", wraps=rule_refinement_indices) as scoped, \
             patch("analyzer.observe.refinement_indices", side_effect=AssertionError("wrong refinement scope")):
            report = self.run_video(max_refinement_frames=8)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(scoped.call_count, 1)
        self.assertEqual(scoped.call_args.kwargs["max_new_frames"], 8)
        self.assertEqual(report["settings"]["refinement_policy"], "absence_and_cash_condition_v1")
        self.assertEqual(report["refinement"]["indices"], [])

    def test_generic_selector_is_preserved_when_flag_disabled(self):
        self.primary.responses = [{"cash_on_worktop": "no"}] * 2
        self.activity.responses = [*PRESENT, DESK, {"resting_on_desk": "no"}] * 2
        with patch("analyzer.observe.rule_refinement_indices", side_effect=AssertionError("unexpected scoped selector")), \
             patch("analyzer.observe.refinement_indices", wraps=refinement_indices) as generic:
            report = self.run_video(skip_cash_when_person_present=False, max_refinement_frames=8)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(generic.call_count, 1)
        self.assertNotIn("refinement_policy", report["settings"])

    def test_observer_executes_priority_order_but_report_keeps_chronological_observations(self):
        self.sample.return_value["frames"] = [fixture.frame(t) for t in (0, .5, 1, 1.5, 2)]
        self.sample.return_value["max_sample_gap_sec"] = .5
        self.activity.responses = PRESENT * 4
        plan = {"indices": [1, 3], "selection_order": [3, 1], "candidate_count": 2, "capped": False}
        with patch("analyzer.observe.rule_refinement_indices", return_value=plan):
            report = self.run_video(max_refinement_frames=2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual([call["timestamp"] for call in self.activity.calls[::2]], [0, 2, 1.5, .5])
        self.assertEqual([item["timestamp_sec"] for item in report["observations"]], [0, .5, 1.5, 2])
        self.assertEqual(report["refinement"]["selection_order"], [3, 1])


if __name__ == "__main__":
    unittest.main()
