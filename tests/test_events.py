"""Failure-oriented tests for descriptive events, without images or inference."""

import copy
import math
import unittest

from analyzer.events import build_observed_events


def rows(cash, person=None, machine=None, times=None):
    result = []
    for index, state in enumerate(cash):
        states = {"cash_on_worktop": state, "person_in_work_area": person[index] if person else "yes"}
        if machine is not None:
            states["notes_in_machine"] = machine[index]
        result.append({"frame_id": f"f{index}", "timestamp_sec": times[index] if times else index,
                       "states": states})
    return result


def change(observations, feature="cash_on_worktop", before="no", after="yes", positions=(1, 2, 3)):
    return {"feature": feature, "from_state": before, "to_state": after,
            "interval_start_sec": observations[positions[0]]["timestamp_sec"],
            "interval_end_sec": observations[positions[1]]["timestamp_sec"],
            "confirmed_at_sec": observations[positions[2]]["timestamp_sec"],
            "evidence_frame_ids": [observations[index]["frame_id"] for index in positions]}


class ObservedEventsTests(unittest.TestCase):
    def test_four_visual_changes_preserve_evidence_and_pts(self):
        for feature, before, after, kind in (
            ("cash_on_worktop", "no", "yes", "cash_appeared_on_worktop"),
            ("cash_on_worktop", "yes", "no", "cash_no_longer_visible_on_worktop"),
            ("notes_in_machine", "no", "yes", "notes_appeared_in_machine"),
            ("notes_in_machine", "yes", "no", "notes_no_longer_visible_in_machine"),
        ):
            with self.subTest(kind=kind):
                observations = rows([before, before, after, after], times=[0, 1.25, 2.5, 3.75])
                for row in observations:
                    row["states"][feature] = row["states"]["cash_on_worktop"]
                transition = change(observations, feature, before, after)
                event, = build_observed_events(observations, [transition])
                self.assertEqual(event["type"], kind)
                for key in ("interval_start_sec", "interval_end_sec", "confirmed_at_sec", "evidence_frame_ids"):
                    self.assertEqual(event[key], transition[key])
                self.assertNotIn("verdict", event)
                self.assertTrue(event["description"])
                self.assertTrue(event["title"])

    def test_changes_without_supplied_transitions_are_not_invented(self):
        self.assertEqual(build_observed_events(rows(["no", "no", "yes", "yes"]), []), [])

    def test_simultaneous_changes_do_not_claim_transfer_or_counting(self):
        observations = rows(["yes", "yes", "no", "no"], machine=["no", "no", "yes", "yes"])
        changes = [change(observations, before="yes", after="no"), change(observations, "notes_in_machine")]
        self.assertEqual({event["type"] for event in build_observed_events(observations, changes)},
                         {"cash_no_longer_visible_on_worktop", "notes_appeared_in_machine"})

    def test_single_matching_frame_does_not_confirm_presence_event(self):
        self.assertEqual(build_observed_events(rows(["yes"], ["no"]), []), [])

    def test_one_event_per_run_with_first_confirming_and_last_evidence(self):
        observations = rows(["yes"] * 5, ["no"] * 5, times=[0, 1.5, 3, 4.5, 6])
        event, = build_observed_events(observations, [])
        self.assertEqual(event["type"], "cash_visible_without_person_in_work_area")
        self.assertEqual((event["start_sec"], event["end_sec"], event["confirmed_at_sec"]), (0, 6, 1.5))
        self.assertEqual(event["observed_frame_count"], 5)
        self.assertEqual(event["evidence_frame_ids"], ["f0", "f1", "f4"])

    def test_two_frame_run_does_not_duplicate_confirming_evidence(self):
        event, = build_observed_events(rows(["yes", "yes"], ["no", "no"]), [])
        self.assertEqual(event["evidence_frame_ids"], ["f0", "f1"])

    def test_unknown_missing_and_opposite_states_break_joint_chain(self):
        for feature, value in (("person_in_work_area", "uncertain"), ("cash_on_worktop", "uncertain"),
                               ("person_in_work_area", "yes"), ("cash_on_worktop", "no"),
                               ("person_in_work_area", None), ("cash_on_worktop", None)):
            observations = rows(["yes"] * 5, ["no"] * 5)
            if value is None:
                observations[2]["states"].pop(feature)
            else:
                observations[2]["states"][feature] = value
            with self.subTest(feature=feature, value=value):
                events = build_observed_events(observations, [])
                self.assertEqual([event["evidence_frame_ids"] for event in events], [["f0", "f1"], ["f3", "f4"]])

    def test_large_gap_splits_runs_but_equal_gap_is_allowed(self):
        observations = rows(["yes"] * 4, ["no"] * 4, times=[0, 4, 8.01, 12.01])
        self.assertEqual([event["evidence_frame_ids"] for event in build_observed_events(observations, [])],
                         [["f0", "f1"], ["f2", "f3"]])

    def test_technical_failures_break_joint_chain_despite_definite_values(self):
        for update in (
            {"status": "error"}, {"status": "partial"},
            {"errors": ["request failed"]}, {"errors": [{"message": "request failed"}]},
            {"errors": [{"feature": "all"}]}, {"errors": [{"feature": "*"}]},
            {"errors": [{"feature": "cash_on_worktop"}]},
            {"errors": [{"feature": "person_in_work_area"}]},
        ):
            observations = rows(["yes"] * 3, ["no"] * 3)
            observations[1].update(update)
            with self.subTest(update=update):
                self.assertEqual(build_observed_events(observations, []), [])

    def test_unrelated_field_errors_keep_joint_chain(self):
        for feature in ("cabinet_door", "notes_in_machine", "another_optional_feature"):
            observations = rows(["yes", "yes"], ["no", "no"])
            observations[1].update(status="partial", errors=[{"feature": feature, "message": "no answer"}])
            with self.subTest(feature=feature):
                self.assertEqual(len(build_observed_events(observations, [])), 1)

    def test_transition_requires_usable_support_for_both_states(self):
        for broken_index in range(4):
            observations = rows(["no", "no", "yes", "yes"])
            observations[broken_index].update(status="partial", errors=[{"feature": "cash_on_worktop"}])
            with self.subTest(broken_index=broken_index):
                self.assertEqual(build_observed_events(observations, [change(observations)]), [])

    def test_transition_ignores_errors_in_other_fields(self):
        observations = rows(["no", "no", "yes", "yes"])
        for row in observations:
            row.update(status="partial", errors=[{"feature": "cabinet_door"}, {"feature": "person_in_work_area"}])
        self.assertEqual(len(build_observed_events(observations, [change(observations)])), 1)

    def test_transition_cannot_skip_unknown_or_error_inside_confirmation(self):
        for update in ({"states": {"cash_on_worktop": "uncertain"}},
                       {"errors": [{"feature": "cash_on_worktop"}]}, {"status": "error"}):
            observations = rows(["no", "no", "yes", "yes", "yes"])
            observations[3].update(update)
            with self.subTest(update=update):
                self.assertEqual(build_observed_events(observations, [change(observations, positions=(1, 2, 4))]), [])

    def test_transition_cannot_bridge_gap_in_old_or_new_confirmation(self):
        for times in ([0, 5, 6, 7], [0, 1, 6, 7], [0, 1, 2, 7]):
            observations = rows(["no", "no", "yes", "yes"], times=times)
            with self.subTest(times=times):
                self.assertEqual(build_observed_events(observations, [change(observations)]), [])

    def test_brief_opposite_frame_does_not_destroy_previously_confirmed_old_state(self):
        observations = rows(["no", "no", "yes", "no", "yes", "yes"])
        event, = build_observed_events(observations, [change(observations, positions=(3, 4, 5))])
        self.assertEqual(event["evidence_frame_ids"], ["f3", "f4", "f5"])

    def test_reversed_changes_and_longer_confirmation_are_supported(self):
        observations = rows(["no", "no", "yes", "yes", "yes", "no", "no", "no"])
        changes = [change(observations, before="yes", after="no", positions=(4, 5, 7)),
                   change(observations, positions=(1, 2, 4))]
        events = build_observed_events(observations, changes)
        self.assertEqual([event["confirmed_at_sec"] for event in events], [4, 7])

    def test_duplicate_transitions_do_not_duplicate_events_and_inputs_are_unchanged(self):
        observations = rows(["no", "no", "yes", "yes"])
        changes = [change(observations)] * 2
        before = copy.deepcopy((observations, changes))
        event, = build_observed_events(observations, changes)
        self.assertEqual((observations, changes), before)
        event["evidence_frame_ids"].append("new")
        self.assertEqual((observations, changes), before)

    def test_unrelated_transitions_are_ignored(self):
        self.assertEqual(build_observed_events([], [{"feature": "cabinet_door"}]), [])

    def test_malformed_transition_evidence_and_pts_are_rejected(self):
        for key, value in (
            ("from_state", "uncertain"), ("to_state", "no"), ("feature", []),
            ("interval_start_sec", True), ("interval_end_sec", 2.1), ("confirmed_at_sec", 2),
            ("evidence_frame_ids", ["f1", "f2", "missing"]),
            ("evidence_frame_ids", ["f1", "f2", "f2"]),
            ("evidence_frame_ids", ["f1", "f3", "f2"]),
            ("evidence_frame_ids", "f1,f2,f3"),
        ):
            observations = rows(["no", "no", "yes", "yes"])
            transition = change(observations)
            transition[key] = value
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                build_observed_events(observations, [transition])

    def test_malformed_observations_and_parameters_are_rejected(self):
        for key, value in (("frame_id", " "), ("timestamp_sec", math.nan), ("timestamp_sec", -1),
                           ("timestamp_sec", True), ("states", []), ("errors", "error"), ("status", "done")):
            observations = rows(["yes"])
            observations[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((TypeError, ValueError)):
                build_observed_events(observations, [])
        for feature in ("person_in_work_area", "cash_on_worktop", "notes_in_machine"):
            observations = rows(["yes"])
            observations[0]["states"][feature] = "present"
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                build_observed_events(observations, [])
        for value in (0, -1, True, "4", None, math.inf, math.nan):
            with self.subTest(gap=value), self.assertRaises((TypeError, ValueError)):
                build_observed_events([], [], max_gap_sec=value)
        for observations, changes in ((None, []), ({}, []), ([None], []), ([], None), ([], {}), ([], [None])):
            with self.subTest(observations=observations, changes=changes), self.assertRaises((TypeError, ValueError)):
                build_observed_events(observations, changes)

    def test_duplicate_frame_ids_and_nonincreasing_pts_are_rejected(self):
        for key, value in (("frame_id", "f0"), ("timestamp_sec", 0), ("timestamp_sec", -1)):
            observations = rows(["yes", "yes"], ["no", "no"])
            observations[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_observed_events(observations, [])

    def test_no_default_negative_for_error_only_frames(self):
        observations = [{"frame_id": f"f{index}", "timestamp_sec": index, "status": "error"} for index in range(2)]
        self.assertEqual(build_observed_events(observations, []), [])


if __name__ == "__main__":
    unittest.main()
