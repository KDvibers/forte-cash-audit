"""Tests for temporal confirmation, evidence and uncertainty boundaries."""

import copy
import math
import unittest

from analyzer.timeline import build_transitions


def observations(person_states, *, times=None, door_states=None):
    if times is None:
        times = range(len(person_states))
    result = []
    for index, (timestamp, state) in enumerate(zip(times, person_states)):
        states = {"person_in_work_area": state}
        if door_states is not None:
            states["cabinet_door"] = door_states[index]
        result.append({"frame_id": f"f{index}", "timestamp_sec": timestamp, "states": states})
    return result


class TimelineTests(unittest.TestCase):
    def test_confirmed_transition_has_bracket_and_actual_evidence(self):
        rows = observations(["yes", "yes", "no", "no"], times=[0, 1.25, 2.5, 3.75])
        self.assertEqual(build_transitions(rows), [{
            "feature": "person_in_work_area", "from_state": "yes", "to_state": "no",
            "interval_start_sec": 1.25, "interval_end_sec": 2.5, "confirmed_at_sec": 3.75,
            "evidence_frame_ids": ["f1", "f2", "f3"],
        }])

    def test_both_features_and_reverse_changes_are_independent(self):
        rows = observations(
            ["yes", "yes", "no", "no", "yes", "yes"],
            door_states=["open", "open", "closed", "closed", "closed", "closed"],
        )
        changes = build_transitions(rows)
        self.assertEqual([(item["feature"], item["from_state"], item["to_state"]) for item in changes], [
            ("person_in_work_area", "yes", "no"),
            ("cabinet_door", "open", "closed"),
            ("person_in_work_area", "no", "yes"),
        ])

    def test_initial_state_requires_confirmation(self):
        self.assertEqual(build_transitions(observations(["yes", "no", "no"])), [])

    def test_new_state_requires_confirmation(self):
        self.assertEqual(build_transitions(observations(["yes", "yes", "no"])), [])

    def test_consecutive_confirmation_and_noise_cancellation(self):
        rows = observations(["yes", "yes", "no", "yes", "no", "no"])
        changes = build_transitions(rows)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["evidence_frame_ids"], ["f3", "f4", "f5"])

    def test_larger_confirmation_count(self):
        rows = observations(["yes", "yes", "yes", "no", "no", "no"])
        self.assertEqual(build_transitions(rows[:-1], confirmation_frames=3), [])
        change = build_transitions(rows, confirmation_frames=3)[0]
        self.assertEqual(change["evidence_frame_ids"], ["f2", "f3", "f5"])

    def test_uncertain_breaks_old_state_and_pending_confirmation(self):
        for states in (["yes", "yes", "uncertain", "no", "no"],
                       ["yes", "yes", "no", "uncertain", "no", "no"]):
            with self.subTest(states=states):
                self.assertEqual(build_transitions(observations(states)), [])

    def test_missing_field_and_not_visible_break_only_that_feature(self):
        rows = observations(
            ["yes", "yes", "no", "no"], door_states=["open", "open", "not_visible", "closed"]
        )
        self.assertEqual([item["feature"] for item in build_transitions(rows)], ["person_in_work_area"])
        rows[2]["states"].pop("cabinet_door")
        self.assertEqual([item["feature"] for item in build_transitions(rows)], ["person_in_work_area"])

    def test_state_is_reconfirmed_after_unknown(self):
        rows = observations(["yes", "yes", "uncertain", "no", "no", "yes", "yes"])
        change = build_transitions(rows)[0]
        self.assertEqual(change["from_state"], "no")
        self.assertEqual(change["evidence_frame_ids"], ["f4", "f5", "f6"])

    def test_gap_above_limit_breaks_chain_including_pending_confirmation(self):
        for times in ([0, 1, 6, 7], [0, 1, 2, 7]):
            with self.subTest(times=times):
                self.assertEqual(build_transitions(observations(["yes", "yes", "no", "no"], times=times)), [])

    def test_gap_equal_to_limit_is_allowed(self):
        rows = observations(["yes", "yes", "no", "no"], times=[0, 4, 8, 12])
        self.assertEqual(len(build_transitions(rows)), 1)

    def test_error_status_breaks_chain_even_when_states_look_valid(self):
        rows = observations(["yes", "yes", "no", "no", "no"])
        rows[2].update(status="error", errors=["model request failed"])
        self.assertEqual(build_transitions(rows), [])

    def test_partial_keeps_usable_feature_without_interpreting_error_text(self):
        rows = observations(
            ["yes", "yes", "no", "no"], door_states=["open", "open", "uncertain", "closed"]
        )
        rows[2].update(status="partial", errors=[{"feature": "cabinet_door", "message": "no valid answer"}])
        self.assertEqual([item["feature"] for item in build_transitions(rows)], ["person_in_work_area"])

    def test_error_only_observation_has_no_default_negative_state(self):
        rows = observations(["yes", "yes", "no", "no", "no"])
        rows[2] = {"frame_id": "error", "timestamp_sec": 2, "status": "error", "errors": ["no response"]}
        self.assertEqual(build_transitions(rows), [])

    def test_unknown_features_are_ignored_and_input_is_not_modified(self):
        rows = observations(["yes", "yes", "no", "no"])
        rows[0]["states"]["not_a_feature"] = {"arbitrary": "metadata"}
        original = copy.deepcopy(rows)
        self.assertEqual(len(build_transitions(rows)), 1)
        self.assertEqual(rows, original)

    def test_empty_and_unchanged_series(self):
        self.assertEqual(build_transitions([]), [])
        self.assertEqual(build_transitions(observations(["yes"] * 10)), [])

    def test_rejects_decreasing_or_equal_timestamps(self):
        for times in ([1, 0], [1, 1]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                build_transitions(observations(["yes", "yes"], times=times))

    def test_rejects_invalid_timestamp_types_and_values(self):
        for value in (None, "1", True, [], -1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                build_transitions(observations(["yes"], times=[value]))

    def test_rejects_invalid_configuration(self):
        for value in (None, True, "4", 0, -1, math.nan, math.inf):
            with self.subTest(max_gap=value), self.assertRaises((ValueError, TypeError)):
                build_transitions([], max_gap_sec=value)
        for value in (None, True, "2", 2.0, 0, 1, -1):
            with self.subTest(confirmation=value), self.assertRaises((ValueError, TypeError)):
                build_transitions([], confirmation_frames=value)

    def test_rejects_invalid_input_structure(self):
        for value in (None, {}, (), [None], ["frame"]):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                build_transitions(value)
        for key, value in (("frame_id", " "), ("frame_id", 1), ("states", []),
                           ("status", "failed"), ("status", None), ("errors", "error")):
            rows = observations(["yes"])
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                build_transitions(rows)

    def test_rejects_unknown_values_of_known_features(self):
        for feature, value in (("person_in_work_area", "present"), ("person_in_work_area", False),
                               ("cabinet_door", "no"), ("cabinet_door", [])):
            rows = observations(["yes"])
            rows[0]["states"][feature] = value
            with self.subTest(feature=feature, value=value), self.assertRaises(ValueError):
                build_transitions(rows)

    def test_rejects_duplicate_frame_ids(self):
        rows = observations(["yes", "yes"])
        rows[1]["frame_id"] = rows[0]["frame_id"]
        with self.assertRaises(ValueError):
            build_transitions(rows)


class CashTimelineTests(unittest.TestCase):
    def test_cash_locations_form_independent_confirmed_state_changes(self):
        snapshots = [
            ("yes", "no", "no"), ("yes", "no", "no"),
            ("no", "yes", "no"), ("no", "yes", "no"),
            ("no", "no", "yes"), ("no", "no", "yes"),
            ("yes", "no", "no"), ("yes", "no", "no"),
        ]
        rows = [
            {"frame_id": f"cash{index}", "timestamp_sec": index / 2,
             "states": dict(zip(("cash_on_worktop", "cash_in_hands", "notes_in_machine"), states))}
            for index, states in enumerate(snapshots)
        ]
        changes = build_transitions(rows)
        self.assertEqual({(row["feature"], row["from_state"], row["to_state"], row["confirmed_at_sec"])
                          for row in changes}, {
            ("cash_on_worktop", "yes", "no", 1.5), ("cash_in_hands", "no", "yes", 1.5),
            ("cash_in_hands", "yes", "no", 2.5), ("notes_in_machine", "no", "yes", 2.5),
            ("cash_on_worktop", "no", "yes", 3.5), ("notes_in_machine", "yes", "no", 3.5),
        })
        self.assertEqual(len(changes), 6)
        first_change = next(row for row in changes if row["feature"] == "cash_on_worktop")
        self.assertEqual(first_change["evidence_frame_ids"], ["cash1", "cash2", "cash3"])
        self.assertNotIn("action", first_change)
        self.assertNotIn("verdict", first_change)

    def test_cash_fields_appearing_late_do_not_invent_an_initial_no(self):
        rows = observations(["yes", "yes", "no", "no"])
        rows[2]["states"]["cash_on_worktop"] = "yes"
        rows[3]["states"]["cash_on_worktop"] = "yes"
        self.assertEqual([row["feature"] for row in build_transitions(rows)], ["person_in_work_area"])

    def test_missing_cash_sample_breaks_chain_and_requires_reconfirmation(self):
        states = ["yes", "yes", None, "no", "no", "yes", "yes"]
        rows = [{"frame_id": f"f{index}", "timestamp_sec": index,
                 "states": {} if state is None else {"cash_on_worktop": state}}
                for index, state in enumerate(states)]
        changes = build_transitions(rows)
        self.assertEqual(len(changes), 1)
        self.assertEqual((changes[0]["from_state"], changes[0]["to_state"]), ("no", "yes"))
        self.assertEqual(changes[0]["evidence_frame_ids"], ["f4", "f5", "f6"])

    def test_partial_cash_failure_does_not_erase_other_cash_or_original_features(self):
        rows = observations(["yes", "yes", "no", "no"],
                            door_states=["open", "open", "closed", "closed"])
        for index, row in enumerate(rows):
            row["states"].update({"cash_on_worktop": "yes" if index < 2 else "no",
                                  "cash_in_hands": "no" if index < 2 else "yes",
                                  "notes_in_machine": "no" if index < 2 else "yes"})
        rows[2]["states"]["cash_on_worktop"] = "uncertain"
        rows[2].update(status="partial", errors=[{"feature": "cash_on_worktop", "message": "timeout"}])
        changes = build_transitions(rows)
        self.assertEqual({row["feature"] for row in changes},
                         {"person_in_work_area", "cabinet_door", "cash_in_hands", "notes_in_machine"})

    def test_global_error_breaks_all_cash_chains_even_with_known_looking_states(self):
        rows = [{"frame_id": f"f{index}", "timestamp_sec": index,
                 "states": {feature: "yes" if index < 2 else "no"
                            for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine")}}
                for index in range(5)]
        rows[2].update(status="error", errors=["request failed"])
        self.assertEqual(build_transitions(rows), [])

    def test_cash_cannot_use_cabinet_specific_or_coerced_states(self):
        for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine"):
            for value in ("open", "not_visible", False, None, ["yes"]):
                row = {"frame_id": "f0", "timestamp_sec": 0, "states": {feature: value}}
                with self.subTest(feature=feature, value=value), self.assertRaises(ValueError):
                    build_transitions([row])


if __name__ == "__main__":
    unittest.main()
