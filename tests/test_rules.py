"""Visual-rule semantics on synthetic observations, without images or models."""

from copy import deepcopy
import json
import unittest

from analyzer.rules import evaluate_rules


def observation(index, person, cash, *, timestamp=None, status="ok", errors=None):
    states = {"cabinet_door": "uncertain", "cash_in_hands": "uncertain", "notes_in_machine": "uncertain"}
    if person is not None:
        states["person_in_work_area"] = person
    if cash is not None:
        states["cash_on_worktop"] = cash
    return {"frame_id": f"f{index}", "timestamp_sec": index if timestamp is None else timestamp,
            "states": states, "status": status, "errors": [] if errors is None else errors}


def report(rows, **changes):
    value = {"observations": rows, "status": "completed", "errors": [], "budget_exceeded": False,
             "selected_frame_count": len(rows), "processed_frame_count": len(rows),
             "all_selected_frames_processed": True, "selected_timestamps": [r["timestamp_sec"] for r in rows]}
    value.update(changes)
    return value


def presence():
    return [observation(0, "yes", "uncertain"), observation(1, "yes", "uncertain")]


class RuleTests(unittest.TestCase):
    def test_confirmed_history_and_two_matching_frames_make_review_candidate(self):
        value = evaluate_rules(report(presence() + [observation(2, "no", "yes"), observation(3, "no", "yes")]))
        self.assertEqual(len(value["candidates"]), 1)
        self.assertEqual(value["blocked_candidates"], [])
        event = value["candidates"][0]
        self.assertEqual(event["type"], "cash_visible_after_person_disappears")
        self.assertEqual(event["status"], "candidate")
        self.assertEqual((event["start_sec"], event["end_sec"], event["timecode_sec"], event["confirmed_at_sec"]), (2, 3, 3, 3))
        self.assertEqual(event["timecode_semantics"], "candidate_confirmation_not_departure")
        self.assertEqual(value["policy"]["revision"], 2)
        self.assertEqual(value["policy"]["timecode_semantics"], event["timecode_semantics"])
        self.assertEqual(event["evidence_ids"], ["f1", "f2", "f3"])
        self.assertEqual(event["prior_presence_frame_ids"], ["f0", "f1"])
        self.assertFalse(value["can_conclude_no_candidate"])
        self.assertNotIn('"verdict":', json.dumps(value))
        self.assertNotIn('"violations":', json.dumps(value))

    def test_initial_absence_with_cash_is_blocked_not_an_invented_departure(self):
        value = evaluate_rules(report([observation(0, "no", "yes"), observation(1, "no", "yes")]))
        self.assertEqual(value["candidates"], [])
        event = value["blocked_candidates"][0]
        self.assertEqual(event["reason"], "missing_prior_confirmed_presence")
        self.assertIsNone(event["last_present_id"])
        self.assertEqual(event["evidence_ids"], ["f0", "f1"])
        self.assertFalse(value["can_conclude_no_candidate"])

    def test_single_presence_frame_does_not_establish_confirmed_history(self):
        value = evaluate_rules(report([observation(0, "yes", "no"), observation(1, "no", "yes"),
                                       observation(2, "no", "yes")]))
        self.assertFalse(value["candidates"])
        self.assertEqual(value["blocked_candidates"][0]["reason"], "missing_prior_confirmed_presence")

    def test_unknown_person_between_presence_and_candidate_is_recorded_not_bridged(self):
        rows = presence() + [observation(2, "uncertain", "yes"), observation(3, "no", "yes"), observation(4, "no", "yes")]
        value = evaluate_rules(report(rows))
        event = value["candidates"][0]
        self.assertEqual(event["presence_context_unknown_frame_ids"], ["f2"])
        self.assertEqual(event["matching_frame_ids"], ["f3", "f4"])
        self.assertEqual(event["start_sec"], 3)
        self.assertEqual(event["timecode_sec"], 4)
        self.assertEqual(event["evidence_ids"], ["f1", "f3", "f4"])
        self.assertEqual(value["coverage"]["unknown_frame_ids"], ["f2"])
        self.assertEqual(value["policy"]["presence_context"], "historical_presence_not_tracked_exit")

    def test_unknown_breaks_new_presence_confirmation(self):
        rows = [observation(0, "yes", "no"), observation(1, "uncertain", "no"),
                observation(2, "yes", "no"), observation(3, "no", "yes"), observation(4, "no", "yes")]
        value = evaluate_rules(report(rows))
        self.assertFalse(value["candidates"])
        self.assertEqual(value["blocked_candidates"][0]["reason"], "missing_prior_confirmed_presence")

    def test_unknown_inside_positive_run_creates_separate_episodes(self):
        rows = presence() + [observation(2, "no", "yes"), observation(3, "no", "yes"),
                             observation(4, "uncertain", "yes"), observation(5, "no", "yes"), observation(6, "no", "yes")]
        value = evaluate_rules(report(rows))
        self.assertEqual([e["matching_frame_ids"] for e in value["candidates"]], [["f2", "f3"], ["f5", "f6"]])
        self.assertEqual(value["candidates"][1]["presence_context_unknown_frame_ids"], ["f4"])
        self.assertEqual(value["candidates"][0]["presence_context_unknown_frame_ids"], [])

    def test_unknown_cannot_join_two_isolated_positive_frames(self):
        rows = presence() + [observation(2, "no", "yes"), observation(3, "no", "uncertain"), observation(4, "no", "yes")]
        value = evaluate_rules(report(rows))
        self.assertFalse(value["candidates"])
        self.assertEqual(len(value["blocked_candidates"]), 2)
        self.assertTrue(all(e["reason"] == "insufficient_consecutive_support" for e in value["blocked_candidates"]))
        self.assertTrue(all(e["timecode_sec"] is None and e["confirmed_at_sec"] is None
                            for e in value["blocked_candidates"]))
        self.assertEqual([e["evidence_ids"] for e in value["blocked_candidates"]], [["f1", "f2"], ["f1", "f4"]])
        self.assertFalse(value["can_conclude_no_candidate"])

    def test_long_run_is_one_event_and_uses_at_most_three_original_frame_ids(self):
        rows = presence() + [observation(i, "no", "yes", timestamp=i+.0333) for i in range(2, 12)]
        value = evaluate_rules(report(rows))
        self.assertEqual(len(value["candidates"]), 1)
        event = value["candidates"][0]
        self.assertEqual(event["evidence_ids"], ["f1", "f2", "f3"])
        self.assertEqual(event["matching_frame_ids"], [f"f{i}" for i in range(2, 12)])
        self.assertEqual(event["observed_frame_count"], 10)
        self.assertEqual(event["confirmed_at_sec"], 3.0333)
        self.assertEqual(event["timecode_sec"], 3.0333)
        self.assertEqual(event["start_sec"], 2.0333)
        self.assertEqual(event["end_sec"], 11.0333)

    def test_gap_over_four_seconds_resets_historical_presence(self):
        for gap, expected in ((4.0, 1), (4.0001, 0)):
            with self.subTest(gap=gap):
                rows = presence() + [observation(2, "no", "yes", timestamp=1+gap),
                                     observation(3, "no", "yes", timestamp=2+gap)]
                value = evaluate_rules(report(rows))
                self.assertEqual(len(value["candidates"]), expected)
                if not expected:
                    self.assertEqual(value["blocked_candidates"][0]["reason"], "missing_prior_confirmed_presence")
                    self.assertEqual(len(value["coverage"]["gaps"]), 1)

    def test_cash_unknown_while_present_does_not_block_presence_context(self):
        rows = presence()
        rows[0].update(status="partial", errors=[{"feature": "cash_on_worktop", "message": "cash request failed"}])
        rows += [observation(2, "no", "yes"), observation(3, "no", "yes")]
        value = evaluate_rules(report(rows, status="partial", all_selected_frames_processed=False))
        self.assertEqual(len(value["candidates"]), 1)
        self.assertEqual(value["candidates"][0]["prior_presence_frame_ids"], ["f0", "f1"])
        self.assertEqual(value["coverage"]["unknown_frame_ids"], [])
        self.assertFalse(value["can_conclude_no_candidate"])

    def test_scoped_unrelated_errors_preserve_candidate_but_required_error_blocks(self):
        for feature, expected in (("cabinet_door", 1), ("cash_in_hands", 1),
                                  ("cash_on_worktop", 0), ("person_in_work_area", 0)):
            with self.subTest(feature=feature):
                rows = presence() + [observation(2, "no", "yes", status="partial", errors=[{"feature": feature}]),
                                     observation(3, "no", "yes")]
                value = evaluate_rules(report(rows, status="partial", all_selected_frames_processed=False))
                self.assertEqual(len(value["candidates"]), expected)

    def test_unscoped_unknown_or_malformed_error_scopes_block_all_features(self):
        for error in ({"message": "transport failure"}, {"feature": "*"}, {"feature": "all"},
                      {"feature": "cash_on_workop"}, {"feature": []}, "timeout", None):
            with self.subTest(error=error):
                rows = presence() + [observation(2, "no", "yes", status="partial", errors=[error]),
                                     observation(3, "no", "yes")]
                value = evaluate_rules(report(rows, status="partial", all_selected_frames_processed=False))
                self.assertFalse(value["candidates"])
                self.assertIn("f2", value["coverage"]["unknown_frame_ids"])

    def test_partial_without_scoped_error_and_error_status_cannot_confirm(self):
        for status in ("partial", "error"):
            with self.subTest(status=status):
                rows = presence() + [observation(2, "no", "yes", status=status), observation(3, "no", "yes")]
                value = evaluate_rules(report(rows, status="partial", all_selected_frames_processed=False))
                self.assertFalse(value["candidates"])

    def test_three_valued_and_does_not_require_irrelevant_operand_to_be_known(self):
        rows = [observation(0, "yes", "uncertain"), observation(1, "uncertain", "no"),
                observation(2, None, "no"), observation(3, "yes", None)]
        value = evaluate_rules(report(rows))
        self.assertTrue(value["can_conclude_no_candidate"])
        self.assertEqual(value["coverage"]["assessed_frame_ids"], ["f0", "f1", "f2", "f3"])
        self.assertFalse(value["coverage"]["unknown_frame_ids"])
        self.assertFalse(value["candidates"])
        self.assertFalse(value["blocked_candidates"])

    def test_missing_or_unknown_relevant_state_is_not_a_clean_negative(self):
        for person, cash in (("uncertain", "yes"), (None, "yes"), ("no", "uncertain"), ("no", None)):
            with self.subTest(person=person, cash=cash):
                value = evaluate_rules(report([observation(0, person, cash)]))
                self.assertFalse(value["can_conclude_no_candidate"])
                self.assertEqual(value["coverage"]["unknown_frame_ids"], ["f0"])

    def test_partial_missing_or_inconsistent_completion_metadata_never_means_negative(self):
        source = report([observation(0, "yes", "no")])
        variants = [{"status": "partial"}, {"status": "running"}, {"status": "failed"},
                    {"all_selected_frames_processed": False}, {"all_selected_frames_processed": 1},
                    {"selected_frame_count": 2}, {"processed_frame_count": 0}, {"selected_timestamps": [99]},
                    {"errors": [{"message": "request failed"}]}, {"budget_exceeded": True}]
        for changes in variants:
            with self.subTest(changes=changes):
                self.assertFalse(evaluate_rules({**source, **changes})["can_conclude_no_candidate"])
        for key in ("status", "selected_frame_count", "processed_frame_count", "all_selected_frames_processed"):
            missing = deepcopy(source)
            del missing[key]
            self.assertFalse(evaluate_rules(missing)["can_conclude_no_candidate"])
        self.assertFalse(evaluate_rules(report([]))["can_conclude_no_candidate"])

    def test_known_negative_frames_separated_by_large_gap_are_incomplete_coverage(self):
        value = evaluate_rules(report([observation(0, "yes", "no"), observation(1, "yes", "no", timestamp=10)]))
        self.assertFalse(value["can_conclude_no_candidate"])
        self.assertIn("excessive_frame_gaps", value["coverage"]["negative_blockers"])

    def test_returning_person_ends_run_and_new_confirmed_presence_replaces_history(self):
        rows = presence() + [observation(2, "no", "yes"), observation(3, "no", "yes"),
                             observation(4, "yes", "yes"), observation(5, "yes", "yes"),
                             observation(6, "no", "yes"), observation(7, "no", "yes")]
        events = evaluate_rules(report(rows))["candidates"]
        self.assertEqual([event["last_present_id"] for event in events], ["f1", "f5"])
        self.assertEqual([event["start_sec"] for event in events], [2, 6])

    def test_bad_observation_structure_never_silently_reorders_or_deduplicates(self):
        normal = presence()
        bad_rows = [None, {}, [None], [normal[0], normal[0]], list(reversed(normal)),
                    [observation(0, "yes", "no", timestamp=float("nan"))],
                    [observation(0, "yes", "no", timestamp=True)],
                    [observation(0, "yes", "no", timestamp=-1)],
                    [observation(0, "yes", "invalid")], [observation(0, "yes", "no", errors={})],
                    [observation(0, "yes", "no", status="done")]]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_rules({"observations": rows})
        invalid_cabinet = observation(0, "yes", "no")
        invalid_cabinet["states"]["cabinet_door"] = "locked"
        with self.assertRaises(ValueError):
            evaluate_rules(report([invalid_cabinet]))

    def test_inputs_and_policy_are_not_mutated_and_forged_transitions_labels_are_ignored(self):
        source = report(presence() + [observation(2, "no", "yes"), observation(3, "no", "yes")])
        expected = evaluate_rules(source)
        source.update(video_id="untrusted", video_path="do-not-read.mov", truth="no_violation",
                      transitions=[{"feature": "cash_on_worktop", "from_state": "yes", "to_state": "no"}],
                      observed_events=[{"type": "invented_legal_violation"}])
        snapshot = deepcopy(source)
        result = evaluate_rules(source)
        self.assertEqual(result, expected)
        self.assertEqual(source, snapshot)
        result["candidates"][0]["matching_frame_ids"].append("invented")
        result["policy"]["max_gap_sec"] = 1000
        self.assertEqual(evaluate_rules(source), expected)


if __name__ == "__main__":
    unittest.main()
