"""Evaluate visual review signals; this module never makes a legal verdict."""

from copy import deepcopy
import hashlib
import json
import math

from .features import FEATURE_STATES


_RULE_ID = "cash_visible_after_person_disappears"
_PERSON = "person_in_work_area"
_CASH = "cash_on_worktop"
_POLICY = {
    "revision": 2,
    "rule_id": _RULE_ID,
    "confirmation_frames": 2,
    "presence_confirmation_frames": 2,
    "max_gap_sec": 4.0,
    "presence_context": "historical_presence_not_tracked_exit",
    "timecode_semantics": "candidate_confirmation_not_departure",
    "evidence_selection": "last_confirmed_present_first_matching_confirmation",
    "unknown_condition": "break_matching_run",
    "unknown_person": "break_new_presence_confirmation_but_keep_confirmed_history_in_same_segment",
    "gap": "reset_matching_run_and_presence_history",
    "interpretation": "sampled_visual_candidate_for_human_review_not_legal_finding",
    "threshold_basis": "engineering_parameters_not_legal_requirements",
}


def _valid_time(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _read_observations(value):
    if not isinstance(value, list):
        raise ValueError("observations must be a list")
    rows, seen = [], set()
    for original in value:
        if not isinstance(original, dict):
            raise ValueError("Each observation must be an object")
        frame_id, timestamp = original.get("frame_id"), original.get("timestamp_sec")
        if not isinstance(frame_id, str) or not frame_id.strip() or frame_id in seen:
            raise ValueError("Observation frame IDs must be nonempty and unique")
        if not _valid_time(timestamp) or (rows and timestamp <= rows[-1]["timestamp_sec"]):
            raise ValueError("Observation timestamps must be finite, nonnegative and strictly increasing")
        states = original.get("states", {})
        if not isinstance(states, dict):
            raise ValueError("Observation states must be an object")
        for feature, allowed in FEATURE_STATES.items():
            if feature in states and (not isinstance(states[feature], str) or states[feature] not in allowed):
                raise ValueError(f"Invalid state for {feature}")
        status, errors = original.get("status", "ok"), original.get("errors", [])
        if not isinstance(status, str) or status not in {"ok", "partial", "error"}:
            raise ValueError("Observation status must be ok, partial or error")
        if not isinstance(errors, list):
            raise ValueError("Observation errors must be a list")
        blocked = set(FEATURE_STATES) if status == "error" or (status == "partial" and not errors) else set()
        for error in errors:
            feature = error.get("feature") if isinstance(error, dict) else None
            if isinstance(feature, str) and feature in FEATURE_STATES:
                blocked.add(feature)
            else:
                # A typo or unknown scope must not make a technical error harmless.
                blocked.update(FEATURE_STATES)
        rows.append({"frame_id": frame_id, "timestamp_sec": timestamp,
                     "person": None if _PERSON in blocked else states.get(_PERSON),
                     "cash": None if _CASH in blocked else states.get(_CASH),
                     "technical_partial": status != "ok" or bool(errors)})
        seen.add(frame_id)
    return rows


def _condition(row):
    # Three-valued AND: one known false operand is enough to disprove the
    # conjunction at this frame, even when the other operand is unknown.
    if row["person"] == "yes" or row["cash"] == "no":
        return "false"
    if row["person"] == "no" and row["cash"] == "yes":
        return "true"
    return "unknown"


def _episode(run, context):
    first, last = run[0], run[-1]
    confirmation = run[1] if len(run) >= _POLICY["confirmation_frames"] else None
    history = context["history"]
    evidence = [history["last_present_id"]] if history else []
    evidence.append(first["frame_id"])
    if confirmation is not None:
        evidence.append(confirmation["frame_id"])
    reasons = []
    if len(run) < _POLICY["confirmation_frames"]:
        reasons.append("insufficient_consecutive_support")
    if history is None:
        reasons.append("missing_prior_confirmed_presence")
    result = {
        "type": _RULE_ID, "source_rule_id": _RULE_ID,
        "status": "blocked" if reasons else "candidate",
        "start_sec": first["timestamp_sec"], "end_sec": last["timestamp_sec"],
        "timecode_sec": confirmation["timestamp_sec"] if confirmation is not None else None,
        "confirmed_at_sec": confirmation["timestamp_sec"] if confirmation is not None else None,
        "timecode_semantics": _POLICY["timecode_semantics"],
        "last_present_id": history["last_present_id"] if history else None,
        "last_present_sec": history["last_present_sec"] if history else None,
        "prior_presence_frame_ids": list(history["prior_presence_frame_ids"]) if history else [],
        "presence_context_unknown_frame_ids": list(context["unknown_ids"]),
        "evidence_ids": evidence,
        "matching_frame_ids": [row["frame_id"] for row in run],
        "observed_frame_count": len(run),
        "interval_interpretation": "matching_sampled_frames_not_continuous_visibility",
    }
    if reasons:
        result.update(reason=reasons[0], reasons=reasons)
    return result


def evaluate_rules(report):
    """Return visual candidates, blocked episodes and rule-specific coverage.

    Cash must be visible while no person is visible on two consecutive usable
    frames. Earlier presence needs two consecutive usable person=yes frames in
    the same PTS segment. Unknown frames break matching runs and new presence
    confirmation, but do not erase an already observed historical presence.
    This is not identity recognition or continuous tracking of an exit.
    The timecode is the second matching selected frame, when the rule gained
    enough model observations; it is not a verified disappearance timestamp.
    Evidence retains prior presence, first match and this confirmation frame.

    Scoped errors block only their known feature; unknown scopes block all.
    Partial reports may retain supported positive candidates. Absence of a
    candidate is conclusive only within this rule and sampling policy, with
    complete processing and no unresolved signal, ambiguity or excessive gap.
    Neither candidates nor can_conclude_no_candidate are API/legal verdicts.
    Input objects, JPEGs, labels and supplied transitions are never modified or
    used to manufacture evidence. Evidence file verification belongs to export.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    rows = _read_observations(report.get("observations"))
    errors = report.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError("Report errors must be a list")
    report_status = report.get("status")
    if report_status is not None and report_status not in ("running", "completed", "partial", "failed"):
        raise ValueError("Unknown report status")
    candidates, blocked_candidates = [], []
    assessed_ids, unknown_ids, true_ids, false_ids, gaps = [], [], [], [], []
    run, run_context, presence_run, history, context_unknown_ids = [], None, [], None, []
    previous = None

    def finish_run():
        if run:
            episode = _episode(run, run_context)
            (blocked_candidates if episode["status"] == "blocked" else candidates).append(episode)

    for row in rows:
        if previous is not None and row["timestamp_sec"]-previous["timestamp_sec"] > _POLICY["max_gap_sec"]:
            finish_run()
            run, run_context, presence_run, history, context_unknown_ids = [], None, [], None, []
            gaps.append({"before_frame_id": previous["frame_id"], "after_frame_id": row["frame_id"],
                         "gap_sec": row["timestamp_sec"]-previous["timestamp_sec"]})
        previous = row
        if row["person"] == "yes":
            presence_run.append(row)
            if len(presence_run) >= _POLICY["presence_confirmation_frames"]:
                history = {"last_present_id": row["frame_id"], "last_present_sec": row["timestamp_sec"],
                           "prior_presence_frame_ids": [item["frame_id"] for item in presence_run[-2:]]}
                context_unknown_ids = []
        else:
            presence_run = []
            if row["person"] != "no":
                context_unknown_ids.append(row["frame_id"])
        condition = _condition(row)
        if condition == "unknown":
            unknown_ids.append(row["frame_id"])
        else:
            assessed_ids.append(row["frame_id"])
            (true_ids if condition == "true" else false_ids).append(row["frame_id"])
        if condition == "true":
            if not run:
                run_context = {"history": deepcopy(history), "unknown_ids": list(context_unknown_ids)}
            run.append(row)
        else:
            finish_run()
            run, run_context = [], None
    finish_run()

    selected, processed = report.get("selected_frame_count"), report.get("processed_frame_count")
    complete_selection = (report.get("all_selected_frames_processed") is True
                          and type(selected) is int and selected == len(rows)
                          and type(processed) is int and processed == len(rows))
    if "selected_timestamps" in report:
        complete_selection = complete_selection and report["selected_timestamps"] == [row["timestamp_sec"] for row in rows]
    technical_ids = [row["frame_id"] for row in rows if row["technical_partial"]]
    negative_blockers = []
    for blocked, reason in (
        (not rows, "no_observations"), (report_status != "completed", "report_not_completed"),
        (not complete_selection, "selected_frames_not_fully_processed"),
        (bool(technical_ids), "observation_technical_errors"), (bool(errors), "report_errors"),
        (report.get("budget_exceeded") is True, "budget_exceeded"),
        (bool(gaps), "excessive_frame_gaps"), (bool(unknown_ids), "unknown_condition_states"),
        (bool(blocked_candidates), "blocked_visual_candidates"), (bool(candidates), "visual_candidates_present"),
    ):
        if blocked:
            negative_blockers.append(reason)
    policy = deepcopy(_POLICY)
    return {
        "kind": "visual_rule_candidates_not_legal_verdict", "report_status": report_status,
        "candidates": candidates, "blocked_candidates": blocked_candidates,
        "can_conclude_no_candidate": not negative_blockers,
        "coverage": {"observed_frame_count": len(rows), "assessed_frame_count": len(assessed_ids),
                     "unknown_frame_count": len(unknown_ids), "assessed_frame_ids": assessed_ids,
                     "unknown_frame_ids": unknown_ids, "condition_true_frame_ids": true_ids,
                     "condition_false_frame_ids": false_ids, "technical_partial_frame_ids": technical_ids,
                     "gaps": gaps, "selected_frame_count": selected, "processed_frame_count": processed,
                     "all_selected_frames_processed": report.get("all_selected_frames_processed") is True,
                     "selection_counts_consistent": complete_selection,
                     "report_partial": report_status != "completed", "negative_blockers": negative_blockers},
        "policy": policy,
        "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
