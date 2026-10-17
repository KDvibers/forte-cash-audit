"""Select bounded action queries; an open door is a cue, never transfer evidence."""

from copy import deepcopy
import hashlib
import json
import math


_ROIS = ("cabinet", "desk")
_UNRELATED_FEATURES = {"person_in_work_area", "cash_on_worktop", "cash_in_hands", "notes_in_machine"}


def combine_action_motion(motion, counter_motion, frames):
    """Use cabinet motion and a separately discovered counter, not the whole desk.

    Counter windows cover at most three seconds. Their centre is an actual frame;
    a later high-motion hand movement must not replace the middle of the window.
    Geometry is inferred independently of action labels or dataset identifiers.
    """
    combined = {"windows": [], "dropped_windows": []}
    for source, roi, offset in ((motion, "cabinet", 0), (counter_motion, "desk", 1)):
        for field in combined:
            for item in source.get(field, []):
                if item.get("roi") != roi:
                    continue
                window = deepcopy(item)
                if _valid_window(window):
                    first, peak, last = window["frame_indices"]
                    if last >= len(frames):
                        raise ValueError("Action window exceeds the sampled frame pool")
                    if window["frame_timestamps"] != [frames[i].timestamp_sec for i in (first, peak, last)]:
                        raise ValueError("Action timestamps do not match actual source frames")
                    window["priority"] = 2*window["priority"]-1+offset
                    if roi == "desk":
                        if frames[last].timestamp_sec-frames[first].timestamp_sec > 3.0+1e-6:
                            raise ValueError("Counter action windows must not exceed three seconds")
                        middle_time = (frames[first].timestamp_sec+frames[last].timestamp_sec)/2
                        middle = min(range(first+1, last), key=lambda i: abs(frames[i].timestamp_sec-middle_time))
                        window["motion_peak_frame_index"] = peak
                        window["frame_indices"] = [first, middle, last]
                        window["frame_timestamps"] = [frames[i].timestamp_sec for i in (first, middle, last)]
                        window["group_id"] = "counter-"+window["group_id"]
                        window["anchor_policy"] = "actual_temporal_midpoint"
                combined[field].append(window)
    return combined


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _valid_window(window):
    if not isinstance(window, dict) or window.get("roi") not in _ROIS:
        return False
    indices, times = window.get("frame_indices"), window.get("frame_timestamps")
    return (isinstance(window.get("group_id"), str) and bool(window["group_id"])
            and type(window.get("priority")) is int and window["priority"] > 0
            and isinstance(indices, list) and len(indices) == 3
            and all(type(index) is int and index >= 0 for index in indices)
            and all(b > a for a, b in zip(indices, indices[1:]))
            and isinstance(times, list) and len(times) == 3
            and all(_finite(t) for t in times)
            and all(b > a for a, b in zip(times, times[1:]))
            and _finite(window.get("start_sec")) and _finite(window.get("end_sec"))
            and window["start_sec"] == times[0] and window["end_sec"] == times[-1])


def _open_times(observations):
    times = set()
    for row in observations:
        if not isinstance(row, dict):
            raise TypeError("Observations must contain objects")
        states, errors = row.get("states", {}), row.get("errors", [])
        if (not isinstance(states, dict) or states.get("cabinet_door") != "open"
                or row.get("status") not in ("ok", "partial")
                or not _finite(row.get("timestamp_sec")) or not isinstance(errors, list)
                or (row.get("status") == "partial" and not errors)):
            continue
        # General errors and errors affecting the cabinet block the cue.
        if any(not isinstance(error, dict) or error.get("feature") not in _UNRELATED_FEATURES
               for error in errors):
            continue
        times.add(row["timestamp_sec"])
    return sorted(times)


def gate_cabinet_claim(check, observations):
    """Require an observed open door for a positive near-open-cabinet claim.

    This consistency gate cannot establish cash transfer or confirm that the
    item is money. Missing support produces uncertainty, never a negative fact.
    Raw calls and every other result head are preserved for review.
    """
    record = deepcopy(check)
    if (record.get("region") != "cabinet"
            or record.get("result", {}).get("cash_near_cabinet") != "yes"):
        return record
    record["cash_near_cabinet_before_gate"] = "yes"
    start, end = record.get("start_sec"), record.get("end_sec")
    opened = ([timestamp for timestamp in _open_times(observations) if start <= timestamp <= end]
              if _finite(start) and _finite(end) and start <= end else [])
    record["cabinet_state_gate_open_times"] = opened
    record["cabinet_state_gate"] = ("observed_open_in_window" if opened else "no_observed_open_in_window")
    if not opened:
        record["result"]["cash_near_cabinet"] = "uncertain"
    return record


def select_action_windows(motion, observations, max_windows=8):
    """Recover cap-dropped triples and prioritize observed open-cabinet episodes.

    Without a trusted open observation inside a cabinet window, motion priority
    is unchanged. Otherwise, rank matching chunks first within their temporal
    groups, then alternate cabinet/desk groups before revisiting any group.
    The original frame triples and motion-refinement allocation are untouched.
    ``windows`` is query order; ``action_priority`` is its one-based rank, while
    ``priority`` retains the original motion rank. No action truth is produced.
    """
    if type(max_windows) is not int or max_windows < 0:
        raise ValueError("max_windows must be a nonnegative integer")
    if not isinstance(motion, dict) or not isinstance(observations, list):
        raise TypeError("motion must be an object and observations a list")
    for field in ("windows", "dropped_windows"):
        if not isinstance(motion.get(field, []), list):
            raise TypeError(f"motion.{field} must be a list")
    opened = _open_times(observations)
    candidates, dropped, seen = [], [], set()
    sources = [(window, "motion_selected") for window in motion.get("windows", [])]
    sources += [(window, "motion_dropped") for window in motion.get("dropped_windows", [])]
    for original, source in sources:
        if not isinstance(original, dict):
            raise TypeError("Motion windows must be objects")
        window = deepcopy(original)
        if source == "motion_dropped" and window.get("reason") != "window_cap":
            dropped.append(window)
            continue
        if not _valid_window(window):
            window["reason"] = "invalid_action_window"
            dropped.append(window)
            continue
        key = (window["roi"], tuple(window["frame_indices"]), tuple(window["frame_timestamps"]))
        if key in seen:
            window["reason"] = "duplicate_action_window"
            dropped.append(window)
            continue
        seen.add(key)
        window.pop("reason", None)
        window["selection_source"] = source
        window["open_observation_times"] = ([t for t in opened if window["start_sec"] <= t <= window["end_sec"]]
                                             if window["roi"] == "cabinet" else [])
        candidates.append(window)
    candidates.sort(key=lambda window: (window["priority"], window["start_sec"], window["roi"]))
    conditioned = any(window["open_observation_times"] for window in candidates)
    ranked = candidates
    if conditioned:
        grouped = {}
        for window in candidates:
            grouped.setdefault((window["roi"], window["group_id"]), []).append(window)
        for group in grouped.values():
            group.sort(key=lambda window: (not bool(window["open_observation_times"]), window["priority"]))
        queues = {roi: sorted((group for (name, _), group in grouped.items() if name == roi),
                              key=lambda group: (not bool(group[0]["open_observation_times"]), group[0]["priority"]))
                  for roi in _ROIS}
        groups = []
        while any(queues.values()):
            for roi in _ROIS:
                if queues[roi]:
                    groups.append(queues[roi].pop(0))
        ranked = [group[depth] for depth in range(max(map(len, groups)))
                  for group in groups if len(group) > depth]
    for index, window in enumerate(ranked, 1):
        window["action_priority"] = index
        window["reason"] = ("observed_open_cabinet" if window["open_observation_times"] else
                            "motion_priority_with_roi_group_coverage" if conditioned else "motion_priority")
    selected = ranked[:max_windows]
    for window in ranked[max_windows:]:
        window["selection_reason"] = window["reason"]
        window["reason"] = "action_window_cap"
        dropped.append(window)
    policy = {"revision": 1, "max_windows": max_windows,
              "cue": "trusted_open_observation_inside_cabinet_window",
              "ranking": "cue_first_within_group_then_alternating_rois_distinct_groups_before_repeats",
              "fallback": "original_motion_priority", "interpretation": "query_selection_not_action_evidence"}
    return {"kind": "action_query_selection_not_actions", "windows": selected, "dropped_windows": dropped,
            "capped": len(candidates) > max_windows, "selected_count": len(selected),
            "candidate_count": len(candidates), "state_conditioned": conditioned, "policy": policy,
            "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()}
