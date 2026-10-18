"""Plan coarse and extra observations using timestamps and states only."""

from bisect import bisect_left, bisect_right
from collections import deque
import math

from .features import FEATURE_STATES, KNOWN_STATES

_EPSILON_SEC = 1e-6
_MAX_WINDOW_GAP_SEC = 4.0
_PADDING_SEC = 0.5


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _times(frames):
    if not isinstance(frames, list):
        raise TypeError("frames must be a list")
    times = []
    for frame in frames:
        timestamp = _number(getattr(frame, "timestamp_sec", None), "frame timestamp_sec")
        if timestamp < 0 or (times and timestamp < times[-1]):
            raise ValueError("Frame timestamps must be nonnegative and nondecreasing")
        times.append(timestamp)
    return times


def _unique_indices(times):
    return [index for index, timestamp in enumerate(times)
            if index == 0 or timestamp != times[index - 1]]


def coarse_indices(frames, interval_sec=2.0, max_frames=90):
    """Select source indices nearest uniform PTS targets, retaining both ends.

    Frames need only expose ``timestamp_sec``; their images are never accessed.
    The interval is nominal: source gaps and the frame cap may make actual gaps
    larger. Repeated source PTS use only the first frame at that time. No source
    index is returned twice, even when targets share a nearest frame. Empty and
    one-unique-time inputs return zero and one indices respectively.
    """
    interval = _number(interval_sec, "interval_sec")
    if interval <= 0:
        raise ValueError("interval_sec must be positive")
    limit = _integer(max_frames, "max_frames", 2)
    original_times = _times(frames)
    original_indices = _unique_indices(original_times)
    times = [original_times[index] for index in original_indices]
    if len(times) <= 1:
        return original_indices
    span = times[-1] - times[0]
    ratio = span / interval
    # Cap before ceil: very small positive intervals may produce an infinite ratio.
    count = limit if ratio >= limit - 1 else max(2, math.ceil(ratio) + 1)
    if count >= len(times):
        return original_indices
    chosen = {0, len(times) - 1}
    for index in range(1, count - 1):
        target = times[0] + span * (index / (count - 1))
        right = bisect_left(times, target)
        candidates = [position for position in (right - 1, right) if 0 <= position < len(times)]
        chosen.add(min(candidates, key=lambda position: (abs(times[position] - target), position)))
    return [original_indices[index] for index in sorted(chosen)]


def _observations(observations):
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    rows = []
    ids = set()
    for row in observations:
        if not isinstance(row, dict):
            raise TypeError("Each observation must be an object")
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip() or frame_id in ids:
            raise ValueError("Observation frame IDs must be nonempty and unique")
        ids.add(frame_id)
        timestamp = _number(row.get("timestamp_sec"), "observation timestamp_sec")
        if timestamp < 0 or (rows and timestamp <= rows[-1][0]):
            raise ValueError("Observation timestamps must be nonnegative and strictly increasing")
        states = row.get("states", {})
        if not isinstance(states, dict):
            raise TypeError("Observation states must be an object")
        for feature, allowed in FEATURE_STATES.items():
            if feature in states and (not isinstance(states[feature], str) or states[feature] not in allowed):
                raise ValueError(f"Invalid state for {feature}")
        status = row.get("status", "ok")
        if not isinstance(status, str) or status not in {"ok", "partial", "error"}:
            raise ValueError("Observation status must be ok, partial or error")
        errors = row.get("errors", [])
        if not isinstance(errors, list):
            raise TypeError("Observation errors must be a list")
        blocked = set()
        if status == "error" or (status == "partial" and not errors):
            blocked.update(KNOWN_STATES)
        for error in errors:
            feature = error.get("feature") if isinstance(error, dict) else None
            if isinstance(feature, str) and feature in KNOWN_STATES:
                blocked.add(feature)
            else:
                blocked.update(KNOWN_STATES)
        rows.append((timestamp, states, blocked))
    return rows


def _suspicious(left, right):
    for feature, known in KNOWN_STATES.items():
        if feature in left[2] or feature in right[2]:
            continue
        before, after = left[1].get(feature), right[1].get(feature)
        if before in known and after in known and before != after:
            return True
        if (before == "uncertain" and after in known) or (after == "uncertain" and before in known):
            return True
    return False


def refinement_indices(frames, observations, max_new_frames=24):
    """Choose unseen dense frames around suspicious adjacent observations.

    Known changes and known/uncertain pairs get a half-second margin on both
    sides. Gaps over four seconds are never expanded. Technical errors block
    their named feature; general errors block all. Error rows are not removed
    or bridged. ``not_visible`` alone is not an uncertainty trigger.

    Repeated dense PTS use only the first frame at each time. All observed
    timestamps, including failed observations, are excluded within 1 microsecond.
    With a cap, selection takes one available frame per window in
    chronological round-robin order, nearest each window centre first. The
    returned indices are unique and sorted in source order. ``candidate_count``
    counts unique eligible frames before the cap, not the number of windows.
    This plans observations only; it does not infer an action or violation.
    """
    return _select_refinement_indices(
        frames, observations, max_new_frames,
        lambda left, right: 0 if _suspicious(left, right) else None)


def _select_refinement_indices(frames, observations, max_new_frames, window_priority,
                               *, include_selection_order=False):
    """Share actual-PTS window construction and bounded fair selection."""
    limit = _integer(max_new_frames, "max_new_frames", 0)
    times = _times(frames)
    rows = _observations(observations)
    if not times or len(rows) < 2:
        result = {"indices": [], "candidate_count": 0, "capped": False}
        if include_selection_order:
            result["selection_order"] = []
        return result
    unique_indices = set(_unique_indices(times))
    observed_times = [row[0] for row in rows]
    already_seen = set()
    for index, timestamp in enumerate(times):
        right = bisect_left(observed_times, timestamp)
        if any(abs(observed_times[position] - timestamp) <= _EPSILON_SEC
               for position in (right - 1, right) if 0 <= position < len(observed_times)):
            already_seen.add(index)
    windows_by_priority = {}
    all_candidates = set()
    for left, right in zip(rows, rows[1:]):
        if right[0] - left[0] > _MAX_WINDOW_GAP_SEC:
            continue
        priority = window_priority(left, right)
        if priority is None:
            continue
        start = max(0.0, left[0] - _PADDING_SEC)
        end = min(times[-1], right[0] + _PADDING_SEC)
        first = bisect_left(times, start - _EPSILON_SEC)
        last = bisect_right(times, end + _EPSILON_SEC)
        candidates = [index for index in range(first, last)
                      if index in unique_indices and index not in already_seen]
        if not candidates:
            continue
        centre = left[0] + (right[0] - left[0]) / 2
        candidates.sort(key=lambda index: (abs(times[index] - centre), times[index], index))
        windows_by_priority.setdefault(priority, []).append(deque(candidates))
        all_candidates.update(candidates)
    selected, selection_order = set(), []
    for priority in sorted(windows_by_priority):
        while len(selected) < limit:
            progressed = False
            for candidates in windows_by_priority[priority]:
                while candidates and candidates[0] in selected:
                    candidates.popleft()
                if candidates:
                    index = candidates.popleft()
                    selected.add(index)
                    selection_order.append(index)
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
    result = {"indices": sorted(selected), "candidate_count": len(all_candidates),
              "capped": len(all_candidates) > limit}
    if include_selection_order:
        result["selection_order"] = selection_order
    return result


def _cash_absence_condition(row):
    _, states, blocked = row
    person = None if "person_in_work_area" in blocked else states.get("person_in_work_area")
    cash = None if "cash_on_worktop" in blocked else states.get("cash_on_worktop")
    if person == "yes" or cash == "no":
        return "false"
    if person == "no" and cash == "yes":
        return "true"
    return "unknown"


def _rule_window_priority(left, right):
    before, after = _cash_absence_condition(left), _cash_absence_condition(right)
    if before == after:
        return None
    return 1 if "unknown" in (before, after) else 0


def rule_refinement_indices(frames, observations, max_new_frames=8):
    """Refine only the sampled absence-AND-worktop-cash condition.

    Known true/false boundaries have priority over known/unknown boundaries.
    Windows of each priority share the cap in chronological round-robin order,
    nearest their centres first. A known person=yes or cash=no makes the
    conjunction false even if its other operand was deliberately not checked.
    Equal false, true or unknown endpoints do not trigger refinement; short
    actions between equal endpoints can still be missed.

    Feature-scoped errors block their operand; an unknown error scope blocks
    both. A blocked value is never used as a known boundary. Nearby new frames
    may be selected at the secondary unknown/known priority; failed observed
    frames themselves are never selected again. No gap over four seconds is
    bridged. Source indices and actual timestamps are retained, without reading
    images, inferring a violation or changing any input observations.
    """
    return _select_refinement_indices(frames, observations, max_new_frames, _rule_window_priority,
                                      include_selection_order=True)
