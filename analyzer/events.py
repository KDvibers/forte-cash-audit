"""Describe supported visual changes, without inferring actions or violations."""

from dataclasses import dataclass
import math


_MONEY = ("cash_on_worktop", "notes_in_machine")
_FEATURES = frozenset((*_MONEY, "person_in_work_area"))
_STATES = frozenset({"yes", "no", "uncertain"})
_EVENTS = {
    ("cash_on_worktop", "no", "yes"): (
        "cash_appeared_on_worktop", "Банкноты появились на рабочем столе",
        "На более раннем кадре банкноты на рабочем столе не видны, "
        "на последующих кадрах видны. Способ их появления не установлен.",
    ),
    ("cash_on_worktop", "yes", "no"): (
        "cash_no_longer_visible_on_worktop", "Банкноты перестали быть видны на рабочем столе",
        "На более раннем кадре банкноты на рабочем столе видны, "
        "на последующих кадрах не видны. Место их перемещения не установлено.",
    ),
    ("notes_in_machine", "no", "yes"): (
        "notes_appeared_in_machine", "Банкноты появились в счётной машине",
        "На более раннем кадре банкноты в счётной машине не видны, "
        "на последующих кадрах видны. Работа машины и пересчёт не подтверждены.",
    ),
    ("notes_in_machine", "yes", "no"): (
        "notes_no_longer_visible_in_machine", "Банкноты перестали быть видны в счётной машине",
        "На более раннем кадре банкноты в счётной машине видны, "
        "на последующих кадрах не видны. Пересчёт и место перемещения не установлены.",
    ),
}


@dataclass(frozen=True)
class _Observation:
    frame_id: str
    timestamp: float
    states: dict
    blocked: frozenset

    def state(self, feature):
        return None if feature in self.blocked else self.states.get(feature)


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


def _observations(observations):
    if not isinstance(observations, list):
        raise TypeError("observations must be a list")
    rows, ids = [], set()
    for row in observations:
        if not isinstance(row, dict):
            raise TypeError("Each observation must be an object")
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip() or frame_id in ids:
            raise ValueError("Observation frame IDs must be nonempty and unique")
        timestamp = _number(row.get("timestamp_sec"), "timestamp_sec")
        if timestamp < 0 or (rows and timestamp <= rows[-1].timestamp):
            raise ValueError("Observation timestamps must be nonnegative and strictly increasing")
        states = row.get("states", {})
        if not isinstance(states, dict):
            raise TypeError("states must be an object")
        for feature in _FEATURES:
            if feature in states and (not isinstance(states[feature], str) or states[feature] not in _STATES):
                raise ValueError(f"Invalid state for {feature}")
        status = row.get("status", "ok")
        if not isinstance(status, str) or status not in {"ok", "partial", "error"}:
            raise ValueError("status must be ok, partial or error")
        errors = row.get("errors", [])
        if not isinstance(errors, list):
            raise TypeError("errors must be a list")
        blocked = set()
        if status == "error" or (status == "partial" and not errors):
            blocked.update(_FEATURES)
        for error in errors:
            feature = error.get("feature") if isinstance(error, dict) else None
            if isinstance(feature, str) and feature.strip() and feature not in {"all", "*"}:
                blocked.add(feature)
            else:
                blocked.update(_FEATURES)
        rows.append(_Observation(frame_id, timestamp, states, frozenset(blocked)))
        ids.add(frame_id)
    return rows


def _confirmed_states(rows, max_gap):
    """Check that a supplied transition's old state really had two-frame support."""
    stable, pending, result = {}, {}, []
    previous = None
    for row in rows:
        if previous is not None and row.timestamp - previous > max_gap:
            stable.clear()
            pending.clear()
        previous = row.timestamp
        for feature in _MONEY:
            state = row.state(feature)
            if state not in {"yes", "no"}:
                stable.pop(feature, None)
                pending.pop(feature, None)
            elif stable.get(feature) == state:
                pending.pop(feature, None)
            elif pending.get(feature) == state:
                stable[feature] = state
                pending.pop(feature, None)
            else:
                pending[feature] = state
        result.append(stable.copy())
    return result


def _transition_event(transition, rows, indices, stable, max_gap):
    feature = transition.get("feature")
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError("Transition feature must be a nonempty string")
    if feature not in _MONEY:
        return None
    before, after = transition.get("from_state"), transition.get("to_state")
    if not isinstance(before, str) or not isinstance(after, str) or (feature, before, after) not in _EVENTS:
        raise ValueError("Money transitions must change between yes and no")
    time_keys = ("interval_start_sec", "interval_end_sec", "confirmed_at_sec")
    times = [_number(transition.get(key), key) for key in time_keys]
    if not 0 <= times[0] < times[1] < times[2]:
        raise ValueError("Transition times must satisfy 0 <= start < end < confirmed")
    evidence = transition.get("evidence_frame_ids")
    if (not isinstance(evidence, list) or len(evidence) != 3
            or any(not isinstance(item, str) or item not in indices for item in evidence)
            or len(set(evidence)) != 3):
        raise ValueError("Transition evidence must contain three distinct existing frame IDs")
    positions = [indices[item] for item in evidence]
    if positions != sorted(positions):
        raise ValueError("Transition evidence must follow timestamp order")
    if any(rows[position].timestamp != timestamp for position, timestamp in zip(positions, times)):
        raise ValueError("Transition timestamps must match the evidence frames")
    start, end, confirmed = positions
    # Missing or failed observations cannot be skipped to bridge a transition.
    if end != start + 1 or stable[start].get(feature) != before or rows[start].state(feature) != before:
        return None
    for index in range(end, confirmed + 1):
        if rows[index].state(feature) != after or rows[index].timestamp - rows[index - 1].timestamp > max_gap:
            return None
    kind, title, description = _EVENTS[feature, before, after]
    return {
        "type": kind, "title": title, "description": description,
        "feature": feature, "from_state": before, "to_state": after,
        **{key: transition[key] for key in time_keys},
        "evidence_frame_ids": evidence.copy(),
    }


def _cash_without_person(run):
    if len(run) < 2:
        return None
    first, confirming, last = run[0], run[1], run[-1]
    evidence = [first.frame_id, confirming.frame_id]
    if last.frame_id != confirming.frame_id:
        evidence.append(last.frame_id)
    return {
        "type": "cash_visible_without_person_in_work_area",
        "title": "Банкноты видны на столе без видимого человека в рабочей зоне",
        "description": "На последовательных проверенных кадрах банкноты видны на рабочем столе, "
                       "а человек в наблюдаемой рабочей зоне не виден. "
                       "Обстановка между кадрами и вне видимой зоны не установлена.",
        "start_sec": first.timestamp, "end_sec": last.timestamp,
        "interval_start_sec": first.timestamp, "interval_end_sec": last.timestamp,
        "confirmed_at_sec": confirming.timestamp,
        "observed_frame_count": len(run), "evidence_frame_ids": evidence,
    }


def build_observed_events(observations, transitions, max_gap_sec=4.0):
    """Return descriptive events backed by known, technically usable observations.

    Money changes require supplied confirmed transitions; this function does not
    invent transitions from frames alone. Their three evidence IDs and PTS must
    agree with observations, both states need at least two-frame support, and no
    unknown/error/gap may be skipped. Structurally malformed inputs raise errors;
    transitions contradicted by usable states or broken support are omitted.

    The separate cash-without-person event requires two consecutive matching
    observations. Each uninterrupted run yields one event, with its first,
    confirming and last frame (at most three unique IDs). The interval bounds
    sampled observations, not uninterrupted visibility between frames.

    Missing/uncertain states and affected-field errors reset support. Unscoped
    errors, error status, and partial status without scoped errors block all
    relevant fields. An error explicitly belonging to another feature does not.
    No transfer, machine operation, identity, intent or legal verdict is inferred.
    Inputs are not modified; returned events are ordered by confirmation time.
    """
    max_gap = _number(max_gap_sec, "max_gap_sec")
    if max_gap <= 0:
        raise ValueError("max_gap_sec must be positive")
    rows = _observations(observations)
    if not isinstance(transitions, list):
        raise TypeError("transitions must be a list")
    indices = {row.frame_id: index for index, row in enumerate(rows)}
    stable = _confirmed_states(rows, max_gap)
    events, seen = [], set()
    for transition in transitions:
        if not isinstance(transition, dict):
            raise TypeError("Each transition must be an object")
        event = _transition_event(transition, rows, indices, stable, max_gap)
        if event is not None:
            identity = (event["type"], tuple(event["evidence_frame_ids"]))
            if identity not in seen:
                seen.add(identity)
                events.append(event)
    run = []
    for row in rows:
        matches = row.state("person_in_work_area") == "no" and row.state("cash_on_worktop") == "yes"
        if not matches or (run and row.timestamp - run[-1].timestamp > max_gap):
            event = _cash_without_person(run)
            if event is not None:
                events.append(event)
            run = []
        if matches:
            run.append(row)
    event = _cash_without_person(run)
    if event is not None:
        events.append(event)
    return sorted(events, key=lambda item: (item["confirmed_at_sec"], item["interval_start_sec"], item["type"]))
