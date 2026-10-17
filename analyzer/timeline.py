"""Build confirmed changes of visible states, without inferring actions or violations."""

from dataclasses import dataclass
import math

from .features import FEATURE_STATES, UNKNOWN_STATES


@dataclass
class _Run:
    state: str
    first_id: str
    first_sec: float
    last_id: str
    last_sec: float
    count: int = 1

    def extend(self, frame_id: str, timestamp: float) -> None:
        self.last_id = frame_id
        self.last_sec = timestamp
        self.count += 1


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _validate(observations: list[dict]) -> list[tuple[str, float, dict, str]]:
    if not isinstance(observations, list):
        raise TypeError("observations must be a list.")
    validated = []
    previous_time = None
    frame_ids = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise TypeError(f"observations[{index}] must be an object.")
        frame_id = observation.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(f"observations[{index}].frame_id must be a nonempty string.")
        if frame_id in frame_ids:
            raise ValueError("frame_id values must be unique.")
        frame_ids.add(frame_id)
        timestamp = _finite_number(observation.get("timestamp_sec"), "timestamp_sec")
        if timestamp < 0:
            raise ValueError("timestamp_sec cannot be negative.")
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("timestamp_sec values must be strictly increasing.")
        previous_time = timestamp
        status = observation.get("status", "ok")
        if not isinstance(status, str) or status not in {"ok", "partial", "error"}:
            raise ValueError("status must be ok, partial or error.")
        if "errors" in observation and not isinstance(observation["errors"], list):
            raise TypeError("errors must be a list when provided.")
        states = observation.get("states", {})
        if not isinstance(states, dict):
            raise TypeError("states must be an object.")
        for feature, allowed in FEATURE_STATES.items():
            if feature in states:
                value = states[feature]
                if not isinstance(value, str) or value not in allowed:
                    raise ValueError(f"Invalid state for {feature}.")
        validated.append((frame_id, timestamp, states, status))
    return validated


def build_transitions(
    observations: list[dict], max_gap_sec: float = 4.0, confirmation_frames: int = 2
) -> list[dict]:
    """Return confirmed state changes with the interval in which each occurred.

    Input timestamps must strictly increase and frame IDs must be unique.
    Both the initial state and each replacement require ``confirmation_frames``
    consecutive known observations. A brief opposing run does not replace a
    confirmed state; returning to that state cancels the pending replacement.

    A gap greater than ``max_gap_sec`` resets all features. Missing states,
    ``uncertain`` and ``not_visible`` reset only the affected feature. An error
    status resets all, while a partial status preserves usable fields. The
    caller must mark failed fields uncertain; error messages are never parsed
    as observations. Other feature names are ignored.

    Each interval spans the last observed old state and first observed new one;
    it is not an exact action time. Evidence contains those two frame IDs and
    the confirming frame. No missing action, absence outside the visible area,
    legal conclusion or final API verdict is inferred.
    """
    max_gap = _finite_number(max_gap_sec, "max_gap_sec")
    if max_gap <= 0:
        raise ValueError("max_gap_sec must be positive.")
    if isinstance(confirmation_frames, bool) or not isinstance(confirmation_frames, int):
        raise TypeError("confirmation_frames must be an integer.")
    if confirmation_frames < 2:
        raise ValueError("confirmation_frames must be at least 2.")
    validated = _validate(observations)
    stable: dict[str, _Run] = {}
    pending: dict[str, _Run] = {}
    transitions = []
    previous_time = None
    for frame_id, timestamp, states, status in validated:
        if previous_time is not None and timestamp - previous_time > max_gap:
            stable.clear()
            pending.clear()
        previous_time = timestamp
        for feature in FEATURE_STATES:
            state = states.get(feature)
            if status == "error" or state is None or state in UNKNOWN_STATES:
                stable.pop(feature, None)
                pending.pop(feature, None)
                continue
            confirmed = stable.get(feature)
            if confirmed is not None and state == confirmed.state:
                confirmed.extend(frame_id, timestamp)
                pending.pop(feature, None)
                continue
            candidate = pending.get(feature)
            if candidate is None or candidate.state != state:
                candidate = _Run(state, frame_id, timestamp, frame_id, timestamp)
                pending[feature] = candidate
            else:
                candidate.extend(frame_id, timestamp)
            if candidate.count < confirmation_frames:
                continue
            if confirmed is not None:
                transitions.append({
                    "feature": feature,
                    "from_state": confirmed.state,
                    "to_state": candidate.state,
                    "interval_start_sec": confirmed.last_sec,
                    "interval_end_sec": candidate.first_sec,
                    "confirmed_at_sec": candidate.last_sec,
                    "evidence_frame_ids": [confirmed.last_id, candidate.first_id, candidate.last_id],
                })
            stable[feature] = candidate
            pending.pop(feature, None)
    return transitions
