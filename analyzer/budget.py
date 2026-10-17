"""Order existing observations for temporal coverage without changing selection."""

import math


def coverage_order(frames, indices):
    """Return selected indices: earliest, latest, then fill the largest time gap.

    Each gap is split at its selected frame nearest the actual time midpoint.
    Ties between gaps or midpoint candidates prefer the earlier time. Only gaps
    containing an unprocessed selected frame are eligible. Images are not read;
    this is a processing order, not evidence of continuous temporal coverage.
    """
    if not isinstance(frames, (list, tuple)) or not isinstance(indices, (list, tuple)):
        raise TypeError("frames and indices must be lists or tuples")
    times = []
    for frame in frames:
        value = getattr(frame, "timestamp_sec", None)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid or (times and value <= times[-1]):
            raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
        times.append(value)
    if any(type(index) is not int or not 0 <= index < len(frames) for index in indices):
        raise ValueError("Indices must be integers identifying existing frames")
    if len(set(indices)) != len(indices):
        raise ValueError("Indices must be unique")
    selected = sorted(indices)
    if len(selected) <= 2:
        return selected
    result = [selected[0], selected[-1]]
    gaps = [(0, len(selected)-1)]
    while gaps:
        left, right = min(gaps, key=lambda pair: (
            -(times[selected[pair[1]]] - times[selected[pair[0]]]), times[selected[pair[0]]]))
        gaps.remove((left, right))
        start, end = times[selected[left]], times[selected[right]]
        midpoint = start + (end-start)/2
        middle = min(range(left+1, right), key=lambda pos: (abs(times[selected[pos]]-midpoint), times[selected[pos]]))
        result.append(selected[middle])
        if middle-left > 1:
            gaps.append((left, middle))
        if right-middle > 1:
            gaps.append((middle, right))
    return result
