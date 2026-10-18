"""CPU-only proposals for dense frames omitted by earlier action queries.

The selector never changes an action result. ``no`` and ``uncertain`` are query
priorities, not proof that the unshown part of an interval is empty. Execution,
model calls and their shared deadline belong to the caller.
"""

from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math


_EPS = 1e-9
_PRIORITIES = {"uncertain": 0, "no": 1, "yes": 2}
_ROIS = ("desk", "cabinet")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _box(value):
    return (isinstance(value, list) and len(value) == 4
            and all(type(v) is int and 0 <= v <= 1000 for v in value)
            and value[0] < value[2] and value[1] < value[3])


def _iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


def _source(row, times):
    if row.get("status") != "ok" or not isinstance(row.get("errors", []), list) or row.get("errors"):
        return None, "incomplete_source_check"
    roi = row.get("region")
    if roi not in _ROIS:
        return None, "unsupported_region"
    if row.get("object_visibility") != "yes" or not _box(row.get("object_bbox_normalized")):
        return None, "object_not_verified"
    feature = "interaction" if roi == "desk" else "cash_near_cabinet"
    result = row.get("result", {})
    state = result.get(feature) if isinstance(result, dict) else None
    if not isinstance(state, str) or state not in _PRIORITIES:
        return None, "invalid_source_result"
    evidence = row.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 3:
        return None, "invalid_source_evidence"
    stamps = [item.get("timestamp_sec") if isinstance(item, dict) else None for item in evidence]
    if not all(_finite(t) for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        return None, "invalid_source_evidence"
    if (not _finite(row.get("start_sec")) or not _finite(row.get("end_sec"))
            or abs(row["start_sec"]-stamps[0]) > _EPS or abs(row["end_sec"]-stamps[-1]) > _EPS):
        return None, "invalid_source_interval"
    indices = []
    for timestamp in stamps:
        right = bisect_left(times, timestamp)
        matches = [i for i in (right-1, right) if 0 <= i < len(times) and abs(times[i]-timestamp) <= _EPS]
        if not matches:
            return None, "evidence_not_in_frame_pool"
        indices.append(min(matches, key=lambda i: (abs(times[i]-timestamp), i)))
    if len(set(indices)) != len(indices):
        return None, "duplicate_source_frame"
    return {"id": row["window_id"], "roi": roi, "state": state,
            "box": tuple(row["object_bbox_normalized"]), "indices": indices,
            "start": indices[0], "end": indices[-1]}, None


def _episodes(rows, times):
    episodes = []
    for row in sorted(rows, key=lambda r: (r["start"], r["end"], r["roi"], r["id"])):
        matches = [e for e in episodes if e["roi"] == row["roi"]
                   and times[row["start"]] <= times[e["end"]]+1.0+_EPS
                   and all(_iou(row["box"], other["box"]) >= .5 for other in e["rows"])]
        if matches:
            episode = max(matches, key=lambda e: (e["end"], -e["number"]))
            episode["end"] = max(episode["end"], row["end"])
            episode["rows"].append(row)
        else:
            episodes.append({"roi": row["roi"], "start": row["start"], "end": row["end"],
                             "rows": [row], "number": len(episodes)})
    return episodes


def select_short_action_windows(frames, action_checks, max_windows=4, *, max_span_sec=1.5):
    """Select bounded triples around previously unshown dense source frames.

    Inputs are the unchanged sampled frame pool and completed action checks
    including their real ``evidence`` timestamps. No JPEG is decoded or copied.
    Source windows need visible valid object geometry, but no positive action is
    required. Uncertain checks have priority over negative, then positive ones.
    Within each priority band, desk/cabinet and distinct temporal episodes take
    turns before a second query in the same episode. Each candidate consists of
    three consecutive pool frames entirely inside a source window. It must add
    at least one frame not shown by any geometrically compatible source check
    in its episode, and must respect ``max_span_sec``.

    The cap and omitted frames remain explicit. This does not recover windows
    dropped upstream, motion outside known objects, or sub-pool actions. A query
    may expose a new view without changing the uncertain interpretation.
    """
    if type(max_windows) is not int or max_windows < 0:
        raise ValueError("max_windows must be a nonnegative integer")
    if not _finite(max_span_sec) or max_span_sec <= 0:
        raise ValueError("max_span_sec must be finite and positive")
    if not isinstance(frames, (list, tuple)) or not isinstance(action_checks, list):
        raise TypeError("frames must be a list/tuple and action_checks a list")
    times = [getattr(frame, "timestamp_sec", None) for frame in frames]
    if not all(_finite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
    dropped, by_id = [], {}
    for index, row in enumerate(action_checks):
        if not isinstance(row, dict):
            raise TypeError("action_checks must contain objects")
        source_id = row.get("window_id")
        if not isinstance(source_id, str) or not source_id.strip():
            dropped.append({"input_index": index, "reason": "invalid_window_id"})
            continue
        source, reason = _source(row, times)
        by_id.setdefault(source_id, []).append((source, reason))
    rows = []
    for source_id, entries in sorted(by_id.items()):
        signatures = {json.dumps(pair, sort_keys=True) for pair in entries}
        if len(signatures) > 1:
            dropped.append({"source_window_ids": [source_id], "reason": "conflicting_window_id"})
            continue
        source, reason = entries[0]
        if reason:
            dropped.append({"source_window_ids": [source_id], "reason": reason})
        else:
            rows.append(source)
        if len(entries) > 1:
            dropped.append({"source_window_ids": [source_id], "reason": "duplicate_window_id",
                            "omitted_count": len(entries)-1})
    episodes = _episodes(rows, times)
    candidates = []
    dense_inside, shown, eligible_new = set(), set(), set()
    for episode in episodes:
        episode_id = f"short_episode_{episode['number']:04d}"
        observed = {i for row in episode["rows"] for i in row["indices"]}
        inside = {i for row in episode["rows"] for i in range(row["start"], row["end"]+1)}
        # Include object identity in coverage keys: identical PTS at another ROI
        # does not mean that the relevant object crop was previously inspected.
        shown.update((episode_id, i) for i in observed)
        dense_inside.update((episode_id, i) for i in inside)
        unique = {}
        for row in episode["rows"]:
            for centre in range(row["start"]+1, row["end"]):
                indices = (centre-1, centre, centre+1)
                new = [i for i in indices if i not in observed]
                if not new or times[indices[-1]]-times[indices[0]] > max_span_sec+_EPS:
                    continue
                eligible_new.update((episode_id, i) for i in new)
                # The source box is retained, never widened by a union.
                key = indices, row["box"]
                item = unique.setdefault(key, {"roi": row["roi"], "episode_id": episode_id,
                    "object_bbox_normalized": list(row["box"]), "source_window_ids": [],
                    "source_results": {}, "frame_indices": list(indices),
                    "frame_timestamps": [times[i] for i in indices], "new_frame_indices": new,
                    "new_frame_timestamps": [times[i] for i in new],
                    "start_sec": times[indices[0]], "end_sec": times[indices[-1]],
                    "max_anchor_gap_sec": max(times[b]-times[a] for a, b in zip(indices, indices[1:]))})
                item["source_window_ids"].append(row["id"])
                item["source_results"][row["id"]] = row["state"]
        for item in unique.values():
            item["source_window_ids"] = sorted(set(item["source_window_ids"]))
            item["source_priority"] = min(item["source_results"].values(), key=_PRIORITIES.get)
            item["reason"] = "previously_unshown_dense_frames_inside_source_window"
            candidates.append(item)
    # Deterministic priority bands. Round-robin episodes and regions prevent
    # every extra query being spent in a single long, high-motion episode.
    ranked = []
    for state in _PRIORITIES:
        queues = {}
        for episode in episodes:
            episode_id = f"short_episode_{episode['number']:04d}"
            queue = [w for w in candidates if w["episode_id"] == episode_id and w["source_priority"] == state]
            queue.sort(key=lambda w: (-len(w["new_frame_indices"]), w["start_sec"],
                                     w["frame_indices"], w["object_bbox_normalized"]))
            if queue:
                queues.setdefault(episode["roi"], []).append(queue)
        ordered = []
        depth = 0
        while any(len(groups) > depth for groups in queues.values()):
            for roi in _ROIS:
                if len(queues.get(roi, [])) > depth:
                    ordered.append(queues[roi][depth])
            depth += 1
        depth = 0
        while any(len(queue) > depth for queue in ordered):
            ranked.extend(queue[depth] for queue in ordered if len(queue) > depth)
            depth += 1
    selected, covered_new, selected_triples = [], set(), set()
    for window in ranked:
        keys = {(window["episode_id"], i) for i in window["new_frame_indices"]}
        triple = window["episode_id"], tuple(window["frame_indices"])
        if triple in selected_triples or not keys-covered_new:
            dropped.append({**window, "reason": "no_additional_frame_coverage"})
        elif len(selected) >= max_windows:
            dropped.append({**window, "reason": "short_action_window_cap"})
        else:
            selected_triples.add(triple)
            covered_new.update(keys)
            selected.append({**window, "window_id": f"short_action_{len(selected):04d}",
                             "priority": len(selected)+1})
    policy = {"revision": 1, "max_windows": max_windows, "max_span_sec": max_span_sec,
              "priority": list(_PRIORITIES), "roi_order": list(_ROIS),
              "episode_gap_sec": 1.0, "episode_all_pairs_min_bbox_iou": .5,
              "anchors": "three_consecutive_actual_frames_inside_source_window",
              "novelty": "unshown_by_compatible_source_checks_in_episode",
              "ranking": "state_then_round_robin_roi_episode_then_more_new_frames_earlier_time",
              "interpretation": "query_selection_not_action_evidence"}
    counts = Counter(w["reason"] for w in dropped)
    return {"kind": "short_action_query_selection_not_actions", "windows": selected,
            "dropped_windows": dropped, "candidate_count": len(candidates), "selected_count": len(selected),
            "capped": bool(counts["short_action_window_cap"]), "policy": policy,
            "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
            "coverage": {"available_frame_count": len(times), "input_action_count": len(action_checks),
                "eligible_source_count": len(rows), "temporal_episode_count": len(episodes),
                "source_shown_frame_references": len(shown), "dense_inside_frame_references": len(dense_inside),
                "unshown_inside_frame_references": len(dense_inside-shown),
                "eligible_new_frame_references": len(eligible_new), "selected_new_frame_references": len(covered_new),
                "remaining_eligible_frame_references": len(eligible_new-covered_new),
                "drop_counts_by_reason": dict(sorted(counts.items())),
                "upstream_dropped_windows_recovered": False, "blind_subsample_actions": True}}
