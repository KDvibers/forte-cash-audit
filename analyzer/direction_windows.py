"""Select real frames for experimental direction queries, without model calls.

These windows are proposals. A positive interaction, shared geometry, or motion
does not establish banknote loading, removal, successful counting or a violation.
"""

from collections import Counter
import hashlib
import json
import math


_EPS = 1e-9
_MERGE_GAP = 1.0
_MAX_SPAN = 6.0
_CONTEXT = 0.5
_MAX_FRAMES = 5
_MIN_IOU = 0.5


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) is int and 0 <= v <= 1000 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def _stable_box(boxes):
    """Use an observed medoid, rather than inventing or widening the geometry."""
    boxes = sorted(set(boxes))
    return min(boxes, key=lambda box: (-sum(_iou(box, other) for other in boxes), box))


def _check(row, times):
    if row.get("status") != "ok":
        return "action_not_ok"
    if row.get("region") != "desk":
        return "region_not_desk"
    if not isinstance(row.get("errors", []), list) or row.get("errors"):
        return "action_errors"
    if not isinstance(row.get("result"), dict) or row["result"].get("interaction") != "yes":
        return "interaction_not_yes"
    if row.get("object_visibility") != "yes":
        return "object_not_visible"
    if not _valid_box(row.get("object_bbox_normalized")):
        return "invalid_bbox"
    start, end = row.get("start_sec"), row.get("end_sec")
    if not _finite(start) or not _finite(end) or end < start:
        return "invalid_interval"
    if not times:
        return "no_frame_pool"
    if start < times[0]-_EPS or end > times[-1]+_EPS:
        return "interval_outside_frame_pool"
    if "evidence" in row:
        evidence = row["evidence"]
        if not isinstance(evidence, list) or not evidence:
            return "invalid_source_evidence"
        stamps = [item.get("timestamp_sec") if isinstance(item, dict) else None for item in evidence]
        if (not all(_finite(t) for t in stamps)
                or any(b <= a for a, b in zip(stamps, stamps[1:]))
                or abs(stamps[0]-start) > _EPS or abs(stamps[-1]-end) > _EPS
                or any(not any(abs(t-actual) <= _EPS for actual in times) for t in stamps)):
            return "invalid_source_evidence"
    return None


def _collect(checks, times, dropped):
    by_id = {}
    for index, row in enumerate(checks):
        if not isinstance(row, dict):
            raise TypeError("action_checks must contain objects")
        source_id = row.get("window_id")
        if not isinstance(source_id, str) or not source_id.strip():
            dropped.append({"source_window_ids": [], "input_index": index, "reason": "invalid_window_id"})
            continue
        reason = _check(row, times)
        candidate = None if reason else {"start": row["start_sec"], "end": row["end_sec"],
                                         "box": tuple(row["object_bbox_normalized"]), "ids": [source_id]}
        signature = (reason,) if reason else (None, candidate["start"], candidate["end"], candidate["box"])
        by_id.setdefault(source_id, []).append((index, reason, candidate, signature))
    candidates = []
    for source_id, entries in sorted(by_id.items()):
        conflict = len({entry[3] for entry in entries}) > 1
        for number, (index, reason, candidate, _) in enumerate(entries):
            reason = "conflicting_window_id" if conflict else reason or ("duplicate_window_id" if number else None)
            if reason:
                dropped.append({"source_window_ids": [source_id], "input_index": index, "reason": reason})
            else:
                candidates.append(candidate)
    unique = {}
    for candidate in sorted(candidates, key=lambda c: (c["start"], c["end"], c["box"], c["ids"])):
        key = candidate["start"], candidate["end"], candidate["box"]
        if key in unique:
            dropped.append({"source_window_ids": list(candidate["ids"]), "reason": "duplicate_action_window",
                            "duplicate_of_window_id": unique[key]["ids"][0]})
            unique[key]["ids"].extend(candidate["ids"])
        else:
            unique[key] = candidate
    return list(unique.values())


def _episodes(candidates):
    episodes = []
    for row in candidates:
        matches = [(index, episode) for index, episode in enumerate(episodes)
                   if row["start"] <= episode["end"] + _MERGE_GAP + _EPS
                   and all(_iou(row["box"], box) >= _MIN_IOU for box in episode["boxes"])]
        if matches:
            _, episode = min(matches, key=lambda pair: (
                -min(_iou(row["box"], box) for box in pair[1]["boxes"]), -pair[1]["end"], pair[0]))
            episode["end"] = max(episode["end"], row["end"])
            episode["boxes"].add(row["box"])
            episode["rows"].append(row)
        else:
            episodes.append({"start": row["start"], "end": row["end"], "boxes": {row["box"]}, "rows": [row]})
    return sorted(episodes, key=lambda e: (e["start"], e["end"], _stable_box(e["boxes"])))


def _chunks(episode, times):
    parts, empty_splits_skipped = [], 0
    for row in episode["rows"]:
        start, end = row["start"], row["end"]
        if end-start <= _MAX_SPAN+_EPS:
            parts.append({"start": start, "end": end, "ids": list(row["ids"])})
            continue
        stride = _MAX_SPAN-_MERGE_GAP
        last_part = math.ceil((end-start-_MAX_SPAN)/stride)
        # Bound work by the sampled pool, not by a possibly enormous PTS gap.
        # Only split windows near an actual frame can supply query anchors.
        possible = {0, last_part}
        for timestamp in times:
            if start-_CONTEXT-_EPS <= timestamp <= end+_CONTEXT+_EPS:
                part = math.floor((timestamp-start)/stride)
                possible.update(i for i in (part-1, part, part+1) if 0 <= i <= last_part)
        empty_splits_skipped += last_part+1-len(possible)
        for part in sorted(possible):
            left = min(end, start+stride*part)
            parts.append({"start": left, "end": min(end, left+_MAX_SPAN), "ids": list(row["ids"])})
    chunks = []
    for part in sorted(parts, key=lambda p: (p["start"], p["end"], p["ids"])):
        if (chunks and part["start"] <= chunks[-1]["end"]+_MERGE_GAP+_EPS
                and max(chunks[-1]["end"], part["end"])-chunks[-1]["start"] <= _MAX_SPAN+_EPS):
            chunks[-1]["end"] = max(chunks[-1]["end"], part["end"])
            chunks[-1]["ids"] = sorted(set(chunks[-1]["ids"]+part["ids"]))
        else:
            chunks.append(part)
    return chunks, empty_splits_skipped


def _anchors(chunk, times, *, reserve_context=False):
    start, end = chunk["start"], chunk["end"]
    inside = [i for i, t in enumerate(times) if start-_EPS <= t <= end+_EPS]
    if not inside:
        return None
    if reserve_context:
        # Keep the actual interior endpoints and at most one middle frame.
        # The two remaining slots are for context, even at a video boundary;
        # missing context must not silently become more interior observations.
        selected = {inside[0], inside[-1]}
        if len(inside) > 2:
            midpoint = start + (end-start)/2
            selected.add(min(inside[1:-1], key=lambda i: (abs(times[i]-midpoint), times[i], i)))
    else:
        count = min(_MAX_FRAMES, len(inside))
        targets = [start+j*(end-start)/(count-1) for j in range(count)] if count > 1 else [start]
        selected = {min(inside, key=lambda i: (abs(times[i]-target), times[i], i)) for target in targets}
    before = [i for i, t in enumerate(times) if start-_CONTEXT-_EPS <= t < start-_EPS]
    after = [i for i, t in enumerate(times) if end+_EPS < t <= end+_CONTEXT+_EPS]
    added, context_dropped = [], []
    for side, index in (("before", before[0] if before else None), ("after", after[-1] if after else None)):
        reason = None
        if index is None:
            reason = "video_edge_or_no_frame_in_context"
        elif len(selected) >= _MAX_FRAMES:
            reason = "no_spare_frame_slot"
        elif max(times[max(selected)], times[index])-min(times[min(selected)], times[index]) > _MAX_SPAN+_EPS:
            reason = "duration_cap"
        if reason:
            context_dropped.append({"side": side, "frame_index": index, "reason": reason})
        else:
            selected.add(index)
            added.append(index)
    if len(selected) < 2:
        return None
    indices = sorted(selected)
    actual = [times[i] for i in indices]
    return {"frame_indices": indices, "frame_timestamps": actual,
            "start_sec": actual[0], "end_sec": actual[-1],
            "max_anchor_gap_sec": max(b-a for a, b in zip(actual, actual[1:])),
            "context_indices_added": added, "context_candidates_dropped": context_dropped,
            "context_before_sec": max(0, start-actual[0]), "context_after_sec": max(0, actual[-1]-end),
            "unselected_dense_frame_count": sum(actual[0] <= t <= actual[-1] for t in times)-len(indices)}


def select_direction_windows(frames, action_checks, max_windows=4, *, reserve_context=False):
    """Select <=5 actual frames per bounded query, preserving input objects.

    ``frames`` must be a list/tuple with finite, nonnegative, strictly increasing
    ``timestamp_sec`` attributes. No JPEG bytes are decoded, generated or copied.
    Only completed positive desk interactions with visible valid integer boxes
    qualify. When supplied, evidence times must agree with the source frame pool.
    Contact-head timestamps and dataset labels never affect selection.

    IoU agreement is required against every box in a temporal episode; this
    prevents pairwise chaining from drifting between objects. The stable box is
    an observed medoid. Episodes may require multiple <=6-second chunks. A cap
    visits different episodes in chronological order before their second chunks.
    By default context cannot evict an already selected interior frame. The
    opt-in ``reserve_context=True`` instead chooses up to three interior frames
    (actual endpoints and nearest middle), reserving one slot for each context
    side. Missing context and the six-second limit remain explicit; neither
    mode establishes direction or recovers upstream missed interactions.
    """
    if type(max_windows) is not int or max_windows < 0:
        raise ValueError("max_windows must be a nonnegative integer")
    if type(reserve_context) is not bool:
        raise ValueError("reserve_context must be a boolean")
    if not isinstance(frames, (list, tuple)) or not isinstance(action_checks, list):
        raise TypeError("frames must be a list/tuple and action_checks a list")
    times = [getattr(frame, "timestamp_sec", None) for frame in frames]
    if (not all(_finite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
    dropped = []
    candidates = _collect(action_checks, times, dropped)
    episodes = _episodes(candidates)
    queues, sequence_number, empty_splits_skipped = [], 0, 0
    for episode_index, episode in enumerate(episodes):
        queue = []
        chunks, skipped = _chunks(episode, times)
        empty_splits_skipped += skipped
        for chunk_index, chunk in enumerate(chunks):
            anchors = _anchors(chunk, times, reserve_context=reserve_context)
            if anchors is None:
                dropped.append({"source_window_ids": sorted(chunk["ids"]), "reason": "insufficient_actual_frames",
                                "interaction_start_sec": chunk["start"], "interaction_end_sec": chunk["end"]})
                continue
            queue.append({"window_id": f"direction_{sequence_number:04d}", "roi": "desk",
                          "kind": "direction_query_candidate", "source_window_ids": sorted(chunk["ids"]),
                          "episode_id": f"direction_episode_{episode_index:04d}", "chunk_index": chunk_index,
                          "episode_start_sec": episode["start"], "episode_end_sec": episode["end"],
                          "interaction_start_sec": chunk["start"], "interaction_end_sec": chunk["end"],
                          "object_bbox_normalized": list(_stable_box(episode["boxes"])),
                          "bbox_agreement_min_iou": min(_iou(a, b) for a in episode["boxes"] for b in episode["boxes"]),
                          "distinct_bbox_count": len(episode["boxes"]), **anchors})
            sequence_number += 1
        if queue:
            queues.append(queue)
    ranked = [queue[depth] for depth in range(max(map(len, queues), default=0)) for queue in queues if len(queue) > depth]
    for priority, window in enumerate(ranked, 1):
        window["priority"] = priority
    selected = ranked[:max_windows]
    cap_drops = [{**window, "reason": "direction_window_cap"} for window in ranked[max_windows:]]
    dropped.extend(cap_drops)
    selected_ids = {source_id for window in selected for source_id in window["source_window_ids"]}
    cap_ids = {source_id for window in cap_drops for source_id in window["source_window_ids"]}
    policy = {"revision": 1, "max_windows": max_windows, "max_frames_per_window": _MAX_FRAMES,
              "merge_gap_sec": _MERGE_GAP, "max_window_span_sec": _MAX_SPAN, "context_sec": _CONTEXT,
              "minimum_bbox_iou": _MIN_IOU, "bbox_agreement": "all_pairs_within_episode",
              "stable_bbox": "observed_medoid_unique_boxes_lexicographic_tie",
              "eligibility": "status_ok_no_errors_desk_interaction_yes_visible_valid_bbox",
              "anchors": "nearest_actual_to_uniform_inside_targets_earlier_tie",
              "context": "spare_slots_only_preserve_interior_before_then_after",
              "cap_order": "chronological_episodes_round_robin_chunks",
              "oversized_window_overlap_sec": _MERGE_GAP, "contact_evidence_sec_used": False,
              "oversized_empty_splits": "skip_when_no_sampled_frame_can_be_inside_or_in_context",
              "minimum_actual_frames": 2, "interpretation": "experimental_query_selection_not_direction"}
    if reserve_context:
        # Do not change the legacy policy fingerprint when the option is off.
        policy.update(revision=2, max_interior_frames_per_window=3,
                      anchors="actual_inside_endpoints_then_nearest_interior_midpoint_earlier_tie",
                      context="reserved_slots_farthest_within_context_before_then_after_duration_cap")
    return {"kind": "direction_query_selection_not_actions", "windows": selected, "dropped_windows": dropped,
            "candidate_count": len(ranked), "selected_count": len(selected), "capped": len(ranked) > max_windows,
            "policy": policy, "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
            "coverage": {"available_frame_count": len(times), "input_action_count": len(action_checks),
                         "eligible_unique_action_count": len(candidates), "temporal_episode_count": len(episodes),
                         "actual_max_frame_gap_sec": max((b-a for a, b in zip(times, times[1:])), default=0),
                         "selected_max_anchor_gap_sec": max((w["max_anchor_gap_sec"] for w in selected), default=0),
                         "selected_image_reference_count": sum(len(w["frame_indices"]) for w in selected),
                         "selected_distinct_frame_count": len({i for w in selected for i in w["frame_indices"]}),
                         "cap_dropped_window_count": len(cap_drops), "fully_cap_dropped_source_window_ids": sorted(cap_ids-selected_ids),
                         "empty_split_windows_skipped": empty_splits_skipped,
                         "partially_cap_dropped_source_window_ids": sorted(cap_ids & selected_ids),
                         "drop_counts_by_reason": dict(sorted(Counter(d["reason"] for d in dropped).items())),
                         "blind_shorter_actions": len(times) > 1,
                         "upstream_missed_or_uncertain_interactions_recovered": False}}
