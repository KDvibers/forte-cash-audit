"""CPU-only motion windows for later visual review; motion is not an action."""

from bisect import bisect_left, bisect_right
from collections import deque
from io import BytesIO
import hashlib
import json
import math

from PIL import Image, ImageChops, ImageFilter


_ROI_FIELDS = {"cabinet": "cabinet_bbox_normalized", "desk": "desk_bbox_normalized"}


def _number(value, name, *, minimum=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"Invalid {name}")
    return value


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _compatible(left, right):
    intersection = (max(0, min(left[2], right[2]) - max(left[0], right[0]))
                    * max(0, min(left[3], right[3]) - max(left[1], right[1])))
    return (intersection / (_area(left) + _area(right) - intersection) >= 0.25
            and 0.25 <= _area(right) / _area(left) <= 4)


def _regions(observations, padding):
    if not isinstance(observations, list) or any(not isinstance(row, dict) for row in observations):
        raise TypeError("coarse_observations must be a list of objects")
    result = {}
    for name, field in _ROI_FIELDS.items():
        boxes, invalid, missing = [], 0, 0
        for row in observations:
            box = row.get(field)
            if box is None or row.get("status") == "error":
                missing += 1
            elif (isinstance(box, list) and len(box) == 4
                  and all(type(v) is int and 0 <= v <= 1000 for v in box)
                  and box[2] > box[0] and box[3] > box[1]):
                boxes.append(box)
            else:
                invalid += 1
        info = {"bbox_normalized": None, "source_field": field, "valid_box_count": len(boxes),
                "missing_box_count": missing, "invalid_box_count": invalid,
                "accepted_box_count": 0, "outlier_box_count": 0}
        if len(boxes) < 2:
            info["reason"] = "at_least_two_coarse_boxes_required"
        else:
            # Largest overlap cluster; equal support keeps the first coarse box.
            pivot = max(boxes, key=lambda box: sum(_compatible(box, other) for other in boxes))
            accepted = [box for box in boxes if _compatible(pivot, box)]
            info.update(accepted_box_count=len(accepted), outlier_box_count=len(boxes)-len(accepted))
            if len(accepted) < 2:
                info["reason"] = "no_supported_box_cluster"
            else:
                union = [min(b[0] for b in accepted), min(b[1] for b in accepted),
                         max(b[2] for b in accepted), max(b[3] for b in accepted)]
                width, height = union[2]-union[0], union[3]-union[1]
                info["union_bbox_normalized"] = union
                info["bbox_normalized"] = [max(0, math.floor(union[0]-padding*width)),
                                          max(0, math.floor(union[1]-padding*height)),
                                          min(1000, math.ceil(union[2]+padding*width)),
                                          min(1000, math.ceil(union[3]+padding*height))]
                info["reason"] = "coarse_box_cluster_union"
        result[name] = info
    return result


def _frames(frames):
    if not isinstance(frames, list):
        raise TypeError("frames must be a list")
    times = []
    for frame in frames:
        timestamp = _number(getattr(frame, "timestamp_sec", None), "frame timestamp_sec")
        if times and timestamp <= times[-1]:
            raise ValueError("Frame timestamps must strictly increase")
        if not isinstance(getattr(frame, "jpeg", None), (bytes, bytearray)):
            raise TypeError("Frame jpeg must contain bytes")
        _integer(getattr(frame, "width", None), "frame width", 1)
        _integer(getattr(frame, "height", None), "frame height", 1)
        times.append(timestamp)
    return times


def _pictures(frames, regions, side):
    pictures = {name: [] for name, region in regions.items() if region["bbox_normalized"] is not None}
    errors = []
    readable = 0
    for index, frame in enumerate(frames):
        if not pictures:
            break
        try:
            with Image.open(BytesIO(frame.jpeg)) as source:
                if source.format != "JPEG" or source.size != (frame.width, frame.height):
                    raise ValueError("JPEG format or dimensions do not match frame metadata")
                source.load()
                crops = {}
                for name in pictures:
                    box = regions[name]["bbox_normalized"]
                    bounds = (math.floor(box[0]*source.width/1000), math.floor(box[1]*source.height/1000),
                              math.ceil(box[2]*source.width/1000), math.ceil(box[3]*source.height/1000))
                    crop = source.crop(bounds).convert("L")
                    crop.thumbnail((side, side), Image.Resampling.BILINEAR)
                    crops[name] = crop.filter(ImageFilter.GaussianBlur(0.6))
                for name in pictures:
                    pictures[name].append(crops[name])
                readable += 1
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"frame_index": index, "timestamp_sec": frame.timestamp_sec,
                           "type": type(error).__name__, "message": str(error)})
            for name in pictures:
                pictures[name].append(None)
    return pictures, errors, readable


def _median_intensity(image):
    total, halfway = 0, image.width * image.height / 2
    for value, count in enumerate(image.histogram()):
        total += count
        if total >= halfway:
            return value


def _scores(pictures, frames, times, pixel_threshold, max_gap):
    pairs, segments = [], {}
    segment_id = -1
    frame_segments = []
    for index, picture in enumerate(pictures):
        if picture is None:
            frame_segments.append(None)
            continue
        connected = (index > 0 and frame_segments[index-1] is not None
                     and times[index]-times[index-1] <= max_gap
                     and (frames[index].width, frames[index].height) == (frames[index-1].width, frames[index-1].height))
        if not connected:
            segment_id += 1
            segments[segment_id] = [index, index]
        segments[segment_id][1] = index
        frame_segments.append(segment_id)
    for index in range(1, len(pictures)):
        before, after = pictures[index-1], pictures[index]
        row = {"frame_index": index, "start_sec": times[index-1], "end_sec": times[index]}
        if before is None or after is None:
            row["skipped_reason"] = "unreadable_frame"
        elif times[index]-times[index-1] > max_gap:
            row["skipped_reason"] = "frame_gap"
        elif frame_segments[index] != frame_segments[index-1]:
            row["skipped_reason"] = "frame_geometry_changed"
        else:
            shift = _median_intensity(after) - _median_intensity(before)
            adjusted = after.point([max(0, min(255, value-shift)) for value in range(256)])
            histogram = ImageChops.difference(before, adjusted).histogram()
            pixels = before.width * before.height
            row.update(changed_fraction=sum(histogram[pixel_threshold:])/pixels,
                       mean_abs_change=sum(value*count for value, count in enumerate(histogram))/pixels,
                       median_luma_shift=shift, segment_id=frame_segments[index])
        pairs.append(row)
    return pairs, segments


def _groups(pairs, threshold, merge_gap):
    groups = []
    for pair in pairs:
        if pair.get("changed_fraction", -1) < threshold:
            continue
        if (groups and groups[-1]["segment_id"] == pair["segment_id"]
                and pair["start_sec"]-groups[-1]["end_sec"] <= merge_gap + 1e-9):
            groups[-1]["end_sec"] = pair["end_sec"]
            groups[-1]["pairs"].append(pair)
        else:
            groups.append({"start_sec": pair["start_sec"], "end_sec": pair["end_sec"],
                           "segment_id": pair["segment_id"], "pairs": [pair]})
    return groups


def _windows(name, groups, segments, times, context, max_window):
    windows, dropped = [], []
    core = max_window - 2*context
    for group_index, group in enumerate(groups):
        segment_start, segment_end = segments[group["segment_id"]]
        chunks = {}
        for pair in group["pairs"]:
            position = (pair["end_sec"]-group["start_sec"])/core
            if not math.isfinite(position):
                raise ValueError("max_window_sec is too small for the timestamp range")
            chunk = max(0, math.ceil(position-1e-9)-1)
            chunks.setdefault(chunk, []).append(pair)
        for chunk, active in chunks.items():
            lower = group["start_sec"] + chunk*core
            upper = min(group["end_sec"], lower+core)
            start = max(segment_start, bisect_left(times, lower-context-1e-9))
            end = min(segment_end, bisect_right(times, upper+context+1e-9)-1)
            info = {"roi": name, "group_id": f"{name}-{group_index:04d}", "chunk_index": chunk,
                    "group_start_sec": group["start_sec"], "group_end_sec": group["end_sec"],
                    "start_sec": times[start], "end_sec": times[end], "split_group": len(chunks) > 1}
            interior = [pair for pair in active if start < pair["frame_index"] < end]
            if not interior:
                dropped.append({**info, "reason": "insufficient_three_frame_context"})
                continue
            peak = max(interior, key=lambda pair: (pair["changed_fraction"], -pair["frame_index"]))
            indices = [start, peak["frame_index"], end]
            windows.append({**info, "peak_sec": peak["end_sec"], "score": peak["changed_fraction"],
                            "frame_indices": indices, "frame_timestamps": [times[index] for index in indices]})
    return windows, dropped


def _prioritize(windows):
    """First represent distinct temporal groups, alternating available ROIs."""
    grouped = {}
    for window in windows:
        grouped.setdefault(window["group_id"], []).append(window)
    for group in grouped.values():
        group.sort(key=lambda item: (-item["score"], item["start_sec"], item["chunk_index"]))
    queues = {}
    for name in _ROI_FIELDS:
        queues[name] = sorted((group for group in grouped.values() if group[0]["roi"] == name),
                              key=lambda group: (-group[0]["score"], group[0]["start_sec"], group[0]["group_id"]))
    groups = []
    while any(queues.values()):
        for name in _ROI_FIELDS:
            if queues[name]:
                groups.append(queues[name].pop(0))
    ranked = []
    depth = 0
    while any(len(group) > depth for group in groups):
        for group in groups:
            if len(group) > depth:
                ranked.append({**group[depth], "priority": len(ranked)+1})
        depth += 1
    return ranked


def _refinement(windows, times, observed, limit):
    queues, candidates = [], set()
    for window in sorted(windows, key=lambda item: item["priority"]):
        all_indices = range(window["frame_indices"][0], window["frame_indices"][-1]+1)
        eligible = [index for index in all_indices if index not in observed]
        window["eligible_refinement_indices"] = eligible
        window["already_observed_indices"] = [index for index in all_indices if index in observed]
        centre = (window["start_sec"]+window["end_sec"])/2
        queues.append(deque(sorted(eligible, key=lambda index: (abs(times[index]-centre), times[index]))))
        candidates.update(eligible)
    selected, order = set(), []
    while len(selected) < limit:
        progressed = False
        for queue in queues:
            while queue and queue[0] in selected:
                queue.popleft()
            if queue:
                index = queue.popleft()
                selected.add(index)
                order.append(index)
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    for window in windows:
        eligible = window["eligible_refinement_indices"]
        window["refinement_indices"] = [index for index in eligible if index in selected]
        window["refinement_dropped_indices"] = [index for index in eligible if index not in selected]
        window["refinement_complete"] = not window["refinement_dropped_indices"]
    return {"indices": sorted(selected), "selection_order": order, "candidate_count": len(candidates),
            "dropped_indices": sorted(candidates-selected), "capped": len(candidates) > limit}


def propose_motion_windows(
    frames, coarse_observations, *, max_windows=8, pixel_threshold=20,
    min_changed_fraction=0.04, roi_padding=0.1, analysis_side=128,
    merge_gap_sec=0.5, context_sec=1.0, max_window_sec=6.0, max_pair_gap_sec=1.0,
    max_refinement_frames=24, observed_indices=None,
):
    """Suggest bounded three-frame windows without calling a model.

    ``frames`` are dense, strictly increasing-PTS objects with jpeg/width/height.
    ``coarse_observations`` supplies automatically detected cabinet/desk boxes;
    the caller should exclude refinements. No state, video name, label, action
    timestamp or fixed scene coordinate participates in selection. At least two
    overlapping boxes are required for an ROI. A union of the strongest overlap
    cluster rejects isolated boxes; it does not verify that the object is real.

    Pixel changes are measured after grayscale blur and median-brightness shift
    correction. Consecutive above-threshold pairs form temporal groups. Long
    groups are split; distinct groups receive priority before more windows from
    one long group. ROIs alternate so one region cannot consume the entire cap.
    Each selected window refers to three distinct actual source-frame indices.
    ``refinement_indices`` separately selects dense frames inside those windows,
    excluding ``observed_indices``. Selection alternates windows, nearest each
    window's centre first, up to ``max_refinement_frames``. Per-window omissions
    are retained; a partial window may be insufficient for state confirmation.

    Corrupt frames, resolution changes and excessive PTS gaps break continuity.
    Dropped windows and actual spacing remain visible in the return value.
    Motion can be a person, reflection or lighting change, not necessarily an
    action. Subsample actions, low-contrast changes and activity outside the ROIs
    can be missed. The absence of a proposal cannot establish absence of action.
    """
    _integer(max_windows, "max_windows", 0)
    _integer(max_refinement_frames, "max_refinement_frames", 0)
    _integer(pixel_threshold, "pixel_threshold", 1)
    if pixel_threshold > 255:
        raise ValueError("pixel_threshold must not exceed 255")
    _integer(analysis_side, "analysis_side", 8)
    threshold = _number(min_changed_fraction, "min_changed_fraction", strict=True)
    padding = _number(roi_padding, "roi_padding")
    if threshold > 1 or padding > 1:
        raise ValueError("Changed fraction and ROI padding must not exceed 1")
    merge_gap = _number(merge_gap_sec, "merge_gap_sec")
    context = _number(context_sec, "context_sec")
    max_window = _number(max_window_sec, "max_window_sec", strict=True)
    max_gap = _number(max_pair_gap_sec, "max_pair_gap_sec", strict=True)
    if max_window <= 2*context:
        raise ValueError("max_window_sec must exceed twice context_sec")
    times = _frames(frames)
    if observed_indices is None:
        observed = set()
    else:
        if (not isinstance(observed_indices, (list, tuple, set, frozenset))
                or any(type(index) is not int or not 0 <= index < len(frames) for index in observed_indices)):
            raise ValueError("observed_indices must contain valid source-frame indices")
        observed = set(observed_indices)
    regions = _regions(coarse_observations, padding)
    pictures, errors, readable = _pictures(frames, regions, analysis_side)
    scores, proposals, dropped = {}, [], []
    group_count = 0
    for name, images in pictures.items():
        pairs, segments = _scores(images, frames, times, pixel_threshold, max_gap)
        scores[name] = pairs
        groups = _groups(pairs, threshold, merge_gap)
        group_count += len(groups)
        candidates, omitted = _windows(name, groups, segments, times, context, max_window)
        proposals.extend(candidates)
        dropped.extend(omitted)
    ranked = _prioritize(proposals)
    selected = ranked[:max_windows]
    dropped.extend({**window, "reason": "window_cap"} for window in ranked[max_windows:])
    refinement = _refinement(selected, times, observed, max_refinement_frames)
    policy = {"revision": 1, "max_windows": max_windows, "pixel_threshold": pixel_threshold,
              "min_changed_fraction": threshold, "roi_padding": padding, "analysis_side": analysis_side,
              "merge_gap_sec": merge_gap, "context_sec": context, "max_window_sec": max_window,
              "max_pair_gap_sec": max_gap, "blur_radius": 0.6, "minimum_roi_boxes": 2,
              "max_refinement_frames": max_refinement_frames,
              "roi_cluster_min_iou": 0.25, "roi_cluster_area_ratio": [0.25, 4.0]}
    return {
        "kind": "motion_proposals_not_actions", "policy": policy,
        "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        "rois": regions, "windows": sorted(selected, key=lambda item: (item["start_sec"], item["roi"], item["priority"])),
        "dropped_windows": dropped, "scores": scores, "frame_errors": errors,
        "refinement_indices": refinement["indices"], "refinement": refinement,
        "coverage": {
            "input_frame_count": len(frames), "decoded_frame_count": readable,
            "image_decode_skipped_no_roi": not pictures,
            "first_frame_sec": times[0] if times else None, "last_frame_sec": times[-1] if times else None,
            "actual_max_frame_gap_sec": max((b-a for a, b in zip(times, times[1:])), default=0.0),
            "compared_pairs_by_roi": {name: sum("changed_fraction" in pair for pair in pairs) for name, pairs in scores.items()},
            "skipped_pairs_by_roi": {name: sum("skipped_reason" in pair for pair in pairs) for name, pairs in scores.items()},
            "candidate_group_count": group_count, "candidate_window_count": len(selected)+len(dropped),
            "selected_window_count": len(selected), "dropped_window_count": len(dropped),
            "capped": len(ranked) > max_windows, "blind_shorter_actions": True,
            "refinement_capped": refinement["capped"],
            "limitations": ["Actions entirely between sampled frames can be missed.",
                            "Motion is not an action; person motion, reflections and lighting may trigger proposals.",
                            "Low-contrast motion or motion outside the detected ROIs may be missed.",
                            "Three chosen frames need not capture every phase of an action.",
                            "No proposal does not establish absence of an action."],
        },
    }
