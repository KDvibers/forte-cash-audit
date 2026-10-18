"""Bounded video-only resampling and CPU motion proposals for local review.

These helpers select actual frames, not actions or violations. Focused sampling
can still miss motion between source frames, and image changes can be lighting.
"""

from fractions import Fraction
from io import BytesIO
import math
from os import PathLike
from pathlib import Path
from time import perf_counter

import av
from PIL import Image, ImageChops, ImageStat

from .video import VideoFrame, _duration, _scan, _select, _timestamp


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return Fraction(str(value))


def sample_video_intervals(path, windows, interval_sec=.1, max_frames_per_window=32,
                           max_windows=4, max_side=1280):
    """Decode two video-only passes and return real JPEGs inside requested bounds.

    Every window is a dict with a unique nonempty ``window_id``, ``start_sec``
    and ``end_sec`` relative to the video's first presentation timestamp. Each
    interval is at most three seconds and must lie within the video duration.
    Exceeding ``max_windows`` raises instead of silently dropping a request.

    The first pass scans the whole video; the second decodes through the latest
    selected frame. No audio stream is decoded. Selection uses actual PTS within
    each closed interval; no nearest frame outside the interval is substituted.
    Duplicate PTS keep their first decoded frame. Overlapping windows reference
    one shared sorted ``frames`` pool. Images preserve aspect ratio, never upscale,
    and match the actual frame timestamp. An interval containing no frame raises.

    ``sampling_capped`` describes the requested nominal interval versus the cap;
    measured gaps and unobserved boundary intervals describe actual coverage.
    A successful selection does not confirm that any action occurred.
    """
    started = perf_counter()
    spacing = _number(interval_sec, "interval_sec", positive=True)
    _integer(max_frames_per_window, "max_frames_per_window", 2)
    _integer(max_windows, "max_windows", 0)
    _integer(max_side, "max_side", 1)
    if not isinstance(windows, list):
        raise TypeError("windows must be a list of objects")
    if len(windows) > max_windows:
        raise ValueError("Requested window count exceeds max_windows")
    requests, identifiers = [], set()
    for row in windows:
        if not isinstance(row, dict):
            raise TypeError("windows must contain objects")
        identifier = row.get("window_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Every window needs a unique nonempty window_id")
        start = _number(row.get("start_sec"), "start_sec")
        end = _number(row.get("end_sec"), "end_sec")
        if end < start or end-start > 3:
            raise ValueError("Window bounds must be ordered and span at most three seconds")
        requests.append((identifier, start, end))
        identifiers.add(identifier)
    if not isinstance(path, (str, PathLike)) or not str(path).strip():
        raise ValueError("path must identify an existing local video file")
    video_path = Path(path).expanduser().resolve(strict=True)
    if not video_path.is_file():
        raise ValueError("path must identify a file, not a directory")

    stamps, stream_duration = _scan(video_path)
    origin = stamps[0].timestamp
    duration = _duration(stamps, stream_duration)
    # _scan sorts by (PTS, decoded index), so setdefault selects deterministically.
    distinct = {}
    for stamp in stamps:
        distinct.setdefault(stamp.timestamp, stamp)
    unique = list(distinct.values())
    choices, wanted = [], {}
    for identifier, start, end in requests:
        if end > Fraction(str(duration)):
            raise ValueError(f"Window {identifier!r} extends beyond video duration")
        eligible = [s for s in unique if start <= s.timestamp-origin <= end]
        if not eligible:
            raise ValueError(f"Window {identifier!r} contains no actual presentation frame")
        needed = max(2, math.ceil((end-start)/spacing)+1)
        chosen = _select(eligible, min(max_frames_per_window, needed))
        choices.append((identifier, start, end, chosen, needed, len(eligible)))
        wanted.update((s.index, s) for s in chosen)

    sampled = {}
    if wanted:
        with av.open(str(video_path), mode="r") as container:
            if not container.streams.video:
                raise ValueError("The file no longer contains a video stream")
            for index, frame in enumerate(container.decode(container.streams.video[0])):
                if index not in wanted:
                    continue
                timestamp = _timestamp(frame)
                if timestamp != wanted[index].timestamp:
                    raise RuntimeError("Video timestamps changed between sampling passes")
                picture = frame.to_image().convert("RGB")
                picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                encoded = BytesIO()
                picture.save(encoded, format="JPEG", quality=90)
                sampled[index] = VideoFrame(float(timestamp-origin), encoded.getvalue(),
                                            picture.width, picture.height)
                if len(sampled) == len(wanted):
                    break
        if len(sampled) != len(wanted):
            raise RuntimeError("Not all selected video frames could be decoded")
    ordered = sorted(wanted.values(), key=lambda s: (s.timestamp, s.index))
    pool_indices = {s.index: i for i, s in enumerate(ordered)}
    records = []
    for identifier, start, end, chosen, needed, available in choices:
        times = [float(s.timestamp-origin) for s in chosen]
        gaps = [float(b.timestamp-a.timestamp) for a, b in zip(chosen, chosen[1:])]
        left_gap = float(chosen[0].timestamp-origin-start)
        right_gap = float(end-(chosen[-1].timestamp-origin))
        records.append({"window_id": identifier, "requested_start_sec": float(start),
                        "requested_end_sec": float(end), "start_sec": times[0], "end_sec": times[-1],
                        "frame_indices": [pool_indices[s.index] for s in chosen],
                        "frame_timestamps": times, "max_sample_gap_sec": max(gaps, default=0.0),
                        "left_boundary_gap_sec": left_gap, "right_boundary_gap_sec": right_gap,
                        "max_unobserved_interval_sec": max([left_gap, right_gap, *gaps]),
                        "available_distinct_frame_count": available, "requested_target_count": needed,
                        "sampling_capped": needed > max_frames_per_window})
    return {"kind": "focused_actual_video_frames_not_actions", "frames": [sampled[s.index] for s in ordered],
            "windows": records, "duration_sec": duration, "extraction_seconds": perf_counter()-started,
            "sampling_capped": any(row["sampling_capped"] for row in records),
            "duplicate_source_pts_removed": len(stamps)-len(unique),
            "settings": {"interval_sec": float(spacing), "max_frames_per_window": max_frames_per_window,
                         "max_windows": max_windows, "max_side": max_side, "max_window_sec": 3.0}}


def motion_peak_triple(frames, box, *, max_pair_gap_sec=.25):
    """Propose three neighboring sampled frames around the largest ROI change.

    ``box`` is a valid integer normalized [left, top, right, bottom] in 0..1000.
    Frames must contain real JPEGs and strictly increasing actual times. Corrupt
    data raises; resolution changes and gaps above ``max_pair_gap_sec`` break
    continuity and are reported. The largest eligible adjacent pair wins, ties
    favor the earlier pair. A third frame after the pair is preferred, otherwise
    a preceding one is used. All three are consecutive in the supplied sample
    list; they need not be consecutive frames in the original video.

    Scores are mean absolute grayscale differences in 0..255, with no inference,
    generated images, object labels or action threshold. Zero change yields no
    proposal. Lighting, reflections, camera motion and empty hands can trigger a
    proposal; absence of a proposal never proves absence of an action.
    """
    max_gap = float(_number(max_pair_gap_sec, "max_pair_gap_sec", positive=True))
    if not isinstance(frames, (list, tuple)) or len(frames) < 3:
        raise ValueError("At least three actual frames are required")
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(v) is not int or not 0 <= v <= 1000 for v in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("box must be a valid normalized integer rectangle")
    times, pictures, geometry = [], [], []
    for frame in frames:
        timestamp = float(_number(getattr(frame, "timestamp_sec", None), "frame timestamp"))
        if times and timestamp <= times[-1]:
            raise ValueError("Frame timestamps must strictly increase")
        width, height = getattr(frame, "width", None), getattr(frame, "height", None)
        _integer(width, "frame width", 1)
        _integer(height, "frame height", 1)
        if not isinstance(getattr(frame, "jpeg", None), (bytes, bytearray)):
            raise ValueError("Frame jpeg must contain JPEG bytes")
        with Image.open(BytesIO(frame.jpeg)) as picture:
            if picture.format != "JPEG" or picture.size != (width, height):
                raise ValueError("JPEG format or dimensions differ from frame metadata")
            picture.load()
            pixels = (math.floor(box[0]*width/1000), math.floor(box[1]*height/1000),
                      math.ceil(box[2]*width/1000), math.ceil(box[3]*height/1000))
            pictures.append(picture.crop(pixels).convert("L"))
        times.append(timestamp)
        geometry.append((width, height))
    scores = []
    for i in range(1, len(frames)):
        row = {"frame_indices": [i-1, i], "start_sec": times[i-1], "end_sec": times[i],
               "gap_sec": times[i]-times[i-1]}
        if row["gap_sec"] > max_gap + 1e-9:
            row["skipped_reason"] = "frame_gap"
        elif geometry[i-1] != geometry[i]:
            row["skipped_reason"] = "frame_geometry_changed"
        else:
            row["score"] = ImageStat.Stat(ImageChops.difference(pictures[i-1], pictures[i])).mean[0]
        scores.append(row)
    candidates = []
    for pair_index, row in enumerate(scores):
        if "score" not in row:
            continue
        right = pair_index+1
        if pair_index+1 < len(scores) and "score" in scores[pair_index+1]:
            indices = [right-1, right, right+1]
        elif pair_index > 0 and "score" in scores[pair_index-1]:
            indices = [right-2, right-1, right]
        else:
            row["selection_skipped_reason"] = "insufficient_three_frame_context"
            continue
        candidates.append((row["score"], pair_index, indices))
    result = {"kind": "motion_peak_candidate_not_action", "selected": False, "frame_indices": [],
              "frame_timestamps": [], "score": None, "peak_pair_indices": [], "scores": scores,
              "max_source_sample_gap_sec": max(b-a for a, b in zip(times, times[1:])),
              "policy": {"score": "mean_absolute_gray_change_0_255", "max_pair_gap_sec": max_gap,
                         "roi_normalized": list(box), "tie": "earlier_pair", "context": "after_then_before"}}
    if not candidates:
        result["reason"] = "no_three_frame_continuous_context"
        return result
    score, pair_index, indices = max(candidates, key=lambda item: (item[0], -item[1]))
    if score == 0:
        result.update(reason="no_pixel_change", score=0.0)
        return result
    result.update(selected=True, reason="largest_eligible_pair_change", score=score,
                  frame_indices=indices, frame_timestamps=[times[i] for i in indices],
                  peak_pair_indices=[pair_index, pair_index+1],
                  max_sample_gap_sec=max(times[b]-times[a] for a, b in zip(indices, indices[1:])))
    return result
