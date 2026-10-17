"""Sample real JPEG frames from a local video using presentation timestamps."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from os import PathLike
from pathlib import Path
from time import perf_counter

import av
from PIL import Image


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A sampled frame; its timestamp is relative to the first video frame."""

    timestamp_sec: float
    jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _FrameStamp:
    index: int
    timestamp: Fraction
    duration: Fraction | None


def _timestamp(frame: av.VideoFrame) -> Fraction:
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        raise ValueError("Video frame has no usable presentation timestamp (PTS).")
    return frame.pts * Fraction(frame.time_base)


def _scan(path: Path) -> tuple[list[_FrameStamp], Fraction | None]:
    stamps: list[_FrameStamp] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError("The file does not contain a video stream.")
        stream = container.streams.video[0]
        stream_duration = (
            stream.duration * Fraction(stream.time_base)
            if stream.duration is not None
            and stream.duration > 0
            and stream.time_base is not None
            and stream.time_base > 0
            else None
        )
        # Selecting this stream explicitly never decodes an audio stream.
        for index, frame in enumerate(container.decode(stream)):
            timestamp = _timestamp(frame)
            duration_ticks = getattr(frame, "duration", None)
            duration = (
                duration_ticks * Fraction(frame.time_base)
                if duration_ticks is not None and duration_ticks > 0
                else None
            )
            stamps.append(_FrameStamp(index, timestamp, duration))
    if not stamps:
        raise ValueError("The video does not contain any decodable frames.")
    # Variable intervals and reordered timestamps are handled on the PTS axis.
    stamps.sort(key=lambda stamp: (stamp.timestamp, stamp.index))
    return stamps, stream_duration


def _select(stamps: list[_FrameStamp], max_frames: int) -> list[_FrameStamp]:
    if len(stamps) <= max_frames:
        return stamps
    timestamps = [stamp.timestamp for stamp in stamps]
    start, end = timestamps[0], timestamps[-1]
    chosen = {0, len(stamps) - 1}
    for sample_index in range(1, max_frames - 1):
        target = start + (end - start) * Fraction(sample_index, max_frames - 1)
        right = bisect_left(timestamps, target)
        candidates = [index for index in (right - 1, right) if 0 <= index < len(stamps)]
        chosen.add(min(candidates, key=lambda index: (abs(timestamps[index] - target), index)))
    return [stamps[index] for index in sorted(chosen)]


def _duration(stamps: list[_FrameStamp], stream_duration: Fraction | None) -> float:
    first, last = stamps[0], stamps[-1]
    tail_duration = last.duration
    if tail_duration is None:
        # Use an observed PTS interval, never frame count divided by nominal FPS.
        previous = next(
            (stamp for stamp in reversed(stamps[:-1]) if stamp.timestamp < last.timestamp),
            None,
        )
        tail_duration = (
            last.timestamp - previous.timestamp
            if previous is not None
            else stream_duration or Fraction(0)
        )
    end = max(
        last.timestamp + tail_duration,
        max(stamp.timestamp + (stamp.duration or Fraction(0)) for stamp in stamps),
    )
    return float(end - first.timestamp)


def sample_video(
    path: str | PathLike[str], max_frames: int = 12, max_side: int = 768
) -> dict:
    """Return ``duration_sec``, ``frames`` and ``extraction_seconds``.

    Two sequential video-only passes select real frames nearest to evenly spaced
    PTS targets across the complete clip. The first and last presentation frames
    are always included; a one-frame video returns one frame. A large gap between
    timestamps can make multiple targets select the same frame, so fewer than
    ``max_frames`` frames may be returned. No frame is duplicated or interpolated.

    Timestamps use the first decoded presentation timestamp as zero. Duration
    includes the final frame's display duration; if that duration is unavailable,
    the last distinct PTS interval is an estimate of its duration. JPEG dimensions
    preserve aspect ratio and never exceed ``max_side``; images are not enlarged.

    ``max_frames`` must be an integer of at least 2, and ``max_side`` a positive
    integer. Missing PTS, empty/non-video files and decode failures raise errors;
    callers must not translate these errors into a ``no_violation`` verdict.
    """
    started = perf_counter()
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 2:
        raise ValueError("max_frames must be an integer of at least 2.")
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 1:
        raise ValueError("max_side must be a positive integer.")
    if not isinstance(path, (str, PathLike)) or not str(path).strip():
        raise ValueError("path must identify an existing local video file.")
    video_path = Path(path).expanduser().resolve(strict=True)
    if not video_path.is_file():
        raise ValueError("path must identify a file, not a directory.")

    stamps, stream_duration = _scan(video_path)
    selected = _select(stamps, max_frames)
    wanted = {stamp.index: stamp for stamp in selected}
    sampled: dict[int, VideoFrame] = {}
    origin = stamps[0].timestamp
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError("The file no longer contains a video stream.")
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index not in wanted:
                continue
            timestamp = _timestamp(frame)
            if timestamp != wanted[index].timestamp:
                raise RuntimeError("Video timestamps changed between sampling passes.")
            picture = frame.to_image().convert("RGB")
            picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            encoded = BytesIO()
            picture.save(encoded, format="JPEG", quality=90)
            sampled[index] = VideoFrame(
                timestamp_sec=float(timestamp - origin),
                jpeg=encoded.getvalue(),
                width=picture.width,
                height=picture.height,
            )
            if len(sampled) == len(wanted):
                break
    if len(sampled) != len(wanted):
        raise RuntimeError("Not all selected video frames could be decoded.")

    return {
        "duration_sec": _duration(stamps, stream_duration),
        "frames": [sampled[stamp.index] for stamp in selected],
        "extraction_seconds": perf_counter() - started,
    }
