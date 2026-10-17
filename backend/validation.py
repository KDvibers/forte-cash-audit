from __future__ import annotations

import base64
import binascii
import io
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

from PIL import Image


class InvalidVideo(ValueError):
    pass


def validate_video(path: Path, ffprobe_path: str | None = None) -> dict:
    """Check a local MOV container and decode an actual frame with a bounded runtime.

    The analyzer must still handle errors later in the stream. This check cannot
    establish that every frame of the input is intact.
    """
    total = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(min(total, 1024 * 1024))
    offset = 0
    found = False
    while offset + 8 <= len(header):
        size, atom = struct.unpack(">I4s", header[offset:offset + 8])
        minimum = 8
        if size == 1:
            if offset + 16 > len(header):
                break
            size = struct.unpack(">Q", header[offset + 8:offset + 16])[0]
            minimum = 16
        elif size == 0:
            size = total - offset
        if size < minimum or offset + size > total:
            raise InvalidVideo("Файл повреждён: некорректный контейнер MOV.")
        if atom in {b"ftyp", b"moov", b"mdat"}:
            found = True
        if atom == b"ftyp" and size < 16:
            raise InvalidVideo("Файл повреждён: некорректный заголовок MOV.")
        offset += size
        if offset >= len(header):
            break
    if not found:
        raise InvalidVideo("Содержимое файла не похоже на видео MOV.")

    info = {}
    probe = ffprobe_path or shutil.which("ffprobe")
    if probe:
        try:
            run = subprocess.run([probe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
                                  "format=duration:stream=codec_type,codec_name", "-of", "json", str(path)],
                                 capture_output=True, timeout=10, check=False)
            info = json.loads(run.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            raise InvalidVideo("Не удалось прочитать видео. Проверьте файл и повторите загрузку.") from exc
        if run.returncode or not any(stream.get("codec_type") == "video" for stream in info.get("streams", [])):
            raise InvalidVideo("В файле не найден читаемый видеопоток.")

    # ffprobe only reads stream metadata; a successful probe alone is insufficient.
    try:
        import imageio_ffmpeg
        decode = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-nostdin", "-threads", "1",
             "-protocol_whitelist", "file,pipe", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
             "-vf", "scale=320:-2", "-threads", "1", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1"],
            capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired, ImportError, RuntimeError) as exc:
        raise InvalidVideo("Проверка видео недоступна: проверьте установку FFmpeg на сервере.") from exc
    if decode.returncode != 0 or not decode.stdout.startswith(b"\xff\xd8"):
        raise InvalidVideo("Видео повреждено или не содержит читаемых кадров.")
    return {**info, "validation": "first_frame_decode"}


def _time(value, field: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return value


def validate_result(result) -> dict:
    """Validate the agreed contract without inventing events or correcting a verdict."""
    if not isinstance(result, dict):
        raise ValueError("Analyzer result must be an object")
    verdict, events = result.get("verdict"), result.get("violations")
    if not isinstance(verdict, str) or verdict not in {"violation", "no_violation"} or not isinstance(events, list):
        raise ValueError("Invalid verdict or violations")
    if (verdict == "violation") != bool(events):
        raise ValueError("Verdict contradicts violations")
    if len(events) > 100:
        raise ValueError("Too many events in one result")
    verified = []
    total_image_bytes = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Event must be an object")
        for field in ("type", "description"):
            if not isinstance(event.get(field), str) or not event[field].strip() or len(event[field]) > 20000:
                raise ValueError(f"Invalid event {field}")
        timestamp = _time(event.get("timecode_sec"), "timecode_sec")
        frames = event.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise ValueError("Each event needs one to three evidence frames")
        output_frames = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Frame must be an object")
            frame_time = _time(frame.get("timestamp_sec"), "timestamp_sec")
            encoded = frame.get("image_base64")
            if not isinstance(encoded, str) or not encoded or len(encoded) > 8 * 1024 * 1024:
                raise ValueError("Invalid JPEG base64 length")
            try:
                raw = base64.b64decode(encoded, validate=True)
                total_image_bytes += len(raw)
                if total_image_bytes > 30 * 1024 * 1024:
                    raise ValueError("Evidence result is too large")
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "JPEG" or image.width * image.height > 40_000_000:
                        raise ValueError("Evidence must be a JPEG image")
                    image.verify()
                # JPEG verify() alone does not decode truncated pixel data.
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
            except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("Evidence is not valid JPEG base64") from exc
            output_frames.append({"timestamp_sec": frame_time, "image_base64": encoded})
        verified.append({"type": event["type"], "timecode_sec": timestamp, "description": event["description"],
                         "evidence_frames": output_frames})
    return {"verdict": verdict, "violations": verified}
