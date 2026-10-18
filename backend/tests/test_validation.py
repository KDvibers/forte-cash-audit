"""Reject misleading analyzer payloads before a job can become done."""
from __future__ import annotations

import base64
from copy import deepcopy
import io
import struct
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

from backend.mock_analyzer import analyze
from backend import validation
from backend.validation import validate_result, validate_video, VideoValidationTimeout


@pytest.fixture
def result():
    return analyze("one", 0)


@pytest.mark.parametrize("bad_time", [True, "1.5", -1, float("nan"), float("inf")])
def test_event_and_frame_times_are_finite_nonnegative_numbers(result, bad_time):
    for target, field in [(result["violations"][0], "timecode_sec"),
                          (result["violations"][0]["evidence_frames"][0], "timestamp_sec")]:
        original = target[field]
        target[field] = bad_time
        with pytest.raises(ValueError):
            validate_result(result)
        target[field] = original


@pytest.mark.parametrize("verdict,events", [
    ("unknown", []), ("violation", []), ("no_violation", "keep"),
])
def test_verdict_matches_events(result, verdict, events):
    result["verdict"] = verdict
    if events != "keep":
        result["violations"] = events
    with pytest.raises(ValueError):
        validate_result(result)


@pytest.mark.parametrize("case", ["invalid-base64", "jpeg-header-only", "png", "data-url", "no-frames", "too-many-frames"])
def test_evidence_must_be_one_to_three_decodable_jpegs(result, case):
    frames = result["violations"][0]["evidence_frames"]
    if case == "invalid-base64":
        frames[0]["image_base64"] = "!invalid!"
    elif case == "jpeg-header-only":
        frames[0]["image_base64"] = base64.b64encode(b"\xff\xd8\xff\xd9").decode()
    elif case == "png":
        output = io.BytesIO()
        Image.new("RGB", (2, 2)).save(output, format="PNG")
        frames[0]["image_base64"] = base64.b64encode(output.getvalue()).decode()
    elif case == "data-url":
        frames[0]["image_base64"] = "data:image/jpeg;base64," + frames[0]["image_base64"]
    elif case == "no-frames":
        frames.clear()
    else:
        frames.append(deepcopy(frames[0]))
    with pytest.raises(ValueError):
        validate_result(result)


def test_new_event_types_and_complete_description_are_preserved(result):
    event = result["violations"][0]
    event["type"] = "Новая категория анализатора / не из списка интерфейса"
    event["description"] = "Наблюдение и проверенная нормативная ссылка.\n" * 100
    validated = validate_result(result)
    assert validated["violations"][0]["type"] == event["type"]
    assert validated["violations"][0]["description"] == event["description"]


@pytest.fixture
def mov_header(tmp_path):
    # Container marker only: subprocess results below are synthetic test doubles.
    path = tmp_path / "synthetic.mov"
    path.write_bytes(struct.pack(">I4s8s", 16, b"ftyp", b"qt      "))
    return path


def test_probe_and_decode_share_remaining_deadline(mov_header, monkeypatch):
    clock, timeouts = [100.0], []
    monkeypatch.setattr(validation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(validation.shutil, "which", lambda name: "synthetic-ffprobe")

    def run(command, *, timeout, **kwargs):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            clock[0] += 9
            return SimpleNamespace(returncode=0, stdout=b'{"streams":[{"codec_type":"video"}]}')
        clock[0] += 2
        return SimpleNamespace(returncode=0, stdout=b"\xff\xd8synthetic")

    monkeypatch.setattr(validation.subprocess, "run", run)
    assert validate_video(mov_header, deadline_monotonic=112)["validation"] == "first_frame_decode"
    assert timeouts == [10, 3]


@pytest.mark.parametrize("phase", ["before", "probe", "decode"])
def test_validation_deadline_expiry_never_produces_success(mov_header, monkeypatch, phase):
    clock, calls = [100.0], []
    deadline = 99 if phase == "before" else 105
    monkeypatch.setattr(validation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(validation.shutil, "which", lambda name: "synthetic-ffprobe")

    def run(command, *, timeout, **kwargs):
        calls.append(timeout)
        if phase == "probe" or len(calls) == 2:
            clock[0] = 105
            raise subprocess.TimeoutExpired(command, timeout)
        clock[0] += 4
        return SimpleNamespace(returncode=0, stdout=b'{"streams":[{"codec_type":"video"}]}')

    monkeypatch.setattr(validation.subprocess, "run", run)
    with pytest.raises(VideoValidationTimeout):
        validate_video(mov_header, deadline_monotonic=deadline)
    assert len(calls) == {"before": 0, "probe": 1, "decode": 2}[phase]
    if phase == "decode":
        assert calls == [5, 1]


def test_validation_without_deadline_keeps_previous_phase_limits(mov_header, monkeypatch):
    timeouts = []
    monkeypatch.setattr(validation.shutil, "which", lambda name: "synthetic-ffprobe")

    def run(command, *, timeout, **kwargs):
        timeouts.append(timeout)
        return SimpleNamespace(returncode=0, stdout=(b'{"streams":[{"codec_type":"video"}]}'
                               if len(timeouts) == 1 else b"\xff\xd8synthetic"))

    monkeypatch.setattr(validation.subprocess, "run", run)
    validate_video(mov_header)
    assert timeouts == [10, 20]
