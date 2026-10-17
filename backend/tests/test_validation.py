"""Reject misleading analyzer payloads before a job can become done."""
from __future__ import annotations

import base64
from copy import deepcopy
import io

from PIL import Image
import pytest

from backend.mock_analyzer import analyze
from backend.validation import validate_result


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
