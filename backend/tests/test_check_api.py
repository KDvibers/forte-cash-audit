"""Public API check diagnostics, using synthetic responses and no model or network."""
from __future__ import annotations

import base64
from itertools import count
import re
import sys

import pytest

from backend import check_api
from backend.mock_analyzer import analyze


KEY = "test-private-api-key-must-never-be-printed"
PRIVATE_DESCRIPTION = "private-observation-must-not-appear-in-console"
JOB_ID = "b0e53542-a057-4adf-9df4-eb58df757b45"
URL = "http://127.0.0.1:8000"


def done(scenario="one"):
    result = {"job_id": JOB_ID, "status": "done", **analyze(scenario, 0)}
    for event in result["violations"]:
        event["description"] = PRIVATE_DESCRIPTION
    return result


def mock_main(monkeypatch, *, mode="real"):
    monkeypatch.setattr(check_api, "read_key", lambda path: KEY)
    monkeypatch.setattr(check_api, "build_opener", lambda *args: object())
    monkeypatch.setattr(check_api, "wait_ready", lambda *args: {
        "mode": mode, "analyzer_ready": True, "message": PRIVATE_DESCRIPTION,
    })


@pytest.mark.parametrize(("expected", "actual"), [("real", "mock"), ("mock", "real")])
def test_mode_mismatch_stops_before_video_creation_or_submission(monkeypatch, capsys, expected, actual):
    mock_main(monkeypatch, mode=actual)
    monkeypatch.setattr(sys, "argv", ["check_api", "--expect-mode", expected])
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        pytest.fail("An incompatible server must not receive or generate a video")

    monkeypatch.setattr(check_api, "create_sample", forbidden)
    monkeypatch.setattr(check_api, "submit", forbidden)
    assert check_api.main() == 1
    assert called == []
    output = capsys.readouterr().out
    assert KEY not in output
    assert PRIVATE_DESCRIPTION not in output


@pytest.mark.parametrize("scenario", ["one", "clean"])
def test_poll_checks_and_accepts_complete_success_results(monkeypatch, capsys, scenario):
    payload = done(scenario)
    requests = []

    def response(opener, request, **kwargs):
        requests.append(request)
        return payload

    monkeypatch.setattr(check_api, "request_json", response)
    assert check_api.poll(object(), URL, KEY, JOB_ID) == 0
    assert len(requests) == 1
    assert requests[0].full_url == f"{URL}/api/v1/jobs/{JOB_ID}"
    assert requests[0].get_header("X-api-key") == KEY
    output = capsys.readouterr().out
    assert "done" in output
    assert payload["verdict"] in output
    assert KEY not in output
    assert PRIVATE_DESCRIPTION not in output
    if payload["violations"]:
        assert payload["violations"][0]["evidence_frames"][0]["image_base64"] not in output


@pytest.mark.parametrize("kind", ["missing-description", "invalid-time", "contradictory-verdict",
                                  "invalid-base64", "truncated-jpeg"])
def test_poll_rejects_malformed_results_and_undecodable_jpegs(monkeypatch, capsys, kind):
    payload = done()
    event = payload["violations"][0]
    if kind == "missing-description":
        del event["description"]
    elif kind == "invalid-time":
        event["timecode_sec"] = "1.5"
    elif kind == "contradictory-verdict":
        payload["verdict"] = "no_violation"
    elif kind == "invalid-base64":
        event["evidence_frames"][0]["image_base64"] = "!invalid!"
    else:
        # JPEG markers are valid base64 but are not a decodable JPEG image.
        event["evidence_frames"][0]["image_base64"] = base64.b64encode(b"\xff\xd8\xff\xd9").decode()
    monkeypatch.setattr(check_api, "request_json", lambda *args, **kwargs: payload)
    with pytest.raises(check_api.CheckError) as captured:
        check_api.poll(object(), URL, KEY, JOB_ID)
    output = capsys.readouterr().out + str(captured.value)
    assert KEY not in output
    assert PRIVATE_DESCRIPTION not in output
    assert "вердикт:" not in output


def test_poll_failed_is_distinct_from_no_violations(monkeypatch, capsys):
    monkeypatch.setattr(check_api, "request_json", lambda *args, **kwargs: {
        "job_id": JOB_ID, "status": "failed", "detail": KEY,
    })
    assert check_api.poll(object(), URL, KEY, JOB_ID) == 2
    output = capsys.readouterr().out
    assert "failed" in output
    assert "no_violation" not in output
    assert KEY not in output


@pytest.mark.parametrize(("status", "exit_code"), [("done", 0), ("failed", 2), ("malformed", 1)])
def test_main_reports_phase_timings_without_keys_or_evidence(monkeypatch, tmp_path, capsys, status, exit_code):
    mock_main(monkeypatch)
    video = tmp_path / "neutral.mov"
    # The network is stubbed: this test needs a path, not video decoding.
    video.write_bytes(b"neutral API-check test input")
    monkeypatch.setattr(sys, "argv", ["check_api", "--video", str(video), "--expect-mode", "real"])
    submitted = []

    def submit(opener, url, key, path):
        submitted.append((url, key, path))
        return JOB_ID

    monkeypatch.setattr(check_api, "submit", submit)
    payload = done()
    if status == "failed":
        payload = {"job_id": JOB_ID, "status": "failed", "detail": PRIVATE_DESCRIPTION}
    elif status == "malformed":
        payload["violations"][0]["evidence_frames"][0]["image_base64"] = "!invalid!"
    monkeypatch.setattr(check_api, "request_json", lambda *args, **kwargs: payload)
    clock = count(100, 0.25)
    monkeypatch.setattr(check_api.time, "monotonic", lambda: next(clock))
    assert check_api.main() == exit_code
    assert submitted == [(URL, KEY, video)]
    output = capsys.readouterr().out
    assert KEY not in output
    assert PRIVATE_DESCRIPTION not in output
    if payload.get("violations"):
        assert payload["violations"][0]["evidence_frames"][0]["image_base64"] not in output
    for label in ("Передача и приём файла", "Ожидание и проверка ответа",
                  "Всего от отправки до проверки ответа"):
        match = re.search(re.escape(label) + r": (\d+\.\d{2}) с\.", output)
        assert match is not None, f"Missing timing: {label}"
        assert float(match.group(1)) >= 0
