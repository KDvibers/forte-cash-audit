"""Acceptance-client tests use synthetic fixtures, not an actual real model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from email.message import Message
import json
import threading

import pytest

from backend import check_real
from backend.mock_analyzer import analyze

ORIGIN = "https://acceptance.example"
KEY = "secret-key-never-report-or-print-this"
COOKIE = "cash_audit_session=private-cookie; Path=/ui; Secure; HttpOnly; SameSite=Strict"


def reply(payload, status=200, *, session=False):
    headers = Message()
    headers["Content-Type"] = "application/json"
    if session:
        headers["Set-Cookie"] = COOKIE
    return check_real.Reply(status, headers, json.dumps(payload).encode())


class FixtureClient:
    def __init__(self, *, scenario="one", mode="real", ready=True, status="done", corrupt=False,
                 clock=None, upload_time=0, poll_time=0, recovery_wrong=False):
        self.calls = []
        self.mode, self.ready, self.status = mode, ready, status
        self.result = {"job_id": "fixture-job", "status": status, **analyze(scenario, 0)}
        if corrupt:
            self.result["violations"][0]["evidence_frames"][0]["image_base64"] = "invalid"
        self.clock, self.upload_time, self.poll_time = clock, upload_time, poll_time
        self.recovery_wrong = recovery_wrong
        self.upload_size = 0

    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        headers = kwargs.get("headers", {})
        if path == "/health":
            return reply({"mode": self.mode, "status": "ok", "analyzer_ready": self.ready})
        if path == "/ui/session":
            return reply({"mode": self.mode, "analyzer_ready": self.ready, "csrf_token": "a" * 40}, session=True)
        if method == "POST":
            if path.startswith("/api") and headers.get("X-API-Key") != KEY:
                return reply({}, 401)
            if path.startswith("/ui") and (headers.get("Origin") != ORIGIN or not headers.get("X-CSRF-Token")):
                return reply({}, 403)
            body = b"".join(kwargs["chunks"])
            self.upload_size = len(body)
            if self.clock:
                self.clock.value += self.upload_time
            return reply({"job_id": "fixture-job"})
        if path.startswith("/ui/recovery/"):
            return reply({"job_id": "other" if self.recovery_wrong else "fixture-job", "status": "done",
                          "mode": "real", "file_size": 7})
        if "/jobs/" in path:
            if self.clock:
                self.clock.value += self.poll_time
            return reply(self.result)
        raise AssertionError("Unexpected fixture route")


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "private_filename.mov"
    path.write_bytes(b"fixture")
    return path


def run(video, client, **kwargs):
    return check_real.check_one(ORIGIN, KEY, video, check_real.file_metadata(video),
                                client_factory=lambda origin: client, **kwargs)


@pytest.mark.parametrize(("scenario", "count", "frames"), [("clean", 0, 0), ("one", 1, 3), ("multiple", 2, 4)])
def test_valid_contract_counts_without_private_evidence(video, scenario, count, frames):
    client = FixtureClient(scenario=scenario)
    result = run(video, client)
    assert result["outcome"] == "passed"
    assert result["events"] == count and result["jpeg_frames"] == frames
    assert result["within_300_seconds"] is True
    assert not {"failure_phase", "transport_phase", "body_bytes_handed_to_socket",
                "declared_body_bytes"}.intersection(result)
    saved = json.dumps(result)
    assert KEY not in saved and str(video) not in saved and video.name not in saved
    assert "image_base64" not in saved and "description" not in saved
    assert result["job_id"] == "fixture-job"  # Ignored local report only: queue audit lookup.
    assert len([call for call in client.calls if call[0] == "POST"]) == 1


def test_positive_case_cannot_pass_without_events(video):
    result = run(video, FixtureClient(scenario="clean"), require_events=True)
    assert result["outcome"] == "expected_events_missing"
    assert result["jpeg_frames"] == 0


@pytest.mark.parametrize("header", ["X-Cash-Audit-Mode", "X-Cash-Audit-Job-Mode"])
@pytest.mark.parametrize("stage", ["upload", "poll"])
def test_mode_change_after_preflight_cannot_pass(video, header, stage):
    class ChangedModeClient(FixtureClient):
        def request(self, method, path, **kwargs):
            response = super().request(method, path, **kwargs)
            if (stage == "upload" and method == "POST") or (stage == "poll" and "/jobs/" in path):
                response.headers[header] = "mock"
            return response
    client = ChangedModeClient()
    assert run(video, client)["outcome"] == "wrong_mode"
    assert len([call for call in client.calls if call[0] == "POST"]) == 1


def test_optional_real_headers_are_accepted(video):
    class RealHeaders(FixtureClient):
        def request(self, *args, **kwargs):
            response = super().request(*args, **kwargs)
            response.headers["X-Cash-Audit-Mode"] = "real"
            response.headers["X-Cash-Audit-Job-Mode"] = "real"
            return response
    assert run(video, RealHeaders())["outcome"] == "passed"


@pytest.mark.parametrize(("mode", "ready", "outcome"), [("mock", True, "wrong_mode"), ("real", False, "not_ready")])
def test_preflight_stops_upload(video, mode, ready, outcome):
    client = FixtureClient(mode=mode, ready=ready)
    assert run(video, client)["outcome"] == outcome
    assert all(method == "GET" for method, _, _ in client.calls)


@pytest.mark.parametrize("kind", ["jpeg", "timecode", "frame-time", "identity", "status", "failed"])
def test_invalid_contract_and_failed_are_not_success(video, kind):
    client = FixtureClient(corrupt=kind == "jpeg")
    if kind == "timecode":
        client.result["violations"][0]["timecode_sec"] = -1
    if kind == "frame-time":
        client.result["violations"][0]["evidence_frames"][0]["timestamp_sec"] = "0.1"
    if kind == "identity":
        client.result["job_id"] = "other"
    if kind in {"failed", "status"}:
        client.result["status"] = "failed" if kind == "failed" else "other"
    result = run(video, client)
    assert result["outcome"] == ("analysis_failed" if kind == "failed" else "invalid_result")
    assert result["within_300_seconds"] is False


class Clock:
    value = 10.0
    def now(self):
        return self.value


@pytest.mark.parametrize(("upload", "poll"), [(301, 0), (270, 31), (0, 301)])
def test_total_deadline_counts_upload_and_rejects_late_done(video, monkeypatch, upload, poll):
    clock = Clock()
    monkeypatch.setattr(check_real.time, "monotonic", clock.now)
    client = FixtureClient(clock=clock, upload_time=upload, poll_time=poll)
    result = run(video, client)
    assert result["outcome"] == "timeout"
    assert result["total_seconds"] >= 300
    assert result["within_300_seconds"] is False
    assert len([call for call in client.calls if call[0] == "POST"]) == 1


def test_jpeg_validation_time_is_inside_deadline(video, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(check_real.time, "monotonic", clock.now)
    original = check_real.validate_result
    def slow_validate(payload):
        value = original(payload)
        clock.value += 301
        return value
    monkeypatch.setattr(check_real, "validate_result", slow_validate)
    result = run(video, FixtureClient())
    assert result["outcome"] == "timeout"


@pytest.mark.parametrize("wrong", [False, True])
def test_ui_recovery_uses_same_session_gets_and_never_reuploads(video, wrong):
    client = FixtureClient(recovery_wrong=wrong)
    result = run(video, client, channel="ui")
    assert result["outcome"] == ("recovery_error" if wrong else "passed")
    posts = [call for call in client.calls if call[0] == "POST"]
    assert len(posts) == 1
    identifier = posts[0][2]["headers"]["X-Upload-ID"]
    assert any(path == "/ui/recovery/" + identifier for _, path, _ in client.calls)
    assert all("X-API-Key" not in options.get("headers", {}) for _, _, options in client.calls)


def test_preflight_checks_no_key_wrong_key_origin_and_csrf():
    client = FixtureClient()
    check_real.access_checks(client, ORIGIN)
    posts = [call for call in client.calls if call[0] == "POST"]
    assert len(posts) == 4
    assert all(call[2]["length"] == 0 for call in posts)
    assert client.upload_size == 0


def test_two_concurrent_checks_have_distinct_uploads_and_deadlines(video):
    clients = [FixtureClient(), FixtureClient()]
    gate = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, video, client, gate=gate) for client in clients]
        assert all(future.result()["outcome"] == "passed" for future in futures)
    deadlines = [next(opts["deadline"] for method, _, opts in client.calls if method == "POST") for client in clients]
    assert len(deadlines) == 2 and all(value > 300 for value in deadlines)
    assert all(len([call for call in client.calls if call[0] == "POST"]) == 1 for client in clients)


def test_reports_cannot_be_written_outside_ignored_directory(tmp_path):
    with pytest.raises(check_real.AcceptanceError):
        check_real.report_path(tmp_path / "report.json")
    assert check_real.report_path(None).is_relative_to(check_real.REPORT_ROOT)


def test_cli_saves_metadata_only_and_does_not_print_key_or_path(monkeypatch, tmp_path, video, capsys):
    key_file = tmp_path / "private-key.txt"
    key_file.write_text(KEY)
    report = tmp_path / "report.json"
    monkeypatch.setattr(check_real, "report_path", lambda value: report)
    monkeypatch.setattr(check_real, "HTTPSClient", lambda origin: FixtureClient())
    original = check_real.check_one
    monkeypatch.setattr(check_real, "check_one", lambda *args, **kwargs:
                        original(*args, **kwargs, client_factory=lambda origin: FixtureClient()))
    assert check_real.main(["--base-url", ORIGIN, "--key-file", str(key_file), "--video", str(video)]) == 0
    output = capsys.readouterr().out
    assert KEY not in output and str(video) not in output and "fixture-job" not in output
    saved = json.loads(report.read_text())
    assert saved["quality_evaluated"] is False
    assert saved["checks"][0]["outcome"] == "passed"
    assert saved["checks"][0]["job_id"] == "fixture-job"
    assert "image_base64" not in report.read_text() and KEY not in report.read_text()


def test_cli_refuses_existing_report_before_network(monkeypatch, tmp_path, video):
    key_file = tmp_path / "private-key.txt"
    key_file.write_text(KEY)
    report = tmp_path / "report.json"
    report.write_text("previous")
    monkeypatch.setattr(check_real, "report_path", lambda value: report)
    monkeypatch.setattr(check_real, "HTTPSClient", lambda *args: pytest.fail("No upload before report validation"))
    assert check_real.main(["--base-url", ORIGIN, "--key-file", str(key_file), "--video", str(video)]) == 1
    assert report.read_text() == "previous"


def test_main_requires_explicit_video_and_never_generates_it():
    with pytest.raises(SystemExit):
        check_real.main(["--base-url", ORIGIN, "--key-file", "missing.txt"])


def test_https_transport_uses_original_host_certificate_check_and_no_redirect(monkeypatch):
    connections = []
    class FakeSocket:
        def settimeout(self, value):
            assert 0 < value <= 15
    class FakeResponse:
        status = 302
        headers = Message()
        headers["Location"] = "https://other.invalid/"
        def read1(self, size):
            return b""
    class FakeConnection:
        def __init__(self, host, port, *, timeout, context):
            assert host == "acceptance.example" and port == 443
            assert context.check_hostname
            self.sock = FakeSocket()
            self.sent = []
            connections.append(self)
        def connect(self): pass
        def putrequest(self, method, path, **kwargs): self.sent.append((method, path))
        def putheader(self, name, value): self.sent.append((name, value))
        def endheaders(self): pass
        def getresponse(self): return FakeResponse()
        def close(self): self.closed = True
    monkeypatch.setattr(check_real.http.client, "HTTPSConnection", FakeConnection)
    response = check_real.HTTPSClient(ORIGIN).request("GET", "/health")
    assert response.status == 302 and len(connections) == 1
    assert connections[0].closed
    with pytest.raises(check_real.AcceptanceError, match="HTTP 302"):
        check_real.get_json(response)


def test_per_chunk_deadline_stops_drip_response(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(check_real.time, "monotonic", clock.now)
    class FakeSocket:
        def settimeout(self, value): pass
    class FakeResponse:
        status = 200
        headers = Message()
        def read1(self, size):
            clock.value += 6
            return b"x"
    class FakeConnection:
        def __init__(self, *args, **kwargs): self.sock = FakeSocket()
        def connect(self): pass
        def putrequest(self, *args, **kwargs): pass
        def putheader(self, *args): pass
        def endheaders(self): pass
        def getresponse(self): return FakeResponse()
        def close(self): pass
    monkeypatch.setattr(check_real.http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(check_real.AcceptanceError) as failure:
        check_real.HTTPSClient(ORIGIN).request("GET", "/health", deadline=20)
    assert failure.value.code == "timeout"


@pytest.mark.parametrize("phase", ["preflight", "upload", "poll", "recovery"])
def test_failure_reports_current_phase_without_secrets_or_previous_progress(video, phase):
    class FailingClient(FixtureClient):
        def request(self, method, path, **kwargs):
            should_fail = ((phase == "preflight" and path == "/health")
                           or (phase == "upload" and method == "POST")
                           or (phase == "poll" and "/jobs/" in path)
                           or (phase == "recovery" and "/recovery/" in path))
            if should_fail:
                self.transport_progress = {
                    "transport_phase": "wait_headers", "body_bytes_handed_to_socket": 3,
                    "declared_body_bytes": 7, "headers": {"X-API-Key": KEY}, "body": COOKIE,
                }
                raise OSError(KEY)
            return super().request(method, path, **kwargs)

    client = FailingClient()
    result = run(video, client, channel="ui" if phase == "recovery" else "api")
    assert result["outcome"] == "network_error"
    assert result["failure_phase"] == phase
    assert result["transport_phase"] == "wait_headers"
    assert result["body_bytes_handed_to_socket"] == 3 and result["declared_body_bytes"] == 7
    assert KEY not in json.dumps(result) and COOKIE not in json.dumps(result)
    assert "headers" not in result and "body" not in result
    assert len([call for call in client.calls if call[0] == "POST"]) <= 1


@pytest.mark.parametrize("failure,expected_bytes", [
    ("connect", 0), ("send_body", 3), ("wait_headers", 7), ("read_response", 7),
])
def test_transport_counts_only_completed_body_sends_and_identifies_blocked_stage(monkeypatch, failure, expected_bytes):
    connections = []

    class FakeSocket:
        def settimeout(self, value):
            assert 0 < value <= 15

    class FakeResponse:
        status = 200
        headers = Message()
        def read1(self, size):
            raise TimeoutError("synthetic response timeout")

    class FakeConnection:
        def __init__(self, *args, **kwargs):
            self.sock = FakeSocket()
            self.send_count = 0
            self.response_count = 0
            self.closed = False
            connections.append(self)
        def connect(self):
            if failure == "connect":
                raise TimeoutError("synthetic connect timeout")
        def putrequest(self, *args, **kwargs): pass
        def putheader(self, *args): pass
        def endheaders(self): pass
        def send(self, chunk):
            self.send_count += 1
            if failure == "send_body" and self.send_count == 2:
                # Some bytes may have reached the kernel before failure; they
                # cannot be counted as a completed send without an ACK contract.
                raise TimeoutError("synthetic partial send")
        def getresponse(self):
            self.response_count += 1
            if failure == "wait_headers":
                raise TimeoutError("synthetic header timeout")
            return FakeResponse()
        def close(self):
            self.closed = True

    monkeypatch.setattr(check_real.http.client, "HTTPSConnection", FakeConnection)
    client = check_real.HTTPSClient(ORIGIN)
    with pytest.raises(check_real.AcceptanceError) as caught:
        client.request("POST", "/api/v1/analyze", headers={"X-API-Key": KEY},
                       chunks=iter([b"abc", b"defg"]), length=7)
    assert caught.value.code == "timeout"
    assert client.transport_progress == {
        "transport_phase": failure, "body_bytes_handed_to_socket": expected_bytes, "declared_body_bytes": 7,
    }
    assert len(connections) == 1 and connections[0].closed
    assert connections[0].response_count == int(failure in {"wait_headers", "read_response"})


def test_stale_preflight_progress_is_not_attached_to_local_upload_failure(video, monkeypatch):
    class Client(FixtureClient):
        def request(self, *args, **kwargs):
            self.transport_progress = {"transport_phase": "read_response", "body_bytes_handed_to_socket": 0,
                                       "declared_body_bytes": 0}
            return super().request(*args, **kwargs)
    monkeypatch.setattr(check_real, "multipart", lambda video: (_ for _ in ()).throw(OSError("synthetic file error")))
    result = run(video, Client())
    assert result["failure_phase"] == "upload"
    assert "transport_phase" not in result and "body_bytes_handed_to_socket" not in result
