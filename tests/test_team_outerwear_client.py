import base64
from copy import deepcopy
from email.message import Message
import http.client
from io import BytesIO
import json
import socket
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from analyzer.observations import ObservationDeadline, ObservationError
from analyzer.outerwear import outerwear_policy_fingerprint
from analyzer.team_outerwear_client import TeamOuterwearWorkerClient, TeamOuterwearError, _NoRedirect
from analyzer.video import VideoFrame


RUNTIME = {"model_path": "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "model_alias": "forte-qwen",
           "model_ftype": "Q4_K - Medium", "build_info": "sha256:" + "a" * 64,
           "modalities": {"vision": True}, "total_slots": 1}
HOST = "team-test.tailtest.ts.net"
FRAME = VideoFrame(1.5, b"synthetic-jpeg-placeholder", 12, 8)


def record(frame_id="frame_01", timestamp=1.5):
    return {"frame_id": frame_id, "timestamp_sec": timestamp, "state": "no", "status": "ok", "errors": [],
            "policy_sha256": outerwear_policy_fingerprint(), "primary_state": "no", "elapsed_seconds": 0.03,
            "calls": [{"question": "removed_outerwear", "completed": True, "model_role": "local_activity_4b",
                       "request_seconds": 0.02, "response": {"finish_reason": "stop", "valid_json": True,
                                                               "observation": {"removed_outerwear": "no"}}}]}


class Response(BytesIO):
    def __init__(self, value, *, raw=False, content_type="application/json", length=None):
        data = value if raw else json.dumps(value).encode()
        super().__init__(data)
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(data) if length is None else length)
        self.body_timeouts = []
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=self.body_timeouts.append)))


class Peer:
    def __init__(self, mutate=None, error=None, response_factory=None):
        self.requests = []
        self.mutate, self.error, self.response_factory = mutate, error, response_factory

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        if request.data is None:
            value = {"status": "ok", "protocol_version": 1, "runtime": deepcopy(RUNTIME),
                     "observer_policy_sha256": outerwear_policy_fingerprint()}
        else:
            payload = json.loads(request.data)
            value = {key: payload[key] for key in ("protocol_version", "request_id", "frame_id", "timestamp_sec", "image_sha256")}
            value.update(runtime=deepcopy(RUNTIME), record=record(payload["frame_id"], payload["timestamp_sec"]))
        if self.mutate:
            self.mutate(value)
        return self.response_factory(value) if self.response_factory else Response(value)


def client(peer=None):
    value = TeamOuterwearWorkerClient(f"https://{HOST}:8443", expected_hostname=HOST, expected_runtime=RUNTIME)
    value.opener = peer or Peer()
    return value


def test_health_and_original_record_protocol_with_no_prompt_or_auth_secret():
    peer = Peer()
    value = client(peer)
    assert value.server_info() == RUNTIME
    result = value.observe(FRAME, "frame_01", deadline=time.perf_counter() + 30)
    assert result == record()
    assert len(peer.requests) == 2
    request, timeout = peer.requests[1]
    payload = json.loads(request.data)
    assert set(payload) == {"protocol_version", "request_id", "frame_id", "timestamp_sec", "image_base64",
                            "image_sha256", "width", "height", "budget_sec"}
    assert base64.b64decode(payload["image_base64"]) == FRAME.jpeg
    assert 0 < payload["budget_sec"] <= 30 and 0 < timeout <= 30
    assert request.full_url == f"https://{HOST}:8443/outerwear/check"
    assert request.get_header("Authorization") is None
    assert request.get_header("X-api-key") is None


@pytest.mark.parametrize("url", [f"http://{HOST}:8443", f"https://{HOST}", f"https://{HOST}:443",
                                  f"https://evil.tailtest.ts.net:8443", f"https://secret@{HOST}:8443",
                                  f"https://{HOST}:8443/path", f"https://{HOST}:8443/?x=1",
                                  f"https://{HOST}:8443/#fragment", "http://127.0.0.1:8093"])
def test_only_explicit_private_https_destination_allowed(url):
    with pytest.raises(ObservationError):
        TeamOuterwearWorkerClient(url, expected_hostname=HOST, expected_runtime=RUNTIME)


@pytest.mark.parametrize("field,value", [("status", "busy"), ("status", "quarantined"),
                                         ("protocol_version", True), ("observer_policy_sha256", "wrong")])
def test_unready_or_unknown_health_refuses(field, value):
    peer = Peer(mutate=lambda body: body.update({field: value}))
    with pytest.raises(ObservationError):
        client(peer).server_info()
    assert len(peer.requests) == 1


@pytest.mark.parametrize("mutation", [
    lambda body: body.update(request_id="0" * 32),
    lambda body: body.update(frame_id="different"),
    lambda body: body.update(timestamp_sec=True),
    lambda body: body.update(image_sha256="0" * 64),
    lambda body: body.update(extra="unexpected"),
    lambda body: body["runtime"].update(total_slots=2),
    lambda body: body["record"].update(state="yes"),
    lambda body: body["record"].update(status="partial"),
    lambda body: body["record"]["calls"][0].update(completed=False),
    lambda body: body["record"]["calls"].append(deepcopy(body["record"]["calls"][0])),
])
def test_record_and_envelope_tampering_is_not_usable(mutation):
    peer = Peer(mutate=mutation)
    with pytest.raises(ObservationError):
        client(peer).observe(FRAME, "frame_01", deadline=time.perf_counter() + 10)
    assert len(peer.requests) == 1


@pytest.mark.parametrize("error", [TimeoutError("timeout"), urllib.error.URLError("no route"),
                                   urllib.error.HTTPError("https://unused", 503, "unavailable", {}, None)])
def test_unknown_completion_never_retries(error):
    peer = Peer(error=error)
    with pytest.raises((ObservationError, ObservationDeadline)):
        client(peer).observe(FRAME, "frame_01", deadline=time.perf_counter() + 10)
    assert len(peer.requests) == 1


def test_expired_deadline_never_dispatches():
    peer = Peer()
    with pytest.raises(ObservationDeadline):
        client(peer).observe(FRAME, "frame_01", deadline=time.perf_counter() - 1)
    assert peer.requests == []


@pytest.mark.parametrize("factory", [
    lambda body: Response(body, content_type="text/html"),
    lambda body: Response(body, length=999999),
    lambda body: Response(body, length=1),
    lambda body: Response(b'{"status":"ok","status":"busy"}', raw=True),
    lambda body: Response(b'{"time":NaN}', raw=True),
    lambda body: Response(b'x' * 65537, raw=True),
])
def test_malformed_or_unbounded_response_rejected(factory):
    with pytest.raises(ObservationError):
        client(Peer(response_factory=factory)).server_info()


def test_redirect_handler_never_forwards_request():
    with pytest.raises(TeamOuterwearError):
        _NoRedirect().redirect_request(None, None, 307, "redirect", {}, "https://other.invalid")


class SocketBodyPeer:
    """Actual HTTPResponse/socket reads; no model, TLS peer or external network."""

    def __init__(self, pieces, *, chunked=False):
        self.pieces = pieces
        self.chunked = chunked
        self.requests = []
        self.stop = threading.Event()
        self.reader, self.writer = socket.socketpair()
        self.thread = None

    def open(self, request, *, timeout):
        self.requests.append(request)
        self.reader.settimeout(timeout)
        framing = (b"Transfer-Encoding: chunked\r\n" if self.chunked else
                   f"Content-Length: {sum(len(data) for _, data in self.pieces)}\r\n".encode())

        def send():
            try:
                self.writer.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" + framing + b"\r\n")
                for delay, data in self.pieces:
                    if self.stop.wait(delay):
                        return
                    self.writer.sendall(data)
            except OSError:
                pass  # Client may close immediately after the deadline.

        self.thread = threading.Thread(target=send, daemon=True)
        self.thread.start()
        response = http.client.HTTPResponse(self.reader)
        response.begin()
        # urllib closes its original socket ownership after returning headers;
        # HTTPResponse's SocketIO keeps it alive only until the body is read.
        self.reader.close()
        return response

    def close(self):
        self.stop.set()
        self.reader.close()
        self.writer.close()
        if self.thread is not None:
            self.thread.join(1)
            assert not self.thread.is_alive()


@pytest.mark.parametrize("pieces", [
    [(0.5, b'{"ok":true}')],
    [(0.04, bytes([value])) for value in b'{"ok":true}'],
])
def test_real_socket_slow_body_returns_at_deadline_without_second_post(pieces):
    peer = SocketBodyPeer(pieces)
    value = client(peer)
    started = time.perf_counter()
    try:
        with pytest.raises(ObservationDeadline):
            value._request("/outerwear/check", deadline=started + 0.12, payload={"synthetic": True})
        elapsed = time.perf_counter() - started
        # Both former read() cases take >=0.44 s or 0.5 s. Permit normal
        # scheduler jitter, while still proving we did not await the full body.
        assert elapsed < 0.30
        assert len(peer.requests) == 1
        assert peer.requests[0].method == "POST"
    finally:
        peer.close()


def test_real_socket_fragmented_body_completes_before_deadline():
    peer = SocketBodyPeer([(0, b'{"ok":'), (0.01, b'true}')])
    try:
        assert client(peer)._request("/outerwear/check", deadline=time.perf_counter() + 1,
                                     payload={"synthetic": True}) == {"ok": True}
        assert len(peer.requests) == 1
    finally:
        peer.close()


def test_chunked_body_is_refused_before_waiting_for_its_framing():
    peer = SocketBodyPeer([(0.5, b'2\r\n{}\r\n0\r\n\r\n')], chunked=True)
    started = time.perf_counter()
    try:
        with pytest.raises(TeamOuterwearError, match="Chunked"):
            client(peer)._request("/outerwear/check", deadline=started + 1, payload={"synthetic": True})
        assert time.perf_counter() - started < 0.30
        assert len(peer.requests) == 1
    finally:
        peer.close()


def test_unfamiliar_body_adapter_is_refused_instead_of_using_read():
    def unsupported(body):
        response = Response(body)
        response.fp = None
        return response
    peer = Peer(response_factory=unsupported)
    with pytest.raises(TeamOuterwearError, match="Unsupported worker body transport"):
        client(peer).server_info()
    assert len(peer.requests) == 1


def test_each_body_read_receives_the_remaining_timeout_and_late_bytes_are_rejected(monkeypatch):
    clock = [0.0]
    response = Response({})
    calls = []
    def read1(amount):
        calls.append(amount)
        clock[0] += 2.0
        return b"{" if len(calls) == 1 else b"}"
    response.read1 = read1
    monkeypatch.setattr("analyzer.team_outerwear_client.time.perf_counter", lambda: clock[0])
    peer = Peer(response_factory=lambda body: response)
    with pytest.raises(ObservationDeadline):
        client(peer).observe(FRAME, "frame_01", deadline=3.0)
    assert response.body_timeouts == [3.0, 1.0]
    assert len(calls) == 2 and len(peer.requests) == 1
