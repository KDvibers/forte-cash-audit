"""Public readiness checks use synthetic fixtures and local ASGI, never the internet."""
from __future__ import annotations

from email.message import Message
from http.cookiejar import CookieJar
import io
import json
import time
from urllib.error import HTTPError, URLError

from fastapi.testclient import TestClient
import pytest

from backend import check_public
from backend.config import Settings
from backend.main import create_app
from backend.mock_analyzer import analyze


ORIGIN = "https://public.example"
KEY = "private-public-check-key-not-for-console"


def headers(media_type="application/json", cookie=None):
    result = Message()
    result["Content-Type"] = media_type
    if cookie:
        result["Set-Cookie"] = cookie
    return result


class Response(io.BytesIO):
    def __init__(self, body, response_headers=None, status=200):
        super().__init__(body)
        self.headers = response_headers or headers()
        self.status = status

    def info(self):
        return self.headers


class StubOpener:
    def __init__(self, body=None, response_headers=None, error=None):
        self.body = body if body is not None else b"{}"
        self.headers = response_headers
        self.error = error
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        if self.error:
            raise self.error
        return Response(self.body, self.headers)


@pytest.mark.parametrize("value", [
    "http://public.example", "https://public.example/path", "https://user:secret@public.example",
    "https://public.example?key=secret", "https://public.example#secret", " https://public.example",
    "https://public.example\n", "https://public.example:99999", "https://public.example:",
    "https://public.example\\evil", "https://127.0.0.1", "https://[::1]",
])
def test_rejects_non_public_or_ambiguous_origins(value):
    with pytest.raises(check_public.CheckError):
        check_public.https_origin(value)


def test_normalizes_https_default_port():
    assert check_public.https_origin("https://PUBLIC.example:443/") == ORIGIN


@pytest.mark.parametrize("health", [
    {"mode": "real", "analyzer_ready": True, "status": "ok"},
    {"mode": "mock", "analyzer_ready": False, "status": "degraded"},
])
def test_incompatible_health_stops_before_posts_or_video(monkeypatch, health):
    api = StubOpener(json.dumps(health).encode())
    monkeypatch.setattr(check_public, "create_sample", lambda *args: pytest.fail("No video may be created"))
    with pytest.raises(check_public.CheckError):
        check_public.run_check(ORIGIN, "mock", KEY, api, StubOpener())
    assert len(api.requests) == 1
    assert api.requests[0].get_method() == "GET"


@pytest.mark.parametrize("cookie", [
    "cash_audit_session=token; Path=/ui; HttpOnly; SameSite=Strict",
    "cash_audit_session=token; Path=/ui; Secure; SameSite=Strict",
    "cash_audit_session=token; Path=/ui; Secure; HttpOnly; SameSite=Lax",
    "cash_audit_session=token; Path=/; Secure; HttpOnly; SameSite=Strict",
    "cash_audit_session=token; Path=/ui; Secure; HttpOnly; SameSite=Strict; Domain=public.example",
])
def test_requires_secure_restricted_cookie(cookie):
    with pytest.raises(check_public.CheckError):
        check_public.check_cookie(headers(cookie=cookie))


def test_valid_cookie_attributes():
    check_public.check_cookie(headers(cookie="cash_audit_session=token; Path=/ui; Secure; HttpOnly; SameSite=Strict"))


def test_response_size_and_content_type_are_bounded():
    with pytest.raises(check_public.CheckError):
        check_public.fetch(StubOpener(b"x" * 12), check_public.Request(ORIGIN), limit=10)
    with pytest.raises(check_public.CheckError):
        check_public.json_response(StubOpener(b"{}", headers("text/html")), check_public.Request(ORIGIN))


def test_upload_network_failure_does_not_retry(tmp_path):
    video = tmp_path / "neutral.mov"
    video.write_bytes(b"neutral")
    opener = StubOpener(error=URLError(KEY))
    with pytest.raises(URLError):
        check_public.upload(opener, ORIGIN, video, {"X-API-Key": KEY})
    assert len(opener.requests) == 1


@pytest.mark.parametrize("payload", [
    {"job_id": "other", "status": "done", "verdict": "no_violation", "violations": []},
    {"job_id": "own", "status": "failed", "detail": KEY},
    {"job_id": "own", "status": "unexpected"},
    {"job_id": "own", "status": "done", "verdict": "violation", "violations": []},
])
def test_poll_rejects_wrong_identity_failure_and_bad_contract(payload):
    with pytest.raises(check_public.CheckError):
        check_public.poll(StubOpener(json.dumps(payload).encode()), ORIGIN, "own", {})


def test_poll_rejects_invalid_jpeg():
    payload = {"job_id": "own", "status": "done", **analyze("one", 0)}
    payload["violations"][0]["evidence_frames"][0]["image_base64"] = "invalid"
    with pytest.raises(check_public.CheckError, match="JPEG"):
        check_public.poll(StubOpener(json.dumps(payload).encode()), ORIGIN, "own", {})


def test_cli_sanitizes_errors_and_uses_no_redirect_proxy_or_shared_cookie(monkeypatch, tmp_path, capsys):
    key_file = tmp_path / "key.txt"
    key_file.write_text(KEY)
    handlers = []
    def opener(*args):
        handlers.append(args)
        return object()
    def fail(*args):
        raise HTTPError(ORIGIN, 302, KEY, headers(), None)
    monkeypatch.setattr(check_public, "build_opener", opener)
    monkeypatch.setattr(check_public, "run_check", fail)
    assert check_public.main(["--base-url", ORIGIN, "--expect-mode", "mock", "--key-file", str(key_file)]) == 1
    assert KEY not in capsys.readouterr().out
    assert len(handlers) == 2
    for group in handlers:
        assert any(isinstance(handler, check_public.NoRedirect) for handler in group)
        assert any(isinstance(handler, check_public.ProxyHandler) and handler.proxies == {} for handler in group)
    assert not any(isinstance(handler, check_public.HTTPCookieProcessor) for handler in handlers[0])
    assert any(isinstance(handler, check_public.HTTPCookieProcessor) for handler in handlers[1])


class AsgiOpener:
    """Adapt urllib requests to local HTTPS ASGI while applying its real cookie jar."""
    def __init__(self, client):
        self.client = client
        self.cookies = CookieJar()
        self.requests = []

    def open(self, request, timeout):
        self.cookies.add_cookie_header(request)
        self.requests.append(request)
        data = request.data
        if data is not None and not isinstance(data, bytes):
            data = b"".join(data)
        # Use urllib's jar, not httpx's automatic jar, to exercise Secure/Path behavior.
        self.client.cookies.clear()
        result = self.client.request(request.get_method(), request.full_url,
                                     headers=dict(request.header_items()), content=data,
                                     follow_redirects=False)
        response_headers = Message()
        for key, value in result.headers.multi_items():
            response_headers[key] = value
        response = Response(result.content, response_headers, result.status_code)
        self.cookies.extract_cookies(response, request)
        if result.status_code >= 300:
            raise HTTPError(request.full_url, result.status_code, "ASGI test failure", response_headers, response)
        return response


def test_entire_public_check_through_https_asgi_routes(tmp_path, capsys):
    settings = Settings(mode="mock", data_dir=tmp_path / "service", api_key=KEY,
                        mock_scenario="multiple", mock_delay_seconds=0, public_origin=ORIGIN,
                        upload_rate_per_minute=100)
    with TestClient(create_app(settings), base_url=ORIGIN) as client:
        deadline = time.monotonic() + 20
        while not client.get("/health").json().get("analyzer_ready"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        api, browser = AsgiOpener(client), AsgiOpener(client)
        check_public.run_check(ORIGIN, "mock", KEY, api, browser)
    assert len([request for request in api.requests if request.data is not None and request.data != b""]) == 1
    assert len([request for request in browser.requests if request.data is not None and request.data != b""]) == 1
    assert all(request.get_header("X-api-key") is None for request in browser.requests)
    assert any(request.get_header("Cookie") for request in browser.requests if "/ui/jobs/" in request.full_url)
    output = capsys.readouterr().out
    assert KEY not in output
    assert "восстановление" in output
    assert "событий: 2" in output
