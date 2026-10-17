"""HTTPS gateway boundaries use Uvicorn's real proxy middleware, without a tunnel."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.browser_sessions import BrowserSessions, COOKIE_NAME
from backend.config import Settings
from backend.main import create_app
from backend.service import JobService


ORIGIN = "https://audit.example"
KEY = "test-public-access-key-not-for-deployment"
PROXY_HEADERS = {"X-Forwarded-Proto": "https"}


@pytest.mark.parametrize("value", [
    "http://audit.example", "https://name:secret@audit.example", "https://audit.example/path",
    "https://audit.example?key=secret", "https://audit.example#secret", "https://audit.example?",
    "https://audit.example#", "https://127.0.0.1", "https://127.0.0.0x1", "https://[::1]", "https://localhost",
    "https://app.localhost", "https://app.local", "https://*.audit.example",
    "https://audit.example.", "https://bad_name.example", "https://audit.example:0",
    "https://audit.example:65536", "https://audit.example:", " https://audit.example",
    "https://audit.example\\attacker", "https://audit.example\n", "",
])
def test_public_origin_rejects_non_origins_without_echoing_value(tmp_path, value):
    with pytest.raises(ValueError, match="CASH_AUDIT_PUBLIC_ORIGIN") as failure:
        Settings(data_dir=tmp_path, public_origin=value)
    assert "secret" not in str(failure.value)


def test_public_origin_normalizes_host_default_port_and_trailing_slash(tmp_path, monkeypatch):
    assert Settings(data_dir=tmp_path).public_origin is None
    assert Settings(data_dir=tmp_path, public_origin="https://Audit.Example:443/").public_origin == ORIGIN
    assert Settings(data_dir=tmp_path, public_origin="https://audit.example:8443").public_origin == ORIGIN + ":8443"
    monkeypatch.setenv("CASH_AUDIT_PUBLIC_ORIGIN", ORIGIN)
    assert Settings.from_env().public_origin == ORIGIN


@pytest.fixture
def public_app(tmp_path):
    settings = Settings(mode="mock", data_dir=tmp_path, api_key=KEY, public_origin=ORIGIN)
    app = create_app(settings)
    # These tests cover HTTP guards; no inference or background worker is needed.
    service = JobService(settings)
    service.ready = True
    service.readiness_message = "Synthetic gateway test"
    app.state.service = service
    app.state.browser_sessions = BrowserSessions(service.store)
    return ProxyHeadersMiddleware(app, trusted_hosts=["127.0.0.1"])


@pytest.fixture
def public_client(public_app):
    # A browser uses HTTPS to the edge; the local tunnel forwards HTTP to Uvicorn.
    client = TestClient(public_app, base_url="http://audit.example", client=("127.0.0.1", 45678))
    client.headers.update(PROXY_HEADERS)
    yield client
    client.close()


def browser_headers(client):
    response = client.get("/ui/session")
    assert response.status_code == 200
    # Explicitly emulate the cookie forwarded by the HTTPS edge over local HTTP.
    return {"Cookie": f"{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}",
            "Origin": ORIGIN, "X-CSRF-Token": response.json()["csrf_token"]}


def test_trusted_proxy_sets_https_and_secure_host_only_cookie(public_client):
    response = public_client.get("/ui/session")
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    for flag in ("; secure", "httponly", "samesite=strict", "path=/ui"):
        assert flag in cookie
    assert "domain=" not in cookie
    assert KEY not in response.text
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_nonlocal_client_cannot_forge_forwarded_https(public_app):
    client = TestClient(public_app, base_url="http://audit.example", client=("203.0.113.10", 45678))
    try:
        response = client.get("/ui/session", headers=PROXY_HEADERS)
        assert response.status_code == 400
        assert "HTTPS" in response.json()["detail"]
        assert "set-cookie" not in response.headers
    finally:
        client.close()


@pytest.mark.parametrize("host", ["attacker.example", "audit.example.attacker.example", "127.0.0.1",
                                  "audit.example:8000", "audit.example,attacker.example"])
def test_public_host_is_fixed_even_when_forwarded_host_looks_valid(public_client, host):
    response = public_client.get("/health", headers={"Host": host, "X-Forwarded-Host": "audit.example"})
    assert response.status_code == 400
    assert "location" not in response.headers
    assert response.headers["x-frame-options"] == "DENY"


def test_duplicate_host_is_rejected(public_client):
    response = public_client.get("/health", headers=[("Host", "audit.example"), ("Host", "attacker.example")])
    assert response.status_code == 400


def test_empty_host_is_rejected(public_client):
    assert public_client.get("/health", headers={"Host": ""}).status_code == 400


def test_forwarded_host_never_changes_the_accepted_origin(public_client):
    assert public_client.get("/health", headers={"X-Forwarded-Host": "attacker.example"}).status_code == 200
    headers = browser_headers(public_client)
    response = public_client.post("/ui/analyze", headers={**headers, "X-Forwarded-Host": "attacker.example"})
    assert response.status_code == 422  # Authentication passed; the required video is absent.


@pytest.mark.parametrize("method,path", [("GET", "/ui/session"), ("POST", "/api/v1/analyze")])
def test_public_http_is_rejected_without_redirect_or_cookie(public_client, method, path):
    response = public_client.request(method, path, headers={"X-Forwarded-Proto": "http"}, content=b"invalid")
    assert response.status_code == 400
    assert "HTTPS" in response.json()["detail"]
    assert "location" not in response.headers
    assert "set-cookie" not in response.headers


def test_public_api_still_requires_key_before_body_parsing(public_client):
    for supplied in ({}, {"X-API-Key": "wrong"}):
        assert public_client.post("/api/v1/analyze", headers=supplied, content=b"not multipart").status_code == 401
        assert public_client.get("/api/v1/jobs/missing", headers=supplied).status_code == 401
    assert public_client.get("/api/v1/jobs/missing", headers={"X-API-Key": KEY}).status_code == 404


@pytest.mark.parametrize("origin", [None, "null", "http://audit.example", "https://attacker.example",
                                    "https://audit.example.attacker.example"])
def test_public_browser_origin_and_csrf_are_checked_before_body(public_client, origin):
    headers = browser_headers(public_client)
    headers.pop("Origin")
    if origin is not None:
        headers["Origin"] = origin
    assert public_client.post("/ui/analyze", headers=headers, content=b"not multipart").status_code == 403


def test_public_bootstrap_rejects_foreign_origin_and_duplicate_origin(public_client):
    assert public_client.get("/ui/session", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert public_client.get("/ui/session", headers=[("Origin", ORIGIN), ("Origin", ORIGIN)]).status_code == 403


def test_valid_origin_does_not_replace_csrf(public_client):
    headers = browser_headers(public_client)
    headers["X-CSRF-Token"] = "forged"
    assert public_client.post("/ui/analyze", headers=headers, content=b"not multipart").status_code == 403
