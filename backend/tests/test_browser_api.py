"""Browser upload flow and separation from the key-authenticated public API.

All videos are generated test patterns; no bank footage or real API keys are used.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from backend.config import Settings
from backend.main import create_app
from backend.make_sample import create_sample


KEY = "browser-tests-only-secret-never-embed-in-frontend"
COOKIE = "cash_audit_session"
ORIGIN = "http://testserver"


@pytest.fixture(scope="session")
def browser_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("browser-synthetic-media") / "sample.mov"
    return create_sample(path).read_bytes()


@contextmanager
def running_browser(data_dir, **overrides):
    config = dict(mode="mock", data_dir=data_dir, api_key=KEY,
                  mock_delay_seconds=0.05, upload_rate_per_minute=100)
    config.update(overrides)
    with TestClient(create_app(Settings(**config))) as client:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if client.get("/health").json()["analyzer_ready"]:
                break
            time.sleep(0.05)
        else:
            pytest.fail("The mock analyzer did not become ready")
        yield client


@pytest.fixture
def browser_service(tmp_path):
    with running_browser(tmp_path / "browser-service") as client:
        yield client


def session(client):
    response = client.get("/ui/session")
    assert response.status_code == 200, response.text
    return response.json(), client.cookies.get(COOKIE)


def use_cookie(client, value):
    client.cookies.clear()
    if value:
        client.cookies.set(COOKIE, value, domain="testserver.local", path="/ui")


def upload(client, video, token, upload_id=None):
    headers = {"Origin": ORIGIN, "X-CSRF-Token": token}
    if upload_id:
        headers["X-Upload-ID"] = upload_id
    response = client.post("/ui/analyze", headers=headers,
                           files={"file": ("synthetic.mov", video, "video/quicktime")})
    assert response.status_code in (200, 202), response.text
    assert set(response.json()) == {"job_id"}
    return response.json()["job_id"]


def finish(client, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"done", "failed"}:
            return result
        assert result == {"job_id": job_id, "status": "processing"}
        time.sleep(0.05)
    pytest.fail("The browser job never reached a terminal state")


def test_session_cookie_and_bootstrap_do_not_expose_api_key(browser_service):
    client = browser_service
    response = client.get("/ui/session")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"csrf_token", "mode", "analyzer_ready", "message",
                            "max_upload_bytes", "poll_interval_ms", "job_timeout_seconds"}
    assert isinstance(payload["csrf_token"], str) and len(payload["csrf_token"]) >= 32
    assert payload["mode"] == "mock"
    assert payload["analyzer_ready"] is True
    assert payload["max_upload_bytes"] == 100 * 1024 * 1024
    assert 100 <= payload["poll_interval_ms"] <= 10_000
    assert payload["job_timeout_seconds"] > 0
    cookie = client.cookies.get(COOKIE)
    assert cookie and len(cookie) >= 32
    flags = response.headers["set-cookie"].lower()
    for expected in ("httponly", "samesite=strict", "path=/ui", "max-age=86400"):
        assert expected in flags
    assert "no-store" in response.headers["cache-control"]
    assert KEY not in response.text and cookie not in response.text
    assert COOKIE not in client.build_request("GET", "/api/v1/jobs/missing").headers.get("cookie", "")
    repeated, repeated_cookie = session(client)
    assert repeated_cookie == cookie
    assert repeated["csrf_token"] == payload["csrf_token"]


@pytest.mark.parametrize(("scenario", "verdict", "count"), [
    ("one", "violation", 1), ("clean", "no_violation", 0),
])
def test_browser_upload_reaches_real_server_result(tmp_path, browser_video, scenario, verdict, count):
    with running_browser(tmp_path / scenario, mock_scenario=scenario) as client:
        bootstrap, _ = session(client)
        job_id = upload(client, browser_video, bootstrap["csrf_token"])
        result = finish(client, job_id)
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        assert result["status"] == "done"
        assert result["verdict"] == verdict
        assert len(result["violations"]) == count


def test_browser_analyzer_failure_is_not_a_clean_verdict(tmp_path, browser_video):
    with running_browser(tmp_path / "failure", mock_scenario="error") as client:
        bootstrap, _ = session(client)
        job_id = upload(client, browser_video, bootstrap["csrf_token"])
        assert finish(client, job_id) == {"job_id": job_id, "status": "failed"}


@pytest.mark.parametrize("cookie", [None, "forged-session"])
def test_missing_or_forged_session_is_rejected_before_body_parse(browser_service, cookie):
    client = browser_service
    use_cookie(client, cookie)
    headers = {"Origin": ORIGIN, "X-CSRF-Token": "forged-token",
               "Content-Type": "multipart/form-data; boundary=broken"}
    assert client.post("/ui/analyze", headers=headers, content=b"not multipart").status_code == 401
    assert client.get("/ui/jobs/missing").status_code == 401


@pytest.mark.parametrize("origin", [None, "null", "https://testserver",
                                    "http://testserver.evil.example", "http://evil.example"])
def test_origin_is_checked_before_multipart_parsing(browser_service, origin):
    bootstrap, _ = session(browser_service)
    headers = {"X-CSRF-Token": bootstrap["csrf_token"],
               "Content-Type": "multipart/form-data; boundary=broken"}
    if origin is not None:
        headers["Origin"] = origin
    response = browser_service.post("/ui/analyze", headers=headers, content=b"not multipart")
    assert response.status_code == 403, response.text


def test_csrf_and_fetch_metadata_are_checked_before_body_parse(browser_service):
    bootstrap, _ = session(browser_service)
    for token in (None, "forged-token"):
        headers = {"Origin": ORIGIN, "Content-Type": "multipart/form-data; boundary=broken"}
        if token is not None:
            headers["X-CSRF-Token"] = token
        assert browser_service.post("/ui/analyze", headers=headers,
                                    content=b"not multipart").status_code == 403
    response = browser_service.post("/ui/analyze", content=b"not multipart", headers={
        "Origin": ORIGIN, "X-CSRF-Token": bootstrap["csrf_token"], "Sec-Fetch-Site": "cross-site",
    })
    assert response.status_code == 403


def test_cross_site_page_cannot_create_browser_session(browser_service):
    response = browser_service.get("/ui/session", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403
    assert "set-cookie" not in response.headers
    assert browser_service.cookies.get(COOKIE) is None


def test_https_session_cookie_is_secure(browser_service):
    browser_service.base_url = "https://testserver"
    response = browser_service.get("/ui/session")
    assert response.status_code == 200
    assert "; secure" in response.headers["set-cookie"].lower()


def test_browser_and_api_owners_cannot_read_each_others_jobs(browser_service, browser_video):
    client = browser_service
    alice, alice_cookie = session(client)
    alice_job = upload(client, browser_video, alice["csrf_token"])
    assert finish(client, alice_job)["status"] == "done"

    use_cookie(client, None)
    bob, bob_cookie = session(client)
    assert bob_cookie != alice_cookie
    assert client.get(f"/ui/jobs/{alice_job}").status_code == 404
    assert client.get(f"/api/v1/jobs/{alice_job}", headers={"X-API-Key": KEY}).status_code == 404
    # Another browser's valid CSRF token does not authorize this session.
    assert client.post("/ui/analyze", content=b"invalid", headers={
        "Origin": ORIGIN, "X-CSRF-Token": alice["csrf_token"],
    }).status_code == 403

    public = client.post("/api/v1/analyze", headers={"X-API-Key": KEY},
                         files={"file": ("synthetic.mov", browser_video, "video/quicktime")})
    assert public.status_code in (200, 202), public.text
    public_job = public.json()["job_id"]
    assert client.get(f"/ui/jobs/{public_job}").status_code == 404
    # Having a browser session never substitutes for the public API key.
    assert client.get(f"/api/v1/jobs/{public_job}").status_code == 401
    use_cookie(client, alice_cookie)
    assert client.get(f"/ui/jobs/{public_job}").status_code == 404
    assert client.get(f"/ui/jobs/{alice_job}").json()["status"] == "done"


def test_upload_retry_identifier_is_scoped_to_browser_owner(browser_service, browser_video):
    client = browser_service
    identifier = str(uuid.uuid4())
    alice, _ = session(client)
    first = upload(client, browser_video, alice["csrf_token"], identifier)
    assert upload(client, browser_video, alice["csrf_token"], identifier) == first
    assert finish(client, first)["status"] == "done"

    use_cookie(client, None)
    bob, _ = session(client)
    second = upload(client, browser_video, bob["csrf_token"], identifier)
    assert second != first
    assert finish(client, second)["status"] == "done"


def test_browser_session_and_job_survive_server_restart(tmp_path, browser_video):
    data_dir = tmp_path / "persistent-browser"
    with running_browser(data_dir) as client:
        bootstrap, cookie = session(client)
        job_id = upload(client, browser_video, bootstrap["csrf_token"])
        original = finish(client, job_id)
    with running_browser(data_dir) as client:
        use_cookie(client, cookie)
        renewed, renewed_cookie = session(client)
        assert renewed_cookie == cookie
        assert renewed["csrf_token"] == bootstrap["csrf_token"]
        assert client.get(f"/ui/jobs/{job_id}").json() == original


def test_expired_browser_session_cannot_reclaim_previous_jobs(browser_service, browser_video):
    client = browser_service
    bootstrap, cookie = session(client)
    job_id = upload(client, browser_video, bootstrap["csrf_token"])
    assert finish(client, job_id)["status"] == "done"
    with client.app.state.service.store.connection() as db:
        db.execute("UPDATE browser_sessions SET expires_at=0")
    assert client.get(f"/ui/jobs/{job_id}").status_code == 401
    assert client.post("/ui/analyze", content=b"invalid", headers={
        "Origin": ORIGIN, "X-CSRF-Token": bootstrap["csrf_token"],
    }).status_code == 401
    renewed, new_cookie = session(client)
    assert new_cookie != cookie
    assert renewed["csrf_token"] != bootstrap["csrf_token"]
    assert client.get(f"/ui/jobs/{job_id}").status_code == 404


def test_only_explicit_frontend_assets_are_public(browser_service):
    client = browser_service
    for path, content_type in (("/", "text/html"), ("/assets/styles.css", "text/css"),
                               ("/assets/app.js", "javascript")):
        response = client.get(path)
        assert response.status_code == 200, path
        assert content_type in response.headers["content-type"], path
        assert KEY not in response.text, path
        assert "no-store" in response.headers["cache-control"]
    for path in ("/assets/", "/assets/README.md", "/assets/preview.ps1", "/assets/index.html",
                 "/backend/.local/api-key.txt", "/backend/.local/jobs.sqlite3", "/.env",
                 "/.git/config", "/uploads/", "/assets/%2e%2e%2fbackend%2fmain.py"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert KEY not in response.text, path
