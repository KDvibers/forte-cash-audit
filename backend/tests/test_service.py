"""Stage 2: public HTTP contract, process isolation, and durable job state.

All media in this suite is generated locally. No bank recordings are used.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import io
import time
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.config import Settings
from backend.main import create_app
from backend.make_sample import create_sample
from backend.store import JobStore


KEY = "test-only-key-never-use-in-production"
AUTH = {"X-API-Key": KEY}


@pytest.fixture(scope="session")
def video_bytes(tmp_path_factory):
    return create_sample(tmp_path_factory.mktemp("synthetic-media") / "sample.mov").read_bytes()


def wait_ready(client, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = client.get("/health")
        assert response.status_code == 200
        if response.json()["analyzer_ready"]:
            return
        time.sleep(0.05)
    pytest.fail("The mock worker did not become ready")


@contextmanager
def running(data_dir, **overrides):
    config = dict(mode="mock", data_dir=data_dir, api_key=KEY,
                  mock_delay_seconds=0.05, upload_rate_per_minute=100)
    config.update(overrides)
    with TestClient(create_app(Settings(**config))) as client:
        wait_ready(client)
        yield client


@pytest.fixture
def service(tmp_path):
    with running(tmp_path / "service") as client:
        yield client


def upload(client, video, *, name="synthetic.mov", headers=None):
    return client.post("/api/v1/analyze", headers={**AUTH, **(headers or {})},
                       files={"file": (name, video, "video/quicktime")})


def finish(client, job_id, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] in {"processing", "done", "failed"}
        if result["status"] != "processing":
            return result
        assert result == {"job_id": job_id, "status": "processing"}
        time.sleep(0.05)
    pytest.fail("Job stayed processing beyond the test deadline")


def accepted(response):
    assert response.status_code in (200, 202), response.text
    assert set(response.json()) == {"job_id"}
    assert isinstance(response.json()["job_id"], str)
    return response.json()["job_id"]


def test_authentication_precedes_invalid_body_and_unknown_jobs(service):
    # Missing or malformed multipart bodies must not bypass authentication.
    for headers in ({}, {"X-API-Key": "wrong"}):
        assert service.post("/api/v1/analyze", content=b"not multipart", headers=headers).status_code == 401
        assert service.get("/api/v1/jobs/missing", headers=headers).status_code == 401
    assert service.get("/api/v1/jobs/missing", headers=AUTH).status_code == 404
    assert service.post("/api/v1/analyze", headers=AUTH).status_code == 422


@pytest.mark.parametrize(("scenario", "verdict", "count"), [
    ("clean", "no_violation", 0), ("one", "violation", 1), ("multiple", "violation", 2),
])
def test_success_contract_and_actual_jpegs(tmp_path, video_bytes, scenario, verdict, count):
    with running(tmp_path / scenario, mock_scenario=scenario) as client:
        job_id = accepted(upload(client, video_bytes))
        result = finish(client, job_id)
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        assert result["status"] == "done"
        assert result["verdict"] == verdict
        assert len(result["violations"]) == count
        for event in result["violations"]:
            assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert event["type"] and event["description"]
            assert 0 <= event["timecode_sec"] <= 12
            assert 1 <= len(event["evidence_frames"]) <= 3
            for frame in event["evidence_frames"]:
                assert set(frame) == {"timestamp_sec", "image_base64"}
                assert 0 <= frame["timestamp_sec"] <= 12
                raw = base64.b64decode(frame["image_base64"], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    assert image.format == "JPEG"
                    image.verify()


def test_analysis_exception_is_failed_not_negative_result(tmp_path, video_bytes):
    with running(tmp_path / "failure", mock_scenario="error") as client:
        job_id = accepted(upload(client, video_bytes))
        assert finish(client, job_id) == {"job_id": job_id, "status": "failed"}


def test_bad_uploads_are_rejected_without_jobs(service, video_bytes):
    cases = [("wrong.txt", video_bytes, 415), ("empty.mov", b"", 422),
             ("broken.mov", b"not a movie", 422)]
    for name, content, status in cases:
        response = upload(service, content, name=name)
        assert response.status_code == status, response.text
        assert "job_id" not in response.json()


def test_client_filename_cannot_escape_job_storage(tmp_path, video_bytes):
    data_dir = tmp_path / "service"
    with running(data_dir) as client:
        job_id = accepted(upload(client, video_bytes, name="../../outside.mov"))
        assert finish(client, job_id)["status"] == "done"
    assert not (tmp_path / "outside.mov").exists()
    saved = [path for path in data_dir.rglob("*.mov") if path.read_bytes() == video_bytes]
    assert len(saved) == 1
    assert saved[0].resolve().is_relative_to(data_dir.resolve())


def test_retried_upload_id_returns_original_job(service, video_bytes):
    headers = {"X-Upload-ID": str(uuid.uuid4())}
    first = accepted(upload(service, video_bytes, headers=headers))
    second = accepted(upload(service, video_bytes, headers=headers))
    assert second == first
    assert finish(service, first)["status"] == "done"


@pytest.mark.parametrize("chunked", [False, True])
def test_upload_size_limit_including_unknown_content_length(tmp_path, video_bytes, chunked):
    with running(tmp_path / "small-limit", max_upload_mb=0.01) as client:
        if not chunked:
            response = upload(client, video_bytes)
        else:
            boundary = "stage2-boundary"
            prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                      "filename=\"synthetic.mov\"\r\nContent-Type: video/quicktime\r\n\r\n").encode()
            chunks = [prefix, *(video_bytes[i:i + 4096] for i in range(0, len(video_bytes), 4096)),
                      f"\r\n--{boundary}--\r\n".encode()]
            response = client.post("/api/v1/analyze", content=iter(chunks),
                                   headers={**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert response.status_code == 413, response.text


def test_two_jobs_do_not_block_http_and_keep_separate_results(tmp_path, video_bytes):
    with running(tmp_path / "two-jobs", mock_delay_seconds=1.2) as client:
        first = accepted(upload(client, video_bytes))
        second = accepted(upload(client, video_bytes))
        assert first != second
        started = time.monotonic()
        health = client.get("/health")
        pending = client.get(f"/api/v1/jobs/{second}", headers=AUTH)
        assert time.monotonic() - started < 1.0
        assert health.json()["analyzer_ready"] is True
        assert pending.json() == {"job_id": second, "status": "processing"}
        for job_id in (first, second):
            assert finish(client, job_id)["status"] == "done"


def test_queue_is_bounded_and_deadline_fails_unfinished_jobs(tmp_path, video_bytes):
    with running(tmp_path / "deadlines", mock_delay_seconds=4,
                 job_timeout_seconds=1.5, max_pending_jobs=2) as client:
        first = accepted(upload(client, video_bytes))
        second = accepted(upload(client, video_bytes))
        third = upload(client, video_bytes)
        assert third.status_code == 429, third.text
        for job_id in (first, second):
            assert finish(client, job_id) == {"job_id": job_id, "status": "failed"}


def test_completed_result_survives_server_restart(tmp_path, video_bytes):
    data_dir = tmp_path / "persistent"
    with running(data_dir) as client:
        job_id = accepted(upload(client, video_bytes))
        original = finish(client, job_id)
    with running(data_dir) as client:
        assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == original


def test_shutdown_and_restart_never_leave_old_processing_job(tmp_path, video_bytes):
    data_dir = tmp_path / "interrupted"
    with running(data_dir, mock_delay_seconds=30) as client:
        job_id = accepted(upload(client, video_bytes))
        assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()["status"] == "processing"
    with running(data_dir) as client:
        assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == {
            "job_id": job_id, "status": "failed"}


def test_preflight_validation_counts_toward_processing_deadline(tmp_path, video_bytes, monkeypatch):
    from backend import main as main_module

    original_validate = main_module.validate_video

    def slow_validate(*args, **kwargs):
        time.sleep(0.25)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(main_module, "validate_video", slow_validate)
    with running(tmp_path / "slow-preflight", job_timeout_seconds=0.2,
                 mock_delay_seconds=0) as client:
        job_id = accepted(upload(client, video_bytes))
        assert finish(client, job_id) == {"job_id": job_id, "status": "failed"}


def test_startup_removes_only_orphan_job_directories_and_recovers_old_job(tmp_path):
    data_dir = tmp_path / "restart-cleanup"
    upload_dir = data_dir / "uploads"
    known_id = str(uuid.uuid4())
    known = upload_dir / known_id
    orphan = upload_dir / str(uuid.uuid4())
    manual = upload_dir / "manually-created-folder"
    outside = tmp_path / str(uuid.uuid4())
    for directory in (known, orphan, manual, outside):
        directory.mkdir(parents=True)
        (directory / "source.mov").write_bytes(b"preserve unless orphan")
    store = JobStore(data_dir / "jobs.sqlite3")
    store.create({"job_id": known_id, "owner": "api", "file_name": "source.mov",
                  "file_size": (known / "source.mov").stat().st_size,
                  "video_path": str((known / "source.mov").resolve()), "created_at": time.time()})

    with running(data_dir) as client:
        assert not orphan.exists()
        for directory in (known, manual, outside):
            assert (directory / "source.mov").read_bytes() == b"preserve unless orphan"
        assert client.get(f"/api/v1/jobs/{known_id}", headers=AUTH).json() == {
            "job_id": known_id, "status": "failed"}
        assert store.get(known_id)["error_code"] == "server_restarted"
