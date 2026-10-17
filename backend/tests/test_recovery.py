"""Browser recovery keeps ownership and original result mode across restarts."""
from __future__ import annotations

import sqlite3
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from backend.config import Settings
from backend.main import create_app
from backend.make_sample import create_sample
from backend.service import JobService
from backend.store import JobStore
from backend.tests.test_browser_api import KEY, COOKIE, running_browser, session, use_cookie, upload, finish


@pytest.fixture(scope='module')
def recovery_video(tmp_path_factory):
    return create_sample(tmp_path_factory.mktemp('recovery-media') / 'neutral.mov').read_bytes()


def test_recovery_finds_accepted_upload_without_post_acknowledgement(tmp_path, recovery_video):
    upload_id = str(uuid.uuid4())
    with running_browser(tmp_path / 'service') as client:
        bootstrap, _ = session(client)
        job_id = upload(client, recovery_video, bootstrap['csrf_token'], upload_id)
        response = client.get(f'/ui/recovery/{upload_id}')
        assert response.status_code == 200
        metadata = response.json()
        assert set(metadata) == {'job_id', 'status', 'mode', 'file_name', 'file_size', 'created_at'}
        assert metadata['job_id'] == job_id
        assert metadata['mode'] == 'mock'
        assert metadata['file_name'] == 'synthetic.mov'
        assert metadata['file_size'] == len(recovery_video)
        assert metadata['created_at'] <= time.time()
        assert 'no-store' in response.headers['cache-control']
        assert client.get(f'/ui/jobs/{job_id}').headers['X-Cash-Audit-Job-Mode'] == 'mock'
        finish(client, job_id)
        for _ in range(3):
            assert client.get(f'/ui/recovery/{upload_id}').json()['status'] == 'done'
        with client.app.state.service.store.connection() as db:
            assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1


def test_recovery_never_exposes_other_browser_or_public_api_jobs(tmp_path, recovery_video):
    with running_browser(tmp_path / 'ownership') as client:
        bootstrap, first_cookie = session(client)
        own_upload = str(uuid.uuid4())
        job_id = upload(client, recovery_video, bootstrap['csrf_token'], own_upload)
        finish(client, job_id)
        public_upload = str(uuid.uuid4())
        public_response = client.post('/api/v1/analyze', headers={'X-API-Key': KEY, 'X-Upload-ID': public_upload},
                                      files={'file': ('neutral.mov', recovery_video, 'video/quicktime')})
        assert public_response.status_code == 200
        assert client.get(f'/ui/recovery/{public_upload}').status_code == 404
        use_cookie(client, None)
        assert client.get(f'/ui/recovery/{own_upload}').status_code == 401
        session(client)
        assert client.get(f'/ui/recovery/{own_upload}').status_code == 404
        assert client.get(f'/ui/jobs/{job_id}').status_code == 404
        use_cookie(client, first_cookie)
        assert client.get(f'/ui/recovery/{own_upload}').status_code == 200
        assert client.get(f'/ui/recovery/{own_upload}', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
        assert client.get('/ui/recovery/not-a-uuid').status_code == 404


def test_saved_mock_result_remains_available_when_real_worker_unavailable(tmp_path, recovery_video, monkeypatch):
    data_dir = tmp_path / 'restart'
    upload_id = str(uuid.uuid4())
    with running_browser(data_dir) as client:
        bootstrap, cookie = session(client)
        job_id = upload(client, recovery_video, bootstrap['csrf_token'], upload_id)
        original = finish(client, job_id)

    # Independent of whether a real analyzer has been installed on this machine.
    def unavailable_worker(self):
        self.ready = False
        self.readiness_message = 'Analyzer intentionally unavailable in recovery test.'

    monkeypatch.setattr(JobService, '_launch_worker', unavailable_worker)
    with TestClient(create_app(Settings(mode='real', data_dir=data_dir, api_key=KEY))) as client:
        use_cookie(client, cookie)
        bootstrap, _ = session(client)
        assert bootstrap['mode'] == 'real'
        assert bootstrap['analyzer_ready'] is False
        assert client.get(f'/ui/recovery/{upload_id}').json()['mode'] == 'mock'
        response = client.get(f'/ui/jobs/{job_id}')
        assert response.json() == original
        assert response.headers['X-Cash-Audit-Mode'] == 'real'
        assert response.headers['X-Cash-Audit-Job-Mode'] == 'mock'


def test_expired_session_and_deleted_job_cannot_be_recovered(tmp_path, recovery_video):
    upload_id = str(uuid.uuid4())
    with running_browser(tmp_path / 'expiry') as client:
        bootstrap, cookie = session(client)
        job_id = upload(client, recovery_video, bootstrap['csrf_token'], upload_id)
        finish(client, job_id)
        store = client.app.state.service.store
        with store.connection() as db:
            db.execute('UPDATE browser_sessions SET expires_at=?', (time.time() - 1,))
        assert client.get(f'/ui/recovery/{upload_id}').status_code == 401
        session(client)  # A new session must not inherit the expired session's job.
        assert client.cookies.get(COOKIE) != cookie
        assert client.get(f'/ui/recovery/{upload_id}').status_code == 404
        store.delete(job_id)
        assert client.get(f'/ui/jobs/{job_id}').status_code == 404


def test_old_database_migration_preserves_jobs_without_guessing_mode(tmp_path):
    path = tmp_path / 'old.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT 'api', file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL, video_path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'processing',
            result_json TEXT, error_code TEXT, created_at REAL NOT NULL, started_at REAL,
            finished_at REAL, scenario TEXT NOT NULL DEFAULT 'one', upload_id TEXT, UNIQUE(owner, upload_id)
        )''')
        db.execute("INSERT INTO jobs (job_id,file_name,file_size,video_path,created_at) VALUES (?,?,?,?,?)",
                   ('legacy-job', 'legacy.mov', 1, '/unused/legacy.mov', time.time()))
    for _ in range(2):  # Migration is idempotent on later application starts.
        store = JobStore(path)
        old = store.get('legacy-job')
        assert old['mode'] is None
        assert old['file_name'] == 'legacy.mov'
        assert old['status'] == 'processing'


def test_unknown_legacy_mode_is_explicit_and_public_contract_is_unchanged(tmp_path, recovery_video):
    upload_id = str(uuid.uuid4())
    with running_browser(tmp_path / 'legacy-mode') as client:
        bootstrap, _ = session(client)
        job_id = upload(client, recovery_video, bootstrap['csrf_token'], upload_id)
        finish(client, job_id)
        with client.app.state.service.store.connection() as db:
            db.execute('UPDATE jobs SET mode=NULL WHERE job_id=?', (job_id,))
        assert client.get(f'/ui/recovery/{upload_id}').json()['mode'] is None
        response = client.get(f'/ui/jobs/{job_id}')
        assert response.headers['X-Cash-Audit-Job-Mode'] == 'unknown'
        assert set(response.json()) == {'job_id', 'status', 'verdict', 'violations'}
        assert client.get(f'/api/v1/jobs/{job_id}', headers={'X-API-Key': KEY}).status_code == 404
