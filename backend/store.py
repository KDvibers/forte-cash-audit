"""SQLite job metadata; connections are never shared between threads."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, owner TEXT NOT NULL DEFAULT 'api', file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL, video_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing' CHECK(status IN ('processing','done','failed')),
                    result_json TEXT, error_code TEXT, created_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, scenario TEXT NOT NULL DEFAULT 'one', upload_id TEXT,
                    mode TEXT CHECK(mode IN ('mock', 'real') OR mode IS NULL),
                    UNIQUE(owner, upload_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
            """)
            # Earlier jobs have no recorded mode. Do not guess real vs synthetic.
            columns = {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            if 'mode' not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN mode TEXT CHECK(mode IN ('mock', 'real') OR mode IS NULL)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def recover(self):
        with self.connection() as db:
            db.execute("UPDATE jobs SET status='failed', result_json=NULL, error_code='server_restarted', "
                       "finished_at=? WHERE status='processing'", (time.time(),))

    def create(self, job: dict):
        allowed = {"job_id", "owner", "file_name", "file_size", "video_path", "created_at", "scenario", "upload_id", "mode"}
        if not set(job).issubset(allowed):
            raise ValueError("Unexpected job fields")
        keys = tuple(job)
        with self.connection() as db:
            db.execute(f"INSERT INTO jobs ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", tuple(job.values()))

    def get(self, job_id: str):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def by_upload_id(self, owner: str, upload_id: str | None):
        if not upload_id:
            return None
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE owner=? AND upload_id=?", (owner, upload_id)).fetchone()
            return dict(row) if row else None

    def pending(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE status='processing' ORDER BY created_at")]

    def start(self, job_id: str):
        with self.connection() as db:
            db.execute("UPDATE jobs SET started_at=? WHERE job_id=? AND status='processing'", (time.time(), job_id))

    def finish(self, job_id: str, result: dict | None = None, error: str | None = None):
        if (result is None) == (error is None):
            raise ValueError("A job must finish with either a result or an error")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("A failure requires an error code")
        with self.connection() as db:
            db.execute("UPDATE jobs SET status=?, result_json=?, error_code=?, finished_at=? "
                       "WHERE job_id=? AND status='processing'",
                       ("failed" if error else "done", json.dumps(result, ensure_ascii=False, allow_nan=False)
                        if result is not None else None, error, time.time(), job_id))

    def expired_jobs(self, cutoff: float):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE status != 'processing' AND finished_at < ?", (cutoff,))]

    def delete(self, job_id: str):
        with self.connection() as db:
            db.execute("DELETE FROM jobs WHERE job_id=? AND status != 'processing'", (job_id,))
