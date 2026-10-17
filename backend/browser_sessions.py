"""Anonymous browser ownership, independent of the evaluator's API key."""
from __future__ import annotations

import hashlib
import secrets
import time

from .store import JobStore

COOKIE_NAME = "cash_audit_session"
SESSION_TTL_SECONDS = 24 * 60 * 60


class BrowserSessions:
    """Persist only hashes of bearer cookies; keep ownership across restarts."""

    def __init__(self, store: JobStore):
        self.store = store
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS browser_sessions (
                    session_id TEXT PRIMARY KEY,
                    csrf_token TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS browser_sessions_expiry
                    ON browser_sessions(expires_at);
            """)

    @staticmethod
    def session_id(token: str | None) -> str | None:
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            return None
        return hashlib.sha256(token.encode()).hexdigest()

    def get(self, token: str | None) -> dict | None:
        identity = self.session_id(token)
        if identity is None:
            return None
        with self.store.connection() as db:
            row = db.execute(
                "SELECT session_id, csrf_token, expires_at FROM browser_sessions "
                "WHERE session_id=? AND expires_at>?", (identity, time.time()),
            ).fetchone()
        return dict(row) if row else None

    def open(self, token: str | None) -> tuple[str, dict]:
        """Create or renew a session when the same-origin page connects."""
        now = time.time()
        identity = self.session_id(token)
        with self.store.connection() as db:
            db.execute("DELETE FROM browser_sessions WHERE expires_at<=?", (now,))
            row = None
            if identity is not None:
                row = db.execute(
                    "SELECT session_id, csrf_token, expires_at FROM browser_sessions "
                    "WHERE session_id=?", (identity,),
                ).fetchone()
            if row:
                session = dict(row)
                session["expires_at"] = now + SESSION_TTL_SECONDS
                db.execute("UPDATE browser_sessions SET expires_at=? WHERE session_id=?",
                           (session["expires_at"], identity))
            else:
                token = secrets.token_urlsafe(32)
                session = {"session_id": self.session_id(token),
                           "csrf_token": secrets.token_urlsafe(32),
                           "expires_at": now + SESSION_TTL_SECONDS}
                db.execute("INSERT INTO browser_sessions (session_id, csrf_token, expires_at) "
                           "VALUES (?, ?, ?)",
                           (session["session_id"], session["csrf_token"], session["expires_at"]))
        return token, session

    @staticmethod
    def owner(session: dict) -> str:
        return "ui:" + session["session_id"]
