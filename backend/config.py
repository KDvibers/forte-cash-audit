"""Local server settings. Mock responses require an explicit opt-in."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def normalize_public_origin(value: str | None) -> str | None:
    """Accept one HTTPS DNS origin, never a URL with credentials or a route."""
    if value is None:
        return None
    error = "CASH_AUDIT_PUBLIC_ORIGIN must be an HTTPS DNS origin without credentials, path, query or fragment"
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(character.isspace() for character in value)
            or any(character in value for character in ("\\", "?", "#"))):
        raise ValueError(error)
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError(error) from None
    if (parsed.scheme != "https" or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or not host or len(host) > 253
            or "." not in host or host.endswith((".localhost", ".local"))):
        raise ValueError(error)
    # ASCII DNS labels include punycode; IP literals and all-numeric suffixes do not.
    if (not re.fullmatch(r"[A-Za-z0-9.-]+", host)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", host.rsplit(".", 1)[-1])
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                   for label in host.split("."))
            or (port is not None and not 1 <= port <= 65535)
            or parsed.netloc.endswith(":")):
        raise ValueError(error)
    authority = host if port in (None, 443) else f"{host}:{port}"
    return f"https://{authority}"


@dataclass
class Settings:
    mode: str = "real"
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / ".local")
    api_key: str | None = None
    max_upload_mb: float = 100
    job_timeout_seconds: float = 285
    mock_delay_seconds: float = 2
    mock_scenario: str = "one"
    max_pending_jobs: int = 2
    upload_rate_per_minute: int = 12
    retention_hours: float = 24
    worker_startup_seconds: float = 90
    ffprobe_path: str | None = None
    public_origin: str | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.mode = self.mode.lower()
        self.public_origin = normalize_public_origin(self.public_origin)
        if self.mode not in {"real", "mock"}:
            raise ValueError("CASH_AUDIT_MODE must be real or mock")
        if self.mock_scenario not in {"clean", "one", "multiple", "error"}:
            raise ValueError("CASH_AUDIT_MOCK_SCENARIO must be clean, one, multiple or error")
        for name in ("max_upload_mb", "job_timeout_seconds", "retention_hours", "worker_startup_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("max_pending_jobs", "upload_rate_per_minute"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.mock_delay_seconds) or self.mock_delay_seconds < 0:
            raise ValueError("mock_delay_seconds must be finite and nonnegative")
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_mb must allow at least one byte")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        fields = {
            "MODE": ("mode", str), "DATA_DIR": ("data_dir", Path), "API_KEY": ("api_key", str),
            "MAX_UPLOAD_MB": ("max_upload_mb", float), "JOB_TIMEOUT_SECONDS": ("job_timeout_seconds", float),
            "MOCK_DELAY_SECONDS": ("mock_delay_seconds", float), "MOCK_SCENARIO": ("mock_scenario", str),
            "MAX_PENDING_JOBS": ("max_pending_jobs", int), "UPLOAD_RATE_PER_MINUTE": ("upload_rate_per_minute", int),
            "RETENTION_HOURS": ("retention_hours", float), "WORKER_STARTUP_SECONDS": ("worker_startup_seconds", float),
            "FFPROBE_PATH": ("ffprobe_path", str),
            "PUBLIC_ORIGIN": ("public_origin", str),
        }
        return cls(**{name: cast(os.environ["CASH_AUDIT_" + key]) for key, (name, cast) in fields.items()
                      if os.environ.get("CASH_AUDIT_" + key)})
