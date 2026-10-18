"""Explicit inactive team profile; loading configuration never contacts a peer."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import stat
import time

from .local_client import LocalModelClient
from .observations import ObservationDeadline, ObservationError
from .outerwear import outerwear_policy_fingerprint
from .outerwear_cascade import (CascadeOuterwearObserver, _identities,
                                canonical_runtime, cascade_policy_fingerprint)
from .team_outerwear_client import TeamOuterwearWorkerClient


CONFIG_ENV = "FORTE_TEAM_OUTERWEAR_CONFIG"
MAX_CONFIG_BYTES = 64 * 1024
_FIELDS = {"schema_version", "profile", "peer_hostname", "review_runtime", "filter_runtime",
           "observer_policy_sha256", "cascade_policy_sha256"}
_RUNTIME_FIELDS = {"model_path", "model_alias", "model_ftype", "build_info", "modalities", "total_slots"}


class TeamOuterwearProfileError(ObservationError):
    """An explicit profile cannot be accepted; there is no automatic fallback."""


def _require(condition, message):
    if not condition:
        raise TeamOuterwearProfileError(message)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _check_deadline(deadline):
    if deadline is not None:
        _require(_finite(deadline), "A finite absolute deadline is required")
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Team outerwear profile deadline exhausted")


def _object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "Duplicate profile JSON field")
        value[key] = item
    return value


def _constant(_value):
    raise TeamOuterwearProfileError("Nonfinite profile JSON value")


def _read_config(path, deadline):
    _check_deadline(deadline)
    _require(path.is_absolute(), "The team profile path must be absolute")
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not part.is_junction(), "Profile symlinks and junctions are forbidden")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), "The team profile must be a regular file")
    _require(0 < before.st_size <= MAX_CONFIG_BYTES, "The team profile must fit within 64 KiB")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 "The profile changed while opening")
        raw = stream.read(MAX_CONFIG_BYTES + 1)
        after = os.fstat(stream.fileno())
    final = path.lstat()
    fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    _require(fields(before) == fields(after) == fields(final) and not path.is_symlink(),
             "The profile changed while reading")
    _require(0 < len(raw) <= MAX_CONFIG_BYTES, "The team profile must fit within 64 KiB")
    _check_deadline(deadline)
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, TeamOuterwearProfileError):
            raise
        raise TeamOuterwearProfileError("Invalid profile JSON") from error
    _require(isinstance(value, dict) and set(value) == _FIELDS, "Unknown or missing team profile fields")
    return value


def load_configured_observer(*, deadline=None):
    """Return None only when explicitly unconfigured; never make HTTP requests.

    A present but invalid configuration raises. This helper does not verify a
    benchmark, enable any rule or change publication/transport recovery gates.
    """
    configured = os.environ.get(CONFIG_ENV)
    if not configured:
        return None
    value = _read_config(Path(configured), deadline)
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["profile"] == "private_team_outerwear_v1", "Unknown team profile version")
    _require(value["observer_policy_sha256"] == outerwear_policy_fingerprint()
             and value["cascade_policy_sha256"] == cascade_policy_fingerprint(), "Team profile policy changed")
    for name in ("filter_runtime", "review_runtime"):
        _require(isinstance(value[name], dict) and set(value[name]) == _RUNTIME_FIELDS,
                 "Each profile runtime must contain exactly six identity fields")
    expected = _identities({"filter4b": value["filter_runtime"], "review8b": value["review_runtime"]})
    _check_deadline(deadline)
    # The remote constructor validates the exact HTTPS host, hashed build
    # identity and peer protocol. It only creates a client, never connects.
    hostname = value["peer_hostname"]
    peer = TeamOuterwearWorkerClient(f"https://{hostname}:8443", expected_hostname=hostname,
                                    expected_runtime=value["review_runtime"], timeout=75)
    local = LocalModelClient("http://127.0.0.1:8081", timeout=20)
    observer = CascadeOuterwearObserver(local, review_worker=peer, expected_runtimes=expected)
    _check_deadline(deadline)
    return observer


def verify_configured_observer(observer, deadline):
    """Check both declared identities within one deadline; no retries/fallback.

    Returns the two canonical runtime snapshots only after both match. Each
    server_info request receives at most five seconds and the remaining budget;
    temporary timeout changes are restored even when the request raises.
    """
    _check_deadline(deadline)
    _require(deadline is not None, "Verification requires an absolute deadline")
    _require(type(observer) is CascadeOuterwearObserver and observer.review_worker is not None,
             "An explicit team cascade observer is required")
    expected = _identities(observer.expected_runtimes)
    _require(isinstance(observer.clients, dict) and set(observer.clients) == {"filter4b", "review8b"}
             and observer.clients["review8b"] is observer.review_worker, "Team observer clients are inconsistent")
    verified = {}
    for name in ("filter4b", "review8b"):
        _check_deadline(deadline)
        client = observer.clients[name]
        previous = client.timeout
        _require(_finite(previous) and previous > 0, "Invalid team client timeout")
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise ObservationDeadline("Team identity deadline exhausted before dispatch")
        client.timeout = min(previous, 5.0, remaining)
        try:
            current = canonical_runtime(client.server_info())
        finally:
            client.timeout = previous
        _check_deadline(deadline)
        _require(json.dumps(current, sort_keys=True, allow_nan=False)
                 == json.dumps(expected[name], sort_keys=True, allow_nan=False), "Team runtime identity changed")
        verified[name] = current
    _check_deadline(deadline)
    return deepcopy(verified)
