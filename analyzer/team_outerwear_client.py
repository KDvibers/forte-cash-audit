"""Inactive, narrow client for a team-owned private Tailscale observer.

This is not a remote model URL option for LocalModelClient. Only the original
outerwear observer protocol is allowed. Production does not import this module.
"""
from copy import deepcopy
import base64
import hashlib
import json
import math
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .observations import ObservationDeadline, ObservationError
from .outerwear import outerwear_policy_fingerprint
from .outerwear_cascade import _clean_original


_RUNTIME_KEYS = {"model_path", "model_alias", "model_ftype", "build_info", "modalities", "total_slots"}
_MODEL = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
_BODY_LIMIT = 65536


class TeamOuterwearError(ObservationError):
    """A private worker response cannot establish a usable observation."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TeamOuterwearError("Team worker redirects are forbidden")


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _require(condition, message):
    if not condition:
        raise TeamOuterwearError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON response field")
        result[key] = value
    return result


def _constant(_value):
    raise TeamOuterwearError("Nonfinite JSON response value")


def _same(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _runtime(value):
    _require(isinstance(value, dict) and set(value) == _RUNTIME_KEYS, "Invalid worker runtime fields")
    _require(value["model_path"] == _MODEL, "Wrong worker model")
    _require(isinstance(value["build_info"], str)
             and re.fullmatch(r"sha256:[0-9a-f]{64}", value["build_info"]), "Invalid worker build identity")
    _require(isinstance(value["modalities"], dict) and value["modalities"].get("vision") is True,
             "Worker does not declare vision support")
    _require(type(value["total_slots"]) is int and value["total_slots"] == 1, "Worker must have one slot")
    _require(isinstance(value["model_alias"], str)
             and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value["model_alias"]), "Invalid model alias")
    _require(isinstance(value["model_ftype"], str)
             and re.fullmatch(r"[A-Za-z0-9_ .()+-]{1,80}", value["model_ftype"]), "Invalid model quantization identity")
    _require(_same(value, value), "Runtime is not finite JSON")
    return deepcopy(value)


class TeamOuterwearWorkerClient:
    """One explicitly configured HTTPS peer; no retries or environment proxies.

    The operator supplies a runtime captured from the verified team deployment.
    TLS authenticates the hostname. The peer's private Serve authorizes the
    caller's Tailscale identity; no shared API key is placed in this client.
    """

    def __init__(self, base_url, *, expected_hostname, expected_runtime, timeout=75):
        _require(isinstance(expected_hostname, str) and re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.[a-z0-9-]+\.ts\.net", expected_hostname),
            "A fixed private Tailscale hostname is required")
        parsed = urllib.parse.urlsplit(base_url)
        _require(parsed.scheme == "https" and parsed.hostname == expected_hostname and parsed.port == 8443
                 and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
                 and parsed.path in ("", "/"), "Only the configured private HTTPS peer on 8443 is allowed")
        _require(_finite(timeout) and 0 < timeout <= 75, "Invalid worker timeout")
        self.base_url = f"https://{expected_hostname}:8443"
        self.expected_runtime = _runtime(expected_runtime)
        self.timeout = timeout
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def _request(self, path, *, deadline, payload=None):
        _require(_finite(deadline), "Finite absolute deadline is required")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ObservationDeadline("Team worker deadline exhausted before dispatch")
        data = None if payload is None else json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json", "Accept": "application/json"},
                                         method="GET" if payload is None else "POST")
        try:
            # A failed/unknown POST is never replayed. Its completion is unknown.
            with self.opener.open(request, timeout=min(self.timeout, remaining)) as response:
                _require(response.status == 200, "Unexpected worker HTTP status")
                _require(response.headers.get_content_type() == "application/json", "Worker response is not JSON")
                lengths = response.headers.get_all("Content-Length", [])
                _require(len(lengths) <= 1, "Duplicate worker Content-Length")
                if lengths:
                    _require(lengths[0].isdigit() and int(lengths[0]) <= _BODY_LIMIT, "Worker response too large")
                # The worker sends a fixed-length body. HTTPResponse's chunked
                # read1 may perform multiple reads while parsing chunk framing,
                # so it cannot provide the same remaining-budget bound.
                _require(not getattr(response, "chunked", False), "Chunked worker responses are unsupported")
                # CPython's HTTP(S)Response wraps SocketIO (also for TLS).
                # Refuse unfamiliar adapters rather than silently using an
                # unbounded read. This covers the body, not DNS/header reads.
                body_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                _require(callable(getattr(body_socket, "settimeout", None))
                         and callable(getattr(response, "read1", None)), "Unsupported worker body transport")
                chunks, size = [], 0
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise ObservationDeadline("Team worker response exceeded the deadline")
                    body_socket.settimeout(min(self.timeout, remaining))
                    part = response.read1(min(16384, _BODY_LIMIT + 1 - size))
                    if time.perf_counter() >= deadline:
                        raise ObservationDeadline("Team worker response exceeded the deadline")
                    if not part:
                        break
                    size += len(part)
                    _require(size <= _BODY_LIMIT, "Worker response too large")
                    chunks.append(part)
                    if lengths and size == int(lengths[0]):
                        # HTTPResponse may close its SocketIO at the declared
                        # end; no further socket operation or EOF read is needed.
                        break
                if lengths:
                    _require(size == int(lengths[0]), "Worker response was truncated")
        except urllib.error.HTTPError as error:
            raise TeamOuterwearError(f"Team worker refused request (HTTP {error.code}); no retry") from error
        except (TimeoutError, urllib.error.URLError) as error:
            # A response may be lost while the worker still runs. Fail closed.
            raise ObservationDeadline("Team worker transport did not establish completion") from error
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Team worker response arrived after the deadline")
        try:
            result = json.loads(b"".join(chunks), object_pairs_hook=_object, parse_constant=_constant)
        except (ValueError, UnicodeError) as error:
            raise TeamOuterwearError("Invalid worker JSON response") from error
        _require(isinstance(result, dict), "Worker response must be an object")
        return result

    def server_info(self):
        health = self._request("/health", deadline=time.perf_counter() + min(5, self.timeout))
        _require(set(health) == {"status", "protocol_version", "runtime", "observer_policy_sha256"},
                 "Unknown worker health fields")
        _require(type(health["protocol_version"]) is int and health["protocol_version"] == 1
                 and health["status"] == "ok"
                 and health["observer_policy_sha256"] == outerwear_policy_fingerprint(),
                 "Worker is not ready with the expected observer policy")
        runtime = _runtime(health["runtime"])
        _require(_same(runtime, self.expected_runtime), "Worker runtime identity changed")
        return runtime

    def observe(self, frame, frame_id, *, deadline):
        _require(_finite(deadline), "Finite absolute deadline is required")
        _require(isinstance(frame_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", frame_id), "Invalid frame ID")
        _require(_finite(frame.timestamp_sec), "Invalid frame timestamp")
        _require(isinstance(frame.jpeg, bytes) and 0 < len(frame.jpeg) <= 2 * 1024 * 1024, "Invalid JPEG byte size")
        _require(all(type(v) is int and 1 <= v <= 1280 for v in (frame.width, frame.height)), "Invalid frame dimensions")
        remaining = min(self.timeout, 75, deadline - time.perf_counter())
        if remaining <= 0:
            raise ObservationDeadline("Team worker deadline exhausted before dispatch")
        request_id, digest = uuid.uuid4().hex, hashlib.sha256(frame.jpeg).hexdigest()
        payload = {"protocol_version": 1, "request_id": request_id, "frame_id": frame_id,
                   "timestamp_sec": frame.timestamp_sec, "image_base64": base64.b64encode(frame.jpeg).decode("ascii"),
                   "image_sha256": digest, "width": frame.width, "height": frame.height, "budget_sec": remaining}
        value = self._request("/outerwear/check", deadline=deadline, payload=payload)
        _require(set(value) == {"protocol_version", "request_id", "frame_id", "timestamp_sec", "image_sha256", "runtime", "record"},
                 "Unknown worker observation fields")
        _require(type(value["protocol_version"]) is int and value["protocol_version"] == 1
                 and value["request_id"] == request_id and value["frame_id"] == frame_id
                 and _finite(value["timestamp_sec"]) and value["timestamp_sec"] == frame.timestamp_sec
                 and value["image_sha256"] == digest, "Worker response does not identify this request and frame")
        _require(_same(_runtime(value["runtime"]), self.expected_runtime), "Worker runtime identity changed")
        _clean_original(value["record"], frame_id, frame.timestamp_sec)
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Team worker validation exceeded the deadline")
        return deepcopy(value["record"])
