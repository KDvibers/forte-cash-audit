"""Inactive experimental team worker; starting it does not configure Serve.

Only a trusted Tailscale Serve HTTP proxy may reach this loopback listener.
Local processes are in the trust boundary. No production profile imports this
module, and the caller must configure private access separately after testing.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import warnings

from PIL import Image

from .local_client import LocalModelClient
from .outerwear import OuterwearObserver, outerwear_policy_fingerprint
from .outerwear_cascade import _clean_original
from .video import VideoFrame


PROTOCOL_VERSION = 1
MODEL_NAME = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MODEL_URL = "http://127.0.0.1:8092"
MAX_BODY_BYTES = 3 * 1024 * 1024
MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_BUDGET_SEC = 75.0
MAX_REQUEST_IDS = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_FIELDS = {"protocol_version", "request_id", "frame_id", "timestamp_sec", "image_base64",
           "image_sha256", "width", "height", "budget_sec"}


class WorkerError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


def _number(value, *, positive=False):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_payload(raw):
    try:
        def invalid_constant(value):
            raise ValueError("Nonfinite JSON value")
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise WorkerError(400, "invalid_json") from None


def sanitize_runtime(info):
    """Retain declared model identity, never filesystem or account details."""
    if not isinstance(info, dict):
        raise ValueError("Invalid model metadata")
    name = str(info.get("model_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
    modalities = info.get("modalities")
    if (name != MODEL_NAME or info.get("model_alias") != "forte-qwen"
            or info.get("model_ftype") != "Q4_K - Medium"
            or not isinstance(info.get("build_info"), str) or not info["build_info"]
            or len(info["build_info"]) > 1024
            or not isinstance(modalities, dict) or modalities.get("vision") is not True
            or any(key not in {"vision", "video", "audio", "text"} or type(value) is not bool
                   for key, value in modalities.items())
            or type(info.get("total_slots")) is not int or info["total_slots"] != 1):
        raise ValueError("Unexpected local worker model identity")
    return {"model_path": name, "model_alias": "forte-qwen", "model_ftype": "Q4_K - Medium",
            "build_info": "sha256:" + hashlib.sha256(info["build_info"].encode("utf-8")).hexdigest(),
            "modalities": dict(modalities), "total_slots": 1}


def validate_payload(payload):
    if (not isinstance(payload, dict) or set(payload) != _FIELDS
            or type(payload["protocol_version"]) is not int or payload["protocol_version"] != 1
            or not isinstance(payload["request_id"], str) or not _HEX32.fullmatch(payload["request_id"])
            or not isinstance(payload["frame_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", payload["frame_id"])
            or not _number(payload["timestamp_sec"])
            or not _number(payload["budget_sec"], positive=True) or payload["budget_sec"] > MAX_BUDGET_SEC
            or not isinstance(payload["image_sha256"], str) or not _HEX64.fullmatch(payload["image_sha256"])
            or any(type(payload[key]) is not int or not 1 <= payload[key] <= 1280 for key in ("width", "height"))
            or not isinstance(payload["image_base64"], str)):
        raise WorkerError(400, "invalid_request")
    encoded = payload["image_base64"]
    if len(encoded) > 4 * ((MAX_JPEG_BYTES+2)//3):
        raise WorkerError(413, "image_too_large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise WorkerError(400, "invalid_image") from None
    if not raw or len(raw) > MAX_JPEG_BYTES:
        raise WorkerError(413, "image_too_large")
    if hashlib.sha256(raw).hexdigest() != payload["image_sha256"]:
        raise WorkerError(400, "image_hash_mismatch")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as picture:
                if picture.format != "JPEG" or picture.size != (payload["width"], payload["height"]):
                    raise ValueError("Wrong image format or dimensions")
                picture.verify()
            with Image.open(BytesIO(raw)) as picture:
                picture.load()
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise WorkerError(400, "invalid_image") from None
    return VideoFrame(payload["timestamp_sec"], raw, payload["width"], payload["height"])


def _clean_record(record, frame, frame_id):
    # Client, cascade and worker share the same journal validation policy.
    try:
        _clean_original(record, frame_id, frame.timestamp_sec)
        return True
    except (KeyError, TypeError, ValueError):
        return False


class TeamOuterwearWorker:
    """One observer; all real dependencies can be replaced by CPU test fakes."""
    def __init__(self, observer, runtime, allowed_login_sha256, *, identity_getter, clock=time.perf_counter):
        allowed = frozenset(allowed_login_sha256)
        if not allowed or any(not isinstance(item, str) or not _HEX64.fullmatch(item) for item in allowed):
            raise ValueError("A nonempty SHA256 identity allowlist is required")
        self.observer, self.runtime = observer, sanitize_runtime(runtime)
        self.allowed = allowed
        self.identity_getter, self.clock = identity_getter, clock
        self._mutex = threading.Lock()
        self._busy, self._quarantined = False, False
        self._seen = set()

    def authorize(self, headers):
        values = headers.get_all("Tailscale-User-Login", [])
        if headers.get_all("Tailscale-Funnel-Request", []):
            raise WorkerError(403, "forbidden")
        if (len(values) != 1 or not values[0] or not values[0].isascii()
                or values[0] != values[0].strip()
                or any(ord(char) < 33 or ord(char) > 126 for char in values[0])
                or hashlib.sha256(values[0].encode("ascii")).hexdigest() not in self.allowed):
            raise WorkerError(403, "forbidden")

    def quarantine(self):
        with self._mutex:
            self._quarantined = True

    def health(self):
        with self._mutex:
            status = "quarantined" if self._quarantined else "busy" if self._busy else "ok"
        return {"status": status, "protocol_version": 1, "runtime": deepcopy(self.runtime),
                "observer_policy_sha256": outerwear_policy_fingerprint()}

    def check(self, payload, *, started=None):
        began = self.clock() if started is None else started
        frame = validate_payload(payload)
        deadline = began + payload["budget_sec"]
        if self.clock() >= deadline:
            raise WorkerError(504, "deadline_exceeded")
        request_id = payload["request_id"]
        with self._mutex:
            if request_id in self._seen:
                raise WorkerError(409, "request_replayed")
            if self._quarantined:
                raise WorkerError(503, "quarantined")
            if self._busy:
                raise WorkerError(429, "busy")
            if len(self._seen) >= MAX_REQUEST_IDS:
                self._quarantined = True
                raise WorkerError(503, "replay_capacity_exhausted")
            self._seen.add(request_id)
            self._busy = True
        try:
            current = sanitize_runtime(self.identity_getter(deadline=deadline))
            if current != self.runtime:
                raise WorkerError(503, "model_identity_changed")
            if self.clock() >= deadline:
                raise WorkerError(504, "deadline_exceeded")
            record = self.observer.observe(frame, payload["frame_id"], deadline=deadline)
            if self.clock() >= deadline:
                raise WorkerError(504, "deadline_exceeded")
            if not _clean_record(record, frame, payload["frame_id"]):
                raise WorkerError(503, "incomplete_observation")
            result = {"protocol_version": 1, "request_id": request_id, "frame_id": payload["frame_id"],
                      "timestamp_sec": frame.timestamp_sec, "image_sha256": payload["image_sha256"],
                      "runtime": deepcopy(self.runtime), "record": record}
            if len(_json_bytes(result)) > MAX_RESPONSE_BYTES:
                raise WorkerError(503, "invalid_response")
            if self.clock() >= deadline:
                raise WorkerError(504, "deadline_exceeded")
            return result
        except Exception as error:
            self.quarantine()
            if isinstance(error, WorkerError):
                raise
            if isinstance(error, TimeoutError):
                raise WorkerError(504, "deadline_exceeded") from None
            raise WorkerError(503, "worker_error") from None
        finally:
            with self._mutex:
                self._busy = False


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        # Never emit arbitrary exceptions, paths, request bodies or identity.
        pass


def make_server(worker, *, host="127.0.0.1", port=8093):
    if host != "127.0.0.1" or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("The worker must listen on IPv4 loopback only")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass

        def send_error(self, code, message=None, explain=None):
            self._reply(code, {"protocol_version": 1, "error": "invalid_http"})

        def _reply(self, status, value):
            encoded = _json_bytes(value)
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, error):
            self._reply(error.status, {"protocol_version": 1, "error": error.code})

        def do_GET(self):
            try:
                worker.authorize(self.headers)
                if self.path != "/health":
                    raise WorkerError(404, "not_found")
                health = worker.health()
                self._reply(503 if health["status"] == "quarantined" else 200, health)
            except WorkerError as error:
                self._error(error)
            except (OSError, TimeoutError):
                self.close_connection = True

        def do_POST(self):
            started = worker.clock()
            try:
                worker.authorize(self.headers)
                if self.path != "/outerwear/check":
                    raise WorkerError(404, "not_found")
                lengths = self.headers.get_all("Content-Length", [])
                types = self.headers.get_all("Content-Type", [])
                if (self.headers.get_all("Transfer-Encoding", [])
                        or self.headers.get_all("Content-Encoding", [])
                        or len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0])
                        or len(types) != 1 or types[0].lower() not in
                            {"application/json", "application/json; charset=utf-8"}):
                    raise WorkerError(400, "invalid_framing")
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY_BYTES:
                    raise WorkerError(413, "body_too_large")
                data = bytearray()
                while len(data) < length:
                    remaining = started + MAX_BUDGET_SEC - worker.clock()
                    if remaining <= 0:
                        raise WorkerError(504, "deadline_exceeded")
                    self.connection.settimeout(min(10, remaining))
                    chunk = self.rfile.read1(min(65536, length-len(data)))
                    if not chunk:
                        raise WorkerError(400, "incomplete_body")
                    data.extend(chunk)
                payload = parse_payload(bytes(data))
                result = worker.check(payload, started=started)
                if worker.clock() >= started+payload["budget_sec"]:
                    worker.quarantine()
                    raise WorkerError(504, "deadline_exceeded")
                self._reply(200, result)
            except WorkerError as error:
                self._error(error)
            except (TimeoutError, socket.timeout):
                self._error(WorkerError(408, "body_timeout"))
            except OSError:
                self.close_connection = True

    return _Server((host, port), Handler)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Local model redirect refused")


def build_local_worker(allowed_login_sha256):
    """Explicit startup probe only; never launches or downloads a model."""
    client = LocalModelClient(MODEL_URL, timeout=20)
    client.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def identity(*, deadline):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Identity deadline")
        previous = client.timeout
        client.timeout = min(previous, remaining)
        try:
            result = client.server_info()
        finally:
            client.timeout = previous
        if time.perf_counter() >= deadline:
            raise TimeoutError("Identity deadline")
        return result

    runtime = identity(deadline=time.perf_counter()+5)
    return TeamOuterwearWorker(OuterwearObserver(client), runtime, allowed_login_sha256,
                               identity_getter=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--allowed-login-sha256", action="append", required=True,
                        help="SHA256 of the exact ASCII Serve Login; no email or API secret")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.port == 8092:
        parser.error("Choose a separate local port between 1024 and 65535")
    try:
        worker = build_local_worker(args.allowed_login_sha256)
        with make_server(worker, port=args.port) as server:
            print("Experimental loopback worker ready; private Serve is not configured by this command.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return
    except Exception:
        print("Worker startup or execution failed; no model or network configuration was changed.", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
