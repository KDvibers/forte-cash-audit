"""CPU fixtures only: no real model, Tailscale daemon or production listener."""
import base64
from copy import deepcopy
from email.message import Message
import hashlib
import http.client
from io import BytesIO
import json
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

from PIL import Image

from analyzer import team_outerwear_worker as module
from analyzer.outerwear import OuterwearObserver, outerwear_policy_fingerprint
from tests.test_outerwear import Client, Clock, primary, inventory


LOGIN = "allowed@example.invalid"
LOGIN_HASH = hashlib.sha256(LOGIN.encode("ascii")).hexdigest()
RUNTIME = {"model_path": "C:/synthetic/private/model/" + module.MODEL_NAME,
           "model_alias": "forte-qwen", "model_ftype": "Q4_K - Medium",
           "build_info": "b11669-synthetic", "modalities": {"vision": True, "video": True, "audio": False},
           "total_slots": 1}


def payload(request_id="1"*32, *, image_format="JPEG"):
    output = BytesIO()
    Image.new("RGB", (16, 12), (0, 70, 90)).save(output, image_format)
    raw = output.getvalue()
    return {"protocol_version": 1, "request_id": request_id, "frame_id": "outerwear_frame_004",
            "timestamp_sec": 4.25, "image_base64": base64.b64encode(raw).decode("ascii"),
            "image_sha256": hashlib.sha256(raw).hexdigest(), "width": 16, "height": 12, "budget_sec": 20}


def headers(*pairs):
    value = Message()
    for key, item in pairs:
        value[key] = item
    return value


class WorkerCoreTests(unittest.TestCase):
    def worker(self, *, responses=None, clock=None, observer=None, identity_getter=None):
        client = Client([primary("no")] if responses is None else responses)
        getter = identity_getter or Mock(return_value=deepcopy(RUNTIME))
        worker = module.TeamOuterwearWorker(observer or OuterwearObserver(client), deepcopy(RUNTIME),
                     [LOGIN_HASH], identity_getter=getter, clock=clock or time.perf_counter)
        return worker, client, getter

    def assert_worker_error(self, status, function, *args, **kwargs):
        with self.assertRaises(module.WorkerError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.status, status)
        return caught.exception.code

    def test_runtime_contains_only_sanitized_declared_identity(self):
        worker, _, getter = self.worker()
        info = worker.health()
        self.assertEqual(set(info), {"status", "protocol_version", "runtime", "observer_policy_sha256"})
        self.assertEqual(set(info["runtime"]), set(RUNTIME))
        self.assertEqual(info["runtime"]["model_path"], module.MODEL_NAME)
        self.assertNotIn("private", json.dumps(info))
        self.assertRegex(info["runtime"]["build_info"], r"^sha256:[a-f0-9]{64}$")
        getter.assert_not_called()
        info["runtime"]["modalities"]["vision"] = False
        self.assertTrue(worker.health()["runtime"]["modalities"]["vision"])

    def test_unexpected_model_slots_vision_or_alias_rejected_at_construction(self):
        for change in ({"model_path": "other.gguf"}, {"total_slots": 2}, {"total_slots": True},
                       {"modalities": {"vision": False}}, {"model_alias": "private@example.invalid"},
                       {"build_info": None}, {"model_ftype": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.sanitize_runtime({**RUNTIME, **change})
        for allowed in ([], [LOGIN], ["a"*63]):
            with self.assertRaises(ValueError):
                module.TeamOuterwearWorker(None, RUNTIME, allowed, identity_getter=lambda **kw: RUNTIME)

    def test_identity_is_exact_and_missing_duplicate_funnel_or_nonascii_is_denied(self):
        worker, _, _ = self.worker()
        worker.authorize(headers(("tailscale-user-login", LOGIN)))
        rejected = [headers(), headers(("Tailscale-User-Login", LOGIN.upper())),
                    headers(("Tailscale-User-Login", "неизвестный")),
                    headers(("Tailscale-User-Login", LOGIN), ("Tailscale-User-Login", LOGIN)),
                    headers(("Tailscale-User-Login", LOGIN), ("Tailscale-Funnel-Request", "")),
                    headers(("Tailscale-User-Login", LOGIN + " "))]
        for index, value in enumerate(rejected):
            with self.subTest(index=index):
                self.assert_worker_error(403, worker.authorize, value)

    def test_valid_frame_keeps_bytes_pts_hash_record_and_shared_deadline(self):
        worker, client, getter = self.worker(responses=[primary("yes"), inventory()])
        request = payload()
        started = time.perf_counter()
        result = worker.check(request, started=started)
        self.assertEqual(set(result), {"protocol_version", "request_id", "frame_id", "timestamp_sec",
                                     "image_sha256", "runtime", "record"})
        self.assertEqual(result["record"]["state"], "yes")
        self.assertEqual(result["record"]["policy_sha256"], outerwear_policy_fingerprint())
        self.assertEqual(result["image_sha256"], request["image_sha256"])
        self.assertEqual(client.calls[0]["frame"].timestamp_sec, request["timestamp_sec"])
        self.assertEqual(client.calls[0]["frame"].jpeg, base64.b64decode(request["image_base64"]))
        self.assertEqual(getter.call_args.kwargs["deadline"], started + request["budget_sec"])
        self.assertEqual(worker.health()["status"], "ok")

    def test_clean_uncertain_remains_uncertain_and_healthy(self):
        worker, _, _ = self.worker(responses=[primary("uncertain")])
        self.assertEqual(worker.check(payload())["record"]["state"], "uncertain")
        self.assertEqual(worker.health()["status"], "ok")

    def test_bad_fields_hash_dims_png_truncated_or_huge_values_never_infer(self):
        worker, client, getter = self.worker()
        base = payload()
        cases = [{**base, "prompt": "injected"}, {**base, "url": "https://example.invalid"},
                 {**base, "timestamp_sec": True}, {**base, "timestamp_sec": 10**400},
                 {**base, "timestamp_sec": float("nan")}, {**base, "budget_sec": 76},
                 {**base, "budget_sec": 0}, {**base, "width": True}, {**base, "width": 1281},
                 {**base, "width": 17}, {**base, "request_id": "wrong"},
                 {**base, "frame_id": "../private/file"}, {**base, "image_sha256": "a"*64},
                 {**base, "image_base64": "!!!"}, payload(image_format="PNG")]
        raw = base64.b64decode(base["image_base64"])[:-50]
        cases.append({**base, "image_base64": base64.b64encode(raw).decode(),
                      "image_sha256": hashlib.sha256(raw).hexdigest()})
        for index, request in enumerate(cases):
            with self.subTest(index=index):
                self.assert_worker_error(400, worker.check, request)
        self.assert_worker_error(413, worker.check, {**base, "image_base64": "A"*(3*1024*1024)})
        self.assertEqual(client.calls, [])
        getter.assert_not_called()
        self.assertEqual(worker.health()["status"], "ok")

    def test_duplicate_json_and_nonfinite_json_are_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{'):
            self.assert_worker_error(400, module.parse_payload, raw)

    def test_replay_is_not_a_second_inference(self):
        worker, client, _ = self.worker()
        request = payload()
        worker.check(request)
        self.assert_worker_error(409, worker.check, request)
        self.assertEqual(len(client.calls), 1)

    def test_concurrent_call_is_busy_and_inflight_request_id_is_replayed(self):
        entered, release = threading.Event(), threading.Event()
        real = OuterwearObserver(Client([primary("no")]))
        def blocking(frame, frame_id, **kw):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("fixture did not release")
            return real.observe(frame, frame_id, **kw)
        worker, _, _ = self.worker(observer=SimpleNamespace(observe=blocking))
        answers = []
        thread = threading.Thread(target=lambda: answers.append(worker.check(payload())))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(worker.health()["status"], "busy")
            self.assert_worker_error(429, worker.check, payload("2"*32))
            self.assert_worker_error(409, worker.check, payload())
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(answers), 1)
        self.assertEqual(worker.health()["status"], "ok")

    def test_changed_runtime_quarantines_before_observer(self):
        worker, client, _ = self.worker(identity_getter=lambda **kw: {**RUNTIME, "build_info": "changed"})
        self.assert_worker_error(503, worker.check, payload())
        self.assertEqual(client.calls, [])
        self.assertEqual(worker.health()["status"], "quarantined")
        self.assert_worker_error(503, worker.check, payload("2"*32))

    def test_identity_timeout_and_schema_failure_remain_unhealthy(self):
        getter = Mock(side_effect=TimeoutError("private details must not escape"))
        worker, client, _ = self.worker(identity_getter=getter)
        self.assert_worker_error(504, worker.check, payload())
        self.assertEqual(client.calls, [])
        self.assertEqual(worker.health()["status"], "quarantined")
        worker, _, _ = self.worker(responses=[primary("bad enum")])
        self.assert_worker_error(503, worker.check, payload())
        self.assertEqual(worker.health()["status"], "quarantined")

    def test_identity_or_observer_late_answer_never_publishes_positive(self):
        clock = Clock()
        def late_identity(**kw):
            clock.now += 20
            return RUNTIME
        worker, client, _ = self.worker(clock=clock, identity_getter=late_identity)
        self.assert_worker_error(504, worker.check, payload())
        self.assertEqual(client.calls, [])
        self.assertEqual(worker.health()["status"], "quarantined")
        clock = Clock()
        real = OuterwearObserver(Client([primary("yes"), inventory()]))
        def late_observer(frame, frame_id, **kw):
            record = real.observe(frame, frame_id)
            clock.now += 20
            return record
        worker, _, _ = self.worker(clock=clock, observer=SimpleNamespace(observe=late_observer))
        self.assert_worker_error(504, worker.check, payload())
        self.assertEqual(worker.health()["status"], "quarantined")

    def test_corrupted_frame_record_is_not_an_envelope(self):
        real = OuterwearObserver(Client([primary("no")]))
        def wrong(frame, frame_id, **kw):
            record = real.observe(frame, frame_id, **kw)
            record["frame_id"] = "other"
            return record
        worker, _, _ = self.worker(observer=SimpleNamespace(observe=wrong))
        self.assert_worker_error(503, worker.check, payload())
        self.assertEqual(worker.health()["status"], "quarantined")

    def test_shared_validator_rejects_extra_fields_trace_disagreement_and_partial_calls(self):
        def extra(record):
            record["untracked_pass"] = "forbidden"
        def partial(record):
            record["calls"][0]["completed"] = False
        def wrong_text(record):
            record["calls"][0]["response"]["response_text"] = '{"removed_outerwear":"yes"}'
        def unknown_call(record):
            record["calls"][0]["additional_data"] = "forbidden"
        def extra_response(record):
            record["calls"][0]["response"]["file_path"] = "forbidden"
        for mutate in (extra, partial, wrong_text, unknown_call, extra_response):
            with self.subTest(mutate=mutate.__name__):
                real = OuterwearObserver(Client([primary("no")]))
                def corrupt(frame, frame_id, **kw):
                    record = real.observe(frame, frame_id, **kw)
                    mutate(record)
                    return record
                worker, _, _ = self.worker(observer=SimpleNamespace(observe=corrupt))
                self.assert_worker_error(503, worker.check, payload())
                self.assertEqual(worker.health()["status"], "quarantined")

    def test_identity_is_checked_again_for_each_new_request(self):
        getter = Mock(side_effect=[deepcopy(RUNTIME), {**RUNTIME, "build_info": "changed"}])
        worker, client, _ = self.worker(identity_getter=getter)
        worker.check(payload())
        self.assert_worker_error(503, worker.check, payload("2"*32))
        self.assertEqual(getter.call_count, 2)
        self.assertEqual(len(client.calls), 1)

    def test_replay_capacity_fails_closed_without_evicting_old_ids(self):
        worker, client, _ = self.worker()
        with patch.object(module, "MAX_REQUEST_IDS", 1):
            worker.check(payload())
            self.assert_worker_error(503, worker.check, payload("2"*32))
            self.assert_worker_error(409, worker.check, payload())
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(worker.health()["status"], "quarantined")

    def test_jpeg_decompression_warning_is_a_rejection_before_identity(self):
        worker, client, getter = self.worker()
        with patch.object(module.Image, "open", side_effect=Image.DecompressionBombWarning("fixture")):
            self.assert_worker_error(400, worker.check, payload())
        getter.assert_not_called()
        self.assertEqual(client.calls, [])

    def test_local_startup_is_fixed_loopback_and_redirects_are_refused(self):
        client = Mock(timeout=20)
        client.server_info.return_value = deepcopy(RUNTIME)
        with patch.object(module, "LocalModelClient", return_value=client) as factory:
            worker = module.build_local_worker([LOGIN_HASH])
        factory.assert_called_once_with("http://127.0.0.1:8092", timeout=20)
        client.server_info.assert_called_once()
        self.assertEqual(worker.health()["status"], "ok")
        with self.assertRaises(urllib.error.URLError):
            module._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid")


class WorkerHTTPTests(unittest.TestCase):
    def setUp(self):
        self.client = Client([primary("no"), primary("uncertain")])
        self.getter = Mock(return_value=deepcopy(RUNTIME))
        self.worker = module.TeamOuterwearWorker(OuterwearObserver(self.client), RUNTIME,
                         [LOGIN_HASH], identity_getter=self.getter)
        self.server = module.make_server(self.worker, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, body=None, extra=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        try:
            connection.request(method, path, body, {"Tailscale-User-Login": LOGIN,
                                                  "Content-Type": "application/json", **(extra or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def raw_headers_only(self, lines):
        # No body is sent; auth/framing rejection must answer without waiting.
        connection = socket.create_connection(self.server.server_address, timeout=2)
        try:
            connection.sendall(("POST /outerwear/check HTTP/1.1\r\nHost: fixture\r\n" +
                                "\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
            response = http.client.HTTPResponse(connection)
            response.begin()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_and_clean_response_use_exact_protocol_and_no_inference_for_health(self):
        status, health = self.request("GET", "/health")
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.getter.assert_not_called()
        request = payload()
        status, result = self.request("POST", "/outerwear/check", json.dumps(request))
        self.assertEqual(status, 200)
        self.assertEqual(result["request_id"], request["request_id"])
        self.assertEqual(result["runtime"], health["runtime"])
        self.assertEqual(result["record"]["state"], "no")
        status, error = self.request("POST", "/outerwear/check", json.dumps(request))
        self.assertEqual((status, error["error"]), (409, "request_replayed"))

    def test_auth_denial_precedes_reading_even_a_large_or_invalid_framed_body(self):
        cases = [[], ["Tailscale-User-Login: unknown@example.invalid"],
                 ["Tailscale-User-Login: " + LOGIN, "Tailscale-Funnel-Request: ?1"],
                 ["Tailscale-User-Login: " + LOGIN, "Tailscale-User-Login: " + LOGIN]]
        for extra in cases:
            with self.subTest(extra=extra):
                status, body = self.raw_headers_only(extra + ["Content-Length: 999999999", "Transfer-Encoding: chunked"])
                self.assertEqual((status, body), (403, {"protocol_version": 1, "error": "forbidden"}))
        self.assertEqual(self.client.calls, [])

    def test_oversize_and_ambiguous_framing_are_rejected_without_body(self):
        common = ["Tailscale-User-Login: " + LOGIN, "Content-Type: application/json"]
        cases = [(413, ["Content-Length: " + str(module.MAX_BODY_BYTES+1)]),
                 (400, ["Content-Length: 10", "Content-Length: 10"]),
                 (400, ["Content-Length: 10", "Transfer-Encoding: chunked"]),
                 (400, ["Content-Length: 10", "Content-Encoding: gzip"]),
                 (400, [])]
        for expected, extra in cases:
            with self.subTest(extra=extra):
                status, _ = self.raw_headers_only(common + extra)
                self.assertEqual(status, expected)
        self.assertEqual(self.client.calls, [])

    def test_unknown_route_json_duplicates_and_no_identity_health_are_refused(self):
        self.assertEqual(self.request("GET", "/health?secret=x")[0], 404)
        self.assertEqual(self.request("POST", "/anything", "{}")[0], 404)
        self.assertEqual(self.request("POST", "/outerwear/check", '{"x":1,"x":2}')[0], 400)
        self.assertEqual(self.request("GET", "/health", extra={"Tailscale-User-Login": ""})[0], 403)
        self.assertEqual(self.client.calls, [])

    def test_quarantined_health_is_not_http_healthy(self):
        self.worker.quarantine()
        status, body = self.request("GET", "/health")
        self.assertEqual((status, body["status"]), (503, "quarantined"))
        self.assertEqual(self.request("POST", "/outerwear/check", json.dumps(payload()))[0], 503)

    def test_listener_cannot_bind_lan_or_all_interfaces(self):
        for host in ("0.0.0.0", "::", "192.0.2.1", "localhost", "::1"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                module.make_server(self.worker, host=host, port=0)


if __name__ == "__main__":
    unittest.main()
