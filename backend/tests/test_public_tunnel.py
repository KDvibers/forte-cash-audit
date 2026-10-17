"""Public launcher ownership and failure paths, without network or subprocesses."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import doctor, public_tunnel as tunnel
from backend.config import Settings


ORIGIN = "https://synthetic-test.trycloudflare.com"
SECRET = "private-test-value-never-print"


class FakeProcess:
    def __init__(self, output="", events=None, stubborn=False):
        self.stdout = io.StringIO(output)
        self.events = events if events is not None else []
        self.returncode = None
        self.stubborn = stubborn

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("terminate-tunnel")
        if not self.stubborn:
            self.returncode = 0

    def wait(self, timeout):
        self.events.append(("wait-tunnel", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("synthetic-process", timeout)
        return self.returncode

    def kill(self):
        self.events.append("kill-tunnel")
        self.returncode = -9


class InlineThread:
    def __init__(self, target=None, kwargs=None, **options):
        self.target = target
        self.kwargs = kwargs or {}
        self.options = options

    def start(self):
        if self.target:
            self.target(**self.kwargs)

    def join(self, timeout):
        pass

    def is_alive(self):
        return True


def fast_clock(monkeypatch):
    counter = itertools.count()
    monkeypatch.setattr(tunnel.time, "monotonic", lambda: next(counter))
    monkeypatch.setattr(tunnel.time, "sleep", lambda seconds: None)


def test_binary_checksum_rejects_modified_or_missing_executable(tmp_path, monkeypatch):
    binary = tmp_path / "cloudflared.exe"
    original = b"synthetic executable, never executed"
    binary.write_bytes(original)
    monkeypatch.setattr(tunnel, "CLOUDFLARED_SHA256", hashlib.sha256(original).hexdigest())
    assert tunnel.verified_binary(binary) == binary.resolve()
    binary.write_bytes(original + b"modified")
    with pytest.raises(tunnel.LaunchError, match="Контрольная сумма"):
        tunnel.verified_binary(binary)
    with pytest.raises(tunnel.LaunchError, match="Не найден"):
        tunnel.verified_binary(tmp_path / "missing.exe")


def test_tunnel_does_not_inherit_account_tokens_or_application_secrets(monkeypatch):
    for name in ("TUNNEL_TOKEN", "TUNNEL_HTTP_HOST_HEADER", "CLOUDFLARED_CONFIG", "CASH_AUDIT_API_KEY"):
        monkeypatch.setenv(name, SECRET)
    monkeypatch.setenv("PUBLIC_LAUNCH_TEST_PATH", "preserved")
    environment = tunnel.tunnel_environment()
    assert not any(name.upper().startswith(("TUNNEL_", "CLOUDFLARED_", "CASH_AUDIT_")) for name in environment)
    assert SECRET not in environment.values()
    assert environment["PUBLIC_LAUNCH_TEST_PATH"] == "preserved"


@pytest.mark.parametrize("value", [
    ORIGIN + ".attacker.example", ORIGIN + "-attacker", ORIGIN + "@attacker.example",
    ORIGIN + "%2eattacker.example", "https://synthetic-test.trycloudflare.com_attacker",
])
def test_tunnel_url_parser_does_not_accept_a_trusted_prefix_of_foreign_url(value):
    assert tunnel.QUICK_URL.search(value) is None


def test_tunnel_uses_explicit_loopback_config_and_reaps_only_owned_process(tmp_path, monkeypatch):
    process = FakeProcess("INF Tunnel ready: " + ORIGIN + "\n", stubborn=True)
    calls = []

    def spawn(arguments, **options):
        calls.append((arguments, options))
        return process

    monkeypatch.setattr(tunnel.subprocess, "Popen", spawn)
    monkeypatch.setattr(tunnel.threading, "Thread", InlineThread)
    with pytest.raises(RuntimeError, match="caller stopped"):
        with tunnel.tunnel_process(Path("verified.exe"), 8010, tmp_path / "run") as (origin, owned):
            assert origin == ORIGIN
            assert owned is process
            raise RuntimeError("caller stopped")
    arguments, options = calls[0]
    assert arguments[arguments.index("--url") + 1] == "http://127.0.0.1:8010"
    assert arguments[arguments.index("--metrics") + 1] == "127.0.0.1:0"
    assert "--no-autoupdate" in arguments
    assert Path(arguments[arguments.index("--config") + 1]).read_text(encoding="utf-8") == "{}\n"
    assert options["stdout"] == subprocess.PIPE
    assert process.events == ["terminate-tunnel", ("wait-tunnel", 10), "kill-tunnel", ("wait-tunnel", 5)]
    assert process.stdout.closed


def test_tunnel_failure_before_url_is_not_treated_as_ready(tmp_path, monkeypatch):
    process = FakeProcess("INF connection unavailable\n")
    process.returncode = 1
    monkeypatch.setattr(tunnel.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(tunnel.threading, "Thread", InlineThread)
    with pytest.raises(tunnel.LaunchError, match="до выдачи адреса"):
        with tunnel.tunnel_process(Path("verified.exe"), 8010, tmp_path / "run"):
            pytest.fail("Failed tunnel must not yield a public address")
    assert process.stdout.closed
    assert "terminate-tunnel" not in process.events


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def read(self, limit):
        assert limit == 4097
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("payload", [
    {"mode": "real", "analyzer_ready": True}, {"mode": "mock", "analyzer_ready": False},
    {"mode": "mock", "analyzer_ready": "true"}, [],
])
def test_public_readiness_fails_closed_for_wrong_mode_or_unready_response(monkeypatch, payload):
    fast_clock(monkeypatch)
    calls = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(tunnel, "build_opener", lambda *handlers: SimpleNamespace(open=open_request))
    with pytest.raises(tunnel.LaunchError, match="готовности не прошла"):
        tunnel.wait_for_public(ORIGIN, "mock", FakeProcess(), InlineThread(), timeout=3)
    assert calls
    assert all(request.full_url == ORIGIN + "/health" and timeout == 4 for request, timeout in calls)
    assert all(not request.has_header("X-api-key") for request, _ in calls)


def test_readiness_uses_no_proxy_or_redirect_and_accepts_only_expected_mode(monkeypatch):
    handlers_used = []

    def opener(*handlers):
        handlers_used.extend(handlers)
        return SimpleNamespace(open=lambda *a, **k: FakeResponse(b'{"mode":"mock","analyzer_ready":true}'))

    monkeypatch.setattr(tunnel, "build_opener", opener)
    tunnel.wait_for_public(ORIGIN, "mock", FakeProcess(), InlineThread(), timeout=3)
    assert any(isinstance(handler, tunnel.ProxyHandler) and handler.proxies == {} for handler in handlers_used)
    redirect = next(handler for handler in handlers_used if isinstance(handler, tunnel.NoRedirect))
    assert redirect.redirect_request(None, None, 302, None, None, "https://attacker.example") is None


def test_readiness_stops_if_owned_tunnel_exits(monkeypatch):
    process = FakeProcess()
    process.returncode = 1
    # Building a safe opener is harmless; no request may be issued after poll detects exit.
    monkeypatch.setattr(tunnel, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: pytest.fail("No HTTP")))
    with pytest.raises(tunnel.LaunchError, match="завершился"):
        tunnel.wait_for_public(ORIGIN, "mock", process, InlineThread())


def test_listener_failure_closes_only_the_socket_it_created(monkeypatch):
    calls = []

    class BusySocket:
        def setsockopt(self, *args):
            calls.append("exclusive")

        def bind(self, address):
            calls.append(address)
            raise OSError("occupied test port")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(tunnel.socket, "socket", lambda *args: BusySocket())
    with pytest.raises(tunnel.LaunchError, match="чужие процессы не останавливались"):
        tunnel.reserve_listener(8010)
    assert ("127.0.0.1", 8010) in calls
    assert calls[-1] == "close"


@pytest.mark.parametrize("stuck", [False, True])
def test_application_server_limits_proxy_trust_and_stops_its_worker(monkeypatch, tmp_path, stuck):
    import uvicorn

    events = []
    application = SimpleNamespace(state=SimpleNamespace(service=SimpleNamespace(stop=lambda: events.append("stop-worker"))))
    monkeypatch.setattr("backend.main.create_app", lambda settings: application)
    configs = []

    def configuration(app, **kwargs):
        assert app is application
        configs.append(kwargs)
        return SimpleNamespace(**kwargs)

    servers = []

    class FakeServer:
        should_exit = False
        force_exit = False

        def __init__(self, config):
            servers.append(self)

        def run(self, sockets):
            events.append("start-server")
            assert len(sockets) == 1

    class ServerThread(InlineThread):
        alive = True

        def join(self, timeout):
            events.append(("join", timeout))
            if not stuck:
                self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(uvicorn, "Config", configuration)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(tunnel.threading, "Thread", ServerThread)
    listener = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8010))
    expected_error = tunnel.LaunchError if stuck else RuntimeError
    with pytest.raises(expected_error):
        with tunnel.application_server(Settings(mode="mock", data_dir=tmp_path), listener):
            raise RuntimeError("caller failed")
    assert configs[0]["host"] == "127.0.0.1"
    assert configs[0]["forwarded_allow_ips"] == "127.0.0.1"
    assert configs[0]["proxy_headers"] is True
    assert configs[0]["workers"] == 1
    assert configs[0]["limit_concurrency"] == 16
    assert servers[0].should_exit is True
    assert ("stop-worker" in events) is stuck
    assert servers[0].force_exit is stuck


def prepare_main(monkeypatch, tmp_path, mode="mock"):
    events = []
    process = FakeProcess(events=events)
    observed = {}
    monkeypatch.setattr(sys, "argv", ["public_tunnel", "--mode", mode])
    monkeypatch.setattr(tunnel, "ROOT", tmp_path)
    monkeypatch.setattr(tunnel, "verified_binary", lambda path: Path("verified.exe"))

    def checks(**kwargs):
        kwargs["data_dir"].mkdir(parents=True, exist_ok=True)
        return [doctor.Check("OK", "test", "synthetic")]

    monkeypatch.setattr(doctor, "run_checks", checks)
    monkeypatch.setattr("backend.make_sample.create_sample", lambda path: events.append("create-sample"))
    monkeypatch.setattr(tunnel, "reserve_listener", lambda port: SimpleNamespace(close=lambda: events.append("close-listener")))

    @contextmanager
    def fake_tunnel(*args):
        events.append("start-tunnel")
        try:
            yield ORIGIN, process
        finally:
            events.append("exit-tunnel-context")

    @contextmanager
    def fake_application(settings, listener):
        observed["settings"] = settings
        events.append("start-app")
        try:
            yield SimpleNamespace(), InlineThread()
        finally:
            events.append("stop-app")

    monkeypatch.setattr(tunnel, "tunnel_process", fake_tunnel)
    monkeypatch.setattr(tunnel, "application_server", fake_application)
    return events, observed


@pytest.mark.parametrize("mode", ["mock", "real"])
def test_main_failed_https_probe_stops_owned_tunnel_before_server(monkeypatch, tmp_path, capsys, mode):
    events, observed = prepare_main(monkeypatch, tmp_path, mode)

    def fail_probe(*args):
        raise tunnel.LaunchError("Synthetic HTTPS probe failed")

    monkeypatch.setattr(tunnel, "wait_for_public", fail_probe)
    assert tunnel.main() == 1
    assert events.index("terminate-tunnel") < events.index("stop-app")
    assert events[-1] == "close-listener"
    assert "HTTPS готов:" not in capsys.readouterr().out
    settings = observed["settings"]
    assert settings.mode == mode
    assert settings.data_dir == (tmp_path / "backend" / ".local" / ("public-" + mode)).resolve()
    assert settings.public_origin == ORIGIN
    assert settings.max_upload_mb == 90
    assert ("create-sample" in events) is (mode == "mock")


def test_main_keyboard_interrupt_stops_tunnel_and_server(monkeypatch, tmp_path, capsys):
    events, _ = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(tunnel, "wait_for_public", lambda *args: None)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(tunnel.time, "sleep", interrupt)
    assert tunnel.main() == 0
    assert events.index("terminate-tunnel") < events.index("stop-app")
    assert events[-1] == "close-listener"
    assert SECRET not in capsys.readouterr().out


def test_main_diagnostics_failure_never_starts_tunnel(monkeypatch, tmp_path, capsys):
    events, _ = prepare_main(monkeypatch, tmp_path, "real")
    monkeypatch.setattr(doctor, "run_checks", lambda **kwargs: [doctor.Check("FAIL", "analyzer", "missing")])
    assert tunnel.main() == 1
    assert not events
    assert "HTTPS готов:" not in capsys.readouterr().out


def test_direct_python_launch_fails_closed_for_invalid_environment(monkeypatch, tmp_path, capsys):
    events, _ = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setenv("CASH_AUDIT_JOB_TIMEOUT_SECONDS", SECRET)
    monkeypatch.setattr(doctor, "run_checks", lambda **kwargs: [doctor.check_environment(kwargs["mode"], kwargs["data_dir"])])
    assert tunnel.main() == 1
    assert not events
    output = capsys.readouterr().out
    assert SECRET not in output
    assert "HTTPS готов:" not in output
