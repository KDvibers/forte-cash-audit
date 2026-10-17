"""Funnel ownership, readiness and failure paths without account/network changes."""
from __future__ import annotations

from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from backend import doctor, public_funnel as funnel


HOST = "synthetic-device.synthetic-tailnet.ts.net"
ORIGIN = "https://" + HOST
SECRET = "sensitive-test-value-do-not-print"


def status():
    return {"BackendState": "Running", "Self": {"DNSName": HOST + "."}, "CertDomains": [HOST]}


def config(port=8020):
    return {"Foreground": {"synthetic-session": {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {HOST + ":443": {"Handlers": {"/": {"Proxy": f"http://127.0.0.1:{port}"}}}},
        "AllowFunnel": {HOST + ":443": True},
    }}}


class FakeProcess:
    def __init__(self, events=None, stubborn=False):
        self.events = events if events is not None else []
        self.returncode = None
        self.stubborn = stubborn

    def poll(self):
        return self.returncode

    def terminate(self):
        self.events.append("terminate-funnel")
        if not self.stubborn:
            self.returncode = 0

    def wait(self, timeout):
        self.events.append(("wait-funnel", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("synthetic-funnel", timeout)
        return self.returncode

    def kill(self):
        self.events.append("kill-funnel")
        self.returncode = -9


def fast_clock(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(funnel.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(funnel.time, "sleep", lambda _: None)


def test_origin_uses_exact_device_name_and_certificate(monkeypatch):
    monkeypatch.delenv("CASH_AUDIT_PUBLIC_ORIGIN", raising=False)
    assert funnel.device_origin(status()) == ORIGIN


@pytest.mark.parametrize("host", [
    HOST + ".attacker.example", HOST + "@attacker.example", HOST + "/path", HOST + ":8443",
    HOST + "?secret=private", HOST + "#section", "localhost", "127.0.0.1", "https://" + HOST,
    "other-device.synthetic-tailnet.ts.net", "-bad.synthetic-tailnet.ts.net",
])
def test_origin_rejects_other_devices_injected_urls_and_non_tailscale_hosts(host):
    payload = status()
    payload["Self"]["DNSName"] = host
    with pytest.raises(funnel.LaunchError):
        funnel.device_origin(payload)


@pytest.mark.parametrize("payload", [{}, {"BackendState": "NeedsLogin"},
    {"BackendState": "Running", "Self": None, "CertDomains": []},
    {"BackendState": "Running", "Self": {"DNSName": HOST}, "CertDomains": HOST},
])
def test_missing_auth_or_certificate_fails_closed(payload):
    with pytest.raises(funnel.LaunchError):
        funnel.device_origin(payload)


def test_old_public_origin_cannot_redirect_funnel_to_other_host(monkeypatch):
    monkeypatch.setenv("CASH_AUDIT_PUBLIC_ORIGIN", "https://old.trycloudflare.com")
    with pytest.raises(funnel.LaunchError, match="отличается"):
        funnel.device_origin(status())


def test_cli_environment_omits_tokens_and_alternate_account_controls(monkeypatch):
    for key in ("TS_AUTHKEY", "TS_SOCKET", "TAILSCALE_CONTROLURL", "CASH_AUDIT_API_KEY", "TUNNEL_TOKEN"):
        monkeypatch.setenv(key, SECRET)
    monkeypatch.setenv("FUNNEL_TEST_PRESERVE", "retained")
    environment = funnel.client_environment()
    assert SECRET not in environment.values()
    assert environment["FUNNEL_TEST_PRESERVE"] == "retained"


def test_read_client_json_hides_account_output_and_is_noninteractive(monkeypatch):
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return SimpleNamespace(returncode=1, stdout=SECRET, stderr=SECRET)

    monkeypatch.setattr(funnel.subprocess, "run", run)
    with pytest.raises(funnel.LaunchError) as error:
        funnel.read_client_json(Path("client.exe"), "status", "--json")
    assert SECRET not in str(error.value)
    assert calls[0][0] == ["client.exe", "status", "--json"]
    assert calls[0][1]["stdin"] == subprocess.DEVNULL


def test_concurrent_configuration_changes_are_not_overwritten(monkeypatch):
    results = iter([{}, config()])
    monkeypatch.setattr(funnel, "read_client_json", lambda *args: next(results))
    with pytest.raises(funnel.LaunchError, match="изменилось"):
        funnel.read_publications(Path("client.exe"))


def test_existing_serve_or_funnel_prevents_start(monkeypatch, tmp_path):
    monkeypatch.setattr(funnel, "read_publications", lambda *args: config())
    monkeypatch.setattr(funnel.subprocess, "Popen", lambda *a, **k: pytest.fail("Existing config must not be changed"))
    with pytest.raises(funnel.LaunchError, match="уже настроен"):
        with funnel.funnel_process(Path("client.exe"), ORIGIN, 8020, tmp_path):
            pytest.fail("Must not launch")


def test_exact_foreground_config_is_owned_but_extra_routes_or_background_are_not():
    assert funnel.owns_publication(config(), ORIGIN, 8020)
    assert not funnel.owns_publication(config(8021), ORIGIN, 8020)
    assert not funnel.owns_publication(config()["Foreground"]["synthetic-session"], ORIGIN, 8020)
    extra = config()
    extra["Foreground"]["synthetic-session"]["Web"][HOST + ":443"]["Handlers"]["/other"] = {"Text": "other service"}
    assert not funnel.owns_publication(extra, ORIGIN, 8020)
    extra = config()
    extra["TCP"] = {"8443": {"HTTPS": True}}
    assert not funnel.owns_publication(extra, ORIGIN, 8020)


def test_foreground_launch_stops_only_owned_process_and_checks_removal(monkeypatch, tmp_path):
    events = []
    process = FakeProcess(events, stubborn=True)
    calls = []
    observed = iter([{}, config(), {}])
    monkeypatch.setattr(funnel, "read_publications", lambda *args: next(observed))

    def spawn(arguments, **options):
        calls.append((arguments, options))
        return process

    monkeypatch.setattr(funnel.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="caller failed"):
        with funnel.funnel_process(Path("client.exe"), ORIGIN, 8020, tmp_path) as owned:
            assert owned is process
            raise RuntimeError("caller failed")
    assert calls[0][0] == ["client.exe", "funnel", "--https=443", "http://127.0.0.1:8020"]
    assert "--bg" not in calls[0][0]
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert events == ["terminate-funnel", ("wait-funnel", 10), "kill-funnel", ("wait-funnel", 5)]


def test_failed_foreground_process_never_yields_ready(monkeypatch, tmp_path):
    process = FakeProcess()
    process.returncode = 1
    monkeypatch.setattr(funnel, "read_publications", lambda *args: {})
    monkeypatch.setattr(funnel.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(funnel.LaunchError, match="завершился"):
        with funnel.funnel_process(Path("client.exe"), ORIGIN, 8020, tmp_path):
            pytest.fail("Failed process cannot be ready")
    assert not process.events


def test_cleanup_reports_remaining_configuration_without_global_reset(monkeypatch):
    fast_clock(monkeypatch)
    monkeypatch.setattr(funnel, "read_publications", lambda *args: config(9999))
    monkeypatch.setattr(funnel.subprocess, "run", lambda *a, **k: pytest.fail("Never reset other configuration"))
    with pytest.raises(funnel.LaunchError, match="общий reset не выполнялся"):
        funnel.verify_publication_removed(Path("client.exe"), timeout=3)


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("payload", [
    {"status": "ok", "mode": "real", "analyzer_ready": True},
    {"status": "degraded", "mode": "mock", "analyzer_ready": True},
    {"status": "ok", "mode": "mock", "analyzer_ready": False},
    {"status": "ok", "mode": "mock", "analyzer_ready": "true"}, [],
])
def test_readiness_rejects_wrong_mode_status_and_unready_model(monkeypatch, payload):
    fast_clock(monkeypatch)
    monkeypatch.setattr(funnel, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: FakeResponse(payload)))
    with pytest.raises(funnel.LaunchError, match="готовности"):
        funnel.wait_for_health(ORIGIN, "mock", SimpleNamespace(is_alive=lambda: True), timeout=3)


def test_local_probe_has_exact_host_and_scheme_and_public_probe_has_no_spoofed_headers(monkeypatch):
    requests = []
    handlers = []

    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 4
        return FakeResponse({"status": "ok", "mode": "mock", "analyzer_ready": True})

    def opener(*values):
        handlers.extend(values)
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(funnel, "build_opener", opener)
    thread = SimpleNamespace(is_alive=lambda: True)
    funnel.wait_for_health("http://127.0.0.1:8020", "mock", thread, origin=ORIGIN)
    funnel.wait_for_health(ORIGIN, "mock", thread, process=FakeProcess())
    assert requests[0].get_header("Host") == HOST
    assert requests[0].get_header("X-forwarded-proto") == "https"
    assert not requests[1].has_header("Host")
    assert not requests[1].has_header("X-forwarded-proto")
    assert all(not request.has_header("X-api-key") for request in requests)
    assert all(handler.proxies == {} for handler in handlers if isinstance(handler, funnel.ProxyHandler))
    assert any(isinstance(handler, funnel.NoRedirect) for handler in handlers)


def prepare_main(monkeypatch, tmp_path):
    monkeypatch.delenv("CASH_AUDIT_PUBLIC_ORIGIN", raising=False)
    monkeypatch.setattr(funnel, "ROOT", tmp_path)
    monkeypatch.setattr(funnel, "find_binary", lambda path: Path("client.exe"))
    monkeypatch.setattr(funnel, "read_client_json", lambda *args: status())
    monkeypatch.setattr(funnel, "read_publications", lambda *args: {})
    events = []
    observed = {}

    def checks(**options):
        options["data_dir"].mkdir(parents=True, exist_ok=True)
        return [doctor.Check("OK", "synthetic", "ready")]

    monkeypatch.setattr(doctor, "run_checks", checks)
    monkeypatch.setattr("backend.make_sample.create_sample", lambda path: events.append("create-sample"))
    monkeypatch.setattr(funnel, "reserve_listener", lambda port: SimpleNamespace(close=lambda: events.append("close-listener")))

    @contextmanager
    def application(settings, listener):
        observed["settings"] = settings
        events.append("start-app")
        try:
            yield SimpleNamespace(), SimpleNamespace(is_alive=lambda: True)
        finally:
            events.append("stop-app")

    @contextmanager
    def publication(*args):
        events.append("start-funnel")
        try:
            yield FakeProcess()
        finally:
            events.append("stop-funnel")

    monkeypatch.setattr(funnel, "application_server", application)
    monkeypatch.setattr(funnel, "funnel_process", publication)
    return events, observed


@pytest.mark.parametrize("mode", ["mock", "real"])
def test_https_failure_stops_publication_before_backend_and_closes_listener(monkeypatch, tmp_path, mode, capsys):
    events, observed = prepare_main(monkeypatch, tmp_path)

    def probe(url, *args, **kwargs):
        events.append("probe-local" if url.startswith("http:") else "probe-public")
        if url.startswith("https:"):
            raise funnel.LaunchError("HTTPS test failed")

    monkeypatch.setattr(funnel, "wait_for_health", probe)
    assert funnel.main(["--mode", mode]) == 1
    assert events.index("probe-local") < events.index("start-funnel")
    assert events.index("stop-funnel") < events.index("stop-app")
    assert events[-1] == "close-listener"
    settings = observed["settings"]
    assert settings.mode == mode
    assert settings.data_dir == tmp_path / "backend" / ".local" / ("funnel-" + mode)
    assert settings.public_origin == ORIGIN
    assert settings.max_upload_mb == 100
    assert ("create-sample" in events) is (mode == "mock")
    assert "HTTPS отвечает" not in capsys.readouterr().out


def test_real_preflight_failure_never_starts_backend_or_publication(monkeypatch, tmp_path):
    events, _ = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(doctor, "run_checks", lambda **kw: [doctor.Check("FAIL", "analyzer", "missing")])
    assert funnel.main(["--mode", "real"]) == 1
    assert not events


def test_backend_startup_failure_never_opens_public_access(monkeypatch, tmp_path):
    events, _ = prepare_main(monkeypatch, tmp_path)

    def fail_local_probe(*args, **kwargs):
        raise funnel.LaunchError("Local worker failed")

    monkeypatch.setattr(funnel, "wait_for_health", fail_local_probe)
    assert funnel.main([]) == 1
    assert "start-funnel" not in events
    assert events[-2:] == ["stop-app", "close-listener"]


def test_cleanup_failure_is_reported_instead_of_success_on_interrupt(monkeypatch, tmp_path, capsys):
    events, _ = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(funnel, "wait_for_health", lambda *a, **k: None)

    @contextmanager
    def changed_configuration(*args):
        try:
            yield FakeProcess()
        finally:
            raise funnel.LaunchError("Synthetic remaining publication; inspect status")

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(funnel, "funnel_process", changed_configuration)
    monkeypatch.setattr(funnel.time, "sleep", interrupt)
    assert funnel.main([]) == 1
    output = capsys.readouterr().out
    assert "Synthetic remaining publication" in output
    assert "Собственная публикация Funnel снята" not in output
    assert events[-2:] == ["stop-app", "close-listener"]


def test_keyboard_interrupt_cleans_owned_resources_and_never_prints_api_key(monkeypatch, tmp_path, capsys):
    events, observed = prepare_main(monkeypatch, tmp_path)
    monkeypatch.setenv("CASH_AUDIT_API_KEY", SECRET)
    monkeypatch.setenv("CASH_AUDIT_DATA_DIR", str(tmp_path / "other-running-server"))
    monkeypatch.setenv("CASH_AUDIT_MODE", "real")
    monkeypatch.setattr(funnel, "wait_for_health", lambda *a, **k: None)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(funnel.time, "sleep", interrupt)
    assert funnel.main([]) == 0
    assert events.index("stop-funnel") < events.index("stop-app")
    assert observed["settings"].api_key is None
    assert observed["settings"].mode == "mock"
    assert observed["settings"].data_dir == tmp_path / "backend" / ".local" / "funnel-mock"
    assert SECRET not in capsys.readouterr().out
