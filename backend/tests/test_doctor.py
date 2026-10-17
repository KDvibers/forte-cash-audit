"""Offline diagnostic faults, safe output, and bounded localhost checks."""
from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler

import pytest

from backend import doctor


SECRET = "do-not-print-this-private-value"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "analyzer").mkdir()
    (tmp_path / "backend" / "requirements.txt").write_text(
        "\n".join(f"{name}==1.0.0" for name in doctor.MODULES), encoding="utf-8")
    (tmp_path / "analyzer" / "requirements.txt").write_text("av==16.0.1\nPillow==0.9.0", encoding="utf-8")
    return tmp_path


def rendered(checks):
    return "\n".join(f"{check.status} {check.name} {check.detail}" for check in checks)


def test_invalid_environment_is_reported_without_values(monkeypatch, tmp_path):
    monkeypatch.setenv("CASH_AUDIT_JOB_TIMEOUT_SECONDS", SECRET)
    monkeypatch.setenv("CASH_AUDIT_API_KEY", SECRET)
    result = doctor.check_environment("mock", tmp_path)
    assert result.status == "FAIL"
    assert SECRET not in rendered([result])


def test_dependency_exception_and_process_output_do_not_leak_secrets(monkeypatch, project):
    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(doctor, "run_hidden", fail)
    assert all(item.status == "FAIL" for item in doctor.check_dependencies(project))
    assert SECRET not in rendered(doctor.check_dependencies(project))
    monkeypatch.setattr(doctor, "run_hidden", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=SECRET, stderr=SECRET))
    assert SECRET not in rendered(doctor.check_dependencies(project))


def test_dependency_versions_imports_and_bundled_ffmpeg_are_checked(monkeypatch, project):
    package_dir = project / "installed-imageio"
    package_dir.mkdir()
    executable = package_dir / "ffmpeg.exe"
    executable.write_bytes(b"not executed in this unit test")
    payload = {name: {"ok": True, "version": "1.0.0"} for name in doctor.MODULES}
    payload["imageio-ffmpeg"].update(ffmpeg=str(executable), package_root=str(package_dir))
    calls = []

    def run(arguments, *, timeout, cwd):
        calls.append((arguments, timeout))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) if "-c" in arguments else "ffmpeg version test", stderr="")

    monkeypatch.setattr(doctor, "run_hidden", run)
    assert all(item.status == "OK" for item in doctor.check_dependencies(project))
    assert [timeout for _, timeout in calls] == [25, 5]
    payload["pillow"]["version"] = SECRET
    results = doctor.check_dependencies(project)
    assert next(item for item in results if item.name == "pillow").status == "FAIL"
    assert SECRET not in rendered(results)
    payload["fastapi"] = {"ok": False, "error": SECRET}
    assert next(item for item in doctor.check_dependencies(project) if item.name == "fastapi").status == "FAIL"


def test_ffmpeg_outside_package_is_not_executed(monkeypatch, project):
    package_dir = project / "imageio-package"
    package_dir.mkdir()
    executable = project / "outside.exe"
    executable.write_bytes(b"not a bundled executable")
    payload = {name: {"ok": True, "version": "1.0.0"} for name in doctor.MODULES}
    payload["imageio-ffmpeg"].update(ffmpeg=str(executable), package_root=str(package_dir))
    calls = []
    monkeypatch.setattr(doctor, "run_hidden", lambda *a, **k: (
        calls.append(a) or SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")))
    checks = doctor.check_dependencies(project)
    assert checks[-1].name == "FFmpeg" and checks[-1].status == "FAIL"
    assert len(calls) == 1


def test_occupied_port_fails_without_stopping_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert doctor.check_port(port).status == "FAIL"
        assert listener.getsockname()[1] == port
        assert doctor.check_port(port, skip=True).status == "WARN"
    assert doctor.check_port(0).status == "FAIL"


def test_write_probe_removes_only_its_own_temporary_file(tmp_path):
    existing = tmp_path / "preserve.txt"
    existing.write_text("keep", encoding="utf-8")
    assert doctor.check_storage(tmp_path).status == "OK"
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text(encoding="utf-8") == "keep"
    assert doctor.check_storage(existing).status == "FAIL"


def test_mock_analyzer_warnings_do_not_contact_model(monkeypatch, project):
    def unexpected_health():
        pytest.fail("Mock diagnostics must not contact the model")

    monkeypatch.setattr(doctor, "check_model_health", unexpected_health)
    assert [check.status for check in doctor.check_analyzer(project, "mock")] == ["WARN", "WARN"]


def test_real_missing_module_and_pin_conflict_fail_even_with_ready_health(monkeypatch, project):
    monkeypatch.setattr(doctor, "check_model_health", lambda: doctor.Check("OK", "health", "ready"))
    checks = doctor.check_analyzer(project, "real")
    assert [check.status for check in checks] == ["FAIL", "FAIL", "OK"]
    assert not (project / "analyzer" / "service.py").exists()


@pytest.mark.parametrize("status,payload,expected", [
    (200, {"status": "ok"}, "OK"), (200, {"status": "loading model"}, "FAIL"),
    (200, {"analyzer_ready": True}, "FAIL"), (503, {"status": "ok"}, "FAIL"),
    (200, [], "FAIL"),
])
def test_model_health_checks_body_and_uses_only_bounded_local_request(monkeypatch, status, payload, expected):
    handlers = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count):
            assert count == 4097
            return json.dumps(payload).encode()

    response = Response()
    response.status = status

    class Opener:
        def open(self, request, *, timeout):
            assert request.full_url == "http://127.0.0.1:8080/health"
            assert timeout == 3
            assert request.data is None
            assert "X-api-key" not in request.headers
            return response

    monkeypatch.setattr(doctor, "build_opener", lambda *args: (handlers.extend(args) or Opener()))
    check = doctor.check_model_health()
    assert check.status == expected
    assert any(isinstance(handler, ProxyHandler) and handler.proxies == {} for handler in handlers)
    redirect = next(handler for handler in handlers if isinstance(handler, HTTPRedirectHandler))
    assert redirect.redirect_request(None, None, 302, None, {}, "https://outside.invalid") is None


def test_model_http_failure_hides_error_details(monkeypatch):
    class Opener:
        def open(self, *args, **kwargs):
            raise HTTPError("http://127.0.0.1:8080/health", 503, SECRET, {}, None)

    monkeypatch.setattr(doctor, "build_opener", lambda *args: Opener())
    check = doctor.check_model_health()
    assert check.status == "FAIL"
    assert SECRET not in rendered([check])


def test_cli_exit_status_does_not_treat_warnings_as_success_of_real_mode(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks", lambda **kwargs: [doctor.Check("WARN", "Модуль", "нет")])
    assert doctor.main(["--mode", "mock", "--skip-port"]) == 0
    assert "предупреждений — 1" in capsys.readouterr().out
    monkeypatch.setattr(doctor, "run_checks", lambda **kwargs: [doctor.Check("FAIL", "Модуль", "нет")])
    assert doctor.main(["--mode", "real", "--skip-port"]) == 1
    assert "ошибок — 1" in capsys.readouterr().out
