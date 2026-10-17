"""Fault handling of the local self-check without recursive test processes."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from backend import selfcheck


SECRET = "private-selfcheck-test-value"


class InterruptedCheck(RuntimeError):
    pass


def audit_settings():
    return {key: value for key, value in os.environ.items() if key.startswith("CASH_AUDIT_")}


@pytest.mark.parametrize("interrupted", [False, True])
def test_settings_are_isolated_and_restored_even_after_exception(monkeypatch, interrupted):
    monkeypatch.setenv("CASH_AUDIT_MODE", "real")
    monkeypatch.setenv("CASH_AUDIT_API_KEY", SECRET)
    monkeypatch.setenv("CASH_AUDIT_DATA_DIR", "existing-production-data")
    monkeypatch.setenv("SELFCHECK_UNRELATED_SETTING", "preserve")
    original = audit_settings()

    try:
        with selfcheck.isolated_settings():
            assert audit_settings() == {"CASH_AUDIT_MODE": "mock"}
            assert os.environ["SELFCHECK_UNRELATED_SETTING"] == "preserve"
            os.environ["CASH_AUDIT_CREATED_DURING_CHECK"] = "temporary"
            os.environ["CASH_AUDIT_API_KEY"] = "temporary-key"
            if interrupted:
                raise InterruptedCheck()
    except InterruptedCheck:
        assert interrupted

    assert audit_settings() == original
    assert os.environ["SELFCHECK_UNRELATED_SETTING"] == "preserve"


def test_workspace_removes_only_its_run_after_exception(monkeypatch, tmp_path):
    run_root = tmp_path / "runs"
    neighbor = run_root / "run-existing-neighbor"
    neighbor.mkdir(parents=True)
    preserve = neighbor / "keep.txt"
    preserve.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(selfcheck, "RUN_ROOT", run_root)

    with pytest.raises(InterruptedCheck):
        with selfcheck.workspace() as owned:
            assert owned.parent == run_root.resolve()
            (owned / "nested").mkdir()
            (owned / "nested" / "temporary.txt").write_text("remove", encoding="utf-8")
            raise InterruptedCheck()

    assert not owned.exists()
    assert list(run_root.iterdir()) == [neighbor]
    assert preserve.read_text(encoding="utf-8") == "keep"


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """No pytest, Node, HTTP server, or model process can run from main()."""
    frontend = tmp_path / "frontend" / "tests"
    frontend.mkdir(parents=True)
    (frontend / "example.test.cjs").write_text("// Never executed by this unit test.\n", encoding="utf-8")
    monkeypatch.setattr(selfcheck, "ROOT", tmp_path)
    monkeypatch.setattr(selfcheck, "RUN_ROOT", tmp_path / "temporary-runs")
    monkeypatch.setattr(sys, "argv", ["backend.selfcheck"])
    state = SimpleNamespace(commands=[], http_calls=[], backend_ok=True, version="v22.15.0\n")

    def command(arguments, timeout):
        assert audit_settings() == {"CASH_AUDIT_MODE": "mock"}
        state.commands.append(arguments)
        return state.backend_ok if "pytest" in arguments else True

    def node_version(arguments, **kwargs):
        assert arguments == ["test-node", "--version"]
        return SimpleNamespace(returncode=0, stdout=state.version, stderr="")

    def smoke(data_dir):
        assert audit_settings() == {"CASH_AUDIT_MODE": "mock"}
        state.http_calls.append(data_dir)
        return {"events": 2, "seconds": 0.25}

    monkeypatch.setattr(selfcheck, "run_command", command)
    monkeypatch.setattr(selfcheck, "api_smoke", smoke)
    monkeypatch.setattr(selfcheck.shutil, "which", lambda name: "test-node")
    monkeypatch.setattr(selfcheck.subprocess, "run", node_version)
    return state


@pytest.mark.parametrize("node_problem", ["missing", "old"])
def test_missing_or_old_node_cannot_pass_full_check(cli, monkeypatch, capsys, node_problem):
    if node_problem == "missing":
        monkeypatch.setattr(selfcheck.shutil, "which", lambda name: None)
    else:
        cli.version = "v20.19.0\n"

    assert selfcheck.main() == 1
    output = capsys.readouterr().out
    assert "[FAIL] Frontend" in output
    assert "[OK] Backend" in output
    assert "[OK] HTTP API" in output
    assert not any("--test" in arguments for arguments in cli.commands)
    assert len(cli.http_calls) == 1


def test_backend_failure_is_not_masked_by_successful_frontend_and_api(cli, capsys):
    cli.backend_ok = False
    assert selfcheck.main() == 1
    output = capsys.readouterr().out
    assert "[FAIL] Backend" in output
    assert "[OK] Frontend" in output
    assert "[OK] HTTP API" in output
    assert any("--test" in arguments for arguments in cli.commands)
    assert len(cli.http_calls) == 1


def test_full_success_requires_all_four_checks(cli, capsys):
    assert selfcheck.main() == 0
    output = capsys.readouterr().out
    for phase in ("Окружение", "Backend", "Frontend", "HTTP API"):
        assert f"[OK] {phase}" in output
    assert len(cli.commands) == 3
    assert len(cli.http_calls) == 1


def test_api_exception_fails_safely_and_restores_environment(cli, monkeypatch, capsys):
    monkeypatch.setenv("CASH_AUDIT_MODE", "real")
    monkeypatch.setenv("CASH_AUDIT_API_KEY", SECRET)
    original = audit_settings()

    def fail(data_dir):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(selfcheck, "api_smoke", fail)
    assert selfcheck.main() == 1
    output = capsys.readouterr().out
    assert "[FAIL] HTTP API" in output
    assert SECRET not in output
    assert audit_settings() == original
    assert not list(selfcheck.RUN_ROOT.iterdir())


def test_command_timeout_fails_without_printing_process_output(monkeypatch, capsys):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(SECRET, 1, output=SECRET, stderr=SECRET)

    monkeypatch.setattr(selfcheck.subprocess, "run", timeout)
    assert selfcheck.run_command(["never-executed"], 1) is False
    output = capsys.readouterr().out
    assert "[FAIL]" in output
    assert SECRET not in output


@pytest.mark.parametrize("stuck,stop_raises", [(False, False), (True, False), (True, True)])
def test_mock_server_closes_owned_listener_on_body_or_shutdown_failure(
    monkeypatch, tmp_path, stuck, stop_raises
):
    """Fast fault injection: no sockets, background threads, or inference start."""
    state = SimpleNamespace(closed=False, stopped=False, joined=[], started=False)

    class Listener:
        def setsockopt(self, *args): pass
        def bind(self, address): assert address == ("127.0.0.1", 0)
        def listen(self, backlog): pass
        def getsockname(self): return ("127.0.0.1", 12345)
        def close(self): state.closed = True

    def stop():
        state.stopped = True
        if stop_raises:
            raise InterruptedCheck("owned service could not stop")

    listener = Listener()
    application = SimpleNamespace(state=SimpleNamespace(service=SimpleNamespace(stop=stop)))
    server = SimpleNamespace(should_exit=False, force_exit=False, run=lambda **kwargs: None)

    class Thread:
        def __init__(self, **kwargs):
            assert kwargs["kwargs"]["sockets"] == [listener]
        def start(self): state.started = True
        def join(self, timeout): state.joined.append(timeout)
        def is_alive(self): return stuck

    def create_app(settings):
        assert settings.mode == "mock"
        assert settings.data_dir == tmp_path
        return application

    monkeypatch.setattr(selfcheck, "socket", SimpleNamespace(
        AF_INET=2, SOCK_STREAM=1, SOL_SOCKET=1, SO_EXCLUSIVEADDRUSE=1,
        socket=lambda *args: listener))
    monkeypatch.setattr(selfcheck, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        Config=lambda *args, **kwargs: None, Server=lambda config: server))
    monkeypatch.setitem(sys.modules, "backend.main", SimpleNamespace(create_app=create_app))

    expected = selfcheck.CheckFailed if stuck and not stop_raises else InterruptedCheck
    with pytest.raises(expected):
        with selfcheck.mock_server(tmp_path) as (url, app):
            assert url == "http://127.0.0.1:12345" and app is application
            if not stuck:
                raise InterruptedCheck("body failed")

    assert state.started and server.should_exit
    assert state.closed, "The owned listening socket must close even if service.stop raises"
    assert state.stopped is stuck
    if stuck:
        assert server.force_exit
