"""Real-mode orchestration with synthetic functions, never GPU/model inference."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest

from backend import main, service, worker
from backend.browser_sessions import BrowserSessions
from backend.config import Settings
from backend.main import ApiGuards, create_app
from backend.service import JobService


KEY = "synthetic-runtime-test-key-only"
CLEAN = {"verdict": "no_violation", "violations": []}


class Connection:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.messages = []
        self.closed = False

    def poll(self, timeout=0):
        return bool(self.tasks)

    def recv(self):
        return self.tasks.pop(0)

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


def install_analyzer(monkeypatch, analyze, ready=lambda: None, *, incomplete_type=None):
    module = SimpleNamespace(analyze_video=analyze, check_ready=ready)
    if incomplete_type is not None:
        module.IncompleteAnalysisError = incomplete_type
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)


@pytest.mark.parametrize("analyze", [
    lambda path: CLEAN,
    lambda path, **kwargs: CLEAN,
    lambda path, budget_sec, /: CLEAN,
    lambda path, budget_sec, required: CLEAN,
    None,
])
def test_legacy_or_implicit_budget_interface_is_not_ready(monkeypatch, analyze):
    install_analyzer(monkeypatch, analyze)
    connection = Connection([None])
    worker.worker_entry(connection, "real", 0)
    assert connection.messages == [{"kind": "unavailable", "error": "analyzer_contract"}]
    assert connection.closed


@pytest.mark.parametrize("ready", [None, lambda required: None, lambda: False])
def test_missing_or_failed_readiness_never_starts_analysis(monkeypatch, ready):
    calls = []

    def analyze(path, *, budget_sec):
        calls.append(path)
        return CLEAN

    install_analyzer(monkeypatch, analyze, ready)
    connection = Connection([{"job_id": "a", "video_path": "a.mov", "deadline_monotonic": 999999}, None])
    worker.worker_entry(connection, "real", 0)
    assert connection.messages[0]["kind"] == "unavailable"
    assert not calls


def test_two_real_shaped_calls_are_sequential_and_receive_remaining_budget(monkeypatch, capsys):
    clock = [100.0]
    calls, probes = [], []

    def ready():
        print("private readiness diagnostics")
        probes.append(clock[0])
        clock[0] += 2

    def analyze(path, *, budget_sec):
        print("private synthetic prompt")
        calls.append((path, budget_sec))
        clock[0] += 20
        return CLEAN.copy()

    install_analyzer(monkeypatch, analyze, ready)
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    connection = Connection([
        {"job_id": "first", "video_path": "first.mov", "deadline_monotonic": 200},
        {"job_id": "second", "video_path": "second.mov", "deadline_monotonic": 200}, None,
    ])
    worker.worker_entry(connection, "real", 0)
    assert calls == [("first.mov", 91.0), ("second.mov", 69.0)]
    assert len(probes) == 3  # startup, immediately before each analysis
    assert [event["job_id"] for event in connection.messages[1:]] == ["first", "second"]
    assert all(event["result"] == CLEAN for event in connection.messages[1:])
    assert capsys.readouterr().out == ""


def test_analyzer_budget_has_local_260_second_ceiling(monkeypatch):
    budgets = []

    def analyze(path, budget_sec):
        budgets.append(budget_sec)
        return CLEAN

    install_analyzer(monkeypatch, analyze)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100)
    connection = Connection([{"job_id": "a", "video_path": "a.mov", "deadline_monotonic": 385}, None])
    worker.worker_entry(connection, "real", 0)
    assert budgets == [260]


def test_expired_queue_budget_never_calls_model(monkeypatch):
    calls = []

    def analyze(path, budget_sec):
        calls.append(path)
        return CLEAN

    install_analyzer(monkeypatch, analyze)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100)
    connection = Connection([{"job_id": "a", "video_path": "a.mov", "deadline_monotonic": 115}, None])
    worker.worker_entry(connection, "real", 0)
    assert connection.messages[-1] == {"kind": "result", "job_id": "a", "error": "timeout"}
    assert not calls


@pytest.mark.parametrize("error", [TimeoutError("private URL"), RuntimeError("private prompt")])
def test_real_error_quarantines_without_dispatching_second_job(monkeypatch, error, caplog):
    calls = []

    def analyze(path, budget_sec):
        calls.append(path)
        raise error

    install_analyzer(monkeypatch, analyze)
    deadline = time.monotonic() + 100
    connection = Connection([
        {"job_id": "a", "video_path": "a.mov", "deadline_monotonic": deadline},
        {"job_id": "b", "video_path": "b.mov", "deadline_monotonic": deadline}, None,
    ])
    worker.worker_entry(connection, "real", 0)
    assert calls == ["a.mov"]
    assert connection.messages[-1]["quarantine"] is True
    assert connection.messages[-1]["error"] == ("timeout" if isinstance(error, TimeoutError) else "analysis_failed")
    assert "private" not in caplog.text


@pytest.mark.parametrize("failure", [
    "known_complete", "known_incomplete", "known_missing", "unexpected_complete",
    "timeout_complete", "unregistered_complete",
])
def test_only_typed_completed_uncertainty_can_continue_real_queue(monkeypatch, failure, caplog):
    class IncompleteAnalysisError(RuntimeError):
        pass

    error = (TimeoutError("private timeout") if failure == "timeout_complete" else
             RuntimeError("private arbitrary error") if failure == "unexpected_complete" else
             IncompleteAnalysisError("private unresolved visual evidence"))
    if failure != "known_missing":
        error.inference_completed = failure != "known_incomplete"
    calls = []

    def analyze(path, *, budget_sec):
        calls.append(path)
        if len(calls) == 1:
            raise error
        return CLEAN.copy()

    install_analyzer(monkeypatch, analyze, incomplete_type=(
        None if failure == "unregistered_complete" else IncompleteAnalysisError))
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100)
    connection = Connection([
        {"job_id": "a", "video_path": "a.mov", "deadline_monotonic": 200},
        {"job_id": "b", "video_path": "b.mov", "deadline_monotonic": 200}, None,
    ])
    worker.worker_entry(connection, "real", 0)
    recovered = failure == "known_complete"
    assert calls == (["a.mov", "b.mov"] if recovered else ["a.mov"])
    assert connection.messages[1] == {
        "kind": "result", "job_id": "a", "error": (
            "incomplete_analysis" if recovered else
            "timeout" if failure == "timeout_complete" else "analysis_failed"),
        "quarantine": not recovered,
    }
    if recovered:
        assert connection.messages[2] == {"kind": "result", "job_id": "b", "result": CLEAN}
    assert "private" not in caplog.text


def test_models_stopped_after_startup_fail_before_model_call(monkeypatch):
    probes, calls = [], []

    def ready():
        probes.append(True)
        if len(probes) > 1:
            raise ConnectionError("private server failure")

    def analyze(path, budget_sec):
        calls.append(path)
        return CLEAN

    install_analyzer(monkeypatch, analyze, ready)
    connection = Connection([{"job_id": "a", "video_path": "a.mov", "deadline_monotonic": time.monotonic() + 100}])
    worker.worker_entry(connection, "real", 0)
    assert connection.messages[-1] == {
        "kind": "result", "job_id": "a", "error": "analyzer_not_ready", "quarantine": True,
    }
    assert not calls


def test_idle_model_shutdown_is_reported_unavailable(monkeypatch):
    clock, probes = [100.0], []

    def ready():
        probes.append(True)
        if len(probes) > 1:
            raise ConnectionError

    install_analyzer(monkeypatch, lambda path, budget_sec: CLEAN, ready)
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    connection = Connection([])

    def poll(timeout=0):
        clock[0] += 6
        return False

    connection.poll = poll
    worker.worker_entry(connection, "real", 0)
    assert connection.messages == [{"kind": "ready", "analysis_scope": None}, {"kind": "unavailable", "error": "analyzer_not_ready"}]


def make_service(tmp_path, **overrides):
    return JobService(Settings(data_dir=tmp_path, api_key=KEY, **overrides))


def job_metadata(manager, job_id, owner="api"):
    return {"job_id": job_id, "owner": owner, "file_name": "synthetic.mov", "file_size": 1,
            "video_path": str(manager.upload_dir / job_id / "source.mov"), "created_at": time.time(),
            "mode": manager.settings.mode}


class Steps:
    def __init__(self, count):
        self.count = count

    def wait(self, timeout):
        self.count -= 1
        return self.count < 0

    def is_set(self):
        return self.count < 0


def run_steps(manager, monkeypatch, count=1):
    manager._stop = Steps(count)
    monkeypatch.setattr(manager, "_terminate_worker", lambda: setattr(manager, "ready", False))
    manager._run()


def test_active_real_timeout_fails_queue_and_never_restarts_worker(tmp_path, monkeypatch):
    manager = make_service(tmp_path)
    manager.ready = True
    for name, remaining in (("active", -1), ("queued", 100)):
        manager.register_job(job_metadata(manager, name), time.monotonic() + remaining - 285)
    manager._active = manager.store.get("active")
    manager.store.start("active")
    launches = []
    monkeypatch.setattr(manager, "_launch_worker", lambda: launches.append(True))
    run_steps(manager, monkeypatch)
    assert manager.store.get("active")["error_code"] == "timeout"
    assert manager.store.get("queued")["error_code"] == "real_restart_required"
    assert not manager.ready and not launches
    assert not manager._deadlines


def test_queued_deadline_is_spent_while_waiting_without_model_call(tmp_path, monkeypatch):
    manager = make_service(tmp_path)
    manager.ready = True
    manager._connection = Connection([])
    manager.register_job(job_metadata(manager, "queued"), time.monotonic() - 286)
    run_steps(manager, monkeypatch)
    assert manager.store.get("queued")["error_code"] == "timeout"
    assert manager.store.get("queued")["started_at"] is None
    assert manager._connection.messages == []


@pytest.mark.parametrize("expired", [False, True])
def test_completed_uncertainty_keeps_next_job_but_honors_active_deadline(tmp_path, monkeypatch, expired):
    manager = make_service(tmp_path)
    manager.ready = True
    manager._connection = Connection([])
    monkeypatch.setattr(service.time, "monotonic", lambda: 100.0)
    for name, deadline in (("active", 99.0 if expired else 200.0), ("queued", 250.0)):
        manager.register_job(job_metadata(manager, name), deadline - 285.0)
    manager._active = manager.store.get("active")
    manager.store.start("active")
    manager._handle_event({"kind": "result", "job_id": "active", "error": "incomplete_analysis",
                           "quarantine": False})
    assert manager.store.get("active")["error_code"] == ("timeout" if expired else "incomplete_analysis")
    assert manager.ready and manager._active is None
    assert manager.store.get("queued")["status"] == "processing"
    manager._dispatch(manager.store.get("queued"))
    assert len(manager._connection.messages) == 1
    assert manager._connection.messages[0]["job_id"] == "queued"
    assert manager._connection.messages[0]["deadline_monotonic"] == 250.0


def test_coordinator_dispatches_only_one_at_a_time(tmp_path, monkeypatch):
    manager = make_service(tmp_path)
    manager.ready = True
    sent, responses = [], []
    for name in ("first", "second"):
        manager.register_job(job_metadata(manager, name), time.monotonic() - 30)

    def send(task):
        sent.append(task)
        assert manager._active["job_id"] == task["job_id"]
        if len(sent) == 2:
            assert manager.store.get("first")["status"] == "done"
        responses.append({"kind": "result", "job_id": task["job_id"], "result": CLEAN})

    manager._connection = SimpleNamespace(poll=lambda: bool(responses), recv=lambda: responses.pop(0), send=send)
    run_steps(manager, monkeypatch, 3)
    assert [item["job_id"] for item in sent] == ["first", "second"]
    assert all(250 < item["deadline_monotonic"] - time.monotonic() < 255 for item in sent)
    assert all(manager.store.get(name)["status"] == "done" for name in ("first", "second"))


def test_same_checkout_real_lock_blocks_other_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "REAL_LOCK_PATH", tmp_path / "shared" / "real.lock")
    monkeypatch.setattr(JobService, "_launch_worker", lambda self: None)
    monkeypatch.setattr(JobService, "_run", lambda self: None)
    first = make_service(tmp_path / "demo")
    second = make_service(tmp_path / "funnel")
    first.start()
    try:
        with pytest.raises(RuntimeError, match="real-analyzer lock"):
            second.start()
    finally:
        first.stop()
    second.start()
    second.stop()


def test_real_budget_ceiling_does_not_restrict_mock(tmp_path):
    with pytest.raises(ValueError, match="285"):
        Settings(mode="real", data_dir=tmp_path, job_timeout_seconds=286)
    assert Settings(mode="mock", data_dir=tmp_path, job_timeout_seconds=400).job_timeout_seconds == 400


def upload_scope(manager):
    return {"type": "http", "method": "POST", "path": "/api/v1/analyze", "scheme": "http",
            "client": ("127.0.0.1", 1234), "headers": [(b"x-api-key", KEY.encode())],
            "app": SimpleNamespace(state=SimpleNamespace(service=manager))}


def test_two_uploads_reserve_slots_before_body_and_third_is_429(tmp_path):
    manager = make_service(tmp_path)
    manager.ready = True

    async def scenario():
        release = asyncio.Event()
        entered = []

        async def inner(scope, receive, send):
            entered.append(scope)
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})

        async def receive():
            await release.wait()
            return {"type": "http.request", "body": b"data", "more_body": False}

        guards = ApiGuards(inner, manager.settings)
        outputs = [[], [], []]

        async def request(index):
            async def send(message):
                outputs[index].append(message)
            await guards(upload_scope(manager), receive, send)

        first, second = asyncio.create_task(request(0)), asyncio.create_task(request(1))
        for _ in range(100):
            if len(entered) == 2:
                break
            await asyncio.sleep(0)
        assert len(entered) == 2 and len(manager._reservations) == 2
        await request(2)
        assert outputs[2][0]["status"] == 429
        assert len(entered) == 2
        release.set()
        await asyncio.gather(first, second)
        assert not manager._reservations

    asyncio.run(scenario())


def test_stalled_upload_expires_and_releases_slot(tmp_path):
    manager = make_service(tmp_path, job_timeout_seconds=0.03)
    manager.ready = True

    async def scenario():
        output = []

        async def receive():
            await asyncio.Event().wait()

        async def inner(scope, receive, send):
            await receive()

        async def send(message):
            output.append(message)

        await ApiGuards(inner, manager.settings)(upload_scope(manager), receive, send)
        assert output[0]["status"] == 408
        assert not manager._reservations

    asyncio.run(scenario())


def test_multipart_upload_time_is_part_of_registered_deadline(tmp_path, monkeypatch):
    manager = make_service(tmp_path)
    manager.ready = True
    app = create_app(manager.settings)
    app.state.service = manager
    monkeypatch.setattr(main, "validate_video", lambda *args, **kwargs: None)

    async def scenario():
        received_at = []

        async def body():
            await asyncio.sleep(0.04)
            received_at.append(time.time())
            yield (b'--test\r\nContent-Disposition: form-data; name="file"; filename="test.mov"\r\n'
                   b'Content-Type: video/quicktime\r\n\r\nsynthetic\r\n--test--\r\n')

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            response = await client.post("/api/v1/analyze", content=body(), headers={
                "X-API-Key": KEY, "Content-Type": "multipart/form-data; boundary=test"})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        assert received_at[0] - manager.store.get(job_id)["created_at"] >= 0.03
        assert manager._deadlines[job_id] - time.monotonic() < 284.98
        assert not manager._reservations

    asyncio.run(scenario())


def test_reservation_and_created_job_are_one_admission(tmp_path):
    manager = make_service(tmp_path)
    manager.ready = True
    first = manager.reserve("api", None, "local")
    manager.register_job(job_metadata(manager, first), time.monotonic())
    second = manager.reserve("api", None, "local")
    assert second != first
    with pytest.raises(HTTPException) as raised:
        manager.reserve("api", None, "local")
    assert raised.value.status_code == 429


def test_ui_failure_code_is_allowlisted_and_jury_contract_unchanged(tmp_path):
    manager = make_service(tmp_path)
    app = create_app(manager.settings)
    app.state.service = manager
    app.state.browser_sessions = BrowserSessions(manager.store)
    client = TestClient(app, raise_server_exceptions=True)
    try:
        client.get("/ui/session")
        token = client.cookies.get("cash_audit_session")
        owner = BrowserSessions.owner(app.state.browser_sessions.get(token))
        for name, failure in (("known", "real_restart_required"), ("incomplete", "incomplete_analysis"),
                              ("unknown", "private-token-123")):
            manager.store.create(job_metadata(manager, name, owner))
            manager.store.finish(name, error=failure)
            result = client.get(f"/ui/jobs/{name}").json()
            assert result == {"job_id": name, "status": "failed",
                              "error_code": failure if name != "unknown" else "analysis_failed"}
        manager.store.create(job_metadata(manager, "public"))
        manager.store.finish("public", error="real_restart_required")
        public = client.get("/api/v1/jobs/public", headers={"X-API-Key": KEY})
        assert public.headers["X-Cash-Audit-Job-Mode"] == "real"
        assert public.json() == {
            "job_id": "public", "status": "failed"}
    finally:
        client.close()


def test_unavailable_event_clears_real_health_without_mock_fallback(tmp_path, monkeypatch):
    manager = make_service(tmp_path)
    manager.ready = True
    monkeypatch.setattr(manager, "_terminate_worker", lambda: setattr(manager, "ready", False))
    manager._handle_event({"kind": "unavailable", "error": "analyzer_contract"})
    app = create_app(manager.settings)
    app.state.service = manager
    client = TestClient(app)
    response = client.get("/health")
    assert response.json()["analyzer_ready"] is False
    assert response.json()["mode"] == "real"
    assert "budget_sec" in response.json()["message"]
