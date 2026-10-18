"""Real-worker handshakes and budgets with an in-memory analyzer, never a GPU."""
from types import SimpleNamespace

import pytest

from backend import worker


SCOPE = {"rules": [{"id": "example", "title": "One review signal"}],
         "notice": "Limited scope", "complete_case_coverage": False}


class Connection:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def poll(self, timeout=0):
        return bool(self.tasks)

    def recv(self):
        return self.tasks.pop(0)

    def close(self):
        self.closed = True


def install_analyzer(monkeypatch, *, ready, analyze, scope=SCOPE):
    module = SimpleNamespace(check_ready=ready, analyze_video=analyze)
    if scope is not None:
        module.describe_scope = lambda: scope
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)


def test_real_worker_checks_models_before_ready_and_passes_remaining_budget(monkeypatch):
    order = []

    def analyze(path, *, budget_sec):
        order.append((path, budget_sec))
        return {"verdict": "no_violation", "violations": []}

    install_analyzer(monkeypatch, ready=lambda: order.append("models_checked"), analyze=analyze)
    connection = Connection([{"job_id": "one", "video_path": "synthetic.mov", "deadline_monotonic": 146.5}, None])
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100)
    worker.worker_entry(connection, "real", 0)
    assert order == ["models_checked", "models_checked", ("synthetic.mov", 41.5)]
    assert connection.messages[0] == {"kind": "ready", "analysis_scope": SCOPE}
    assert connection.messages[1]["result"] == {"verdict": "no_violation", "violations": []}
    assert connection.closed


def test_wrong_or_missing_model_never_yields_ready_or_calls_analysis(monkeypatch, caplog):
    def failed_models():
        raise RuntimeError("private-model-error-must-stay-hidden")

    def forbidden(path, *, budget_sec):
        pytest.fail("Unavailable models must not receive a video")

    install_analyzer(monkeypatch, ready=failed_models, analyze=forbidden)
    connection = Connection([])
    worker.worker_entry(connection, "real", 0)
    assert connection.messages == [{"kind": "unavailable", "error": "analyzer_unavailable"}]
    assert "private-model-error-must-stay-hidden" not in caplog.text
    assert connection.closed


def test_optional_scope_and_analysis_errors_preserve_failure_contract(monkeypatch):
    def failed_analysis(path, *, budget_sec):
        raise TimeoutError("deadline")

    install_analyzer(monkeypatch, ready=lambda: None, analyze=failed_analysis, scope=None)
    connection = Connection([{"job_id": "one", "video_path": "synthetic.mov", "deadline_monotonic": 116}, None])
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100)
    worker.worker_entry(connection, "real", 0)
    assert connection.messages == [
        {"kind": "ready", "analysis_scope": None},
        {"kind": "result", "job_id": "one", "error": "timeout", "quarantine": True},
    ]


def test_mock_worker_does_not_import_real_analyzer_or_inherit_scope(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mock readiness must not import a real analyzer or contact models")

    monkeypatch.setattr(worker.importlib, "import_module", forbidden)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    connection = Connection([None])
    worker.worker_entry(connection, "mock", 0)
    assert connection.messages == [{"kind": "ready", "analysis_scope": None}]
