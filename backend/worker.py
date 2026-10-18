"""One persistent, replaceable process owns the analyzer and its model state."""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import signal
import time
from contextlib import redirect_stderr, redirect_stdout

from .validation import validate_result

log = logging.getLogger(__name__)


class AnalyzerContractError(RuntimeError):
    """The published analyzer cannot honor the server's remaining budget."""


class AnalyzerReadinessError(RuntimeError):
    """The analyzer could not verify its local dependencies and model servers."""


def call_private(function, *args, **kwargs):
    # Third-party Python prints must not leak frames, paths, prompts or secrets
    # into the public launcher's console. Keep our own logs to error types.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            return function(*args, **kwargs)


def load_real_analyzer():
    module = call_private(importlib.import_module, "analyzer.service")
    analyzer = getattr(module, "analyze_video", None)
    check_ready = getattr(module, "check_ready", None)
    try:
        if not callable(analyzer) or not callable(check_ready):
            raise TypeError
        signature = inspect.signature(analyzer)
        parameter = signature.parameters.get("budget_sec")
        if parameter is None or parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise TypeError
        signature.bind("video.mov", budget_sec=20.0)
        inspect.signature(check_ready).bind()
    except (TypeError, ValueError):
        raise AnalyzerContractError from None
    incomplete_type = getattr(module, "IncompleteAnalysisError", None)
    if not isinstance(incomplete_type, type) or not issubclass(incomplete_type, Exception):
        incomplete_type = None
    return analyzer, check_ready, getattr(module, "describe_scope", None), incomplete_type


def verify_ready(check_ready):
    # Readiness may return identities or None; an explicit False is failure.
    try:
        if call_private(check_ready) is False:
            raise AnalyzerReadinessError
    except Exception:
        raise AnalyzerReadinessError from None


def worker_entry(connection, mode: str, mock_delay: float):
    # The parent owns shutdown; Windows broadcasts Ctrl+C to both processes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            if mode == "real":
                analyzer, check_ready, describe_scope, incomplete_type = load_real_analyzer()
                verify_ready(check_ready)
                analysis_scope = call_private(describe_scope) if callable(describe_scope) else None
                if analysis_scope is not None and not isinstance(analysis_scope, dict):
                    raise AnalyzerContractError
            elif mode == "mock":
                from .mock_analyzer import analyze
                analyzer = None
                analysis_scope = None
            else:
                raise ValueError("Unknown analyzer mode")
            connection.send({"kind": "ready", "analysis_scope": analysis_scope})
        except Exception as exc:
            log.error("Analyzer startup failed (%s). Check analyzer.service and local dependencies.", type(exc).__name__)
            code = "analyzer_contract" if isinstance(exc, AnalyzerContractError) else "analyzer_unavailable"
            connection.send({"kind": "unavailable", "error": code})
            return
        last_probe = time.monotonic()
        while True:
            try:
                if mode == "real" and not connection.poll(1):
                    if time.monotonic() - last_probe >= 5:
                        try:
                            verify_ready(check_ready)
                            last_probe = time.monotonic()
                        except Exception:
                            connection.send({"kind": "unavailable", "error": "analyzer_not_ready"})
                            break
                    continue
                task = connection.recv()
            except (EOFError, OSError):
                break
            if task is None:
                break
            job_id = task["job_id"]
            quarantine = False
            try:
                if mode == "real":
                    verify_ready(check_ready)
                    # Probe time also consumes the budget. Parent enforces the
                    # same monotonic deadline including uploads and queueing.
                    budget = min(260.0, task["deadline_monotonic"] - time.monotonic() - 5.0)
                    if budget <= 10:
                        connection.send({"kind": "result", "job_id": job_id, "error": "timeout"})
                        continue
                    result = call_private(analyzer, task["video_path"], budget_sec=budget)
                    last_probe = time.monotonic()
                else:
                    result = analyze(task["scenario"], mock_delay)
                result = validate_result(result)
                message = {"kind": "result", "job_id": job_id, "result": result}
            except Exception as exc:
                # Never log video content, evidence, arbitrary exception text or secrets.
                log.error("Analysis failed for job %s (%s)", job_id, type(exc).__name__)
                # Only the analyzer's typed completion signal establishes that
                # uncertainty remains after every local inference call finished.
                # Other failures can leave GPU work running in the model server.
                completed_uncertainty = (
                    mode == "real" and incomplete_type is not None
                    and isinstance(exc, incomplete_type)
                    and getattr(exc, "inference_completed", False) is True)
                error = ("incomplete_analysis" if completed_uncertainty else
                         "analyzer_not_ready" if isinstance(exc, AnalyzerReadinessError) else
                         "timeout" if isinstance(exc, TimeoutError) else "analysis_failed")
                message = {"kind": "result", "job_id": job_id, "error": error}
                quarantine = mode == "real" and not completed_uncertainty
                message["quarantine"] = quarantine
            try:
                connection.send(message)
            except (EOFError, OSError):
                break
            if quarantine:
                break
    finally:
        connection.close()
