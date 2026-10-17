"""One persistent, replaceable process owns the analyzer and its model state."""
from __future__ import annotations

import importlib
import logging
import signal

from .validation import validate_result

log = logging.getLogger(__name__)


def worker_entry(connection, mode: str, mock_delay: float):
    # The parent owns shutdown; Windows broadcasts Ctrl+C to both processes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            if mode == "real":
                module = importlib.import_module("analyzer.service")
                analyzer = getattr(module, "analyze_video")
                if not callable(analyzer):
                    raise TypeError("analyze_video is not callable")
            elif mode == "mock":
                from .mock_analyzer import analyze
                analyzer = None
            else:
                raise ValueError("Unknown analyzer mode")
            connection.send({"kind": "ready"})
        except Exception as exc:
            log.error("Analyzer startup failed (%s). Check analyzer.service and local dependencies.", type(exc).__name__)
            connection.send({"kind": "unavailable", "error": "analyzer_unavailable"})
            return
        while True:
            try:
                task = connection.recv()
            except (EOFError, OSError):
                break
            if task is None:
                break
            job_id = task["job_id"]
            try:
                result = analyzer(task["video_path"]) if mode == "real" else analyze(task["scenario"], mock_delay)
                result = validate_result(result)
                message = {"kind": "result", "job_id": job_id, "result": result}
            except Exception as exc:
                # Never log video content, evidence, arbitrary exception text or secrets.
                log.error("Analysis failed for job %s (%s)", job_id, type(exc).__name__)
                message = {"kind": "result", "job_id": job_id, "error": "analysis_failed"}
            try:
                connection.send(message)
            except (EOFError, OSError):
                break
    finally:
        connection.close()
