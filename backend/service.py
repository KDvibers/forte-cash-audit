"""Single-GPU job coordinator, separated from HTTP request handling."""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import secrets
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import HTTPException

from .config import Settings
from .store import JobStore
from .worker import worker_entry

log = logging.getLogger(__name__)
REAL_LOCK_PATH = Path(__file__).resolve().parent / ".local" / "real-analyzer.lock"
REAL_RESTART_MESSAGE = ("Реальный анализ остановлен. Проверьте и перезапустите локальные серверы моделей, "
                        "затем сервер приложения. Новые задания не запускаются.")


class InstanceLock:
    """An OS-held lock prevents two servers from sharing this worker directory."""
    def __init__(self, path: Path):
        self.file = path.open("a+b")
        try:
            self.file.seek(0)
            if self.file.read(1) == b"":
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("Another server already holds this data directory or real-analyzer lock. Run one real server per checkout, with one Uvicorn worker.") from exc

    def close(self):
        self.file.close()


def ensure_api_key(settings: Settings) -> str:
    if settings.api_key:
        if len(settings.api_key) < 16:
            raise ValueError("CASH_AUDIT_API_KEY must contain at least 16 characters")
        return settings.api_key
    path = settings.data_dir / "api-key.txt"
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    try:
        path.chmod(0o600)
    except OSError:
        pass
    key = path.read_text(encoding="utf-8").strip()
    if len(key) < 16:
        raise ValueError("Local API key is invalid; configure CASH_AUDIT_API_KEY")
    return key


class JobService:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = settings.data_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.api_key = ensure_api_key(settings)
        self.store = JobStore(settings.data_dir / "jobs.sqlite3")
        self._mutex = threading.RLock()
        self._worker_mutex = threading.RLock()
        self._stop = threading.Event()
        self._reservations: set[str] = set()
        self._uploads: set[tuple[str, str]] = set()
        self._rates = defaultdict(deque)
        self._process = None
        self._connection = None
        self._active = None
        self._worker_started = 0
        self._startup_waiting = False
        self.ready = False
        self.analysis_scope = None
        self.readiness_message = "Исполнитель запускается. Подождите несколько секунд."
        self._instance_lock = None
        self._real_lock = None
        self._deadlines: dict[str, float] = {}
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("JobService has already been started")
        self._instance_lock = InstanceLock(self.settings.data_dir / "server.lock")
        try:
            if self.settings.mode == "real":
                REAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                self._real_lock = InstanceLock(REAL_LOCK_PATH)
            self.store.recover()
            self._cleanup_orphan_uploads()
            self._launch_worker()
            self._thread = threading.Thread(target=self._run, name="cash-audit-jobs", daemon=True)
            self._thread.start()
        except BaseException:
            self._terminate_worker()
            self._instance_lock.close()
            self._instance_lock = None
            if self._real_lock:
                self._real_lock.close()
                self._real_lock = None
            raise

    def _launch_worker(self):
        with self._worker_mutex:
            if self._stop.is_set():
                return
            self.ready = False
            self.analysis_scope = None
            self.readiness_message = "Исполнитель запускается. Подождите несколько секунд."
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=worker_entry,
                                      args=(child, self.settings.mode, self.settings.mock_delay_seconds),
                                      name="cash-audit-analyzer", daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                process.close()
                raise
            finally:
                child.close()
            self._connection = parent
            self._process = process
            self._worker_started = time.monotonic()
            self._startup_waiting = True

    def _terminate_worker(self):
        with self._worker_mutex:
            self.ready = False
            self.analysis_scope = None
            self._startup_waiting = False
            if self._process is not None:
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=3)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2)
                if not self._process.is_alive():
                    self._process.close()
                self._process = None
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        self._terminate_worker()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        for job in self.store.pending():
            self.store.finish(job["job_id"], error="server_stopped")
        if self._instance_lock:
            self._instance_lock.close()
            self._instance_lock = None
        if self._real_lock:
            self._real_lock.close()
            self._real_lock = None

    def _unavailable(self, message: str, code: str = "worker_unavailable"):
        self.readiness_message = message
        self._terminate_worker()
        self._active = None
        for job in self.store.pending():
            self.store.finish(job["job_id"], error=code)
            self._deadlines.pop(job["job_id"], None)

    def register_job(self, job: dict, started_monotonic: float):
        """Publish the deadline before the coordinator can discover this job."""
        with self._mutex:
            self._deadlines[job["job_id"]] = started_monotonic + self.settings.job_timeout_seconds
            try:
                self.store.create(job)
            except BaseException:
                self._deadlines.pop(job["job_id"], None)
                raise

    def _deadline(self, job: dict) -> float:
        # Pending jobs are failed on restart, so monotonic timestamps never
        # cross process lifetimes. Fallback supports internal/imported jobs.
        with self._mutex:
            return self._deadlines.setdefault(job["job_id"], time.monotonic() + max(
                0.0, self.settings.job_timeout_seconds - (time.time() - job["created_at"])))

    def _handle_event(self, event: dict):
        kind = event.get("kind")
        if kind == "ready":
            self.analysis_scope = event.get("analysis_scope") if self.settings.mode == "real" else None
            self.ready = True
            self._startup_waiting = False
            self.readiness_message = ("Тестовый режим: ответы и кадры синтетические." if self.settings.mode == "mock" else
                                      "Локальные модели проверены. Анализатор готов; доступные проверки указаны в интерфейсе.")
        elif kind == "result":
            if self._active and event.get("job_id") == self._active["job_id"]:
                if time.monotonic() >= self._deadline(self._active):
                    self.store.finish(self._active["job_id"], error="timeout")
                else:
                    self.store.finish(self._active["job_id"], result=event.get("result"), error=event.get("error"))
                self._deadlines.pop(self._active["job_id"], None)
                self._active = None
                if self.settings.mode == "real" and event.get("quarantine"):
                    self._unavailable(REAL_RESTART_MESSAGE, "real_restart_required")
        else:
            if event.get("error") == "analyzer_contract":
                self._unavailable("Версия анализатора несовместима: нужны analyze_video(video_path, budget_sec=...) "
                                  "и check_ready(). Получите согласованные изменения Данияра и перезапустите сервер.",
                                  "analyzer_contract")
            elif self.settings.mode == "real" and self._active:
                self._unavailable(REAL_RESTART_MESSAGE, "real_restart_required")
            else:
                self._unavailable("Анализатор или локальные модели недоступны. Проверьте оба сервера моделей "
                                  "и зависимости, затем перезапустите приложение.", "analyzer_unavailable")

    def _dispatch(self, job: dict):
        # The upload, validation and queue share one monotonic deadline. The
        # worker subtracts its readiness probe and passes the remainder to AI.
        deadline = self._deadline(job)
        if self.settings.mode == "real" and deadline - time.monotonic() <= 15.0:
            self.store.finish(job["job_id"], error="timeout")
            self._deadlines.pop(job["job_id"], None)
            return
        self._active = job
        self.store.start(job["job_id"])
        self._connection.send({**job, "deadline_monotonic": deadline})

    def _run(self):
        last_cleanup = 0
        try:
            while not self._stop.wait(0.1):
                try:
                    now = time.time()
                    if self._connection is not None and self._connection.poll():
                        try:
                            event = self._connection.recv()
                        except (EOFError, OSError):
                            event = {"kind": "unavailable"}
                        self._handle_event(event)
                    if self._process is not None and not self._process.is_alive():
                        self._unavailable(REAL_RESTART_MESSAGE if self.settings.mode == "real" else
                                          "Исполнитель неожиданно остановился. Проверьте локальные зависимости и перезапустите сервер.",
                                          "real_restart_required" if self.settings.mode == "real" else "worker_unavailable")
                    if self._startup_waiting and time.monotonic() - self._worker_started > self.settings.worker_startup_seconds:
                        self._unavailable("Истекло время запуска анализатора. Проверьте зависимости и перезапустите сервер.", "startup_timeout")
                    for job in self.store.pending():
                        if time.monotonic() >= self._deadline(job):
                            self.store.finish(job["job_id"], error="timeout")
                            self._deadlines.pop(job["job_id"], None)
                            if self._active and self._active["job_id"] == job["job_id"]:
                                self._active = None
                                if self.settings.mode == "real":
                                    self._unavailable(REAL_RESTART_MESSAGE, "real_restart_required")
                                    break
                                else:
                                    self._terminate_worker()
                                    self._launch_worker()
                    if self.ready and self._active is None and not self._stop.is_set():
                        waiting = self.store.pending()
                        if waiting:
                            self._dispatch(waiting[0])
                    if now - last_cleanup > 60:
                        self.cleanup()
                        last_cleanup = now
                except Exception as exc:
                    log.error("Job coordinator error (%s)", type(exc).__name__)
                    self._unavailable(REAL_RESTART_MESSAGE if self.settings.mode == "real" else
                                      "Ошибка исполнителя. Перезапустите сервер после проверки локального журнала.",
                                      "real_restart_required" if self.settings.mode == "real" else "worker_error")
        finally:
            self._terminate_worker()

    def cleanup(self):
        upload_root = self.upload_dir.resolve()
        for job in self.store.expired_jobs(time.time() - self.settings.retention_hours * 3600):
            # Only remove a verified direct child of this server's upload directory.
            directory = Path(job["video_path"]).parent.resolve()
            if directory.parent == upload_root and directory.name == job["job_id"]:
                try:
                    shutil.rmtree(directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                self.store.delete(job["job_id"])
        with self._mutex:
            for key in list(self._rates):
                if not self._rates[key] or self._rates[key][-1] < time.monotonic() - 60:
                    del self._rates[key]

    def _cleanup_orphan_uploads(self):
        """At startup there are no uploads in flight; remove files left before DB commit."""
        upload_root = self.upload_dir.resolve()
        for candidate in self.upload_dir.iterdir():
            try:
                if str(uuid.UUID(candidate.name)) != candidate.name:
                    continue
            except ValueError:
                continue
            directory = candidate.resolve()
            if (candidate.is_symlink() or not directory.is_dir() or directory.parent != upload_root
                    or directory.name != candidate.name or self.store.get(candidate.name) is not None):
                continue
            try:
                shutil.rmtree(directory)
            except OSError:
                log.warning("Could not remove an orphan upload directory; retry on next startup")

    def reserve(self, owner: str, upload_id: str | None, rate_key: str) -> str:
        with self._mutex:
            if not self.ready:
                raise HTTPException(503, detail=self.readiness_message, headers={"Retry-After": "5"})
            if upload_id and (owner, upload_id) in self._uploads:
                raise HTTPException(409, "Этот файл уже загружается. Дождитесь ответа первого запроса.")
            admitted = {job['job_id'] for job in self.store.pending()} | self._reservations
            if len(admitted) >= self.settings.max_pending_jobs:
                raise HTTPException(429, "Очередь заполнена. Повторите загрузку немного позже.", headers={"Retry-After": "5"})
            rate = self._rates[rate_key]
            now = time.monotonic()
            while rate and now - rate[0] >= 60:
                rate.popleft()
            if len(rate) >= self.settings.upload_rate_per_minute:
                raise HTTPException(429, "Слишком много загрузок. Повторите через минуту.", headers={"Retry-After": "60"})
            rate.append(now)
            job_id = str(uuid.uuid4())
            self._reservations.add(job_id)
            if upload_id:
                self._uploads.add((owner, upload_id))
            return job_id

    def release(self, job_id: str, owner: str, upload_id: str | None):
        with self._mutex:
            self._reservations.discard(job_id)
            if upload_id:
                self._uploads.discard((owner, upload_id))

    def public_result(self, job: dict) -> dict:
        result = {"job_id": job["job_id"], "status": job["status"]}
        if job["status"] == "done":
            result.update(json.loads(job["result_json"]))
        return result
