"""Explicit-video HTTPS acceptance check; never creates a synthetic real input.

Each upload has its own 300-second deadline including network transfer, queue,
result transfer and JPEG validation. Ignored local reports contain job IDs,
counts/timings, not video paths, keys, sessions, descriptions or base64 evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
import hashlib
import http.client
from http.cookiejar import CookieJar
import json
from pathlib import Path
import socket
import ssl
import threading
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request
import uuid

from .check_api import CheckError, read_key
from .check_public import check_cookie, https_origin
from .validation import validate_result

ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = ROOT / "backend" / ".local" / "check-real"
TOTAL_SECONDS = 300
MAX_RESULT_BYTES = 48 * 1024 * 1024
CHUNK_BYTES = 64 * 1024


class AcceptanceError(CheckError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise AcceptanceError("timeout", "Истёк срок текущего запроса или проверки; повторной отправки нет.")
    return value


@dataclass
class Reply:
    status: int
    headers: Message
    body: bytes

    def info(self):
        return self.headers


class HTTPSClient:
    """Direct TLS with a fresh connection; no environment proxy or redirects."""
    def __init__(self, origin: str):
        self.origin = https_origin(origin)
        self.parts = urlsplit(self.origin)
        self.cookies = CookieJar()
        self.transport_progress = None

    def request(self, method: str, path: str, *, headers=None, chunks=None,
                length=None, deadline=None, limit=65536) -> Reply:
        # Numeric progress only: a completed send is not proof of server receipt.
        # Reset per request; upload counters include multipart framing, not headers.
        self.transport_progress = {
            "transport_phase": "connect", "body_bytes_handed_to_socket": 0,
            "declared_body_bytes": length if type(length) is int and length >= 0 else 0,
        }
        deadline = deadline if deadline is not None else time.monotonic() + 15
        request = Request(self.origin + path, headers=headers or {}, method=method)
        self.cookies.add_cookie_header(request)
        connection = http.client.HTTPSConnection(
            self.parts.hostname, self.parts.port or 443,
            timeout=remaining(deadline), context=ssl.create_default_context())
        active_socket = [None]

        def expire_connection():
            # Interrupt a peer that keeps dribbling header bytes, too. System DNS
            # resolution itself cannot be cancelled portably; its late result fails.
            transport = active_socket[0] or connection.sock
            if transport is not None:
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

        timer = threading.Timer(remaining(deadline), expire_connection)
        timer.daemon = True
        timer.start()
        try:
            connection.connect()
            active_socket[0] = connection.sock
            remaining(deadline)  # Includes DNS, TCP and TLS; late results cannot pass.
            self.transport_progress["transport_phase"] = "send_body"
            connection.sock.settimeout(remaining(deadline))
            connection.putrequest(method, path, skip_accept_encoding=True)
            for name, value in request.header_items():
                connection.putheader(name, value)
            if length is not None:
                connection.putheader("Content-Length", str(length))
            connection.endheaders()
            if chunks is not None:
                for chunk in chunks:
                    connection.sock.settimeout(remaining(deadline))
                    connection.send(chunk)
                    self.transport_progress["body_bytes_handed_to_socket"] += len(chunk)
                    remaining(deadline)
            self.transport_progress["transport_phase"] = "wait_headers"
            connection.sock.settimeout(remaining(deadline))
            response = connection.getresponse()
            self.transport_progress["transport_phase"] = "read_response"
            # http.client can clear connection.sock for a Connection: close reply.
            response_socket = connection.sock
            if response_socket is None:
                response_socket = getattr(getattr(response.fp, "raw", None), "_sock", None)
            active_socket[0] = response_socket
            body = bytearray()
            while True:
                if response_socket is not None:
                    response_socket.settimeout(remaining(deadline))
                chunk = response.read1(min(CHUNK_BYTES, limit + 1 - len(body)))
                remaining(deadline)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > limit:
                    raise AcceptanceError("invalid_result", "Ответ превышает допустимый размер.")
            reply = Reply(response.status, response.headers, bytes(body))
            self.cookies.extract_cookies(reply, request)
            return reply
        except (socket.timeout, TimeoutError):
            raise AcceptanceError("timeout", "Сетевой запрос не завершился в срок; повторной отправки нет.") from None
        except (OSError, http.client.HTTPException):
            remaining(deadline)  # A watchdog interruption is a timeout, not a failed verdict.
            raise
        finally:
            timer.cancel()
            connection.close()


def get_json(reply: Reply, expected=200) -> dict:
    if reply.status != expected:
        raise AcceptanceError("http_error", f"Получен HTTP {reply.status}; перенаправления и повторная отправка запрещены.")
    if reply.headers.get_content_type() != "application/json":
        raise AcceptanceError("invalid_result", "Сервер вернул ответ не в формате JSON.")
    try:
        result = json.loads(reply.body)
    except (ValueError, UnicodeError):
        raise AcceptanceError("invalid_result", "Ответ JSON повреждён.") from None
    if not isinstance(result, dict):
        raise AcceptanceError("invalid_result", "Ожидался объект JSON.")
    return result


def check_response_modes(reply: Reply) -> None:
    # These diagnostics are optional for compatible API implementations. When
    # present, neither the current service nor the stored job may claim mock.
    for name in ("X-Cash-Audit-Mode", "X-Cash-Audit-Job-Mode"):
        if reply.headers.get(name) not in (None, "real"):
            raise AcceptanceError("wrong_mode", "Ответ помечен другим режимом обработки.")


def strict_health(client) -> None:
    health = get_json(client.request("GET", "/health"))
    if health.get("mode") != "real":
        raise AcceptanceError("wrong_mode", "Ожидался real. Видео не отправлено.")
    if health.get("status") != "ok" or health.get("analyzer_ready") is not True:
        raise AcceptanceError("not_ready", "Настоящий анализатор не готов. Видео не отправлено.")


def browser_session(client) -> str:
    reply = client.request("GET", "/ui/session")
    session = get_json(reply)
    check_cookie(reply.headers)
    token = session.get("csrf_token")
    if (session.get("mode") != "real" or session.get("analyzer_ready") is not True
            or not isinstance(token, str) or not 32 <= len(token) <= 200
            or not token.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in token)):
        raise AcceptanceError("not_ready", "Браузерная сессия не подтвердила real и готовность.")
    return token


def access_checks(client, origin: str) -> None:
    for path, headers in (("/api/v1/analyze", {}),
                          ("/api/v1/analyze", {"X-API-Key": "invalid-" + uuid.uuid4().hex})):
        if client.request("POST", path, headers=headers, chunks=(), length=0).status != 401:
            raise AcceptanceError("access_error", "API должен отклонять запросы без ключа и с неверным ключом.")
    token = browser_session(client)
    for headers in ({"Origin": origin}, {"Origin": "https://wrong-origin.invalid", "X-CSRF-Token": token}):
        if client.request("POST", "/ui/analyze", headers=headers, chunks=(), length=0).status != 403:
            raise AcceptanceError("access_error", "Не подтверждены проверки Origin и CSRF.")


def file_metadata(video: Path) -> dict:
    if not video.is_file() or video.suffix.lower() != ".mov" or not video.stat().st_size:
        raise AcceptanceError("input_error", "Нужен существующий непустой MOV; путь в отчёт не записывается.")
    digest = hashlib.sha256()
    with video.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    # A fingerprint identifies the tested file without storing it or its filename.
    return {"size_bytes": video.stat().st_size, "sha256": digest.hexdigest()}


def multipart(video: Path):
    boundary = "real-check-" + uuid.uuid4().hex
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
              'filename="check.mov"\r\nContent-Type: video/quicktime\r\n\r\n').encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")

    def chunks():
        yield prefix
        with video.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
                yield chunk
        yield suffix

    return chunks(), len(prefix) + video.stat().st_size + len(suffix), f"multipart/form-data; boundary={boundary}"


def recover(client, job_id: str, identifier: str, size: int, deadline: float) -> None:
    # Model readiness may change after completion. Recovery needs the same cookie,
    # not a new inference or a fresh upload; opening /ui/session models page reload.
    refreshed = get_json(client.request("GET", "/ui/session", deadline=deadline))
    if refreshed.get("mode") != "real":
        raise AcceptanceError("wrong_mode", "Режим сервера изменился при восстановлении.")
    saved = get_json(client.request("GET", "/ui/recovery/" + identifier, deadline=deadline))
    if (saved.get("job_id") != job_id or saved.get("status") != "done"
            or saved.get("mode") != "real" or saved.get("file_size") != size):
        raise AcceptanceError("recovery_error", "Восстановление вернуло другое задание или метаданные.")
    existing = get_json(client.request("GET", "/ui/jobs/" + quote(job_id, safe=""),
                                     deadline=deadline, limit=MAX_RESULT_BYTES))
    if existing.get("job_id") != job_id or existing.get("status") != "done":
        raise AcceptanceError("recovery_error", "Готовое задание не восстановилось через GET.")
    validate_result(existing)
    remaining(deadline)


def check_one(origin: str, key: str, video: Path, metadata: dict, *, channel="api",
              require_events=False, client_factory=HTTPSClient, gate=None) -> dict:
    client = client_factory(origin)
    record = {**metadata, "channel": channel, "outcome": "not_started", "within_300_seconds": False}
    started = accepted = completed = None
    phase, previous_progress = "preflight", getattr(client, "transport_progress", None)
    try:
        strict_health(client)
        headers = {"X-API-Key": key}
        identifier = None
        if channel == "ui":
            token = browser_session(client)
            identifier = str(uuid.uuid4())
            headers = {"Origin": origin, "X-CSRF-Token": token, "X-Upload-ID": identifier}
        phase, previous_progress = "upload", getattr(client, "transport_progress", None)
        chunks, length, content_type = multipart(video)
        if gate is not None:
            gate.wait(timeout=30)
        headers["Content-Type"] = content_type
        started = time.monotonic()
        deadline = started + TOTAL_SECONDS
        path = "/ui/analyze" if channel == "ui" else "/api/v1/analyze"
        accepted_reply = client.request("POST", path, headers=headers, chunks=chunks,
                                        length=length, deadline=deadline)
        check_response_modes(accepted_reply)
        accepted_result = get_json(accepted_reply)
        remaining(deadline)
        accepted = time.monotonic()
        job_id = accepted_result.get("job_id")
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 200:
            raise AcceptanceError("invalid_result", "Загрузка не вернула корректный job_id; повторной отправки нет.")
        record["job_id"] = job_id  # Local report only, for a separate metadata-only queue audit.
        phase, previous_progress = "poll", getattr(client, "transport_progress", None)
        get_headers = {} if channel == "ui" else {"X-API-Key": key}
        job_path = ("/ui/jobs/" if channel == "ui" else "/api/v1/jobs/") + quote(job_id, safe="")
        while True:
            remaining(deadline)
            reply = client.request("GET", job_path, headers=get_headers, deadline=deadline, limit=MAX_RESULT_BYTES)
            check_response_modes(reply)
            result = get_json(reply)
            remaining(deadline)
            if result.get("job_id") != job_id:
                raise AcceptanceError("invalid_result", "Получен результат другого задания.")
            status = result.get("status")
            if status == "failed":
                raise AcceptanceError("analysis_failed", "Анализ завершился failed; заключение не выдано.")
            if status == "done":
                try:
                    validated = validate_result(result)
                except ValueError:
                    raise AcceptanceError("invalid_result", "Контракт, таймкоды или JPEG не прошли проверку.") from None
                remaining(deadline)
                completed = time.monotonic()
                events = validated["violations"]
                record.update(verdict=validated["verdict"], events=len(events),
                              jpeg_frames=sum(len(event["evidence_frames"]) for event in events),
                              within_300_seconds=True)
                if require_events and not events:
                    raise AcceptanceError("expected_events_missing", "Для положительного сценария ожидались события, получен no_violation.")
                if channel == "ui":
                    phase, previous_progress = "recovery", getattr(client, "transport_progress", None)
                    recovery_started = time.monotonic()
                    recover(client, job_id, identifier, metadata["size_bytes"], time.monotonic() + 20)
                    record["recovery_get_only"] = True
                    record["recovery_seconds"] = round(time.monotonic() - recovery_started, 3)
                record["outcome"] = "passed"
                break
            if status != "processing":
                raise AcceptanceError("invalid_result", "Неизвестный статус задания.")
            time.sleep(min(0.5, remaining(deadline)))
    except AcceptanceError as exc:
        record["outcome"] = exc.code
    except (OSError, http.client.HTTPException, ssl.SSLError):
        record["outcome"] = "network_error"
    except (ValueError, CheckError):
        record["outcome"] = "invalid_result"
    except Exception:
        # Includes a broken concurrency gate; never print exception bodies/paths.
        record["outcome"] = "check_error"
    finally:
        if record["outcome"] != "passed":
            record["failure_phase"] = phase
            progress = getattr(client, "transport_progress", None)
            # Do not mislabel a previous phase's request or copy arbitrary client data.
            if isinstance(progress, dict) and progress is not previous_progress:
                transport_phase = progress.get("transport_phase")
                if isinstance(transport_phase, str) and transport_phase in {"connect", "send_body", "wait_headers", "read_response"}:
                    record["transport_phase"] = transport_phase
                for field in ("body_bytes_handed_to_socket", "declared_body_bytes"):
                    value = progress.get(field)
                    if type(value) is int and value >= 0:
                        record[field] = value
        ended = completed if completed is not None else time.monotonic()
        if started is not None:
            record["total_seconds"] = round(ended - started, 3)
            if accepted is not None:
                record["upload_seconds"] = round(accepted - started, 3)
                record["wait_and_validate_seconds"] = round(ended - accepted, 3)
    return record


def report_path(path: Path | None) -> Path:
    target = path or REPORT_ROOT / ("run-" + uuid.uuid4().hex + ".json")
    resolved = target.resolve()
    if not resolved.is_relative_to(REPORT_ROOT.resolve()) or resolved.suffix.lower() != ".json":
        raise AcceptanceError("input_error", "Отчёт разрешён только в backend/.local/check-real/*.json вне Git.")
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--second-video", type=Path, help="Второй MOV: две одновременные проверки, отдельный лимит каждой.")
    parser.add_argument("--channel", choices=("api", "ui"), default="api")
    parser.add_argument("--require-events", action="store_true", help="Ожидать хотя бы одно событие в каждом результате.")
    parser.add_argument("--report", type=Path, help="JSON только внутри backend/.local/check-real.")
    args = parser.parse_args(argv)
    try:
        origin = https_origin(args.base_url)
        key = read_key(args.key_file)
        if not key.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in key):
            raise AcceptanceError("input_error", "Неверный формат файла API-ключа.")
        destination = report_path(args.report)
        if destination.exists():
            raise AcceptanceError("input_error", "Такой отчёт уже существует. Укажите новое имя до отправки видео.")
        videos = [args.video] + ([args.second_video] if args.second_video else [])
        metadata = [file_metadata(video) for video in videos]
        preflight = HTTPSClient(origin)
        strict_health(preflight)
        access_checks(preflight, origin)
        print("[OK] Real готов; API-ключ и Origin/CSRF проверены. Отправляются только явно выбранные MOV.")
        gate = threading.Barrier(2) if len(videos) == 2 else None
        with ThreadPoolExecutor(max_workers=len(videos)) as executor:
            futures = [executor.submit(check_one, origin, key, video, info, channel=args.channel,
                                       require_events=args.require_events, gate=gate)
                       for video, info in zip(videos, metadata)]
            records = [future.result() for future in futures]
        report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "real",
                  "deadline_seconds_per_upload": TOTAL_SECONDS, "concurrent_uploads": len(videos),
                  "require_events": args.require_events, "checks": records,
                  "quality_evaluated": False, "independent_external_device_confirmed": False}
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Never replace an existing acceptance record.
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        for number, item in enumerate(records, 1):
            print(f"Проверка {number}: {item['outcome']}; всего {item.get('total_seconds', 0):.2f} с; "
                  f"событий {item.get('events', 0)}; JPEG {item.get('jpeg_frames', 0)}.")
        print("Отчёт сохранён в backend/.local/check-real; ключи, пути видео, тексты и кадры не записаны.")
        print("Проверка JPEG/таймкодов подтверждает формат, но не подлинность кадра или правильность распознавания.")
        print("Запуск с подключённым Tailscale может идти по частному маршруту. Повторите с внешнего устройства без Tailscale.")
        return 0 if all(item["outcome"] == "passed" for item in records) else 1
    except AcceptanceError as exc:
        print(f"[FAIL] {exc.code}: {exc}")
    except (OSError, ValueError, CheckError, http.client.HTTPException):
        print("[FAIL] Ошибка входных данных, сети или формата; секреты и ответы сервера не выводятся.")
    except KeyboardInterrupt:
        print("Проверка остановлена. Уже принятые задания не отменены; повторной загрузки нет.")
        return 130
    except Exception:
        print("[FAIL] Проверка не завершена; внутренние данные и ответы сервера не выводятся.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
