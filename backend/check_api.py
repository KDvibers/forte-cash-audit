"""Run a small public-API smoke check without exposing the API key or image data.

Usage: python -m backend.check_api [--expect-mode real] [--video path.mov] [--key-file path.txt]
The default input is a locally generated neutral MOV, not bank footage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from .config import Settings
from .make_sample import create_sample
from .validation import validate_result


STARTUP_TIMEOUT_SECONDS = 100
JOB_TIMEOUT_SECONDS = 330


class CheckError(Exception):
    """An intentionally safe message, containing neither secrets nor server bodies."""


class NoRedirect(HTTPRedirectHandler):
    """Never forward an authenticated request to a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def base_url(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise CheckError("Укажите HTTP(S) адрес без логина, пароля, параметров и фрагмента.")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def read_key(path: Path | None) -> str:
    if path is not None:
        key = path.read_text(encoding="utf-8-sig").strip()
    elif os.environ.get("CASH_AUDIT_API_KEY"):
        key = os.environ["CASH_AUDIT_API_KEY"].strip()
    else:
        path = Settings.from_env().data_dir / "api-key.txt"
        key = path.read_text(encoding="utf-8-sig").strip()
    if len(key) < 16 or "\r" in key or "\n" in key:
        raise CheckError("Ключ отсутствует или имеет неверный формат.")
    return key


def request_json(opener, request: Request, timeout: float = 15) -> dict:
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise CheckError("API вернул JSON неверного формата.")
    return result


def wait_ready(opener, url: str) -> dict:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    print("Ожидание готовности сервера и исполнителя...")
    while time.monotonic() < deadline:
        try:
            health = request_json(opener, Request(url + "/health"),
                                  timeout=min(15, max(0.1, deadline - time.monotonic())))
            if health.get("analyzer_ready") is True:
                return health
        except HTTPError:
            # A redirect or HTTP error is actionable; do not retry a different destination.
            raise
        except (URLError, TimeoutError):
            pass  # A server that is still starting can refuse the first connection.
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError("Сервер или исполнитель не стали готовы за отведённое время.")


def submit(opener, url: str, key: str, video: Path) -> str:
    boundary = "cash-audit-check-" + uuid.uuid4().hex
    prefix = (f"--{boundary}\r\n"
              'Content-Disposition: form-data; name="file"; filename="check.mov"\r\n'
              "Content-Type: video/quicktime\r\n\r\n").encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")

    def body():
        yield prefix
        with video.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                yield chunk
        yield suffix

    request = Request(url + "/api/v1/analyze", data=body(), method="POST", headers={
        "X-API-Key": key,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(prefix) + video.stat().st_size + len(suffix)),
    })
    result = request_json(opener, request, timeout=60)
    job_id = result.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise CheckError("Ответ загрузки не содержит job_id.")
    return job_id


def poll(opener, url: str, key: str, job_id: str) -> int:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    print("Статус: processing")
    job_url = url + "/api/v1/jobs/" + quote(job_id, safe="")
    while time.monotonic() < deadline:
        result = request_json(opener, Request(job_url, headers={"X-API-Key": key}),
                              timeout=min(15, max(0.1, deadline - time.monotonic())))
        status = result.get("status")
        if status == "done":
            try:
                validated = validate_result(result)
            except ValueError as exc:
                raise CheckError("Ответ done не соответствует контракту результата или содержит повреждённые JPEG-кадры.") from exc
            verdict, events = validated["verdict"], validated["violations"]
            print(f"Статус: done; вердикт: {verdict}; событий: {len(events)}")
            return 0
        if status == "failed":
            print("Статус: failed. Ошибка анализа; это не результат «нарушений нет».")
            return 2
        if status != "processing":
            raise CheckError("API вернул неизвестный статус задания.")
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError("Истекло время ожидания результата; новое задание не создавалось.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", type=Path, help="Путь к MOV; по умолчанию нейтральный тестовый ролик.")
    parser.add_argument("--key-file", type=Path, help="Файл с ключом; значение ключа в командную строку не передаётся.")
    parser.add_argument("--expect-mode", choices=("real", "mock"),
                        help="Остановиться до отправки видео, если сервер работает в другом режиме.")
    args = parser.parse_args()
    try:
        url = base_url(args.base_url)
        key = read_key(args.key_file)
        opener = build_opener(NoRedirect())
        health = wait_ready(opener, url)
        mode = health.get("mode")
        if not isinstance(mode, str) or mode not in {"mock", "real"}:
            raise CheckError("Сервер не сообщил корректный режим обработки.")
        print(f"Режим: {mode}")
        if args.expect_mode is not None and mode != args.expect_mode:
            raise CheckError(f"Ожидался режим {args.expect_mode}, сервер работает в режиме {mode}. Видео не отправлено.")
        if mode == "mock":
            print("ТЕСТОВЫЙ РЕЖИМ: ответы и кадры синтетические.")
        video = args.video
        if video is None:
            video = create_sample(Path(__file__).resolve().parent / ".local" / "sample.mov")
            print("Вход: нейтральный синтетический MOV. Эта проверка не измеряет точность модели.")
        if not video.is_file() or video.suffix.lower() != ".mov":
            raise CheckError("Не найден файл MOV для проверки.")
        started = time.monotonic()
        accepted_at = None
        try:
            job_id = submit(opener, url, key, video)
            accepted_at = time.monotonic()
            print(f"Передача и приём файла: {accepted_at - started:.2f} с.")
            return poll(opener, url, key, job_id)
        finally:
            finished = time.monotonic()
            if accepted_at is not None:
                print(f"Ожидание и проверка ответа: {finished - accepted_at:.2f} с.")
                print(f"Всего от отправки до проверки ответа: {finished - started:.2f} с.")
            else:
                print(f"Время попытки отправки: {finished - started:.2f} с.")
    except HTTPError as exc:
        print(f"Ошибка HTTP {exc.code}. Проверьте адрес, ключ и состояние сервера; перенаправления запрещены.")
    except CheckError as exc:
        print(f"Ошибка проверки: {exc}")
    except (URLError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as exc:
        # No response bodies, request headers, base64 images, or secret values are printed.
        if isinstance(exc, TimeoutError):
            print("Истекло время ожидания сервера или результата.")
        else:
            print("Ошибка соединения, чтения файла или формата ответа. Проверьте сервер и локальные файлы.")
    except KeyboardInterrupt:
        print("Проверка остановлена. Уже принятое задание продолжает обрабатываться на сервере.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
