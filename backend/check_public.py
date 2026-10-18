"""Check a public HTTPS interface and API using two neutral synthetic uploads.

The API key is read from a local file, never from a command-line value. Requests
never follow redirects, retry uploads, or send existing bank recordings.
"""
from __future__ import annotations

import argparse
from http.cookiejar import CookieJar
from http.cookies import CookieError, SimpleCookie
import json
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener
import uuid

from .browser_sessions import COOKIE_NAME
from .check_api import CheckError, NoRedirect, read_key
from .config import normalize_public_origin
from .make_sample import create_sample
from .validation import validate_result


REQUEST_TIMEOUT = 15
JOB_TIMEOUT = 330
MAX_RESULT_BYTES = 48 * 1024 * 1024
ASSETS = {
    "styles.css": "text/css", "results.css": "text/css",
    "app.js": "text/javascript", "results.js": "text/javascript",
    "recovery.js": "text/javascript",
}


def https_origin(value: str) -> str:
    """Accept one explicit HTTPS origin, not a URL containing a path or secret."""
    try:
        return normalize_public_origin(value)
    except ValueError as exc:
        raise CheckError("Укажите HTTPS адрес сервера без пути, логина, пароля, параметров и фрагмента.") from exc


def fetch(opener, request: Request, *, limit=65536, timeout=REQUEST_TIMEOUT):
    with opener.open(request, timeout=timeout) as response:
        body = response.read(limit + 1)
        headers = response.headers
        if response.status != 200 or len(body) > limit:
            raise CheckError("Сервер вернул неожиданный статус или слишком большой ответ.")
    return body, headers


def json_response(opener, request: Request, *, limit=65536, timeout=REQUEST_TIMEOUT):
    body, headers = fetch(opener, request, limit=limit, timeout=timeout)
    if headers.get_content_type() != "application/json":
        raise CheckError("Вместо JSON сервер вернул страницу или другой формат ответа.")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise CheckError("API вернул JSON неверного формата.")
    return result, headers


def expect_status(opener, request: Request, expected: int):
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT):
            pass
    except HTTPError as exc:
        exc.close()
        if exc.code == expected:
            return
    raise CheckError(f"Проверка ограничения доступа не вернула ожидаемый HTTP {expected}.")


def check_frontend(opener, url: str):
    body, headers = fetch(opener, Request(url + "/"), limit=2 * 1024 * 1024)
    if headers.get_content_type() != "text/html" or b"Forte Audit" not in body:
        raise CheckError("По внешнему адресу не найден интерфейс Forte Audit.")
    for name, media_type in ASSETS.items():
        if f"/assets/{name}".encode() not in body:
            raise CheckError("Главная страница не содержит необходимые файлы интерфейса.")
        content, asset_headers = fetch(opener, Request(url + "/assets/" + name), limit=2 * 1024 * 1024)
        if not content.strip() or asset_headers.get_content_type() != media_type:
            raise CheckError("Не удалось получить необходимые стили или код интерфейса.")


def check_cookie(headers):
    cookies = SimpleCookie()
    for value in headers.get_all("Set-Cookie", []):
        cookies.load(value)
    cookie = cookies.get(COOKIE_NAME)
    if (cookie is None or not cookie.value or not cookie["secure"] or not cookie["httponly"]
            or cookie["samesite"].lower() != "strict" or cookie["path"] != "/ui" or cookie["domain"]):
        raise CheckError("Браузерная сессия не получила cookie Secure, HttpOnly, SameSite=Strict и Path=/ui.")


def upload(opener, url: str, video: Path, headers: dict, *, browser=False, upload_id=None):
    boundary = "public-check-" + uuid.uuid4().hex
    prefix = (f"--{boundary}\r\n"
              'Content-Disposition: form-data; name="file"; filename="public-check.mov"\r\n'
              "Content-Type: video/quicktime\r\n\r\n").encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")

    def body():
        yield prefix
        with video.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk
        yield suffix

    headers = {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}",
               "Content-Length": str(len(prefix) + video.stat().st_size + len(suffix))}
    if upload_id:
        headers["X-Upload-ID"] = upload_id
    path = "/ui/analyze" if browser else "/api/v1/analyze"
    result, _ = json_response(opener, Request(url + path, data=body(), method="POST", headers=headers), timeout=60)
    job_id = result.get("job_id")
    if not isinstance(job_id, str) or not 1 <= len(job_id) <= 200:
        raise CheckError("Ответ загрузки не содержит корректный job_id. Повторная загрузка не выполнялась.")
    return job_id


def poll(opener, url: str, job_id: str, headers: dict, *, browser=False):
    deadline = time.monotonic() + JOB_TIMEOUT
    path = "/ui/jobs/" if browser else "/api/v1/jobs/"
    while time.monotonic() < deadline:
        result, _ = json_response(opener, Request(url + path + quote(job_id, safe=""), headers=headers),
                                  limit=MAX_RESULT_BYTES,
                                  timeout=min(REQUEST_TIMEOUT, max(0.1, deadline - time.monotonic())))
        if result.get("job_id") != job_id:
            raise CheckError("Сервер вернул результат другого задания.")
        if result.get("status") == "done":
            try:
                validated = validate_result(result)
            except ValueError as exc:
                raise CheckError("Результат не соответствует контракту или содержит повреждённые JPEG-кадры.") from exc
            return len(validated["violations"])
        if result.get("status") == "failed":
            raise CheckError("Анализ завершился ошибкой. Это не результат «нарушений нет».")
        if result.get("status") != "processing":
            raise CheckError("API вернул неизвестный статус задания.")
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise CheckError("Истекло время ожидания результата. Повторная загрузка не выполнялась.")


def run_check(url: str, expected_mode: str, key: str, api, browser):
    if expected_mode != "mock":
        raise CheckError("Нейтральная проверка предназначена только для mock. Для real используйте scripts/check-real.ps1 с явным MOV.")
    # Readiness and mode are checked before any POST, including access probes.
    health, _ = json_response(api, Request(url + "/health"))
    if health.get("mode") != expected_mode:
        raise CheckError("Режим сервера отличается от ожидаемого. Видео не отправлено.")
    if health.get("analyzer_ready") is not True or health.get("status") != "ok":
        raise CheckError("Сервер или анализатор не готовы. Видео не отправлено.")
    print(f"[OK] HTTPS и готовность сервера; режим: {expected_mode}.")
    if expected_mode == "mock":
        print("ТЕСТОВЫЙ РЕЖИМ: результаты и кадры синтетические.")
    check_frontend(api, url)
    print("[OK] Главная страница, стили и код интерфейса доступны.")
    expect_status(api, Request(url + "/api/v1/analyze", data=b"", method="POST"), 401)
    expect_status(api, Request(url + "/api/v1/jobs/missing", headers={"X-API-Key": "invalid-" + uuid.uuid4().hex}), 401)
    print("[OK] API отклоняет запросы без ключа и с неверным ключом.")

    session, session_headers = json_response(browser, Request(url + "/ui/session"))
    check_cookie(session_headers)
    token = session.get("csrf_token")
    if (not isinstance(token, str) or not 32 <= len(token) <= 200 or not token.isascii()
            or any(ord(char) <= 32 for char in token) or session.get("mode") != expected_mode
            or session.get("analyzer_ready") is not True):
        raise CheckError("Браузерная сессия не подтвердила готовность и режим сервера.")
    browser_headers = {"Origin": url, "X-CSRF-Token": token}
    expect_status(browser, Request(url + "/ui/analyze", data=b"", method="POST",
                                   headers={**browser_headers, "Origin": "https://wrong-origin.invalid"}), 403)
    expect_status(browser, Request(url + "/ui/analyze", data=b"", method="POST", headers={"Origin": url}), 403)
    print("[OK] HTTPS cookie и защита браузерной загрузки по Origin и CSRF.")

    print("Отправляются два нейтральных MOV: один через API, один через браузерный маршрут.")
    with tempfile.TemporaryDirectory(prefix="forte-public-check-") as directory:
        video = create_sample(Path(directory) / "neutral.mov")
        api_job = upload(api, url, video, {"X-API-Key": key})
        count = poll(api, url, api_job, {"X-API-Key": key})
        print(f"[OK] API: загрузка, обработка и результат; событий: {count}.")
        identifier = str(uuid.uuid4())
        browser_job = upload(browser, url, video, browser_headers, browser=True, upload_id=identifier)
        count = poll(browser, url, browser_job, {}, browser=True)
        recovered, _ = json_response(browser, Request(url + "/ui/recovery/" + identifier))
        if (recovered.get("job_id") != browser_job or recovered.get("mode") != expected_mode
                or recovered.get("status") != "done" or recovered.get("file_size") != video.stat().st_size):
            raise CheckError("Восстановление проверки вернуло неверное задание или метаданные.")
        expect_status(browser, Request(url + "/ui/jobs/" + quote(api_job, safe="")), 404)
        print(f"[OK] Браузерная загрузка, результат и восстановление; событий: {count}.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTPS адрес без пути.")
    parser.add_argument("--expect-mode", choices=("mock", "real"), required=True)
    parser.add_argument("--key-file", type=Path, required=True, help="Локальный файл API-ключа.")
    args = parser.parse_args(argv)
    try:
        url = https_origin(args.base_url)
        key = read_key(args.key_file)
        if not key.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in key):
            raise CheckError("Файл API-ключа содержит неподходящий формат.")
        api = build_opener(ProxyHandler({}), NoRedirect())
        browser = build_opener(ProxyHandler({}), NoRedirect(), HTTPCookieProcessor(CookieJar()))
        started = time.monotonic()
        run_check(url, args.expect_mode, key, api, browser)
        print(f"Внешняя проверка пройдена за {time.monotonic() - started:.1f} с.")
        print("Проверены HTTP-маршруты интерфейса и API. Это не визуальная проверка браузера и не оценка точности модели.")
        return 0
    except HTTPError as exc:
        exc.close()
        print(f"Ошибка HTTP {exc.code}. Проверьте адрес, ключ и сервер. Перенаправления запрещены.")
    except CheckError as exc:
        print(f"Ошибка проверки: {exc}")
    except (URLError, TimeoutError, OSError, ValueError, CookieError, subprocess.SubprocessError):
        print("Ошибка соединения, TLS, локального файла или формата ответа. Секреты и ответы сервера не выводятся.")
    except KeyboardInterrupt:
        print("Проверка остановлена. Уже принятые задания продолжают обрабатываться на сервере.")
        return 130
    print("Повторная отправка видео не выполнялась.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
