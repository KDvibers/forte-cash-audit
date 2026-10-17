"""Own a temporary HTTPS tunnel and a loopback-only application in one terminal."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import tempfile
import threading
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import Settings, normalize_public_origin

ROOT = Path(__file__).resolve().parent.parent
CLOUDFLARED_VERSION = '2026.10.1'
CLOUDFLARED_SHA256 = '4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6'
QUICK_URL = re.compile(r'https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com(?![A-Za-z0-9._%:@-])')


class LaunchError(RuntimeError):
    """Operator-facing message that contains no credentials or upstream bodies."""


def verified_binary(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError
        with resolved.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    except (OSError, ValueError) as exc:
        raise LaunchError('Не найден cloudflared. Выполните scripts/install-tunnel.ps1 или public.ps1 -Install.') from exc
    if digest != CLOUDFLARED_SHA256:
        raise LaunchError('Контрольная сумма cloudflared отличается от закреплённого релиза. Исполняемый файл не запускался.')
    return resolved


def tunnel_environment() -> dict[str, str]:
    # Never reuse another tunnel, account token, custom Host override, or debug setting.
    return {key: value for key, value in os.environ.items()
            if not key.upper().startswith(('TUNNEL_', 'CLOUDFLARED_', 'CASH_AUDIT_'))}


def reserve_listener(port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == 'nt':
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(('127.0.0.1', port))
        listener.listen(16)
        return listener
    except OSError as exc:
        listener.close()
        raise LaunchError('Локальный порт занят или недоступен. Выберите другой -Port; чужие процессы не останавливались.') from exc


@contextmanager
def tunnel_process(binary: Path, port: int, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / 'config.yml'
    config.write_text('{}\n', encoding='utf-8')
    events = queue.Queue(maxsize=1)
    process = None
    reader = None
    with (directory / 'cloudflared.log').open('w', encoding='utf-8') as logfile:
        try:
            process = subprocess.Popen([
                str(binary), 'tunnel', '--config', str(config), '--no-autoupdate',
                '--loglevel', 'info', '--protocol', 'http2', '--metrics', '127.0.0.1:0',
                '--url', f'http://127.0.0.1:{port}',
            ], cwd=ROOT, env=tunnel_environment(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

            def read_output():
                for line in process.stdout:
                    # Info-level diagnostics stay local; do not print provider output or HTTP headers.
                    logfile.write(line)
                    logfile.flush()
                    match = QUICK_URL.search(line)
                    if match:
                        try:
                            events.put_nowait(normalize_public_origin(match.group()))
                        except queue.Full:
                            pass

            reader = threading.Thread(target=read_output, name='tunnel-log', daemon=True)
            reader.start()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise LaunchError('Туннель завершился до выдачи адреса. Проверьте локальный журнал cloudflared.log.')
                try:
                    origin = events.get(timeout=0.25)
                    break
                except queue.Empty:
                    pass
            else:
                raise LaunchError('Туннель не выдал адрес за 60 секунд. Проверьте доступ к Cloudflare; входящие порты открывать не требуется.')
            yield origin, process
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if reader is not None:
                    reader.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()


@contextmanager
def application_server(settings: Settings, listener: socket.socket):
    import uvicorn
    from .main import create_app

    application = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(
        application, host='127.0.0.1', port=listener.getsockname()[1], workers=1,
        proxy_headers=True, forwarded_allow_ips='127.0.0.1', access_log=False,
        log_level='warning', timeout_graceful_shutdown=10, limit_concurrency=16,
    ))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]},
                              name='public-http', daemon=True)
    thread.start()
    try:
        yield application, thread
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        if thread.is_alive():
            service = getattr(application.state, 'service', None)
            if service is not None:
                service.stop()
            raise LaunchError('Сервер не завершился штатно. Проверьте процесс перед новым запуском.')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def wait_for_public(origin: str, mode: str, process, thread, timeout: float = 60) -> None:
    opener = build_opener(ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None or not thread.is_alive():
            raise LaunchError('Сервер или туннель завершился во время проверки готовности.')
        try:
            with opener.open(Request(origin + '/health', headers={'Accept': 'application/json'}), timeout=4) as response:
                raw = response.read(4097)
                health = json.loads(raw) if len(raw) <= 4096 else {}
                if response.status == 200 and health.get('mode') == mode and health.get('analyzer_ready') is True:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise LaunchError('HTTPS-проверка готовности не прошла. Проверьте сеть (исходящий TCP 7844), журнал туннеля и готовность анализатора; публичный запуск остановлен.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('mock', 'real'), default='mock')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--scenario', choices=('clean', 'one', 'multiple', 'error'), default='multiple')
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--cloudflared', type=Path, default=ROOT / '.tools' / 'cloudflared' / 'cloudflared.exe')
    args = parser.parse_args()
    listener = None
    try:
        if not 1 <= args.port <= 65535:
            raise LaunchError('Порт должен быть от 1 до 65535.')
        binary = verified_binary(args.cloudflared)
        data = ROOT / 'backend' / '.local' / ('public-' + args.mode)
        settings = Settings(mode=args.mode, data_dir=data, max_upload_mb=90,
                            mock_scenario=args.scenario, mock_delay_seconds=args.delay)
        from .doctor import run_checks
        checks = run_checks(mode=args.mode, port=args.port, data_dir=data, skip_port=True)
        for check in checks:
            print(f'[{check.status}] {check.name}: {check.detail}', flush=True)
        if any(check.status == 'FAIL' for check in checks):
            raise LaunchError('Исправьте ошибки диагностики. Туннель ещё не запускался; автоматической подмены real на mock нет.')
        if args.mode == 'mock':
            from .make_sample import create_sample
            create_sample(data / 'sample.mov')
        listener = reserve_listener(args.port)
        run = Path(tempfile.mkdtemp(prefix='session-', dir=data))
        print('Создаётся временный HTTPS-доступ. Трафик проходит через Cloudflare; анализ выполняется на этом компьютере.', flush=True)
        print(f'Режим: {args.mode}. Лимит файла: 90 МиБ. Локальные данные: {data}', flush=True)
        print(f'Журнал соединения: {run / "cloudflared.log"}', flush=True)
        with tunnel_process(binary, args.port, run) as (origin, process):
            with application_server(replace(settings, public_origin=origin), listener) as (app, thread):
                try:
                    wait_for_public(origin, args.mode, process, thread, settings.worker_startup_seconds + 30)
                    print(f'\nHTTPS готов: {origin}', flush=True)
                    print(f'API: {origin}/api/v1/analyze', flush=True)
                    print(f'Ключ API хранится в {data / "api-key.txt"}; его значение не выводится.', flush=True)
                    if args.mode == 'mock':
                        print(f'ТЕСТОВЫЙ РЕЖИМ: ответы синтетические. Для показа выберите {data / "sample.mov"}', flush=True)
                    print('Откройте ссылку вручную. Не закрывайте этот терминал. Остановка сервера и ссылки: Ctrl+C.', flush=True)
                    while process.poll() is None and thread.is_alive():
                        time.sleep(0.5)
                    raise LaunchError('Соединение или сервер завершился. Для новой ссылки повторите запуск.')
                finally:
                    # Remove external access before waiting for application shutdown.
                    if process.poll() is None:
                        process.terminate()
    except KeyboardInterrupt:
        print('\nПубличный запуск остановлен. Данные сохранены, временная ссылка больше не обслуживается.')
        return 0
    except LaunchError as exc:
        print(f'[FAIL] {exc}')
        return 1
    except Exception:
        print('[FAIL] Не удалось запустить публичный доступ. Проверьте окружение и локальные журналы; подробности и секреты не выводятся.')
        return 1
    finally:
        if listener is not None:
            listener.close()


if __name__ == '__main__':
    raise SystemExit(main())
