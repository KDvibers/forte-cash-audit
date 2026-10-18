"""Run one loopback application and its owned foreground Tailscale Funnel."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, Request, build_opener

from .config import Settings, normalize_public_origin
from .public_tunnel import LaunchError, NoRedirect, application_server, reserve_listener


ROOT = Path(__file__).resolve().parent.parent


def client_environment() -> dict[str, str]:
    """Do not inherit account tokens, alternate sockets or application secrets."""
    return {key: value for key, value in os.environ.items()
            if not key.upper().startswith(("TS_", "TAILSCALE_", "CASH_AUDIT_",
                                           "TUNNEL_", "CLOUDFLARED_"))}


def find_binary(path: Path | None = None) -> Path:
    candidate = path or shutil.which("tailscale")
    if not candidate and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe"
    try:
        resolved = Path(candidate).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise LaunchError("Tailscale CLI не найден. Установите официальный клиент и выполните вход на этом ноутбуке.") from None
    return resolved


def read_client_json(binary: Path, *arguments: str) -> dict:
    try:
        result = subprocess.run([str(binary), *arguments], cwd=ROOT, env=client_environment(),
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=15, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except (OSError, subprocess.TimeoutExpired, ValueError):
        raise LaunchError("Tailscale не вернул корректное состояние. Проверьте службу и вход; вывод клиента скрыт.") from None


def device_origin(status: dict) -> str:
    """Accept only this authenticated device's exact advertised certificate name."""
    if status.get("BackendState") != "Running":
        raise LaunchError("Tailscale не подключён. Выполните вход в официальный клиент; сервер ещё не опубликован.")
    device = status.get("Self")
    certificates = status.get("CertDomains")
    try:
        name = device["DNSName"]
        if not isinstance(name, str):
            raise ValueError
        name = name.removesuffix(".")
        origin = normalize_public_origin("https://" + name)
        if (origin != "https://" + name or not name.endswith(".ts.net") or name.count(".") != 3
                or not isinstance(certificates, list) or name not in certificates):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        raise LaunchError("Не подтверждено HTTPS-имя этого устройства в Tailscale. Проверьте MagicDNS и HTTPS-сертификаты.") from None
    configured = os.environ.get("CASH_AUDIT_PUBLIC_ORIGIN")
    if configured:
        try:
            agrees = normalize_public_origin(configured) == origin
        except ValueError:
            agrees = False
        if not agrees:
            raise LaunchError("CASH_AUDIT_PUBLIC_ORIGIN отличается от адреса этого устройства. Очистите старую настройку; публикация не запускалась.")
    return origin


def read_publications(binary: Path) -> dict:
    # These sequential read-only commands can straddle our own startup/cleanup.
    # Accept only an agreeing pair, then let the caller enforce empty/owned state.
    # Persistent disagreement still fails closed; no reset or ownership bypass.
    for attempt in range(3):
        funnel = read_client_json(binary, "funnel", "status", "--json")
        serve = read_client_json(binary, "serve", "status", "--json")
        if funnel == serve:
            return funnel
        if attempt < 2:
            time.sleep(0.1)
    raise LaunchError("Состояние Serve/Funnel изменилось во время проверки. Повторите запуск; чужие настройки не изменялись.")


def require_empty_publications(binary: Path) -> None:
    if read_publications(binary):
        raise LaunchError("На устройстве уже настроен Serve или Funnel. Сначала проверьте tailscale serve status; существующие публикации не изменялись.")


def owns_publication(configuration: dict, origin: str, port: int) -> bool:
    """Recognize exactly one foreground proxy, never a broader existing service."""
    foreground = configuration.get("Foreground")
    if not isinstance(foreground, dict) or len(foreground) != 1:
        return False
    if any(value for key, value in configuration.items() if key != "Foreground"):
        return False
    configuration = next(iter(foreground.values()))
    if not isinstance(configuration, dict):
        return False
    authority = origin.removeprefix("https://") + ":443"
    expected = {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {authority: {"Handlers": {"/": {"Proxy": f"http://127.0.0.1:{port}"}}}},
        "AllowFunnel": {authority: True},
    }
    return configuration == expected


def stop_owned_process(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def verify_publication_removed(binary: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not read_publications(binary):
            return
        time.sleep(0.5)
    # No 'reset' or port-wide 'off': another operator might have changed state.
    raise LaunchError("Процесс нашего Funnel остановлен, но Serve/Funnel всё ещё содержит настройки. Проверьте tailscale serve status; общий reset не выполнялся.")


@contextmanager
def funnel_process(binary: Path, origin: str, port: int, directory: Path):
    # Repeat immediately before mutation, after application startup.
    require_empty_publications(binary)
    directory.mkdir(parents=True, exist_ok=True)
    process = None
    with (directory / "tailscale-funnel.log").open("w", encoding="utf-8") as logfile:
        try:
            process = subprocess.Popen([
                str(binary), "funnel", "--https=443", f"http://127.0.0.1:{port}",
            ], cwd=ROOT, env=client_environment(), stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise LaunchError("Funnel завершился при запуске. Проверьте право Funnel и локальный tailscale-funnel.log; вход выполняется владельцем ноутбука.")
                configuration = read_publications(binary)
                if owns_publication(configuration, origin, port):
                    break
                if configuration:
                    raise LaunchError("Появилась другая или неподдерживаемая конфигурация Serve/Funnel. Запуск остановлен без сброса настроек.")
                time.sleep(0.5)
            else:
                raise LaunchError("Funnel не подтвердил публикацию за 30 секунд. Проверьте разрешение Funnel и локальный журнал.")
            yield process
        finally:
            if process is not None:
                stop_owned_process(process)
                verify_publication_removed(binary)


def wait_for_health(url: str, mode: str, thread, *, process=None,
                    origin: str | None = None, timeout: float = 120) -> None:
    opener = build_opener(ProxyHandler({}), NoRedirect())
    headers = {"Accept": "application/json"}
    if origin is not None:
        # A loopback-only probe reproduces the trusted local proxy, never exposes
        # a bypass remotely and sends no key, session or video.
        headers.update({"Host": origin.removeprefix("https://"), "X-Forwarded-Proto": "https"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not thread.is_alive() or (process is not None and process.poll() is not None):
            raise LaunchError("Сервер или Funnel завершился во время проверки готовности.")
        try:
            with opener.open(Request(url + "/health", headers=headers), timeout=4) as response:
                raw = response.read(4097)
                payload = json.loads(raw) if len(raw) <= 4096 else {}
                if (response.status == 200 and isinstance(payload, dict)
                        and payload.get("status") == "ok" and payload.get("mode") == mode
                        and payload.get("analyzer_ready") is True):
                    return
        except Exception:
            pass
        time.sleep(1)
    raise LaunchError("Проверка готовности HTTP/HTTPS не прошла. Проверьте модель, HTTPS, DNS и сеть; запуск останавливается.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--scenario", choices=("clean", "one", "multiple", "error"), default="multiple")
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--tailscale", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Run diagnostics without starting HTTP or publishing Funnel")
    args = parser.parse_args(argv)
    listener = None
    try:
        if not 1 <= args.port <= 65535:
            raise LaunchError("Порт должен быть от 1 до 65535.")
        binary = find_binary(args.tailscale)
        origin = device_origin(read_client_json(binary, "status", "--json"))
        require_empty_publications(binary)
        data = ROOT / "backend" / ".local" / ("funnel-" + args.mode)
        settings = Settings(mode=args.mode, data_dir=data, public_origin=origin,
                            mock_scenario=args.scenario, mock_delay_seconds=args.delay)
        from .doctor import run_checks
        checks = run_checks(mode=args.mode, port=args.port, data_dir=data, skip_port=not args.check_only)
        for check in checks:
            print(f"[{check.status}] {check.name}: {check.detail}", flush=True)
        if any(check.status == "FAIL" for check in checks):
            raise LaunchError("Исправьте ошибки диагностики. Публикация ещё не запускалась; автоматической подмены real на mock нет.")
        if args.check_only:
            print("Диагностика завершена: сервер, Funnel и анализ видео не запускались. Проверка не подтверждает качество или время реального анализа.", flush=True)
            return 0
        if args.mode == "mock":
            from .make_sample import create_sample
            create_sample(data / "sample.mov")
        listener = reserve_listener(args.port)
        run = Path(tempfile.mkdtemp(prefix="session-", dir=data))
        print(f"Режим: {args.mode}. Локальные данные: {data}", flush=True)
        print(f"Журнал запуска Funnel: {run / 'tailscale-funnel.log'}", flush=True)
        print("TLS завершается на этом ноутбуке; серверы Funnel пересылают зашифрованный трафик. Анализ локальный.", flush=True)
        with application_server(settings, listener) as (_, thread):
            wait_for_health(f"http://127.0.0.1:{args.port}", args.mode, thread,
                            origin=origin, timeout=settings.worker_startup_seconds + 15)
            with funnel_process(binary, origin, args.port, run) as process:
                wait_for_health(origin, args.mode, thread, process=process, timeout=120)
                print(f"\nHTTPS отвечает с этого ноутбука: {origin}", flush=True)
                print("Для подтверждения публичного доступа проверьте ссылку из другой сети без Tailscale.", flush=True)
                print(f"API: {origin}/api/v1/analyze", flush=True)
                print(f"Ключ API: {data / 'api-key.txt'}; значение не выводится.", flush=True)
                if args.mode == "mock":
                    print(f"ТЕСТОВЫЙ РЕЖИМ: синтетические ответы. Нейтральное видео: {data / 'sample.mov'}", flush=True)
                print("Оставьте этот терминал открытым. Ctrl+C остановит приложение и принадлежащую ему публикацию Funnel.", flush=True)
                while process.poll() is None and thread.is_alive():
                    time.sleep(0.5)
                raise LaunchError("Сервер или Funnel завершился. Повторите запуск после проверки состояния.")
    except KeyboardInterrupt:
        print("\nЗапуск остановлен. Собственная публикация Funnel снята; локальные данные сохранены.")
        return 0
    except LaunchError as exc:
        print(f"[FAIL] {exc}")
        return 1
    except Exception:
        print("[FAIL] Не удалось запустить Funnel. Проверьте окружение и локальный журнал; подробности и секреты не выводятся.")
        return 1
    finally:
        if listener is not None:
            listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
