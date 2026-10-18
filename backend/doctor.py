"""Local preflight checks. No installation, model loading, or inference is performed."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import sys
import tempfile
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import Settings


ROOT = Path(__file__).resolve().parent.parent
MODULES = {
    "fastapi": "fastapi", "starlette": "starlette", "uvicorn": "uvicorn",
    "python-multipart": "multipart", "pillow": "PIL", "imageio-ffmpeg": "imageio_ffmpeg",
}
ASSETS = ("index.html", "styles.css", "app.js", "results.css", "results.js", "recovery.js")


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def check_python() -> Check:
    supported = (sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8
                 and platform.machine().lower() in {"amd64", "x86_64", "x64"})
    return Check("OK" if supported else "FAIL", "Python",
                 "Python 3.12, 64 бита, x64." if supported else
                 "Нужен Python 3.12 для x64; запустите проверку из окружения проекта.")


def check_environment(mode: str, data_dir: Path) -> Check:
    try:
        settings = replace(Settings.from_env(), mode=mode, data_dir=data_dir)
        if settings.api_key and (len(settings.api_key) < 16 or any(c in settings.api_key for c in "\r\n")):
            return Check("FAIL", "Настройки", "Ключ в окружении имеет неверный формат; значение скрыто.")
    except Exception:
        return Check("FAIL", "Настройки", "Проверьте значения CASH_AUDIT_*: конфигурация некорректна. Значения скрыты.")
    return Check("OK", "Настройки", "Параметры окружения корректны; секретные значения не выводятся.")


def pinned_versions(path: Path) -> dict[str, str]:
    pins = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+!-]*)", line)
        if not match:
            raise ValueError("A requirement is not a simple version pin")
        name = match[1].lower().replace("_", "-")
        if name in pins:
            raise ValueError("A requirement is duplicated")
        pins[name] = match[2]
    return pins


DEPENDENCY_PROBE = r'''
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
modules = json.loads(sys.argv[1])
results = {}
for package, module in modules.items():
    try:
        loaded = importlib.import_module(module)
        entry = {"ok": True, "version": importlib.metadata.version(package)}
        if package == "imageio-ffmpeg":
            entry["ffmpeg"] = loaded.get_ffmpeg_exe()
            entry["package_root"] = str(Path(loaded.__file__).resolve().parent)
        results[package] = entry
    except Exception:
        results[package] = {"ok": False}
print(json.dumps(results))
'''


def run_hidden(arguments: list[str], *, timeout: float, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def check_dependencies(root: Path) -> list[Check]:
    try:
        pins = pinned_versions(root / "backend" / "requirements.txt")
        if set(pins) != set(MODULES):
            raise ValueError("Unexpected server requirements")
        process = run_hidden([sys.executable, "-I", "-c", DEPENDENCY_PROBE, json.dumps(MODULES)],
                             timeout=25, cwd=root)
        if process.returncode or len(process.stdout) > 16384:
            raise ValueError("Dependency probe failed")
        result = json.loads(process.stdout)
        if not isinstance(result, dict):
            raise ValueError("Invalid probe output")
    except Exception:
        return [Check("FAIL", "Зависимости сервера",
                      "Не удалось проверить импорты и версии. Проверьте окружение; вывод зависимостей скрыт.")]
    checks = []
    for package in MODULES:
        entry = result.get(package)
        if not isinstance(entry, dict) or entry.get("ok") is not True:
            checks.append(Check("FAIL", package, "Пакет отсутствует или не импортируется; подробности скрыты."))
        elif entry.get("version") != pins[package]:
            checks.append(Check("FAIL", package, "Версия отличается от backend/requirements.txt; ничего не изменено."))
        else:
            checks.append(Check("OK", package, "Импорт работает, версия соответствует требованиям сервера."))
    entry = result.get("imageio-ffmpeg")
    try:
        if not isinstance(entry, dict) or entry.get("ok") is not True:
            raise ValueError("FFmpeg package unavailable")
        executable = Path(entry["ffmpeg"]).resolve(strict=True)
        package_root = Path(entry["package_root"]).resolve(strict=True)
        if not executable.is_file() or not executable.is_relative_to(package_root):
            raise ValueError("FFmpeg is not the bundled executable")
        version = run_hidden([str(executable), "-version"], timeout=5, cwd=root)
        if version.returncode or "ffmpeg version" not in version.stdout.lower():
            raise ValueError("FFmpeg did not respond")
    except Exception:
        checks.append(Check("FAIL", "FFmpeg",
                            "Встроенный FFmpeg недоступен. Проверьте imageio-ffmpeg и настройку IMAGEIO_FFMPEG_EXE."))
    else:
        checks.append(Check("OK", "FFmpeg", "Встроенный исполняемый файл найден и отвечает на -version."))
    return checks


def check_assets(root: Path) -> Check:
    for name in ASSETS:
        try:
            with (root / "frontend" / name).open("rb") as handle:
                if not handle.read(1):
                    raise ValueError("Empty asset")
        except Exception:
            return Check("FAIL", "Интерфейс", f"Файл frontend/{name} отсутствует, пуст или недоступен.")
    return Check("OK", "Интерфейс", "Все файлы интерфейса доступны.")


def check_storage(data_dir: Path) -> Check:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # This context removes only its own newly created probe, including on failure.
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".doctor-", suffix=".tmp", dir=data_dir) as probe:
            probe.write(b"forte-audit-write-check")
            probe.flush()
            os.fsync(probe.fileno())
            probe.seek(0)
            if probe.read() != b"forte-audit-write-check":
                raise OSError("Probe read failed")
    except Exception:
        return Check("FAIL", "Каталог данных", "Не удалось создать, прочитать или удалить временный проверочный файл.")
    return Check("OK", "Каталог данных", "Запись и чтение работают; временный файл удалён.")


def check_port(port: int, skip: bool = False) -> Check:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return Check("FAIL", "Порт API", "Порт должен быть целым числом от 1 до 65535.")
    if skip:
        return Check("WARN", "Порт API", "Проверка свободного порта пропущена по параметру --skip-port.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("127.0.0.1", port))
    except OSError:
        return Check("FAIL", "Порт API", "Локальный порт занят или недоступен. Выберите другой порт; процессы не останавливались.")
    return Check("OK", "Порт API", f"127.0.0.1:{port} доступен сейчас; сервер ещё не запущен.")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_model_health(port: int = 8080) -> Check:
    if isinstance(port, bool) or port not in (8080, 8081):
        return Check("FAIL", "Сервер модели", "Диагностика допускает только объявленные локальные порты 8080 и 8081.")
    address = f"http://127.0.0.1:{port}/health"
    try:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        request = Request(address, headers={"Accept": "application/json"})
        with opener.open(request, timeout=3) as response:
            raw = response.read(4097)
            if response.status != 200 or len(raw) > 4096:
                raise ValueError("Invalid health response")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise ValueError("Model is not ready")
    except Exception:
        return Check("FAIL", f"Сервер модели {port}",
                     f"127.0.0.1:{port}/health не подтвердил готовность. Модель не запускалась, запрос анализа не выполнялся.")
    return Check("OK", f"Сервер модели {port}", "Локальный health подтвердил готовность. Это не проверка качества анализа.")


ANALYZER_PROBE = r'''
import contextlib
import importlib
import inspect
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
result = {"status": "import_failed"}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        module = importlib.import_module("analyzer.service")
        function = getattr(module, "analyze_video", None)
        ready = getattr(module, "check_ready", None)
        result = {"status": "contract_failed"}
        if not callable(function) or not callable(ready):
            raise ValueError()
        signature = inspect.signature(function)
        parameter = signature.parameters.get("budget_sec")
        if parameter is None or parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError()
        signature.bind("not-opened.mov", budget_sec=260.0)
        inspect.signature(ready).bind()
        result = {"status": "models_failed"}
        identities = ready()
        if not isinstance(identities, (list, tuple)) or len(identities) != 2:
            raise ValueError()
        result = {"status": "ok"}
except Exception:
    pass
print(json.dumps(result))
'''


def check_analyzer_runtime(root: Path) -> Check:
    """Import in a short-lived child and call readiness only, never analysis."""
    try:
        process = run_hidden([sys.executable, "-I", "-c", ANALYZER_PROBE, str(root.resolve())],
                             timeout=25, cwd=root)
        if process.returncode or len(process.stdout) > 4096:
            raise ValueError
        payload = json.loads(process.stdout)
        if not isinstance(payload, dict):
            raise ValueError
        state = payload.get("status")
    except Exception:
        state = "probe_failed"
    messages = {
        "ok": "analyze_video принимает budget_sec; проверка идентичности моделей check_ready завершилась успешно. Видео не отправлялось; качество не проверено.",
        "contract_failed": "Нужны analyze_video(video_path, budget_sec=...) и check_ready(). Старая функция с одним аргументом несовместима; подключите актуальную интеграцию Данияра без перезаписи его локальных изменений.",
        "models_failed": "check_ready() не подтвердил обе объявленные локальные модели на 8080 и 8081. Проверьте 2B/4B, порты и идентичность весов; модели не запускались.",
        "import_failed": "Не удалось импортировать analyzer.service. Проверьте актуальные файлы анализатора и зависимости в выбранном окружении; подробности скрыты.",
        "probe_failed": "Проверка интерфейса анализатора не завершилась штатно за отведённое время. Вывод модуля скрыт; анализ видео не запускался.",
    }
    return Check("OK" if state == "ok" else "FAIL", "Интерфейс и модели анализатора",
                 messages.get(state, messages["probe_failed"]))


def check_analyzer_dependencies(root: Path) -> list[Check]:
    """Verify analyzer imports in the same interpreter as the API worker."""
    try:
        pins = pinned_versions(root / "analyzer" / "requirements.txt")
        if set(pins) != {"av", "pillow"}:
            raise ValueError
        process = run_hidden([sys.executable, "-I", "-c", DEPENDENCY_PROBE,
                              json.dumps({"av": "av", "pillow": "PIL"})], timeout=15, cwd=root)
        if process.returncode or len(process.stdout) > 4096:
            raise ValueError
        result = json.loads(process.stdout)
        if not isinstance(result, dict):
            raise ValueError
    except Exception:
        return [Check("FAIL", "Зависимости анализатора", "Не удалось проверить PyAV и Pillow выбранного окружения; вывод зависимостей скрыт.")]
    return [Check("OK" if isinstance(result.get(name), dict) and result[name].get("ok") is True
                  and result[name].get("version") == version else "FAIL", name + " (анализатор)",
                  "Импорт и версия должны соответствовать analyzer/requirements.txt; окружение не изменялось.")
            for name, version in pins.items()]


def check_analyzer(root: Path, mode: str) -> list[Check]:
    level = "FAIL" if mode == "real" else "WARN"
    if (root / "analyzer" / "service.py").is_file():
        checks = [Check("OK", "Модуль анализатора", "Файл analyzer/service.py найден; тяжёлый модуль не импортировался.")]
    else:
        checks = [Check(level, "Модуль анализатора", "analyzer/service.py отсутствует. Реальная интеграция пока не готова.")]
    try:
        server = pinned_versions(root / "backend" / "requirements.txt")
        analyzer = pinned_versions(root / "analyzer" / "requirements.txt")
        if not server.get("pillow") or not analyzer.get("pillow") or server["pillow"] != analyzer["pillow"]:
            raise ValueError("Pillow version pins differ")
    except Exception:
        checks.append(Check(level, "Совместимость Pillow",
                            "Требования backend и analyzer к Pillow не согласованы или недоступны. Не устанавливайте их поверх друг друга."))
    else:
        checks.append(Check("OK", "Совместимость Pillow", "В обоих списках закреплена одинаковая версия Pillow."))
    if mode == "real":
        checks.extend(check_analyzer_dependencies(root))
        checks.extend(check_model_health(port) for port in (8080, 8081))
        checks.append(check_analyzer_runtime(root))
    return checks


def run_checks(*, mode: str, port: int, data_dir: Path, skip_port: bool = False,
               root: Path = ROOT) -> list[Check]:
    return [check_python(), check_environment(mode, data_dir), *check_dependencies(root),
            check_assets(root), check_storage(data_dir), check_port(port, skip_port),
            *check_analyzer(root, mode)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Диагностика локального приложения без установки и запуска модели.")
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backend" / ".local" / "demo")
    parser.add_argument("--skip-port", action="store_true")
    options = parser.parse_args(argv)
    checks = run_checks(mode=options.mode, port=options.port, data_dir=options.data_dir, skip_port=options.skip_port)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    print(f"Итог: ошибок — {failures}, предупреждений — {warnings}.")
    print("Диагностика не проверяет точность модели и не подтверждает готовность к оценке жюри.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
