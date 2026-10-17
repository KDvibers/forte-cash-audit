"""Check the application with synthetic media and an isolated localhost server."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / 'backend' / '.local' / 'selfcheck'


class CheckFailed(RuntimeError):
    """A safe diagnostic message, without response bodies or credentials."""


@contextmanager
def isolated_settings():
    """Never test against inherited production data or load a real analyzer."""
    saved = {key: value for key, value in os.environ.items() if key.startswith('CASH_AUDIT_')}
    for key in saved:
        del os.environ[key]
    os.environ['CASH_AUDIT_MODE'] = 'mock'
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith('CASH_AUDIT_'):
                del os.environ[key]
        os.environ.update(saved)


@contextmanager
def workspace():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    parent = RUN_ROOT.resolve()
    directory = Path(tempfile.mkdtemp(prefix='run-', dir=parent)).resolve()
    try:
        yield directory
    finally:
        # Verify the exact resolved target before recursive cleanup on Windows.
        resolved = directory.resolve()
        if not directory.is_symlink() and resolved.parent == parent and resolved.name.startswith('run-'):
            try:
                shutil.rmtree(resolved)
            except OSError:
                print('[WARN] Временные файлы самопроверки остались в backend/.local/selfcheck; они исключены из Git.')
        else:
            print('[WARN] Каталог самопроверки изменился. Автоматическая очистка не выполнена.')


def find_node(explicit: str | None) -> str:
    candidate = explicit or shutil.which('node')
    if not candidate:
        raise CheckFailed('Не найден Node.js 22+. Установите его или передайте -NodePath в scripts/check.ps1.')
    try:
        result = subprocess.run([candidate, '--version'], capture_output=True, text=True, timeout=10,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CheckFailed('Не удалось запустить Node.js. Проверьте -NodePath.') from exc
    match = re.fullmatch(r'v(\d+)\.\d+\.\d+\s*', result.stdout)
    if result.returncode != 0 or not match or int(match.group(1)) < 22:
        raise CheckFailed('Для полной самопроверки требуется Node.js 22 или новее. Для самого сайта Node.js не нужен.')
    return str(candidate)


def run_command(arguments: list[str], timeout: float) -> bool:
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    try:
        result = subprocess.run(arguments, cwd=ROOT, env=env, timeout=timeout, check=False,
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        # A detached Windows child cannot reliably inherit a PowerShell console.
        # Capturing output also keeps diagnostics readable in redirected terminals.
        if result.stdout:
            print(result.stdout, end='' if result.stdout.endswith('\n') else '\n', flush=True)
        if result.stderr:
            print(result.stderr, end='' if result.stderr.endswith('\n') else '\n', flush=True)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print('[FAIL] Проверка превысила допустимое время.')
    except OSError:
        print('[FAIL] Не удалось запустить проверку. Проверьте установленные зависимости.')
    return False


@contextmanager
def mock_server(data_dir: Path):
    """Own one reserved socket/server; leave all other listeners untouched."""
    import uvicorn
    from .config import Settings
    from .main import create_app

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server = None
    thread = None
    try:
        if os.name == 'nt':
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        application = create_app(Settings(mode='mock', data_dir=data_dir,
                                          mock_scenario='multiple', mock_delay_seconds=0.05,
                                          job_timeout_seconds=30, worker_startup_seconds=30))
        server = uvicorn.Server(uvicorn.Config(application, log_level='error', access_log=False,
                                             host='127.0.0.1', port=port, workers=1))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]},
                                  name='selfcheck-http', daemon=True)
        thread.start()
        yield f'http://127.0.0.1:{port}', application
    finally:
        try:
            if server is not None:
                server.should_exit = True
            if thread is not None:
                thread.join(timeout=25)
                if thread.is_alive():
                    server.force_exit = True
                    thread.join(timeout=5)
                if thread.is_alive():
                    # Stop only this service, including its owned analyzer child.
                    service = getattr(application.state, 'service', None)
                    if service is not None:
                        service.stop()
                    raise CheckFailed('Временный HTTP-сервер не завершился штатно. Самопроверка не считается успешной.')
        finally:
            listener.close()


def api_smoke(data_dir: Path) -> dict:
    from .check_api import NoRedirect, request_json, submit
    from .make_sample import create_sample
    from .validation import validate_result

    video = create_sample(data_dir / 'neutral.mov')
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with mock_server(data_dir / 'service') as (url, application):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                health = request_json(opener, Request(url + '/health'), timeout=2)
                if health.get('analyzer_ready') is True and health.get('mode') == 'mock':
                    break
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.1)
        else:
            raise CheckFailed('Временный тестовый сервер не стал готов. Реальная модель не запускалась.')

        for path, method, headers in (
            ('/api/v1/analyze', 'POST', {}),
            ('/api/v1/jobs/missing', 'GET', {'X-API-Key': 'invalid-selfcheck-key'}),
        ):
            try:
                request_json(opener, Request(url + path, method=method, headers=headers), timeout=5)
            except HTTPError as error:
                if error.code != 401:
                    raise CheckFailed('API не отклонил запрос без правильного ключа ожидаемым кодом 401.') from error
            else:
                raise CheckFailed('API принял запрос без правильного ключа.')

        key = application.state.service.api_key  # Memory only; never printed or passed as a CLI argument.
        started = time.monotonic()
        job_id = submit(opener, url, key, video)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = request_json(opener, Request(url + '/api/v1/jobs/' + job_id,
                                                 headers={'X-API-Key': key}), timeout=5)
            if result.get('job_id') != job_id:
                raise CheckFailed('API вернул другой идентификатор задания.')
            if result.get('status') == 'done':
                verified = validate_result(result)
                if verified['verdict'] != 'violation' or len(verified['violations']) != 2:
                    raise CheckFailed('Тестовый сценарий multiple вернул неожиданный результат.')
                return {'events': len(verified['violations']), 'seconds': time.monotonic() - started}
            if result.get('status') != 'processing':
                raise CheckFailed('Тестовое задание завершилось ошибкой или вернуло неизвестный статус.')
            time.sleep(0.1)
        raise CheckFailed('Временный API не вернул результат за отведённое время.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', help='Полный путь к node.exe, если Node.js отсутствует в PATH.')
    args = parser.parse_args()
    phases = {}
    try:
        with isolated_settings(), workspace() as run:
            print('Самопроверка приложения. Только синтетическое видео и локальный тестовый сервер.', flush=True)
            phases['Окружение'] = run_command([sys.executable, '-m', 'backend.doctor', '--mode', 'mock',
                                               '--skip-port', '--data-dir', str(run / 'probe')], 90)
            if not phases['Окружение']:
                raise CheckFailed('Исправьте ошибки диагностики перед запуском тестов.')
            print('\nПроверка backend:', flush=True)
            phases['Backend'] = run_command([sys.executable, '-m', 'pytest', 'backend/tests', '-q',
                                             '--basetemp', str(run / 'pytest')], 240)
            print('\nПроверка frontend:', flush=True)
            try:
                node = find_node(args.node)
                files = sorted(str(path) for path in (ROOT / 'frontend' / 'tests').glob('*.test.cjs'))
                if not files:
                    raise CheckFailed('Не найдены тесты frontend.')
                phases['Frontend'] = run_command([node, '--test', *files], 90)
            except CheckFailed as exc:
                phases['Frontend'] = False
                print(f'[FAIL] {exc}')
            print('\nПроверка HTTP API на временном сервере:', flush=True)
            try:
                result = api_smoke(run / 'http')
                phases['HTTP API'] = True
                print(f"[OK] Ключ проверен; загрузка → done; событий: {result['events']}; {result['seconds']:.2f} с.")
            except Exception as exc:
                phases['HTTP API'] = False
                print(f'[FAIL] {exc}' if isinstance(exc, CheckFailed) else '[FAIL] Не удалось завершить локальную проверку HTTP API.')
    except CheckFailed as exc:
        print(f'[FAIL] {exc}')
    except (OSError, ValueError):
        phases['Подготовка'] = False
        print('[FAIL] Не удалось подготовить локальное окружение самопроверки.')
    except KeyboardInterrupt:
        print('\nСамопроверка прервана; запущенный ею HTTP-сервер остановлен.')
        return 130
    print('\nИтог:')
    for phase, passed in phases.items():
        print(f"[{'OK' if passed else 'FAIL'}] {phase}")
    success = len(phases) == 4 and all(phases.values())
    print('Самопроверка пройдена.' if success else 'Самопроверка не пройдена. Исправьте отмеченные причины и повторите запуск.')
    print('Качество и скорость реального анализатора этой командой не оцениваются.')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
