"""Asynchronous video API and a same-origin, anonymous browser interface."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import Headers

from .config import Settings
from .browser_sessions import BrowserSessions, COOKIE_NAME, SESSION_TTL_SECONDS
from .service import JobService
from .validation import InvalidVideo, validate_video

log = logging.getLogger(__name__)


class ApiGuards:
    """Authenticate before parsing uploads and bound bodies without Content-Length."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings
        self.public_authority = (settings.public_origin.removeprefix('https://')
                                 if settings.public_origin else None)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        service = getattr(scope['app'].state, 'service', None)
        protected = scope['path'].startswith('/api/v1/')
        browser = scope['path'] == '/ui' or scope['path'].startswith('/ui/')
        response_started = False
        received = 0
        exceeded = False
        replacement_sent = False
        limit = self.settings.max_upload_bytes + 128 * 1024

        async def reject(status, detail):
            response = JSONResponse({'detail': detail}, status_code=status,
                                    headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                                             'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer'})
            await response(scope, receive, send)

        if self.public_authority:
            hosts = headers.getlist('host')
            if len(hosts) != 1 or hosts[0].lower() != self.public_authority:
                return await reject(400, 'Адрес сервера не совпадает с настроенным публичным адресом.')
            # The launcher trusts proxy headers only from the local tunnel process.
            # Reading X-Forwarded-Proto here would also trust arbitrary clients.
            if scope.get('scheme') != 'https':
                return await reject(400, 'Для публичного доступа требуется HTTPS.')
        if protected:
            if service is None:
                return await reject(503, 'Сервер запускается.')
            supplied = headers.get('x-api-key', '')
            if not hmac.compare_digest(supplied.encode(), service.api_key.encode()):
                return await reject(401, 'Неверный или отсутствующий X-API-Key.')
        if browser:
            if headers.get('sec-fetch-site', '').lower() == 'cross-site':
                return await reject(403, 'Откройте интерфейс на адресе этого сервера.')
            if self.settings.public_origin and 'origin' in headers:
                origins = headers.getlist('origin')
                if origins != [self.settings.public_origin]:
                    return await reject(403, 'Откройте интерфейс на настроенном публичном адресе.')
            sessions = getattr(scope['app'].state, 'browser_sessions', None)
            if service is None or sessions is None:
                return await reject(503, 'Сервер запускается.')
            request = Request(scope)
            if not (scope['path'] == '/ui/session' and scope['method'] == 'GET'):
                session = sessions.get(request.cookies.get(COOKIE_NAME))
                if session is None:
                    return await reject(401, 'Сессия истекла. Обновите подключение к серверу.')
                scope.setdefault('state', {})['browser_session'] = session
                if scope['method'] == 'POST':
                    origin = self.settings.public_origin or f'{request.url.scheme}://{request.url.netloc}'
                    if headers.getlist('origin') != [origin]:
                        return await reject(403, 'Загрузка разрешена только из интерфейса этого сервера.')
                    supplied = headers.get('x-csrf-token', '')
                    if not hmac.compare_digest(supplied.encode(), session['csrf_token'].encode()):
                        return await reject(403, 'Проверка сессии не пройдена. Обновите подключение к серверу.')
        if scope['method'] == 'POST':
            value = headers.get('content-length')
            if value is not None:
                try:
                    length = int(value)
                    if length < 0:
                        raise ValueError
                except ValueError:
                    return await reject(400, 'Некорректный Content-Length.')
                if length > limit:
                    return await reject(413, 'Файл превышает допустимый размер.')
            if scope['path'] in {'/api/v1/analyze', '/ui/analyze'} and not service.ready:
                return await reject(503, service.readiness_message)

        async def bounded_receive():
            nonlocal received, exceeded
            message = await receive()
            if scope['method'] == 'POST':
                received += len(message.get('body', b''))
                if received > limit:
                    exceeded = True
                    raise HTTPException(413, 'Файл превышает допустимый размер.')
            return message

        async def guarded_send(message):
            nonlocal response_started, replacement_sent
            if exceeded:
                if not replacement_sent:
                    replacement_sent = True
                    body = json.dumps({'detail': 'Файл превышает допустимый размер.'}, ensure_ascii=False).encode()
                    await send({'type': 'http.response.start', 'status': 413, 'headers': [
                        (b'content-type', b'application/json; charset=utf-8'),
                        (b'content-length', str(len(body)).encode()), (b'cache-control', b'no-store'),
                        (b'x-content-type-options', b'nosniff'),
                        (b'x-frame-options', b'DENY'), (b'referrer-policy', b'no-referrer'),
                    ]})
                    await send({'type': 'http.response.body', 'body': body})
                return
            if message['type'] == 'http.response.start':
                response_started = True
                message['headers'] = list(message.get('headers', [])) + [
                    (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                    (b'x-frame-options', b'DENY'), (b'referrer-policy', b'no-referrer'),
                    (b'x-cash-audit-mode', self.settings.mode.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except HTTPException:
            if exceeded and not response_started and not replacement_sent:
                await reject(413, 'Файл превышает допустимый размер.')
            else:
                raise


def display_filename(value: str | None) -> str:
    name = (value or 'video.mov').replace('\\', '/').split('/')[-1]
    return ''.join(character for character in name if character.isprintable()).strip()[:200] or 'video.mov'


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application):
        service = JobService(settings)
        application.state.service = service
        await asyncio.to_thread(service.start)
        try:
            application.state.browser_sessions = BrowserSessions(service.store)
            yield
        finally:
            await asyncio.to_thread(service.stop)

    application = FastAPI(title='Forte Cash Audit API', version='0.4.0', lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.state.settings = settings
    application.add_middleware(ApiGuards, settings=settings)

    def service() -> JobService:
        return application.state.service

    frontend = Path(__file__).resolve().parent.parent / 'frontend'

    @application.get('/')
    async def index():
        return FileResponse(frontend / 'index.html', media_type='text/html')

    @application.get('/assets/{name}')
    async def frontend_asset(name: str):
        media_types = {'styles.css': 'text/css', 'app.js': 'text/javascript',
                       'results.css': 'text/css', 'results.js': 'text/javascript',
                       'recovery.js': 'text/javascript'}
        if name not in media_types:
            raise HTTPException(404, 'Файл не найден.')
        return FileResponse(frontend / name, media_type=media_types[name])

    @application.get('/health')
    async def health():
        worker = service()
        return {'status': 'ok' if worker.ready else 'degraded', 'mode': settings.mode,
                'analyzer_ready': worker.ready, 'message': worker.readiness_message}

    @application.get('/ui/session')
    async def browser_session(request: Request):
        worker = service()
        token, session = application.state.browser_sessions.open(request.cookies.get(COOKIE_NAME))
        response = JSONResponse({
            'csrf_token': session['csrf_token'], 'mode': settings.mode,
            'analyzer_ready': worker.ready, 'message': worker.readiness_message,
            'max_upload_bytes': settings.max_upload_bytes, 'poll_interval_ms': 2000,
            'job_timeout_seconds': settings.job_timeout_seconds,
        })
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_TTL_SECONDS, path='/ui',
                            httponly=True, secure=bool(settings.public_origin) or request.url.scheme == 'https',
                            samesite='strict')
        return response

    async def accept_upload(request: Request, file: UploadFile, x_upload_id: str | None, owner: str):
        worker = service()
        job_id = None
        directory = None
        created = False
        upload_id = None
        try:
            if x_upload_id:
                try:
                    upload_id = str(uuid.UUID(x_upload_id))
                except (ValueError, AttributeError):
                    raise HTTPException(422, 'X-Upload-ID должен быть UUID.')
                previous = worker.store.by_upload_id(owner, upload_id)
                if previous:
                    return {'job_id': previous['job_id']}
            name = display_filename(file.filename)
            if Path(name).suffix.lower() != '.mov':
                raise HTTPException(415, 'Выберите видео в формате .mov.')
            ip = request.client.host if request.client else 'unknown'
            job_id = worker.reserve(owner, upload_id, 'ip:' + ip)
            processing_started = time.time()
            directory = worker.upload_dir / job_id
            directory.mkdir()
            path = directory / 'source.mov'
            size = 0
            with path.open('xb') as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, 'Файл превышает допустимый размер.')
                    handle.write(chunk)
            if not size:
                raise HTTPException(422, 'Выбранный файл пуст.')
            try:
                await asyncio.to_thread(validate_video, path, settings.ffprobe_path)
            except InvalidVideo as exc:
                raise HTTPException(422, str(exc)) from exc
            if not worker.ready:
                raise HTTPException(503, 'Исполнитель недоступен. Повторите после восстановления сервера.')
            worker.store.create({'job_id': job_id, 'owner': owner, 'file_name': name,
                                 'file_size': size, 'video_path': str(path.resolve()),
                                 'created_at': processing_started, 'scenario': settings.mock_scenario,
                                 'upload_id': upload_id, 'mode': settings.mode})
            created = True
            if time.time() - processing_started >= settings.job_timeout_seconds:
                worker.store.finish(job_id, error='timeout')
            return {'job_id': job_id}
        except OSError as exc:
            log.error('Upload storage failure (%s)', type(exc).__name__)
            raise HTTPException(503, 'Не удалось сохранить видео. Проверьте свободное место на сервере.') from exc
        finally:
            await file.close()
            if job_id:
                worker.release(job_id, owner, upload_id)
            if not created and directory and directory.is_dir():
                resolved = directory.resolve()
                if resolved.parent == worker.upload_dir.resolve() and resolved.name == job_id:
                    shutil.rmtree(resolved)

    @application.post('/api/v1/analyze')
    async def analyze(request: Request, file: UploadFile = File(...),
                      x_upload_id: str | None = Header(None)):
        return await accept_upload(request, file, x_upload_id, 'api')

    @application.post('/ui/analyze')
    async def browser_analyze(request: Request, file: UploadFile = File(...),
                              x_upload_id: str | None = Header(None)):
        return await accept_upload(request, file, x_upload_id,
                                   BrowserSessions.owner(request.state.browser_session))

    def owned_job(job_id: str, owner: str):
        job = service().store.get(job_id)
        if not job or job['owner'] != owner:
            raise HTTPException(404, 'Задание не найдено или срок хранения истёк.')
        return job

    @application.get('/api/v1/jobs/{job_id}')
    async def status(job_id: str):
        return service().public_result(owned_job(job_id, 'api'))

    @application.get('/ui/jobs/{job_id}')
    async def browser_status(job_id: str, request: Request):
        job = owned_job(job_id, BrowserSessions.owner(request.state.browser_session))
        return JSONResponse(service().public_result(job), headers={
            'X-Cash-Audit-Job-Mode': job['mode'] or 'unknown',
        })

    @application.get('/ui/recovery/{upload_id}')
    async def browser_recovery(upload_id: str, request: Request):
        # Read-only lookup also recovers a lost POST acknowledgement, without reuploading.
        try:
            upload_id = str(uuid.UUID(upload_id))
        except ValueError:
            raise HTTPException(404, 'Проверка не найдена в текущей сессии.')
        owner = BrowserSessions.owner(request.state.browser_session)
        job = service().store.by_upload_id(owner, upload_id)
        if not job:
            raise HTTPException(404, 'Загрузка ещё не зарегистрирована или срок хранения проверки истёк.')
        return {'job_id': job['job_id'], 'status': job['status'], 'mode': job['mode'],
                'file_name': job['file_name'], 'file_size': job['file_size'],
                'created_at': job['created_at']}

    return application


app = create_app()
