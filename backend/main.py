"""Stage 2: authenticated asynchronous video API; no frontend routes yet."""
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
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from .config import Settings
from .service import JobService
from .validation import InvalidVideo, validate_video

log = logging.getLogger(__name__)


class ApiGuards:
    """Authenticate before parsing uploads and bound bodies without Content-Length."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        service = getattr(scope['app'].state, 'service', None)
        protected = scope['path'].startswith('/api/v1/')
        response_started = False
        received = 0
        exceeded = False
        replacement_sent = False
        limit = self.settings.max_upload_bytes + 128 * 1024

        async def reject(status, detail):
            response = JSONResponse({'detail': detail}, status_code=status,
                                    headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
            await response(scope, receive, send)

        if protected:
            if service is None:
                return await reject(503, 'Сервер запускается.')
            supplied = headers.get('x-api-key', '')
            if not hmac.compare_digest(supplied.encode(), service.api_key.encode()):
                return await reject(401, 'Неверный или отсутствующий X-API-Key.')
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
            if scope['path'] == '/api/v1/analyze' and not service.ready:
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
                    ]})
                    await send({'type': 'http.response.body', 'body': body})
                return
            if message['type'] == 'http.response.start':
                response_started = True
                message['headers'] = list(message.get('headers', [])) + [
                    (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
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
            yield
        finally:
            await asyncio.to_thread(service.stop)

    application = FastAPI(title='Forte Cash Audit API', version='0.2.0', lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.state.settings = settings
    application.add_middleware(ApiGuards, settings=settings)

    def service() -> JobService:
        return application.state.service

    @application.get('/')
    async def index():
        return {'service': 'Forte Cash Audit API', 'stage': 2, 'mode': settings.mode,
                'health': '/health', 'message': 'Сервер API. Веб-интерфейс появится на следующем этапе.'}

    @application.get('/health')
    async def health():
        worker = service()
        return {'status': 'ok' if worker.ready else 'degraded', 'mode': settings.mode,
                'analyzer_ready': worker.ready, 'message': worker.readiness_message}

    @application.post('/api/v1/analyze')
    async def analyze(request: Request, file: UploadFile = File(...),
                      x_upload_id: str | None = Header(None)):
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
                previous = worker.store.by_upload_id('api', upload_id)
                if previous:
                    return {'job_id': previous['job_id']}
            name = display_filename(file.filename)
            if Path(name).suffix.lower() != '.mov':
                raise HTTPException(415, 'Выберите видео в формате .mov.')
            ip = request.client.host if request.client else 'unknown'
            job_id = worker.reserve('api', upload_id, 'ip:' + ip)
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
            worker.store.create({'job_id': job_id, 'owner': 'api', 'file_name': name,
                                 'file_size': size, 'video_path': str(path.resolve()),
                                 'created_at': processing_started, 'scenario': settings.mock_scenario,
                                 'upload_id': upload_id})
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
                worker.release(job_id, 'api', upload_id)
            if not created and directory and directory.is_dir():
                resolved = directory.resolve()
                if resolved.parent == worker.upload_dir.resolve() and resolved.name == job_id:
                    shutil.rmtree(resolved)

    @application.get('/api/v1/jobs/{job_id}')
    async def status(job_id: str):
        job = service().store.get(job_id)
        if not job or job['owner'] != 'api':
            raise HTTPException(404, 'Задание не найдено или срок хранения истёк.')
        return service().public_result(job)

    return application


app = create_app()
