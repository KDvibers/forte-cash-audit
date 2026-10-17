# Forte Cash Audit

Local video analysis service for the ForteBank hackathon.

## Предпросмотр интерфейса — этап 3

Из корня репозитория, в PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\frontend\preview.ps1
```

Откройте <http://127.0.0.1:4173>. Доступны выбор MOV, перетаскивание,
проверка имени и размера, замена файла и справка. Видео остаётся в браузере;
подключение к анализатору выполняется на следующем этапе.

- [Интерфейс и ограничения предпросмотра](frontend/README.md).
- [Сервер, API и его отдельный запуск](backend/README.md).

Frontend не требует Node.js, установки пакетов или сборки. Для команды
предпросмотра нужен Python; остановка сервера — `Ctrl+C`.
