# Рабочие компоненты и лицензии

Состояние на 18.10.2026. Версии Python-пакетов сверены с установленной `.venv-app` и прямыми зависимостями в [requirements-real.txt](../backend/requirements-real.txt), [backend/requirements.txt](../backend/requirements.txt) и [analyzer/requirements.txt](../analyzer/requirements.txt). Лицензии проверены по первичным источникам проектов. Это перечень рабочего профиля, а не полный юридический аудит всех транзитивных зависимостей, кодеков и системных библиотек.

Репозиторий не поставляет веса моделей, CUDA/llama.cpp, FFmpeg или окружение Python. Они скачиваются из внешних источников и устанавливаются локально; видео для этого не отправляются. Список ниже не означает, что права на все компоненты принадлежат команде.

## Модели и запуск

| Компонент | Зафиксированный вариант | Лицензия и первичный источник |
| --- | --- | --- |
| Qwen3-VL-2B-Instruct-GGUF | Ревизия `52d6c8ffea26cc873ac5ad116f8631268d7eb503`; языковая часть `Q4_K_M`, визуальный проектор `F16`; локальный порт 8080 | Apache-2.0: [карточка Qwen](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF), [метаданные выбранной ревизии](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/raw/52d6c8ffea26cc873ac5ad116f8631268d7eb503/README.md). |
| Qwen3-VL-4B-Instruct-GGUF | Ревизия `1cd86afb9a95c410a6038ab3b40d8b578c892266`; `Q4_K_M` и проектор `F16`; локальный порт 8081 | Apache-2.0: [метаданные выбранной ревизии Qwen](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/raw/1cd86afb9a95c410a6038ab3b40d8b578c892266/README.md). |
| llama.cpp | `b11669`, Windows x64, CUDA-сборка 13.4; два локальных `llama-server` | MIT для кода llama.cpp: [LICENSE этого релиза](https://github.com/ggml-org/llama.cpp/blob/b11669/LICENSE). Это не лицензия всех DLL в архиве. |
| NVIDIA CUDA runtime | Архив `cudart-llama-bin-win-cuda-13.4-x64.zip` выбранного релиза | Отдельные [NVIDIA CUDA EULA и условия распространения](https://docs.nvidia.com/cuda/eula/index.html); MIT llama.cpp на них не переносится. |
| CPython | Установлен Python 3.12.14 | PSF License Version 2; включённые компоненты могут иметь другие условия: [документация Python](https://docs.python.org/3.12/license.html). |
| Tailscale / Funnel | Установлен клиент 1.104.1; внешний сетевой маршрут, не AI-модель | Открытый код Tailscale — [BSD-3-Clause](https://github.com/tailscale/tailscale/blob/v1.104.1/LICENSE). Использование сервиса отдельно регулируют [условия Tailscale](https://tailscale.com/terms); лицензия репозитория не описывает всю службу и установщик. |

Текущий производственный путь использует 2B и 4B. Исследовательские 8B и дообученные детекторы в этот перечень рабочего запуска не включены. Их наличие локально не означает использование сервисом.

## Прямые Python-зависимости

| Пакет | Установленная версия | Назначение | Лицензия и первичный источник |
| --- | --- | --- | --- |
| FastAPI | 0.143.3 | HTTP-приложение | [MIT](https://github.com/fastapi/fastapi/blob/0.143.3/LICENSE) |
| Starlette | 1.8.0 | ASGI, HTTP и сессии | [BSD-3-Clause](https://github.com/Kludex/starlette/blob/1.8.0/LICENSE.md) |
| Uvicorn | 0.54.0 | Сервер ASGI | [BSD-3-Clause](https://github.com/Kludex/uvicorn/blob/0.54.0/LICENSE.md) |
| python-multipart | 0.0.32 | Приём multipart-загрузок | [Apache-2.0](https://github.com/Kludex/python-multipart/blob/0.0.32/LICENSE.txt) |
| Pillow | 11.3.0 | Изображения и JPEG | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/11.3.0/LICENSE) |
| PyAV (`av`) | 16.0.1 | Декодирование видео и временные отметки кадров | [BSD-3-Clause](https://github.com/PyAV-Org/PyAV/blob/v16.0.1/LICENSE.txt) для PyAV; FFmpeg-библиотеки рассмотрены отдельно ниже. |
| imageio-ffmpeg | 0.6.0 | Запуск FFmpeg при проверке файлов и создании нейтральных примеров | [BSD-2-Clause](https://github.com/imageio/imageio-ffmpeg/blob/v0.6.0/LICENSE) для Python-обёртки; не для вложенного `ffmpeg.exe`. |
| pytest | 9.1.1 | Тесты; `requirements-dev.txt` | [MIT](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE) |
| HTTPX | 0.28.1 | Тесты HTTP; `requirements-dev.txt` | [BSD-3-Clause](https://github.com/encode/httpx/blob/0.28.1/LICENSE.md) |

SQLite используется через стандартный модуль Python. Авторы SQLite публикуют основную библиотеку как [public domain](https://www.sqlite.org/copyright.html). Полный состав зависимостей самого Python и Python-пакетов здесь не перечисляется.

## FFmpeg: две разные поставки

**Отдельный исполняемый файл imageio-ffmpeg:** локально проверен `7.1-essentials_build-www.gyan.dev`, с флагами `--enable-gpl`, `--enable-version3`, `--enable-libx264`. Для этой сборки применима GPL версии 3 или более поздней, а не BSD обёртки. Это следует из проверенных флагов и [правил лицензирования FFmpeg 7.1](https://github.com/FFmpeg/FFmpeg/blob/n7.1/LICENSE.md). Backend вызывает этот файл отдельным процессом.

**Библиотеки PyAV:** установленный PyAV сообщает FFmpeg 8.0 (`libavcodec 62.11.100`, `libavformat 62.3.100`). Функции `avcodec_license`, `avformat_license` и `avutil_license` локальных DLL вернули `LGPL version 3 or later`. Однако конфигурация также содержит `--enable-libx264` и `--enable-libx265`: это требует отдельной сверки состава и условий сборки. Нельзя считать весь wheel лицензированным только BSD-3-Clause или подтверждать его полную лицензионную совместимость по одной строке DLL. Связанные кодеки отдельно оговорены в [официальном описании FFmpeg](https://ffmpeg.org/legal.html).

Этот документ не определяет лицензию всего проекта из лицензии одного внешнего процесса. Если команда будет распространять готовую среду или бинарный архив, потребуется отдельно подготовить соответствующие тексты лицензий, уведомления и исходники там, где их требует конкретная лицензия. Текущий Git-репозиторий таких бинарных файлов не содержит.

## Зафиксированные файлы и SHA-256

Источник таблицы — [runtime-lock.json](../analyzer/runtime-lock.json) и [runtime-lock-4b.json](../analyzer/runtime-lock-4b.json). В локальной проверке 18.10.2026 контрольные суммы всех четырёх рабочих GGUF и обоих ZIP совпали: **6 из 6**. При подготовке этого документа значения перенесены из файлов фиксации, повторное хеширование не выполнялось. Хеш подтверждает совпадение файла, а не качество модели или выполнение лицензии.

| Файл | SHA-256 |
| --- | --- |
| `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae` |
| `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | `c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5` |
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a` |
| `mmproj-Qwen3VL-4B-Instruct-F16.gguf` | `256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331` |
| `llama-b11669-bin-win-cuda-13.4-x64.zip` | `3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc` |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` | `738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668` |

Адреса загрузки и имена файлов закреплены в двух runtime-lock. Секреты, личные адреса Funnel, пути пользовательских видео и сами веса в этот документ не включены. Порядок запуска — в [REAL_RUNBOOK.md](REAL_RUNBOOK.md), фактические измерения сервиса — в [протоколе интеграции](FUNNEL_INTEGRATION_STATUS.md).
