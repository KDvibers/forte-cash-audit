"""Render an observation report as an offline, human-readable diagnostic page."""

import argparse
from html import escape
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import quote, urlsplit

from PIL import Image


_BASE_FEATURES = ("person_in_work_area", "cabinet_door")
_FEATURES = {
    "person_in_work_area": "Человек в рабочей зоне", "cabinet_door": "Дверца шкафа",
    "cash_on_worktop": "Банкноты на рабочем столе", "cash_in_hands": "Банкноты в руках",
    "notes_in_machine": "Банкноты в счётной машине",
}
_STATES = {
    "person_in_work_area": {"yes": "Виден в рабочей зоне", "no": "Не виден в рабочей зоне",
                            "uncertain": "Не удалось определить"},
    "cabinet_door": {"open": "Открыта", "closed": "Закрыта", "uncertain": "Не удалось определить",
                     "not_visible": "Шкаф не виден"},
    **{feature: {"yes": "Видны", "no": "Не видны", "uncertain": "Не удалось определить"}
       for feature in ("cash_on_worktop", "cash_in_hands", "notes_in_machine")},
}
_EVENT_NAMES = {
    "cash_appeared_on_worktop": "Банкноты появились на рабочем столе",
    "cash_no_longer_visible_on_worktop": "Банкноты перестали быть видны на рабочем столе",
    "notes_appeared_in_machine": "Банкноты появились в счётной машине",
    "notes_no_longer_visible_in_machine": "Банкноты перестали быть видны в счётной машине",
    "cash_visible_without_person_in_work_area": "Банкноты видны на столе без видимого человека в рабочей зоне",
}
_STATUS = {"completed": "Обработка завершена", "partial": "Частичный результат", "failed": "Ошибка обработки",
           "running": "Обработка ещё идёт — сохранён промежуточный отчёт"}
_FRAME_STATUS = {"ok": "Запросы выполнены", "partial": "Есть незавершённые или ошибочные запросы",
                 "error": "Ошибка наблюдения"}
_SUPPORT_LABELS = {
    "experimental": "Экспериментальный признак", "provisional": "Предварительный признак",
    "not_validated": "Надёжность не подтверждена", "unvalidated": "Надёжность не подтверждена",
    "not_evaluated": "Качество ещё не оценено", "unsupported": "Надёжное наблюдение пока недоступно",
    "disabled": "Признак отключён", "not_implemented": "Наблюдение пока не реализовано",
    "enabled": "Наблюдение включено", "supported": "Наблюдение доступно",
}


def _h(value):
    return escape(str(value), quote=True)


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "неизвестно"
    try:
        if not math.isfinite(value) or value < 0:
            return "неизвестно"
        return f"{value:.3f}".rstrip("0").rstrip(".") + " с"
    except (OverflowError, ValueError):
        return "неизвестно"


def _count(value):
    return str(value) if type(value) is int and value >= 0 else "неизвестно"


def _objects(report, key):
    value = report.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _model_name(report):
    for source in (report, report.get("settings"), report.get("runtime")):
        if not isinstance(source, dict):
            continue
        for key in ("model_label", "model_name", "model_id", "model_repository", "model", "model_path"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return PureWindowsPath(value).name if key == "model_path" else value
            if isinstance(value, dict):
                for field in ("name", "id", "repository"):
                    if isinstance(value.get(field), str) and value[field].strip():
                        return value[field]
    return "Не записана в отчёте"


def _activity_model_name(report):
    for source in (report, report.get("settings")):
        if isinstance(source, dict):
            value = source.get("activity_model_label")
            if isinstance(value, str) and value.strip():
                return value
    runtime = report.get("runtime_activity")
    if isinstance(runtime, dict):
        return _model_name({"runtime": runtime})
    if isinstance(runtime, str) and runtime.strip():
        return runtime
    settings = report.get("settings")
    if isinstance(settings, dict) and settings.get("activity_base_url") is not None:
        return "Не записана в отчёте"
    return None


def _image_details(root, value):
    """Return URL and real dimensions for a JPEG confined to the report directory."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        relative = PurePosixPath(value.replace("\\", "/"))
        if relative.is_absolute() or PureWindowsPath(value).drive or urlsplit(value).scheme:
            return None
        if ".." in relative.parts or any(":" in part for part in relative.parts):
            return None
        actual = (root / Path(*relative.parts)).resolve(strict=True)
        confined = actual.relative_to(root)
        if not actual.is_file() or actual.suffix.lower() not in {".jpg", ".jpeg"}:
            return None
        with actual.open("rb") as source:
            if source.read(3) != b"\xff\xd8\xff":
                return None
        with Image.open(actual) as source:
            if source.format != "JPEG":
                return None
            width, height = source.size
        return quote(confined.as_posix(), safe="/"), width, height
    except (OSError, ValueError, RuntimeError, Image.DecompressionBombError):
        return None


def _errors(value):
    if not value:
        return ""
    entries = value if isinstance(value, list) else [value]
    items = []
    for error in entries:
        if isinstance(error, dict):
            feature = error.get("feature", "")
            label = _FEATURES.get(feature, feature) if isinstance(feature, str) else str(feature)
            parts = [str(part) for part in (label, error.get("type"), error.get("message")) if part]
            text = ": ".join(parts) or json.dumps(error, ensure_ascii=False)
        else:
            text = str(error)
        items.append(f"<li>{_h(text)}</li>")
    return '<div class="errors"><strong>Технические ошибки</strong><ul>' + "".join(items) + "</ul></div>"


def _state_label(feature, state):
    if state is None:
        return "Нет наблюдения"
    return _STATES.get(feature, {}).get(str(state), str(state))


def _support_text(value):
    if isinstance(value, str):
        return _SUPPORT_LABELS.get(value, value)
    if isinstance(value, bool):
        return "Наблюдение доступно" if value else "Надёжное наблюдение пока недоступно"
    if isinstance(value, list):
        return "; ".join(_support_text(item) for item in value)
    if isinstance(value, dict):
        labels = {"status": "Статус", "reason": "Причина", "reasons": "Причины",
                  "note": "Примечание", "message": "Пояснение", "supported": "Доступность"}
        return "; ".join(f"{labels.get(key, key)}: {_support_text(item)}" for key, item in value.items())
    return "Не указано" if value is None else str(value)


def _feature_limits(record):
    """Show declared limitations without treating a configured feature as accurate."""
    notes = []
    for key in ("feature_support", "field_limits"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            entries = list(value.items())
        elif isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("feature"), str):
                    entries.append((item["feature"], {name: text for name, text in item.items() if name != "feature"}))
                else:
                    entries.append((None, item))
        else:
            entries = [(None, value)]
        for feature, detail in entries:
            label = _FEATURES.get(feature, feature) if isinstance(feature, str) else None
            text = _support_text(detail)
            note = f"{label}: {text}" if label else text
            if note and note not in notes:
                notes.append(note)
    if not notes:
        return ""
    return ('<div class="feature-limits"><strong>Ограничения наблюдений</strong><ul>'
            + ''.join(f'<li>{_h(note)}</li>' for note in notes) + '</ul></div>')


def render_review(report_path):
    """Write review.html next to report.json without modifying any source image."""
    path = Path(report_path).expanduser().resolve(strict=True)
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("The observation report must be a JSON object")
    root = path.parent
    output = root / "review.html"
    if output.is_symlink():
        raise ValueError("Refusing to overwrite a symlink instead of generated review.html")
    observations = _objects(report, "observations")
    transitions = _objects(report, "transitions")
    events = _objects(report, "observed_events")
    anchors = {}
    image_urls = {}
    image_sizes = {}
    for index, row in enumerate(observations):
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in anchors:
            raise ValueError("Observation frame_id values must be nonempty and unique")
        anchors[frame_id] = f"frame-{index}"
        details = _image_details(root, row.get("image_path"))
        image_urls[frame_id] = details[0] if details is not None else None
        if details is not None:
            image_sizes[frame_id] = details[1:]

    present_features = set(_BASE_FEATURES)
    for row in observations:
        if isinstance(row.get("states"), dict):
            present_features.update(row["states"])
    for item in transitions:
        if isinstance(item.get("feature"), str):
            present_features.add(item["feature"])
    features = {key: label for key, label in _FEATURES.items() if key in present_features}
    has_cash = any(key not in _BASE_FEATURES for key in features)

    changed_ids = set()

    def evidence_links(item):
        links = []
        evidence = item.get("evidence_frame_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        for frame_id in evidence:
            if isinstance(frame_id, str) and frame_id in anchors:
                changed_ids.add(frame_id)
                missing = " (JPEG недоступен)" if image_urls[frame_id] is None else ""
                links.append(f'<a href="#{anchors[frame_id]}">{_h(frame_id)}</a>{missing}')
            else:
                links.append(f"{_h(frame_id)} (кадр недоступен)")
        return '<br>'.join(links) or 'Нет ссылок на кадры'

    transition_rows = []
    for item in transitions:
        feature = str(item.get("feature", ""))
        transition_rows.append(
            "<tr>"
            f"<td>{_h(_FEATURES.get(feature, feature))}</td>"
            f"<td>{_h(_state_label(feature, item.get('from_state')))} → "
            f"{_h(_state_label(feature, item.get('to_state')))}</td>"
            f"<td>{_h(_seconds(item.get('interval_start_sec')))} — {_h(_seconds(item.get('interval_end_sec')))}</td>"
            f"<td>{_h(_seconds(item.get('confirmed_at_sec')))}</td>"
            f"<td class=links>{evidence_links(item)}</td></tr>"
        )
    if transition_rows:
        transition_html = (
            '<div class="table-scroll"><table><thead><tr><th>Признак</th><th>Изменение</th>'
            '<th>Интервал изменения</th><th>Подтверждено к</th><th>Кадры</th></tr></thead><tbody>'
            + "".join(transition_rows) + "</tbody></table></div>"
        )
    else:
        transition_html = '<p class="empty">Подтверждённых изменений в сохранённых наблюдениях нет. Это не означает, что в видео нет событий или нарушений.</p>'

    event_section = ""
    if "observed_events" in report:
        event_rows = []
        for item in events:
            kind = str(item.get("type", ""))
            title = item.get("title") or _EVENT_NAMES.get(kind, "Наблюдаемое событие")
            description = item.get("description")
            detail = f'<p class="event-detail">{_h(description)}</p>' if description else ""
            is_run = kind == "cash_visible_without_person_in_work_area"
            interval_label = "От первого до последнего кадра цепочки" if is_run else "Между старым и новым состоянием"
            start = item.get("start_sec", item.get("interval_start_sec")) if is_run else item.get("interval_start_sec")
            end = item.get("end_sec", item.get("interval_end_sec")) if is_run else item.get("interval_end_sec")
            count = item.get("observed_frame_count")
            count_text = f'<br><span class="muted">Кадров в цепочке: {_count(count)}</span>' if is_run else ""
            event_rows.append(
                f'<tr><td><strong>{_h(title)}</strong>{detail}</td>'
                f'<td>{_h(_seconds(start))} — {_h(_seconds(end))}'
                f'<br><span class="muted">{interval_label}</span>{count_text}</td>'
                f'<td>{_h(_seconds(item.get("confirmed_at_sec")))}</td>'
                f'<td class="links">{evidence_links(item)}</td></tr>'
            )
        event_html = (
            '<div class="table-scroll"><table><thead><tr><th>Что видно по кадрам</th>'
            '<th>Интервал наблюдения</th><th>Подтверждено к</th><th>Кадры</th></tr></thead><tbody>'
            + ''.join(event_rows) + '</tbody></table></div>'
            if event_rows else
            '<p class="empty">В сохранённых наблюдениях нет цепочек, удовлетворяющих условиям этих событий. '
            'Это не доказывает отсутствие действий между кадрами.</p>'
        )
        event_section = (
            '<section><h2>События для проверки</h2><p class="muted">Модель может ошибаться в описании кадров. '
            'Проверьте каждое событие по изображениям ниже. Здесь описаны изменения видимости '
            'банкнот и совпадения признаков на последовательных кадрах. Для каждого события можно '
            'открыть кадры и проверить описание. Время подтверждения — момент второго подходящего '
            'наблюдения или более позднего подтверждающего кадра.</p>' + event_html + '</section>'
        )

    cards = []
    for row in observations:
        frame_id = row["frame_id"]
        image_url = image_urls[frame_id]
        timestamp = _seconds(row.get("timestamp_sec"))
        if image_url:
            width, height = image_sizes[frame_id]
            image_html = (
                f'<a class="picture" href="{_h(image_url)}"><img loading="lazy" '
                f'width="{width}" height="{height}" '
                f'src="{_h(image_url)}" alt="Исходный кадр {_h(frame_id)}, {_h(timestamp)}"></a>'
            )
        else:
            image_html = '<div class="missing">JPEG недоступен внутри папки этого отчёта. Изображение не подменено.</div>'
        states = row.get("states") if isinstance(row.get("states"), dict) else {}
        state_rows = []
        for feature, label in features.items():
            state = states.get(feature)
            css_class = "unknown" if state is None or state in ("uncertain", "not_visible") else "known"
            state_rows.append(f'<div class="state"><dt>{_h(label)}</dt>'
                              f'<dd class="{css_class}">{_h(_state_label(feature, state))}</dd></div>')
        observation_status = row.get("status", "не указан")
        status_label = _FRAME_STATUS.get(str(observation_status), str(observation_status))
        badge = '<span class="badge">Связанный кадр</span>' if frame_id in changed_ids else ""
        cards.append(
            f'<article class="frame" id="{anchors[frame_id]}"><div class="frame-heading">'
            f'<h3>{_h(timestamp)}</h3>{badge}</div>{image_html}<div class="frame-body">'
            f'<p class="frame-id">{_h(frame_id)} · {_h(status_label)}</p><dl>{"".join(state_rows)}</dl>'
            f'{_feature_limits(row)}{_errors(row.get("errors"))}</div></article>'
        )

    status = str(report.get("status", "не указан"))
    selected = _count(report.get("selected_frame_count"))
    video_name = str(report.get("video_path", "Видео не указано")).replace("\\", "/").rsplit("/", 1)[-1]
    coverage = report.get("all_selected_frames_processed")
    coverage_text = (
        "Все выбранные кадры обработаны без технических ошибок."
        if coverage is True else
        "Обработка выбранных кадров неполная или содержит технические ошибки."
        if coverage is False else
        "Полнота обработки ещё не записана в отчёте."
    )
    extra_notes = []
    if report.get("sampling_capped") is True:
        extra_notes.append("Число кадров ограничено заданным пределом; шаг наблюдений мог увеличиться.")
    if report.get("budget_exceeded") is True:
        extra_notes.append("Бюджет времени исчерпан; результат может быть неполным.")
    extra_html = "".join(f"<p>{_h(note)}</p>" for note in extra_notes)
    source_limitations = report.get("limitations", [])
    if not isinstance(source_limitations, list):
        source_limitations = [source_limitations]
    source_details = (
        '<details><summary>Исходные ограничения из JSON</summary><ul>'
        + "".join(f"<li>{_h(item)}</li>" for item in source_limitations) + "</ul></details>"
        if source_limitations else ""
    )
    activity_model = _activity_model_name(report)
    main_model_label = "Модель шкафа и рабочего стола" if activity_model else "Модель человека и шкафа"
    activity_metric = (
        '<div class="metric"><small>Модель присутствия и положения пачки</small>'
        f'<strong>{_h(activity_model)}</strong></div>' if activity_model else ""
    )
    cash_policy_note = (
        '<p class="muted">Для банкнот на столе сопоставляются два ответа: видны ли банкноты '
        'и лежит ли пачка на поверхности, а не удерживается над ней. '
        'Если ответы не совпали, результат остаётся неопределённым. '
        'Совпадение ответов тоже может быть ошибочным — сверяйте их с кадрами.</p>'
        if activity_model and has_cash else ""
    )
    cash_limitations = (
        '<li>Появление банкнот в счётной машине не доказывает, что машина работала или деньги пересчитаны.</li>'
        '<li>Если банкноты перестали быть видны на столе, по этому признаку нельзя установить, куда их переместили.</li>'
        '<li>Банкноты без видимого человека означают только сочетание признаков на проверенных кадрах; '
        'это не устанавливает обстановку вне кадра или намерение человека.</li>' if has_cash or events else ""
    )
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Проверка наблюдений — {_h(video_name)}</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #172b43; background: #eef2f6; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} main {{ max-width: 1320px; margin: auto; padding: 28px 20px 60px; }}
h1 {{ font-size: clamp(25px, 4vw, 38px); line-height: 1.18; margin: 12px 0; }} h2 {{ margin-top: 32px; }}
p, li {{ line-height: 1.55; }} a {{ color: #1458a3; overflow-wrap: anywhere; }}
.eyebrow, .muted {{ color: #52667e; }} .eyebrow {{ font-size: 13px; letter-spacing: .06em; text-transform: uppercase; }}
.notice {{ background: #fff4db; border-left: 5px solid #c48b14; padding: 12px 18px; border-radius: 8px; }}
.notice p {{ margin: 6px 0; }} .metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap: 12px; margin: 20px 0; }}
.metric {{ background: white; padding: 16px; border-radius: 10px; border: 1px solid #dbe3ed; overflow-wrap: anywhere; }}
.metric small {{ display: block; color: #52667e; margin-bottom: 8px; }} .metric strong {{ font-size: 18px; }}
.table-scroll {{ overflow-x: auto; background: white; border: 1px solid #dbe3ed; border-radius: 10px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }} th, td {{ padding: 13px; text-align: left; vertical-align: top; border-bottom: 1px solid #e6ebf1; }}
th {{ background: #f5f7fa; }} td.links {{ min-width: 140px; }} .empty {{ padding: 18px; background: white; border-radius: 10px; }}
.event-detail {{ margin: 8px 0 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 18px; }}
.frame {{ background: white; border: 1px solid #dbe3ed; border-radius: 12px; overflow: hidden; scroll-margin-top: 18px; }}
.frame:target {{ outline: 4px solid #4f88c7; }} .frame-heading {{ padding: 13px 16px; display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
.frame-heading h3 {{ margin: 0; }} .badge {{ font-size: 12px; background: #e7f0fd; color: #164a86; border-radius: 20px; padding: 5px 9px; }}
.picture {{ display: block; background: #182332; }} .picture img {{ width: 100%; height: auto; display: block; }}
.frame-body {{ padding: 0 16px 16px; }} .frame-id {{ color: #52667e; font-size: 12px; overflow-wrap: anywhere; }}
dl {{ margin: 0; }} .state {{ padding: 9px 0; border-top: 1px solid #edf0f4; }} dt {{ font-size: 13px; color: #52667e; }}
dd {{ margin: 4px 0 0; font-weight: 650; overflow-wrap: anywhere; }} .unknown {{ color: #8b5e10; }}
.errors {{ background: #fff0ed; border-left: 3px solid #bd4d3d; padding: 10px 12px; margin-top: 12px; font-size: 13px; overflow-wrap: anywhere; }}
.errors ul {{ margin: 8px 0 0; padding-left: 20px; }} .missing {{ padding: 40px 20px; background: #f3f4f6; color: #626f80; }}
.feature-limits {{ background: #fff4db; border-radius: 8px; padding: 12px 16px; margin: 14px 0; overflow-wrap: anywhere; }}
.feature-limits ul {{ margin: 8px 0 0; padding-left: 20px; }}
.limitations {{ background: white; border-radius: 10px; padding: 4px 20px 18px; }} details {{ margin-top: 18px; overflow-wrap: anywhere; }}
summary {{ cursor: pointer; }} footer {{ margin-top: 30px; color: #52667e; font-size: 13px; }}
@media(max-width: 420px) {{ main {{ padding: 20px 12px; }} .grid {{ grid-template-columns: 1fr; }} }}
@media print {{ .frame {{ break-inside: avoid; }} body {{ background: white; }} }}
</style></head><body><main>
<header><div class="eyebrow">Локальная диагностика · Forte Cash Audit</div>
<h1>Наблюдения модели, не вывод о нарушении</h1><p class="muted">Видео: {_h(video_name)}</p></header>
<div class="notice"><p><strong>Технический статус не означает правильность ответа.</strong> Даже завершённый запрос может неверно описать кадр. Сверяйте подписи с изображениями.</p></div>
<section class="metrics" aria-label="Сводка запуска">
<div class="metric"><small>{_h(main_model_label)}</small><strong>{_h(_model_name(report))}</strong></div>
{activity_metric}
<div class="metric"><small>Технический статус</small><strong>{_h(_STATUS.get(status, status))}</strong></div>
<div class="metric"><small>Время анализа</small><strong>{_h(_seconds(report.get('total_seconds')))}</strong></div>
<div class="metric"><small>Сохранено наблюдений / выбрано кадров</small><strong>{len(observations)} / {_h(selected)}</strong></div>
</section>
<p>{_h(coverage_text)} Длительность видео: {_h(_seconds(report.get('duration_sec')))}.
Максимальный промежуток между выбранными кадрами: {_h(_seconds(report.get('max_sample_gap_sec')))}.</p>
{extra_html}{cash_policy_note}{_feature_limits(report)}{_errors(report.get('errors'))}
<section><h2>Изменения состояний</h2><p class="muted">Границы показывают последний кадр старого состояния и первый кадр нового. Это интервал наблюдения, а не точная секунда действия. Ссылки ведут к сохранённым кадрам ниже.</p>{transition_html}</section>
{event_section}
<section><h2>Кадры и ответы модели</h2><p class="muted">Нажмите на изображение, чтобы открыть исходный сохранённый JPEG. «Не удалось определить» — неопределённость, а не ответ «нет».</p>
<div class="grid">{''.join(cards) or '<p class="empty">Наблюдения кадров не сохранены.</p>'}</div></section>
<section class="limitations"><h2>Как понимать этот отчёт</h2><ul>
<li>Закрытая дверца не доказывает, что шкаф заперт на ключ.</li>
<li>Человек в видимой рабочей зоне не обязательно кассир; личность и полномочия не определяются.</li>
<li>Два состояния ограничивают интервал изменения, но сами по себе не доказывают конкретное действие.</li>
{cash_limitations}
<li>Короткое действие может произойти между выбранными кадрами. Отсутствие наблюдения не доказывает пропуск процедуры.</li>
<li>Совпадение ответов на две обрезки одного кадра не является независимой проверкой или измеренной уверенностью.</li>
<li>Время анализа не включает загрузку видео, внешнюю очередь и запуск модели. Оно не подтверждает соблюдение полного лимита кейса.</li>
</ul>{source_details}</section>
<footer>Диагностическая страница из локального JSON. Она не заменяет интерфейс продукта и не выдаёт итоговый вердикт. Изображения доступны только пока рядом сохранена папка этого запуска.</footer>
</main></body></html>"""
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to an observe_video report.json")
    args = parser.parse_args()
    try:
        output = render_review(args.report)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Не удалось создать страницу: {error}\n")
    print(f"Локальная страница: {output}")


if __name__ == "__main__":
    main()
