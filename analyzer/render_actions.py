"""Render local action candidates for human review without asserting transfers."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

from .render_review import _h, _image_details, _seconds


_STATUS = {"ok": "Запросы выполнены", "partial": "Часть запросов не завершена или содержит ошибки"}
_TITLES = {"desk": "Взаимодействие у счётчика", "cabinet": "Возможное перемещение у шкафа"}
_INTERACTION = {"yes": "Есть признак взаимодействия", "no": "Взаимодействие не наблюдается",
                "uncertain": "Не удалось определить"}
_NEAR_CABINET = {"yes": "Возможно видна пачка у шкафа", "no": "Пачка у шкафа не наблюдается",
                 "uncertain": "Не удалось определить"}
_COVERAGE_LABELS = {
    "input_frame_count": "Кадров для поиска движения", "decoded_frame_count": "Прочитано кадров",
    "candidate_group_count": "Групп движения", "candidate_window_count": "Найдено временных окон",
    "selected_window_count": "Выбрано окон", "dropped_window_count": "Пропущено окон",
    "first_frame_sec": "Первый кадр", "last_frame_sec": "Последний кадр",
    "actual_max_frame_gap_sec": "Наибольший промежуток между кадрами",
}
_REASONS = {"window_cap": "Достигнут лимит окон", "insufficient_frames": "Недостаточно кадров",
            "insufficient_context": "Недостаточно кадров до или после", "frame_gap": "Большой разрыв между кадрами",
            "action_window_cap": "Достигнут лимит проверок действий",
            "invalid_action_window": "Некорректное окно для проверки действия",
            "duplicate_action_window": "Повторное окно для проверки действия",
            "at_least_two_coarse_boxes_required": "Нужны как минимум две подтверждённые области на разных кадрах",
            "no_supported_box_cluster": "Найденные области не согласуются между кадрами",
            "coarse_box_cluster_union": "Область построена по согласующимся обнаружениям"}


def _objects(value, name):
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _label(value, labels):
    return labels.get(value, "Не удалось определить: отсутствует или некорректно значение") if isinstance(value, str) else "Не удалось определить: нет корректного значения"


def _error_list(value):
    if not value:
        return ""
    values = value if isinstance(value, list) else [value]
    items = []
    for item in values:
        if isinstance(item, dict):
            text = ": ".join(str(item[key]) for key in ("feature", "type", "message") if item.get(key))
            text = text or json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        items.append(f"<li>{_h(text)}</li>")
    return '<div class="warning"><strong>Технические ошибки</strong><ul>' + "".join(items) + "</ul></div>"


def _evidence(root, item, index):
    details = _image_details(root, item.get("image_path"))
    failure = "Кадр недоступен: разрешён только существующий JPEG внутри папки отчёта."
    digest = item.get("image_sha256")
    valid_digest = (isinstance(digest, str) and len(digest) == 64
                    and all(char in "0123456789abcdef" for char in digest.lower()))
    if details is not None:
        if not valid_digest:
            failure = "Кадр не показан: нет корректного SHA256 для проверки исходного изображения."
            details = None
        else:
            try:
                source = (root / unquote(details[0])).resolve(strict=True)
                source.relative_to(root)
                matches = hashlib.sha256(source.read_bytes()).hexdigest() == digest.lower()
            except (OSError, ValueError, RuntimeError):
                matches = False
            if not matches:
                failure = "Кадр не показан: SHA256 не совпадает с отчётом. Возможна подмена или изменение файла."
                details = None
    caption = f"Кадр {index + 1} · {_seconds(item.get('timestamp_sec'))}"
    if details is None:
        body = f'<div class="missing">{_h(failure)}</div>'
    else:
        url, width, height = details
        body = (f'<a href="{_h(url)}"><img src="{_h(url)}" width="{width}" height="{height}" '
                f'alt="{_h(caption)}" loading="lazy"></a>')
    return f'<figure>{body}<figcaption>{_h(caption)}</figcaption></figure>'


def _counts(values, labels):
    entries = []
    for key, label in labels.items():
        if key not in values:
            continue
        value = values[key]
        text = _seconds(value) if key.endswith("_sec") else str(value) if type(value) is int and value >= 0 else "неизвестно"
        entries.append(f"<dt>{_h(label)}</dt><dd>{_h(text)}</dd>")
    return "<dl>" + "".join(entries) + "</dl>" if entries else ""


def _dropped_windows(dropped, *, counter=True):
    if not isinstance(dropped, list):
        return '<p class="warning">Список пропущенных окон имеет неизвестный формат.</p>'
    parts = [f"<p>Пропущенные окна в списке: {len(dropped)}.</p>"]
    items = []
    for row in dropped:
        if not isinstance(row, dict):
            items.append(f"<li>{_h(row)}</li>")
            continue
        region = row.get("region", row.get("roi"))
        titles = {"cabinet": "Область шкафа", "desk": "Область счётчика" if counter else "Область рабочего стола"}
        title = titles.get(region, str(region)) if isinstance(region, str) else "Область не указана"
        reason = row.get("reason", "Причина не указана")
        reason = _REASONS.get(reason, reason) if isinstance(reason, str) else str(reason)
        items.append(f"<li>{_h(title)} · {_h(_seconds(row.get('start_sec')))} — "
                     f"{_h(_seconds(row.get('end_sec')))}: {_h(reason)}</li>")
    if items:
        parts.append("<ul>" + "".join(items) + "</ul>")
    return "".join(parts)


def _coverage(motion, *, counter=False):
    title = "Поиск движения у счётчика" if counter else "Поиск движения для уточнения кадров"
    section_id = "counter-motion" if counter else "frame-motion"
    parts = [f'<section class="coverage" id="{section_id}"><h2>{title}</h2>']
    if not isinstance(motion, dict):
        return "".join(parts) + '<p class="warning">Сведения о покрытии поиска движения не записаны.</p></section>'
    if not counter:
        parts.append('<p>Эти окна используются для уточнения кадров. Их число не равно числу проверок действий: '
                     'план проверки действий показан отдельно выше.</p>')
    coverage = motion.get("coverage")
    if isinstance(coverage, dict):
        parts.append(_counts(coverage, _COVERAGE_LABELS) or "<p>Подробные численные сведения о покрытии не записаны.</p>")
        for key, note in (
            ("capped", "Лимит окон достигнут: часть найденных участков могла остаться непроверенной."),
            ("refinement_capped", "Лимит дополнительных кадров достигнут: не все участки проверены одинаково подробно."),
            ("image_decode_skipped_no_roi", "Область поиска не определена; поиск движения по кадрам был пропущен."),
        ):
            if coverage.get(key) is True:
                parts.append(f'<p class="warning">{_h(note)}</p>')
    else:
        parts.append("<p>Сведения о покрытии не записаны в ожидаемом формате.</p>")
    if counter:
        regions = motion.get("rois")
        roi = regions.get("desk") if isinstance(regions, dict) else None
        roi = roi if isinstance(roi, dict) else {}
        box = roi.get("bbox_normalized")
        valid_box = (isinstance(box, list) and len(box) == 4
                     and all(type(v) is int and 0 <= v <= 1000 for v in box)
                     and box[2] > box[0] and box[3] > box[1])
        if valid_box:
            parts.append("<p>Согласованная область счётчика найдена для поиска движения. Это не подтверждает действие.</p>")
        else:
            parts.append('<p class="warning">Область счётчика для поиска движения не найдена. '
                         'Это пробел покрытия, а не отсутствие действий у счётчика.</p>')
        reason = roi.get("reason")
        if isinstance(reason, str):
            parts.append(f"<p>{_h(_REASONS.get(reason, reason))}.</p>")
        parts.append(_counts(roi, {"valid_box_count": "Обнаружений с корректными координатами",
                                  "accepted_box_count": "Согласующихся обнаружений",
                                  "outlier_box_count": "Отброшено несогласующихся областей",
                                  "invalid_box_count": "Областей с некорректными координатами",
                                  "missing_box_count": "Наблюдений без области"}))
        if isinstance(coverage, dict):
            for key, label in (("compared_pairs_by_roi", "Сравнено пар кадров у счётчика"),
                               ("skipped_pairs_by_roi", "Пропущено пар кадров у счётчика")):
                pairs = coverage.get(key)
                if isinstance(pairs, dict):
                    parts.append(_counts(pairs, {"desk": label}))
    parts.append(_dropped_windows(motion.get("dropped_windows", []), counter=counter))
    parts.append(_error_list(motion.get("frame_errors")))
    parts.append("<p>Движение может быть вызвано человеком, отражением или освещением. Действия между выбранными кадрами "
                 "и вне найденных областей могут быть пропущены. Отсутствие кандидатов не доказывает отсутствие действия.</p></section>")
    return "".join(parts)


def _action_selection(selection, checks):
    parts = ['<section class="coverage" id="action-selection"><h2>Выбор окон для проверки действий</h2>']
    parts.append(f"<p>Записано результатов проверки: {len(checks)}.</p>")
    if not isinstance(selection, dict):
        parts.append('<p>План выбора окон не записан. Поиск движения не заменяет эти сведения.</p>')
    else:
        parts.append(_counts(selection, {"candidate_count": "Кандидатов перед ограничением",
                                         "selected_count": "Выбрано для проверки действий"}))
        if selection.get("capped") is True:
            parts.append('<p class="warning">Лимит проверок действий достигнут: часть окон не выбрана.</p>')
        elif selection.get("capped") is False:
            parts.append('<p>Лимит выбора окон для проверки действий не достигнут.</p>')
        else:
            parts.append('<p>Не записано, ограничен ли выбор окон для проверки действий.</p>')
        selected = selection.get("selected_count")
        if type(selected) is int and selected >= 0 and len(checks) < selected:
            parts.append('<p class="warning">Не все выбранные окна имеют результат. '
                         'Обработка могла ещё идти или завершиться по ограничению времени.</p>')
        if selection.get("state_conditioned") is True:
            parts.append('<p>Открытая дверца помогла выбрать окна для просмотра; это не доказательство переноса.</p>')
        parts.append(_dropped_windows(selection.get("dropped_windows", [])))
    parts.append('<p>План объединяет окна у шкафа и отдельно найденного счётчика. '
                 'Число выбранных окон не означает число обнаруженных действий.</p></section>')
    return "".join(parts)


def _counter_discovery(discovery):
    parts = ['<section class="coverage" id="counter-discovery"><h2>Поиск и проверка счётчика</h2>']
    if not isinstance(discovery, dict):
        return "".join(parts) + '<p>Отдельный поиск счётчика не записан. Отсутствие этих сведений не означает отсутствие действий.</p></section>'
    coverage = discovery.get("coverage")
    if isinstance(coverage, dict):
        parts.append(_counts(coverage, {"coarse_frame_count": "Кадров предварительной выборки",
                                       "unique_coarse_frame_count": "Уникальных кадров предварительной выборки",
                                       "selected_frame_count": "Выбрано кадров для поиска счётчика",
                                       "attempted_frame_count": "Начато проверок кадров",
                                       "completed_frame_count": "Завершено проверок кадров",
                                       "validated_frame_count": "Кадров с подтверждённым типом устройства",
                                       "unvalidated_frame_count": "Проверок без подтверждённого типа устройства",
                                       "skipped_frame_count": "Кадров не проверено"}))
        if coverage.get("budget_exhausted") is True:
            parts.append('<p class="warning">Время поиска счётчика закончилось; часть кадров могла остаться непроверенной.</p>')
    status = discovery.get("status")
    parts.append(f'<p class="technical">{_h(_label(status, _STATUS))}. '
                 'Технический статус не подтверждает правильность найденного устройства.</p>')
    checks = discovery.get("checks", [])
    if isinstance(checks, list):
        items = []
        for check in checks:
            if not isinstance(check, dict):
                continue
            state = check.get("banknote_counter")
            if state == "yes":
                result = "Тип устройства подтверждён моделью; согласованность области между кадрами проверяется отдельно"
            elif state == "no":
                result = "Кандидат отклонён: модель не подтвердила, что это счётчик"
            elif check.get("object_visibility") in ("no", "uncertain"):
                result = "Кандидат не найден или его положение неясно; действие не оценивалось"
            else:
                result = "Тип устройства не установлен; действие не оценивалось"
            items.append(f"<li>{_h(_seconds(check.get('timestamp_sec')))}: {_h(result)}.</li>")
        if items:
            parts.append("<ul>" + "".join(items) + "</ul>")
    parts.append(_error_list(discovery.get("errors")))
    parts.append('<p>Отсутствующий или отклонённый объект не означает отсутствие действий. '
                 'Для поиска движения нужна согласованная область счётчика на нескольких кадрах.</p></section>')
    return "".join(parts)


def render_actions(report_path):
    """Write actions.html next to a report; never replace or create evidence JPEGs."""
    path = Path(report_path).expanduser().resolve(strict=True)
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("The action report must be a JSON object")
    checks = _objects(report.get("action_checks", []), "action_checks")
    root = path.parent
    output = root / "actions.html"
    if output == path or output.is_symlink() or (output.exists() and output.stat().st_nlink > 1):
        raise ValueError("Refusing to overwrite an input file or linked output")
    cards = []
    for index, row in enumerate(checks):
        evidence = _objects(row.get("evidence", []), "evidence")
        result = row.get("result")
        result = result if isinstance(result, dict) else {}
        region = row.get("region")
        title = _TITLES.get(region, "Область не определена") if isinstance(region, str) else "Область не определена"
        status = row.get("status")
        status_text = _STATUS.get(status, "Неизвестный технический статус") if isinstance(status, str) else "Технический статус отсутствует"
        cards.append(f'<section class="window" id="window-{index}"><h2>{_h(title)}</h2>'
                     f'<p>Кандидат {_h(row.get("window_id", index + 1))} · {_h(_seconds(row.get("start_sec")))} — '
                     f'{_h(_seconds(row.get("end_sec")))}</p><p class="technical">{_h(status_text)}. '
                     'Технический статус не означает, что наблюдение верно.</p>')
        if region == "desk":
            cards.append(f'<p><strong>Наблюдение:</strong> {_h(_label(result.get("interaction"), _INTERACTION))}.</p>'
                         '<p class="limit">Направление загрузки или извлечения не подтверждено. Пересчёт денег не установлен.</p>')
        elif region == "cabinet":
            cards.append(f'<p><strong>Наблюдение:</strong> {_h(_label(result.get("cash_near_cabinet"), _NEAR_CABINET))}.</p>'
                         '<p class="limit">Прямой перенос банкнот внутрь шкафа не подтверждён. Исчезновение за дверцей не доказывает помещение внутрь.</p>')
            if row.get("cabinet_state_gate") == "no_observed_open_in_window":
                cards.append('<p class="warning">Первоначальный ответ модели не принят: '
                             'в этом участке нет допустимого наблюдения открытой дверцы. '
                             'Сохраняем неопределённость.</p>')
        else:
            cards.append('<p class="warning">Неизвестная область: вывод о действии не показан.</p>')
        if len(evidence) != 3:
            cards.append(f'<p class="warning">Ожидалось три кадра; в отчёте указано {len(evidence)}. '
                         'Показаны доступные первые три, недостающие кадры не заменяются.</p>')
        cards.append('<div class="evidence">' + "".join(_evidence(root, item, i) for i, item in enumerate(evidence[:3])) + '</div>')
        cards.append(_error_list(row.get("errors")))
        cards.append('</section>')
    body = "".join(cards) if cards else '<p class="warning">Кандидатов для просмотра нет. Это не доказательство отсутствия действий.</p>'
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Просмотр возможных действий</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#17212d}main{max-width:1180px;margin:auto;padding:24px}
h1,h2{line-height:1.2}h2{font-size:21px}.intro,.coverage,.window{background:white;border:1px solid #d3dbe3;border-radius:10px;padding:20px;margin:18px 0}
.warning,.limit{background:#fff4d9;padding:12px;border-left:4px solid #bd7d12}.technical{color:#465565}.evidence{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
figure{margin:0;min-width:0}img{display:block;max-width:100%;height:auto;border:1px solid #c8d1dc}figcaption{padding:8px 0}.missing{padding:24px;background:#edf0f3;min-height:120px}
dl{display:grid;grid-template-columns:minmax(180px,1fr) 1fr;gap:6px 20px}dt{font-weight:600}dd{margin:0}li{overflow-wrap:anywhere}p{overflow-wrap:anywhere}
@media(max-width:700px){.evidence{grid-template-columns:1fr}main{padding:12px}dl{grid-template-columns:1fr}}
</style></head><body><main><h1>Просмотр возможных действий</h1>
<div class="intro"><strong>Кандидаты для ручной проверки.</strong><p>Наблюдения модели, не вывод о нарушении. Неизвестность не означает отсутствие действия.</p>
<p>Здесь показаны только реальные JPEG из папки отчёта с проверкой SHA256. Сырые ответы модели не используются как доказанные факты.</p>
<p>Закрыто не означает заперто. Человек в кадре не устанавливает его роль. Интервал между кадрами не является точной секундой действия.</p></div>'''
    page += f'<p>Источник: {_h(path.name)}. Окон в отчёте: {len(checks)}.</p>'
    overall_status = {"completed": "Выбранные проверки завершены", "partial": "Частичный результат",
                      "failed": "Ошибка обработки", "running": "Обработка ещё идёт"}
    page += f'<p><strong>Общий проход: {_h(overall_status.get(str(report.get("status")), "Статус не записан"))}.</strong></p>'
    budget_control = report.get("budget_control")
    if isinstance(budget_control, dict) and budget_control.get("enabled") is True:
        skipped = budget_control.get("skipped_refinement_indices", [])
        skipped_count = len(skipped) if isinstance(skipped, list) else "неизвестно"
        page += (f'<p class="warning">Для действий оставляли резерв {_h(_seconds(budget_control.get("action_reserve_sec")))}. '
                 f'Уточняющих кадров пропущено: {_h(skipped_count)}. '
                 'Резерв не гарантирует завершения всех действий. Пропущенный кадр не является отрицательным наблюдением.</p>')
    page += _error_list(report.get("errors"))
    page += (_action_selection(report.get("action_selection"), checks)
             + _counter_discovery(report.get("counter_discovery"))
             + _coverage(report.get("counter_motion"), counter=True)
             + _coverage(report.get("motion")) + body + '</main></body></html>')
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        output = render_actions(args.report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
