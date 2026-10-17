"""Render experimental direction answers and their actual local evidence."""

import argparse
import json
import math
from pathlib import Path

from .render_actions import _error_list, _evidence, _objects
from .render_review import _count, _h, _model_name, _seconds, _support_text


_STATUS = {
    "running": "Обработка ещё идёт — промежуточный отчёт",
    "completed": "Обработка завершена", "partial": "Частичный результат",
    "failed": "Ошибка обработки", "error": "Ошибка запроса",
    "ok": "Запрос выполнен", "not_started": "Не запускалось",
}
_MOTION = {
    "hand_to_input": "Модель предполагает загрузку банкнот из руки во входной лоток",
    "output_to_hand": "Модель предполагает извлечение банкнот из выходного лотка в руку",
    "no_transfer": "Модель не различила загрузку или извлечение в выбранных кадрах",
    "uncertain": "Модель не смогла определить направление",
}
_LIMITATIONS = {
    "Experimental model answers are not rule facts or violation verdicts.":
        "Экспериментальные ответы модели не являются фактами для правил или выводами о нарушении.",
    "This pass does not start, stop or switch model servers.":
        "Эта проверка сама не запускает и не переключает модели.",
    "Timing excludes previous analysis, model startup/switching, upload and queueing.":
        "Время не включает предыдущий анализ, запуск и переключение моделей, загрузку видео и очередь.",
    "Decoding and file hashing are not preemptible; deadlines are checked afterwards.":
        "Чтение видео и проверка хешей завершаются до проверки лимита времени.",
    "Source reports without a video hash cannot prove historical identity of unsaved frames.":
        "Без хеша видео в исходном отчёте нельзя подтвердить прежнее содержимое несохранённых кадров.",
    "Selection cannot recover upstream missed or uncertain interactions.":
        "Отбор не восстанавливает взаимодействия, пропущенные или неясные на предыдущем шаге.",
    "Successful requests and stable object geometry do not establish visual correctness.":
        "Успешный запрос и согласованная область счётчика не подтверждают правильность распознавания.",
}


def _status(value):
    return _STATUS.get(value, "Неизвестный технический статус") if isinstance(value, str) else "Технический статус не записан"


def _ids(value):
    return ", ".join(str(item) for item in value) if isinstance(value, list) and value else "не записаны"


def _valid_time(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _coverage(report, checks):
    parts = ['<section class="coverage" id="direction-selection"><h2>Что проверено</h2>']
    selection = report.get("direction_selection")
    if not isinstance(selection, dict):
        parts.append('<p class="warning">Сведения об отборе окон не записаны.</p>')
        selection = {}
    coverage = selection.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    policy = selection.get("policy")
    policy = policy if isinstance(policy, dict) else {}
    settings = report.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    attempted = sum(row.get("attempted") is True for row in checks)
    succeeded = sum(row.get("status") == "ok" and row.get("attempted") is True
                    and not row.get("errors") for row in checks)
    pending = sum(row.get("status") == "not_started" or row.get("attempted") is False for row in checks)
    selected = selection.get("selected_count", report.get("selected_window_count"))
    counts = [
        ("Кандидатов до лимита", selection.get("candidate_count")),
        ("Выбрано окон", selected),
        ("Лимит окон", policy.get("max_windows", settings.get("max_windows"))),
        ("Записано окон в отчёте", len(checks)), ("Начата проверка окон", attempted),
        ("Запрос выполнен без записанных ошибок", succeeded), ("Не запускалось", pending),
        ("Отброшено из-за лимита", coverage.get("cap_dropped_window_count")),
        ("Запросов к модели", report.get("request_count")),
    ]
    parts.append('<dl>' + ''.join(f'<dt>{_h(label)}</dt><dd>{_h(_count(value))}</dd>' for label, value in counts) + '</dl>')
    if selection.get("capped") is True:
        parts.append('<p class="warning">Лимит окон достигнут: часть кандидатов не выбрана для проверки.</p>')
    elif selection.get("capped") is False:
        parts.append('<p>Лимит выбора окон не достигнут. Это не означает, что найдены все действия.</p>')
    if (pending or report.get("all_selected_windows_processed") is False
            or (type(selected) is int and selected > attempted)):
        parts.append('<p class="warning">Не все выбранные окна обработаны. Незавершённая проверка не означает отсутствие действия.</p>')
    if report.get("budget_exceeded") is True:
        parts.append('<p class="warning">Лимит времени исчерпан. Доступен только сохранённый результат.</p>')
    for key, label in (("actual_max_frame_gap_sec", "Наибольший шаг исходной выборки кадров"),
                       ("selected_max_anchor_gap_sec", "Наибольший промежуток между выбранными кадрами")):
        if key in coverage:
            parts.append(f'<p>{_h(label)}: {_h(_seconds(coverage[key]))}.</p>')
    parts.append('<p>Короткое действие между кадрами может быть пропущено. '
                 'Отбор не восстанавливает пропущенные или неясные взаимодействия предыдущего шага.</p>')
    dropped = _objects(selection.get("dropped_windows", []), "direction_selection.dropped_windows")
    if dropped:
        parts.append(f'<details><summary>Не выбранные окна: {len(dropped)}</summary><ul>')
        for row in dropped:
            reason = row.get("reason", "Причина не записана")
            if reason == "direction_window_cap":
                reason = "Достигнут лимит окон направления"
            parts.append(f'<li>{_h(_ids(row.get("source_window_ids")))}: '
                         f'{_h(_seconds(row.get("start_sec")))} — {_h(_seconds(row.get("end_sec")))}; {_h(reason)}.</li>')
        parts.append('</ul></details>')
    parts.append('</section>')
    return ''.join(parts)


def _card(root, row, index):
    evidence = _objects(row.get("evidence", []), "direction_checks.evidence")
    parts = [f'<section class="window" id="direction-{index}"><h2>Окно {_h(row.get("window_id", index + 1))}</h2>',
             f'<p>Исходные окна взаимодействия: {_h(_ids(row.get("source_window_ids")))}.</p>',
             f'<p class="technical">{_h(_status(row.get("status")))}. '
             'Технический статус не подтверждает правильность ответа модели.</p>']
    if evidence:
        parts.append(f'<p>Время кадров по отчёту: {_h(_seconds(evidence[0].get("timestamp_sec")))} — '
                     f'{_h(_seconds(evidence[-1].get("timestamp_sec")))}.</p>')
        times = [item.get("timestamp_sec") for item in evidence]
        if not all(_valid_time(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            parts.append('<p class="warning">Время кадров отсутствует или нарушен хронологический порядок. '
                         'Порядок отчёта сохранён; направление по такой последовательности не подтверждается.</p>')
        elif (row.get("start_sec") != times[0] or row.get("end_sec") != times[-1]):
            parts.append('<p class="warning">Границы окна не совпадают с временем перечисленных кадров.</p>')
    parts.append('<p class="accepted"><strong>Принятый результат: направление не подтверждено (uncertain).</strong> '
                 'Экспериментальный ответ не используется как установленный факт для правил.</p>')
    result = row.get("result")
    if not isinstance(result, dict) or result.get("counter_direction") != "uncertain":
        parts.append('<p class="warning">Поле принятого результата отсутствует или не соответствует экспериментальному протоколу. '
                     'Подтверждённое направление на этой странице не объявляется.</p>')
    parts.append('<div class="raw-answer"><h3>Ответ модели — требует проверки</h3>')
    raw = row.get("raw_banknote_motion")
    label = _MOTION.get(raw) if isinstance(raw, str) else None
    if label:
        parts.append(f'<p>{_h(label)}. <code>{_h(raw)}</code></p>')
        if raw == "no_transfer":
            parts.append('<p>Это не доказывает отсутствие переноса между кадрами или за границей выбранного окна.</p>')
    elif raw is None:
        parts.append('<p>Сырой ответ не записан. Это не отрицательный ответ о действии.</p>')
    else:
        parts.append(f'<p class="warning">Неизвестный формат ответа: {_h(raw)}.</p>')
    description = row.get("model_evidence_description")
    if description:
        parts.append(f'<p class="description"><strong>Описание модели:</strong> {_h(description)}</p>')
    parts.append('</div>')
    if row.get("feature_support") is not None:
        parts.append(f'<p class="limit">{_h(_support_text(row["feature_support"]))}</p>')
    if not 2 <= len(evidence) <= 5:
        parts.append(f'<p class="warning">Ожидалось от двух до пяти кадров; в отчёте указано {len(evidence)}. '
                     'Показаны все перечисленные кадры, недостающие не заменяются.</p>')
    parts.append('<div class="evidence">' + ''.join(_evidence(root, item, i) for i, item in enumerate(evidence)) + '</div>')
    parts.append(_error_list(row.get("errors")))
    parts.append('</section>')
    return ''.join(parts)


def render_direction(report_path):
    """Write direction.html beside the report, without changing any evidence."""
    path = Path(report_path).expanduser().resolve(strict=True)
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("The direction report must be a JSON object")
    checks = _objects(report.get("direction_checks", []), "direction_checks")
    root = path.parent
    output = root / "direction.html"
    if output == path or output.is_symlink() or (output.exists() and output.stat().st_nlink > 1):
        raise ValueError("Refusing to overwrite an input file or linked output")
    body = ''.join(_card(root, row, index) for index, row in enumerate(checks))
    if not body:
        body = '<p class="warning">Нет записанных проверок направления. Это не доказательство отсутствия действий.</p>'
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка загрузки и извлечения</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#17212d}main{max-width:1240px;margin:auto;padding:24px}
h1,h2,h3{line-height:1.25}h2{font-size:22px}h3{font-size:18px}.intro,.coverage,.window,.metadata{background:white;border:1px solid #d3dbe3;border-radius:10px;padding:20px;margin:18px 0}
.warning,.limit,.accepted{background:#fff4d9;padding:12px;border-left:4px solid #bd7d12}.technical{color:#465565}.raw-answer{background:#eef2f6;padding:12px;margin:16px 0}
.evidence{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0;min-width:0}img{display:block;max-width:100%;height:auto;border:1px solid #c8d1dc}
figcaption{padding:8px 0}.missing{padding:24px;background:#edf0f3;min-height:120px}dl{display:grid;grid-template-columns:minmax(220px,1fr) 1fr;gap:6px 20px}dt{font-weight:600}dd{margin:0}
p,li,dd,h2,code{overflow-wrap:anywhere}.description,pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}
@media(max-width:700px){.evidence{grid-template-columns:1fr}main{padding:12px}dl{grid-template-columns:1fr}}
</style></head><body><main><h1>Проверка загрузки и извлечения</h1>
<section class="intro"><strong>Эксперимент для ручной проверки.</strong>
<p>Показаны полные JPEG из папки этого отчёта с проверкой SHA256. Нажмите на кадр, чтобы открыть его в исходном размере.</p>
<p>Сырые ответы модели могут быть ошибочны. Направление переноса пока не подтверждено. Пересчёт денег и нарушение не установлены.</p>
<p>Время кадра не является точной секундой действия. Исчезновение пачки и положение руки сами по себе не доказывают загрузку или извлечение.</p></section>'''
    page += (f'<section class="metadata"><h2>Запуск проверки</h2><p>{_h(_status(report.get("status")))}.</p>'
             f'<p>Модель: {_h(_model_name(report))}.</p><p>Время этого прохода: {_h(_seconds(report.get("total_seconds")))}.</p>'
             '<p>Время не включает предыдущий анализ, запуск и переключение моделей, загрузку видео и очередь.</p>')
    if report.get("feature_support") is not None:
        page += f'<p class="limit">{_h(_support_text(report["feature_support"]))}</p>'
    page += _error_list(report.get("errors"))
    if report.get("runtime_metadata_error"):
        page += f'<p class="warning">Не удалось получить параметры модели: {_h(report["runtime_metadata_error"])}.</p>'
    page += '</section>' + _coverage(report, checks) + body
    limits = report.get("limitations", [])
    if limits:
        limits = limits if isinstance(limits, list) else [limits]
        page += '<section class="metadata"><h2>Ограничения этого отчёта</h2><ul>'
        for item in limits:
            text = _LIMITATIONS.get(item, item) if isinstance(item, str) else _support_text(item)
            page += f'<li>{_h(text)}</li>'
        page += '</ul></section>'
    metadata = {key: report[key] for key in (
        "kind", "model_label", "runtime", "settings", "source_report_path", "source_report_sha256",
        "source_report_snapshot", "source_status", "source_verification", "policy_sha256",
        "source_policy_sha256", "source_action_policy_sha256", "timing_scope",
    ) if key in report}
    if isinstance(report.get("direction_selection"), dict):
        metadata["selection_policy_sha256"] = report["direction_selection"].get("policy_sha256")
    page += ('<details class="metadata"><summary>Параметры запуска и воспроизводимость</summary>'
             f'<pre>{_h(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></details></main></body></html>')
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        output = render_direction(args.report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()
