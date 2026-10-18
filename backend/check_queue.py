"""Read only job timing metadata after a two-upload real acceptance run."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path


def audit(database: Path, report: Path) -> dict:
    record = json.loads(report.read_text(encoding="utf-8"))
    checks = record.get("checks", [])
    if (record.get("mode") != "real" or len(checks) != 2
            or any(check.get("outcome") != "passed" for check in checks)):
        raise ValueError("Нужен успешный отчёт двух настоящих загрузок.")
    ids = [check.get("job_id") for check in checks]
    if any(not isinstance(item, str) or not item for item in ids) or ids[0] == ids[1]:
        raise ValueError("В отчёте должны быть два разных job_id.")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute(
            "SELECT job_id, status, mode, created_at, started_at, finished_at "
            "FROM jobs WHERE job_id IN (?,?)", ids)]
    if len(rows) != 2:
        raise ValueError("Не найдены оба задания в указанной базе.")
    for row in rows:
        times = [row[name] for name in ("created_at", "started_at", "finished_at")]
        if (row["mode"] != "real" or row["status"] != "done"
                or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
                or not times[0] <= times[1] <= times[2]):
            raise ValueError("Нужны два завершённых real-задания с корректными временами.")
    first, second = sorted(rows, key=lambda row: row["started_at"])
    serial = second["started_at"] >= first["finished_at"]
    requests_overlapped = second["created_at"] < first["finished_at"]
    return {"worker_intervals_do_not_overlap": serial, "requests_overlapped": requests_overlapped,
            "passed": serial and requests_overlapped,
            "second_upload_and_wait_seconds": round(second["started_at"] - second["created_at"], 3)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.database, args.report)
    except (OSError, ValueError, TypeError, AttributeError, sqlite3.Error):
        print("[FAIL] Не удалось подтвердить очередь. Проверьте базу, отчёт и успешность обоих заданий.")
        return 1
    print(json.dumps(result, ensure_ascii=False))
    print("Проверены интервалы исполнителя приложения. Это не измерение GPU и не проверка качества модели.")
    print("Время до начала анализа включает приём видео и очередь; отдельно ожидание GPU здесь не измеряется.")
    print("Для замера не меняйте системное время; сторонние программы не должны использовать модели.")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
