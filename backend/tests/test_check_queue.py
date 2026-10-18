import json
import sqlite3

import pytest

from backend.check_queue import audit, main


def fixture(tmp_path, *, second_start=21, second_created=11, mode="real", outcome="passed"):
    database = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE jobs(job_id, status, mode, created_at, started_at, finished_at)")
        db.executemany("INSERT INTO jobs VALUES(?,?,?,?,?,?)", [
            ("one", "done", mode, 10, 12, 20), ("two", "done", mode, second_created, second_start, 30)])
    report = tmp_path / "run.json"
    report.write_text(json.dumps({"mode": "real", "checks": [
        {"job_id": "one", "outcome": outcome}, {"job_id": "two", "outcome": "passed"}]}))
    return database, report


def test_serial_real_jobs_with_observed_wait_pass_without_mutation(tmp_path):
    database, report = fixture(tmp_path)
    before = database.read_bytes()
    result = audit(database, report)
    assert result["passed"] is True
    assert result["second_upload_and_wait_seconds"] == 10
    assert database.read_bytes() == before


@pytest.mark.parametrize("settings", [{"second_start": 19}, {"second_created": 21}])
def test_overlapping_or_not_concurrent_run_does_not_confirm_queue(tmp_path, settings):
    assert audit(*fixture(tmp_path, **settings))["passed"] is False


@pytest.mark.parametrize("settings", [{"mode": "mock"}, {"outcome": "timeout"}])
def test_mock_or_unsuccessful_run_is_not_real_evidence(tmp_path, settings):
    with pytest.raises(ValueError):
        audit(*fixture(tmp_path, **settings))


def test_missing_database_is_not_created(tmp_path):
    _, report = fixture(tmp_path)
    missing = tmp_path / "missing.sqlite3"
    assert main(["--database", str(missing), "--report", str(report)]) == 1
    assert not missing.exists()
