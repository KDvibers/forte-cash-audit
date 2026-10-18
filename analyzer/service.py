"""Local case-review entry point; visual signals are not legal findings."""
import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from PIL import Image

from .local_client import LocalModelClient
from .observe import observe_video
from .rule_catalog import RULES
from .rules import evaluate_rules
from .outerwear import evaluate_outerwear, outerwear_policy_fingerprint
from .outerwear_cascade import (CascadeOuterwearObserver, _identities,
                               cascade_policy_fingerprint, validate_cascade_record)
from .video import VideoFrame


_ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()
_SERVERS = (("http://127.0.0.1:8080", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
            ("http://127.0.0.1:8081", "Qwen3VL-4B-Instruct-Q4_K_M.gguf"))
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_OUTERWEAR_RULE = "removed_outerwear_in_cash_workplace"
_VISUAL_BLOCKERS = {"unknown_condition_states", "blocked_visual_candidates", "excessive_frame_gaps",
                    "unknown_states", "unconfirmed_positive_frames", "visual_candidates_present"}


class IncompleteAnalysisError(RuntimeError):
    """No supported signal and insufficient evidence for a negative result."""

    def __init__(self, message, *, inference_completed=False):
        super().__init__(message)
        # A refusal may let the worker continue only after completed inference.
        # Invalid visual answers can be complete; partial/late HTTP calls cannot.
        self.inference_completed = inference_completed is True


class EvidenceError(RuntimeError):
    """A selected original evidence frame is missing, changed or invalid."""


def _remaining(deadline):
    remaining = deadline-time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Local analysis budget exhausted")
    return remaining


def _sha256(path, *, deadline=None):
    digest = hashlib.sha256()
    if deadline is not None:
        _remaining(deadline)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            if deadline is not None:
                _remaining(deadline)
            digest.update(block)
    if deadline is not None:
        _remaining(deadline)
    return digest.hexdigest()


def _save(path, value, *, deadline=None):
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        if deadline is not None:
            _remaining(deadline)
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        if deadline is not None:
            _remaining(deadline)
        temp.replace(path)
        if deadline is not None:
            try:
                _remaining(deadline)
            except TimeoutError:
                path.unlink(missing_ok=True)
                raise
    finally:
        temp.unlink(missing_ok=True)


def check_ready(*, deadline=None):
    """Check local declared model identities without uploading video or loading weights."""
    identities = []
    for url, expected in _SERVERS:
        timeout = 5 if deadline is None else min(5, _remaining(deadline))
        runtime = LocalModelClient(url, timeout=timeout).server_info()
        if deadline is not None:
            _remaining(deadline)
        declared = str(runtime.get("model_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if declared != expected:
            raise RuntimeError(f"Unexpected model on {url}; expected {expected}")
        identities.append({"url": url, "runtime": runtime})
    return identities


def describe_scope():
    """Describe enabled checks without loading a model or changing the case API."""
    return {
        "rules": [{"id": rule_id, "title": rule["title"]}
                  for rule_id, rule in RULES.items() if rule["enabled"]],
        "notice": (
            "Результат относится к перечисленным проверкам. "
            "Другие действия и требования кассовой дисциплины могут не проверяться. "
            "Обнаруженный признак требует проверки человеком."
        ),
        "complete_case_coverage": False,
    }


def _frame(row, directory):
    relative = row.get("image_path")
    expected = row.get("image_sha256")
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or not isinstance(expected, str) or not _HASH.fullmatch(expected)):
        raise EvidenceError("Evidence requires a relative path and capture-time SHA256")
    candidate = directory / relative
    # Refuse aliases: an evidence path must identify the file saved by this run.
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != directory.parent):
        raise EvidenceError("Linked evidence is not accepted")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(directory)
        stat = resolved.stat()
        if not resolved.is_file() or stat.st_nlink > 1 or not 0 < stat.st_size <= 6 * 1024 * 1024:
            raise EvidenceError("Invalid evidence file")
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise EvidenceError("Evidence SHA256 mismatch")
        with Image.open(BytesIO(raw)) as picture:
            if picture.format != "JPEG" or picture.width * picture.height > 40_000_000:
                raise EvidenceError("Evidence must be an original JPEG frame")
            picture.verify()
        with Image.open(BytesIO(raw)) as picture:
            picture.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise EvidenceError("Evidence is missing or invalid") from error
    return {"timestamp_sec": row["timestamp_sec"], "image_base64": base64.b64encode(raw).decode("ascii")}


def _uses_cascade(report):
    """Recognize new/unknown traces even if their report label was removed."""
    if ("outerwear_observer_kind" in report or "outerwear_expected_runtimes" in report
            or report.get("outerwear_policy_sha256") not in (None, outerwear_policy_fingerprint())):
        return True
    rows = report.get("outerwear_checks", [])
    return isinstance(rows, list) and any(isinstance(row, dict) and (
        "kind" in row or "stages" in row
        or row.get("policy_sha256") not in (None, outerwear_policy_fingerprint())) for row in rows)


def _validate_saved_cascade(report, directory, *, expected_runtimes, deadline=None):
    """Revalidate the full trace against saved source bytes before evaluation.

    The expected runtime snapshot comes from the explicitly supplied observer,
    never from the report being checked. No model or health request is made.
    """
    if deadline is not None:
        _remaining(deadline)
    try:
        if (RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is not True
                or report.get("outerwear_observer_kind") != "inactive_outerwear_cascade_v1"
                or report.get("outerwear_policy_sha256") != cascade_policy_fingerprint()):
            raise ValueError("Unknown or disabled cascade policy")
        expected = _identities(expected_runtimes)
        if (json.dumps(report.get("outerwear_expected_runtimes"), sort_keys=True, allow_nan=False)
                != json.dumps(expected, sort_keys=True, allow_nan=False)):
            raise ValueError("Cascade runtime snapshot changed or is missing")
        rows = report.get("outerwear_checks")
        ids, times = report.get("selected_outerwear_frame_ids"), report.get("selected_outerwear_timestamps")
        if (not isinstance(rows, list) or not rows or not isinstance(ids, list) or not isinstance(times, list)
                or len(ids) != len(times) or len(rows) > len(ids)
                or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in times)
                or any(right <= left for left, right in zip(times, times[1:]))):
            raise ValueError("Invalid cascade selection")
        incomplete = False
        for index, row in enumerate(rows):
            if deadline is not None:
                _remaining(deadline)
            if (not isinstance(row, dict) or row.get("frame_id") != ids[index]
                    or type(row.get("timestamp_sec")) not in (int, float) or row["timestamp_sec"] != times[index]):
                raise ValueError("Cascade row is not the saved selection prefix")
            if row.get("status") != "ok" or row.get("errors") != []:
                incomplete = True
                continue
            evidence = _frame(row, directory)
            raw = base64.b64decode(evidence["image_base64"], validate=True)
            with Image.open(BytesIO(raw)) as picture:
                frame = VideoFrame(times[index], raw, picture.width, picture.height)
            original = {key: value for key, value in row.items() if key != "image_path"}
            validate_cascade_record(original, frame, ids[index], expected_runtimes=expected)
            if deadline is not None:
                _remaining(deadline)
        if (incomplete or report.get("outerwear_status") != "completed" or report.get("outerwear_errors") != []
                or report.get("outerwear_budget_exceeded") is not False
                or report.get("all_selected_outerwear_frames_processed") is not True or len(rows) != len(ids)):
            raise ValueError("Cascade did not complete the entire selected pass")
    except TimeoutError:
        raise
    except (KeyError, TypeError, ValueError, EvidenceError) as error:
        raise IncompleteAnalysisError("Cascade report is not independently verified: " + str(error),
                                      inference_completed=False) from error


def evaluate_report(report, *, report_directory=None, outerwear_expected_runtimes=None, deadline=None):
    """Combine only explicitly enabled rules without losing per-rule coverage."""
    if _uses_cascade(report) or outerwear_expected_runtimes is not None:
        if report_directory is None:
            raise IncompleteAnalysisError("Cascade evaluation requires saved source frames and independent runtimes")
        _validate_saved_cascade(report, Path(report_directory).expanduser().resolve(strict=True),
                                expected_runtimes=outerwear_expected_runtimes, deadline=deadline)
    cash = evaluate_rules(report)
    if RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is not True:
        return cash
    if not isinstance(report.get("outerwear_checks"), list):
        raise IncompleteAnalysisError("Enabled outerwear check is missing from the report")
    outerwear = evaluate_outerwear(report)
    checks = {"cash_visible_after_person_disappears": cash, _OUTERWEAR_RULE: outerwear}
    # Each pass owns its status. Failure of one check must not discard a
    # supported candidate from another, nor publish the failed pass's rows.
    usable = {"cash_visible_after_person_disappears": report.get("status") in {"completed", "partial"},
              _OUTERWEAR_RULE: report.get("outerwear_status") in {"completed", "partial"}}
    blockers = [f"{key}:{reason}" for key, value in checks.items()
                for reason in value["coverage"]["negative_blockers"]]
    return {
        "kind": "combined_visual_candidates_not_legal_verdict", "checks": checks,
        "candidates": sorted([candidate for key, check in checks.items() if usable[key]
                              for candidate in check["candidates"]],
                             key=lambda value: (value["timecode_sec"], value["source_rule_id"])),
        "blocked_candidates": cash["blocked_candidates"] + outerwear["blocked_candidates"],
        "can_conclude_no_candidate": all(value["can_conclude_no_candidate"] for value in checks.values()),
        "coverage": {"negative_blockers": blockers,
                     "coverage_limited": any(reason != "visual_candidates_present" for value in checks.values()
                                             for reason in value["coverage"]["negative_blockers"])},
    }


def _model_call_returned(call):
    seconds = call.get("request_seconds") if isinstance(call, dict) else None
    response = call.get("response", call) if isinstance(call, dict) else None
    return (isinstance(call, dict) and call.get("completed") is True
            and isinstance(response, dict) and response.get("finish_reason") == "stop"
            and not call.get("error_type") and not call.get("error")
            and type(response.get("valid_json")) is bool
            and type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0)


def _recorded_calls_returned(rows):
    return isinstance(rows, list) and all(
        isinstance(row, dict) and isinstance(row.get("calls", []), list)
        and all(_model_call_returned(call) for call in row.get("calls", [])) for row in rows)


def _cash_inference_completed(report, evaluation):
    """Check cash-pass completion before any optional pass exists.

    A clean visual uncertainty is complete inference. An invalid answer or an
    explicit unfinished request cannot authorize another pass on the same GPU.
    """
    rows = report.get("observations")
    if (report.get("status") != "completed" or report.get("errors")
            or report.get("budget_exceeded") is True
            or report.get("all_selected_frames_processed") is not True
            or not rows or not _recorded_calls_returned(rows)
            or any(row.get("status") != "ok" or row.get("errors") for row in rows)):
        return False
    return set(evaluation["coverage"]["negative_blockers"]) <= _VISUAL_BLOCKERS


def _inference_completed(report, evaluation):
    """Prove technical completion separately from a visual yes/no/unknown."""
    checks = evaluation.get("checks", {"cash_visible_after_person_disappears": evaluation})
    if not _cash_inference_completed(report, checks["cash_visible_after_person_disappears"]):
        return False
    if RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is True:
        if (report.get("outerwear_status") != "completed" or report.get("outerwear_errors")
                or report.get("outerwear_budget_exceeded") is True
                or report.get("all_selected_outerwear_frames_processed") is not True
                or any(row.get("status") != "ok" or row.get("errors") for row in report["outerwear_checks"])):
            return False
    return all(set(check["coverage"]["negative_blockers"]) <= _VISUAL_BLOCKERS for check in checks.values())


def _outerwear_schema_refusal_completed(report, evaluation):
    """Prove returned calls for the one known early-stop outerwear failure.

    This proves only that the next job may run. The unprocessed suffix and the
    invalid last answer remain incomplete and cannot publish a live verdict.
    """
    checks = evaluation.get("checks", {})
    cash_rule = "cash_visible_after_person_disappears"
    if (set(checks) != {cash_rule, _OUTERWEAR_RULE}
            or any(rule["enabled"] and key not in checks for key, rule in RULES.items())
            or not _cash_inference_completed(report, checks[cash_rule])):
        return False
    settings = report.get("settings", {})
    if (not isinstance(settings, dict) or settings.get("enable_actions") is not False
            or settings.get("max_motion_windows") != 0 or settings.get("max_refinement_frames") != 0
            or report.get("action_checks") or report.get("counter_discovery")
            or report.get("outerwear_status") != "partial"
            or report.get("outerwear_budget_exceeded") is not False
            or report.get("all_selected_outerwear_frames_processed") is not False
            or report.get("outerwear_policy_sha256") != outerwear_policy_fingerprint()):
        return False
    global_errors = report.get("outerwear_errors")
    if (not isinstance(global_errors, list) or len(global_errors) != 1
            or not isinstance(global_errors[0], dict) or global_errors[0].get("type") != "ObservationError"):
        return False
    cash_rows, rows = report["observations"], report.get("outerwear_checks")
    if not isinstance(rows, list) or not rows:
        return False
    # Cash has a complete selection; outerwear stopped exactly at the last
    # recorded frame. No missing or additional started calls can be inferred.
    if (type(report.get("selected_frame_count")) is not int
            or type(report.get("processed_frame_count")) is not int
            or report["selected_frame_count"] != len(cash_rows)
            or report["processed_frame_count"] != len(cash_rows)
            or report.get("selected_timestamps") != [row.get("timestamp_sec") for row in cash_rows]
            or any(not row.get("calls") for row in cash_rows)
            or type(report.get("request_count")) is not int
            or report["request_count"] != sum(len(row["calls"]) for row in cash_rows)):
        return False
    selected_ids, selected_times = (report.get("selected_outerwear_frame_ids"),
                                    report.get("selected_outerwear_timestamps"))
    if (not isinstance(selected_ids, list) or not isinstance(selected_times, list)
            or len(selected_ids) != len(selected_times) or len(rows) > len(selected_ids)
            or any(not isinstance(value, str) or not value.strip() for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in selected_times)
            or any(b <= a for a, b in zip(selected_times, selected_times[1:]))
            or selected_ids[:len(rows)] != [row.get("frame_id") for row in rows]
            or selected_times[:len(rows)] != [row.get("timestamp_sec") for row in rows]):
        return False
    policy = outerwear_policy_fingerprint()
    for index, row in enumerate(rows):
        calls, errors = row.get("calls"), row.get("errors")
        if (row.get("policy_sha256") != policy or not isinstance(calls, list)
                or [call.get("question") for call in calls] not in
                    (["removed_outerwear"], ["removed_outerwear", "outerwear_structure"])
                or not _recorded_calls_returned([row])):
            return False
        if index < len(rows)-1:
            if row.get("status") != "ok" or errors != []:
                return False
        elif (row.get("status") != "partial" or row.get("state") != "uncertain"
                or not isinstance(errors, list) or len(errors) != 1
                or not isinstance(errors[0], dict) or errors[0].get("type") != "ObservationError"
                or errors[0].get("feature") != "removed_outerwear"):
            return False
    return True


def _refusal_inference_completed(report, evaluation, *, cash_pass_only=False):
    """Allow a failed job to release the worker, never to publish a verdict.

    Narrow fallbacks cover schema failures after fully recorded returned calls.
    Unknown errors, missing call records and unverified passes remain
    quarantined. Publication still requires the stricter _inference_completed.
    cash_pass_only is used solely before an optional pass has been started;
    any optional-pass metadata makes that mode refuse recovery.
    """
    # New cascade refusals have no recovery proof in the original service.
    # Even returned calls must not silently enter the old safe-refusal policy.
    if _uses_cascade(report):
        return False
    if cash_pass_only and any(key.startswith(("outerwear_", "selected_outerwear_", "all_selected_outerwear_"))
                              for key in report):
        return False
    rows = report.get("observations")
    if not isinstance(rows, list) or not rows:
        return False
    outerwear_rows = report.get("outerwear_checks", [])
    if not isinstance(outerwear_rows, list):
        return False
    all_rows = rows + outerwear_rows
    # Even an otherwise clean summary cannot overrule an explicit pending call.
    if not _recorded_calls_returned(all_rows):
        return False
    complete = (_cash_inference_completed(report, evaluation) if cash_pass_only
                else _inference_completed(report, evaluation))
    if complete:
        return True
    if RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is True and not cash_pass_only:
        return _outerwear_schema_refusal_completed(report, evaluation)
    settings = report.get("settings", {})
    if (report.get("status") != "partial" or report.get("errors") != []
            or report.get("budget_exceeded") is not False
            or not isinstance(settings, dict) or settings.get("enable_actions") is not False
            or settings.get("max_motion_windows") != 0 or settings.get("max_refinement_frames") != 0
            or report.get("action_checks") or report.get("counter_discovery")
            or report.get("outerwear_checks")):
        return False
    selected = report.get("selected_frame_count")
    processed = report.get("processed_frame_count")
    times = report.get("selected_timestamps")
    actual_times = [row.get("timestamp_sec") for row in rows]
    ids = [row.get("frame_id") for row in rows]
    if (type(selected) is not int or type(processed) is not int
            or selected != processed or selected != len(rows)
            or not isinstance(times, list) or times != actual_times
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        return False
    if any(not row.get("calls") for row in rows):
        return False
    calls = [call for row in rows for call in row["calls"]]
    if type(report.get("request_count")) is not int or report["request_count"] != len(calls):
        return False
    has_schema_failure = False
    for row in rows:
        errors = row.get("errors")
        if not row["calls"] or not isinstance(errors, list):
            return False
        if errors:
            if row.get("status") != "partial" or any(
                    not isinstance(error, dict) or error.get("type") != "ObservationError" for error in errors):
                return False
            has_schema_failure = True
        elif row.get("status") != "ok":
            return False
    return has_schema_failure


def result_from_report(report, report_directory, *, outerwear_expected_runtimes=None, deadline=None):
    """Export only supported review candidates using the existing case API schema.

    ``violation`` means that signs of a possible violation require review, as
    defined by the case. The description states the conditional normative link.
    ``no_violation`` concerns the enabled checks on sampled frames only.
    Missing evidence or an inconclusive negative result raises an exception.
    """
    if _uses_cascade(report) or outerwear_expected_runtimes is not None:
        evaluation = evaluate_report(report, report_directory=report_directory,
                                     outerwear_expected_runtimes=outerwear_expected_runtimes, deadline=deadline)
    else:
        evaluation = evaluate_report(report)
    outerwear_usable = (RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is True
                       and report.get("outerwear_status") in {"completed", "partial"})
    if report.get("status") not in {"completed", "partial"} and not outerwear_usable:
        raise IncompleteAnalysisError("Observation pass has not produced a usable final report")
    directory = Path(report_directory).expanduser().resolve(strict=True)
    observations = {row["frame_id"]: row for row in report["observations"]}
    if RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is True:
        for row in report["outerwear_checks"]:
            if row["frame_id"] in observations:
                raise EvidenceError("Rule evidence frame IDs must be unique across passes")
            observations[row["frame_id"]] = row
    events = []
    total_bytes = 0
    for candidate in evaluation["candidates"]:
        specification = RULES.get(candidate["source_rule_id"])
        if not specification or specification.get("enabled") is not True:
            raise IncompleteAnalysisError("The selected rule is disabled or absent")
        if (specification.get("kind") != "case_review_signal"
                or specification.get("direct_legal_finding") is not False
                or specification.get("requires_human_review") is not True):
            raise IncompleteAnalysisError("Unsupported rule publication policy")
        ids = candidate["evidence_ids"]
        if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
            raise EvidenceError("A signal needs one to three distinct source frames")
        evidence = [_frame(observations[frame_id], directory) for frame_id in ids]
        total_bytes += sum(len(base64.b64decode(frame["image_base64"])) for frame in evidence)
        if len(events) >= 100 or total_bytes > 30 * 1024 * 1024:
            raise EvidenceError("Case API evidence size limit exceeded")
        source = specification["source"]
        description = (
            f"Первый модельный признак: {candidate['start_sec']:.2f} с; "
            f"последнее совпадение в выбранных кадрах: {candidate['end_sec']:.2f} с. "
            f"второе последовательное модельное совпадение: {candidate['confirmed_at_sec']:.2f} с. "
            "Таймкод указывает первое совпадение, подтверждённое следующим кадром. "
            + specification["description_template"]
            + f" Связанное требование: {source['title']}, пункт {source['paragraph']}. {source['url']}. "
            "Между выбранными кадрами ситуация могла изменяться."
        )
        if candidate.get("last_present_sec") is not None:
            description += (
                f" Человек ранее виден в {candidate['last_present_sec']:.2f} с; "
                "точный момент ухода не установлен."
            )
        bounds = candidate.get("condition_onset_bounds")
        if bounds is not None:
            description += (
                f" Последний кадр без сочетания признаков: {bounds['last_false_sec']:.2f} с; "
                f"первый кадр с сочетанием: {bounds['first_true_sec']:.2f} с. "
                "Это границы наблюдаемого изменения, а не доказательство непрерывности между кадрами."
            )
        if candidate.get("interruptions"):
            description += (
                f" Подтверждённых фрагментов: {len(candidate['matching_segments'])}. "
                "Они сгруппированы по временной истории наблюдений; промежуточные кадры "
                "не подтверждают непрерывное событие."
            )
        limited = evaluation["coverage"].get("coverage_limited",
            any(reason != "visual_candidates_present" for reason in evaluation["coverage"]["negative_blockers"]))
        if limited:
            description += " Полнота остальных проверок ограничена; наличие этого сигнала не означает, что найдены все события."
        events.append({"type": specification["title"], "timecode_sec": candidate["timecode_sec"],
                       "description": description, "evidence_frames": evidence})
    if events:
        return {"verdict": "violation", "violations": events}
    if not evaluation["can_conclude_no_candidate"]:
        raise IncompleteAnalysisError("No supported signal, but the rule check is incomplete: "
                                      + ", ".join(evaluation["coverage"]["negative_blockers"]),
                                      inference_completed=_refusal_inference_completed(report, evaluation))
    return {"verdict": "no_violation", "violations": []}


def analyze_to_directory(video_path, out, *, budget_sec=260.0, outerwear_observer=None):
    """Run enabled rule passes and retain local observations, evaluation and result."""
    if type(budget_sec) not in (int, float) or not math.isfinite(budget_sec) or budget_sec <= 10:
        raise ValueError("budget_sec must be finite and greater than ten seconds")
    cascade_runtimes = None
    if outerwear_observer is not None:
        if (RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is not True
                or type(outerwear_observer) is not CascadeOuterwearObserver):
            raise ValueError("An explicit cascade requires the enabled outerwear rule and exact observer type")
        cascade_runtimes = _identities(deepcopy(outerwear_observer.expected_runtimes))
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("An analysis is already running in this process; use one GPU worker")
    started = time.perf_counter()
    deadline = started+budget_sec
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "rule_scope": [key for key, rule in RULES.items() if rule["enabled"]],
                "verdict_semantics": "case_review_signal_not_established_legal_breach",
                "timing_scope": "local_job_excluding_external_queue_upload_and_model_startup",
                "budget_sec": budget_sec}
    if cascade_runtimes is not None:
        metadata.update(outerwear_observer_kind="inactive_outerwear_cascade_v1",
                        outerwear_expected_runtimes=deepcopy(cascade_runtimes),
                        outerwear_policy_sha256=cascade_policy_fingerprint())
    destination = None
    destination_created = False
    try:
        source = Path(video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("video_path must identify a local file")
        destination = Path(out).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        destination_created = True
        metadata.update(video_path=str(source), source_video_sha256=_sha256(source, deadline=deadline))
        _save(destination / "job.json", metadata)
        metadata["runtime"] = check_ready(deadline=deadline)
        remaining = budget_sec-(time.perf_counter()-started)-3.0
        if remaining <= 0:
            raise TimeoutError("Job budget exhausted before observation")
        report_directory = destination / "observations"
        report = observe_video(source, report_directory, base_url=_SERVERS[0][0],
            activity_base_url=_SERVERS[1][0], cabinet_mode="disabled", interval_sec=2.0,
            max_frames=80, max_refinement_frames=0, max_motion_windows=0,
            enable_actions=False, skip_cash_when_person_present=True, budget_sec=remaining)
        if RULES.get(_OUTERWEAR_RULE, {}).get("enabled") is True:
            cash_evaluation = evaluate_rules(report)
            if not _cash_inference_completed(report, cash_evaluation):
                # Do not overlap a possibly unfinished cash request with another
                # model pass. Preserve the cash-only review, without pretending
                # that the skipped rule was checked or exporting its verdict.
                metadata["skipped_rule_passes"] = [{"rule_id": _OUTERWEAR_RULE,
                    "reason": "cash_pass_not_technically_complete"}]
                _save(report_directory / "report.json", report)
                _save(destination / "rule_evaluation.json", cash_evaluation)
                raise IncompleteAnalysisError("Cash pass did not finish cleanly; optional rule pass was not started",
                    inference_completed=_refusal_inference_completed(report, cash_evaluation, cash_pass_only=True))
            _remaining(deadline-3.0)
            from .outerwear_pass import observe_outerwear_video
            if outerwear_observer is None:
                outerwear = observe_outerwear_video(source, report_directory,
                    client=LocalModelClient(_SERVERS[1][0], timeout=20), deadline=deadline-3.0)
            else:
                outerwear = observe_outerwear_video(source, report_directory,
                    client=outerwear_observer.clients["filter4b"], deadline=deadline-3.0,
                    observer=outerwear_observer)
            report.update(outerwear)
            report["combined_status"] = ("completed" if report.get("status") == "completed"
                                         and report.get("outerwear_status") == "completed" else "partial")
        if _sha256(source, deadline=deadline) != metadata["source_video_sha256"]:
            raise EvidenceError("Source video changed during analysis")
        report["source_video_sha256"] = metadata["source_video_sha256"]
        _save(report_directory / "report.json", report)
        if cascade_runtimes is not None or _uses_cascade(report):
            try:
                evaluation = evaluate_report(report, report_directory=report_directory,
                                             outerwear_expected_runtimes=cascade_runtimes, deadline=deadline)
            except Exception:
                # Preserve independently available cash review without ever
                # evaluating unverified cascade rows or declaring their verdict.
                _save(destination / "rule_evaluation.json", {
                    "kind": "cascade_rejected_cash_review_only",
                    "checks": {"cash_visible_after_person_disappears": evaluate_rules(report)},
                    "cascade_validated": False})
                raise
        else:
            evaluation = evaluate_report(report)
        _save(destination / "rule_evaluation.json", evaluation)
        if not _inference_completed(report, evaluation):
            # A timed-out HTTP client may leave inference running in llama.cpp.
            # Keep candidates for local review, but quarantine the live worker
            # instead of publishing an early positive and admitting another job.
            raise IncompleteAnalysisError("Model pass did not finish cleanly; saved candidates require local review",
                                          inference_completed=_refusal_inference_completed(report, evaluation))
        if cascade_runtimes is None:
            result = result_from_report(report, report_directory)
        else:
            result = result_from_report(report, report_directory,
                                        outerwear_expected_runtimes=cascade_runtimes, deadline=deadline)
        if time.perf_counter()-started >= budget_sec:
            raise TimeoutError("Job result arrived after its local budget")
        _save(destination / "result.json", result, deadline=deadline)
        metadata["status"] = "done"
        return result
    except Exception as error:
        metadata.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        metadata["total_seconds"] = time.perf_counter()-started
        try:
            if destination_created:
                _save(destination / "job.json", metadata)
                if metadata["status"] == "done" and time.perf_counter() >= deadline:
                    # Final metadata I/O is part of the local job as well.
                    (destination / "result.json").unlink(missing_ok=True)
                    error = TimeoutError("Job finalization exceeded its local budget")
                    metadata.update(status="failed", total_seconds=time.perf_counter()-started,
                                    error={"type": type(error).__name__, "message": str(error)})
                    _save(destination / "job.json", metadata)
                    raise error
        finally:
            _LOCK.release()


def analyze_video(video_path: str, *, budget_sec: float = 260.0) -> dict:
    """Entry point consumed by backend.worker; no mock fallback or remote inference."""
    out = _ROOT / "runs" / "stage3" / ("service-" + uuid.uuid4().hex)
    return analyze_to_directory(video_path, out, budget_sec=budget_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New run directory under ignored runs/")
    parser.add_argument("--budget", type=float, default=260.0)
    args = parser.parse_args()
    try:
        result = analyze_to_directory(args.video, args.out, budget_sec=args.budget)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "message": str(error), "directory": str(args.out)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps({"status": "done", "verdict": result["verdict"],
                      "signals": len(result["violations"]), "directory": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
