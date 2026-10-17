"""Local case-review entry point; visual signals are not legal findings."""
import argparse
import base64
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


_ROOT = Path(__file__).resolve().parent.parent
_LOCK = threading.Lock()
_SERVERS = (("http://127.0.0.1:8080", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
            ("http://127.0.0.1:8081", "Qwen3VL-4B-Instruct-Q4_K_M.gguf"))
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class IncompleteAnalysisError(RuntimeError):
    """No supported signal and insufficient evidence for a negative result."""


class EvidenceError(RuntimeError):
    """A selected original evidence frame is missing, changed or invalid."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def check_ready():
    """Check local declared model identities without uploading video or loading weights."""
    identities = []
    for url, expected in _SERVERS:
        runtime = LocalModelClient(url, timeout=5).server_info()
        declared = str(runtime.get("model_path") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if declared != expected:
            raise RuntimeError(f"Unexpected model on {url}; expected {expected}")
        identities.append({"url": url, "runtime": runtime})
    return identities


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


def result_from_report(report, report_directory):
    """Export only supported review candidates using the existing case API schema.

    ``violation`` means that signs of a possible violation require review, as
    defined by the case. The description states the conditional normative link.
    ``no_violation`` concerns this implemented rule on sampled frames only.
    Missing evidence or an inconclusive negative result raises an exception.
    """
    evaluation = evaluate_rules(report)
    if report.get("status") not in {"completed", "partial"}:
        raise IncompleteAnalysisError("Observation pass has not produced a usable final report")
    directory = Path(report_directory).expanduser().resolve(strict=True)
    observations = {row["frame_id"]: row for row in report["observations"]}
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
            f"Человек ранее виден в {candidate['last_present_sec']:.2f} с; "
            f"второе последовательное модельное совпадение: {candidate['confirmed_at_sec']:.2f} с. "
            "Таймкод обозначает срабатывание правила после двух совпадений, а не точный момент ухода. "
            + specification["description_template"]
            + f" Связанное требование: {source['title']}, пункт {source['paragraph']}. {source['url']}. "
            "Между выбранными кадрами ситуация могла изменяться; точный момент ухода не установлен."
        )
        if evaluation["coverage"]["negative_blockers"] != ["visual_candidates_present"]:
            description += " Полнота остальных проверок ограничена; наличие этого сигнала не означает, что найдены все события."
        events.append({"type": specification["title"], "timecode_sec": candidate["timecode_sec"],
                       "description": description, "evidence_frames": evidence})
    if events:
        return {"verdict": "violation", "violations": events}
    if not evaluation["can_conclude_no_candidate"]:
        raise IncompleteAnalysisError("No supported signal, but the rule check is incomplete: "
                                      + ", ".join(evaluation["coverage"]["negative_blockers"]))
    return {"verdict": "no_violation", "violations": []}


def analyze_to_directory(video_path, out, *, budget_sec=260.0):
    """Run the first rule profile and retain local observations, evaluation and result."""
    if type(budget_sec) not in (int, float) or not math.isfinite(budget_sec) or budget_sec <= 10:
        raise ValueError("budget_sec must be finite and greater than ten seconds")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("An analysis is already running in this process; use one GPU worker")
    started = time.perf_counter()
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "rule_scope": [key for key, rule in RULES.items() if rule["enabled"]],
                "verdict_semantics": "case_review_signal_not_established_legal_breach",
                "timing_scope": "local_job_excluding_external_queue_upload_and_model_startup",
                "budget_sec": budget_sec}
    destination = None
    destination_created = False
    try:
        source = Path(video_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("video_path must identify a local file")
        destination = Path(out).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        destination_created = True
        metadata.update(video_path=str(source), source_video_sha256=_sha256(source))
        _save(destination / "job.json", metadata)
        metadata["runtime"] = check_ready()
        remaining = budget_sec-(time.perf_counter()-started)-3.0
        if remaining <= 0:
            raise TimeoutError("Job budget exhausted before observation")
        report_directory = destination / "observations"
        report = observe_video(source, report_directory, base_url=_SERVERS[0][0],
            activity_base_url=_SERVERS[1][0], cabinet_mode="disabled", interval_sec=2.0,
            max_frames=60, max_refinement_frames=0, max_motion_windows=0,
            enable_actions=False, budget_sec=remaining)
        if _sha256(source) != metadata["source_video_sha256"]:
            raise EvidenceError("Source video changed during analysis")
        report["source_video_sha256"] = metadata["source_video_sha256"]
        _save(report_directory / "report.json", report)
        evaluation = evaluate_rules(report)
        _save(destination / "rule_evaluation.json", evaluation)
        result = result_from_report(report, report_directory)
        if time.perf_counter()-started >= budget_sec:
            raise TimeoutError("Job result arrived after its local budget")
        _save(destination / "result.json", result)
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
        finally:
            _LOCK.release()


def analyze_video(video_path: str) -> dict:
    """Entry point consumed by backend.worker; no mock fallback or remote inference."""
    out = _ROOT / "runs" / "stage3" / ("service-" + uuid.uuid4().hex)
    return analyze_to_directory(video_path, out)


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
