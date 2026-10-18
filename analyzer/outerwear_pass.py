"""Bounded local outerwear sampling; separate metadata from the cash pass."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from .observations import ObservationDeadline, ObservationError
from .outerwear import OuterwearObserver, outerwear_policy_fingerprint
from .outerwear_cascade import (CascadeOuterwearObserver, _identities,
                                cascade_policy_fingerprint, validate_cascade_record)
from .video import sample_video


def observe_outerwear_video(video_path, directory, *, client, deadline, observer=None):
    """Save actual sampled JPEGs and a dedicated, atomic outerwear report.

    ``directory`` is an existing observation root. Cash frames/reports are not
    changed. Reusing outerwear output names is rejected rather than overwritten.
    The caller owns the local model client and the absolute monotonic deadline;
    this function neither starts models nor resets their budget. Completion
    describes the selected samples, not all frames or every possible violation.
    An explicit cascade remains experimental. Its complete journals must match
    the frozen observer policy, runtime identities and actual source bytes;
    retaining a partial journal never authorizes publication of its hypotheses.
    """
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("deadline must be a finite absolute monotonic timestamp")
    if observer is not None and type(observer) not in (OuterwearObserver, CascadeOuterwearObserver):
        raise TypeError("Only the original or explicit cascade outerwear observer is supported")
    is_cascade = observer is not None and type(observer) is CascadeOuterwearObserver
    expected_runtimes = _identities(observer.expected_runtimes) if is_cascade else None
    destination = Path(directory).expanduser().resolve(strict=True)
    if not destination.is_dir():
        raise NotADirectoryError(str(destination))
    report_path = destination / "outerwear-report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Outerwear report already exists; use a new observation root")
    started = time.perf_counter()
    report = {
        "outerwear_checks": [], "selected_outerwear_frame_ids": [],
        "selected_outerwear_timestamps": [], "all_selected_outerwear_frames_processed": False,
        "outerwear_status": "failed", "outerwear_errors": [], "outerwear_budget_exceeded": False,
        "outerwear_settings": {"max_frames": 40, "interval_sec": 4.0, "max_side": 1280,
                               "deadline_semantics": "shared_absolute_monotonic_deadline"},
        "outerwear_policy_sha256": outerwear_policy_fingerprint(),
    }
    if is_cascade:
        report.update(outerwear_observer_kind="inactive_outerwear_cascade_v1",
                      outerwear_policy_sha256=cascade_policy_fingerprint(),
                      outerwear_expected_runtimes=deepcopy(expected_runtimes))
    current_frame, current_id, saved_image = None, None, None

    def check_deadline():
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Outerwear pass budget exhausted")

    def attach_image(record, *, interrupted=False):
        if (not isinstance(record, dict) or current_frame is None or saved_image is None
                or record.get("frame_id") != current_id
                or type(record.get("timestamp_sec")) not in (int, float)
                or record["timestamp_sec"] != current_frame.timestamp_sec):
            raise ObservationError("Outerwear record does not identify the current saved frame")
        if (record.get("state") not in {"yes", "no", "uncertain"}
                or record.get("status") not in {"ok", "partial", "error"}
                or not isinstance(record.get("errors"), list)):
            raise ObservationError("Invalid outerwear observation record")
        if not saved_image.is_file() or saved_image.read_bytes() != current_frame.jpeg:
            raise ObservationError("Saved outerwear JPEG changed or is missing")
        value = deepcopy(record)
        if is_cascade:
            if (record.get("kind") != "inactive_outerwear_cascade_v1"
                    or record.get("policy_sha256") != report["outerwear_policy_sha256"]
                    or record.get("image_sha256") != hashlib.sha256(current_frame.jpeg).hexdigest()):
                raise ObservationError("Cascade journal does not identify the frozen policy and current JPEG")
            try:
                value = validate_cascade_record(record, current_frame, current_id,
                                                expected_runtimes=expected_runtimes)
            except ObservationError as invalid:
                if record["status"] == "ok" and not record["errors"]:
                    raise
                # A partial trace is local diagnostic evidence only. Keeping it
                # must not make its observations or runtime claims usable.
                value["errors"].append({"feature": "outerwear_cascade", "type": type(invalid).__name__,
                                        "message": "Partial cascade journal was not validated: " + str(invalid)})
        if interrupted or value["status"] != "ok" or value["errors"]:
            value.update(state="uncertain", status="partial")
        if interrupted:
            value["errors"].append({"feature": "removed_outerwear", "type": "ObservationDeadline",
                                    "message": "Current frame did not complete within the shared deadline"})
        value["image_path"] = saved_image.relative_to(destination).as_posix()
        value["image_sha256"] = hashlib.sha256(current_frame.jpeg).hexdigest()
        return value

    try:
        check_deadline()
        sampled = sample_video(video_path, max_frames=40, interval_sec=4.0, max_side=1280, deadline=deadline)
        check_deadline()
        frames = sampled["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Video has no sampled outerwear frames")
        times = [frame.timestamp_sec for frame in frames]
        if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("Outerwear sample PTS must be finite and strictly increasing")
        identifiers = [f"outerwear_frame_{index:04d}" for index in range(len(frames))]
        report.update(selected_outerwear_frame_ids=identifiers, selected_outerwear_timestamps=times,
                      outerwear_sampling={key: value for key, value in sampled.items() if key != "frames"})
        frames_dir = destination / "frames"
        if frames_dir.is_symlink() or frames_dir.resolve().parent != destination:
            raise ValueError("Outerwear frame directory must stay inside the observation root")
        frames_dir.mkdir(exist_ok=True)
        paths = [frames_dir / f"{frame_id}.jpg" for frame_id in identifiers]
        if any(path.exists() or path.is_symlink() for path in paths):
            raise FileExistsError("Outerwear frame already exists; no existing JPEG may be overwritten")
        if observer is None:
            observer = OuterwearObserver(client)
        for frame, frame_id, path in zip(frames, identifiers, paths):
            current_frame, current_id, saved_image = frame, frame_id, None
            check_deadline()
            # Exclusive creation also protects against a file appearing after preflight.
            with path.open("xb") as handle:
                handle.write(frame.jpeg)
            saved_image = path
            check_deadline()
            record = observer.observe(frame, frame_id, deadline=deadline)
            if time.perf_counter() >= deadline:
                raise ObservationDeadline("Outerwear response arrived after the pass deadline", partial_record=record)
            value = attach_image(record)
            report["outerwear_checks"].append(value)
            if value["status"] != "ok" or value["errors"]:
                raise ObservationError("Outerwear frame has technical errors; remaining samples were not processed")
        check_deadline()
        report.update(outerwear_status="completed", all_selected_outerwear_frames_processed=True)
    except Exception as error:
        if isinstance(error, TimeoutError):
            report["outerwear_budget_exceeded"] = True
            partial_record = getattr(error, "partial_record", None)
            if partial_record is not None:
                try:
                    value = attach_image(partial_record, interrupted=True)
                except Exception as invalid:
                    report["outerwear_errors"].append({"type": type(invalid).__name__, "message": str(invalid)})
                else:
                    report["outerwear_checks"].append(value)
        report["outerwear_errors"].append({"type": type(error).__name__, "message": str(error)})
        report["outerwear_status"] = "partial" if report["outerwear_checks"] else "failed"
        report["all_selected_outerwear_frames_processed"] = False
    def mark_late_save():
        report.update(outerwear_status="partial", all_selected_outerwear_frames_processed=False,
                      outerwear_budget_exceeded=True)
        report["outerwear_errors"].append({"type": "ObservationDeadline",
                                          "message": "Outerwear report finalization exceeded the shared deadline"})

    def save_report():
        # Only this dedicated report is replaced, never the cash pass's report.json.
        temp_path = None
        try:
            report["outerwear_total_seconds"] = time.perf_counter() - started
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination,
                                             prefix=".outerwear-", suffix=".tmp", delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump(report, handle, ensure_ascii=False, indent=2)
            if report["outerwear_status"] == "completed" and time.perf_counter() >= deadline:
                mark_late_save()
                report["outerwear_total_seconds"] = time.perf_counter() - started
                temp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(report_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    save_report()
    # Native filesystem operations cannot be preempted. If replacement itself
    # finished late, persist that fact before returning, rather than claim success.
    if report["outerwear_status"] == "completed" and time.perf_counter() >= deadline:
        mark_late_save()
        save_report()
    return report
