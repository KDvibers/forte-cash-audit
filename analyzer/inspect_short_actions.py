"""Replay overlooked short action candidates on a local model, for review only."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .actions import ActionObserver, ACTION_SUPPORT, action_policy_fingerprint
from .action_windows import gate_cabinet_claim
from .inspect_direction import _dense_times, _file_sha, _verify_images, _sha, _SAMPLING
from .local_client import LocalModelClient
from .observations import ObservationDeadline
from .short_action_windows import select_short_action_windows
from .video import sample_video


def inspect_short_actions(source_report, out, *, grounding_url="http://127.0.0.1:8080",
                          activity_url="http://127.0.0.1:8081", budget_sec=45.,
                          max_windows=4, focus=False, exhaustive_contact=False):
    """Create a new report; never merge additional guesses into source facts.

    Both URLs must point to localhost. The deadline includes reconstruction and
    optional CPU resampling. File decoding/hash work cannot be preempted; after
    it returns, no model request starts if the deadline has passed.
    """
    if type(budget_sec) not in (int, float) or not math.isfinite(budget_sec) or budget_sec <= 0:
        raise ValueError("budget_sec must be finite and positive")
    if type(max_windows) is not int or not 0 <= max_windows <= 4:
        raise ValueError("max_windows must be an integer between zero and four")
    if type(focus) is not bool:
        raise ValueError("focus must be boolean")
    if type(exhaustive_contact) is not bool:
        raise ValueError("exhaustive_contact must be boolean")
    # Reject nonlocal endpoints before reading any video.
    grounding = LocalModelClient(grounding_url, timeout=20)
    activity = LocalModelClient(activity_url, timeout=20)
    start = time.perf_counter()
    deadline = start + budget_sec
    source_path = Path(source_report).expanduser().resolve(strict=True)
    source_bytes = source_path.read_bytes()
    destination = Path(out).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source_report.json").write_bytes(source_bytes)
    (destination / "frames").mkdir()
    report = {
        "schema_version": 1, "kind": "experimental_short_action_review", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_path": str(source_path), "source_report_sha256": _sha(source_bytes),
        "action_checks": [], "errors": [], "action_support": ACTION_SUPPORT,
        "action_policy_sha256": action_policy_fingerprint(exhaustive_contact=exhaustive_contact),
        "settings": {"grounding_url": grounding_url, "activity_url": activity_url,
                     "max_windows": max_windows, "budget_sec": budget_sec, "focus": focus,
                     "exhaustive_contact": exhaustive_contact},
        "source_verification": {"status": "pending"},
        "timing_scope": "additional_pass_including_source_verification_excludes_initial_analysis_upload_queue",
        "limitations": [
            "Additional model answers are review hypotheses, never API verdicts or established transfers.",
            "No answers are OR-merged with previous guesses; disagreement must be reviewed.",
            "Selection cannot recover actions outside previously selected source windows.",
            "A motion peak is a query proposal; head movements, doors and shadows can also cause it.",
            "No visible interaction in selected frames does not establish continuous absence.",
            "Model contact timestamps are estimates, not verified action onset.",
            "Without a historical video hash, matching saved JPEGs cannot establish identity of unsaved frames.",
        ],
    }

    def save():
        report["total_seconds"] = time.perf_counter() - start
        temp = destination / "report.json.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(destination / "report.json")

    def check_deadline():
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Short action replay budget exhausted")

    save()
    try:
        source = json.loads(source_bytes)
        if not isinstance(source, dict) or source.get("status") not in {"completed", "partial"}:
            raise ValueError("Source must be a completed or partial observation report")
        video = Path(source["video_path"]).expanduser()
        if not video.is_absolute():
            raise ValueError("Source video path must be absolute")
        video = video.resolve(strict=True)
        digest = _file_sha(video)
        if source.get("video_sha256") not in (None, digest):
            raise ValueError("Source video SHA256 differs from its report")
        check_deadline()
        sampled = sample_video(video, **_SAMPLING)
        frames = sampled["frames"]
        times = _dense_times(source, frames)
        if type(source.get("duration_sec")) not in (int, float) or not math.isfinite(source["duration_sec"]) or abs(source["duration_sec"]-sampled["duration_sec"]) > 1e-9:
            raise ValueError("Reconstructed duration differs from source report")
        verified = _verify_images(source, source_path.parent, frames, times)
        if _file_sha(video) != digest or source_path.read_bytes() != source_bytes:
            raise ValueError("Source changed during verification")
        report.update(video_path=str(video), video_sha256=digest, duration_sec=sampled["duration_sec"],
                      extraction_seconds=sampled["extraction_seconds"], source_status=source["status"])
        report["source_verification"] = {"status": "ok", "verified_source_images": verified,
            "verified_source_image_count": len(verified), "dense_frame_count": len(frames),
            "source_video_hash_previously_recorded": source.get("video_sha256") is not None}
        selection = select_short_action_windows(frames, source.get("action_checks", []), max_windows=max_windows)
        report["short_action_selection"] = selection
        chosen = [(w, [frames[i] for i in w["frame_indices"]]) for w in selection["windows"]]
        check_deadline()
        if focus and chosen:
            from .focus_video import sample_video_intervals, motion_peak_triple
            focused = sample_video_intervals(video, [w for w, _ in chosen], max_windows=max_windows)
            report["focus_sampling"] = {k: v for k, v in focused.items() if k != "frames"}
            focused_by_id = {w["window_id"]: w for w in focused["windows"]}
            updated = []
            for w, original in chosen:
                local_frames = [focused["frames"][i] for i in focused_by_id[w["window_id"]]["frame_indices"]]
                peak = motion_peak_triple(local_frames, w["object_bbox_normalized"])
                w = dict(w, focus_motion=peak)
                pictures = [local_frames[i] for i in peak["frame_indices"]] if peak["selected"] else original
                updated.append((w, pictures))
            chosen = updated
            if _file_sha(video) != digest:
                raise ValueError("Source video changed during focus extraction")
            check_deadline()
        observer = ActionObserver(grounding, activity)
        if chosen:
            check_deadline()
            grounding.timeout = min(20., deadline-time.perf_counter())
            report["runtime_grounding"] = grounding.server_info()
            check_deadline()
            activity.timeout = min(20., deadline-time.perf_counter())
            report["runtime_activity"] = activity.server_info()
            check_deadline()
        for w, pictures in chosen:
            evidence = []
            for i, frame in enumerate(pictures):
                relative = f"frames/{w['window_id']}_{i}.jpg"
                (destination / relative).write_bytes(frame.jpeg)
                evidence.append({"timestamp_sec": frame.timestamp_sec, "image_path": relative,
                                 "image_sha256": _sha(frame.jpeg)})
            row = {"window_id": w["window_id"], "region": w["roi"], "source_window_ids": w["source_window_ids"],
                   "selection": w, "evidence": evidence, "status": "not_started", "calls": [], "errors": []}
            report["action_checks"].append(row)
        save()
        for row, (w, pictures) in zip(report["action_checks"], chosen):
            check_deadline()
            row["status"] = "running"
            save()
            checked = observer.check(pictures, region=w["roi"], window_id=w["window_id"], deadline=deadline,
                                     counter_box=w["object_bbox_normalized"] if w["roi"] == "desk" else None,
                                     exhaustive_contact=exhaustive_contact)
            row.update(gate_cabinet_claim(checked, source.get("observations", [])))
            save()
        report["status"] = "partial" if source["status"] == "partial" or any(r["status"] != "ok" for r in report["action_checks"]) else "completed"
    except (ObservationDeadline, TimeoutError) as error:
        report["status"] = "partial"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    except Exception as error:
        report["status"] = "failed"
        if report["source_verification"]["status"] == "pending":
            report["source_verification"]["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        report["budget_exceeded"] = time.perf_counter() >= deadline
        try:
            report["source_report_unchanged"] = source_path.read_bytes() == source_bytes
        except OSError:
            report["source_report_unchanged"] = False
        if not report["source_report_unchanged"]:
            report["errors"].append({"type": "SourceChanged", "message": "Source report changed after its snapshot"})
        if (report["budget_exceeded"] or not report["source_report_unchanged"]) and report["status"] == "completed":
            report["status"] = "partial"
        report["request_count"] = sum(len(r["calls"]) for r in report["action_checks"])
        report["selected_window_count"] = len(report.get("short_action_selection", {}).get("windows", []))
        report["processed_window_count"] = sum(r["status"] in {"ok", "partial"} for r in report["action_checks"])
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=45.)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--focus", action="store_true")
    parser.add_argument("--all-anchors", action="store_true", help="Check edge contacts and recheck disagreements")
    args = parser.parse_args()
    report = inspect_short_actions(args.source_report, args.out, budget_sec=args.budget,
                                   max_windows=args.max_windows, focus=args.focus,
                                   exhaustive_contact=args.all_anchors)
    print(json.dumps({k: report[k] for k in ("status", "selected_window_count", "processed_window_count", "total_seconds", "errors")}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
