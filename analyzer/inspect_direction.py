"""Replay verified evidence for a separate, local direction review pass."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import time
from urllib.parse import urlsplit

from .direction import DirectionObserver, DIRECTION_SUPPORT, direction_policy_fingerprint
from .direction_windows import select_direction_windows
from .local_client import LocalModelClient
from .observations import ObservationDeadline
from .smoke import GpuMonitor
from .video import sample_video


_EPS = 1e-9
_SAMPLING = {"max_frames": 600, "max_side": 1280, "interval_sec": .5}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _dense_times(source, frames):
    count = source.get("available_frame_count")
    if type(count) is not int or count < 2 or count != len(frames):
        raise ValueError("Reconstructed dense frame count differs from the source report")
    times = [frame.timestamp_sec for frame in frames]
    if (not all(_finite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Reconstructed actual timestamps are invalid")
    recorded = {}
    for name in ("motion", "counter_motion"):
        section = source.get(name, {})
        scores = section.get("scores", {}) if isinstance(section, dict) else {}
        if not isinstance(scores, dict):
            raise ValueError("Invalid dense timestamp records")
        for rows in scores.values():
            if not isinstance(rows, list):
                raise ValueError("Invalid dense timestamp records")
            for row in rows:
                index = row.get("frame_index") if isinstance(row, dict) else None
                if type(index) is not int or not 1 <= index < count:
                    raise ValueError("Invalid dense pair index")
                for i, t in ((index-1, row.get("start_sec")), (index, row.get("end_sec"))):
                    if not _finite(t) or (i in recorded and abs(recorded[i]-t) > _EPS):
                        raise ValueError("Conflicting or invalid recorded dense timestamps")
                    recorded[i] = t
    if set(recorded) != set(range(count)):
        raise ValueError("Source report does not record the complete dense timestamp pool")
    if any(abs(times[i]-recorded[i]) > _EPS for i in recorded):
        raise ValueError("Reconstructed actual timestamps differ from the source report")
    gap = source.get("dense_max_sample_gap_sec")
    if not _finite(gap) or abs(gap-max(b-a for a, b in zip(times, times[1:]))) > _EPS:
        raise ValueError("Dense sample gap differs from the source report")
    return times


def _image_references(value, require_hash=False):
    if isinstance(value, dict):
        if "image_path" in value:
            yield value, require_hash
        for key, child in value.items():
            yield from _image_references(child, require_hash or key == "evidence")
    elif isinstance(value, list):
        for child in value:
            yield from _image_references(child, require_hash)


def _confined_image(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Source evidence has no image path")
    relative = PurePosixPath(value.replace("\\", "/"))
    if (relative.is_absolute() or PureWindowsPath(value).drive or urlsplit(value).scheme
            or ".." in relative.parts or any(":" in part for part in relative.parts)):
        raise ValueError("Source evidence must be a relative JPEG inside its report folder")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("Source evidence is not a JPEG file")
    return path


def _verify_images(source, root, frames, times):
    """Check every referenced image, plus all saved dense-index JPEGs."""
    verified = {}
    references = list(_image_references(source))
    for path in sorted((root / "frames").glob("frame_*.jpg")):
        match = re.fullmatch(r"frame_(\d+)\.jpg", path.name)
        if not match or int(match[1]) >= len(frames):
            raise ValueError("Stored frame filename has no reconstructed dense index")
        references.append(({"image_path": path.relative_to(root).as_posix(),
                            "timestamp_sec": times[int(match[1])]}, False))
    for row, require_hash in references:
        path = _confined_image(root, row["image_path"])
        timestamp = row.get("timestamp_sec")
        indices = [i for i, t in enumerate(times) if _finite(timestamp) and abs(t-timestamp) <= _EPS]
        if len(indices) != 1:
            raise ValueError("Source evidence time does not identify one actual dense frame")
        index = indices[0]
        data = path.read_bytes()
        digest = _sha(data)
        expected = row.get("image_sha256")
        if (expected is not None or require_hash) and (
                not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected)
                or digest != expected.lower()):
            raise ValueError("Source evidence SHA256 is missing, invalid or mismatched")
        if not data.startswith(b"\xff\xd8\xff") or digest != _sha(frames[index].jpeg):
            raise ValueError("Stored JPEG does not match the reconstructed actual frame")
        key = path.relative_to(root).as_posix()
        previous = verified.get(key)
        if previous and previous["pool_index"] != index:
            raise ValueError("One source JPEG has conflicting actual timestamps")
        verified[key] = {"image_path": key, "pool_index": index, "timestamp_sec": times[index],
                         "image_sha256": digest,
                         "source_recorded_sha256_verified": expected is not None
                         or bool(previous and previous["source_recorded_sha256_verified"])}
    return list(verified.values())


def inspect_direction(source_report, out, *, base_url="http://127.0.0.1:8082", budget_sec=60.0,
                      max_windows=4, reserve_context=False):
    """Create a new report without changing the source or managing model servers.

    Timing covers this pass only, including reconstruction and verification.
    Model startup, switching, previous analysis, upload and queueing are excluded.
    The caller must unload previous models before starting its direction server.
    """
    if type(budget_sec) not in (int, float) or not math.isfinite(budget_sec) or budget_sec <= 0:
        raise ValueError("budget_sec must be finite and positive")
    if type(max_windows) is not int or max_windows < 0:
        raise ValueError("max_windows must be a nonnegative integer")
    if type(reserve_context) is not bool:
        raise ValueError("reserve_context must be boolean")
    started = time.perf_counter()
    deadline = started + budget_sec
    source_path = Path(source_report).expanduser().resolve(strict=True)
    source_bytes = source_path.read_bytes()
    destination = Path(out).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source_report.json").write_bytes(source_bytes)
    (destination / "frames").mkdir()
    report = {
        "schema_version": 1, "kind": "experimental_direction_review", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_path": str(source_path), "source_report_sha256": _sha(source_bytes),
        "source_report_snapshot": "source_report.json", "policy_sha256": direction_policy_fingerprint(),
        "settings": {"base_url": base_url, "budget_sec": budget_sec, "max_windows": max_windows,
                     "sampling": dict(_SAMPLING), "request_timeout_sec": 20.0,
                     "reserve_context": reserve_context},
        "direction_checks": [], "errors": [], "feature_support": dict(DIRECTION_SUPPORT),
        "source_verification": {"status": "pending"},
        "timing_scope": "direction_pass_only_including_reconstruction_and_verification",
        "limitations": [
            "Experimental model answers are not rule facts or violation verdicts.",
            "This pass does not start, stop or switch model servers.",
            "Timing excludes previous analysis, model startup/switching, upload and queueing.",
            "Decoding and file hashing are not preemptible; deadlines are checked afterwards.",
            "Source reports without a video hash cannot prove historical identity of unsaved frames.",
            "Selection cannot recover upstream missed or uncertain interactions.",
            "Successful requests and stable object geometry do not establish visual correctness.",
        ],
    }
    monitor = None

    def save():
        report["total_seconds"] = time.perf_counter()-started
        temporary = destination / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination / "report.json")

    def check_deadline():
        if time.perf_counter() >= deadline:
            raise ObservationDeadline("Direction pass budget exhausted")

    save()
    try:
        source = json.loads(source_bytes)
        if not isinstance(source, dict) or source.get("status") not in {"completed", "partial"}:
            raise ValueError("Source must be a completed or partial observation report")
        report.update(source_status=source["status"], source_policy_sha256=source.get("policy_sha256"),
                      source_action_policy_sha256=source.get("action_policy_sha256"))
        video = Path(source["video_path"]).expanduser()
        if not video.is_absolute():
            raise ValueError("Source video path must be absolute")
        video = video.resolve(strict=True)
        before_hash = _file_sha(video)
        report.update(video_path=str(video), video_sha256=before_hash)
        expected_video_hash = source.get("video_sha256")
        if expected_video_hash is not None and expected_video_hash != before_hash:
            raise ValueError("Source video SHA256 differs from its report")
        check_deadline()
        sampled = sample_video(video, **_SAMPLING)
        frames = sampled["frames"]
        times = _dense_times(source, frames)
        if (not _finite(source.get("duration_sec"))
                or abs(source["duration_sec"]-sampled["duration_sec"]) > _EPS):
            raise ValueError("Reconstructed video duration differs from the source report")
        verified = _verify_images(source, source_path.parent, frames, times)
        if _file_sha(video) != before_hash:
            raise ValueError("Source video changed during reconstruction")
        if source_path.read_bytes() != source_bytes:
            raise ValueError("Source report changed during verification")
        report["source_verification"] = {
            "status": "ok", "dense_frame_count": len(frames), "all_dense_timestamps_equal": True,
            "dense_timestamps_sec": times, "verified_source_images": verified,
            "verified_source_image_count": len(verified),
            "source_video_hash_previously_recorded": expected_video_hash is not None,
            "video_unchanged_during_reconstruction": True,
        }
        report.update(duration_sec=sampled["duration_sec"], extraction_seconds=sampled["extraction_seconds"],
                      dense_max_sample_gap_sec=sampled["max_sample_gap_sec"])
        selection = select_direction_windows(frames, source.get("action_checks", []), max_windows=max_windows,
                                             reserve_context=reserve_context)
        report["direction_selection"] = selection
        for window in selection["windows"]:
            evidence = []
            for index in window["frame_indices"]:
                frame = frames[index]
                relative = f"frames/frame_{index:04d}.jpg"
                (destination / relative).write_bytes(frame.jpeg)
                evidence.append({"pool_index": index, "timestamp_sec": frame.timestamp_sec,
                                 "image_path": relative, "image_sha256": _sha(frame.jpeg)})
            report["direction_checks"].append({
                "window_id": window["window_id"], "source_window_ids": window["source_window_ids"],
                "episode_id": window["episode_id"], "chunk_index": window["chunk_index"],
                "start_sec": window["start_sec"], "end_sec": window["end_sec"],
                "max_anchor_gap_sec": window["max_anchor_gap_sec"], "evidence": evidence,
                "status": "not_started", "attempted": False, "result": {"counter_direction": "uncertain"},
                "raw_banknote_motion": None, "calls": [], "errors": [],
                "feature_support": dict(DIRECTION_SUPPORT),
            })
        save()
        check_deadline()
        if selection["windows"]:
            client = LocalModelClient(base_url, timeout=min(20.0, deadline-time.perf_counter()))
            observer = DirectionObserver(client)
            monitor = GpuMonitor()
            with monitor:
                try:
                    report["runtime"] = client.server_info()
                    if report["runtime"].get("model_path"):
                        report["model_label"] = Path(report["runtime"]["model_path"]).name
                except Exception as error:
                    report["runtime_metadata_error"] = str(error)
                save()
                for row, window in zip(report["direction_checks"], selection["windows"]):
                    check_deadline()
                    row.update(attempted=True, status="running")
                    save()
                    try:
                        result = observer.check([frames[i] for i in window["frame_indices"]],
                            window_id=window["window_id"], counter_box=window["object_bbox_normalized"],
                            deadline=deadline)
                        row.update(result)
                        if time.perf_counter() >= deadline:
                            row["status"] = "partial"
                            row["result"] = {"counter_direction": "uncertain"}
                            row["errors"].append({"type": "ObservationDeadline", "message": "Response completed after pass deadline"})
                    except Exception as error:
                        row.update(status="partial", result={"counter_direction": "uncertain"})
                        row["errors"].append({"type": type(error).__name__, "message": str(error)})
                    save()
        report["status"] = "partial" if source["status"] == "partial" or any(
            row["status"] != "ok" for row in report["direction_checks"]) else "completed"
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
        if report["budget_exceeded"] and report["status"] == "completed":
            report["status"] = "partial"
        try:
            report["source_report_unchanged"] = source_path.read_bytes() == source_bytes
        except OSError:
            report["source_report_unchanged"] = False
        if not report["source_report_unchanged"]:
            report["errors"].append({"type": "SourceChanged", "message": "Source report changed after its snapshot"})
            if report["status"] == "completed":
                report["status"] = "partial"
        checks = report["direction_checks"]
        report["selected_window_count"] = len(report.get("direction_selection", {}).get("windows", []))
        report["processed_window_count"] = sum(row["attempted"] for row in checks)
        report["all_selected_windows_processed"] = (
            "direction_selection" in report and all(row["status"] == "ok" for row in checks))
        report["request_count"] = sum(len(row["calls"]) for row in checks)
        report["gpu_samples"] = monitor.samples if monitor is not None else []
        report["peak_observed_device_memory_mib"] = max(
            (sample["memory_mib"] for sample in report["gpu_samples"]), default=None)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--reserve-context", action="store_true", help="Reserve before/after anchors; experimental")
    args = parser.parse_args()
    report = inspect_direction(args.source_report, args.out, base_url=args.base_url,
                               budget_sec=args.budget, max_windows=args.max_windows,
                               reserve_context=args.reserve_context)
    print(json.dumps({"report": str(args.out / "report.json"), "status": report["status"],
                      "selected": report["selected_window_count"], "processed": report["processed_window_count"],
                      "total_seconds": report["total_seconds"], "errors": report["errors"]}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
