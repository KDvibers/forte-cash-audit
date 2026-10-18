"""Observe sampled frames and build state changes; never issue violation verdicts."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .local_client import LocalModelClient
from .observations import (ImageObserver, ObservationDeadline, policy_fingerprint, FEATURE_SUPPORT,
                           CASH_SHORT_CIRCUIT_RULE)
from .refinement import coarse_indices, refinement_indices, rule_refinement_indices
from .smoke import GpuMonitor
from .timeline import build_transitions
from .events import build_observed_events
from .features import CASH_FEATURES
from .video import sample_video
from .motion import propose_motion_windows
from .actions import ActionObserver, ACTION_SUPPORT, action_policy_fingerprint
from .action_windows import select_action_windows, combine_action_motion, gate_cabinet_claim
from .counter import discover_counter
from .budget import coverage_order


def observe_video(video_path, out, *, base_url="http://127.0.0.1:8080", interval_sec=2.0,
                  max_frames=90, budget_sec=220.0, request_timeout_sec=20.0,
                  max_refinement_frames=24, cabinet_mode="context_crops", activity_base_url=None,
                  max_motion_windows=8, enable_actions=False, max_action_windows=16,
                  action_reserve_sec=0.0, skip_cash_when_person_present=False):
    """Save real full frames, strict observations and a bounded timeline locally.

    ``completed`` means all selected frames were processed without technical
    errors. It does not mean every action was seen or every answer is correct.
    The budget covers extraction and local analysis, excluding model startup,
    external queueing and video upload. Native decoding cannot be interrupted
    mid-call; budget exhaustion is checked again immediately afterwards.

    ``skip_cash_when_person_present`` is opt-in and scoped to the current
    absence-and-cash rule. Skipped cash remains uncertain; it cannot support
    other cash events. Cabinet checks are unaffected.
    """
    for name, value in (("budget_sec", budget_sec), ("request_timeout_sec", request_timeout_sec)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(interval_sec, bool) or not isinstance(interval_sec, (int, float)) or not math.isfinite(interval_sec) or interval_sec <= 0:
        raise ValueError("interval_sec must be finite and positive")
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError("max_frames must be an integer of at least two")
    if type(max_refinement_frames) is not int or max_refinement_frames < 0:
        raise ValueError("max_refinement_frames must be a nonnegative integer")
    if type(max_motion_windows) is not int or max_motion_windows < 0:
        raise ValueError("max_motion_windows must be a nonnegative integer")
    if type(max_action_windows) is not int or max_action_windows < 0:
        raise ValueError("max_action_windows must be a nonnegative integer")
    if type(enable_actions) is not bool or (enable_actions and activity_base_url is None):
        raise ValueError("Action checks require activity_base_url and a boolean enable_actions")
    if type(skip_cash_when_person_present) is not bool:
        raise ValueError("skip_cash_when_person_present must be boolean")
    if skip_cash_when_person_present and (activity_base_url is None or enable_actions):
        raise ValueError("Cash short-circuit requires activity observations and cannot be used with action checks")
    if cabinet_mode == "disabled" and enable_actions:
        raise ValueError("Disabled cabinet observations are intended for the states-only rule pass")
    if (type(action_reserve_sec) not in (int, float) or not math.isfinite(action_reserve_sec)
            or action_reserve_sec < 0 or action_reserve_sec >= budget_sec
            or (action_reserve_sec > 0 and (not enable_actions or max_action_windows == 0))):
        raise ValueError("action_reserve_sec must be nonnegative, below budget, and requires enabled action windows")
    destination = Path(out).expanduser().resolve()
    # Never mix previous frames/reports with a new video.
    destination.mkdir(parents=True, exist_ok=False)
    frames_dir = destination / "frames"
    frames_dir.mkdir()
    started = time.perf_counter()
    deadline = started+budget_sec
    report = {
        "schema_version": 1, "kind": "visual_observations_not_violation_verdict",
        "video_path": str(Path(video_path).resolve()), "started_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_fingerprint(
            cabinet_mode, cash_enabled=activity_base_url is not None,
            skip_cash_when_person_present=skip_cash_when_person_present), "status": "running",
        "settings": {"base_url": base_url, "interval_sec": interval_sec, "max_frames": max_frames,
                     "cabinet_mode": cabinet_mode, "activity_base_url": activity_base_url,
                     "budget_sec": budget_sec, "request_timeout_sec": request_timeout_sec,
                     "refinement_interval_sec": 0.5, "max_refinement_frames": max_refinement_frames,
                     "max_motion_windows": max_motion_windows, "enable_actions": enable_actions,
                     "max_action_windows": max_action_windows,
                     "action_reserve_sec": action_reserve_sec,
                     "skip_cash_when_person_present": skip_cash_when_person_present,
                     "counter_discovery_model_role": "activity_4b",
                     "cabinet_action_gate": "trusted_open_observation_inside_window_v1",
                     "expected_image_min_tokens": 1024, "expected_image_max_tokens": 1280},
        "observations": [], "transitions": [], "observed_events": [], "errors": [],
        "feature_support": FEATURE_SUPPORT if activity_base_url is not None else {},
        "limitations": [
            "Model observations can be wrong; no final violation decision is produced.",
            "Sparse sampling can miss short actions, including opening and closing between samples.",
            "Presence concerns the visible work area, not employee identity or the entire building.",
            "A closed cabinet door does not prove that its lock is engaged.",
            "Agreement between two crops is not independent evidence or calibrated confidence.",
            "Models can report a definite cabinet state even when its front is obscured.",
            "Expected image token settings are not verified through the server metadata endpoint.",
            "Timing excludes model startup, external queueing and upload; decoding is not preemptible.",
            "Cash locations do not establish a transfer, amount, authenticity or successful machine operation.",
            "Spatial agreement is a consistency check by one model, not independent or calibrated confidence.",
        ],
    }
    if skip_cash_when_person_present:
        report["settings"]["refinement_policy"] = "absence_and_cash_condition_v1"
        report["cash_observation_scope"] = {
            "rule_id": CASH_SHORT_CIRCUIT_RULE, "revision": 1,
            "skip_condition": "final_person_yes_without_person_error",
            "skipped_state": "uncertain", "cabinet_checks": "unchanged",
        }
        report["limitations"].append(
            "Cash checks are skipped after confirmed visible presence only for the current absence-and-cash rule; "
            "skipped cash stays uncertain and cannot support other cash events."
        )
    if enable_actions:
        report.update(action_checks=[], action_support=ACTION_SUPPORT,
                      action_policy_sha256=action_policy_fingerprint())
    report["budget_control"] = {
        "enabled": action_reserve_sec > 0,
        "coarse_order": "largest_time_gap_first" if action_reserve_sec > 0 else "chronological",
        "action_reserve_sec": action_reserve_sec,
        "refinement_cost_policy": "maximum_measured_coarse_frame_wall_time_times_1.2",
        "refinement_cost_estimate_sec": None,
        "skipped_refinement_indices": [],
        "limitation": "Reserve is an estimate, not a deadline or a guarantee that all action windows fit.",
    }

    def save():
        report["total_seconds"] = time.perf_counter()-started
        temp = destination / "report.json.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(destination / "report.json")

    def runtime_metadata(client):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise ObservationDeadline("Video budget exhausted before runtime metadata")
        previous_timeout = client.timeout
        client.timeout = min(request_timeout_sec, remaining)
        try:
            try:
                runtime = client.server_info()
            except Exception as error:
                if time.perf_counter() >= deadline:
                    raise ObservationDeadline("Runtime metadata request reached the video deadline") from error
                raise
            if time.perf_counter() >= deadline:
                raise ObservationDeadline("Runtime metadata response arrived after the video deadline")
            return runtime
        finally:
            client.timeout = previous_timeout

    monitor = GpuMonitor()
    try:
        client = LocalModelClient(base_url, timeout=request_timeout_sec)
        try:
            report["runtime"] = runtime_metadata(client)
            if report["runtime"].get("model_path"):
                report["model_label"] = Path(report["runtime"]["model_path"]).name
        except ObservationDeadline:
            raise
        except Exception as error:
            # Some compatible local servers do not expose /props.
            report["runtime_metadata_error"] = str(error)
        activity_client = None
        if activity_base_url is not None:
            activity_client = LocalModelClient(activity_base_url, timeout=request_timeout_sec)
            try:
                report["runtime_activity"] = runtime_metadata(activity_client)
                if report["runtime_activity"].get("model_path"):
                    report["activity_model_label"] = Path(report["runtime_activity"]["model_path"]).name
            except ObservationDeadline:
                raise
            except Exception as error:
                report["activity_runtime_metadata_error"] = str(error)
        observer = ImageObserver(client, cabinet_mode=cabinet_mode, activity_client=activity_client,
                                 skip_cash_when_person_present=skip_cash_when_person_present)
        with monitor:
            # Decode a denser pool once. Only coarse frames and selected short
            # windows are sent to the model; no repeat decode for each window.
            sampled = sample_video(video_path, max_frames=600, max_side=1280, interval_sec=0.5,
                                   deadline=deadline)
            frames = sampled["frames"]
            report.update({key: value for key, value in sampled.items() if key != "frames"})
            report["available_frame_count"] = len(frames)
            report["dense_max_sample_gap_sec"] = sampled["max_sample_gap_sec"]
            selected = coarse_indices(frames, interval_sec=interval_sec, max_frames=max_frames)
            report["coarse_frame_count"] = len(selected)
            report["selected_frame_count"] = len(selected)
            report["selected_timestamps"] = [frames[i].timestamp_sec for i in selected]
            report["coarse_max_sample_gap_sec"] = max(
                (frames[right].timestamp_sec-frames[left].timestamp_sec
                 for left, right in zip(selected, selected[1:])), default=0.0)
            report["coarse_rule_max_gap_sec"] = 4.0
            report["coarse_coverage_blockers"] = (
                ["coarse_gap_exceeds_rule_max_gap"]
                if report["coarse_max_sample_gap_sec"] > report["coarse_rule_max_gap_sec"] else [])
            report["sampling_capped"] |= math.ceil(frames[-1].timestamp_sec/interval_sec)+1 > max_frames
            save()

            def process(index):
                nonlocal relative_image
                if time.perf_counter() >= deadline:
                    raise ObservationDeadline("Video budget exhausted before next frame")
                frame = frames[index]
                frame_id = f"frame_{index:04d}"
                relative_image = f"frames/{frame_id}.jpg"
                (destination / relative_image).write_bytes(frame.jpeg)
                observation = observer.observe(frame, frame_id, deadline=deadline)
                observation["image_path"] = relative_image
                observation["image_sha256"] = hashlib.sha256(frame.jpeg).hexdigest()
                report["observations"].append(observation)
                report["observations"].sort(key=lambda row: row["timestamp_sec"])
                save()

            relative_image = None
            frame_durations = []
            coarse_order = coverage_order(frames, selected) if action_reserve_sec > 0 else selected
            report["coarse_processing_order"] = list(coarse_order)
            for index in coarse_order:
                frame_started = time.perf_counter()
                process(index)
                frame_durations.append(time.perf_counter()-frame_started)
            frame_cost = max(frame_durations, default=0.0)*1.2
            report["budget_control"]["refinement_cost_estimate_sec"] = frame_cost
            refinement_selector = rule_refinement_indices if skip_cash_when_person_present else refinement_indices
            refinement = refinement_selector(frames, report["observations"], max_new_frames=max_refinement_frames)
            report["refinement"] = refinement
            motion = propose_motion_windows(frames, report["observations"], max_windows=max_motion_windows,
                                            max_refinement_frames=max_refinement_frames, observed_indices=selected)
            report["motion"] = motion
            counter_box = None
            if enable_actions:
                discovery = discover_counter(frames, selected, activity_client, activity_client, deadline=deadline)
                discovery["model_role"] = "activity_4b"
                report["counter_discovery"] = discovery
                counter_motion = propose_motion_windows(frames, discovery["observations"],
                    max_windows=max_action_windows, max_refinement_frames=0, roi_padding=.4,
                    max_window_sec=3.0, context_sec=.5)
                report["counter_motion"] = counter_motion
                counter_box = counter_motion["rois"]["desk"].get("union_bbox_normalized")
            # Preserve a single global refinement budget. Motion has priority so
            # identical coarse endpoints cannot hide an entire short cycle.
            combined = list(motion["refinement"]["selection_order"])
            refinement_order = refinement.get("selection_order", refinement["indices"])
            combined.extend(i for i in refinement_order if i not in combined)
            added = combined[:max_refinement_frames]
            report["combined_refinement"] = {"indices": sorted(added), "candidate_count": len(combined),
                "capped": len(combined) > max_refinement_frames,
                "state_proposals_dropped": [i for i in refinement["indices"] if i not in added]}
            selected = sorted(set(selected) | set(added))
            report["selected_frame_count"] = len(selected)
            report["selected_timestamps"] = [frames[i].timestamp_sec for i in selected]
            save()
            for index in added:
                if action_reserve_sec > 0 and deadline-time.perf_counter() < action_reserve_sec+frame_cost:
                    # Keep the skipped frame in the planned selection. A partial
                    # observation pass must not look like complete coverage.
                    report["budget_control"]["skipped_refinement_indices"].append(index)
                    save()
                    continue
                process(index)
            if enable_actions:
                action_observer = ActionObserver(client, activity_client)
                action_pool = combine_action_motion(motion, counter_motion, frames)
                selection = select_action_windows(action_pool, report["observations"], max_windows=max_action_windows)
                report["action_selection"] = selection
                save()
                for number, window in enumerate(selection["windows"]):
                    if time.perf_counter() >= deadline:
                        raise ObservationDeadline("Video budget exhausted before action window")
                    evidence = []
                    for index in window["frame_indices"]:
                        frame = frames[index]
                        path = f"frames/frame_{index:04d}.jpg"
                        (destination / path).write_bytes(frame.jpeg)
                        evidence.append({"timestamp_sec": frame.timestamp_sec, "image_path": path,
                                         "image_sha256": hashlib.sha256(frame.jpeg).hexdigest()})
                    check = action_observer.check([frames[i] for i in window["frame_indices"]],
                        region=window["roi"], window_id=f"action_{number:04d}", deadline=deadline,
                        counter_box=counter_box if window["roi"] == "desk" else None)
                    check = gate_cabinet_claim(check, report["observations"])
                    check.update(evidence=evidence, motion_group_id=window["group_id"],
                                 selection_reason=window["reason"])
                    report["action_checks"].append(check)
                    save()
            report["status"] = "partial" if any(x["errors"] for x in report["observations"]) else "completed"
            if any(x["status"] != "ok" for x in report.get("action_checks", [])):
                report["status"] = "partial"
            if report.get("counter_discovery", {}).get("status", "ok") != "ok":
                report["status"] = "partial"
            if report["budget_control"]["skipped_refinement_indices"]:
                report["status"] = "partial"
                report["errors"].append({"type": "RefinementBudgetReserved",
                    "message": "Some planned refinement frames were skipped to preserve time for action checks"})
    except (ObservationDeadline, TimeoutError) as error:
        report["status"] = "partial"
        partial_record = getattr(error, "partial_record", None)
        if partial_record is not None:
            partial_record["image_path"] = relative_image
            partial_record["image_sha256"] = hashlib.sha256((destination / relative_image).read_bytes()).hexdigest()
            report["observations"].append(partial_record)
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        # Gaps and uncertain states are deliberately not bridged by the timeline.
        report["observations"].sort(key=lambda row: row["timestamp_sec"])
        report["transitions"] = build_transitions(report["observations"], max_gap_sec=4.0)
        report["observed_events"] = build_observed_events(report["observations"], report["transitions"])
        observed_times = [row["timestamp_sec"] for row in report["observations"]]
        report["max_sample_gap_sec"] = max((b-a for a, b in zip(observed_times, observed_times[1:])), default=0.0)
        report["processed_frame_count"] = len(report["observations"])
        report["all_selected_frames_processed"] = (
            len(report["observations"]) == report.get("selected_frame_count", -1)
            and all(row["status"] == "ok" for row in report["observations"])
        )
        report["budget_exceeded"] = time.perf_counter() > deadline
        if report["budget_exceeded"] and report["status"] == "completed":
            report["status"] = "partial"
            report["errors"].append({"type": "ObservationDeadline", "message": "Overall video budget exceeded"})
        report["peak_observed_device_memory_mib"] = max((s["memory_mib"] for s in monitor.samples), default=None)
        report["request_count"] = sum(len(row["calls"]) for row in report["observations"])
        report["request_count"] += sum(len(row["calls"]) for row in report.get("action_checks", []))
        report["request_count"] += len(report.get("counter_discovery", {}).get("calls", []))
        if enable_actions:
            report["all_selected_action_windows_processed"] = (
                "action_selection" in report and len(report["action_checks"]) == len(report["action_selection"]["windows"])
                and all(row["status"] == "ok" for row in report["action_checks"]))
        report["uncertain_counts"] = {
            field: sum(row["states"].get(field) == "uncertain" for row in report["observations"])
            for field in (("person_in_work_area", "cabinet_door") + (CASH_FEATURES if activity_base_url is not None else ()))
        }
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New directory, usually under ignored runs/")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--activity-base-url", help="Second localhost model for people; enables worktop cash observations on the primary model")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--budget", type=float, default=220.0)
    parser.add_argument("--max-refinement-frames", type=int, default=24)
    parser.add_argument("--max-motion-windows", type=int, default=8, help="CPU motion proposals; zero disables")
    parser.add_argument("--actions", action="store_true", help="Review-only three-frame action checks; requires activity server")
    parser.add_argument("--max-action-windows", type=int, default=16)
    parser.add_argument("--action-reserve", type=float, default=0.0,
                        help="Experimental seconds reserved from optional refinement for action checks; skipped frames remain partial")
    parser.add_argument("--cabinet-mode", choices=["context_crops", "full_frame", "disabled"], default="context_crops")
    args = parser.parse_args()
    report = observe_video(args.video, args.out, base_url=args.base_url, interval_sec=args.interval,
                           max_frames=args.max_frames, budget_sec=args.budget,
                           max_refinement_frames=args.max_refinement_frames, cabinet_mode=args.cabinet_mode,
                           activity_base_url=args.activity_base_url, max_motion_windows=args.max_motion_windows,
                           enable_actions=args.actions, max_action_windows=args.max_action_windows,
                           action_reserve_sec=args.action_reserve)
    print(json.dumps({"report": str(args.out / "report.json"), "status": report["status"],
                      "frames": report["processed_frame_count"], "transitions": len(report["transitions"]),
                      "total_seconds": round(report["total_seconds"], 2), "errors": report["errors"]}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
