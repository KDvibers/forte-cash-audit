"""Run a local visual understanding check; this does not issue case verdicts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from .local_client import LocalModelClient, OBSERVATION_PROMPT, SYSTEM_PROMPT
from .video import sample_video


class GpuMonitor:
    """Coarse GPU memory samples; reports device-wide usage, not exact process peak."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                values = output.splitlines()[0].split(",")
                self.samples.append({"memory_mib": int(values[0].strip()), "utilization_percent": int(values[1].strip())})
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(0.75)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--single-image", action="store_true", help="Use the middle sampled frame only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = LocalModelClient(args.base_url)
    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + OBSERVATION_PROMPT).encode()).hexdigest()
    failed = False
    for index, video in enumerate(args.videos, start=1):
        started = time.perf_counter()
        report = {"video_path": str(video.resolve()), "started_utc": datetime.now(timezone.utc).isoformat(), "prompt_sha256": prompt_hash, "requested_frames": args.frames, "max_side": args.max_side, "purpose": "development_visual_smoke_not_independent_evaluation"}
        monitor = GpuMonitor()
        try:
            with monitor:
                sampled = sample_video(video, max_frames=args.frames, max_side=args.max_side)
                frames = sampled["frames"]
                if args.single_image:
                    frames = [frames[len(frames) // 2]]
                report.update({"duration_sec": sampled["duration_sec"], "extraction_seconds": sampled["extraction_seconds"], "frame_timestamps": [frame.timestamp_sec for frame in frames]})
                frames_dir = args.out / f"{index:02d}_frames"
                frames_dir.mkdir(exist_ok=True)
                for number, frame in enumerate(frames):
                    (frames_dir / f"{number:02d}_{frame.timestamp_sec:.3f}.jpg").write_bytes(frame.jpeg)
                report.update(client.describe(frames))
            report["status"] = "completed" if report["valid_json"] and report["finish_reason"] != "length" else "invalid_response"
            failed |= report["status"] != "completed"
        except Exception as error:
            report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            failed = True
        report["gpu_samples"] = monitor.samples
        report["peak_observed_device_memory_mib"] = max((sample["memory_mib"] for sample in monitor.samples), default=None)
        report["total_seconds"] = time.perf_counter() - started
        destination = args.out / f"{index:02d}_{video.stem}.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(destination), "status": report["status"], "total_seconds": round(report["total_seconds"], 2), "peak_device_mib": report.get("peak_observed_device_memory_mib"), "error": report.get("error")}, ensure_ascii=False), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
