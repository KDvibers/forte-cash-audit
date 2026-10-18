"""Dense local detector replay on whole videos with explicitly manual ROIs.

This is visual review material, not action ground truth or production evidence.
No audio is decoded. Annotated preview MP4 timing is approximate; the JSON and
drawn timestamps retain real source times. A missing box is not absence of cash.
"""

import argparse
from fractions import Fraction
import html
from io import BytesIO
import json
import math
from pathlib import Path
import time

import av
from PIL import Image, ImageDraw, ImageFont

from analyzer.bundle_dataset import file_sha
from analyzer.train_bundle_detector import cash_predictions, checked_resolution, detector_model, offline_runtime, write_json
from analyzer.video import sample_video


def validate_plan(plan):
    videos = plan.get("videos")
    if not isinstance(videos, list) or not 1 <= len(videos) <= 12:
        raise ValueError("Expected 1..12 explicitly selected videos")
    seen = set()
    for row in videos:
        if type(row.get("video_id")) is not int or row["video_id"] in seen:
            raise ValueError("Video IDs must be unique integers")
        seen.add(row["video_id"])
        roi = row.get("roi_xyxy")
        if (not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi)
                or not 0 <= roi[0] < roi[2] or not 0 <= roi[1] < roi[3]):
            raise ValueError("Expected a manual integer ROI")
        if not isinstance(row.get("source_video_sha256"), str) or len(row["source_video_sha256"]) != 64:
            raise ValueError("Every source video requires its frozen SHA256")
        source = Path(row["source_video_path"]).resolve(strict=True)
        if file_sha(source) != row["source_video_sha256"]:
            raise ValueError("Video hash differs from the frozen replay plan")
    return videos


def replay(args):
    resolution = checked_resolution(args)
    if not math.isfinite(args.interval) or not .05 <= args.interval <= 1:
        raise ValueError("Sampling interval must be in .05..1 seconds")
    videos = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    weights = args.weights.resolve(strict=True)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {"kind": "whole_video_manual_roi_detector_replay_not_actions", "status": "starting",
              "weights_sha256": file_sha(weights), "plan_sha256": file_sha(args.plan),
              "interval_sec": args.interval, "score_threshold": .5, "resolution": resolution, "videos": [],
              "limits": ["Manual workspace ROI, not automatic scene localization.",
                         "Unannotated dense frames do not provide precision or recall.",
                         "Preview overlays are not original API evidence; JSON times are real."]}
    started = time.perf_counter()
    write_json(destination/"report.json", report)
    try:
        torch, model_class = offline_runtime()
        model = detector_model(model_class, weights, resolution)
        if model.class_names != {1: "banknote_bundle"}:
            raise ValueError("Expected the locally trained cash checkpoint")
        for video in videos:
            folder = destination/str(video["video_id"])
            folder.mkdir()
            source = Path(video["source_video_path"]).resolve(strict=True)
            sampled = sample_video(source, max_frames=2500, max_side=4096, interval_sec=args.interval)
            if sampled["sampling_capped"]:
                raise ValueError("Video exceeds the declared replay sampling budget")
            rows = []
            output = av.open(str(folder/"preview.mp4"), mode="w")
            stream = output.add_stream("libx264", rate=Fraction(1)/Fraction(str(args.interval)))
            first = sampled["frames"][0]
            roi = video["roi_xyxy"]
            # Output has a small header and even dimensions for yuv420p.
            stream.width, stream.height = first.width+first.width % 2, first.height+first.height % 2+30
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "fast"}
            try:
                for index, frame in enumerate(sampled["frames"]):
                    if (frame.width, frame.height) != (first.width, first.height) or roi[2] > frame.width or roi[3] > frame.height:
                        raise ValueError("Manual ROI or source geometry changed")
                    source_jpeg = folder/f"{index:04d}.jpg"
                    source_jpeg.write_bytes(frame.jpeg)
                    with Image.open(BytesIO(frame.jpeg)) as image:
                        picture = image.convert("RGB")
                    detections = model.predict(picture.crop(roi), threshold=.05)
                    boxes, scores = cash_predictions(detections, roi)
                    row = {"index": index, "timestamp_sec": frame.timestamp_sec,
                           "jpeg": source_jpeg.name, "jpeg_sha256": file_sha(source_jpeg),
                           "boxes_xyxy": boxes, "scores": scores}
                    rows.append(row)
                    preview = Image.new("RGB", (stream.width, stream.height), "#111820")
                    preview.paste(picture, (0, 30))
                    draw = ImageDraw.Draw(preview)
                    font = ImageFont.load_default(size=13)
                    draw.text((8, 7), f"Video {video['video_id']} | SOURCE {frame.timestamp_sec:.3f} s | boxes != actions", font=font, fill="white")
                    draw.rectangle((roi[0],roi[1]+30,roi[2],roi[3]+30), outline="cyan", width=1)
                    for box, score in zip(boxes, scores):
                        if score < .5:
                            continue
                        draw.rectangle((box[0],box[1]+30,box[2],box[3]+30), outline="#ff4b4b", width=2)
                        draw.text((box[0], max(30,box[1]+16)), f"{score:.2f}", font=font, fill="#ff4b4b")
                    encoded = av.VideoFrame.from_image(preview)
                    for packet in stream.encode(encoded):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            finally:
                output.close()
            if file_sha(source) != video["source_video_sha256"]:
                raise ValueError("Source video changed during replay")
            item = {**video, "frames": rows, "duration_sec": sampled["duration_sec"],
                    "max_sample_gap_sec": sampled["max_sample_gap_sec"],
                    "sampling_capped": sampled["sampling_capped"], "preview": f"{video['video_id']}/preview.mp4"}
            write_json(folder/"predictions.json", item)
            report["videos"].append({k:v for k,v in item.items() if k != "frames"} | {"frame_count": len(rows)})
            write_json(destination/"report.json", report)
        report["status"] = "completed"
        cards = [f'<section><h2>Видео {r["video_id"]}</h2><p>{r["frame_count"]} кадров; максимальный промежуток {r["max_sample_gap_sec"]:.3f} с.</p><video controls preload="metadata" src="{html.escape(r["preview"])}" style="max-width:100%;max-height:700px"></video><p><a href="{r["video_id"]}/predictions.json">Исходные времена и рамки</a></p></section>' for r in report["videos"]]
        (destination/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Проверка детектора на видео</title><style>body{background:#131923;color:#eef;font:16px system-ui;max-width:1100px;margin:auto;padding:24px}a{color:#8df}section{display:inline-block;vertical-align:top;max-width:48%;margin:1%}</style><h1>Частый просмотр видео с рамками</h1><p>Область рабочего места выбрана вручную. Это визуальная проверка; рамки не подтверждают действия или нарушения. В заголовке каждого кадра — настоящее время источника. Звука нет.</p>'+''.join(cards), encoding="utf-8")
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        write_json(destination/"report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.1)
    parser.add_argument("--resolution", type=int, choices=(384, 512), default=384)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
