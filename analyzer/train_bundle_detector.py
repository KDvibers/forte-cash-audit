"""Offline RF-DETR Nano experiment; never imported by the production service.

Requires the separate requirements-detector.txt environment and local weights.
The manually selected workspace crop is part of this diagnostic experiment.
Object boxes alone must not be converted into action or violation verdicts.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from PIL import Image

from analyzer.bundle_dataset import file_sha
from analyzer.bundle_metrics import evaluate_detections


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def checked_resolution(args):
    resolution = getattr(args, "resolution", 384)
    if type(resolution) is not int or resolution not in (384, 512):
        raise ValueError("The bounded Nano experiment supports resolution 384 or 512")
    return resolution


def detector_model(model_class, weights, resolution):
    """Set and verify the actual resize used by RF-DETR, including evaluation."""
    model = model_class(pretrain_weights=str(weights), device="cuda", resolution=resolution)
    if (model.model.resolution != resolution or model.model_config.resolution != resolution):
        raise ValueError("Loaded detector resolution differs from the requested experiment")
    return model


def offline_runtime():
    """Disable optional trackers/downloads and reject Python socket connections.

    This is a process-local guard, not a claim about a system-wide firewall.
    Packages and upstream weights must be downloaded before starting this CLI.
    """
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", WANDB_MODE="disabled", MPLBACKEND="Agg")

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError("Network access is disabled in the local detector experiment")

    sys.addaudithook(audit)
    import numpy as np
    import torch
    from rfdetr import RFDETRNano
    if importlib.metadata.version("rfdetr") != "1.4.3":
        raise RuntimeError("This experiment was checked against rfdetr==1.4.3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the detector environment")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Avoid oversubscribing this laptop with CPU data/preprocessing threads.
    torch.set_num_threads(4)
    return torch, RFDETRNano


def verified_dataset(directory):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "local_cash_detector_experiment":
        raise ValueError("Expected a verified bundle_dataset export")
    for split in ("train", "valid", "test"):
        annotations = root/split/"_annotations.coco.json"
        if file_sha(annotations) != manifest.get("coco_sha256", {}).get(split):
            raise ValueError("COCO annotation hash mismatch")
    for row in manifest["rows"]:
        picture = (root / row["export_image_path"]).resolve(strict=True)
        picture.relative_to(root)
        if file_sha(picture) != row["export_image_sha256"]:
            raise ValueError("Export image hash mismatch")
    return root, manifest


def train(args):
    resolution = checked_resolution(args)
    dataset, manifest = verified_dataset(args.dataset)
    weights = args.weights.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(dataset_dir=str(dataset), output_dir=str(out), epochs=args.epochs,
                    batch_size=1, grad_accum_steps=2, num_workers=0,
                    multi_scale=False, expanded_scales=False, use_ema=False,
                    early_stopping=False, tensorboard=False, wandb=False, run_test=False,
                    lr=1e-4, lr_encoder=1.5e-4, lr_drop=100, checkpoint_interval=10)
    report = dict(status="starting", created_utc=datetime.now(timezone.utc).isoformat(),
                  settings=settings, seed=42, model="RFDETRNano", resolution=resolution,
                  initial_weights_sha256=file_sha(weights), dataset_manifest_sha256=file_sha(dataset/"manifest.json"),
                  split_by_video=manifest["split_by_video"],
                  checkpoint_selection="best validation COCO mAP; test is not run during training",
                  score_threshold=0.5, iou_threshold=0.5,
                  network_policy="Python socket connections/DNS/sendto denied; offline libraries; no trackers")
    write_json(out/"experiment.json", report)
    started = time.perf_counter()
    try:
        torch, model_class = offline_runtime()
        torch.cuda.reset_peak_memory_stats()
        report.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda,
                      packages={name: importlib.metadata.version(name) for name in
                                ("rfdetr", "torchvision", "transformers", "peft", "numpy", "Pillow")})
        model = detector_model(model_class, weights, resolution)
        report["status"] = "training"
        write_json(out/"experiment.json", report)
        model.train(**settings)
        checkpoint = out/"checkpoint_best_total.pth"
        if not checkpoint.is_file():
            raise RuntimeError("Training did not produce the selected checkpoint")
        report.update(status="completed", selected_checkpoint=checkpoint.name,
                      selected_checkpoint_sha256=file_sha(checkpoint),
                      peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        write_json(out/"experiment.json", report)


def cash_predictions(detections, roi):
    """Validate model arrays and restore crop boxes to full-image coordinates."""
    x, y, right, bottom = roi
    boxes, scores = [], []
    if not len(detections.xyxy) == len(detections.confidence) == len(detections.class_id):
        raise ValueError("Detection arrays have different lengths")
    for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
                or not box[0] < box[2] or not box[1] < box[3]
                or not math.isfinite(float(score)) or not 0 <= float(score) <= 1
                or not math.isfinite(float(class_id)) or int(class_id) != class_id):
            raise ValueError("Invalid detection coordinates, score or class")
        if int(class_id) != 1:
            continue
        # RF-DETR can regress slightly outside the crop; clip to input bounds.
        box = [max(x, min(right, float(box[0])+x)), max(y, min(bottom, float(box[1])+y)),
               max(x, min(right, float(box[2])+x)), max(y, min(bottom, float(box[3])+y))]
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(box)
            scores.append(float(score))
    return boxes, scores


def evaluate(args):
    resolution = checked_resolution(args)
    dataset, manifest = verified_dataset(args.dataset)
    weights = args.weights.resolve(strict=True)
    rows = [r for r in manifest["rows"] if r["split"] == args.split]
    if not rows:
        raise ValueError("Requested split is empty")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch, model_class = offline_runtime()
    started = time.perf_counter()
    model = detector_model(model_class, weights, resolution)
    if model.class_names != {1: "banknote_bundle"}:
        raise ValueError("Checkpoint does not contain the trained cash class")
    load_seconds = time.perf_counter()-started
    predictions = []
    for row in rows:
        with Image.open(dataset/row["export_image_path"]) as source:
            picture = source.convert("RGB")
        torch.cuda.synchronize()
        began = time.perf_counter()
        detections = model.predict(picture, threshold=0.05)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        boxes, scores = cash_predictions(detections, row["roi_xyxy"])
        predictions.append(dict(id=row["id"], boxes_xyxy=boxes, scores=scores, inference_seconds=elapsed))
        write_json(out/"predictions.partial.json", predictions)
    report = evaluate_detections(rows, predictions, score_threshold=0.5, iou_threshold=0.5)
    report.update(split=args.split, weights_sha256=file_sha(weights), model_load_seconds=load_seconds,
                  resolution=model.model.resolution,
                  dataset_manifest_sha256=file_sha(dataset/"manifest.json"),
                  elapsed_seconds=time.perf_counter()-started, independent_jury_test=False,
                  roi_source=manifest["roi_source"], limits=manifest["limits"])
    write_json(out/"predictions.json", predictions)
    write_json(out/"metrics.json", report)
    print(json.dumps({"totals": report["totals"], "by_context": report["by_context"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resolution", type=int, choices=(384, 512), default=384)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("epochs must be in 1..100 for this bounded experiment")
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
