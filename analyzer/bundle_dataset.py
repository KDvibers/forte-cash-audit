"""Export hash-verified local cash-object annotations for a detector experiment.

No labels are inferred here. Cropped training inputs are not API evidence.
Whole source videos (including exact duplicates) must stay in one split.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a local workspace path")
    path = (root / value).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file():
        raise ValueError("Expected a local file")
    return path


def _box(value, bounds):
    if (not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value)
            or not bounds[0] <= value[0] < value[2] <= bounds[2]
            or not bounds[1] <= value[1] < value[3] <= bounds[3]):
        raise ValueError("Box must be a nonempty integer rectangle inside its bounds")
    return value


def load_annotations(manifests, workspace, split_by_video):
    """Validate all inputs before export; retain uncertainty as explicit regions."""
    root = Path(workspace).resolve(strict=True)
    splits = {str(k): v for k, v in split_by_video.items()}
    if not splits or any(v not in {"train", "valid", "test"} for v in splits.values()):
        raise ValueError("Split names must be train, valid or test")
    rows, ids, image_splits, video_splits, video_ids, hashes = [], set(), {}, {}, {}, {}
    for manifest in manifests:
        path = Path(manifest).resolve(strict=True)
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1 or not isinstance(data.get("rows"), list):
            raise ValueError("Expected annotation schema 1 with rows")
        for original in data["rows"]:
            row = dict(original)
            identifier = row.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in ids:
                raise ValueError("Annotation IDs must be unique safe names")
            ids.add(identifier)
            video_id = str(row.get("video_id"))
            if video_id not in splits:
                raise ValueError("Every annotated video requires a predeclared split")
            split = splits[video_id]
            video = _path(root, row.get("source_video_path"))
            if video not in hashes:
                hashes[video] = file_sha(video)
            digest = hashes[video]
            if row.get("source_video_sha256") != digest:
                raise ValueError("Source video hash mismatch")
            if video_splits.setdefault(digest, split) != split:
                raise ValueError("Duplicate source video crosses splits")
            if video_ids.setdefault(video_id, digest) != digest:
                raise ValueError("One video ID refers to different source bytes")
            picture = _path(root, row.get("image_path"))
            image_bytes = picture.read_bytes()
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            if row.get("image_sha256") != image_sha:
                raise ValueError("Image hash mismatch")
            if image_splits.setdefault(image_sha, split) != split:
                raise ValueError("Duplicate image crosses splits")
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format != "JPEG" or image.size != (row.get("width"), row.get("height")):
                    raise ValueError("Source JPEG dimensions differ from annotation")
                image.load()
            bounds = [0, 0, row["width"], row["height"]]
            roi = _box(row["roi_xyxy"], bounds)
            for name in ("boxes_xyxy", "ignore_boxes_xyxy"):
                if not isinstance(row.get(name), list):
                    raise ValueError("Annotation boxes must be lists")
                for box in row[name]:
                    _box(box, roi)
            if row.get("fully_annotated") is not True:
                raise ValueError("ROI must be fully reviewed; mark ambiguity with ignore regions")
            contexts = row.get("box_contexts", ["unclear"]*len(row["boxes_xyxy"]))
            if len(contexts) != len(row["boxes_xyxy"]) or any(c not in {"in_hand", "in_tray", "on_table", "unclear"} for c in contexts):
                raise ValueError("One allowed context must correspond to every positive box")
            for positive in row["boxes_xyxy"]:
                for ignored in row["ignore_boxes_xyxy"]:
                    if min(positive[2], ignored[2]) > max(positive[0], ignored[0]) and min(positive[3], ignored[3]) > max(positive[1], ignored[1]):
                        raise ValueError("Positive and ignored regions overlap")
            row.update(split=split, box_contexts=contexts, source_manifest=str(path),
                       source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                       resolved_image_path=str(picture), source_video_hash_verified=True)
            rows.append(row)
    return rows


def export_dataset(manifests, workspace, split_by_video, out):
    """Write COCO folders only after verification, never overwrite an export.

    Training rows with ambiguous regions are excluded rather than teaching those
    objects as background. Evaluation keeps them as crowd/ignore regions; custom
    metrics must also exclude unmatched detections substantially inside them.
    """
    rows = load_annotations(manifests, workspace, split_by_video)
    destination = Path(out).resolve()
    if destination.exists():
        raise FileExistsError("Use a new dataset output directory")
    included = [r for r in rows if not (r["split"] == "train" and r["ignore_boxes_xyxy"])]
    if any(not any(r["split"] == s for r in included) for s in ("train", "valid", "test")):
        raise ValueError("Every split needs at least one eligible row")
    if not any(r["boxes_xyxy"] for r in included if r["split"] == "train"):
        raise ValueError("Training requires positive annotations")
    destination.mkdir(parents=True)
    export_rows = []
    coco_sha256 = {}
    for split in ("train", "valid", "test"):
        directory = destination / split
        directory.mkdir()
        coco = {"info": {"description": "Local diagnostic cash-object dataset"}, "licenses": [],
                "categories": [{"id": 0, "name": "objects", "supercategory": "none"},
                               {"id": 1, "name": "banknote_bundle", "supercategory": "objects"}],
                "images": [], "annotations": []}
        for row in (r for r in included if r["split"] == split):
            data = Path(row["resolved_image_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
                raise ValueError("Image changed during export")
            with Image.open(BytesIO(data)) as image:
                crop = image.crop(row["roi_xyxy"])
                image_path = directory / (row["id"]+".jpg")
                crop.save(image_path, format="JPEG", quality=95)
                width, height = crop.size
            image_id = len(coco["images"])+1
            coco["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
            x, y = row["roi_xyxy"][:2]
            for ignored, boxes in ((False, row["boxes_xyxy"]), (True, row["ignore_boxes_xyxy"])):
                for box in boxes:
                    bbox = [box[0]-x, box[1]-y, box[2]-box[0], box[3]-box[1]]
                    coco["annotations"].append({"id": len(coco["annotations"])+1, "image_id": image_id,
                        "category_id": 1, "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": int(ignored),
                        "segmentation": [], "ignore": int(ignored)})
            export_rows.append({**row, "export_image_path": image_path.relative_to(destination).as_posix(),
                                "export_image_sha256": file_sha(image_path), "coco_image_id": image_id})
        annotation_path = directory / "_annotations.coco.json"
        annotation_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        coco_sha256[split] = file_sha(annotation_path)
    report = {"schema_version": 1, "kind": "local_cash_detector_experiment", "rows": export_rows,
        "coco_sha256": coco_sha256,
        "excluded_rows": [{"id": r["id"], "reason": "ambiguous_training_region"} for r in rows if r not in included],
        "split_by_video": {str(k): v for k, v in split_by_video.items()}, "independent_jury_test": False,
        "roi_source": "manual_annotation_context_not_automatic_localization",
        "limits": ["Source videos previously reviewed; split is independent of new training only.",
                   "Object detection does not establish motion, authenticity, amount or violation.",
                   "Export crops are model inputs; original source JPEGs remain evidence."]}
    (destination / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = export_dataset(args.manifest, args.workspace, json.loads(args.split_plan.read_text(encoding="utf-8")), args.out)
    print(json.dumps({"rows": len(report["rows"]), "excluded": len(report["excluded_rows"])}))


if __name__ == "__main__":
    main()
