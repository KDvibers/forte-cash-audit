"""Dataset provenance and split integrity without models or private videos."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from analyzer.bundle_dataset import export_dataset, file_sha, load_annotations


class BundleDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "labels.json"
        self.out = self.root / "dataset"
        self.splits = {"1": "train", "2": "valid", "3": "test"}
        self.rows = []
        for i in range(1, 4):
            video = self.root / f"v{i}.mov"
            video.write_bytes(f"synthetic-video-{i}".encode())
            image = self.root / f"i{i}.jpg"
            Image.new("RGB", (60, 40), (i*60, 40, 10)).save(image)
            self.rows.append({"id": f"frame{i}", "video_id": i,
                "source_video_path": video.name, "source_video_sha256": file_sha(video),
                "image_path": image.name, "image_sha256": file_sha(image), "width": 60, "height": 40,
                "timestamp_sec": 2., "roi_xyxy": [10, 5, 55, 35], "boxes_xyxy": [[20, 10, 30, 20]],
                "ignore_boxes_xyxy": [], "fully_annotated": True, "box_contexts": ["in_hand"]})

    def save(self):
        self.manifest.write_text(json.dumps({"schema_version": 1, "rows": self.rows}), encoding="utf-8")

    def export(self):
        self.save()
        return export_dataset([self.manifest], self.root, self.splits, self.out)

    def test_round_trip_crops_and_transforms_boxes_but_preserves_sources(self):
        source = (self.root / "i1.jpg").read_bytes()
        report = self.export()
        coco = json.loads((self.out / "train/_annotations.coco.json").read_text())
        self.assertEqual(coco["annotations"][0]["bbox"], [10, 5, 10, 10])
        with Image.open(self.out / "train/frame1.jpg") as image:
            self.assertEqual(image.size, (45, 30))
        self.assertEqual(source, (self.root / "i1.jpg").read_bytes())
        self.assertFalse(report["independent_jury_test"])

    def test_duplicate_video_under_different_names_cannot_cross_split(self):
        (self.root / "v3.mov").write_bytes((self.root / "v1.mov").read_bytes())
        self.rows[2]["source_video_sha256"] = self.rows[0]["source_video_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate source video"):
            self.export()
        self.assertFalse(self.out.exists())

    def test_duplicate_image_cannot_cross_split(self):
        (self.root / "i3.jpg").write_bytes((self.root / "i1.jpg").read_bytes())
        self.rows[2]["image_sha256"] = self.rows[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.export()

    def test_video_hash_and_image_hash_fail_before_output(self):
        for field in ("source_video_sha256", "image_sha256"):
            with self.subTest(field=field):
                original = self.rows[0][field]
                self.rows[0][field] = "0"*64
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.export()
                self.assertFalse(self.out.exists())
                self.rows[0][field] = original

    def test_ambiguous_train_row_excluded_but_eval_crowd_preserved(self):
        extra = deepcopy(self.rows[0]); extra["id"] = "train_uncertain"
        extra["ignore_boxes_xyxy"] = [[35, 20, 45, 30]]
        self.rows.append(extra)
        self.rows[2]["ignore_boxes_xyxy"] = [[35, 20, 45, 30]]
        report = self.export()
        self.assertEqual(report["excluded_rows"], [{"id": "train_uncertain", "reason": "ambiguous_training_region"}])
        coco = json.loads((self.out / "test/_annotations.coco.json").read_text())
        self.assertEqual([a["iscrowd"] for a in coco["annotations"]], [0, 1])

    def test_context_geometry_and_overlap_validation(self):
        for key, bad in [("box_contexts", []), ("boxes_xyxy", [[0, 0, 20, 20]]),
                         ("ignore_boxes_xyxy", [[22, 12, 28, 18]]), ("fully_annotated", False)]:
            with self.subTest(key=key):
                original = self.rows[0][key]; self.rows[0][key] = bad
                with self.assertRaises(ValueError):
                    self.export()
                self.rows[0][key] = original

    def test_existing_export_not_overwritten(self):
        self.export()
        before = (self.out / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(before, (self.out / "manifest.json").read_bytes())

    def test_unassigned_video_and_duplicate_annotation_ids_rejected(self):
        self.splits.pop("3")
        with self.assertRaises(ValueError):
            self.export()
        self.splits["3"] = "test"
        self.rows[2]["id"] = self.rows[0]["id"]
        with self.assertRaises(ValueError):
            self.export()


if __name__ == "__main__":
    unittest.main()
