"""Object evaluation stays honest when inference or labels are incomplete."""

from copy import deepcopy
import math
import unittest

from analyzer.bundle_metrics import evaluate_detections


def row(identifier="a", boxes=None, contexts=None, ignored=None, video=1):
    boxes = [[10, 10, 20, 20]] if boxes is None else boxes
    return {"id": identifier, "video_id": video, "roi_xyxy": [0, 0, 100, 100],
            "boxes_xyxy": boxes, "box_contexts": ["in_hand"] * len(boxes) if contexts is None else contexts,
            "ignore_boxes_xyxy": [] if ignored is None else ignored, "fully_annotated": True}


def prediction(identifier="a", boxes=None, scores=None):
    boxes = [[10, 10, 20, 20]] if boxes is None else boxes
    return {"id": identifier, "boxes_xyxy": boxes, "scores": [.8] * len(boxes) if scores is None else scores}


class BundleMetricsTests(unittest.TestCase):
    def test_exact_match_and_context_recall(self):
        result = evaluate_detections([row()], [prediction()])
        self.assertEqual((result["totals"]["tp"], result["totals"]["fp"], result["totals"]["fn"]), (1, 0, 0))
        self.assertEqual(result["totals"]["precision"], 1)
        self.assertEqual(result["by_context"]["in_hand"]["recall"], 1)
        self.assertIsNone(result["by_context"]["in_tray"]["recall"])

    def test_highest_score_matches_first_and_duplicate_is_false_positive(self):
        result = evaluate_detections([row()], [prediction(boxes=[[10, 10, 20, 20]] * 2, scores=[.6, .9])])
        self.assertEqual(result["rows"][0]["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["rows"][0]["false_positive_indices"], [0])
        self.assertEqual(result["totals"]["precision"], .5)

    def test_matching_chooses_best_available_iou_and_cannot_match_one_pred_twice(self):
        r = row(boxes=[[10, 10, 20, 20], [12, 10, 22, 20]], contexts=["in_hand", "in_tray"])
        result = evaluate_detections([r], [prediction(boxes=[[12, 10, 22, 20]])])
        self.assertEqual(result["rows"][0]["matches"][0]["ground_truth_index"], 1)
        self.assertEqual(result["rows"][0]["false_negative_indices"], [0])
        self.assertEqual(result["by_context"]["in_hand"]["recall"], 0)
        self.assertEqual(result["by_context"]["in_tray"]["recall"], 1)

    def test_iou_and_score_thresholds_are_inclusive(self):
        result = evaluate_detections([row()], [prediction(boxes=[[10, 10, 30, 20]], scores=[.5])])
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["rows"][0]["matches"][0]["iou"], .5)

    def test_below_threshold_is_false_negative_not_false_positive(self):
        result = evaluate_detections([row()], [prediction(scores=[.4999])])
        self.assertEqual(result["totals"]["below_threshold_predictions"], 1)
        self.assertEqual(result["totals"]["fn"], 1)
        self.assertEqual(result["totals"]["fp"], 0)
        self.assertIsNone(result["totals"]["precision"])
        self.assertEqual(result["totals"]["recall"], 0)

    def test_successful_empty_prediction_is_a_measured_miss(self):
        result = evaluate_detections([row()], [prediction(boxes=[])])
        self.assertEqual(result["totals"]["fn"], 1)
        self.assertIsNone(result["totals"]["precision"])

    def test_negative_frame_has_no_fabricated_recall(self):
        result = evaluate_detections([row(boxes=[])], [prediction()])
        self.assertEqual(result["totals"]["fp"], 1)
        self.assertEqual(result["totals"]["precision"], 0)
        self.assertIsNone(result["totals"]["recall"])
        empty = evaluate_detections([row(boxes=[])], [prediction(boxes=[])])
        self.assertIsNone(empty["totals"]["precision"])
        self.assertIsNone(empty["totals"]["recall"])

    def test_ignore_uses_prediction_area_not_iou(self):
        r = row(boxes=[], ignored=[[0, 0, 50, 50]])
        result = evaluate_detections([r], [prediction(boxes=[[40, 10, 60, 20]])])
        self.assertEqual(result["totals"]["ignored_predictions"], 1)
        self.assertEqual(result["totals"]["fp"], 0)
        self.assertIsNone(result["totals"]["precision"])

    def test_ignore_threshold_is_not_summed_across_separate_regions(self):
        r = row(boxes=[], ignored=[[10, 10, 14, 20], [16, 10, 20, 20]])
        result = evaluate_detections([r], [prediction()])
        self.assertEqual(result["totals"]["fp"], 1)

    def test_real_match_and_duplicate_cannot_be_hidden_by_ignore(self):
        r = row(ignored=[[0, 0, 50, 50]])
        result = evaluate_detections([r], [prediction(boxes=[[10, 10, 20, 20]] * 2)])
        self.assertEqual(result["totals"]["tp"], 1)
        self.assertEqual(result["totals"]["fp"], 1)
        self.assertEqual(result["totals"]["ignored_predictions"], 0)

    def test_video_and_context_counts_use_gt_contexts(self):
        rows = [row(video=14), row("b", [[30, 30, 40, 40]], ["in_tray"], video=35),
                row("c", [[60, 60, 70, 70]], ["on_table"], video=35)]
        preds = [prediction(), prediction("c", boxes=[]), prediction("b", boxes=[[30, 30, 40, 40], [0, 0, 5, 5]])]
        result = evaluate_detections(rows, preds)
        self.assertEqual((result["totals"]["tp"], result["totals"]["fp"], result["totals"]["fn"]), (2, 1, 1))
        self.assertAlmostEqual(result["totals"]["precision"], 2 / 3)
        self.assertEqual(result["by_video"]["35"]["recall"], .5)
        self.assertEqual(result["by_context"]["on_table"]["fn"], 1)
        self.assertEqual(sum(v["tp"] for v in result["by_context"].values()), 2)

    def test_full_image_coordinates_are_not_shifted_by_roi(self):
        r = row(boxes=[[60, 60, 70, 70]])
        r["roi_xyxy"] = [50, 50, 100, 100]
        result = evaluate_detections([r], [prediction(boxes=[[60, 60, 70, 70]])])
        self.assertEqual(result["totals"]["tp"], 1)
        wrong = evaluate_detections([r], [prediction(boxes=[[10, 10, 20, 20]])])
        self.assertEqual((wrong["totals"]["fp"], wrong["totals"]["fn"]), (1, 1))

    def test_source_and_predictions_remain_unchanged(self):
        rows, preds = [row()], [prediction()]
        before = deepcopy((rows, preds))
        evaluate_detections(rows, preds)
        self.assertEqual((rows, preds), before)

    def test_missing_extra_and_duplicate_ids_rejected(self):
        cases = [([], []), ([row()], []), ([row()], [prediction(), prediction("b")]),
                 ([row()], [prediction(), prediction()]), ([row(), row()], [prediction()])]
        for rows, preds in cases:
            with self.subTest(rows=rows, preds=preds), self.assertRaises(ValueError):
                evaluate_detections(rows, preds)

    def test_failed_or_incomplete_inference_is_never_a_negative(self):
        for extra in ({"error": "timeout"}, {"error": ""}, {"errors": []}, {"status": "failed"},
                      {"status": "processing"}, {"success": False}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                evaluate_detections([row()], [{**prediction(boxes=[]), **extra}])
        for missing in ("scores", "boxes_xyxy"):
            p = prediction()
            del p[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                evaluate_detections([row()], [p])

    def test_nonfinite_invalid_boxes_and_scores_rejected_even_below_threshold(self):
        for invalid in ([10, 10, 10, 20], [20, 10, 10, 20], [-1, 0, 5, 5], [10, 10, math.inf, 20],
                        [10, math.nan, 20, 20], [10, 10, 20], [True, 10, 20, 20]):
            with self.subTest(box=invalid), self.assertRaises(ValueError):
                evaluate_detections([row()], [prediction(boxes=[invalid], scores=[.1])])
        for score in (math.nan, math.inf, -1, 1.1, True, "0.8"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                evaluate_detections([row()], [prediction(scores=[score])])
        with self.assertRaises(ValueError):
            evaluate_detections([row()], [prediction(scores=[])])

    def test_malformed_annotations_cannot_be_scored(self):
        for change in ({"fully_annotated": False}, {"video_id": None}, {"box_contexts": []},
                       {"box_contexts": ["invented"]}, {"roi_xyxy": [0, 0, 5, 5]},
                       {"boxes_xyxy": [[10, 10, math.inf, 20]]}, {"ignore_boxes_xyxy": [[2, 2, 2, 2]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_detections([{**row(), **change}], [prediction()])

    def test_invalid_thresholds_rejected(self):
        for name, value in (("score_threshold", math.nan), ("score_threshold", 1.1),
                            ("iou_threshold", 0), ("iou_threshold", True), ("ignore_overlap", -.1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                evaluate_detections([row()], [prediction()], **{name: value})


if __name__ == "__main__":
    unittest.main()
