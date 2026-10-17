"""Check report attribution, visible limitations and usable evidence links."""

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from analyzer.render_review import render_review


class _Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.images = set(), [], []
        self.image_sizes = []
        self.state_count = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.links.append(attrs["href"][1:])
        if tag == "img":
            self.images.append(attrs.get("src"))
            self.image_sizes.append((attrs.get("width"), attrs.get("height")))
        if tag == "div" and attrs.get("class") == "state":
            self.state_count += 1


class RenderReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="forte-review-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "frames").mkdir()
        Image.new("RGB", (16, 16), "white").save(self.root / "frames/f0.jpg", "JPEG")
        self.report = {
            "model_label": "2B-local", "status": "completed",
            "observations": [{"frame_id": "f0", "timestamp_sec": 1.25, "status": "ok",
                              "image_path": "frames/f0.jpg", "states": {
                                  "person_in_work_area": "yes", "cabinet_door": "closed"}}],
        }

    def render(self):
        path = self.root / "report.json"
        path.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        page = render_review(path).read_text(encoding="utf-8")
        return page, _Page(page)

    def test_legacy_report_does_not_invent_missing_cash_features(self):
        page, parsed = self.render()
        self.assertEqual(parsed.state_count, 2)
        self.assertIn("Модель человека и шкафа", page)
        self.assertNotIn("Банкноты в руках", page)
        self.assertNotIn("События для проверки", page)
        self.assertNotIn("Модель присутствия и положения пачки", page)

    def test_second_model_is_attributed_to_person_not_cash(self):
        self.report["runtime_activity"] = {"model_path": r"C:\private\models\4B-local.gguf"}
        page, _ = self.render()
        self.assertIn("Модель шкафа и рабочего стола", page)
        self.assertIn("Модель присутствия и положения пачки", page)
        self.assertIn("2B-local", page)
        self.assertIn("4B-local.gguf", page)
        self.assertNotIn("C:\\private", page)
        self.assertNotIn("Модель денежных признаков", page)

    def test_declared_unverified_fields_and_reasons_remain_visible_and_escaped(self):
        self.report["feature_support"] = {
            "cash_in_hands": {"status": "not_validated", "reason": "Недостаточно надёжных наблюдений"},
            "notes_in_machine": {"status": "disabled", "reason": "Модель путает машину с банкнотами"},
        }
        self.report["field_limits"] = {"notes_in_machine": ["Нет вывода о пересчёте", "<script>bad</script>"]}
        page, parsed = self.render()
        self.assertIn("Банкноты в руках: Статус: Надёжность не подтверждена", page)
        self.assertIn("Банкноты в счётной машине: Статус: Признак отключён", page)
        self.assertIn("Модель путает машину с банкнотами", page)
        self.assertIn("Нет вывода о пересчёте", page)
        self.assertIn("&lt;script&gt;bad&lt;/script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertEqual(parsed.state_count, 2)

    def test_missing_second_model_metadata_does_not_reassign_person_to_first_model(self):
        self.report["settings"] = {"activity_base_url": "http://127.0.0.1:8081"}
        self.report["activity_runtime_metadata_error"] = "Metadata unavailable"
        page, _ = self.render()
        self.assertIn("Модель шкафа и рабочего стола", page)
        self.assertIn("Модель присутствия и положения пачки", page)
        self.assertIn("Не записана в отчёте", page)
        self.assertNotIn("Модель человека и шкафа", page)

    def test_cash_policy_explains_disagreement_without_claiming_action_detection(self):
        self.report["activity_model_label"] = "4B-local"
        self.report["observations"][0]["states"]["cash_on_worktop"] = "uncertain"
        page, _ = self.render()
        self.assertIn("лежит ли пачка на поверхности, а не удерживается над ней", page)
        self.assertIn("Если ответы не совпали, результат остаётся неопределённым", page)
        self.assertIn("Совпадение ответов тоже может быть ошибочным", page)
        self.assertIn("Не удалось определить", page)

    def test_frame_level_reason_does_not_turn_uncertainty_into_absence(self):
        row = self.report["observations"][0]
        row["states"]["cash_in_hands"] = "uncertain"
        row["field_limits"] = [{"feature": "cash_in_hands", "reason": "Признак пока не проверен"}]
        page, parsed = self.render()
        self.assertEqual(parsed.state_count, 3)
        self.assertIn("Не удалось определить", page)
        self.assertIn("Банкноты в руках: Причина: Признак пока не проверен", page)
        self.assertNotIn('class="known">Не видны', page)

    def test_events_warn_about_model_error_and_never_invent_evidence_anchors(self):
        self.report["observed_events"] = [{
            "type": "cash_appeared_on_worktop", "interval_start_sec": 0,
            "interval_end_sec": 1.25, "confirmed_at_sec": 2.5,
            "evidence_frame_ids": ["f0", "missing"],
        }]
        page, parsed = self.render()
        self.assertIn("События для проверки", page)
        self.assertIn("Модель может ошибаться в описании кадров", page)
        self.assertEqual(parsed.links, ["frame-0"])
        self.assertTrue(all(link in parsed.ids for link in parsed.links))
        self.assertEqual(parsed.images, ["frames/f0.jpg"])
        self.assertIn("missing (кадр недоступен)", page)

    def test_real_jpeg_dimensions_reserve_layout_before_lazy_loading(self):
        path = self.root / "frames/f0.jpg"
        Image.new("RGB", (320, 180), "white").save(path, "JPEG")
        original = path.read_bytes()
        _, parsed = self.render()
        self.assertEqual(parsed.image_sizes, [("320", "180")])
        self.assertEqual(path.read_bytes(), original)

    def test_jpeg_header_without_readable_image_is_marked_unavailable(self):
        (self.root / "frames/f0.jpg").write_bytes(b"\xff\xd8\xffnot-a-real-jpeg")
        page, parsed = self.render()
        self.assertEqual(parsed.images, [])
        self.assertIn("JPEG недоступен", page)


if __name__ == "__main__":
    unittest.main()
