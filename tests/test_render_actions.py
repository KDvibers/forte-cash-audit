"""Action review pages preserve evidence and never promote candidates to facts."""

import copy
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from PIL import Image

from analyzer.render_actions import render_actions


class _Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.images = []
        self.links = []
        self.tags = []
        self.attributes = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.tags.append(tag)
        self.attributes.extend(attrs)
        if tag == "img":
            self.images.append(attrs)
        if tag == "a":
            self.links.append(attrs.get("href", ""))


class RenderActionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="forte-actions-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve() / "run"
        (self.root / "frames").mkdir(parents=True)
        evidence = []
        for index, color in enumerate(("red", "green", "blue")):
            name = f"frames/f{index}.jpg"
            Image.new("RGB", (32 + index, 18), color).save(self.root / name, "JPEG")
            evidence.append({"timestamp_sec": index * .5, "image_path": name,
                             "image_sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()})
        self.report = {"action_checks": [{
            "window_id": "window-original", "region": "desk", "start_sec": 0, "end_sec": 1,
            "status": "ok", "result": {"interaction": "yes", "counter_direction": "uncertain"},
            "errors": [], "evidence": evidence,
        }]}

    @property
    def check(self):
        return self.report["action_checks"][0]

    def render(self):
        source = self.root / "report.json"
        source.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        output = render_actions(source)
        text = output.read_text(encoding="utf-8")
        return output, text, _Page(text)

    def test_original_three_jpegs_are_linked_without_rewriting_evidence(self):
        before = {path: path.read_bytes() for path in (self.root / "frames").iterdir()}
        output, text, page = self.render()
        self.assertEqual(output, self.root / "actions.html")
        self.assertEqual([row["src"] for row in page.images], [f"frames/f{i}.jpg" for i in range(3)])
        self.assertEqual(page.links, [row["src"] for row in page.images])
        self.assertEqual([row["width"] for row in page.images], ["32", "33", "34"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertIn("Кандидаты для ручной проверки", text)
        self.assertIn("Технический статус не означает, что наблюдение верно", text)
        self.assertIn("не вывод о нарушении", text)
        self.assertNotIn("script", page.tags)
        self.assertNotIn("link", page.tags)
        for url in page.links + [row["src"] for row in page.images]:
            self.assertFalse(urlsplit(url).scheme)
            self.assertTrue((self.root / unquote(url)).is_file())

    def test_overwriting_generated_html_does_not_change_report(self):
        output, _, _ = self.render()
        source = self.root / "report.json"
        original = source.read_bytes()
        output.write_text("old generated page", encoding="utf-8")
        self.assertEqual(render_actions(source), output)
        self.assertEqual(source.read_bytes(), original)
        self.assertIn("Кандидаты для ручной проверки", output.read_text(encoding="utf-8"))

    def test_direction_transfer_and_raw_calls_never_become_displayed_facts(self):
        self.check["result"]["counter_direction"] = "load-UNTRUSTED-DIRECTION"
        self.check["calls"] = [{"response": {"response_text": "UNTRUSTED-MODEL-TRANSFER"}}]
        cabinet = copy.deepcopy(self.check)
        cabinet.update(region="cabinet", window_id="cabinet-window")
        cabinet["result"] = {"cash_near_cabinet": "yes", "cash_into_cabinet": "UNTRUSTED-TRANSFER"}
        self.report["action_checks"].append(cabinet)
        _, text, _ = self.render()
        self.assertIn("Возможно видна пачка у шкафа", text)
        self.assertIn("Направление загрузки или извлечения не подтверждено", text)
        self.assertIn("Пересчёт денег не установлен", text)
        self.assertIn("Прямой перенос банкнот внутрь шкафа не подтверждён", text)
        self.assertNotIn("UNTRUSTED", text)

    def test_unknown_or_missing_interaction_is_not_absence(self):
        for state in ("uncertain", "invalid", None, False, [], {}):
            with self.subTest(state=state):
                self.check["result"]["interaction"] = state
                _, text, _ = self.render()
                self.assertIn("Не удалось определить", text)
                self.assertNotIn("Взаимодействие не наблюдается", text)

    def test_missing_evidence_is_explicit_and_never_replaced(self):
        self.check["evidence"] = self.check["evidence"][:1]
        _, text, page = self.render()
        self.assertEqual(len(page.images), 1)
        self.assertIn("в отчёте указано 1", text)
        self.assertIn("недостающие кадры не заменяются", text)

    def test_external_traversal_and_platform_paths_never_load(self):
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes((self.root / "frames/f0.jpg").read_bytes())
        paths = ["../outside.jpg", "frames/../../outside.jpg", str(outside),
                 "https://example.com/image.jpg", "//example.com/image.jpg", "data:image/jpeg;base64,AAAA",
                 "file:///C:/outside.jpg", r"C:\outside.jpg", r"\\host\share\outside.jpg",
                 "frames/f0.jpg:secret", "frames\\..\\..\\outside.jpg"]
        for path in paths:
            with self.subTest(path=path):
                self.check["evidence"] = [{**self.check["evidence"][0], "image_path": path}]
                _, text, page = self.render()
                self.assertEqual(page.images, [])
                self.assertEqual(page.links, [])
                self.assertIn("Кадр недоступен", text)

    def test_replaced_missing_hash_and_fake_jpeg_are_not_displayed(self):
        row = copy.deepcopy(self.check["evidence"][0])
        self.check["evidence"] = [row]
        for digest in (None, "x" * 64, "0" * 64, [], 42):
            with self.subTest(digest=digest):
                row["image_sha256"] = digest
                _, text, page = self.render()
                self.assertEqual(page.images, [])
                self.assertIn("SHA256", text)
        path = self.root / row["image_path"]
        Image.new("RGB", (32, 18), "red").save(path, "PNG")
        row["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _, text, page = self.render()
        self.assertEqual(page.images, [])
        self.assertIn("JPEG", text)

    def test_uppercase_hash_and_unicode_filename_are_supported(self):
        row = self.check["evidence"][0]
        target = self.root / "frames/кадр 0 & 1.jpg"
        (self.root / row["image_path"]).rename(target)
        row.update(image_path="frames/кадр 0 & 1.jpg", image_sha256=row["image_sha256"].upper())
        _, _, page = self.render()
        self.assertEqual(len(page.images), 3)
        self.assertEqual(unquote(page.images[0]["src"]), row["image_path"])
        self.assertIn("%20", page.images[0]["src"])

    def test_untrusted_strings_are_text_not_html_and_coverage_limits_are_visible(self):
        attack = '<script>alert("unsafe")</script><img src="https://bad" onerror="bad">'
        self.check["window_id"] = attack
        self.check["errors"] = [{"feature": attack, "message": attack}]
        self.report["motion"] = {"coverage": {"input_frame_count": 30, "selected_window_count": 1,
                                              "dropped_window_count": 2, "capped": True,
                                              "refinement_capped": True},
                                 "dropped_windows": [{"region": attack, "reason": attack,
                                                      "start_sec": float("nan"), "end_sec": -1}]}
        _, text, page = self.render()
        self.assertEqual(len(page.images), 3)
        self.assertNotIn("script", page.tags)
        self.assertNotIn("onerror", page.attributes)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("Лимит окон достигнут", text)
        self.assertIn("Лимит дополнительных кадров достигнут", text)
        self.assertIn("Пропущенные окна в списке: 1", text)
        self.assertIn("неизвестно", text)

    def test_empty_report_does_not_mean_no_actions(self):
        self.report = {}
        _, text, page = self.render()
        self.assertEqual(page.images, [])
        self.assertIn("Это не доказательство отсутствия действий", text)
        self.assertIn("Сведения о покрытии поиска движения не записаны", text)

    def test_action_selection_is_separate_from_frame_refinement_window_count(self):
        self.report["action_checks"] = [copy.deepcopy(self.check) for _ in range(16)]
        self.report["motion"] = {"coverage": {"selected_window_count": 8, "capped": True},
                                 "dropped_windows": [{"roi": "desk", "start_sec": 20, "end_sec": 22,
                                                      "reason": "window_cap"}]}
        self.report["action_selection"] = {"selected_count": 16, "candidate_count": 18, "capped": True,
                                            "dropped_windows": [{"roi": "cabinet", "start_sec": 30, "end_sec": 33,
                                                                 "reason": "action_window_cap"}]}
        _, text, _ = self.render()
        selection = text.split('id="action-selection"', 1)[1].split("</section>", 1)[0]
        motion = text.split('id="frame-motion"', 1)[1].split("</section>", 1)[0]
        self.assertIn("Записано результатов проверки: 16", selection)
        self.assertIn("Выбрано для проверки действий</dt><dd>16", selection)
        self.assertIn("Лимит проверок действий достигнут", selection)
        self.assertIn("30 с — 33 с", selection)
        self.assertNotIn("20 с — 22 с", selection)
        self.assertIn("Выбрано окон</dt><dd>8", motion)
        self.assertIn("Их число не равно числу проверок действий", motion)
        self.assertIn("20 с — 22 с", motion)
        self.assertNotIn("Не все выбранные окна имеют результат", selection)

    def test_unfinished_selected_windows_are_not_reported_as_completed_checks(self):
        self.report["action_selection"] = {"selected_count": 16, "candidate_count": 16,
                                            "capped": False, "dropped_windows": []}
        _, text, _ = self.render()
        selection = text.split('id="action-selection"', 1)[1].split("</section>", 1)[0]
        self.assertIn("Записано результатов проверки: 1", selection)
        self.assertIn("Не все выбранные окна имеют результат", selection)
        self.assertIn("Лимит выбора окон для проверки действий не достигнут", selection)

    def test_discovery_rejections_and_errors_are_visible_without_claiming_no_actions(self):
        self.report["counter_discovery"] = {
            "status": "partial", "coverage": {"selected_frame_count": 3, "validated_frame_count": 0,
                                                   "skipped_frame_count": 1, "budget_exhausted": True},
            "checks": [{"timestamp_sec": 0, "banknote_counter": "no", "object_visibility": "yes"},
                       {"timestamp_sec": 10, "banknote_counter": "uncertain", "object_visibility": "no"}],
            "errors": [{"feature": "counter_roi", "message": "<img src=x onerror=unsafe> timeout"}],
            "calls": [{"response": {"response_text": "RAW-COUNTER-CLAIM"}}],
        }
        _, text, page = self.render()
        discovery = text.split('id="counter-discovery"', 1)[1].split("</section>", 1)[0]
        self.assertIn("Кандидат отклонён", discovery)
        self.assertIn("Кандидат не найден или его положение неясно", discovery)
        self.assertIn("Время поиска счётчика закончилось", discovery)
        self.assertIn("Кадров не проверено</dt><dd>1", discovery)
        self.assertIn("не означает отсутствие действий", discovery)
        self.assertIn("&lt;img", discovery)
        self.assertNotIn("onerror", page.attributes)
        self.assertNotIn("RAW-COUNTER-CLAIM", text)

    def test_one_confirmed_device_does_not_imply_usable_motion_roi(self):
        self.report["counter_discovery"] = {
            "status": "ok", "coverage": {"validated_frame_count": 1},
            "checks": [{"timestamp_sec": 0, "banknote_counter": "yes"}],
        }
        self.report["counter_motion"] = {
            "rois": {"desk": {"bbox_normalized": None, "valid_box_count": 1, "accepted_box_count": 0,
                               "reason": "at_least_two_coarse_boxes_required"}},
            "coverage": {"image_decode_skipped_no_roi": True, "decoded_frame_count": 0},
        }
        _, text, _ = self.render()
        counter = text.split('id="counter-motion"', 1)[1].split("</section>", 1)[0]
        self.assertIn("Область счётчика для поиска движения не найдена", counter)
        self.assertIn("Это пробел покрытия, а не отсутствие действий", counter)
        self.assertIn("Нужны как минимум две подтверждённые области", counter)
        self.assertNotIn("Согласованная область счётчика найдена", counter)

    def test_counter_roi_coverage_exposes_agreement_outliers_and_missed_pairs(self):
        self.report["counter_motion"] = {
            "rois": {"desk": {"bbox_normalized": [100, 100, 800, 700], "valid_box_count": 3,
                               "accepted_box_count": 2, "outlier_box_count": 1,
                               "reason": "coarse_box_cluster_union"}},
            "coverage": {"selected_window_count": 16, "decoded_frame_count": 100,
                         "compared_pairs_by_roi": {"desk": 92}, "skipped_pairs_by_roi": {"desk": 7}},
            "frame_errors": [{"type": "ValueError", "message": "Unreadable JPEG"}],
        }
        _, text, _ = self.render()
        counter = text.split('id="counter-motion"', 1)[1].split("</section>", 1)[0]
        self.assertIn("Согласованная область счётчика найдена", counter)
        self.assertIn("Согласующихся обнаружений</dt><dd>2", counter)
        self.assertIn("Отброшено несогласующихся областей</dt><dd>1", counter)
        self.assertIn("Сравнено пар кадров у счётчика</dt><dd>92", counter)
        self.assertIn("Пропущено пар кадров у счётчика</dt><dd>7", counter)
        self.assertIn("Unreadable JPEG", counter)

    def test_selection_and_roi_untrusted_strings_stay_text(self):
        attack = '<script>unsafe()</script>'
        self.report["action_selection"] = {"selected_count": True, "candidate_count": -1,
                                            "state_conditioned": True, "capped": [],
                                            "dropped_windows": [{"roi": attack, "reason": attack}]}
        self.report["counter_motion"] = {"rois": {"desk": {"bbox_normalized": [0, 0, 0, 100],
                                                           "reason": attack}}}
        _, text, page = self.render()
        self.assertNotIn("script", page.tags)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("Выбрано для проверки действий</dt><dd>неизвестно", text)
        self.assertIn("это не доказательство переноса", text)
        self.assertIn("Область счётчика для поиска движения не найдена", text)

    def test_old_report_missing_new_sections_does_not_fabricate_coverage(self):
        _, text, _ = self.render()
        self.assertIn("План выбора окон не записан", text)
        self.assertIn("Отдельный поиск счётчика не записан", text)
        self.assertNotIn("Выбрано для проверки действий</dt><dd>0", text)
        self.assertNotIn("Согласованная область счётчика найдена", text)

    def test_malformed_structure_fails_without_overwriting_existing_page(self):
        output = self.root / "actions.html"
        for report in ([], {"action_checks": {}}, {"action_checks": [None]},
                       {"action_checks": [{"evidence": "invalid"}]}):
            with self.subTest(report=report):
                output.write_text("preserve", encoding="utf-8")
                self.report = report
                with self.assertRaises(ValueError):
                    self.render()
                self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_output_cannot_replace_input_named_actions_html(self):
        path = self.root / "actions.html"
        path.write_text(json.dumps(self.report), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            render_actions(path)
        self.assertEqual(path.read_bytes(), before)

    def test_image_symlink_outside_report_is_rejected(self):
        source = self.root.parent / "outside.jpg"
        source.write_bytes((self.root / "frames/f0.jpg").read_bytes())
        link = self.root / "frames/link.jpg"
        try:
            link.symlink_to(source)
        except OSError as error:
            self.skipTest(f"File symlinks unavailable: {error}")
        self.check["evidence"] = [{**self.check["evidence"][0], "image_path": "frames/link.jpg"}]
        _, _, page = self.render()
        self.assertEqual(page.images, [])

    def test_output_hardlink_is_rejected_without_overwriting_target(self):
        source = self.root.parent / "outside.txt"
        source.write_text("preserve", encoding="utf-8")
        try:
            os.link(source, self.root / "actions.html")
        except OSError as error:
            self.skipTest(f"Hardlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            self.render()
        self.assertEqual(source.read_text(encoding="utf-8"), "preserve")

    def test_output_symlink_is_rejected_without_overwriting_target(self):
        source = self.root.parent / "outside.txt"
        source.write_text("preserve", encoding="utf-8")
        try:
            (self.root / "actions.html").symlink_to(source)
        except OSError as error:
            self.skipTest(f"File symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            self.render()
        self.assertEqual(source.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
