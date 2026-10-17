"""Direction review preserves evidence and the distinction between claim and fact."""

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

from analyzer.render_direction import render_direction


class _Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.images, self.links, self.tags, self.attributes = [], [], [], []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.tags.append(tag)
        self.attributes.extend(attrs)
        if tag == "img":
            self.images.append(attrs)
        if tag == "a":
            self.links.append(attrs.get("href", ""))


class RenderDirectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="forte-direction-page-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / "run"
        (self.root / "frames").mkdir(parents=True)
        evidence = []
        for index, timestamp in enumerate((1.0333, 1.5333, 2.0333, 2.5333, 3.0333)):
            name = f"frames/frame_{index:04d}.jpg"
            Image.new("RGB", (32+index, 18+index), (index*40, 20, 60)).save(self.root / name, "JPEG")
            evidence.append({"pool_index": index, "timestamp_sec": timestamp, "image_path": name,
                             "image_sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()})
        self.report = {
            "kind": "experimental_direction_review", "status": "completed", "model_label": "local-test-model",
            "total_seconds": 12.5, "request_count": 1, "selected_window_count": 1,
            "processed_window_count": 1, "all_selected_windows_processed": True,
            "feature_support": {"status": "not_validated", "reason": "Проверка направления ещё не закончена"},
            "settings": {"max_windows": 4},
            "direction_selection": {"candidate_count": 1, "selected_count": 1, "capped": False,
                                    "policy": {"max_windows": 4}, "policy_sha256": "selection-test-hash",
                                    "coverage": {"cap_dropped_window_count": 0,
                                                 "actual_max_frame_gap_sec": .5,
                                                 "selected_max_anchor_gap_sec": .5}, "dropped_windows": []},
            "direction_checks": [{"window_id": "direction_0000", "status": "ok", "attempted": True,
                                  "source_window_ids": ["action_0002", "action_0003"],
                                  "start_sec": 1.0333, "end_sec": 3.0333,
                                  "result": {"counter_direction": "uncertain"},
                                  "raw_banknote_motion": "hand_to_input",
                                  "model_evidence_description": "A bundle moves toward a tray.",
                                  "feature_support": {"status": "not_validated"},
                                  "evidence": evidence, "errors": [], "calls": []}],
            "errors": [], "limitations": ["Selection cannot recover upstream missed or uncertain interactions."],
        }

    @property
    def check(self):
        return self.report["direction_checks"][0]

    def render(self):
        source = self.root / "report.json"
        source.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        output = render_direction(source)
        text = output.read_text(encoding="utf-8")
        return output, text, _Page(text)

    def test_all_five_full_images_preserve_order_dimensions_bytes_and_historical_page(self):
        historic = self.root / "actions.html"
        historic.write_bytes(b"historical action review")
        originals = {p: p.read_bytes() for p in (self.root / "frames").iterdir()}
        output, text, page = self.render()
        self.assertEqual(output, self.root / "direction.html")
        self.assertEqual([item["src"] for item in page.images], [item["image_path"] for item in self.check["evidence"]])
        self.assertEqual(page.links, [item["src"] for item in page.images])
        self.assertEqual([item["width"] for item in page.images], [str(32+i) for i in range(5)])
        self.assertEqual([item["height"] for item in page.images], [str(18+i) for i in range(5)])
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        self.assertEqual(historic.read_bytes(), b"historical action review")
        self.assertIn("Кадр 5 · 3.033 с", text)
        self.assertIn("action_0002, action_0003", text)
        self.assertIn("Направление переноса пока не подтверждено", text)
        self.assertNotIn("script", page.tags)
        for url in page.links:
            self.assertFalse(urlsplit(url).scheme)
            self.assertTrue((self.root / unquote(url)).is_file())

    def test_two_to_five_frames_are_supported_without_replacing_or_reordering(self):
        original = copy.deepcopy(self.check["evidence"])
        for size in (2, 3, 4, 5):
            with self.subTest(size=size):
                self.check["evidence"] = original[:size]
                self.check["end_sec"] = original[size-1]["timestamp_sec"]
                _, text, page = self.render()
                self.assertEqual(len(page.images), size)
                self.assertNotIn("Ожидалось от двух до пяти", text)

    def test_raw_direction_claim_is_labelled_and_never_promoted(self):
        self.check["calls"] = [{"response": {"response_text": "NOT-A-DISPLAYED-FACT"}}]
        for raw in ("hand_to_input", "output_to_hand", "no_transfer", "uncertain"):
            with self.subTest(raw=raw):
                self.check["raw_banknote_motion"] = raw
                _, text, _ = self.render()
                self.assertIn("Ответ модели — требует проверки", text)
                self.assertIn("Принятый результат: направление не подтверждено (uncertain)", text)
                self.assertIn(f"<code>{raw}</code>", text)
                self.assertNotIn("NOT-A-DISPLAYED-FACT", text)
                if raw == "no_transfer":
                    self.assertIn("Это не доказывает отсутствие переноса между кадрами", text)
        self.check["result"]["counter_direction"] = "hand_to_input"
        _, text, _ = self.render()
        self.assertIn("не соответствует экспериментальному протоколу", text)
        self.assertIn("Принятый результат: направление не подтверждено", text)

    def test_missing_response_and_not_started_are_not_negative_model_answers(self):
        self.check.update(status="not_started", attempted=False, raw_banknote_motion=None)
        self.check.pop("model_evidence_description")
        self.report.update(status="partial", processed_window_count=0, all_selected_windows_processed=False,
                           budget_exceeded=True, request_count=0)
        _, text, _ = self.render()
        self.assertIn("Не запускалось</dt><dd>1", text)
        self.assertIn("Начата проверка окон</dt><dd>0", text)
        self.assertIn("Запрос выполнен без записанных ошибок</dt><dd>0", text)
        self.assertIn("Сырой ответ не записан. Это не отрицательный ответ", text)
        self.assertIn("Не все выбранные окна обработаны", text)
        self.assertIn("Лимит времени исчерпан", text)

    def test_partial_attempt_counts_separately_from_success_and_displays_errors(self):
        self.check.update(status="partial", raw_banknote_motion=None,
                          errors=[{"type": "TimeoutError", "message": "request exceeded deadline"}])
        self.report.update(status="partial", errors=[{"message": "job incomplete"}])
        _, text, _ = self.render()
        self.assertIn("Начата проверка окон</dt><dd>1", text)
        self.assertIn("Запрос выполнен без записанных ошибок</dt><dd>0", text)
        self.assertIn("request exceeded deadline", text)
        self.assertIn("job incomplete", text)
        self.assertIn("Частичный результат", text)

    def test_coverage_cap_runtime_and_unknown_metadata_are_explicit(self):
        self.report["direction_selection"].update(candidate_count=5, selected_count=2, capped=True,
            dropped_windows=[{"source_window_ids": ["action_0020"], "start_sec": 10, "end_sec": 12,
                              "reason": "direction_window_cap"}])
        self.report["direction_selection"]["coverage"]["cap_dropped_window_count"] = 3
        _, text, _ = self.render()
        self.assertIn("Кандидатов до лимита</dt><dd>5", text)
        self.assertIn("Выбрано окон</dt><dd>2", text)
        self.assertIn("Лимит окон</dt><dd>4", text)
        self.assertIn("Отброшено из-за лимита</dt><dd>3", text)
        self.assertIn("Лимит окон достигнут", text)
        self.assertIn("Не все выбранные окна обработаны", text)
        self.assertIn("action_0020: 10 с — 12 с", text)
        self.assertIn("local-test-model", text)
        self.assertIn("Время этого прохода: 12.5 с", text)
        self.assertIn("Время не включает предыдущий анализ", text)
        self.assertIn("Отбор не восстанавливает взаимодействия", text)
        self.assertIn("selection-test-hash", text)
        self.report = {}
        _, text, page = self.render()
        self.assertFalse(page.images)
        self.assertIn("Нет записанных проверок направления. Это не доказательство отсутствия действий", text)
        self.assertIn("Сведения об отборе окон не записаны", text)
        self.assertIn("Выбрано окон</dt><dd>неизвестно", text)

    def test_empty_rows_with_selected_windows_warn_about_incomplete_processing(self):
        self.report["direction_checks"] = []
        _, text, _ = self.render()
        self.assertIn("Не все выбранные окна обработаны", text)
        self.assertIn("Это не доказательство отсутствия действий", text)

    def test_tampered_or_unhashed_frame_is_never_displayed(self):
        for digest in ("0"*64, None, "x"*64):
            with self.subTest(digest=digest):
                self.check["evidence"][2]["image_sha256"] = digest
                _, text, page = self.render()
                self.assertEqual(len(page.images), 4)
                self.assertNotIn("frames/frame_0002.jpg", page.links)
                self.assertIn("SHA256", text)
                self.assertIn("Кадр 3 · 2.033 с", text)

    def test_absolute_external_and_parent_paths_are_not_evidence_sources(self):
        row = self.check["evidence"][0]
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes((self.root / row["image_path"]).read_bytes())
        self.check["evidence"] = [row]
        for name in (str(outside), "../outside.jpg", "https://example.test/frame.jpg", r"C:\outside.jpg",
                     r"\\server\share\frame.jpg", "file:///C:/outside.jpg", "frames/frame_0000.jpg:stream"):
            with self.subTest(name=name):
                row["image_path"] = name
                _, text, page = self.render()
                self.assertFalse(page.images)
                self.assertFalse(page.links)
                self.assertIn("Кадр недоступен", text)

    def test_hostile_values_are_only_text_everywhere(self):
        attack = '<script>alert("bad")</script><img src="https://bad" onerror="bad">'
        self.check.update(window_id=attack, source_window_ids=[attack], raw_banknote_motion=attack,
                          model_evidence_description=attack, feature_support={"reason": attack},
                          errors=[{"message": attack}])
        self.report.update(model_label=attack, feature_support={"status": attack}, limitations=[attack],
                           runtime={"model_path": attack}, runtime_metadata_error=attack, source_report_path=attack,
                           errors=[attack])
        self.report["direction_selection"]["dropped_windows"] = [{"source_window_ids": [attack], "reason": attack}]
        _, text, page = self.render()
        self.assertEqual(len(page.images), 5)
        self.assertNotIn("script", page.tags)
        self.assertNotIn("onerror", page.attributes)
        self.assertNotIn("iframe", page.tags)
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn(attack, text)

    def test_bad_timestamps_warn_without_reordering_actual_files(self):
        original_paths = [item["image_path"] for item in self.check["evidence"]]
        for timestamp in (float("nan"), None, -1, False, 1.0333):
            with self.subTest(timestamp=timestamp):
                self.check["evidence"][2]["timestamp_sec"] = timestamp
                _, text, page = self.render()
                self.assertIn("нарушен хронологический порядок", text)
                self.assertEqual([item["src"] for item in page.images], original_paths)

    def test_missing_evidence_and_mismatched_boundary_are_not_silently_fixed(self):
        self.check["start_sec"] = 0
        _, text, _ = self.render()
        self.assertIn("Границы окна не совпадают", text)
        self.check["evidence"] = []
        _, text, page = self.render()
        self.assertFalse(page.images)
        self.assertIn("в отчёте указано 0", text)
        self.assertIn("недостающие не заменяются", text)

    def test_invalid_structure_does_not_replace_an_existing_page(self):
        source, output = self.root / "report.json", self.root / "direction.html"
        valid = copy.deepcopy(self.report)
        bad_reports = [[], {"direction_checks": {}}, {"direction_checks": [None]},
                       {"direction_checks": [{"evidence": "wrong"}]},
                       {"direction_selection": {"dropped_windows": [False]}}]
        for report in bad_reports:
            with self.subTest(report=report):
                source.write_text(json.dumps(report), encoding="utf-8")
                output.write_text("previous page", encoding="utf-8")
                with self.assertRaises(ValueError):
                    render_direction(source)
                self.assertEqual(output.read_text(encoding="utf-8"), "previous page")
        self.report = valid

    def test_plain_output_can_be_refreshed_without_changing_input(self):
        output, _, _ = self.render()
        source = self.root / "report.json"
        before = source.read_bytes()
        output.write_text("old page", encoding="utf-8")
        self.assertEqual(render_direction(source), output)
        self.assertEqual(source.read_bytes(), before)
        self.assertIn("Проверка загрузки и извлечения", output.read_text(encoding="utf-8"))

    def test_output_cannot_replace_input_even_if_input_has_html_name(self):
        source = self.root / "direction.html"
        original = json.dumps(self.report).encode()
        source.write_bytes(original)
        with self.assertRaises(ValueError):
            render_direction(source)
        self.assertEqual(source.read_bytes(), original)

    def test_hardlinked_output_cannot_overwrite_another_file(self):
        source = self.root / "report.json"
        source.write_text(json.dumps(self.report), encoding="utf-8")
        target = self.root.parent / "keep.txt"
        target.write_bytes(b"keep external data")
        os.link(target, self.root / "direction.html")
        with self.assertRaises(ValueError):
            render_direction(source)
        self.assertEqual(target.read_bytes(), b"keep external data")

    def test_symlinked_output_cannot_overwrite_another_file(self):
        source = self.root / "report.json"
        source.write_text(json.dumps(self.report), encoding="utf-8")
        target = self.root.parent / "keep.txt"
        target.write_bytes(b"keep external data")
        try:
            (self.root / "direction.html").symlink_to(target)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            render_direction(source)
        self.assertEqual(target.read_bytes(), b"keep external data")


if __name__ == "__main__":
    unittest.main()
