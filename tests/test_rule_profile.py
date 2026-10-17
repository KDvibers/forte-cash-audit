"""Verify the rule pass asks only its supported visual questions."""
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from PIL import Image

from analyzer.observations import ImageObserver, policy_fingerprint
from analyzer.video import VideoFrame


class Client:
    timeout = 20

    def __init__(self, *, position="front", doorway="no"):
        self.calls = []
        self.position = position
        self.doorway = doorway

    def ask_image(self, frame, prompt, schema, **kwargs):
        keys = set(schema["properties"])
        self.calls.append(keys)
        if keys == {"position"}:
            value = {"position": self.position}
        elif keys == {"doorway_activity"}:
            value = {"doorway_activity": self.doorway}
        elif keys == {"person_in_work_area"}:
            value = {"person_in_work_area": "yes" if self.position == "front" else "no"}
        elif keys == {"visible", "bbox"}:
            value = {"visible": "yes", "bbox": [200, 300, 800, 700]}
        elif keys in ({"cash_on_worktop"}, {"resting_on_desk"}):
            value = {next(iter(keys)): "yes"}
        else:
            raise AssertionError("The rule pass must not query cabinet or actions")
        return {"finish_reason": "stop", "valid_json": True, "observation": value}


class RuleProfileTests(unittest.TestCase):
    def test_skipping_cabinet_preserves_both_agreement_checks(self):
        encoded = BytesIO()
        Image.new("RGB", (160, 240), "gray").save(encoded, "JPEG")
        frame = VideoFrame(3.5, encoded.getvalue(), 160, 240)
        primary, activity = Client(), Client()
        result = ImageObserver(primary, "disabled", activity_client=activity).observe(frame, "frame")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["states"]["person_in_work_area"], "yes")
        self.assertEqual(result["states"]["cash_on_worktop"], "yes")
        self.assertEqual(result["person_consistency"], "agreed")
        self.assertEqual(result["cash_consistency"], "agreed")
        self.assertEqual(result["states"]["cabinet_door"], "uncertain")
        self.assertEqual(result["cabinet_consistency"], "disabled_for_rule_scope")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(activity.calls), 4)
        self.assertEqual(result["desk_localization_model_role"], "activity")
        self.assertEqual(primary.calls, [{"cash_on_worktop"}])
        self.assertEqual(activity.calls, [{"position"}, {"person_in_work_area"},
                                         {"visible", "bbox"}, {"resting_on_desk"}])
        self.assertNotIn({"doorway_activity"}, activity.calls)
        self.assertNotEqual(policy_fingerprint("disabled", cash_enabled=True),
                            policy_fingerprint("context_crops", cash_enabled=True))

    def test_rule_profile_keeps_doorway_gate_when_cabinet_is_disabled(self):
        for doorway, expected in (("no", "no"), ("yes", "uncertain")):
            with self.subTest(doorway=doorway):
                encoded = BytesIO()
                Image.new("RGB", (160, 240), "gray").save(encoded, "JPEG")
                frame = VideoFrame(3.5, encoded.getvalue(), 160, 240)
                primary, activity = Client(), Client(position="behind", doorway=doorway)
                result = ImageObserver(primary, "disabled", activity_client=activity).observe(frame, "frame")
                self.assertEqual(result["states"]["person_in_work_area"], expected)
                self.assertEqual(result["person_doorway_check"], doorway)
                self.assertEqual(result["states"]["cash_on_worktop"], "yes")
                self.assertEqual(result["states"]["cabinet_door"], "uncertain")
                self.assertEqual(result["cabinet_consistency"], "disabled_for_rule_scope")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(activity.calls, [{"position"}, {"person_in_work_area"},
                                                 {"doorway_activity"}, {"visible", "bbox"}, {"resting_on_desk"}])
                self.assertEqual(primary.calls, [{"cash_on_worktop"}])

    def test_policy_fingerprint_records_the_desk_route_without_changing_legacy_mode(self):
        dumps = json.dumps
        configurations = []

        def capture(value, **kwargs):
            configurations.append(value.copy())
            return dumps(value, **kwargs)

        with patch("analyzer.observations.json.dumps", side_effect=capture):
            active = policy_fingerprint("disabled", cash_enabled=True)
            legacy = policy_fingerprint("disabled", cash_enabled=False)
        self.assertNotEqual(active, legacy)
        self.assertEqual(configurations[0]["desk_localization_model_role"], "activity")
        self.assertEqual(configurations[0]["activity_revision"], 6)
        self.assertNotIn("desk_localization_model_role", configurations[1])
        self.assertNotIn("activity_revision", configurations[1])


if __name__ == "__main__":
    unittest.main()
