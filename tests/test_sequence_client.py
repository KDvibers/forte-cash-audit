"""Bounded sequence payload and transport tests; every HTTP request is mocked."""

import base64
from io import BytesIO
import json
import math
import unittest
from unittest.mock import Mock, patch

from analyzer.local_client import LocalModelClient
from analyzer.video import VideoFrame


SCHEMA = {"type": "object", "properties": {"interaction": {"type": "string", "enum": ["yes", "no", "uncertain"]}},
          "required": ["interaction"], "additionalProperties": False}
PROMPT = "Describe only whether hands touch the machine."


def frames_at(times=(0.0, 0.333333, 0.9)):
    # Encoding, not decoding, is under test; distinct bytes detect reordering.
    return [VideoFrame(timestamp, f"synthetic-image-{index}".encode(), 100, 80)
            for index, timestamp in enumerate(times)]


def server_response(text, *, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12},
            "timings": {"predicted_ms": 15}}


def mock_transport(client, response):
    client.opener = Mock()
    client.opener.open.return_value = BytesIO(json.dumps(response).encode("utf-8"))


class SequencePayloadTests(unittest.TestCase):
    def setUp(self):
        self.client = LocalModelClient()
        patcher = patch.object(self.client, "_complete", return_value={"marker": "response"})
        self.complete = patcher.start()
        self.addCleanup(patcher.stop)

    def test_two_to_five_frames_keep_order_actual_times_and_no_filename_metadata(self):
        for count in (2, 3, 4, 5):
            with self.subTest(count=count):
                frames = frames_at((0.0, 0.333333, 0.9, 1.2, 1.75))[:count]
                result = self.client.ask_sequence(frames, PROMPT, SCHEMA)
                self.assertEqual(result, {"marker": "response"})
                payload = self.complete.call_args.args[0]
                self.assertEqual(set(payload), {"model", "messages", "temperature", "seed", "max_tokens",
                                                "cache_prompt", "response_format"})
                self.assertEqual(len(payload["messages"]), 1)
                self.assertEqual(payload["messages"][0]["role"], "user")
                content = payload["messages"][0]["content"]
                self.assertEqual(len(content), 2 * count + 1)
                for index, frame in enumerate(frames):
                    self.assertEqual(content[2 * index], {
                        "type": "text", "text": f"Frame {index + 1}; time {frame.timestamp_sec:.3f} seconds."})
                    image = content[2 * index + 1]
                    self.assertEqual(set(image), {"type", "image_url"})
                    self.assertEqual(image["type"], "image_url")
                    self.assertEqual(set(image["image_url"]), {"url"})
                    prefix, encoded = image["image_url"]["url"].split(",", 1)
                    self.assertEqual(prefix, "data:image/jpeg;base64")
                    self.assertEqual(base64.b64decode(encoded), frame.jpeg)
                self.assertEqual(content[-1], {"type": "text", "text": PROMPT})
                self.assertEqual(payload["max_tokens"], 128)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["seed"], 42)
                self.assertIs(payload["cache_prompt"], False)
                self.assertEqual(payload["response_format"], {"type": "json_object", "schema": SCHEMA})
                self.assertEqual(self.complete.call_args.kwargs, {"timeout": None})

    def test_invalid_frame_counts_never_reach_transport(self):
        for count in (0, 1, 6, 10):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.client.ask_sequence(frames_at(range(count)), PROMPT, SCHEMA)
        self.complete.assert_not_called()

    def test_invalid_timestamps_are_rejected_before_transport(self):
        for times in ((0, 0), (1, 0), (0, 2, 1), (-1, 1),
                      (math.nan, 1), (0, math.nan), (0, math.inf), (-math.inf, 1),
                      (False, 1), (0, True), ("0", 1), (None, 1)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.client.ask_sequence(frames_at(times), PROMPT, SCHEMA)
        self.complete.assert_not_called()

    def test_token_budget_rejects_out_of_range_and_non_integer_values(self):
        for budget in (0, -1, 257, 1000, True, False, 128.0, "128", None, math.nan, math.inf):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.client.ask_sequence(frames_at(), PROMPT, SCHEMA, max_tokens=budget)
        self.complete.assert_not_called()

    def test_valid_token_boundaries_and_timeout_are_forwarded(self):
        for budget in (1, 128, 256):
            with self.subTest(budget=budget):
                self.client.ask_sequence(frames_at(), PROMPT, SCHEMA, max_tokens=budget, timeout=2.75)
                self.assertEqual(self.complete.call_args.args[0]["max_tokens"], budget)
                self.assertEqual(self.complete.call_args.kwargs, {"timeout": 2.75})


class SequenceTransportTests(unittest.TestCase):
    def test_request_uses_local_json_endpoint_and_explicit_timeout(self):
        client = LocalModelClient("http://127.0.0.1:8080/", timeout=30)
        mock_transport(client, server_response('{"interaction":"yes"}'))
        result = client.ask_sequence(frames_at(), PROMPT, SCHEMA, timeout=1.25)
        request = client.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        payload = json.loads(request.data)
        self.assertEqual(payload["response_format"]["schema"], SCHEMA)
        self.assertEqual(client.opener.open.call_args.kwargs, {"timeout": 1.25})
        self.assertEqual(result["observation"], {"interaction": "yes"})
        self.assertTrue(result["valid_json"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertGreaterEqual(result["request_seconds"], 0)

    def test_default_timeout_is_used_without_override(self):
        client = LocalModelClient(timeout=17)
        mock_transport(client, server_response('{"interaction":"no"}'))
        client.ask_sequence(frames_at(), PROMPT, SCHEMA)
        self.assertEqual(client.opener.open.call_args.kwargs, {"timeout": 17})

    def test_invalid_json_non_objects_and_truncation_remain_auditable(self):
        for text, finish, valid in (("not JSON", "stop", False),
                                    ('{"interaction":', "length", False),
                                    ('{"interaction":"yes"}', "length", True),
                                    ('{"interaction":"yes","extra":1}', "stop", True),
                                    ("[]", "stop", False), ("null", "stop", False)):
            with self.subTest(text=text, finish=finish):
                client = LocalModelClient()
                raw = server_response(text, finish=finish)
                mock_transport(client, raw)
                result = client.ask_sequence(frames_at(), PROMPT, SCHEMA)
                # The client reports syntax only. ActionObserver must reject
                # incomplete finish reasons and extra keys, even if JSON parses.
                self.assertEqual(result["valid_json"], valid)
                self.assertEqual(result["response_text"], text)
                self.assertEqual(result["finish_reason"], finish)
                self.assertEqual(result["usage"], raw["usage"])
                self.assertEqual(result["timings"], raw["timings"])

    def test_timeout_propagates_instead_of_becoming_negative_observation(self):
        client = LocalModelClient()
        client.opener = Mock()
        client.opener.open.side_effect = TimeoutError("local request timed out")
        with self.assertRaisesRegex(TimeoutError, "local request timed out"):
            client.ask_sequence(frames_at(), PROMPT, SCHEMA, timeout=0.5)
        self.assertEqual(client.opener.open.call_count, 1)


class SequenceLocalityTests(unittest.TestCase):
    def test_remote_https_and_credential_urls_are_rejected_without_network(self):
        for url in ("https://127.0.0.1:8080", "http://example.com:8080", "http://192.168.1.5:8080",
                    "http://localhost.example.com:8080", "http://127.0.0.2:8080",
                    "http://user:password@localhost:8080", "http://localhost:8080?route=other",
                    "http://localhost:8080#fragment", "file:///tmp/model"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalModelClient(url)

    def test_loopback_addresses_are_allowed_and_environment_proxies_are_disabled(self):
        for url in ("http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080"):
            with self.subTest(url=url), patch("analyzer.local_client.urllib.request.build_opener") as build:
                client = LocalModelClient(url)
                self.assertEqual(client.base_url, url)
                build.assert_called_once()
                handler = build.call_args.args[0]
                self.assertEqual(handler.proxies, {})
                self.assertIs(client.opener, build.return_value)
                build.return_value.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
