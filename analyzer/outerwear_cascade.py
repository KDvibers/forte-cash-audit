"""Inactive CPU-testable composition; not connected to the service or network.

Both stages use the unchanged full OuterwearObserver. A future transport must
authenticate the worker; this validator proves record consistency, not that an
untrusted host really ran the declared weights. No publication/refusal gate is
changed by importing this module.
"""
from copy import deepcopy
import hashlib
import json
import math
import time

from .observations import ObservationDeadline, ObservationError
from .outerwear import (OuterwearObserver, PRIMARY_SCHEMA, VERIFICATION_SCHEMA,
                        _read_answer, outerwear_policy_fingerprint)


_STAGES = ("filter4b", "review8b")
_MODELS = {"filter4b": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
           "review8b": "Qwen3VL-8B-Instruct-Q4_K_M.gguf"}
_ROLES = {"filter4b": "local_activity_4b", "review8b": "team_activity_8b"}
_FEATURES = {"sleeve", "collar", "garment_opening"}
_TOP_KEYS = {"kind", "frame_id", "timestamp_sec", "image_sha256", "state", "status",
             "errors", "calls", "stages", "policy_sha256", "selected_stage", "elapsed_seconds"}
_STAGE_KEYS = {"name", "model_role", "runtime", "frame_sha256", "record"}


class CascadeValidationError(ObservationError):
    """The cascade record is not a complete, supported observation."""


def cascade_policy_fingerprint():
    policy = {"revision": 1, "observer_policy_sha256": outerwear_policy_fingerprint(),
              "filter": "full_original_4b_observer", "review": "full_original_8b_observer",
              "dispatch": "clean_no_skips_review_clean_yes_or_uncertain_requires_review",
              "mapping": "final_is_review_state_without_fallback",
              "errors": "stop_no_additional_calls", "deadline": "unchanged_absolute_deadline",
              "ledger": "original_stage_records_and_exact_flattened_calls_v1",
              "models": _MODELS, "roles": _ROLES}
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise CascadeValidationError(message)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _same_json(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _frame_identity(frame, frame_id):
    _require(isinstance(frame_id, str) and bool(frame_id.strip()), "Invalid frame ID")
    _require(_finite(frame.timestamp_sec), "Invalid frame timestamp")
    _require(isinstance(frame.jpeg, bytes) and bool(frame.jpeg), "Frame must contain source JPEG bytes")
    return hashlib.sha256(frame.jpeg).hexdigest()


def canonical_runtime(runtime):
    """Keep the declared identity, with only the model basename in records.

    The six ordinary server_info fields are required. Optional explicit launch
    metadata is retained and compared, never inferred from a filename.
    """
    required = {"model_path", "model_alias", "model_ftype", "build_info", "modalities", "total_slots"}
    optional = {"context_size", "gpu_layers", "weights_sha256", "projector_sha256"}
    _require(isinstance(runtime, dict) and required <= set(runtime) <= required | optional, "Invalid runtime fields")
    result = deepcopy(runtime)
    for field in ("model_path", "model_alias", "model_ftype", "build_info"):
        _require(isinstance(result[field], str) and bool(result[field].strip()), "Missing runtime identity")
    result["model_path"] = result["model_path"].replace("\\", "/").rsplit("/", 1)[-1]
    _require(isinstance(result["modalities"], dict) and result["modalities"].get("vision") is True
             and all(isinstance(k, str) and type(v) is bool for k, v in result["modalities"].items()),
             "Runtime must declare vision support")
    _require(type(result["total_slots"]) is int and result["total_slots"] == 1, "Cascade requires one inference slot")
    for field in ("context_size", "gpu_layers"):
        if field in result:
            _require(type(result[field]) is int and result[field] >= (1 if field == "context_size" else 0), "Invalid launch setting")
    for field in ("weights_sha256", "projector_sha256"):
        if field in result:
            value = result[field]
            _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Invalid weight hash")
    return result


def _identities(expected):
    _require(isinstance(expected, dict) and set(expected) == set(_STAGES), "Both runtime identities are required")
    result = {name: canonical_runtime(runtime) for name, runtime in expected.items()}
    for name, runtime in result.items():
        _require(runtime["model_path"] == _MODELS[name], "Unexpected stage model")
    return result


def _flatten(stages):
    calls = []
    for stage in stages:
        original = stage.get("record")
        if not isinstance(original, dict) or not isinstance(original.get("calls"), list):
            continue
        for index, original_call in enumerate(original["calls"]):
            _require(isinstance(original_call, dict), "Invalid original call")
            call = deepcopy(original_call)
            call.update(stage=stage["name"], stage_call_index=index,
                        original_model_role=original_call.get("model_role"),
                        model_role=stage["model_role"], actual_model=_MODELS[stage["name"]])
            calls.append(call)
    return calls


def _clean_original(record, frame_id, timestamp):
    _require(isinstance(record, dict), "Missing original observer record")
    required = {"frame_id", "timestamp_sec", "state", "status", "calls", "errors",
                "policy_sha256", "primary_state", "elapsed_seconds"}
    _require(required <= set(record) <= required | {"verification"}, "Unknown or missing observer fields")
    _require(record["frame_id"] == frame_id and _finite(record["timestamp_sec"])
             and record["timestamp_sec"] == timestamp, "Original frame identity changed")
    _require(record["policy_sha256"] == outerwear_policy_fingerprint(), "Unknown original observer policy")
    _require(record["status"] == "ok" and record["errors"] == [], "Observer did not finish cleanly")
    _require(_finite(record["elapsed_seconds"]), "Invalid observer duration")
    calls = record["calls"]
    _require(isinstance(calls, list) and 1 <= len(calls) <= 2, "Missing or extra observer calls")
    for call in calls:
        _require(isinstance(call, dict) and set(call) == {
            "question", "completed", "model_role", "response", "request_seconds"}, "Unknown or incomplete call fields")
        _require(call["completed"] is True and _finite(call["request_seconds"]), "Call completion not proven")
        # The original implementation labels both injected clients 4b. Preserve
        # that raw record; the separate exact ledger contains the actual role.
        _require(call["model_role"] == "local_activity_4b", "Unexpected original call role")
        response = call["response"]
        _require(isinstance(response, dict), "Missing call response")
        _require(set(response) <= {"request_seconds", "response_text", "finish_reason", "usage", "timings",
                                   "observation", "valid_json"}, "Unknown response fields")
        if "request_seconds" in response:
            _require(_finite(response["request_seconds"]), "Invalid response duration")
        if "response_text" in response:
            try:
                decoded = json.loads(response["response_text"])
            except (ValueError, TypeError) as error:
                raise CascadeValidationError("Response text is invalid JSON") from error
            _require(decoded == response.get("observation"), "Response text and parsed answer disagree")
    try:
        primary = _read_answer(calls[0]["response"], PRIMARY_SCHEMA)["removed_outerwear"]
        _require(calls[0]["question"] == "removed_outerwear" and record["primary_state"] == primary,
                 "Primary trace disagrees")
        expected = primary
        if primary == "yes":
            _require(len(calls) == 2 and calls[1]["question"] == "outerwear_structure", "Verification missing or reordered")
            inventory = _read_answer(calls[1]["response"], VERIFICATION_SCHEMA)
            _require(record.get("verification") == inventory, "Verification summary disagrees")
            expected = "yes" if (inventory["outerwear"] == "resting_in_workplace"
                                  and inventory["garment_feature"] in _FEATURES) else "uncertain"
        else:
            _require(len(calls) == 1 and "verification" not in record, "Unexpected verification call")
    except ObservationError as error:
        raise CascadeValidationError(str(error)) from error
    _require(record["state"] == expected, "Observer state is not supported by its calls")
    return expected


def validate_cascade_record(record, frame, frame_id, *, expected_runtimes):
    """Return a detached validated row; reject partial or inconsistent records.

    This is intentionally NOT a safe-refusal proof. The old service helper must
    continue rejecting this new policy until a separate integration is tested.
    """
    digest = _frame_identity(frame, frame_id)
    expected_runtimes = _identities(expected_runtimes)
    _require(isinstance(record, dict) and set(record) == _TOP_KEYS, "Unknown or missing cascade fields")
    _require(record["kind"] == "inactive_outerwear_cascade_v1"
             and record["policy_sha256"] == cascade_policy_fingerprint(), "Unknown cascade policy")
    _require(record["frame_id"] == frame_id and _finite(record["timestamp_sec"])
             and record["timestamp_sec"] == frame.timestamp_sec and record["image_sha256"] == digest,
             "Cascade frame identity changed")
    _require(record["status"] == "ok" and record["errors"] == [] and _finite(record["elapsed_seconds"]),
             "Cascade did not finish cleanly")
    stages = record["stages"]
    _require(isinstance(stages, list) and 1 <= len(stages) <= 2, "Invalid stage count")
    states = []
    for index, stage in enumerate(stages):
        name = _STAGES[index]
        _require(isinstance(stage, dict) and set(stage) == _STAGE_KEYS, "Unknown stage fields")
        _require(stage["name"] == name and stage["model_role"] == _ROLES[name], "Reordered or unknown stage")
        _require(_same_json(stage["runtime"], expected_runtimes[name]), "Runtime identity changed")
        _require(stage["frame_sha256"] == digest, "Stage source bytes changed")
        states.append(_clean_original(stage["record"], frame_id, frame.timestamp_sec))
    expected_count = 1 if states[0] == "no" else 2
    _require(len(stages) == expected_count, "Missing or forbidden review stage")
    _require(isinstance(record["calls"], list) and _same_json(record["calls"], _flatten(stages)), "Flattened ledger is incomplete or altered")
    _require(record["selected_stage"] == _STAGES[expected_count-1] and record["state"] == states[-1],
             "Unsupported cascade state reduction")
    return deepcopy(record)


class CascadeOuterwearObserver:
    """Explicit inactive candidate. Callers retain cash guards and publication gates."""

    def __init__(self, filter_client, review_client=None, *, review_worker=None, expected_runtimes):
        _require((review_client is None) != (review_worker is None), "Choose exactly one review client or observer worker")
        if review_worker is not None:
            _require(callable(getattr(review_worker, "observe", None)), "Worker must return original observer records")
        self.expected_runtimes = _identities(expected_runtimes)
        self.review_worker = review_worker
        self.clients = {"filter4b": filter_client, "review8b": review_worker if review_worker is not None else review_client}

    def observe(self, frame, frame_id, *, deadline):
        digest = _frame_identity(frame, frame_id)
        _require(_finite(deadline), "An absolute finite deadline is required")
        started = time.perf_counter()
        record = {"kind": "inactive_outerwear_cascade_v1", "frame_id": frame_id,
                  "timestamp_sec": frame.timestamp_sec, "image_sha256": digest,
                  "state": "uncertain", "status": "partial", "errors": [], "calls": [], "stages": [],
                  "policy_sha256": cascade_policy_fingerprint(), "selected_stage": None, "elapsed_seconds": 0.0}

        def check_deadline():
            if time.perf_counter() >= deadline:
                raise ObservationDeadline("Cascade absolute deadline exhausted")

        def finish():
            record["calls"] = _flatten(record["stages"])
            record["elapsed_seconds"] = time.perf_counter() - started

        try:
            for name in _STAGES:
                check_deadline()
                client = self.clients[name]
                previous_timeout = client.timeout
                _require(_finite(previous_timeout) and previous_timeout > 0, "Invalid client timeout")
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    raise ObservationDeadline("Cascade deadline exhausted before runtime dispatch")
                client.timeout = min(previous_timeout, 5.0, remaining)
                try:
                    runtime = canonical_runtime(client.server_info())
                finally:
                    client.timeout = previous_timeout
                check_deadline()
                _require(_same_json(runtime, self.expected_runtimes[name]), "Runtime identity changed before dispatch")
                stage = {"name": name, "model_role": _ROLES[name], "runtime": runtime,
                         "frame_sha256": digest, "record": None}
                record["stages"].append(stage)
                try:
                    observer = self.review_worker if name == "review8b" and self.review_worker is not None else OuterwearObserver(client)
                    stage["record"] = deepcopy(observer.observe(frame, frame_id, deadline=deadline))
                except ObservationDeadline as error:
                    stage["record"] = deepcopy(error.partial_record)
                    raise
                check_deadline()
                state = _clean_original(stage["record"], frame_id, frame.timestamp_sec)
                if name == "review8b" or state == "no":
                    record.update(state=state, status="ok", selected_stage=name)
                    finish()
                    validated = validate_cascade_record(record, frame, frame_id, expected_runtimes=self.expected_runtimes)
                    check_deadline()
                    return validated
        except Exception as error:
            record.update(state="uncertain", status="partial", selected_stage=None)
            record["errors"].append({"feature": "outerwear_cascade", "type": type(error).__name__, "message": str(error)})
            finish()
            if isinstance(error, TimeoutError):
                raise ObservationDeadline(str(error), partial_record=record) from error
            return record
        raise AssertionError("Cascade did not select a terminal stage")
