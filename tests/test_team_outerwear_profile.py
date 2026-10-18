"""Configuration/identity checks use temporary JSON and fake HTTP methods only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from analyzer import team_outerwear_profile as module
from analyzer.observations import ObservationDeadline
from analyzer.outerwear import outerwear_policy_fingerprint
from analyzer.outerwear_cascade import CascadeOuterwearObserver, cascade_policy_fingerprint


def runtime(size):
    return {"model_path": f"Qwen3VL-{size}B-Instruct-Q4_K_M.gguf", "model_alias": "forte-qwen",
            "model_ftype": "Q4_K - Medium", "build_info": "synthetic-build" if size == 4 else "sha256:"+"a"*64,
            "modalities": {"vision": True, "audio": False}, "total_slots": 1}


def config():
    return {"schema_version": 1, "profile": "private_team_outerwear_v1",
            "peer_hostname": "synthetic.test-tailnet.ts.net", "filter_runtime": runtime(4),
            "review_runtime": runtime(8), "observer_policy_sha256": outerwear_policy_fingerprint(),
            "cascade_policy_sha256": cascade_policy_fingerprint()}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/"profile.json"
        self.path.write_text(json.dumps(config()), encoding="utf-8")
        self.env = patch.dict(os.environ, {module.CONFIG_ENV: str(self.path)})
        self.env.start(); self.addCleanup(self.env.stop)
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No real network in tests"))
        self.open = self.network.start(); self.addCleanup(self.network.stop)

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def load(self):
        return module.load_configured_observer()

    def test_absent_environment_is_disabled_without_clients_or_http(self):
        os.environ.pop(module.CONFIG_ENV)
        with patch.object(module, "LocalModelClient") as local, patch.object(module, "TeamOuterwearWorkerClient") as peer:
            self.assertIsNone(self.load())
        local.assert_not_called(); peer.assert_not_called(); self.open.assert_not_called()

    def test_loading_valid_config_creates_only_fixed_clients_and_no_http(self):
        observer = self.load()
        self.assertIs(type(observer), CascadeOuterwearObserver)
        self.assertEqual(observer.clients["filter4b"].base_url, "http://127.0.0.1:8081")
        self.assertEqual(observer.clients["filter4b"].timeout, 20)
        self.assertEqual(observer.review_worker.base_url, "https://synthetic.test-tailnet.ts.net:8443")
        self.assertEqual(observer.review_worker.timeout, 75)
        self.assertIs(observer.review_worker, observer.clients["review8b"])
        self.assertEqual(observer.expected_runtimes, {"filter4b":runtime(4),"review8b":runtime(8)})
        self.open.assert_not_called()

    def test_filter_path_is_canonicalized_but_review_requires_worker_basename(self):
        value = config(); value["filter_runtime"]["model_path"] = "C:/synthetic/"+runtime(4)["model_path"]
        self.write(value)
        self.assertEqual(self.load().expected_runtimes["filter4b"]["model_path"], runtime(4)["model_path"])
        value["review_runtime"]["model_path"] = "C:/synthetic/"+runtime(8)["model_path"]
        self.write(value)
        with self.assertRaises(ValueError):
            self.load()

    def test_relative_missing_directory_symlink_and_junction_are_rejected(self):
        for location in ("relative.json", str(self.path.parent), str(self.path.parent/"missing.json")):
            with self.subTest(location=Path(location).name), patch.dict(os.environ,{module.CONFIG_ENV:location}):
                with self.assertRaises((ValueError, OSError)):
                    self.load()
        for name in ("is_symlink", "is_junction"):
            with self.subTest(kind=name), patch.object(Path,name,return_value=True):
                with self.assertRaisesRegex(ValueError,"symlinks and junctions"):
                    self.load()
        self.open.assert_not_called()

    def test_oversize_empty_and_changed_file_are_rejected(self):
        for raw in (b"", b" "*(module.MAX_CONFIG_BYTES+1)):
            self.path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,"64 KiB"):
                self.load()
        self.write(config())
        before = self.path.stat()
        fake = Mock(st_mode=before.st_mode, st_dev=before.st_dev, st_ino=before.st_ino+1)
        with patch.object(module.os,"fstat",return_value=fake), self.assertRaisesRegex(ValueError,"changed while opening"):
            self.load()

    def test_duplicate_fields_invalid_json_nonfinite_and_nonobject_are_rejected(self):
        cases = [b'{"profile":1,"profile":2}', b'{"a":{"x":1,"x":2}}', b'{"a":NaN}',
                 b'{"a":Infinity}', b'[]', b'null', b'{', b'\xff']
        for index, raw in enumerate(cases):
            with self.subTest(index=index):
                self.path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.load()
        self.open.assert_not_called()

    def test_schema_policy_extra_fields_bad_model_and_raw_remote_build_are_rejected(self):
        cases = []
        for key,value in (("schema_version",True),("schema_version",2),("profile","other"),
                          ("observer_policy_sha256","0"*64),("cascade_policy_sha256","0"*64),
                          ("unexpected",True),("peer_hostname","example.invalid"),
                          ("peer_hostname","synthetic.test-tailnet.ts.net:443")):
            cases.append({**config(),key:value})
        missing = config(); missing.pop("review_runtime"); cases.append(missing)
        for name,key,value in (("filter_runtime","model_path",runtime(8)["model_path"]),
                               ("review_runtime","model_path",runtime(4)["model_path"]),
                               ("review_runtime","build_info","unhashed-build"),
                               ("filter_runtime","total_slots",2),("review_runtime","total_slots",True),
                               ("review_runtime","modalities",{"vision":False}),
                               ("filter_runtime","extra_field",True)):
            bad=config(); bad[name][key]=value; cases.append(bad)
        for index,value in enumerate(cases):
            with self.subTest(index=index):
                self.write(value)
                with self.assertRaises(ValueError):
                    self.load()
        self.open.assert_not_called()

    def test_expired_load_deadline_stops_before_file_read_or_client_construction(self):
        with (patch.object(module.time,"perf_counter",return_value=10),
              patch.object(Path,"open") as opened,
              patch.object(module,"LocalModelClient") as local,
              self.assertRaises(ObservationDeadline)):
            module.load_configured_observer(deadline=10)
        opened.assert_not_called(); local.assert_not_called()

    def test_parse_finishing_at_deadline_does_not_construct_clients(self):
        now=[10]
        original=json.loads
        def slow(*args,**kwargs):
            result=original(*args,**kwargs)
            now[0]=20
            return result
        with (patch.object(module.time,"perf_counter",side_effect=lambda:now[0]),
              patch.object(module.json,"loads",side_effect=slow),
              patch.object(module,"LocalModelClient") as local,
              patch.object(module,"TeamOuterwearWorkerClient") as peer,
              self.assertRaises(ObservationDeadline)):
            module.load_configured_observer(deadline=20)
        local.assert_not_called(); peer.assert_not_called()

    def test_verify_both_health_calls_share_deadline_and_restore_timeouts(self):
        observer=self.load(); now=[10.0]; observed=[]
        def health(name):
            observed.append((name,observer.clients[name].timeout))
            if name=="filter4b": now[0]+=4
            return deepcopy(observer.expected_runtimes[name])
        for name in observer.clients:
            observer.clients[name].server_info=Mock(side_effect=lambda name=name:health(name))
        with patch.object(module.time,"perf_counter",side_effect=lambda:now[0]):
            result=module.verify_configured_observer(observer,18)
        self.assertEqual(observed,[("filter4b",5.0),("review8b",4.0)])
        self.assertEqual((observer.clients["filter4b"].timeout,observer.review_worker.timeout),(20,75))
        self.assertEqual(result,observer.expected_runtimes)
        result["filter4b"]["build_info"]="changed"
        self.assertNotEqual(result,observer.expected_runtimes)
        self.open.assert_not_called()

    def test_runtime_mismatch_stops_without_retry_or_other_stage_fallback(self):
        for wrong in ("filter4b","review8b"):
            with self.subTest(wrong=wrong):
                observer=self.load()
                for name,client in observer.clients.items():
                    value=deepcopy(observer.expected_runtimes[name])
                    if name==wrong: value["build_info"]="changed"
                    client.server_info=Mock(return_value=value)
                with patch.object(module.time,"perf_counter",return_value=10), self.assertRaisesRegex(ValueError,"identity changed"):
                    module.verify_configured_observer(observer,20)
                self.assertEqual(observer.clients["filter4b"].server_info.call_count,1)
                self.assertEqual(observer.review_worker.server_info.call_count,0 if wrong=="filter4b" else 1)
                self.assertEqual((observer.clients["filter4b"].timeout,observer.review_worker.timeout),(20,75))

    def test_health_exception_is_propagated_and_timeouts_restored(self):
        for failing in ("filter4b","review8b"):
            with self.subTest(failing=failing):
                observer=self.load(); failure=RuntimeError("synthetic transport error")
                for name,client in observer.clients.items():
                    client.server_info=Mock(return_value=deepcopy(observer.expected_runtimes[name]),
                                            side_effect=failure if name==failing else None)
                with patch.object(module.time,"perf_counter",return_value=10), self.assertRaises(RuntimeError) as caught:
                    module.verify_configured_observer(observer,20)
                self.assertIs(caught.exception,failure)
                self.assertEqual((observer.clients["filter4b"].timeout,observer.review_worker.timeout),(20,75))
                self.assertEqual(observer.review_worker.server_info.call_count,0 if failing=="filter4b" else 1)

    def test_late_first_health_never_starts_second_and_never_confirms_ready(self):
        observer=self.load(); now=[10]
        def late():
            now[0]=20
            return deepcopy(observer.expected_runtimes["filter4b"])
        observer.clients["filter4b"].server_info=Mock(side_effect=late)
        observer.review_worker.server_info=Mock()
        with patch.object(module.time,"perf_counter",side_effect=lambda:now[0]), self.assertRaises(ObservationDeadline):
            module.verify_configured_observer(observer,20)
        observer.review_worker.server_info.assert_not_called()
        self.assertEqual(observer.clients["filter4b"].timeout,20)

    def test_corrupted_expected_identity_is_rejected_before_any_health(self):
        observer=self.load()
        observer.expected_runtimes["filter4b"]["model_path"]=runtime(8)["model_path"]
        for client in observer.clients.values(): client.server_info=Mock()
        with patch.object(module.time,"perf_counter",return_value=10), self.assertRaises(ValueError):
            module.verify_configured_observer(observer,20)
        for client in observer.clients.values(): client.server_info.assert_not_called()

    def test_expired_invalid_deadline_unknown_observer_or_timeout_never_dispatches(self):
        observer=self.load()
        for client in observer.clients.values(): client.server_info=Mock()
        for deadline in (None,True,float("nan"),float("inf"),10):
            with self.subTest(deadline=deadline), patch.object(module.time,"perf_counter",return_value=10):
                with self.assertRaises((ValueError,ObservationDeadline)):
                    module.verify_configured_observer(observer,deadline)
        with patch.object(module.time,"perf_counter",return_value=10), self.assertRaises(ValueError):
            module.verify_configured_observer(object(),20)
        for value in (0,True,float("nan"),-1):
            observer.clients["filter4b"].timeout=value
            with patch.object(module.time,"perf_counter",return_value=10), self.assertRaises(ValueError):
                module.verify_configured_observer(observer,20)
        for client in observer.clients.values(): client.server_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
