import copy
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import create_one_t4 as m

CONFIG = {"project": "test-project", "zones": ["asia-northeast1-c", "asia-northeast1-a"],
          "model": "SPOT", "snapshot": "", "service_account": ""}


class FakeApi:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def call(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        expected_method, expected_path, value = self.events.pop(0)
        assert method == expected_method, (method, expected_method)
        assert expected_path in path, (path, expected_path)
        if isinstance(value, Exception):
            raise value
        return value


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = m.State(Path(self.tmp.name) / "state.json", copy.deepcopy(CONFIG))
        self.state.data["intent"] = {"zone": CONFIG["zones"][0], "request_id": "saved-uuid",
            "created_at": time.time(), "body": m.make_body(CONFIG, CONFIG["zones"][0], "image")}
        self.state.save()

    def test_timeout_reuses_same_request_and_body(self):
        api = FakeApi([
            ("POST", "requestId=saved-uuid", m.TransportError("timeout")),
            ("POST", "requestId=saved-uuid", {"name": "op1", "status": "DONE"}),
            ("GET", "/instances/soup-", {"status": "TERMINATED"}),
        ])
        with self.assertRaisesRegex(m.Halt, "no second VM"):
            m.finish_intent(api, self.state, sleep=lambda _: None)
        self.assertEqual(api.calls[0], api.calls[1])
        self.assertEqual(self.state.data["next_zone"], 0)

    def test_saved_operation_is_polled_without_another_create(self):
        self.state.data["intent"]["operation"] = {"name": "op1", "status": "RUNNING"}
        api = FakeApi([("GET", "/operations/op1", {"name": "op1", "status": "DONE",
            "error": {"errors": [{"code": "QUOTA_EXCEEDED"}]}})])
        with self.assertRaisesRegex(m.Halt, "no other zone"):
            m.finish_intent(api, self.state, sleep=lambda _: None)
        self.assertEqual(len(api.calls), 1)

    def test_quota_and_permission_errors_stop_immediately(self):
        for status in (400, 401, 403, 409, 429):
            with self.subTest(status=status):
                api = FakeApi([("POST", "requestId=", m.ApiError(status, {"error": "blocked"}))])
                with self.assertRaises(m.ApiError):
                    m.finish_intent(api, self.state, sleep=lambda _: None)
                self.assertEqual(len(api.calls), 1)
                self.assertIsNotNone(self.state.data["intent"])

    def test_capacity_advances_only_after_empty_inventory_and_no_disk(self):
        failed = {"name": "op1", "status": "DONE", "error": {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED"}]}}
        api = FakeApi([("POST", "requestId=", failed),
            ("GET", "aggregated/instances", {"items": {}}),
            ("GET", "/disks/soup-", m.ApiError(404, {}))])
        self.assertFalse(m.finish_intent(api, self.state, sleep=lambda _: None))
        self.assertIsNone(self.state.data["intent"])
        self.assertEqual(self.state.data["next_zone"], 1)

    def test_orphan_disk_stops_capacity_retry(self):
        failed = {"name": "op1", "status": "DONE", "error": {"errors": [{"code": "RESOURCE_POOL_EXHAUSTED"}]}}
        api = FakeApi([("POST", "requestId=", failed),
            ("GET", "aggregated/instances", {"items": {}}),
            ("GET", "/disks/soup-", {"name": "leftover"})])
        with self.assertRaisesRegex(m.Halt, "Disk .* exists"):
            m.finish_intent(api, self.state, sleep=lambda _: None)
        self.assertIsNotNone(self.state.data["intent"])
        self.assertEqual(self.state.data["next_zone"], 0)

    def test_partial_inventory_and_multiple_vms_stop(self):
        for payload in ({"unreachables": ["zone"]}, {"items": {"zone": {"warning": {"code": "UNREACHABLE"}}}}):
            with self.assertRaises(m.Halt):
                m.inventory(FakeApi([("GET", "aggregated/instances", payload)]))
        vms = [{"name": "redacted-machine"}, {"name": "redacted-machine"}]
        with self.assertRaisesRegex(m.Halt, "Multiple"):
            m.remember_existing(FakeApi([("GET", "aggregated/instances", {"items": {"x": {"instances": vms}}})]), self.state)

    def test_existing_running_vm_stops_creation(self):
        vm = {"name": "redacted-machine", "zone": "zones/asia-northeast1-c",
              "status": "RUNNING", "labels": {"recovery-tool": "create-one-t4-v1"}}
        api = FakeApi([("GET", "aggregated/instances", {"items": {"z": {"instances": [vm]}}})])
        m.run(api, self.state, 30, 1)
        self.assertEqual(len(api.calls), 1)
        self.assertIsNotNone(self.state.data["winner"])

    def test_uncertain_after_three_attempts_keeps_intent(self):
        api = FakeApi([("POST", "requestId=saved-uuid", m.ApiError(503, {}))] * 3)
        with self.assertRaisesRegex(m.Halt, "uncertain"):
            m.finish_intent(api, self.state, sleep=lambda _: None)
        self.assertEqual(len({path for _, path, _ in api.calls}), 1)
        self.assertIsNotNone(self.state.data["intent"])

    def test_configuration_change_cannot_bypass_saved_intent(self):
        changed = dict(CONFIG, model="STANDARD")
        with self.assertRaisesRegex(m.Halt, "Arguments differ"):
            m.State(self.state.path, changed)

    def test_spot_and_snapshot_preserve_disk_and_stop_on_preemption(self):
        config = dict(CONFIG, snapshot="https://compute.googleapis.com/compute/v1/projects/test-project/global/snapshots/backup")
        body = m.make_body(config, CONFIG["zones"][0], None)
        self.assertEqual(body["scheduling"]["instanceTerminationAction"], "STOP")
        self.assertFalse(body["scheduling"]["automaticRestart"])
        self.assertFalse(body["disks"][0]["autoDelete"])
        self.assertIn("sourceSnapshot", body["disks"][0]["initializeParams"])
        self.assertNotIn("sourceImage", body["disks"][0]["initializeParams"])


if __name__ == "__main__":
    unittest.main()
