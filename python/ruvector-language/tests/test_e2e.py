import copy
import json
from pathlib import Path
import unittest

from ruvector_language import RuGraph, RuGraphError

EXAMPLES = Path(__file__).resolve().parents[3] / "crates/ruvector-language/examples"


class RuGraphE2E(unittest.TestCase):
    def setUp(self):
        self.client = RuGraph()
        self.request = {
            "program": self.client.compile((EXAMPLES / "incident.rvg").read_text()),
            "snapshot": json.loads((EXAMPLES / "snapshot.json").read_text()),
            "context": json.loads((EXAMPLES / "context.json").read_text()),
        }

    def test_counterfactual_audit_and_aggregate_limit(self):
        options = {"interventions": [{"kind": "source", "id": "telemetry-a"}, {"kind": "edge", "id": "api-db"}], "max_execution_work": 300000}
        report = self.client.audit(self.request, options)
        self.assertEqual(report, self.client.audit(self.request, options))
        self.assertTrue(all(c["after"] == "unknown" for c in report["trials"][0]["changes"]))
        self.assertTrue(all(c["after"] is None for c in report["trials"][1]["changes"]))
        options["max_execution_work"] = 1
        with self.assertRaises(RuGraphError) as error:
            self.client.audit(self.request, options)
        self.assertEqual(error.exception.code, "AUDIT_BUDGET")

    def test_audit_private_file_cleanup_on_success_and_failure(self):
        import os
        for fail in (False, True):
            paths = []
            def invoke(args, payload):
                path = Path(args[-1])
                paths.append(path)
                if os.name == "posix":
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                if fail:
                    raise RuGraphError("TEST", "expected")
                return {}
            client = RuGraph()
            client._invoke = invoke
            if fail:
                with self.assertRaises(RuGraphError):
                    client.audit({}, {})
            else:
                client.audit({}, {})
            self.assertFalse(paths[0].exists())

    def test_native_process_execute_replay(self):
        receipt = self.client.run(self.request)
        self.assertEqual(self.client.replay(self.request, receipt), receipt)
        self.assertTrue(any(x["subject"] == "database" and x["predicate"] == "Degraded" and x["status"] == "supported" for x in receipt["output"]["items"]))

    def test_expiry_and_retraction(self):
        for mutate in (lambda r: r["context"].update(now=1100), lambda r: r["snapshot"].update(observations=[])):
            request = copy.deepcopy(self.request)
            mutate(request)
            receipt = self.client.run(request)
            self.assertTrue(all(x["status"] == "unknown" for x in receipt["output"]["items"]))

    def test_permissions_and_replay_mismatch(self):
        receipt = self.client.run(self.request)
        self.request["context"]["now"] = 1001
        with self.assertRaises(RuGraphError) as result:
            self.client.replay(self.request, receipt)
        self.assertEqual(result.exception.code, "REPLAY_MISMATCH")
        self.request["context"]["capabilities"] = []
        with self.assertRaises(RuGraphError) as result:
            self.client.run(self.request)
        self.assertEqual(result.exception.code, "UNAUTHORIZED")

    def test_nonfinite_rejected(self):
        self.request["context"]["now"] = float("nan")
        with self.assertRaises(RuGraphError) as result:
            self.client.run(self.request)
        self.assertEqual(result.exception.code, "INVALID_INPUT")

    def test_full_u64_timestamps_are_lossless(self):
        maximum = 2**64 - 1
        self.request["context"]["now"] = maximum - 2
        for observation in self.request["snapshot"]["observations"]:
            observation["observed_at"] = maximum - 3
            observation["expires_at"] = maximum
        receipt = self.client.run(self.request)
        self.assertEqual(receipt["now"], maximum - 2)
        observed = [e for d in receipt["output"]["items"] for e in d["evidence"]]
        self.assertTrue(observed)
        self.assertTrue(all(e["expires_at"] == maximum for e in observed))
        self.assertEqual(self.client.replay(self.request, receipt), receipt)


if __name__ == "__main__":
    unittest.main()
