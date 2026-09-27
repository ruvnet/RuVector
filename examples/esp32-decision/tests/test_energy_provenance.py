"""Frozen independent evaluator for energy evidence provenance.

The evaluator was authored against parent
21fd8753ea1d54473ef797e9c6ebaaa62c7749bf before the candidate change.
All fixtures are synthetic and do not represent physical measurements.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def rig_fixture(rounds=11):
    def arm_fixture(arm):
        return {
            "meta": {
                "kernel_sha256": f"synthetic-{arm}-kernel",
                "target": "esp32s3",
            },
            "image": {
                "sha256": f"synthetic-{arm}-image",
                "bytes": 123456,
            },
            "energy_batch": {
                "energy_batch": True,
                "marker_gpio": 4,
                "runs": 256,
                "batch_us": 100000,
            },
        }

    return {
        "schema": 1,
        "execution": "physical",
        "physical_hardware": True,
        "model_sha256": "synthetic-model",
        "rounds": [
            {
                "baseline": arm_fixture("baseline"),
                "candidate": arm_fixture("candidate"),
                "exact_replay_match": True,
            }
            for _ in range(rounds)
        ],
        "acceptance": {"correctness_pass": True},
        "provenance_note": "Synthetic software gate fixture; no board or power analyzer.",
    }


def energy_fixture(rig, rig_digest):
    pairs = []
    for round_index, rig_round in enumerate(rig["rounds"]):
        pair = {}
        for arm, value in (("baseline", 0.00012), ("candidate", 0.00010)):
            source = rig_round[arm]
            pair[arm] = {
                "schema": 1,
                "execution": "physical",
                "physical_hardware": True,
                "trace_sha256": f"synthetic-trace-{round_index}-{arm}",
                "rig_report_sha256": rig_digest,
                "image": copy.deepcopy(source["image"]),
                "kernel_sha256": source["meta"]["kernel_sha256"],
                "model_sha256": rig["model_sha256"],
                "target": source["meta"]["target"],
                "scope": "kernel_batch",
                "decisions": 256,
                "round": round_index,
                "arm": arm,
                "gross_joules_per_decision": value,
            }
        pairs.append(pair)
    return pairs


def invoke_comparator(folder, rig, mutate=None):
    folder = Path(folder)
    report = folder / "rig.json"
    report.write_text(json.dumps(rig, sort_keys=True))
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    pairs = energy_fixture(rig, digest)
    if mutate is not None:
        mutate(pairs)

    pair_paths = []
    for index, pair in enumerate(pairs):
        paths = {}
        for arm, value in pair.items():
            path = folder / f"energy-{index}-{arm}.json"
            path.write_text(json.dumps(value, sort_keys=True))
            paths[arm] = str(path)
        pair_paths.append(paths)
    pair_file = folder / "pairs.json"
    pair_file.write_text(json.dumps(pair_paths))
    output = folder / "result.json"
    process = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "energy_compare.py"),
            str(pair_file),
            "--rig-report",
            str(report),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return process, output


class EnergyProvenanceTests(unittest.TestCase):
    def assert_rejected(self, mutation):
        with tempfile.TemporaryDirectory() as folder:
            process, output = invoke_comparator(folder, rig_fixture(), mutation)
            self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertFalse(output.exists(), "Invalid provenance must not emit a result")

    def test_valid_reports_preserve_retention_result(self):
        with tempfile.TemporaryDirectory() as folder:
            process, output = invoke_comparator(folder, rig_fixture())
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["retain"])
            self.assertEqual(result["metric"], "joules_per_decision")
            self.assertEqual(len(result["paired_speedups"]), 11)

    def test_rejects_cross_round_pairing(self):
        def mutation(pairs):
            pairs[0]["candidate"]["round"] = 1

        self.assert_rejected(mutation)

    def test_rejects_duplicate_round_reuse_with_distinct_traces(self):
        def mutation(pairs):
            for arm in ("baseline", "candidate"):
                pairs[1][arm]["round"] = 0

        self.assert_rejected(mutation)

    def test_rejects_kernel_mismatch_with_rig_arm(self):
        def mutation(pairs):
            pairs[3]["candidate"]["kernel_sha256"] = "wrong-kernel"

        self.assert_rejected(mutation)

    def test_rejects_target_or_image_mismatch_with_rig_arm(self):
        mutations = {
            "target": lambda pairs: [
                value.__setitem__("target", "esp32c6")
                for pair in pairs
                for value in pair.values()
            ],
            "image": lambda pairs: pairs[4]["candidate"]["image"].__setitem__(
                "sha256", "wrong-image"
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                self.assert_rejected(mutation)

    def test_rejects_decision_count_or_scope_mismatch_with_rig_arm(self):
        mutations = {
            "decisions": lambda pairs: [
                value.__setitem__("decisions", 128)
                for pair in pairs
                for value in pair.values()
            ],
            "scope": lambda pairs: [
                value.__setitem__("scope", "whole_device")
                for pair in pairs
                for value in pair.values()
            ],
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                self.assert_rejected(mutation)


if __name__ == "__main__":
    unittest.main()
