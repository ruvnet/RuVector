"""Independent finite benchmark evaluator, frozen before candidate implementation.

Finite fixture captured from parent 6fdb5f12ac7fe16752eeb60405e52567081706db.
Synthetic physical provenance exercises software gates only; no board measurement.
"""
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import rig
from energy_compare import compare_energy


def rounds(a=1.2, b=1.0, count=11):
    return [{'baseline': {'profile': {'mean_cycles': a}},
             'candidate': {'profile': {'mean_cycles': b}}} for _ in range(count)]


def energy_pairs(a=1.2, b=1.0, count=11):
    return [{arm: dict(trace_sha256=f'{i}:{arm}', arm=arm,
                       model_sha256='synthetic-model', target='esp32s3',
                       scope='kernel_batch', decisions=256, physical_hardware=True,
                       execution='physical', gross_joules_per_decision=value)
             for arm, value in [('baseline', a), ('candidate', b)]}
            for i in range(count)]


class FiniteBenchmarkTests(unittest.TestCase):
    def test_frozen_finite_parent_fixture_exact(self):
        values = [1.04, 1.08, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28]
        records = [rounds(v, 1, 1)[0] for v in values]
        expected = dict(metric='mean_cycles', paired_speedups=values,
                        median_speedup=1.18, bootstrap_95_lower=1.12,
                        minimum_rounds=11, required_speedup=1.10,
                        correctness_pass=True, physical_execution=True, retain=True)
        self.assertEqual(rig.retention(records, 'mean_cycles', 'physical', True), expected)

    def test_existing_thresholds_and_provenance_remain(self):
        for count, gain, execution, correct, expected in [
            (11, 1.1, 'physical', True, True),
            (10, 1.2, 'physical', True, False),
            (11, 1.099, 'physical', True, False),
            (11, 1.2, 'physical', False, False),
            (11, 1.2, 'native', True, False),
            (11, 1.2, 'emulator', True, False),
        ]:
            with self.subTest(count=count, gain=gain, execution=execution, correct=correct):
                result = rig.retention(rounds(gain, 1, count), 'mean_cycles', execution, correct)
                self.assertEqual(result['retain'], expected)
                self.assertEqual(result['minimum_rounds'], 11)
                self.assertEqual(result['required_speedup'], 1.10)
        # Median gain alone cannot overcome a confidence bound at or below one.
        records = [rounds(v, 1, 1)[0] for v in [0.9] * 5 + [1.2] * 6]
        self.assertFalse(rig.retention(records, 'mean_cycles', 'physical', True)['retain'])

    def test_reject_invalid_numeric_inputs_in_every_arm(self):
        bad = [float('nan'), float('inf'), float('-inf'), True, False,
               '1.2', None, [], {}, Decimal('1.2'), 0, -1, -0.0]
        for value in bad:
            for arm in ('baseline', 'candidate'):
                with self.subTest(value=repr(value), arm=arm):
                    records = rounds()
                    records[5][arm]['profile']['mean_cycles'] = value
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        rig.retention(records, 'mean_cycles', 'physical', True)

    def test_reject_huge_integer_conversion_overflow(self):
        for a, b in [(10 ** 1000, 10 ** 999), (10 ** 1000, 1), (1, 10 ** 1000)]:
            with self.subTest(a_digits=len(str(a)), b_digits=len(str(b))):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    rig.retention(rounds(a, b), 'mean_cycles', 'physical', True)

    def test_reject_nonfinite_and_underflowed_ratios(self):
        for a, b in [(1e308, 1e-308), (1e-308, 1e308)]:
            with self.subTest(a=a, b=b):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    rig.retention(rounds(a, b), 'mean_cycles', 'physical', True)

    def test_reject_nonfinite_aggregate_statistics(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=repr(value)):
                with patch.object(rig.statistics, 'median', return_value=value):
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        rig.retention(rounds(), 'mean_cycles', 'physical', True)

    def test_energy_inherits_invalid_numeric_rejection(self):
        for value in (float('nan'), float('inf'), float('-inf'), True,
                      False, '1.2', None, 0, -1):
            for arm in ('baseline', 'candidate'):
                with self.subTest(value=repr(value), arm=arm):
                    pairs = energy_pairs()
                    pairs[5][arm]['gross_joules_per_decision'] = value
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        compare_energy(pairs, True)
        for a, b in [(1e308, 1e-308), (1e-308, 1e308), (10 ** 1000, 10 ** 999)]:
            with self.subTest(a=repr(a)[:20], b=repr(b)[:20]):
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    compare_energy(energy_pairs(a, b), True)

    def test_energy_finite_parity_and_existing_gates(self):
        result = compare_energy(energy_pairs(), True)
        expected = rig.retention(rounds(), 'mean_cycles', 'physical', True)
        expected['metric'] = 'joules_per_decision'
        self.assertEqual(result, expected)
        self.assertFalse(compare_energy(energy_pairs(count=10), True)['retain'])
        self.assertFalse(compare_energy(energy_pairs(), False)['retain'])
        pairs = energy_pairs()
        pairs[0]['candidate']['physical_hardware'] = False
        self.assertFalse(compare_energy(pairs, True)['retain'])
        pairs = energy_pairs()
        pairs[1]['candidate']['trace_sha256'] = pairs[0]['candidate']['trace_sha256']
        with self.assertRaises(ValueError):
            compare_energy(pairs, True)

    def test_real_json_energy_cli_fails_closed(self):
        # 1e309 is valid JSON numeric syntax but Python decodes it as infinity.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder / 'rig.json'
            report.write_text(json.dumps({'model_sha256': 'synthetic-model',
                                          'acceptance': {'correctness_pass': True}}))
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            paths = []
            for i, pair in enumerate(energy_pairs()):
                item = {}
                for arm, value in pair.items():
                    value['rig_report_sha256'] = digest
                    path = folder / f'{i}-{arm}.json'
                    text = json.dumps(value)
                    if arm == 'baseline':
                        text = text.replace('"gross_joules_per_decision": 1.2',
                                            '"gross_joules_per_decision": 1e309')
                    path.write_text(text)
                    item[arm] = str(path)
                paths.append(item)
            pairs_path = folder / 'pairs.json'
            pairs_path.write_text(json.dumps(paths))
            output = folder / 'result.json'
            process = subprocess.run([sys.executable, str(TOOLS / 'energy_compare.py'),
                                      str(pairs_path), '--rig-report', str(report),
                                      '--output', str(output)], capture_output=True, text=True,
                                     timeout=20)
            self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertFalse(output.exists(), 'Malformed energy evidence must not emit a result')


if __name__ == '__main__':
    unittest.main()
