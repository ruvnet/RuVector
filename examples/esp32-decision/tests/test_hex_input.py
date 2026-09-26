"""Exact hex-float input (`inferx`) must decide identically to decimal `infer`.

Each random binary32 value is sent twice: as `%.9g` decimal text, which
round-trips exactly through strtof, and as its 8-hex-digit bit pattern. Every
reply field except timing must match. Malformed and non-finite hex values must
be rejected with the same error contract as the decimal path.
"""
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e2e import ROOT, CORE, INCLUDE


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def hexf(x):
    return struct.pack('>f', x).hex()


class HexInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / 'firmware'
        subprocess.run(['gcc', '-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'main'), '-I', str(INCLUDE), str(CORE), str(ROOT / 'main/app.c'),
                        str(ROOT / 'main/profile.c'), str(ROOT / 'main/sensor.c'), str(ROOT / 'tests/host_main.c'),
                        '-lm', '-o', str(cls.binary)], check=True)
        boot = subprocess.run([str(cls.binary)], input='meta\n', text=True, capture_output=True, check=True, timeout=30)
        cls.dims = json.loads(boot.stdout.splitlines()[0])['dims']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_commands(self, commands):
        p = subprocess.run([str(self.binary)], input='\n'.join(commands) + '\n', text=True,
                           capture_output=True, check=True, timeout=60)
        replies = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual(len(replies), len(commands) + 1)
        return replies[1:]

    def test_hex_matches_decimal_for_random_rows(self):
        rng = random.Random(20260926)
        rows = [[f32(rng.uniform(-1.5, 1.5)) for _ in range(self.dims)] for _ in range(400)]
        rows += [[f32(v) for v in (0.0, -0.0, 1e-38, -3.4e38, 1.17549435e-38)[:1] * self.dims]]
        commands = []
        for row in rows:
            commands.append('infer ' + ' '.join('%.9g' % v for v in row))
            commands.append('inferx ' + ' '.join(hexf(v) for v in row))
        replies = self.run_commands(commands)
        timing = ('inference_us', 'parse_us', 'preprocess_us')
        self.assertLess(sum('error' in r for r in replies[:800:2]), 4, 'random rows should be valid inputs')
        self.assertIn('error', replies[-2], 'zero vector is rejected by the decimal path too')
        for i in range(0, len(replies), 2):
            text, hexr = replies[i], replies[i + 1]
            self.assertEqual({k: v for k, v in text.items() if k not in timing},
                             {k: v for k, v in hexr.items() if k not in timing}, f'row {i // 2}')

    def test_rejects_malformed_and_non_finite_hex(self):
        ok = ' '.join(['3f800000'] * self.dims)
        bad = ['inferx ' + ok.replace('3f800000', '7fc00000', 1),        # NaN
               'inferx ' + ok.replace('3f800000', '7f800000', 1),        # +inf
               'inferx ' + ok.replace('3f800000', 'ff800000', 1),        # -inf
               'inferx ' + ok.replace('3f800000', '3f80000', 1),         # 7 digits
               'inferx ' + ok.replace('3f800000', '3f8000000', 1),       # 9 digits
               'inferx ' + ok.replace('3f800000', '3f80000g', 1),        # non-hex
               'inferx ' + ok.replace('3f800000', '0x3f8000', 1),        # prefix
               'inferx ' + ' '.join(['3f800000'] * (self.dims + 1)),     # too many
               'inferx 3f800000',                                        # too few
               'inferx ',                                                # empty
               'sensorx ' + ok]                                          # no preprocessing in default model
        replies = self.run_commands(bad + ['inferx ' + ok, 'selftest'])
        for command, reply in zip(bad, replies):
            self.assertIn('error', reply, command)
            self.assertFalse(reply['accepted'], command)
        self.assertNotIn('error', replies[-2])
        self.assertTrue(replies[-1]['selftest_pass'])


if __name__ == '__main__':
    unittest.main()
