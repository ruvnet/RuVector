"""Host-side contract tests for tools/ota.py against a simulated board.

FakeBoard mirrors main/ota.c: `ota status`, `ota begin <bytes> <sha256>`, a raw
byte stream acknowledged per 1 KiB block, digest check, restart, then either a
self-test commit or a rollback to the previous slot. Physical behaviour is
recorded separately in tests/evidence/ota-esp32c6-physical.json.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import ota

BLOCK = 1024


class FakeBoard:
    healthy = True
    running = 'ota_0'

    def __init__(self, port, baudrate, timeout):
        self.out = []
        self.line = b''
        self.stream = None
        self.writes = []

    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def reset_input_buffer(self): pass
    def flush(self): pass

    def emit(self, obj): self.out.append((json.dumps(obj) + '\n').encode())

    def readline(self): return self.out.pop(0) if self.out else b''

    def status(self):
        other = 'ota_1' if FakeBoard.running == 'ota_0' else 'ota_0'
        self.emit({'ota': 'status', 'running': FakeBoard.running, 'state': 'valid', 'next_update': other})

    def write(self, data):
        self.writes.append(bytes(data))
        if self.stream is not None:
            self.stream['buf'] += data
            got = len(self.stream['buf'])
            if got < self.stream['total']:
                if got % BLOCK == 0: self.emit({'ota_ack': got})
                return
            if hashlib.sha256(self.stream['buf']).hexdigest() != self.stream['digest']:
                self.stream = None
                self.emit({'error': 'ota_digest', 'accepted': False}); return
            target = 'ota_1' if FakeBoard.running == 'ota_0' else 'ota_0'
            self.stream = None
            self.emit({'ota': 'done', 'slot': target, 'restart': True})
            self.emit({'event': 'ready', 'selftest_pass': FakeBoard.healthy})
            if FakeBoard.healthy:
                FakeBoard.running = target
                self.emit({'ota': 'committed', 'slot': target})
            else:
                self.emit({'ota': 'rollback', 'slot': target, 'reason': 'selftest'})
                self.emit({'event': 'ready', 'selftest_pass': True})
            return
        self.line += data
        while b'\n' in self.line:
            cmd, self.line = self.line.split(b'\n', 1)
            cmd = cmd.decode()
            if cmd == 'ota status': self.status()
            elif cmd.startswith('ota begin '):
                size, digest = cmd[10:].split(' ')
                self.stream = {'total': int(size), 'digest': digest, 'buf': b''}
                self.emit({'ota': 'ready', 'bytes': int(size), 'block': BLOCK})


def app_image(size=3000):
    return bytes([0xE9]) + bytes(range(256)) * (size // 256) + b'\x00' * (size % 256 - 1)


class OtaToolTests(unittest.TestCase):
    def setUp(self):
        FakeBoard.running, FakeBoard.healthy = 'ota_0', True
        self.patches = [mock.patch.dict(sys.modules, {'serial': types.SimpleNamespace(Serial=FakeBoard)}),
                        mock.patch.object(ota.time, 'sleep', lambda s: None)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches: p.stop()

    def run_update(self, data):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'app.bin'
        path.write_bytes(data)
        return ota.update('fake', path)

    def test_committed_update_switches_slot(self):
        data = app_image(3000)
        report = self.run_update(data)
        self.assertEqual(report['outcome'], 'committed')
        self.assertTrue(report['switched'])
        self.assertEqual(report['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual((report['before']['running'], report['after']['running']), ('ota_0', 'ota_1'))

    def test_failed_selftest_reports_rollback_and_keeps_slot(self):
        FakeBoard.healthy = False
        report = self.run_update(app_image(2048))
        self.assertEqual(report['outcome'], 'rollback')
        self.assertFalse(report['switched'])
        self.assertEqual(report['after']['running'], 'ota_0')

    def test_rejects_non_image_and_factory_image_before_sending(self):
        with self.assertRaisesRegex(ValueError, 'magic'):
            self.run_update(b'\x00' * 100)
        factory = bytearray(app_image(0x9000)); factory[0x8000:0x8002] = b'\xaa\x50'
        with self.assertRaisesRegex(ValueError, 'factory image'):
            self.run_update(bytes(factory))


if __name__ == '__main__':
    unittest.main()
