"""Host contract for negotiated link speed (tools/transport.negotiate_baud).

The simulated board mirrors main/main.c: `baud <rate>` replies at the old rate,
then only a `baud ok` line received at the new rate confirms the switch. The
bare newline the host sends first yields a rejected empty line, which the host
must skip. Physical behaviour is recorded in tests/evidence/baud-*.json.
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from transport import negotiate_baud


class FakeBoard:
    def __init__(self, accept=True, confirm=True):
        self.accept, self.confirm = accept, confirm
        self.baudrate, self.out, self.line, self.writes = 115200, [], b'', []

    def emit(self, obj): self.out.append((json.dumps(obj) + '\n').encode())
    def reset_input_buffer(self): self.out.clear()
    def flush(self): pass
    def read_until(self, sep, size): return self.out.pop(0) if self.out else b''

    def write(self, data):
        self.writes.append((self.baudrate, bytes(data)))
        self.line += data
        while b'\n' in self.line:
            cmd, self.line = self.line.split(b'\n', 1)
            cmd = cmd.decode()
            if cmd.startswith('baud ') and cmd != 'baud ok':
                rate = int(cmd[5:])
                if self.accept: self.emit({'baud': rate, 'confirm': 'baud ok', 'within_ms': 2000})
                else: self.emit({'error': 'baud_rate', 'accepted': False})
            elif cmd == 'baud ok' and self.confirm:
                self.emit({'baud': self.baudrate, 'confirmed': True})
            else:
                self.emit({'error': 'unknown_command', 'accepted': False})


class BaudTests(unittest.TestCase):
    def test_switches_and_confirms_at_new_rate(self):
        board = FakeBoard()
        self.assertEqual(negotiate_baud(board, 921600), 921600)
        self.assertEqual(board.baudrate, 921600)
        self.assertEqual(board.writes[0], (115200, b'baud 921600\n'))
        self.assertEqual(board.writes[-1], (921600, b'\nbaud ok\n'))

    def test_refused_rate_keeps_link(self):
        board = FakeBoard(accept=False)
        with self.assertRaises(RuntimeError):
            negotiate_baud(board, 460800)
        self.assertEqual(board.baudrate, 115200)

    def test_missing_confirmation_times_out(self):
        with self.assertRaises(TimeoutError):
            negotiate_baud(FakeBoard(confirm=False), 230400, timeout=0.2)

    def test_rejects_unlisted_rate_before_writing(self):
        board = FakeBoard()
        with self.assertRaises(ValueError):
            negotiate_baud(board, 2000000)
        self.assertEqual(board.writes, [])


if __name__ == '__main__':
    unittest.main()
