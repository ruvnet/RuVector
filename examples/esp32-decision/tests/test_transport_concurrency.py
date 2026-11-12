"""Frozen evaluator for serial transport transaction concurrency.

Authored against parent 47b042815052be2263498d0ca9ae2ae1f31cd7e1.
The fake adapter is installed as ``serial.Serial`` so every check exercises the
normal ``Device(port=...)`` construction path.  It emits one ordered JSON reply
per command, matching the firmware's line-oriented request/reply contract.
"""

from collections import deque
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import transport
from transport import Device


PARENT_SHA = "47b042815052be2263498d0ca9ae2ae1f31cd7e1"
PAIR_COUNT = 8


class ControlledSerial:
    """Thread-safe reply queue with a controlled pause after alpha's write."""

    instances = []

    def __init__(self, port, baudrate, timeout, write_timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self._condition = threading.Condition()
        self._replies = deque()
        self._last_command = {}
        self.alpha_flush_entered = threading.Event()
        self.alpha_reply_consumed_by_beta = threading.Event()
        self.release_alpha_flush = threading.Event()
        self.block_alpha_flush = False
        self.closed = False
        self.__class__.instances.append(self)

    def reset_input_buffer(self):
        return None

    def write(self, payload):
        command = payload.decode("utf-8").removesuffix("\n")
        reply = json.dumps(
            {"command": command, "serial_port": self.port}, sort_keys=True
        ).encode("utf-8") + b"\n"
        ident = threading.get_ident()
        with self._condition:
            self._last_command[ident] = command
            self._replies.append((command, reply))
            self._condition.notify_all()
        return len(payload)

    def flush(self):
        if (
            self.block_alpha_flush
            and self._last_command.get(threading.get_ident()) == "alpha"
        ):
            self.alpha_flush_entered.set()
            if not self.release_alpha_flush.wait(2):
                raise TimeoutError("test coordinator did not release alpha flush")

    def read_until(self, delimiter, size):
        del delimiter
        deadline = transport.time.monotonic() + self.timeout
        with self._condition:
            while not self._replies:
                remaining = deadline - transport.time.monotonic()
                if remaining <= 0:
                    return b""
                self._condition.wait(remaining)
            command, reply = self._replies.popleft()
        if threading.current_thread().name.startswith("beta-") and command == "alpha":
            self.alpha_reply_consumed_by_beta.set()
        return reply[:size]

    def close(self):
        self.closed = True


class TransportConcurrencyTests(unittest.TestCase):
    def make_device(self, pair):
        ControlledSerial.instances.clear()
        serial_module = types.SimpleNamespace(Serial=ControlledSerial)
        with mock.patch.dict(sys.modules, {"serial": serial_module}):
            with mock.patch.object(transport.time, "sleep", return_value=None):
                device = Device(port=f"fake://pair-{pair}")
        self.addCleanup(device.close)
        self.assertEqual(
            device.meta,
            {"command": "meta", "serial_port": f"fake://pair-{pair}"},
        )
        return device, ControlledSerial.instances[-1]

    @staticmethod
    def query_in_thread(device, command, results, errors):
        try:
            results[command] = device.query(command)
        except Exception as error:  # Preserve the worker failure for the assertion.
            errors[command] = error

    def test_concurrent_callers_receive_their_own_replies(self):
        for pair in range(PAIR_COUNT):
            with self.subTest(pair=pair, parent=PARENT_SHA):
                device, serial = self.make_device(pair)
                serial.block_alpha_flush = True
                results = {}
                errors = {}
                alpha = threading.Thread(
                    target=self.query_in_thread,
                    args=(device, "alpha", results, errors),
                    name=f"alpha-{pair}",
                )
                beta = threading.Thread(
                    target=self.query_in_thread,
                    args=(device, "beta", results, errors),
                    name=f"beta-{pair}",
                )

                alpha.start()
                self.assertTrue(serial.alpha_flush_entered.wait(1))
                beta.start()

                # On the unlocked parent beta reaches read_until and consumes
                # alpha's queued reply.  A serialized candidate keeps beta out;
                # release alpha after this bounded observation either way.
                serial.alpha_reply_consumed_by_beta.wait(0.25)
                serial.release_alpha_flush.set()
                alpha.join(2)
                beta.join(2)

                self.assertFalse(alpha.is_alive())
                self.assertFalse(beta.is_alive())
                self.assertEqual(errors, {})
                self.assertEqual(
                    results,
                    {
                        "alpha": {
                            "command": "alpha",
                            "serial_port": f"fake://pair-{pair}",
                        },
                        "beta": {
                            "command": "beta",
                            "serial_port": f"fake://pair-{pair}",
                        },
                    },
                )

    def test_sequential_queries_preserve_request_reply_order(self):
        device, _ = self.make_device("sequential")
        for command in ("alpha", "beta", "meta", "predict sample 7"):
            with self.subTest(command=command):
                self.assertEqual(
                    device.query(command),
                    {"command": command, "serial_port": "fake://pair-sequential"},
                )

    def assert_split_command_rejected_without_write(self, text):
        device, _ = self.make_device(repr(text))
        with self.assertRaises(ValueError):
            device.query(text)
        self.assertEqual(
            device.query("meta"),
            {"command": "meta", "serial_port": f"fake://pair-{text!r}"},
        )

    def test_rejects_embedded_lf_before_write(self):
        self.assert_split_command_rejected_without_write("alpha\nbeta")

    def test_rejects_embedded_cr_before_write(self):
        self.assert_split_command_rejected_without_write("alpha\rbeta")

    def test_rejects_embedded_crlf_before_write(self):
        self.assert_split_command_rejected_without_write("alpha\r\nbeta")

    def test_rejects_embedded_lfcr_before_write(self):
        self.assert_split_command_rejected_without_write("alpha\n\rbeta")


if __name__ == "__main__":
    unittest.main()
