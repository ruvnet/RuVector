"""Frozen evaluator for transport command-boundary validation.

Authored against parent b65e0c7cad47e5ad97ad64a55091797694967cff.
The inline subprocess mirrors rd_app_byte: CR is ignored and LF terminates one
command and emits exactly one JSON reply.
"""

import json
from pathlib import Path
import sys
import textwrap
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from transport import Device


FAKE_FIRMWARE = textwrap.dedent(
    r"""
    import json
    import sys

    sequence = 0

    def reply(command):
        global sequence
        sequence += 1
        payload = {"command": command.decode("utf-8"), "sequence": sequence}
        sys.stdout.buffer.write(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()

    sys.stdout.buffer.write(b'{"command":"startup","sequence":0}\n')
    sys.stdout.buffer.flush()

    command = bytearray()
    while True:
        byte = sys.stdin.buffer.read(1)
        if not byte:
            break
        if byte == b"\r":
            continue
        elif byte == b"\n":
            reply(bytes(command))
            command.clear()
        else:
            command.extend(byte)
    """
)


SPLIT_COMMANDS = {
    "embedded_lf": "alpha\nbeta",
    "embedded_cr": "alpha\rbeta",
    "embedded_crlf": "alpha\r\nbeta",
    "embedded_lfcr": "alpha\n\rbeta",
}


class TransportBoundaryTests(unittest.TestCase):
    def make_device(self):
        device = Device(command=[sys.executable, "-u", "-c", FAKE_FIRMWARE])
        self.addCleanup(device.close)
        self.assertEqual(device.meta, {"command": "startup", "sequence": 0})
        return device

    def assert_split_command_rejected_without_write(self, text):
        device = self.make_device()
        invalid_reply = None
        error = None
        try:
            invalid_reply = device.query(text)
        except Exception as caught:
            error = caught

        followup = device.query("meta")
        observed = {
            "exception": type(error).__name__ if error is not None else None,
            "invalid_reply": invalid_reply,
            "followup": followup,
        }
        expected = {
            "exception": "ValueError",
            "invalid_reply": None,
            "followup": {"command": "meta", "sequence": 1},
        }
        self.assertEqual(observed, expected)

    def test_rejects_embedded_lf_before_write(self):
        self.assert_split_command_rejected_without_write(SPLIT_COMMANDS["embedded_lf"])

    def test_rejects_embedded_cr_before_write(self):
        self.assert_split_command_rejected_without_write(SPLIT_COMMANDS["embedded_cr"])

    def test_rejects_embedded_crlf_before_write(self):
        self.assert_split_command_rejected_without_write(SPLIT_COMMANDS["embedded_crlf"])

    def test_rejects_embedded_lfcr_before_write(self):
        self.assert_split_command_rejected_without_write(SPLIT_COMMANDS["embedded_lfcr"])

    def test_valid_single_line_commands_preserve_spaces_and_tabs(self):
        device = self.make_device()
        commands = (
            "meta",
            "predict sample 7",
            "predict\tsample\t7",
            " \t padded command \t ",
        )
        for sequence, command in enumerate(commands, start=1):
            with self.subTest(command=repr(command)):
                self.assertEqual(
                    device.query(command),
                    {"command": command, "sequence": sequence},
                )


if __name__ == "__main__":
    unittest.main()
