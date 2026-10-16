import sys
import threading
import unittest

from ruvector_language import RuGraph, RuGraphError


class RuGraphBoundaries(unittest.TestCase):
    def child(self, source, timeout=2):
        try:
            return RuGraph(binary=sys.executable, timeout=timeout)._invoke(["-c", source], "")
        finally:
            self.assertFalse(any(t.name.startswith("ruvector-language-") for t in threading.enumerate()))

    def assert_code(self, expected, action):
        with self.assertRaises(RuGraphError) as error:
            action()
        self.assertEqual(error.exception.code, expected)

    def test_output_bound_while_running(self):
        self.assert_code("OUTPUT_LIMIT", lambda: self.child(
            "import sys; b=b'A'*(9*1024*1024); sys.stdout.buffer.write(b); sys.stderr.buffer.write(b)", 5))

    def test_timeout_and_missing_executable(self):
        self.assert_code("TIMEOUT", lambda: self.child("import time; time.sleep(10)", 0.03))
        self.assert_code("EXECUTABLE_ERROR", lambda: RuGraph(binary="/no/such/binary").check(""))

    def test_bad_error_shapes_preserve_error_contract(self):
        for data in ["null", "[]", '{"code":1,"message":2}',
                     '{"code":"X","message":"bad","position":true}']:
            self.assert_code("PROCESS_ERROR", lambda: self.child(
                f"import sys; sys.stderr.write({data!r}); sys.exit(2)"))

    def test_invalid_outputs_and_nonfinite_numbers(self):
        for data in ["[]", "null", '{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}']:
            self.assert_code("INVALID_OUTPUT", lambda: self.child(f"print({data!r})"))
        self.assert_code("INVALID_OUTPUT", lambda: self.child(
            "import sys; sys.stdout.buffer.write(bytes([255]))"))

    def test_invalid_host_configuration_and_source(self):
        for timeout in [True, "30", float("nan"), float("inf"), 0, 301]:
            self.assert_code("INVALID_INPUT", lambda: RuGraph(timeout=timeout))
        for binary in ["", "bad\0path", 4]:
            self.assert_code("INVALID_INPUT", lambda: RuGraph(binary=binary))
        self.assert_code("INVALID_INPUT", lambda: RuGraph().check({}))
        self.assert_code("INVALID_INPUT", lambda: RuGraph().check("\ud800"))
        self.assert_code("INVALID_INPUT", lambda: RuGraph().run({}, models_path="bad\0path"))

    def test_circular_input_is_a_structured_error(self):
        cycle = {}
        cycle["self"] = cycle
        self.assert_code("INVALID_INPUT", lambda: RuGraph().run(cycle))


if __name__ == "__main__":
    unittest.main()
