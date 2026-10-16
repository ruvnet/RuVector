"""RuGraph process bindings. Binary and model paths are trusted host configuration."""
from __future__ import annotations

import json
import os
import math
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping

_LIMIT = 16 * 1024 * 1024


class RuGraphError(RuntimeError):
    def __init__(self, code: str, message: str, position: int | None = None):
        super().__init__(message)
        self.code = code
        self.position = position


def _encode(value: Any) -> str:
    try:
        result = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise RuGraphError("INVALID_INPUT", str(error)) from error
    if len(result.encode("utf-8")) > _LIMIT:
        raise RuGraphError("INPUT_LIMIT", "Input exceeds 16 MiB")
    return result


def _models(path: str | None) -> list[str]:
    if path is None:
        return []
    if not isinstance(path, str) or not path or "\0" in path or path == "-" or path.startswith("--"):
        raise RuGraphError("INVALID_INPUT", "Invalid models_path")
    return ["--models", path]


class RuGraph:
    def __init__(self, binary: str | None = None, timeout: float = 30):
        self.binary = os.environ.get("RUVECTOR_LANGUAGE_BIN", "rvg") if binary is None else binary
        if (not isinstance(self.binary, str) or not self.binary or "\0" in self.binary
                or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= 300):
            raise RuGraphError("INVALID_INPUT", "Invalid executable or timeout")
        self.timeout = timeout

    def _invoke(self, args: list[str], payload: str) -> dict[str, Any]:
        if not isinstance(payload, str):
            raise RuGraphError("INVALID_INPUT", "Expected string source")
        try:
            encoded = payload.encode("utf-8")
        except UnicodeError as error:
            raise RuGraphError("INVALID_INPUT", "Source must be valid UTF-8") from error
        if len(encoded) > _LIMIT:
            raise RuGraphError("INPUT_LIMIT", "Input exceeds 16 MiB")
        try:
            code, stdout, stderr = _execute([self.binary, *args], encoded, self.timeout)
        except OSError as error:
            raise RuGraphError("EXECUTABLE_ERROR", str(error)) from error
        try:
            out, err = stdout.decode("utf-8"), stderr.decode("utf-8")
        except UnicodeError as error:
            raise RuGraphError("INVALID_OUTPUT", "Runtime returned invalid UTF-8") from error
        if code:
            try:
                data = _decode(err)
            except RuGraphError:
                data = {}
            position = data.get("position")
            if (not isinstance(data.get("code"), str) or not data["code"]
                    or not isinstance(data.get("message"), str)
                    or (position is not None and (type(position) is not int or position < 0))):
                raise RuGraphError("PROCESS_ERROR", err or f"Exit {code}")
            raise RuGraphError(data["code"], data["message"], position)
        return _decode(out)

    def check(self, source: str) -> dict[str, Any]:
        return self._invoke(["check", "-"], source)

    def compile(self, source: str) -> dict[str, Any]:
        return self._invoke(["compile", "-"], source)

    def run(self, request: Mapping[str, Any], *, models_path: str | None = None) -> dict[str, Any]:
        return self._invoke(["run", "--request", "-", *_models(models_path)], _encode(request))

    def audit(self, request: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
        encoded_request, encoded_options = _encode(request), _encode(options)
        try:
            with tempfile.TemporaryDirectory(prefix="rugraph-audit-") as directory:
                path = Path(directory) / "options.json"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    output.write(encoded_options)
                return self._invoke(["audit", "--request", "-", "--audit-options", str(path)], encoded_request)
        except OSError as error:
            raise RuGraphError("IO_ERROR", str(error)) from error

    def replay(self, request: Mapping[str, Any], receipt: Mapping[str, Any], *, models_path: str | None = None) -> dict[str, Any]:
        encoded_request, encoded_receipt = _encode(request), _encode(receipt)
        models = _models(models_path)
        try:
            with tempfile.TemporaryDirectory(prefix="ruvector-language-") as directory:
                path = Path(directory) / "receipt.json"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    output.write(encoded_receipt)
                return self._invoke(["replay", "--request", "-", "--receipt", str(path), *models], encoded_request)
        except OSError as error:
            raise RuGraphError("IO_ERROR", str(error)) from error


def _decode(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON number: {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result

    try:
        data = json.loads(text, parse_constant=reject_constant,
                          parse_float=finite_float, object_pairs_hook=object_pairs)
        if not isinstance(data, dict):
            raise ValueError("Runtime must return a JSON object")
        return data
    except (ValueError, TypeError, RecursionError) as error:
        raise RuGraphError("INVALID_OUTPUT", "Runtime returned invalid JSON") from error


def _execute(args: list[str], payload: bytes, timeout: float) -> tuple[int, bytes, bytes]:
    """Drain both streams concurrently and stop before buffering over 16 MiB."""
    with tempfile.TemporaryFile() as input_file:
        input_file.write(payload)
        input_file.seek(0)
        process = subprocess.Popen(args, stdin=input_file, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=False,
                                   start_new_session=os.name == "posix")
        deadline = time.monotonic() + timeout
        lock = threading.Lock()
        state: dict[str, Any] = {"size": 0, "error": None}
        output = [bytearray(), bytearray()]

        def terminate() -> None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass

        def drain(stream: Any, index: int) -> None:
            try:
                while True:
                    chunk = stream.read1(64 * 1024)
                    if not chunk:
                        return
                    with lock:
                        if state["error"] is not None:
                            return
                        if state["size"] + len(chunk) > _LIMIT:
                            state["error"] = RuGraphError("OUTPUT_LIMIT", "Output exceeds 16 MiB")
                        else:
                            state["size"] += len(chunk)
                            output[index].extend(chunk)
                    if state["error"] is not None:
                        terminate()
                        return
            except OSError as error:
                with lock:
                    if state["error"] is None:
                        state["error"] = RuGraphError("IO_ERROR", str(error))
                terminate()
            finally:
                stream.close()

        readers = [threading.Thread(target=drain, args=(stream, index), daemon=True,
                                    name=f"ruvector-language-{process.pid}-{index}")
                   for index, stream in enumerate((process.stdout, process.stderr))]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
            for reader in readers:
                reader.join(timeout=max(0, deadline - time.monotonic()))
            if any(reader.is_alive() for reader in readers):
                raise subprocess.TimeoutExpired(args, timeout)
        except subprocess.TimeoutExpired as error:
            terminate()
            process.wait()
            for reader in readers:
                reader.join(timeout=1)
            if state["error"] is not None:
                raise state["error"]
            raise RuGraphError("TIMEOUT", "RuGraph exceeded the process timeout") from error
        finally:
            if process.poll() is None:
                terminate()
                process.wait()
        if state["error"] is not None:
            raise state["error"]
        return process.returncode, bytes(output[0]), bytes(output[1])
