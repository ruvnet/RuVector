"""Real RVF WebAssembly vector queries in a bounded, import-free sandbox."""
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
import math
from itertools import islice
import struct
from threading import RLock
import wasmtime

__all__ = ["VectorStore", "Match"]

@dataclass(frozen=True)
class Match:
    id: int
    distance: float

class VectorStore:
    """In-memory brute-force RVF store; each instance owns an isolated Wasm VM.

    Metrics are squared L2, negative inner product, and cosine distance.
    A trap poisons the instance; construct another instead of reusing it.
    """
    def __init__(self, dimensions: int, metric: str = "l2", *,
                 fuel: int = 10_000_000, memory_bytes: int = 64 * 1024 * 1024):
        if type(dimensions) is not int or not 1 <= dimensions <= 4096:
            raise ValueError("dimensions must be an integer between 1 and 4096")
        if metric not in ("l2", "inner_product", "cosine"):
            raise ValueError("unsupported metric")
        if type(fuel) is not int or not 1 <= fuel <= 1_000_000_000:
            raise ValueError("fuel must be between 1 and 1000000000")
        if type(memory_bytes) is not int or not 2**21 <= memory_bytes <= 2**28:
            raise ValueError("memory_bytes must be between 2 MiB and 256 MiB")
        self.dimensions = dimensions
        self._metric = {"l2": 0, "inner_product": 1, "cosine": 2}[metric]
        self._fuel = fuel
        self._lock = RLock()
        self._closed = False
        self._ids: set[int] = set()
        resources = files(__package__)
        blob = resources.joinpath("rvf_wasm_bg.wasm").read_bytes()
        provenance = json.loads(resources.joinpath("provenance.json").read_text())
        if hashlib.sha256(blob).hexdigest() != provenance["wasm_sha256"]:
            raise RuntimeError("bundled Wasm integrity mismatch")
        config = wasmtime.Config()
        config.consume_fuel = True
        engine = wasmtime.Engine(config)
        module = wasmtime.Module(engine, blob)
        if module.imports:
            raise RuntimeError("Wasm imports are forbidden")
        self._store = wasmtime.Store(engine)
        self._store.set_limits(memory_size=memory_bytes, instances=1, memories=1, tables=1, table_elements=4096)
        self._store.set_fuel(fuel)
        self._exports = wasmtime.Instance(self._store, module, []).exports(self._store)
        self._memory = self._exports["memory"]
        self._handle = self._call("rvf_store_create", dimensions, self._metric)

    def _call(self, name, *args):
        try:
            result = self._exports[name](self._store, *args)
        except (wasmtime.Trap, wasmtime.WasmtimeError) as exc:
            self._closed = True
            raise RuntimeError("Wasm execution failed; this store is closed") from exc
        if isinstance(result, int) and result < 0:
            raise RuntimeError(f"{name} failed with code {result}")
        return result

    def _begin(self):
        if self._closed:
            raise RuntimeError("store is closed")
        self._store.set_fuel(self._fuel)

    @contextmanager
    def _buffer(self, data):
        size = len(data)
        ptr = self._call("rvf_alloc", size)
        if not ptr:
            self._closed = True
            raise MemoryError("Wasm allocation failed")
        try:
            self._memory.write(self._store, data, ptr)
            yield ptr
        finally:
            if not self._closed:
                self._call("rvf_free", ptr, size)

    def _vector(self, vector):
        values = tuple(islice(iter(vector), self.dimensions + 1))
        if len(values) != self.dimensions:
            raise ValueError("vector dimension mismatch")
        try:
            values = tuple(float(x) for x in values)
            if not all(math.isfinite(x) and abs(x) <= 1e15 for x in values):
                raise ValueError("vectors must be finite with magnitude at most 1e15")
            if self._metric == 2:
                norm = math.hypot(*values)
                if norm == 0:
                    raise ValueError("cosine vectors must have nonzero norm")
                values = tuple(x / norm for x in values)
            return struct.pack(f"<{self.dimensions}f", *values)
        except (TypeError, OverflowError, struct.error) as exc:
            raise ValueError("invalid float32 vector") from exc

    def add(self, id: int, vector) -> None:
        """Insert a unique unsigned 64-bit ID and one vector."""
        if type(id) is not int or not 0 <= id < 2**64:
            raise ValueError("id must be an unsigned 64-bit integer")
        data = self._vector(vector)
        with self._lock:
            self._begin()
            if id in self._ids:
                raise ValueError("duplicate id")
            if len(self._ids) >= 100_000:
                raise ValueError("store limit is 100000 vectors")
            with self._buffer(data) as vectors, self._buffer(struct.pack("<Q", id)) as ids:
                if self._call("rvf_store_ingest", self._handle, vectors, ids, 1) != 1:
                    raise RuntimeError("ingest did not insert one vector")
            self._ids.add(id)

    def query(self, vector, k: int = 10) -> list[Match]:
        """Return nearest vectors using the shipped Rust distance implementation."""
        if type(k) is not int or not 1 <= k <= 10_000:
            raise ValueError("k must be between 1 and 10000")
        data = self._vector(vector)
        with self._lock:
            self._begin()
            k = min(k, len(self._ids))
            if not k:
                return []
            with self._buffer(data) as query, self._buffer(bytes(k * 12)) as out:
                count = self._call("rvf_store_query", self._handle, query, k, self._metric, out)
                if count > k:
                    self._closed = True
                    raise RuntimeError("invalid Wasm result count")
                raw = self._memory.read(self._store, out, out + count * 12)
                matches = [Match(*struct.unpack_from("<Qf", raw, i * 12)) for i in range(count)]
                if not all(math.isfinite(hit.distance) for hit in matches):
                    raise RuntimeError("Wasm returned a non-finite distance")
                return matches

    def close(self):
        with self._lock:
            if not self._closed:
                self._begin()
                self._call("rvf_store_close", self._handle)
                self._closed = True
            self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
