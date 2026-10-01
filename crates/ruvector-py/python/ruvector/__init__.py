"""Local typed bindings to RuVector's Rust core. Scores are distances: lower is better."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, TypeAlias, cast

from ._native import (
    UPSTREAM_COMMIT,
    UPSTREAM_VERSION,
    ClosedError,
    DimensionError,
    DuplicateIDError,
    IndexError,
    InvalidVectorError,
    NativeDB,
    RuVectorError,
    StorageError,
)

__version__ = "0.1.0"
__all__ = [
    "VectorDB",
    "VectorRecord",
    "SearchResult",
    "DistanceMetric",
    "HNSWConfig",
    "DatabaseOptions",
    "JSONValue",
    "Metadata",
    "RuVectorError",
    "InvalidVectorError",
    "DimensionError",
    "DuplicateIDError",
    "StorageError",
    "IndexError",
    "ClosedError",
    "UPSTREAM_COMMIT",
    "UPSTREAM_VERSION",
]

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Metadata: TypeAlias = Mapping[str, JSONValue]


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


_NATIVE_METRICS = {
    DistanceMetric.EUCLIDEAN: "Euclidean",
    DistanceMetric.COSINE: "Cosine",
    DistanceMetric.MANHATTAN: "Manhattan",
}


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2**63 - 1:
        raise InvalidVectorError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class HNSWConfig:
    """Native HNSW construction/search parameters; capacity defaults to 100,000."""

    m: int = 16
    ef_construction: int = 100
    ef_search: int = 100
    max_elements: int = 100_000

    def __post_init__(self) -> None:
        for name in ("m", "ef_construction", "ef_search", "max_elements"):
            _positive(getattr(self, name), name)
        if self.m < 2 or self.m > 256:
            raise InvalidVectorError("m must be between 2 and 256")

    def _asdict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in ("m", "ef_construction", "ef_search", "max_elements")
        }


@dataclass(frozen=True)
class DatabaseOptions:
    dimensions: int
    distance_metric: DistanceMetric
    storage_path: str
    hnsw: HNSWConfig


@dataclass(frozen=True)
class VectorRecord:
    id: str | None
    vector: Sequence[float]
    metadata: Metadata | None = None


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    vector: list[float] | None
    metadata: dict[str, JSONValue] | None


def _id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidVectorError("id must be a nonempty string")
    return value


def _metadata(value: Metadata | None) -> dict[str, JSONValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvalidVectorError("metadata/filter must be a mapping")

    def check(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise InvalidVectorError("metadata keys must be strings")
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is None or isinstance(item, (str, bool, int)):
            pass
        elif isinstance(item, float) and math.isfinite(item):
            pass
        else:
            raise InvalidVectorError("metadata must contain JSON values with finite numbers")

    result = dict(value)
    check(result)
    return result


class VectorDB:
    """Persistent or in-memory native RuVector database.

    One live handle per persistent path is recommended. Existing files restore
    their saved configuration, which is exposed through options/dimensions.
    Operations on a handle are serialized in Rust and release the Python GIL.
    """

    def __init__(
        self,
        dimensions: int = 384,
        *,
        path: str | os.PathLike[str] | None = None,
        distance_metric: DistanceMetric | str = DistanceMetric.COSINE,
        hnsw: HNSWConfig | None = None,
    ) -> None:
        _positive(dimensions, "dimensions")
        try:
            metric = DistanceMetric(distance_metric)
        except ValueError as exc:
            raise InvalidVectorError(f"unknown distance metric: {distance_metric!r}") from exc
        config = hnsw if hnsw is not None else HNSWConfig()
        storage = "memory://python" if path is None else os.fspath(path)
        if not isinstance(storage, str) or not storage:
            raise InvalidVectorError("path must be a nonempty string or path-like object")
        if path is not None and storage.startswith("memory://"):
            raise InvalidVectorError("use path=None for an in-memory database")
        self._native = NativeDB(
            json.dumps(
                {
                    "dimensions": dimensions,
                    "distance_metric": _NATIVE_METRICS[metric],
                    "storage_path": storage,
                    "hnsw_config": config._asdict(),
                    "quantization": None,
                }
            )
        )
        actual = self._call("options", None)
        actual_metric = next(
            k for k, v in _NATIVE_METRICS.items() if v == actual["distance_metric"]
        )
        self._options = DatabaseOptions(
            actual["dimensions"],
            actual_metric,
            actual["storage_path"],
            HNSWConfig(**actual["hnsw_config"]),
        )

    @property
    def options(self) -> DatabaseOptions:
        return self._options

    @property
    def dimensions(self) -> int:
        return self._options.dimensions

    def _call(self, operation: str, args: object) -> Any:
        return json.loads(
            self._native.execute(operation, json.dumps(args, allow_nan=False, ensure_ascii=True))
        )

    def _vector(self, values: Sequence[float]) -> list[float]:
        try:
            vector: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise InvalidVectorError("vector values must be real numbers")
                number = float(value)
                if not math.isfinite(number) or abs(number) > 3.4028234663852886e38:
                    raise InvalidVectorError("vector values must be finite and fit float32")
                vector.append(number)
        except (TypeError, OverflowError) as exc:
            raise InvalidVectorError("vector must be a sequence of finite real numbers") from exc
        if len(vector) != self.dimensions:
            raise DimensionError(f"expected {self.dimensions} dimensions, got {len(vector)}")
        return vector

    def _entry(self, entry: VectorRecord) -> dict[str, object]:
        if not isinstance(entry, VectorRecord):
            raise InvalidVectorError("batch entries must be VectorRecord instances")
        return {
            "id": None if entry.id is None else _id(entry.id),
            "vector": self._vector(entry.vector),
            "metadata": _metadata(entry.metadata),
        }

    def insert(
        self, vector: Sequence[float], *, id: str | None = None, metadata: Metadata | None = None
    ) -> str:
        """Insert a new ID (or generate UUID). Existing IDs raise DuplicateIDError."""
        return cast(str, self._call("insert", self._entry(VectorRecord(id, vector, metadata))))

    def insert_batch(self, entries: Iterable[VectorRecord]) -> list[str]:
        """Validate the whole input, then use native insert_batch. Not transactional."""
        return cast(list[str], self._call("insert_batch", [self._entry(e) for e in entries]))

    def _query(self, vector: Sequence[float], k: int, filter: Metadata | None) -> dict[str, object]:
        return {
            "vector": self._vector(vector),
            "k": _positive(k, "k"),
            "filter": _metadata(filter),
            "ef_search": None,
        }

    def search(
        self, vector: Sequence[float], *, k: int = 10, filter: Metadata | None = None
    ) -> list[SearchResult]:
        """Approximate top-k, then exact equality AND filter; may return fewer than k."""
        return [SearchResult(**r) for r in self._call("search", self._query(vector, k, filter))]

    def search_batch(
        self, vectors: Iterable[Sequence[float]], *, k: int = 10, filter: Metadata | None = None
    ) -> list[list[SearchResult]]:
        """Run native searches under one lock/GIL release; preserves input order."""
        queries = [self._query(v, k, filter) for v in vectors]
        return [[SearchResult(**r) for r in rows] for rows in self._call("search_batch", queries)]

    def get(self, id: str) -> VectorRecord | None:
        data = self._call("get", _id(id))
        return VectorRecord(**data) if data is not None else None

    def delete(self, id: str) -> bool:
        """Delete by ID; returns False for a missing ID."""
        return self.delete_batch([id])[0]

    def delete_batch(self, ids: Iterable[str]) -> list[bool]:
        """Sequential native deletions, returning one boolean per ID; not transactional."""
        return cast(list[bool], self._call("delete_batch", [_id(id) for id in ids]))

    def keys(self) -> list[str]:
        return cast(list[str], self._call("keys", None))

    def __len__(self) -> int:
        return cast(int, self._call("len", None))

    def __getitem__(self, id: str) -> VectorRecord:
        record = self.get(id)
        if record is None:
            raise KeyError(id)
        return record

    def close(self) -> None:
        """Release the native index and file lock. Repeated close is harmless."""
        self._call("close", None)

    def __enter__(self) -> VectorDB:
        self._call("len", None)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
