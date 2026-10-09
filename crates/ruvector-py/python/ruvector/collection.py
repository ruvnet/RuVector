"""``Collection`` — the generic vector-DB surface added in ADR-352.

Thin pure-Python layer over PyO3-bound native index classes (per CLAUDE.md:
core logic stays in Rust; Python is a typed binding/UX layer). Two backends:

- ``backend="hnsw"`` (**default**, ADR-352 M2 — "make the default Collection
  the fast path"): :class:`ruvector._native.HnswIndex`
  (``ruvector_core::vector_db::VectorDB``). Metadata lives natively in Rust;
  dict-based ``search(..., filter={...})`` is evaluated **in Rust**, not
  Python; ``delete`` is a real delete, not a tombstone.
- ``backend="rabitq"`` (the original M1.5 backend):
  :class:`ruvector._native.RabitqIndex` (RaBitQ+ 1-bit quantization).
  Smaller memory footprint at the cost of recall/latency tradeoffs (see
  ADR-352's benchmark section) and no native filtering — this module still
  does the filtering client-side for this backend (over-fetch + retry,
  documented in :meth:`Collection.search`), and delete is a tombstone
  (:meth:`Collection.vacuum` reclaims space).

One inherent limitation applies to **both** backends: a *callable* filter
predicate (as opposed to an exact-match dict) can never be pushed into
Rust — there's no way to ship an arbitrary Python function across the
PyO3 boundary — so that path always falls back to over-fetch-and-filter-
in-Python. This is a structural fact, not something a future milestone
fixes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ._native import HnswIndex, RabitqIndex, RuVectorError

_META_SUFFIX = ".meta.json"
_DEFAULT_OVERFETCH = 4
_BACKENDS = ("hnsw", "rabitq")
# RabitqIndex stores ids as u32 (ruvector_rabitq::index's `ids: Vec<u32>`,
# `self.ids.push(id as u32)` with no bounds check at that layer). Only the
# rabitq backend is affected — the hnsw backend stores ids as strings
# (`str(int_id)`), with no such ceiling.
_MAX_ID = 2**32 - 1

# Resource bounds. Without them one hostile (or buggy) call such as
# `search(q, k=10**12)` makes the Rust side attempt an 8 TB allocation and
# abort the whole process (reproduced on the hnsw backend), which no Python
# `try/except` can catch. These are far above any realistic use for an
# in-memory ANN index; they exist to turn "process abort" into a clean error.
MAX_K = 10_000
MAX_DIM = 8192
MAX_RERANK_FACTOR = 10_000
MAX_OVERFETCH = 64

Index = Union[HnswIndex, RabitqIndex]


class CollectionError(RuVectorError):
    """Raised for Collection-level misuse (distinct from index-level errors)."""


def _check_ids_fit_u32(ids: Iterable[int]) -> None:
    for i in ids:
        if i < 0 or i > _MAX_ID:
            raise CollectionError(f"id {i} out of range — the rabitq backend stores ids as u32 (0..{_MAX_ID})")


@dataclass
class SearchHit:
    """One search result: id, distance score, and metadata (if any)."""

    id: int
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def __iter__(self) -> "Iterator[Union[int, float]]":
        # Keeps `for id, score in coll.search(...)` working for callers who
        # only want the M1 two-tuple shape.
        yield self.id
        yield self.score


@dataclass
class CollectionStats:
    count: int
    dim: int
    backend: str
    rerank_factor: int
    memory_bytes: int
    tombstoned: int


def _check_finite(arr: NDArray[np.float32], name: str) -> None:
    """NaN/inf poison an index silently (hnsw scores them 0.0, rabitq NaN),
    so they are rejected at every entry point. Checked on the float32 array,
    so a float64 value that overflows float32 (1e39) is caught too."""
    if not np.isfinite(arr).all():
        raise CollectionError(f"{name} must contain only finite values (no NaN or inf)")


def _validate_vector(v: NDArray[np.float32], dim: int, name: str) -> NDArray[np.float32]:
    with np.errstate(over="ignore"):  # float64 -> float32 overflow becomes inf; caught below
        arr = np.ascontiguousarray(v, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise CollectionError(f"{name} must be a 1D float32 array of length {dim}, got shape {arr.shape}")
    _check_finite(arr, name)
    return arr


def _check_int_range(value: object, name: str, lo: int, hi: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise CollectionError(f"{name} must be an integer in [{lo}, {hi}], got {value!r}")


def _check_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise CollectionError(f"unknown backend {backend!r}; expected one of {_BACKENDS}")


class Collection:
    """A named set of vectors with metadata, filtering, and delete.

    Construct via :meth:`create` (empty) or :meth:`from_vectors` (bulk load).
    Do not call ``Collection(...)`` directly.
    """

    # Declared at class level (PEP 526) rather than only assigned in
    # __init__: every constructor path (create/from_vectors/load) sets
    # these right after calling __init__, and mypy --strict needs the
    # declaration to exist somewhere unconditional to recognize the
    # attribute at all on later `coll._dim` reads.
    _dim: int
    _backend: str
    _rerank_factor: int
    _seed: int
    _metric: str
    _hnsw_m: int
    _hnsw_ef_construction: int
    _hnsw_ef_search: int

    def __init__(
        self,
        *,
        _index: Optional[Index],
        _metadata: Dict[int, Dict[str, Any]],
        _tombstones: "set[int]",
        _next_id: int,
    ) -> None:
        self._index = _index
        self._metadata = _metadata
        self._tombstones = _tombstones
        self._next_id = _next_id

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        dim: int,
        *,
        backend: str = "hnsw",
        rerank_factor: int = 20,
        seed: int = 42,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> "Collection":
        """Create an empty collection of the given dimensionality.

        ``backend="hnsw"`` (default) builds the index immediately — unlike
        RaBitQ, HNSW needs no rotation fit, so there's no lazy-build step.
        ``backend="rabitq"`` still lazily builds on the first :meth:`insert`
        (RaBitQ+ needs >=1 vector to fit a rotation); ``rerank_factor``/
        ``seed`` are remembered for that first build.
        """
        _check_int_range(dim, "dim", 1, MAX_DIM)
        _check_int_range(rerank_factor, "rerank_factor", 1, MAX_RERANK_FACTOR)
        _check_backend(backend)
        index: Optional[Index]
        if backend == "hnsw":
            index = HnswIndex.create(dim=dim, metric=metric, m=m, ef_construction=ef_construction, ef_search=ef_search)
        else:
            index = None
        coll = cls(_index=index, _metadata={}, _tombstones=set(), _next_id=0)
        coll._dim = dim
        coll._backend = backend
        coll._rerank_factor = rerank_factor
        coll._seed = seed
        coll._metric = metric
        coll._hnsw_m = m
        coll._hnsw_ef_construction = ef_construction
        coll._hnsw_ef_search = ef_search
        return coll

    @classmethod
    def from_vectors(
        cls,
        vectors: NDArray[np.float32],
        *,
        ids: Optional[Sequence[int]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        backend: str = "hnsw",
        rerank_factor: int = 20,
        seed: int = 42,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> "Collection":
        """Bulk-build a collection from an ``(n, dim)`` array in one call.

        ``backend="hnsw"`` (default) inserts every row via
        ``HnswIndex.insert_batch`` (GIL released around the loop).
        ``backend="rabitq"`` uses ``RabitqIndex.build`` (parallel
        rotate+pack over rayon) instead of a loop.
        """
        _check_backend(backend)
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise CollectionError(f"vectors must be 2D, got {arr.ndim}D")
        _check_int_range(arr.shape[1], "dim", 1, MAX_DIM)
        _check_int_range(rerank_factor, "rerank_factor", 1, MAX_RERANK_FACTOR)
        _check_finite(arr, "vectors")
        n = arr.shape[0]
        id_list: List[int] = list(range(n)) if ids is None else list(ids)
        if len(id_list) != n:
            raise CollectionError(f"ids length ({len(id_list)}) must match vectors row count ({n})")
        if len(set(id_list)) != len(id_list):
            raise CollectionError("ids must be unique")
        if metadatas is not None and len(metadatas) != n:
            raise CollectionError(f"metadatas length ({len(metadatas)}) must match vectors row count ({n})")

        index: Index
        if backend == "hnsw":
            index = HnswIndex.create(dim=arr.shape[1], metric=metric, m=m, ef_construction=ef_construction, ef_search=ef_search)
            index.insert_batch([str(i) for i in id_list], arr, metadatas=list(metadatas) if metadatas is not None else None)
        else:
            if ids is not None:
                _check_ids_fit_u32(id_list)
            # `RabitqIndex.build`'s optional `ids` kwarg stores the caller's
            # own ids directly (added in review — an earlier cut of this
            # method accepted `ids` and silently remapped search results
            # back to row indices instead; see ADR-352/LOOP-STATE).
            ids_arr = None if ids is None else np.asarray(id_list, dtype=np.uint64)
            index = RabitqIndex.build(arr, ids=ids_arr, rerank_factor=rerank_factor, seed=seed)

        metadata: Dict[int, Dict[str, Any]] = {}
        if metadatas is not None:
            for row_id, md in zip(id_list, metadatas):
                if md:
                    metadata[row_id] = dict(md)

        coll = cls(_index=index, _metadata=metadata, _tombstones=set(), _next_id=max(id_list) + 1)
        coll._dim = arr.shape[1]
        coll._backend = backend
        coll._rerank_factor = rerank_factor
        coll._seed = seed
        coll._metric = metric
        coll._hnsw_m = m
        coll._hnsw_ef_construction = ef_construction
        coll._hnsw_ef_search = ef_search
        return coll

    # ── mutation ─────────────────────────────────────────────────────────

    def insert(self, vector: NDArray[np.float32], *, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert one vector, returning its assigned internal id."""
        new_id = self._next_id
        vec = _validate_vector(vector, self._dim, "vector")
        if self._backend == "hnsw":
            assert isinstance(self._index, HnswIndex)  # always constructed in create()/from_vectors()
            self._index.insert(str(new_id), vec, metadata=metadata)
        else:
            if self._index is None:
                self._index = RabitqIndex.build(vec.reshape(1, -1), rerank_factor=self._rerank_factor, seed=self._seed)
            else:
                assert isinstance(self._index, RabitqIndex)
                self._index.add(new_id, vec)
        if metadata:
            self._metadata[new_id] = dict(metadata)
        self._next_id += 1
        return new_id

    def insert_batch(
        self,
        vectors: NDArray[np.float32],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[int]:
        """Insert many vectors at once (releases the GIL in Rust)."""
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise CollectionError(f"vectors must be (n, {self._dim}), got shape {arr.shape}")
        _check_finite(arr, "vectors")
        n = arr.shape[0]
        if metadatas is not None and len(metadatas) != n:
            raise CollectionError(f"metadatas length ({len(metadatas)}) must match row count ({n})")
        new_ids = list(range(self._next_id, self._next_id + n))
        if self._backend == "hnsw":
            assert isinstance(self._index, HnswIndex)
            self._index.insert_batch([str(i) for i in new_ids], arr, metadatas=list(metadatas) if metadatas is not None else None)
        else:
            if self._index is None:
                self._index = RabitqIndex.build(arr, rerank_factor=self._rerank_factor, seed=self._seed)
                # build() assigns row ids 0..n-1; since _next_id was 0,
                # these coincide with new_ids by construction.
            else:
                assert isinstance(self._index, RabitqIndex)
                self._index.add_batch(np.asarray(new_ids, dtype=np.uint64), arr)
        if metadatas:
            for i, md in enumerate(metadatas):
                if md:
                    self._metadata[new_ids[i]] = dict(md)
        self._next_id += n
        return new_ids

    def delete(self, id: int) -> None:
        """Delete ``id``.

        ``backend="hnsw"``: a real delete (``HnswIndex.delete``) — gone
        from count/search/get_metadata immediately. ``backend="rabitq"``:
        a soft delete (tombstone); call :meth:`vacuum` to physically
        reclaim the space once the tombstone fraction gets large.
        """
        if self._backend == "hnsw":
            assert isinstance(self._index, HnswIndex)
            self._index.delete(str(id))
        else:
            self._tombstones.add(id)
        self._metadata.pop(id, None)

    def vacuum(self) -> int:
        """Physically rebuild the index, reclaiming tombstoned rows.

        ``backend="hnsw"``: always returns 0. ``HnswIndex.delete`` already
        performed a real delete (not a Python tombstone), so there is
        nothing queued to drop at this layer. The only unreclaimed cost is
        internal `hnsw_rs` graph memory from deleted nodes (no live-delete
        in the vendored library) — today's API has no way to measure that,
        so this returns 0 rather than a fabricated estimate.

        ``backend="rabitq"``: rebuilds via ``export_items`` + a fresh
        ``build`` call (same cost as the initial bulk build), returning
        the number of rows actually dropped.
        """
        if self._backend == "hnsw":
            return 0
        if self._index is None or not self._tombstones:
            return 0
        assert isinstance(self._index, RabitqIndex)
        items = self._index.export_items()
        kept = [(i, v) for i, v in items if i not in self._tombstones]
        dropped = len(items) - len(kept)
        if not kept:
            self._index = None
            self._metadata = {}
            self._tombstones = set()
            self._next_id = 0
            return dropped
        vecs = np.stack([v for _, v in kept]).astype(np.float32)
        # Rebuild with row-index ids 0..len(kept)-1 (build()'s only mode),
        # then remap metadata from old id -> new row index.
        self._index = RabitqIndex.build(vecs, rerank_factor=self._rerank_factor, seed=self._seed)
        old_ids = [i for i, _ in kept]
        new_metadata: Dict[int, Dict[str, Any]] = {}
        for new_row, old_id in enumerate(old_ids):
            if old_id in self._metadata:
                new_metadata[new_row] = self._metadata[old_id]
        self._metadata = new_metadata
        self._tombstones = set()
        self._next_id = len(kept)
        return dropped

    # ── query ────────────────────────────────────────────────────────────

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        *,
        filter: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]] = None,
        rerank_factor: Optional[int] = None,
        overfetch: int = _DEFAULT_OVERFETCH,
    ) -> List[SearchHit]:
        """Search for ``k`` nearest neighbours.

        ``filter`` may be an exact-match dict (``{"category": "news"}``) or
        a predicate callable. **An exact-match dict is evaluated in Rust**
        when ``backend="hnsw"`` (the literal fix for "filtering in Rust,
        not Python" — see ADR-352) — but Rust's ``VectorDB::search`` fetches
        only ``k`` ANN candidates *before* filtering (no over-fetch for
        selectivity at that layer — a selective filter can legitimately
        return fewer than ``k``), so this method still runs the
        over-fetch-and-retry *loop* in Python, widening how many
        candidates it asks Rust for each round; only the per-row equality
        test itself happens in Rust, not the loop orchestration. A
        predicate callable always falls back to fetching unfiltered
        candidates and testing them in Python, on both backends, because
        an arbitrary Python function cannot be shipped across the PyO3
        boundary — a structural limitation, not a missing feature.
        ``backend="rabitq"`` filters every candidate client-side either
        way (no native filter pushdown in that backend, documented since
        M1.5).
        """
        _check_int_range(k, "k", 1, MAX_K)
        _check_int_range(overfetch, "overfetch", 1, MAX_OVERFETCH)
        if rerank_factor is not None:
            _check_int_range(rerank_factor, "rerank_factor", 1, MAX_RERANK_FACTOR)
        if self._index is None or len(self._index) == 0:
            return []
        qvec = _validate_vector(query, self._dim, "query")

        if self._backend == "hnsw":
            assert isinstance(self._index, HnswIndex)
            return self._search_hnsw(qvec, k, filter, overfetch)

        assert isinstance(self._index, RabitqIndex)
        return self._search_rabitq(qvec, k, filter, rerank_factor, overfetch)

    def _search_hnsw(
        self,
        qvec: NDArray[np.float32],
        k: int,
        filter: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]],
        overfetch: int,
    ) -> List[SearchHit]:
        assert isinstance(self._index, HnswIndex)
        if filter is None:
            # Fast path: nothing to widen for. The overfetch loop below
            # exists only because VectorDB's own filter has no overfetch
            # for selectivity (a selective filter can return fewer than k
            # from one call) - with no filter at all, asking Rust for
            # `k * overfetch` candidates on every unfiltered search would
            # be pure waste (this was a real perf bug, caught while
            # investigating a benchmark number that looked too slow - see
            # ADR-352's benchmark section).
            raw = self._index.search(qvec, k, filter=None)
            return [SearchHit(id=int(i), score=s, metadata=m) for i, s, m in raw]

        rust_filter: Optional[Dict[str, Any]]
        py_pred: Optional[Callable[[Dict[str, Any]], bool]]
        if isinstance(filter, dict):
            rust_filter, py_pred = dict(filter), None
        else:
            rust_filter, py_pred = None, filter

        width = k
        seen_widths: "set[int]" = set()
        hits: List[SearchHit] = []
        while True:
            effective_k = min(width * overfetch, len(self._index))
            raw = self._index.search(qvec, max(effective_k, k), filter=rust_filter)
            hits = []
            for i, s, m in raw:
                if py_pred is not None and not py_pred(m or {}):
                    continue
                hits.append(SearchHit(id=int(i), score=s, metadata=m))
                if len(hits) >= k:
                    break
            if len(hits) >= k or effective_k >= len(self._index) or width in seen_widths:
                break
            seen_widths.add(width)
            width *= 2
        return hits[:k]

    def _search_rabitq(
        self,
        qvec: NDArray[np.float32],
        k: int,
        filter: Optional[Union[Dict[str, Any], Callable[[Dict[str, Any]], bool]]],
        rerank_factor: Optional[int],
        overfetch: int,
    ) -> List[SearchHit]:
        assert isinstance(self._index, RabitqIndex)
        pred: Optional[Callable[[Dict[str, Any]], bool]]
        if filter is None:
            pred = None
        elif isinstance(filter, dict):
            # `isinstance` (not `callable(filter)`) is the branch mypy can
            # actually narrow a Dict|Callable union on; `callable()` doesn't
            # exclude Callable from the union in the general case.
            filt_dict: Dict[str, Any] = dict(filter)

            def pred(meta: Dict[str, Any]) -> bool:  # noqa: F811
                return all(meta.get(key) == val for key, val in filt_dict.items())

        else:
            pred = filter

        tombstones = self._tombstones
        width = k
        hits: List[SearchHit] = []
        seen_widths: "set[int]" = set()
        while True:
            effective_k = min(width * overfetch, len(self._index))
            raw = self._index.search(qvec, max(effective_k, k), rerank_factor=rerank_factor)
            hits = []
            for id_, score in raw:
                if id_ in tombstones:
                    continue
                meta = self._metadata.get(id_)
                if pred is not None and not pred(meta or {}):
                    continue
                hits.append(SearchHit(id=id_, score=score, metadata=meta))
                if len(hits) >= k:
                    break
            if len(hits) >= k or effective_k >= len(self._index) or width in seen_widths:
                break
            seen_widths.add(width)
            width *= 2
        return hits[:k]

    def get_metadata(self, id: int) -> Optional[Dict[str, Any]]:
        return self._metadata.get(id)

    def export_live_items(
        self,
    ) -> List[Tuple[int, NDArray[np.float32], Optional[Dict[str, Any]]]]:
        """Return ``(id, vector, metadata)`` for every live row, sorted by
        id. Used by the CLI's ``export`` command, by :meth:`save`, and by
        anything else that needs a read-only snapshot without reaching
        into backend internals directly.
        """
        if self._index is None:
            return []
        if self._backend == "hnsw":
            assert isinstance(self._index, HnswIndex)
            items = sorted(self._index.export_items(), key=lambda kv: int(kv[0]))
            return [(int(i), v, m) for i, v, m in items]
        assert isinstance(self._index, RabitqIndex)
        rabitq_items = sorted(self._index.export_items(), key=lambda kv: kv[0])
        return [(i, v, self._metadata.get(i)) for i, v in rabitq_items if i not in self._tombstones]

    # ── introspection ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._index is None:
            return 0
        if self._backend == "hnsw":
            return len(self._index)
        return len(self._index) - len(self._tombstones)

    def stats(self) -> CollectionStats:
        memory_bytes = 0
        if self._backend == "rabitq" and isinstance(self._index, RabitqIndex):
            memory_bytes = self._index.memory_bytes
        return CollectionStats(
            count=len(self),
            dim=self._dim,
            backend=self._backend,
            rerank_factor=self._rerank_factor if self._backend == "rabitq" else 0,
            memory_bytes=memory_bytes,
            tombstoned=len(self._tombstones),
        )

    @property
    def metric(self) -> str:
        """The distance function ``search()`` scores are computed with.

        For ``backend="hnsw"``: whatever ``metric=`` was passed at
        construction (``"cosine"`` by default — see ``create()``).
        For ``backend="rabitq"``: always ``"squared_l2"`` — the
        ``metric=`` constructor kwarg is accepted but **not actually
        applied** to this backend (``RabitqPlusIndex`` always scores via
        squared L2 internally; `self._metric` would otherwise report
        whatever default was passed, which could silently lie about what
        distance is actually in use). Added for
        ``ruvector.integrations.llamaindex``'s distance->similarity
        conversion, which needs to know this to avoid returning a raw
        distance in a field callers expect to be a similarity.
        """
        if self._backend == "rabitq":
            return "squared_l2"
        return self._metric

    def __repr__(self) -> str:
        return f"Collection(backend={self._backend!r}, n={len(self)}, dim={self._dim}, tombstoned={len(self._tombstones)})"

    # ── persistence ──────────────────────────────────────────────────────

    @staticmethod
    def meta_path(path: Union[str, os.PathLike[str]]) -> Path:
        """Path of the JSON sidecar for a given index path.

        **Existence of a collection must be checked against this path, not
        the index path** — an empty ``backend="rabitq"`` :meth:`create`d
        collection has no ``.rbpx`` file yet (RaBitQ+ needs >=1 vector to
        build a rotation) but always has a sidecar once :meth:`save` has
        run once. Both the CLI and the MCP server use this (fixed after
        the MCP server's duplicate-collection check used the wrong path
        and silently let a second ``vector_create_collection`` call
        through — see ``tests/test_mcp_server.py::test_create_twice_errors``).
        """
        path = Path(path)
        return path.with_suffix(path.suffix + _META_SUFFIX)

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save to ``path`` plus a ``<path>.meta.json`` sidecar.

        ``backend="rabitq"``: ``path`` is the native ``.rbpx`` index file
        (written by ``RabitqIndex.save``); the sidecar carries metadata,
        tombstones, and build config.

        ``backend="hnsw"``: ``HnswIndex`` has no native on-disk format (it's
        always in-memory — see ``src/hnsw.rs``'s module docstring), so
        ``path`` is written as a raw ``np.save`` of the stacked vectors
        (ids/metadata/config live in the sidecar instead, same two-file
        split as the rabitq backend for a uniform persistence story).

        Each file is written to a temp name and moved into place with
        :func:`os.replace`, so a crash or error mid-write never leaves a
        truncated file at the real path. The two files are still replaced
        one after the other, so a crash *between* the two renames can leave
        a new index with the previous sidecar; :meth:`load` validates the
        pair and raises :class:`CollectionError` rather than serving it.
        """
        path = Path(path)
        tmp_index = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        meta = self.meta_path(path)
        tmp_meta = meta.with_name(f"{meta.name}.tmp-{os.getpid()}")
        index_written = False
        try:
            if self._backend == "hnsw":
                items = self.export_live_items()
                vecs = (
                    np.stack([v for _, v, _ in items]).astype(np.float32)
                    if items
                    else np.zeros((0, self._dim), dtype=np.float32)
                )
                with open(tmp_index, "wb") as f:
                    np.save(f, vecs)
                index_written = True
                sidecar = {
                    "backend": "hnsw",
                    "dim": self._dim,
                    "metric": self._metric,
                    "hnsw_m": self._hnsw_m,
                    "hnsw_ef_construction": self._hnsw_ef_construction,
                    "hnsw_ef_search": self._hnsw_ef_search,
                    "next_id": self._next_id,
                    "ids": [i for i, _, _ in items],
                    "metadata": {str(i): m for i, _, m in items if m},
                    "empty": self._index is None,
                }
            else:
                if self._index is not None:
                    assert isinstance(self._index, RabitqIndex)
                    self._index.save(str(tmp_index))
                    index_written = True
                sidecar = {
                    "backend": "rabitq",
                    "dim": self._dim,
                    "rerank_factor": self._rerank_factor,
                    "seed": self._seed,
                    "next_id": self._next_id,
                    "tombstones": sorted(self._tombstones),
                    "metadata": {str(k): v for k, v in self._metadata.items()},
                    "empty": self._index is None,
                }
            tmp_meta.write_text(json.dumps(sidecar))
            if index_written:
                os.replace(tmp_index, path)
            os.replace(tmp_meta, meta)
        finally:
            for tmp in (tmp_index, tmp_meta):
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    @classmethod
    def load(cls, path: Union[str, os.PathLike[str]]) -> "Collection":
        """Load a collection saved by :meth:`save`.

        The sidecar and index are validated (types, ranges, ids/rows
        consistency, finite values); anything malformed raises
        :class:`CollectionError` rather than a bare ``KeyError`` /
        ``JSONDecodeError`` / Rust panic.
        """
        path = Path(path)
        meta_path = cls.meta_path(path)
        if not meta_path.exists():
            raise CollectionError(f"missing sidecar metadata file: {meta_path}")
        try:
            sidecar = json.loads(meta_path.read_text())
            if not isinstance(sidecar, dict):
                raise CollectionError(f"{meta_path}: sidecar must be a JSON object")
            return cls._load_validated(path, sidecar)
        except CollectionError:
            raise
        except RuVectorError as exc:
            raise CollectionError(f"corrupt collection at {path}: {exc}") from exc
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
            raise CollectionError(f"corrupt or unreadable collection at {path}: {type(exc).__name__}: {exc}") from exc

    @classmethod
    def _load_validated(cls, path: Path, sidecar: Dict[str, Any]) -> "Collection":
        # Sidecars written before ADR-352's M2 slice have no "backend" key
        # at all — they are always rabitq (the only backend that existed).
        backend = sidecar.get("backend", "rabitq")
        _check_backend(backend)
        dim = sidecar["dim"]
        _check_int_range(dim, "dim", 1, MAX_DIM)
        next_id = sidecar["next_id"]
        _check_int_range(next_id, "next_id", 0, 2**63 - 1)
        metadata_raw = sidecar["metadata"]
        if not isinstance(metadata_raw, dict) or not all(isinstance(m, dict) for m in metadata_raw.values()):
            raise CollectionError("sidecar 'metadata' must map ids to objects")
        metadata = {int(k): v for k, v in metadata_raw.items()}

        if backend == "hnsw":
            for key in ("hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"):
                _check_int_range(sidecar[key], key, 1, 100_000)
            if not isinstance(sidecar["metric"], str):
                raise CollectionError("sidecar 'metric' must be a string")
            index: Optional[Index] = None
            if not sidecar["empty"]:
                ids = sidecar["ids"]
                if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
                    raise CollectionError("sidecar 'ids' must be a list of integers")
                if len(set(ids)) != len(ids):
                    raise CollectionError("sidecar 'ids' contains duplicates")
                with open(path, "rb") as f:
                    vecs = np.load(f, allow_pickle=False)
                if vecs.ndim != 2 or vecs.shape != (len(ids), dim) or vecs.dtype != np.float32:
                    raise CollectionError(
                        f"index file shape/dtype {vecs.shape}/{vecs.dtype} does not match sidecar "
                        f"({len(ids)} ids, dim {dim}, float32) - index and sidecar are out of sync"
                    )
                _check_finite(vecs, "stored vectors")
                index = HnswIndex.create(
                    dim=dim,
                    metric=sidecar["metric"],
                    m=sidecar["hnsw_m"],
                    ef_construction=sidecar["hnsw_ef_construction"],
                    ef_search=sidecar["hnsw_ef_search"],
                )
                index.insert_batch([str(i) for i in ids], vecs, metadatas=[metadata_raw.get(str(i)) for i in ids])
            coll = cls(_index=index, _metadata=metadata, _tombstones=set(), _next_id=next_id)
            coll._dim = dim
            coll._backend = "hnsw"
            coll._metric = sidecar["metric"]
            coll._hnsw_m = sidecar["hnsw_m"]
            coll._hnsw_ef_construction = sidecar["hnsw_ef_construction"]
            coll._hnsw_ef_search = sidecar["hnsw_ef_search"]
            coll._rerank_factor = 0
            coll._seed = 0
            return coll

        _check_int_range(sidecar["rerank_factor"], "rerank_factor", 1, MAX_RERANK_FACTOR)
        tombstones = sidecar["tombstones"]
        if not isinstance(tombstones, list) or not all(isinstance(t, int) for t in tombstones):
            raise CollectionError("sidecar 'tombstones' must be a list of integers")
        rabitq_index: Optional[Index] = None if sidecar["empty"] else RabitqIndex.load(str(path))
        coll = cls(_index=rabitq_index, _metadata=metadata, _tombstones=set(tombstones), _next_id=next_id)
        coll._dim = dim
        coll._backend = "rabitq"
        coll._rerank_factor = sidecar["rerank_factor"]
        coll._seed = sidecar["seed"]
        coll._metric = "cosine"
        coll._hnsw_m = 0
        coll._hnsw_ef_construction = 0
        coll._hnsw_ef_search = 0
        return coll


__all__ = ["Collection", "CollectionError", "SearchHit", "CollectionStats"]
