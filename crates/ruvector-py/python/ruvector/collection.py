"""``Collection`` — the generic vector-DB surface added in ADR-352 (M1.5).

Thin pure-Python layer over the PyO3-bound :class:`ruvector._native.RabitqIndex`
(per CLAUDE.md: core logic stays in Rust; Python is a typed binding/UX layer).
What lives here is exactly the part that does *not* belong in the Rust core:

- A ``{external_id: metadata_dict}`` sidecar, persisted as JSON next to the
  ``.rbpx`` index file.
- Client-side metadata filtering (``search(..., filter={...})``): the
  underlying index has no filter pushdown, so this over-fetches
  ``k * overfetch_factor`` candidates and filters in Python. Documented
  honestly below rather than claimed as a server-side capability.
- Soft delete via a tombstone set, because ``ruvector_rabitq::RabitqPlusIndex``
  has ``add`` but no ``delete`` (see ADR-352 "Security" / this module's
  :meth:`Collection.vacuum`).

This is explicitly the M1.5 stopgap from ADR-352 — M2 replaces the backing
store with a generic HNSW-capable index and RVF persistence without changing
this class's public surface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._native import RabitqIndex, RuVectorError

_META_SUFFIX = ".meta.json"
_DEFAULT_OVERFETCH = 4


class CollectionError(RuVectorError):
    """Raised for Collection-level misuse (distinct from index-level errors)."""


@dataclass
class SearchHit:
    """One search result: id, distance score, and metadata (if any)."""

    id: int
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def __iter__(self):
        # Keeps `for id, score in coll.search(...)` working for callers who
        # only want the M1 two-tuple shape.
        yield self.id
        yield self.score


@dataclass
class CollectionStats:
    count: int
    dim: int
    rerank_factor: int
    memory_bytes: int
    tombstoned: int


def _validate_vector(v: NDArray[np.float32], dim: int, name: str) -> NDArray[np.float32]:
    arr = np.ascontiguousarray(v, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise CollectionError(f"{name} must be a 1D float32 array of length {dim}, got shape {arr.shape}")
    return arr


class Collection:
    """A named set of vectors with metadata, filtering, and soft delete.

    Construct via :meth:`create` (empty) or :meth:`from_vectors` (bulk load).
    Do not call ``Collection(...)`` directly.
    """

    def __init__(
        self,
        *,
        _index: RabitqIndex,
        _metadata: Dict[int, Dict[str, Any]],
        _tombstones: set,
        _next_id: int,
    ) -> None:
        self._index = _index
        self._metadata = _metadata
        self._tombstones = _tombstones
        self._next_id = _next_id

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def create(cls, dim: int, *, rerank_factor: int = 20, seed: int = 42) -> "Collection":
        """Create an empty collection of the given dimensionality.

        RaBitQ needs at least one vector to build a rotation, so the first
        :meth:`insert` call lazily builds the index. ``dim``/``rerank_factor``/
        ``seed`` are remembered for that first build.
        """
        if dim <= 0:
            raise CollectionError("dim must be > 0")
        coll = cls(_index=None, _metadata={}, _tombstones=set(), _next_id=0)  # type: ignore[arg-type]
        coll._dim = dim
        coll._rerank_factor = rerank_factor
        coll._seed = seed
        return coll

    @classmethod
    def from_vectors(
        cls,
        vectors: NDArray[np.float32],
        *,
        ids: Optional[Sequence[int]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        rerank_factor: int = 20,
        seed: int = 42,
    ) -> "Collection":
        """Bulk-build a collection from an ``(n, dim)`` array in one call.

        This is the fast path — it uses ``RabitqIndex.build`` (parallel
        rotate+pack over rayon), not a loop of ``insert``.
        """
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise CollectionError(f"vectors must be 2D, got {arr.ndim}D")
        n = arr.shape[0]
        if ids is None:
            ids = list(range(n))
        if len(ids) != n:
            raise CollectionError(f"ids length ({len(ids)}) must match vectors row count ({n})")
        if len(set(ids)) != len(ids):
            raise CollectionError("ids must be unique")
        if metadatas is not None and len(metadatas) != n:
            raise CollectionError(f"metadatas length ({len(metadatas)}) must match vectors row count ({n})")

        # RabitqIndex.build assigns row-index ids internally; remap to the
        # caller's external ids via add_batch after an initial single-row
        # build would be wasteful, so instead we build directly with the
        # caller's ids by using add_batch from a throwaway 1-row index.
        # Simpler and correct: build with row ids 0..n-1, then if the
        # caller's ids differ from identity, re-key via a second pass.
        index = RabitqIndex.build(arr, rerank_factor=rerank_factor, seed=seed)
        identity = list(range(n))
        metadata: Dict[int, Dict[str, Any]] = {}
        if list(ids) != identity:
            # Re-key: export (row-id, vector), rebuild with caller ids as a
            # batch add onto a fresh 1-vector-seeded index. We instead just
            # rebuild once more with a remapped id array, which is the
            # correct & simple approach since `build` takes ids implicitly
            # as row index — so rebuild is avoided by overwriting metadata
            # keyed on row index -> caller id via a parallel map, and
            # relying on callers to pass ids == row order for now.
            #
            # NOTE (documented limitation): `RabitqIndex.build` does not
            # accept external ids in M1/M1.5. We therefore keep the
            # row-index ids as the *internal* ids (what `search()` returns)
            # and store the caller's requested id as metadata["_id"] so
            # round-tripping is still possible. A future M2 Collection can
            # push external ids into the Rust layer directly.
            for row, caller_id in enumerate(ids):
                meta = dict(metadatas[row]) if metadatas and metadatas[row] else {}
                meta["_external_id"] = caller_id
                metadata[row] = meta
        elif metadatas is not None:
            for row, m in enumerate(metadatas):
                if m:
                    metadata[row] = dict(m)

        coll = cls(_index=index, _metadata=metadata, _tombstones=set(), _next_id=n)
        coll._dim = arr.shape[1]
        coll._rerank_factor = rerank_factor
        coll._seed = seed
        return coll

    # ── mutation ─────────────────────────────────────────────────────────

    def insert(self, vector: NDArray[np.float32], *, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert one vector, returning its assigned internal id."""
        new_id = self._next_id
        vec = _validate_vector(vector, self._dim, "vector")
        if self._index is None:
            self._index = RabitqIndex.build(
                vec.reshape(1, -1), rerank_factor=self._rerank_factor, seed=self._seed
            )
        else:
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
        n = arr.shape[0]
        if metadatas is not None and len(metadatas) != n:
            raise CollectionError(f"metadatas length ({len(metadatas)}) must match row count ({n})")
        new_ids = list(range(self._next_id, self._next_id + n))
        if self._index is None:
            self._index = RabitqIndex.build(arr, rerank_factor=self._rerank_factor, seed=self._seed)
            # build() assigns row ids 0..n-1; since _next_id was 0, these
            # coincide with new_ids by construction.
        else:
            self._index.add_batch(np.asarray(new_ids, dtype=np.uint64), arr)
        if metadatas:
            for i, m in enumerate(metadatas):
                if m:
                    self._metadata[new_ids[i]] = dict(m)
        self._next_id += n
        return new_ids

    def delete(self, id: int) -> None:
        """Soft-delete: ``id`` is excluded from future :meth:`search` results.

        The underlying RaBitQ index has no physical delete. Call
        :meth:`vacuum` periodically to reclaim space once the tombstone
        fraction gets large (see ADR-352).
        """
        self._tombstones.add(id)

    def vacuum(self) -> int:
        """Physically rebuild the index excluding tombstoned ids.

        Returns the number of rows dropped. O(n) — rebuilds via
        ``export_items`` + a fresh ``build`` call, same cost as the
        initial bulk build.
        """
        if self._index is None or not self._tombstones:
            return 0
        items = self._index.export_items()
        kept = [(i, v) for i, v in items if i not in self._tombstones]
        dropped = len(items) - len(kept)
        if not kept:
            self._index = None
            self._metadata = {}
            self._tombstones = set()
            self._next_id = 0
            return dropped
        ids = np.asarray([i for i, _ in kept], dtype=np.uint64)
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

        ``filter`` is applied client-side (post-hoc), *not* pushed into the
        Rust scan — see this module's docstring. If ``filter`` is set and
        fewer than ``k`` results survive after one overfetched pass, the
        overfetch width doubles (up to 8x) before giving up and returning
        however many matched. ``filter`` may be an exact-match dict
        (``{"category": "news"}``) or a predicate callable.
        """
        if self._index is None:
            return []
        if k <= 0:
            raise CollectionError("k must be > 0")
        qvec = _validate_vector(query, self._dim, "query")

        pred: Optional[Callable[[Dict[str, Any]], bool]]
        if filter is None:
            pred = None
        elif callable(filter):
            pred = filter
        else:
            filt_dict = dict(filter)

            def pred(meta: Dict[str, Any]) -> bool:  # noqa: F811
                return all(meta.get(key) == val for key, val in filt_dict.items())

        tombstones = self._tombstones
        width = k
        hits: List[SearchHit] = []
        seen_widths = set()
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
    ) -> List["tuple[int, NDArray[np.float32], Optional[Dict[str, Any]]]"]:
        """Return ``(id, vector, metadata)`` for every non-tombstoned row,
        sorted by id. Used by the CLI's ``export`` command and by anything
        else that needs a read-only snapshot without reaching into
        ``_index``/``_tombstones`` directly.
        """
        if self._index is None:
            return []
        items = sorted(self._index.export_items(), key=lambda kv: kv[0])
        return [
            (i, v, self._metadata.get(i)) for i, v in items if i not in self._tombstones
        ]

    # ── introspection ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._index is None:
            return 0
        return len(self._index) - len(self._tombstones)

    def stats(self) -> CollectionStats:
        return CollectionStats(
            count=len(self),
            dim=self._dim,
            rerank_factor=self._rerank_factor,
            memory_bytes=self._index.memory_bytes if self._index is not None else 0,
            tombstoned=len(self._tombstones),
        )

    def __repr__(self) -> str:
        return f"Collection(n={len(self)}, dim={self._dim}, tombstoned={len(self._tombstones)})"

    # ── persistence ──────────────────────────────────────────────────────

    @staticmethod
    def meta_path(path: Union[str, os.PathLike]) -> Path:
        """Path of the JSON sidecar for a given index path.

        **Existence of a collection must be checked against this path, not
        the index path** — an empty :meth:`create`d collection has no
        ``.rbpx`` file yet (the underlying index needs >=1 vector to build
        a rotation) but always has a sidecar once :meth:`save` has run once.
        Both the CLI and the MCP server use this (fixed in this session
        after the MCP server's duplicate-collection check used the wrong
        path and silently let a second ``vector_create_collection`` call
        through — see ``tests/test_mcp_server.py::test_create_twice_errors``).
        """
        path = Path(path)
        return path.with_suffix(path.suffix + _META_SUFFIX)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save to ``path`` (the ``.rbpx`` index) plus a ``<path>.meta.json``
        sidecar (metadata dict, tombstones, dim/rerank_factor/seed/next_id).

        Both files are written; a partial write (index saved, sidecar not)
        is possible on a crash between the two calls — documented, not
        silently hidden. M2's RVF persistence (ADR-352) fixes this with a
        single-file container.
        """
        path = Path(path)
        if self._index is not None:
            self._index.save(str(path))
        meta_path = self.meta_path(path)
        sidecar = {
            "dim": self._dim,
            "rerank_factor": self._rerank_factor,
            "seed": self._seed,
            "next_id": self._next_id,
            "tombstones": sorted(self._tombstones),
            "metadata": {str(k): v for k, v in self._metadata.items()},
            "empty": self._index is None,
        }
        meta_path.write_text(json.dumps(sidecar))

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Collection":
        path = Path(path)
        meta_path = cls.meta_path(path)
        if not meta_path.exists():
            raise CollectionError(f"missing sidecar metadata file: {meta_path}")
        sidecar = json.loads(meta_path.read_text())
        index = None if sidecar["empty"] else RabitqIndex.load(str(path))
        coll = cls(
            _index=index,
            _metadata={int(k): v for k, v in sidecar["metadata"].items()},
            _tombstones=set(sidecar["tombstones"]),
            _next_id=sidecar["next_id"],
        )
        coll._dim = sidecar["dim"]
        coll._rerank_factor = sidecar["rerank_factor"]
        coll._seed = sidecar["seed"]
        return coll


__all__ = ["Collection", "CollectionError", "SearchHit", "CollectionStats"]
