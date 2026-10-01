"""Type stub for the compiled PyO3 extension module ``ruvector._native``.

Hand-written per ``docs/sdk/02-strategy.md`` § "Type stubs" (same rationale
as ``__init__.pyi``, split into its own file so ``from ._native import ...``
in ``collection.py`` type-checks under ``mypy --strict`` — without this file
mypy cannot find an implementation for the compiled module and every
subclass of ``RuVectorError`` elsewhere resolves to ``Any``).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__version__: str

class RuVectorError(Exception):
    """Base class for every error raised by the ruvector extension."""

class RabitqIndex:
    """RaBitQ+ index — symmetric 1-bit scan with exact f32 rerank.

    Backed by ``ruvector_rabitq::RabitqPlusIndex``. Build with
    :meth:`build`, query with :meth:`search`, persist via :meth:`save` /
    :meth:`load`.
    """

    @staticmethod
    def build(
        vectors: NDArray[np.float32],
        *,
        ids: Optional[NDArray[np.uint64]] = ...,
        rerank_factor: int = ...,
        seed: int = ...,
    ) -> "RabitqIndex":
        """Build an index from an ``(n, dim)`` float32 array.

        ``vectors`` must be C-contiguous; non-contiguous arrays raise
        ``TypeError``. ``ids`` (optional) assigns the search-result id for
        each row instead of the default ``0..n`` row index; every id must
        fit in ``u32`` (the index's storage width) or this raises
        ``ValueError``. ``rerank_factor`` defaults to 20 (the ADR-154
        recommendation for 100% recall@10 at D=128). ``seed`` defaults
        to 42 for deterministic builds.
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        *,
        rerank_factor: Optional[int] = ...,
    ) -> List[Tuple[int, float]]:
        """Search for the ``k`` nearest neighbours of ``query``.

        Returns a list of ``(id, score)`` tuples in ascending score
        order (squared L2). ``rerank_factor=None`` (the default) reuses
        the value the index was built with.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to ``path`` in the ``.rbpx`` v1 format."""
        ...

    @staticmethod
    def load(path: str) -> "RabitqIndex":
        """Load an index previously written by :meth:`save`."""
        ...

    def add(self, id: int, vector: NDArray[np.float32]) -> None:
        """Append one vector in place (true incremental add, no rebuild).

        Not GIL-released — see ``docs/sdk/02-strategy.md`` § "GIL story".
        Prefer :meth:`add_batch` for more than a few inserts.
        """
        ...

    def add_batch(self, ids: NDArray[np.uint64], vectors: NDArray[np.float32]) -> None:
        """Append many vectors at once. Releases the GIL around the loop.

        Accepts u64 ids but, like :meth:`build`, every id must fit in
        ``u32`` (the index's storage width) or this raises ``ValueError``.
        """
        ...

    def export_items(self) -> List[Tuple[int, NDArray[np.float32]]]:
        """Return every ``(id, vector)`` pair currently held.

        Used by ``ruvector.Collection.vacuum()`` to physically drop
        tombstoned rows by rebuilding without them — there is no
        ``delete`` on the underlying index.
        """
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def memory_bytes(self) -> int: ...
    @property
    def rerank_factor(self) -> int: ...

class HnswIndex:
    """Generic HNSW-backed index — ``ruvector_core::vector_db::VectorDB``.

    Metadata-aware (arbitrary JSON-compatible dict per vector) and filters
    in Rust (``search(..., filter=...)``), not in Python — the ADR-352 M2
    default backend for :class:`ruvector.Collection`. Always in-memory;
    persistence goes through :meth:`export_items` + the Python-side save/
    load sidecar, same idiom as :class:`RabitqIndex`.
    """

    @staticmethod
    def create(
        dim: int,
        *,
        metric: str = ...,
        m: int = ...,
        ef_construction: int = ...,
        ef_search: int = ...,
    ) -> "HnswIndex":
        """``metric``: one of ``cosine`` (default), ``euclidean``/``l2``,
        ``dot``/``dot_product``, ``manhattan``/``l1``.
        """
        ...

    def insert(
        self,
        id: str,
        vector: NDArray[np.float32],
        metadata: Optional[Dict[str, Any]] = ...,
    ) -> str:
        """Returns ``id`` (echoed back for symmetry with the Rust API,
        which can auto-generate an id when none is given — this binding
        always supplies one explicitly)."""
        ...

    def insert_batch(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = ...,
    ) -> List[str]: ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        *,
        filter: Optional[Dict[str, Any]] = ...,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """``filter`` is an exact-match dict, applied in Rust as a
        post-ANN-search retain (not pushed into the HNSW graph traversal
        itself — see the Rust module docstring for the precise claim).

        No per-call ``ef_search``: tune it at :meth:`create` time — see
        ``src/hnsw.rs``'s doc comment on this method for why a per-call
        override isn't offered (``VectorDB::search`` never reads one).
        """
        ...

    def delete(self, id: str) -> bool:
        """Real delete (not a Python tombstone). The underlying HNSW
        graph node is not physically removed until a rebuild (no
        live-delete in the vendored ``hnsw_rs``), but the id is gone from
        every result/count/get immediately.
        """
        ...

    def export_items(
        self,
    ) -> List[Tuple[str, NDArray[np.float32], Optional[Dict[str, Any]]]]: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def dim(self) -> int: ...

class GraphDB:
    """In-memory property graph — ``ruvector_graph::GraphDB``.

    Raw CRUD surface: create/get nodes, create/get edges, outgoing-edge
    traversal. No Cypher here — see ``src/graph.rs``'s module docstring.
    """

    def __init__(self) -> None: ...
    def create_node(
        self,
        labels: Optional[Sequence[str]] = ...,
        properties: Optional[Dict[str, Any]] = ...,
        *,
        id: Optional[str] = ...,
    ) -> str:
        """Returns the new node's id. If ``id`` is given and already
        exists, raises ``RuVectorError`` rather than silently overwriting
        (the underlying Rust ``create_node`` has no such guard and would
        leave stale label/property index entries behind)."""
        ...

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """``{"id": str, "labels": List[str], "properties": Dict[str, Any]}``
        or ``None`` if ``node_id`` does not exist."""
        ...

    def create_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = ...,
        *,
        id: Optional[str] = ...,
    ) -> str:
        """Returns the new edge's id. Raises ``RuVectorError`` if
        ``from_id``/``to_id`` doesn't exist, or if ``id`` is given and
        already exists (see :meth:`create_node`)."""
        ...

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """``{"id": str, "from": str, "to": str, "type": str,
        "properties": Dict[str, Any]}`` or ``None`` if ``edge_id`` does
        not exist."""
        ...

    def get_outgoing_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """Edges whose ``from`` is ``node_id``. Empty list for an unknown
        node id (not an error)."""
        ...

    def query_cypher(self, cypher: str) -> Dict[str, List[Dict[str, Any]]]:
        """Execute a Cypher string, limited to ``MATCH`` execution (no
        cross-pattern joins, variable-length paths, or aggregations).

        Returns ``{"nodes": [...], "edges": [...]}`` using the same
        per-row dict shape as :meth:`get_node`/:meth:`get_edge` — **not**
        a ``RETURN``-projected row set. ``RETURN`` is valid syntax but is
        never applied as a projection: the result is always every
        node/edge the ``MATCH`` touched. Raises ``RuVectorError`` on a
        parse error, on ``CREATE`` (use :meth:`create_node`/
        :meth:`create_edge` instead), or on anything the executor could
        not honour (e.g. a variable-length relationship).
        """
        ...

    def __len__(self) -> int:
        """Node count (networkx convention: not nodes + edges)."""
        ...

    def __repr__(self) -> str: ...

__all__ = ["RabitqIndex", "HnswIndex", "RuVectorError", "__version__", "GraphDB"]
