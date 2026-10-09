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

class GnnLayer:
    """GNN forward-pass rerank layer — ``ruvector_gnn::layer::RuvectorLayer``.

    Weights are randomly initialised (Xavier/Glorot) at construction time;
    there is no training step in this binding, so ``forward()`` on a
    freshly-built layer is a random projection, not a quality improvement,
    until the weights are trained or loaded via :meth:`from_json`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float = ...,
    ) -> None:
        """``heads`` must divide ``hidden_dim``; ``dropout`` must be in
        ``[0.0, 1.0]`` (both raise ``RuVectorError`` otherwise).
        """
        ...

    def forward(
        self,
        node: NDArray[np.float32],
        neighbors: NDArray[np.float32],
        weights: Optional[NDArray[np.float32]] = ...,
    ) -> NDArray[np.float32]:
        """``node``: shape ``(input_dim,)``. ``neighbors``: shape
        ``(n, input_dim)``, ``n`` may be ``0``. ``weights``: optional
        shape ``(n,)``, defaults to uniform. Returns shape ``(hidden_dim,)``.
        """
        ...

    def to_json(self) -> str:
        """Serialize this layer (including its random/trained weights)."""
        ...

    @staticmethod
    def from_json(data: str) -> "GnnLayer":
        """Deserialize a value previously produced by :meth:`to_json`."""
        ...

    def __repr__(self) -> str: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def hidden_dim(self) -> int: ...
    @property
    def heads(self) -> int: ...
    @property
    def dropout(self) -> float: ...

class AttentionReranker:
    """Attention-based rerank — ``softmax(QK^T/√d)V``, trainless and
    deterministic (``ruvector_attention::attention::ScaledDotProductAttention``).
    """

    def __init__(self, dim: int) -> None: ...
    def rerank(
        self,
        query: NDArray[np.float32],
        candidates: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """``query``: shape ``(dim,)``. ``candidates``: shape ``(n, dim)``,
        used as both keys and values. Returns ``(blended, weights)``:
        the attention-weighted blend (shape ``(dim,)``) and the raw
        per-candidate softmax weight (shape ``(n,)``, sums to ~1.0, same
        row order as ``candidates``).
        """
        ...

    def __repr__(self) -> str: ...
    @property
    def dim(self) -> int: ...

def kmeans(
    vectors: NDArray[np.float32],
    k: int,
    *,
    iters: int = ...,
) -> Tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    """Run Lloyd's k-means over ``vectors`` (shape ``(n, dim)``).

    Backed by ``ruvector_cluster_rag::cluster::kmeans``. Returns
    ``(assignments, centroids, cohesion, cluster_sizes)``:

    - ``assignments``: ``int64[n]`` cluster id per input row.
    - ``centroids``: ``float32[k, dim]`` final cluster centroids.
    - ``cohesion``: ``float32[k]`` mean cosine similarity of each
      cluster's members to their centroid (higher is tighter).
    - ``cluster_sizes``: ``int64[k]`` member count per cluster.

    ``vectors`` must be C-contiguous float32 (``TypeError`` otherwise).
    Raises ``ValueError`` for ``k == 0``, ``k > n``, an empty input, or
    any non-finite (``NaN``/``inf``) coordinate.
    """
    ...

class SonaEngine:
    """SONA (inference-only) adaptive-LoRA engine —
    ``ruvector_sona::SonaEngine``.

    Binds the forward-pass half only (``apply_micro_lora``,
    ``apply_base_lora``, ``stats``, ``save_state``/``load_state``). The
    online-learning API (trajectories, ``tick``, ``force_learn``,
    ``find_patterns``) is not exposed by this binding.

    On a freshly constructed engine, both LoRA forward passes are an
    **exact identity transform** (zero-initialised projections + residual
    forward pass) — this is a usable API hook, not a quality improvement,
    until the (unexposed) online-learning loop has actually adapted the
    weights.
    """

    def __init__(self, hidden_dim: int) -> None:
        """``hidden_dim`` must be > 0."""
        ...

    def apply_micro_lora(self, input: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the micro-LoRA transform. ``input`` length must equal
        :attr:`hidden_dim`. Identity on a fresh, untrained engine."""
        ...

    def apply_base_lora(
        self, layer_idx: int, input: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Apply the base-LoRA transform for layer ``layer_idx``.
        ``input`` length must equal :attr:`hidden_dim`; ``layer_idx`` must
        be ``< num_layers`` (raises ``ValueError`` otherwise — unlike the
        underlying Rust fn, which silently no-ops out of range).
        """
        ...

    def stats(self) -> Dict[str, Any]:
        """Engine statistics: ``trajectories_recorded``,
        ``trajectories_buffered``, ``trajectories_dropped``,
        ``buffer_success_rate``, ``patterns_stored``, ``patterns_learned``,
        ``ewc_tasks``, ``instant_enabled``, ``background_enabled``.
        """
        ...

    def save_state(self) -> str:
        """Serialize learned patterns + EWC task count + enabled flags to
        JSON. Does **not** include LoRA weights."""
        ...

    def load_state(self, state_json: str) -> int:
        """Restore patterns from a ``save_state`` JSON string; returns the
        count restored. Raises ``RuVectorError`` on malformed JSON (unlike
        the NAPI binding, which swallows the error and returns 0)."""
        ...

    def __repr__(self) -> str: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def hidden_dim(self) -> int: ...
    @property
    def is_enabled(self) -> bool: ...
    @is_enabled.setter
    def is_enabled(self, value: bool) -> None: ...

__all__ = [
    "RabitqIndex",
    "HnswIndex",
    "RuVectorError",
    "__version__",
    "GraphDB",
    "GnnLayer",
    "AttentionReranker",
    "kmeans",
    "SonaEngine",
]
