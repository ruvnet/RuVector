"""LlamaIndex ``VectorStore`` adapter over :class:`ruvector.Collection`.

Verified against ``llama-index-core==0.14.25`` (``pip show llama-index-core``
in the dev venv used to write this module). Package layout has moved across
llama-index major versions; in this version the live contract lives at
``llama_index.core.vector_stores.types``, which exposes **two** shapes:

- ``VectorStore``: a ``@runtime_checkable`` ``Protocol`` — duck-typed, no
  base class required.
- ``BasePydanticVectorStore(BaseComponent, ABC)``: a pydantic ``BaseModel``
  with ``abstractmethod``-decorated ``client`` (property), ``add``,
  ``delete``, and ``query``. This is the class every real first-party
  integration in ``llama_index.core.vector_stores`` (e.g.
  ``SimpleVectorStore``) actually subclasses, and the one the rest of
  llama-index's indexing/retrieval code assumes it can construct via
  pydantic validation — so this module subclasses it, not the bare
  Protocol.

Import boundary: same reasoning as ``ruvector.integrations.langchain`` —
subclassing ``BasePydanticVectorStore`` needs the real class at
class-definition time, so ``llama_index.core`` is imported at module level
here (not lazily), with a clear ``ImportError`` + install hint if it is
missing. Plain ``import ruvector`` never imports this module.

Node/metadata convention: this module reuses llama-index's own
``node_to_metadata_dict`` / ``metadata_dict_to_node`` helpers (the same ones
``SimpleVectorStore``, the Chroma/Pinecone/Qdrant integrations, etc. all
use) to serialize a full ``TextNode`` (text, llama-index metadata, the
node's own id/relationships) into the one JSON-compatible metadata ``dict``
that ``ruvector.Collection`` accepts, with ``remove_text=False`` —
deliberately, after checking the round trip: ``metadata_dict_to_node``
only recovers a node's text from the ``_node_content`` JSON blob itself
(its optional ``text=`` override is for stores like Chroma/Pinecone that
keep text in a separate column outside metadata); dropping it via
``remove_text=True`` as the top-level metadata key gets stripped would
silently lose ``page_content`` on every read back from this adapter,
since ruvector has no second text-storage channel to repopulate it from.
``stores_text = True`` on this class signals to llama-index's query layer
that ``query()``'s result already carries full ``TextNode`` objects, not
just ids to resolve against an external docstore.

ID handling: same bridging problem as the LangChain adapter and the same
fix — ``ruvector.Collection`` ids are backend-assigned ``int``s,
llama-index node/ref_doc ids are caller-chosen ``str``s. A process-local
``str -> int`` map (plus its reverse) is kept on the instance; it is not
persisted by ``Collection.save``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..collection import Collection


def _distance_to_similarity(distance: float, metric: str) -> float:
    """Convert a ``Collection.search`` distance score into a similarity.

    Found in review: this adapter originally put ``hit.score`` (a
    *distance* — lower means closer, see ``Collection.search``'s
    docstring) directly into ``VectorStoreQueryResult.similarities``,
    which every llama-index consumer (``SimilarityPostprocessor``'s
    ``similarity_cutoff``, retrievers that sort or threshold on
    "similarity") assumes is *higher-means-closer*. Left as distance,
    the two best (closest) hits would be the ones a
    ``similarity_cutoff`` filter drops first — silently inverted
    ranking for any downstream similarity-threshold logic, not just a
    cosmetic label mismatch.

    For ``metric="cosine"``: exact conversion, since
    ``ruvector_core::encoding::metric_distance``'s cosine branch is
    literally ``(1.0 - cosine_similarity).max(0.0)`` — so
    ``1.0 - distance`` recovers the real cosine similarity (clamped to
    ``>= 0`` the same way the distance already was).

    For any other metric (``euclidean``/``l2``, ``dot``, ``manhattan``,
    or the rabitq backend's ``squared_l2``): there is no universally
    "correct" bounded similarity to invert to — these are unbounded
    distances. ``1.0 / (1.0 + distance)`` is used instead: monotonic
    decreasing in distance (so rank order and any relative
    similarity-threshold comparisons stay correct), bounded to ``(0, 1]``
    (so it behaves like the ``[0, 1]`` range llama-index callers expect
    from a similarity), but it is **not** a normalized similarity score
    with a principled meaning for these metrics — documented here
    rather than presented as equivalent to the cosine case.
    """
    if metric == "cosine":
        return 1.0 - distance
    return 1.0 / (1.0 + distance)

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        FilterCondition,
        FilterOperator,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )
except ImportError as exc:  # pragma: no cover - exercised in tests via sys.modules patch
    raise ImportError(
        "ruvector.integrations.llamaindex requires the 'llama-index-core' package. "
        "Install it with: pip install 'ruvector[llamaindex]'"
    ) from exc

from llama_index.core.bridge.pydantic import PrivateAttr


def _match_filter_value(operator: "FilterOperator", value: Any, metadata_value: Any) -> bool:
    """Evaluate one ``MetadataFilter`` operator against a stored metadata value.

    Mirrors ``llama_index.core.vector_stores.utils.build_metadata_filter_fn``'s
    operator semantics, but operates directly on a metadata ``dict`` rather
    than through an id -> metadata lookup indirection — ruvector's own
    filter callable (``Collection.search(..., filter=...)``) already
    receives the metadata dict directly, so there is no id to look up.
    """
    if metadata_value is None:
        return operator in (FilterOperator.NE, FilterOperator.NIN)
    if operator == FilterOperator.EQ:
        return bool(metadata_value == value)
    if operator == FilterOperator.NE:
        return bool(metadata_value != value)
    if operator == FilterOperator.GT:
        return bool(metadata_value > value)
    if operator == FilterOperator.GTE:
        return bool(metadata_value >= value)
    if operator == FilterOperator.LT:
        return bool(metadata_value < value)
    if operator == FilterOperator.LTE:
        return bool(metadata_value <= value)
    if operator == FilterOperator.IN:
        return bool(metadata_value in value)
    if operator == FilterOperator.NIN:
        return bool(metadata_value not in value)
    if operator == FilterOperator.CONTAINS:
        return bool(value in metadata_value)
    if operator == FilterOperator.TEXT_MATCH:
        return bool(isinstance(value, str) and isinstance(metadata_value, str) and value in metadata_value)
    if operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
        return bool(
            isinstance(value, str) and isinstance(metadata_value, str) and value.lower() in metadata_value.lower()
        )
    if operator == FilterOperator.ALL:
        return all(v in metadata_value for v in value)
    if operator == FilterOperator.ANY:
        return any(v in metadata_value for v in value)
    if operator == FilterOperator.IS_EMPTY:
        return metadata_value is None or metadata_value == [] or metadata_value == ""
    raise ValueError(f"Unsupported FilterOperator: {operator!r}")


def _filters_to_predicate(filters: "MetadataFilters") -> Callable[[Dict[str, Any]], bool]:
    """Translate a ``MetadataFilters`` tree into a ``Collection.search``-compatible predicate.

    Always falls back to ruvector's Python-side filter path (the structural
    limitation ``Collection.search`` already documents for callable
    predicates — an arbitrary Python function cannot cross the PyO3
    boundary into the Rust filter pushdown), rather than only handling the
    EQ+AND subset that could be pushed down as an exact-match dict. This
    keeps every llama-index filter shape (nested ``MetadataFilters``, OR/NOT
    conditions, non-EQ operators like ``>``/``text_match``) correct, at the
    cost of the Rust-side filter pushdown fast path the ``hnsw`` backend
    has for plain dict filters — a real tradeoff, not an oversight.
    """

    def predicate(metadata: Dict[str, Any]) -> bool:
        results = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                results.append(_filters_to_predicate(f)(metadata))
            else:
                results.append(_match_filter_value(f.operator, f.value, metadata.get(f.key)))
        condition = filters.condition or FilterCondition.AND
        if condition == FilterCondition.OR:
            return any(results)
        if condition == FilterCondition.NOT:
            return not any(results)
        return all(results)

    return predicate


class RuVectorStore(BasePydanticVectorStore):
    """llama-index ``BasePydanticVectorStore`` backed by a :class:`ruvector.Collection`."""

    stores_text: bool = True
    is_embedding_query: bool = True

    _collection: Collection = PrivateAttr()
    _id_to_int: Dict[str, int] = PrivateAttr()
    _int_to_id: Dict[int, str] = PrivateAttr()

    def __init__(self, collection: Collection, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._collection = collection
        self._id_to_int = {}
        self._int_to_id = {}

    @classmethod
    def class_name(cls) -> str:
        return "RuVectorStore"

    @property
    def client(self) -> Collection:
        """The wrapped :class:`ruvector.Collection` (llama-index's "native client" escape hatch)."""
        return self._collection

    def add(self, nodes: Sequence["BaseNode"], **add_kwargs: Any) -> List[str]:
        """Insert ``nodes`` (each already carrying its embedding) and return their node ids."""
        out_ids: List[str] = []
        for node in nodes:
            if node.node_id in self._id_to_int:
                raise ValueError(f"node id {node.node_id!r} already exists in this RuVectorStore")
            embedding = node.get_embedding()
            metadata = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
            vec = np.asarray(embedding, dtype=np.float32)
            int_id = self._collection.insert(vec, metadata=metadata)
            self._id_to_int[node.node_id] = int_id
            self._int_to_id[int_id] = node.node_id
            out_ids.append(node.node_id)
        return out_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node whose ``ref_doc_id`` (source document id) matches.

        llama-index's ``BasePydanticVectorStore.delete`` contract keys on
        ``ref_doc_id``, not on a node's own id — a document can have been
        chunked into many nodes, and deleting the document means deleting
        all of them. The per-node ``ref_doc_id`` was captured by
        ``node_to_metadata_dict`` into the ``"ref_doc_id"`` metadata key at
        ``add`` time (same key every llama-index vector-store integration
        reads back for this).
        """
        to_drop = [
            node_id
            for node_id, int_id in self._id_to_int.items()
            if (self._collection.get_metadata(int_id) or {}).get("ref_doc_id") == ref_doc_id
        ]
        for node_id in to_drop:
            int_id = self._id_to_int.pop(node_id)
            self._int_to_id.pop(int_id, None)
            self._collection.delete(int_id)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional["MetadataFilters"] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete by node id and/or metadata filter (the finer-grained sibling of :meth:`delete`)."""
        predicate = _filters_to_predicate(filters) if filters is not None else None
        candidates = node_ids if node_ids is not None else list(self._id_to_int.keys())
        for node_id in list(candidates):
            int_id = self._id_to_int.get(node_id)
            if int_id is None:
                continue
            if predicate is not None:
                metadata = self._collection.get_metadata(int_id) or {}
                if not predicate(metadata):
                    continue
            self._id_to_int.pop(node_id, None)
            self._int_to_id.pop(int_id, None)
            self._collection.delete(int_id)

    def clear(self) -> None:
        for int_id in list(self._int_to_id):
            self._collection.delete(int_id)
        self._id_to_int.clear()
        self._int_to_id.clear()

    def query(self, query: "VectorStoreQuery", **kwargs: Any) -> "VectorStoreQueryResult":
        """Run a similarity search.

        Only ``VectorStoreQueryMode.DEFAULT`` (plain dense kNN) is
        supported — ``ruvector.Collection`` has no sparse/BM25/hybrid
        index, no learner-fit modes (SVM/logistic/linear regression over
        embeddings), and no native MMR, so ``SPARSE``/``HYBRID``/
        ``TEXT_SEARCH``/``SEMANTIC_HYBRID``/``SVM``/``LOGISTIC_REGRESSION``/
        ``LINEAR_REGRESSION``/``MMR`` all raise ``NotImplementedError``
        rather than silently degrading to a dense search that would ignore
        the mode's actual semantics (e.g. MMR's diversity re-ranking).
        """
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"RuVectorStore only supports VectorStoreQueryMode.DEFAULT, got {query.mode!r}")
        if query.query_embedding is None:
            raise ValueError("RuVectorStore.query requires query.query_embedding (no sparse/text-only mode)")

        predicate = _filters_to_predicate(query.filters) if query.filters is not None else None
        query_vec = np.asarray(query.query_embedding, dtype=np.float32)
        hits = self._collection.search(query_vec, query.similarity_top_k, filter=predicate)

        if query.node_ids is not None:
            allowed = set(query.node_ids)
            hits = [h for h in hits if self._int_to_id.get(h.id) in allowed]

        metric = self._collection.metric
        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []
        for hit in hits:
            metadata = hit.metadata or {}
            node = metadata_dict_to_node(metadata)
            nodes.append(node)
            similarities.append(_distance_to_similarity(hit.score, metric))
            ids.append(node.node_id)
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


__all__ = ["RuVectorStore"]
