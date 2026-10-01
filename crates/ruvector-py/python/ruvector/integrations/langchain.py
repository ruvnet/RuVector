"""LangChain ``VectorStore`` adapter over :class:`ruvector.Collection`.

Verified against ``langchain-core==1.6.6`` (``pip show langchain-core`` in the
dev venv used to write this module). The live ABC
(``langchain_core.vectorstores.VectorStore``) only *requires*
``similarity_search`` and ``from_texts`` (``VectorStore.__abstractmethods__``
== ``{"similarity_search", "from_texts"}``) — everything else
(``add_texts``, ``similarity_search_with_score``, ``delete``, the
``embeddings`` property, ``_select_relevance_score_fn``) is a concrete
method on the base class that raises ``NotImplementedError`` unless a
subclass overrides it. This module overrides all of those except
``_select_relevance_score_fn`` (see the docstring on that method below for
why).

Import boundary (deliberate choice, not an oversight): ``langchain_core`` is
imported at **module level**, right here — not lazily inside functions. A
user who writes ``import ruvector.integrations.langchain`` or
``from ruvector.integrations.langchain import RuVectorStore`` has already
opted into the langchain dependency; subclassing ``VectorStore`` requires
the base class to exist at class-definition time, so there is no honest way
to defer this further while still exporting a real ``VectorStore`` subclass
from this module. Plain ``import ruvector`` (and ``import
ruvector.integrations``) remain langchain-free: this module is never
imported by either of those. If ``langchain-core`` is not installed, the
``ImportError`` below carries an explicit install hint rather than a bare
traceback.

Text/metadata convention: a LangChain ``Document``'s ``page_content`` is
stored in the ruvector metadata dict under the key given by ``text_key``
(default ``"text"``), merged with the document's own ``metadata`` dict. On
read, that key is popped back out into ``Document.page_content`` and the
rest of the dict becomes ``Document.metadata``. If a caller's own metadata
already has a ``text_key`` entry, the page content silently overwrites it
on write (documented here, not hidden) — pick a non-colliding
``text_key`` if that matters.

ID handling: ``ruvector.Collection`` ids are backend-assigned, dense,
non-negative ``int``s (see ``Collection.insert``'s return value);
LangChain's ``VectorStore`` contract is entirely in terms of caller-chosen
or caller-visible ``str`` ids (``add_texts(..., ids: list[str] | None)``,
``delete(ids: list[str] | None)``). This adapter keeps a ``str -> int``
id map (and its reverse) alongside the wrapped ``Collection`` to bridge
the two id spaces; the map is process-local, in-memory, and is not part of
``Collection.save``/``Collection.load`` — persisting a ``RuVectorStore``
across processes with stable LangChain-visible ids is out of scope here.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

import numpy as np

from ..collection import Collection, SearchHit

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - exercised in tests via sys.modules patch
    raise ImportError(
        "ruvector.integrations.langchain requires the 'langchain-core' package. "
        "Install it with: pip install 'ruvector[langchain]'"
    ) from exc

VST = TypeVar("VST", bound="RuVectorStore")

_DEFAULT_TEXT_KEY = "text"


class RuVectorStore(VectorStore):
    """LangChain ``VectorStore`` backed by a :class:`ruvector.Collection`.

    Construct directly around an existing collection::

        coll = Collection.create(dim=384)
        store = RuVectorStore(coll, my_embeddings)

    or via the standard LangChain classmethod, which creates a new
    in-memory collection sized to the embedding's own output dimension::

        store = RuVectorStore.from_texts(texts, my_embeddings)
    """

    def __init__(
        self,
        collection: Collection,
        embedding: "Embeddings",
        *,
        text_key: str = _DEFAULT_TEXT_KEY,
    ) -> None:
        self._collection = collection
        self._embedding = embedding
        self._text_key = text_key
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: Dict[int, str] = {}

    # ── LangChain-required surface ──────────────────────────────────────

    @property
    def embeddings(self) -> Optional["Embeddings"]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed ``texts`` and insert them, returning their LangChain ids.

        ``ids``, if given, must not collide with ids already tracked by
        this store instance (mirrors most LangChain integrations' "ids are
        caller-managed" contract); when omitted, a fresh ``uuid4`` hex
        string is minted per text, matching the convention used by
        LangChain's own in-memory/FAISS-style stores.
        """
        text_list = list(texts)
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError(f"metadatas length ({len(metadatas)}) must match texts length ({len(text_list)})")
        if ids is not None and len(ids) != len(text_list):
            raise ValueError(f"ids length ({len(ids)}) must match texts length ({len(text_list)})")
        if ids is not None:
            for doc_id in ids:
                if doc_id in self._id_to_int:
                    raise ValueError(f"id {doc_id!r} already exists in this RuVectorStore")

        if not text_list:
            return []

        embeddings = self._embedding.embed_documents(text_list)
        resolved_ids = list(ids) if ids is not None else [uuid.uuid4().hex for _ in text_list]

        out_ids: List[str] = []
        for i, (text, vec) in enumerate(zip(text_list, embeddings)):
            doc_id = resolved_ids[i]
            full_metadata: Dict[str, Any] = dict(metadatas[i]) if metadatas is not None else {}
            full_metadata[self._text_key] = text
            arr = np.asarray(vec, dtype=np.float32)
            int_id = self._collection.insert(arr, metadata=full_metadata)
            self._id_to_int[doc_id] = int_id
            self._int_to_id[int_id] = doc_id
            out_ids.append(doc_id)
        return out_ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return ``(Document, score)`` pairs.

        ``score`` is whatever :class:`ruvector.SearchHit.score` reports —
        a **distance** for the default cosine/L2 backends (lower is
        closer), not a LangChain-style "higher is more similar" relevance
        score. This is called out explicitly because LangChain's own
        convention varies by integration; this adapter does not override
        ``_select_relevance_score_fn`` (so
        ``similarity_search_with_relevance_scores`` raises
        ``NotImplementedError``, the base class's own default when a
        subclass declines to define a distance->relevance mapping) rather
        than fabricate a normalization that would not hold for every
        ``Collection`` backend/metric combination.
        """
        filter_ = kwargs.get("filter")
        query_vec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        hits = self._collection.search(query_vec, k, filter=filter_)
        return [(self._hit_to_document(hit), hit.score) for hit in hits]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by LangChain id. ``ids=None`` deletes every id this store instance knows about."""
        target_ids = list(ids) if ids is not None else list(self._id_to_int.keys())
        ok = True
        for doc_id in target_ids:
            int_id = self._id_to_int.pop(doc_id, None)
            if int_id is None:
                ok = False
                continue
            self._int_to_id.pop(int_id, None)
            self._collection.delete(int_id)
        return ok

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: "Embeddings",
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        text_key: str = _DEFAULT_TEXT_KEY,
        collection_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VST:
        """Build a fresh in-memory :class:`Collection` sized to ``embedding``'s output dim.

        The dimension can only be known by actually calling
        ``embedding.embed_documents`` (no LangChain ``Embeddings`` exposes a
        ``dim`` property), so for an empty ``texts`` list this raises —
        there is no way to size the collection without at least one vector.
        ``collection_kwargs`` is forwarded to ``Collection.create`` (e.g.
        ``{"backend": "rabitq", "metric": "cosine"}``); unrecognized
        top-level ``**kwargs`` are ignored, matching the permissive
        ``**kwargs: Any`` on the base class's ``from_texts`` signature.
        """
        if not texts:
            raise ValueError("RuVectorStore.from_texts requires at least one text to size the collection's dim")
        probe_vec = embedding.embed_documents([texts[0]])[0]
        dim = len(probe_vec)
        coll = Collection.create(dim=dim, **(collection_kwargs or {}))
        store = cls(coll, embedding, text_key=text_key)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    # ── helpers ──────────────────────────────────────────────────────────

    def _hit_to_document(self, hit: SearchHit) -> Document:
        metadata = dict(hit.metadata) if hit.metadata else {}
        text = metadata.pop(self._text_key, "")
        doc_id = self._int_to_id.get(hit.id)
        return Document(page_content=text, metadata=metadata, id=doc_id)


__all__ = ["RuVectorStore"]
