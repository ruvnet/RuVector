"""LionAGI MemoryStore backed by a bounded RuVectorStore and an injected embedder."""
from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from uuid import UUID

from lionagi.protocols.graph.node import Node
from lionagi.protocols.memory import MemoryItem, MemoryQuery

from . import RuVectorStore, SearchHit, _integer, _text

__all__ = ["RuVectorMemoryStore"]

_ENVELOPE = "ruvector_lionagi_memory_v1"


class RuVectorMemoryStore:
    """Text memory implementing LionAGI's structural MemoryStore protocol.

    The host supplies an async ``embedder(text)`` and a dedicated store for its
    authorization scope. This adapter neither selects providers nor handles
    credentials. Metadata filters are ANDed; tags match any supplied tag.
    Filtered semantic searches rank every matching record exactly, bounded by
    the underlying store capacity. Unfiltered searches use approximate HNSW.

    IDs are append-only: duplicate IDs are rejected by the underlying store.
    Async cancellation can leave an already-started insertion committed. Its
    payload and vector are committed together, so retrieve the UUID before
    retrying. Results are detached snapshots, with string content only.
    """

    def __init__(
        self,
        store: RuVectorStore,
        embedder: Callable[[str], Awaitable[Sequence[float]]],
    ) -> None:
        if not isinstance(store, RuVectorStore):
            raise TypeError("store must be a RuVectorStore")
        if not callable(embedder):
            raise TypeError("embedder must be an async callable")
        self._store = store
        self._embedder = embedder

    def _tags(self, tags: Any) -> list[str]:
        if type(tags) is not list or len(tags) > 128:
            raise ValueError("tags must be a list containing at most 128 strings")
        return [_text(tag, 256, "tag") for tag in tags]

    async def store(self, item: MemoryItem) -> UUID:
        """Embed, validate and insert a detached text memory; return its UUID."""
        if not isinstance(item, MemoryItem):
            raise TypeError("item must be a LionAGI MemoryItem")
        content = _text(item.content, self._store.max_content_bytes, "content")
        item_id = UUID(str(item.id))
        created_at = float(item.created_at)
        if not math.isfinite(created_at):
            raise ValueError("created_at must be finite")
        envelope = {
            _ENVELOPE: {
                "created_at": created_at,
                "tags": self._tags(item.tags),
                "metadata": item.metadata,
            }
        }
        # Bound and detach before awaiting user-supplied embedding code.
        metadata = json.loads(self._store._metadata(envelope))
        vector = await self._embedder(content)
        # Validate before Node's float coercion so bools/strings cannot bypass
        # the vector input contract. The adapter normalizes again on insertion.
        normalized = self._store._vector(vector)
        node = Node(
            id=item_id,
            content=content,
            metadata=metadata,
            embedding=normalized.tolist(),
        )
        await self._store.aadd_nodes([node])
        return item_id

    @staticmethod
    def _memory(hit: SearchHit) -> MemoryItem:
        payload = hit.metadata.get(_ENVELOPE)
        if type(payload) is not dict:
            raise ValueError("record is not a RuVector LionAGI memory")
        # Never use Element.from_dict/from_json: those dispatch on lion_class.
        return MemoryItem(
            id=UUID(hit.node_id),
            created_at=payload["created_at"],
            content=hit.content,
            tags=payload["tags"],
            metadata=payload["metadata"],
        )

    async def retrieve(self, item_id: UUID) -> MemoryItem | None:
        """Read a detached record without calling the embedding provider."""
        if not isinstance(item_id, UUID):
            raise TypeError("item_id must be a UUID")
        hit = await asyncio.to_thread(self._store.get, str(item_id))
        return None if hit is None else self._memory(hit)

    async def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """Search text semantically, or enumerate matching memories without text."""
        if not isinstance(query, MemoryQuery):
            raise TypeError("query must be a LionAGI MemoryQuery")
        limit = _integer(query.limit, "limit", self._store.max_k)
        tags = self._tags(query.tags) if query.tags is not None else []
        filters = query.filters if query.filters is not None else {}
        # Reuse the same finite JSON/depth/byte bounds as stored metadata.
        filters = json.loads(self._store._metadata(filters))
        text = query.text
        if text is not None:
            text = _text(text, self._store.max_content_bytes, "query text")
        wanted_tags = frozenset(tags)

        def matches(metadata: dict[str, Any]) -> bool:
            payload = metadata.get(_ENVELOPE)
            if type(payload) is not dict:
                return False
            record_tags = payload.get("tags")
            record_metadata = payload.get("metadata")
            if type(record_tags) is not list or type(record_metadata) is not dict:
                return False
            return (not wanted_tags or bool(wanted_tags.intersection(record_tags))) and all(
                record_metadata.get(key) == value for key, value in filters.items()
            )

        if not text:
            hits = await asyncio.to_thread(self._store.scan, predicate=matches, k=limit)
        else:
            vector = await self._embedder(text)
            # A dedicated store contains only memory envelopes. Explicit filters
            # trigger exact ranking across all eligible records, avoiding the
            # recall loss inherent in filtering an approximate top-k afterward.
            predicate = matches if tags or filters else None
            hits = await self._store.asearch(vector, k=limit, predicate=predicate)
        return [self._memory(hit) for hit in hits]
