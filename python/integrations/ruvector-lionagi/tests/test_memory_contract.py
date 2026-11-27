"""LionAGI MemoryStore protocol, semantic search and snapshot isolation."""
import asyncio

import pytest
from lionagi import Branch
from lionagi.protocols.memory import MemoryItem, MemoryQuery, MemoryStore
from ruvector_lionagi import RuVectorMemoryStore, RuVectorStore


async def embed(text):
    return [1.0, 0.0] if "apple" in text else [0.0, 1.0]


def test_memory_roundtrip_and_branch_attachment():
    async def run():
        memory = RuVectorMemoryStore(RuVectorStore(2), embed)
        assert isinstance(memory, MemoryStore)
        Branch(memory=memory)
        item = MemoryItem(content="apple", tags=["fruit"], metadata={"nested": {"owner": "one"}})
        ident = await memory.store(item)
        assert ident == item.id
        item.metadata["nested"]["owner"] = "mutated"
        got = await memory.retrieve(ident)
        assert got.content == "apple"
        assert got.tags == ["fruit"]
        assert got.metadata["nested"]["owner"] == "one"
        got.tags.append("changed")
        got.metadata["nested"]["owner"] = "changed"
        again = await memory.retrieve(ident)
        assert again.tags == ["fruit"]
        assert again.metadata["nested"]["owner"] == "one"
    asyncio.run(run())


def test_memory_search_tags_any_metadata_all():
    async def run():
        memory = RuVectorMemoryStore(RuVectorStore(2), embed)
        a = MemoryItem(content="apple", tags=["fruit"], metadata={"owner": "a", "public": True})
        b = MemoryItem(content="other", tags=["vegetable"], metadata={"owner": "a", "public": False})
        c = MemoryItem(content="apple private", tags=["fruit"], metadata={"owner": "b", "public": True})
        for item in (a, b, c):
            await memory.store(item)
        hits = await memory.search(MemoryQuery(text="apple", filters={"owner": "a", "public": True}, limit=2))
        assert [h.id for h in hits] == [a.id]
        hits = await memory.search(MemoryQuery(tags=["fruit", "vegetable"], filters={"owner": "a"}))
        assert {h.id for h in hits} == {a.id, b.id}
        hits = await memory.search(MemoryQuery(text="apple", limit=1))
        assert len(hits) == 1 and hits[0].content.startswith("apple")
    asyncio.run(run())


def test_missing_memory_returns_none():
    async def run():
        from uuid import uuid4
        memory = RuVectorMemoryStore(RuVectorStore(2), embed)
        assert await memory.retrieve(uuid4()) is None
        assert await memory.search(MemoryQuery(text="apple")) == []
    asyncio.run(run())


def test_bad_embedder_leaves_store_empty():
    async def run():
        async def bad(text):
            return [float("nan"), 1]
        store = RuVectorStore(2)
        memory = RuVectorMemoryStore(store, bad)
        with pytest.raises((ValueError, TypeError)):
            await memory.store(MemoryItem(content="apple"))
        assert len(store) == 0
    asyncio.run(run())


def test_memory_snapshot_is_taken_before_embedder_await():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def delayed(text):
            started.set()
            await release.wait()
            return [1, 0]
        memory = RuVectorMemoryStore(RuVectorStore(2), delayed)
        item = MemoryItem(content="original", tags=["original"], metadata={"nested": ["original"]})
        task = asyncio.create_task(memory.store(item))
        await started.wait()
        item.content = "changed"
        item.tags.append("changed")
        item.metadata["nested"].append("changed")
        release.set()
        ident = await task
        got = await memory.retrieve(ident)
        assert got.content == "original"
        assert got.tags == ["original"]
        assert got.metadata == {"nested": ["original"]}
    asyncio.run(run())


def test_cancel_during_embedding_writes_nothing():
    async def run():
        started = asyncio.Event()
        async def delayed(text):
            started.set()
            await asyncio.Future()
        store = RuVectorStore(2)
        memory = RuVectorMemoryStore(store, delayed)
        item = MemoryItem(content="cancel me")
        task = asyncio.create_task(memory.store(item))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(store) == 0
        assert await memory.retrieve(item.id) is None
    asyncio.run(run())
