"""Offline acceptance example: real native search, Branch tool and MemoryStore."""
from __future__ import annotations

import asyncio

from lionagi import Branch, MemoryItem, MemoryQuery, Node
from ruvector_lionagi import RuVectorMemoryStore, RuVectorStore


async def example_embedding(text: str) -> list[float]:
    """Deterministic toy vectors for this example, not a semantic language model."""
    lowered = text.lower()
    if any(word in lowered for word in ("apple", "fruit", "pear")):
        return [1.0, 0.1, 0.0]
    if any(word in lowered for word in ("rust", "vector", "search")):
        return [0.0, 0.1, 1.0]
    return [0.0, 1.0, 0.0]


async def main() -> None:
    # The application supplies embeddings. These examples need no API keys.
    store = RuVectorStore(dimensions=3, capacity=100)
    document = Node(
        content="RuVector provides native Rust vector search.",
        embedding=await example_embedding("Rust vector search"),
        metadata={"source": "local-example"},
    )
    await store.aadd_nodes([document])
    branch = Branch(tools=[store.as_tool(example_embedding)])
    responses = await branch.act(
        {"function": "ruvector_search", "arguments": {"query": "vector search", "k": 1}},
        suppress_errors=False,
    )
    output = responses[0].output
    assert output["untrusted_data"] is True
    assert output["hits"][0]["node_id"] == str(document.id)

    # Keep memory in its own store: its payload carries memory tags/provenance.
    memory = RuVectorMemoryStore(RuVectorStore(3, capacity=100), example_embedding)
    memory_branch = Branch(memory=memory)
    item = MemoryItem(
        content="Apple and pear are fruit.",
        tags=["food"],
        metadata={"source": "local-example"},
    )
    item_id = await memory_branch.memory.store(item)
    restored = await memory_branch.memory.retrieve(item_id)
    assert restored is not None and restored.content == item.content
    matches = await memory_branch.memory.search(
        MemoryQuery(text="fruit", tags=["food"], filters={"source": "local-example"}, limit=1)
    )
    assert [match.id for match in matches] == [item_id]
    excluded = await memory_branch.memory.search(MemoryQuery(text="fruit", tags=["hardware"]))
    assert excluded == []
    print("PASS: native retrieval, Branch.act, memory roundtrip and pre-ranking filters")


if __name__ == "__main__":
    asyncio.run(main())
