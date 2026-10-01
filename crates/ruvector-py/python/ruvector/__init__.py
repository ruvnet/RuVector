"""ruvector — ultra-low-latency vector search, backed by a Rust core.

M1 surface: ``RabitqIndex`` plus the ``RuVectorError`` base exception.
M1.5 (ADR-352) adds ``Collection``: ids, metadata, filtering, delete, and a
persistence sidecar over the same M1 index. M2 adds ``HnswIndex`` — the
generic, metadata-aware, Rust-filtered backend that is ``Collection``'s
*default* as of this milestone (``backend="hnsw"``; ``backend="rabitq"``
still works). See ``docs/sdk/04-milestones.md`` for what M3/M4 still add
(Embedder, A2aClient) and ``docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md``
for the full CLI/MCP/Collection/integrations scope this module now carries.

Attribute access is lazy (PEP 562 module ``__getattr__``) rather than eager
top-level imports: ``import ruvector`` alone does not pull in numpy or the
compiled ``_native`` extension, only ``ruvector.RabitqIndex`` /
``ruvector.Collection`` (or ``from ruvector import ...`` either name) does.
This matters for ``ruvector.cli``'s fast-startup requirement — measured via
``python -X importtime -c "from ruvector.cli import main"`` in
``docs/sdk/LOOP-STATE.md``; eager imports here previously made `ruvector
--help` pay for all of numpy before this fix.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    from ruvector._native import HnswIndex as HnswIndex
    from ruvector._native import RabitqIndex as RabitqIndex
    from ruvector._native import RuVectorError as RuVectorError
    from ruvector.collection import Collection as Collection
    from ruvector.collection import CollectionError as CollectionError
    from ruvector.collection import CollectionStats as CollectionStats
    from ruvector.collection import SearchHit as SearchHit

__all__ = [
    "RabitqIndex",
    "HnswIndex",
    "RuVectorError",
    "__version__",
    "Collection",
    "CollectionError",
    "CollectionStats",
    "SearchHit",
]

_NATIVE_NAMES = {"RabitqIndex", "HnswIndex", "RuVectorError", "__version__"}
_COLLECTION_NAMES = {"Collection", "CollectionError", "CollectionStats", "SearchHit"}


def __getattr__(name: str) -> Any:
    if name in _NATIVE_NAMES:
        from ruvector import _native

        return getattr(_native, name)
    if name in _COLLECTION_NAMES:
        from ruvector import collection as _collection_mod

        return getattr(_collection_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
