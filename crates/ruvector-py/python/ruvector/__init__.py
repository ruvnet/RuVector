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

``__getattr__`` below dispatches by probing the compiled ``_native`` module
and ``collection`` module *dynamically* (``hasattr``) rather than against a
hardcoded name set. This is a deliberate fix, not just a style choice: the
original hardcoded ``_NATIVE_NAMES`` set was once out of sync with the
compiled module (``HnswIndex`` was addable to ``_native`` in Rust but
unreachable as ``ruvector.HnswIndex`` until someone remembered to also add
its name here — a real bug, found during integration testing). Dynamic
dispatch makes that whole bug class structurally impossible: a new PyO3
class is reachable the moment ``lib.rs`` registers it, with nothing else
to remember to update in this file. ``__all__`` is still a static list
(needed for ``from ruvector import *`` and some IDE tooling), so it does
still need a one-line addition per new public class — but getting that
wrong only affects wildcard-import/autocomplete discoverability, never
correctness, which is the bug class this split actually eliminates.
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
    from ruvector._native import GraphDB as GraphDB

__all__ = [
    "RabitqIndex",
    "HnswIndex",
    "RuVectorError",
    "__version__",
    "Collection",
    "CollectionError",
    "CollectionStats",
    "SearchHit",
    "GraphDB",
    "GnnLayer",
    "AttentionReranker",
]

# Names that must NOT be dynamically resolved against _native/collection even
# though they're public-looking — reserved for genuine ruvector submodules
# (ruvector.cli, ruvector.mcp_server, ruvector.integrations.*) so that e.g.
# `import ruvector.cli` isn't shadowed by a same-named attribute probe.
_RESERVED_SUBMODULES = frozenset({"cli", "mcp_server", "integrations", "collection", "_native"})


def __getattr__(name: str) -> Any:
    # `__version__` is the one legitimate dunder re-export (mirrors
    # Cargo.toml's package version via _native); every other leading-
    # underscore name is treated as private and rejected outright rather
    # than risk exposing an internal helper through dynamic dispatch.
    if (name.startswith("_") and name != "__version__") or name in _RESERVED_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from ruvector import _native

    if hasattr(_native, name):
        return getattr(_native, name)

    from ruvector import collection as _collection_mod

    if hasattr(_collection_mod, name):
        return getattr(_collection_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
