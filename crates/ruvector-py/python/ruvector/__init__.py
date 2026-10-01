"""ruvector — vector similarity search via RaBitQ 1-bit quantization.

M1 surface: ``RabitqIndex`` plus the ``RuVectorError`` base exception.
M1.5 (ADR-352) adds ``Collection``: ids, metadata, filtering, soft delete,
and a persistence sidecar over the same M1 index. See
``docs/sdk/04-milestones.md`` for what M2/M3/M4 add (RuLake, Embedder,
A2aClient) and ``docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`` for the
CLI/MCP/Collection scope this module now carries.
"""

from ruvector._native import RabitqIndex, RuVectorError, __version__
from ruvector.collection import Collection, CollectionError, CollectionStats, SearchHit

__all__ = [
    "RabitqIndex",
    "RuVectorError",
    "__version__",
    "Collection",
    "CollectionError",
    "CollectionStats",
    "SearchHit",
]
