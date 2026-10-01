"""ruvector MCP server (ADR-352 M1.5).

Built on the official ``mcp`` Python SDK (``mcp.server.mcpserver.MCPServer``,
mcp>=2.0 — note the v1->v2 rename from ``FastMCP``). Exposes the
``Collection`` surface as tools, plus one ChatGPT-Apps-SDK widget tool
(``vector_explore``) whose ``_meta`` matches the shape live-verified in
ADR-352 against ``web-based-chatgpt-mcp-starter.ruv.chatgpt.site/api/mcp``.

Security boundary (ADR-352 "Security"): tool arguments that name a
collection are **names**, not paths — ``_safe_path`` resolves a name to a
file under ``RUVECTOR_MCP_DATA_DIR`` (default ``~/.ruvector/collections``)
and rejects anything that would escape that root (directory traversal,
absolute paths, path separators). This is a different, stricter rule than
the CLI's `--path` (the CLI runs with the caller's own filesystem
authority; an MCP tool call may come from a remote, less-trusted client).
Resource reads (the ``ui://`` widget) get the SDK's own
``ResourceSecurity(reject_path_traversal=True, reject_absolute_paths=True,
reject_null_bytes=True)`` for free — that guard is on by default in
``MCPServer.__init__``, not something this module has to add.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from ruvector.collection import Collection

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# mcp>=2's ToolAnnotations are pydantic fields (read_only_hint, not
# readOnlyHint — the camelCase on the wire, visible in ADR-352's live probe
# of the starter site, is a pydantic alias). Four reusable instances cover
# every tool below; building them once keeps `mypy --strict` happy (a bare
# dict literal doesn't type-check against ToolAnnotations even though
# pydantic would coerce it at runtime).
_RO = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
_DELETE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True)


def _data_root() -> Path:
    root = Path(os.environ.get("RUVECTOR_MCP_DATA_DIR", "~/.ruvector/collections")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_path(name: str) -> Path:
    """Resolve a collection *name* (never a path) to a file under the data
    root. Rejects anything but ``[A-Za-z0-9_-]`` — no ``/``, no ``..``, no
    null bytes, no absolute paths — so there is no traversal surface to
    exploit regardless of what a remote MCP client sends.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid collection name {name!r}: must match {_NAME_RE.pattern} "
            "(letters, digits, '-', '_' only — no path separators)"
        )
    root = _data_root()
    candidate = (root / f"{name}.rbpx").resolve()
    candidate.relative_to(root)  # raises ValueError if this ever stops holding
    return candidate


# In-process cache so repeated tool calls against the same collection within
# one server lifetime don't pay a disk round-trip every time. Mutating
# operations (insert/delete) always re-save to disk immediately — this is a
# read cache, not a write-behind cache, so a crash never loses an
# acknowledged write.
_cache: Dict[str, "Any"] = {}


def _load(name: str) -> "Collection":
    from ruvector.collection import Collection, CollectionError

    path = _safe_path(name)
    if name in _cache:
        return _cache[name]  # type: ignore[no-any-return]
    if not Collection.meta_path(path).exists():
        raise CollectionError(f"no such collection: {name!r} (create it first with vector_create_collection)")
    coll = Collection.load(path)
    _cache[name] = coll
    return coll


def _save(name: str, coll: "Collection") -> None:
    coll.save(_safe_path(name))
    _cache[name] = coll


server = MCPServer(
    name="ruvector",
    title="RuVector",
    version="0.1.0",
    instructions=(
        "ultra-low-latency vector search backed by a Rust RaBitQ core. "
        "Create a collection, insert vectors with optional metadata, search "
        "with optional exact-match filters, and explore results visually "
        "via vector_explore."
    ),
)


@server.tool(
    name="vector_create_collection",
    description="Create a new empty vector collection.",
    annotations=_WRITE,
)
def vector_create_collection(name: str, dim: int, rerank_factor: int = 20, seed: int = 42) -> Dict[str, Any]:
    from ruvector.collection import Collection

    path = _safe_path(name)
    if Collection.meta_path(path).exists():
        raise ValueError(f"collection {name!r} already exists")
    coll = Collection.create(dim=dim, rerank_factor=rerank_factor, seed=seed)
    _save(name, coll)
    return {"name": name, "dim": dim, "rerank_factor": rerank_factor}


@server.tool(
    name="vector_insert",
    description="Insert one vector (with optional metadata) into a collection.",
    annotations=_WRITE,
)
def vector_insert(name: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import numpy as np

    coll = _load(name)
    new_id = coll.insert(np.asarray(vector, dtype=np.float32), metadata=metadata)
    _save(name, coll)
    return {"id": new_id, "count": len(coll)}


@server.tool(
    name="vector_insert_batch",
    description="Insert many vectors (with optional per-row metadata) into a collection.",
    annotations=_WRITE,
)
def vector_insert_batch(
    name: str, vectors: List[List[float]], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    import numpy as np

    coll = _load(name)
    ids = coll.insert_batch(np.asarray(vectors, dtype=np.float32), metadatas=metadatas)
    _save(name, coll)
    return {"ids": ids, "count": len(coll)}


@server.tool(
    name="vector_search",
    description="Search a collection for the k nearest neighbours of a query vector, with an optional exact-match metadata filter.",
    annotations=_RO,
)
def vector_search(
    name: str,
    query: List[float],
    k: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    rerank_factor: Optional[int] = None,
) -> Dict[str, Any]:
    import numpy as np

    coll = _load(name)
    hits = coll.search(np.asarray(query, dtype=np.float32), k, filter=filter, rerank_factor=rerank_factor)
    return {"hits": [{"id": h.id, "score": h.score, "metadata": h.metadata} for h in hits]}


@server.tool(
    name="vector_delete",
    description="Soft-delete one id from a collection; optionally vacuum immediately to reclaim space.",
    annotations=_DELETE,
)
def vector_delete(name: str, id: int, vacuum: bool = False) -> Dict[str, Any]:
    coll = _load(name)
    coll.delete(id)
    dropped = coll.vacuum() if vacuum else 0
    _save(name, coll)
    return {"count": len(coll), "vacuumed": dropped}


@server.tool(
    name="vector_stats",
    description="Get stats (count, dim, rerank_factor, memory_bytes, tombstoned) for a collection.",
    annotations=_RO,
)
def vector_stats(name: str) -> Dict[str, Any]:
    from dataclasses import asdict

    return asdict(_load(name).stats())


@server.tool(
    name="vector_list_collections",
    description="List every collection under the server's data root.",
    annotations=_RO,
)
def vector_list_collections() -> Dict[str, Any]:
    root = _data_root()
    names = sorted(p.stem for p in root.glob("*.rbpx"))
    return {"collections": names, "data_root": str(root)}


# ── ui:// widget (ChatGPT Apps SDK convention, live-verified in ADR-352) ────

_WIDGET_URI = "ui://ruvector/explore.html"

_WIDGET_META = {
    "ui": {"resourceUri": _WIDGET_URI},
    "openai/outputTemplate": _WIDGET_URI,
    "openai/widgetAccessible": True,
    "ui/resourceUri": _WIDGET_URI,
}

_WIDGET_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>ruvector explore</title>
<style>
  body { font: 13px -apple-system, system-ui, sans-serif; margin: 0; padding: 12px; color: #1a1a1a; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #e5e5e5; }
  th { color: #666; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .bar { height: 6px; background: #4c6ef5; border-radius: 3px; }
  .barwrap { background: #eef0ff; border-radius: 3px; width: 120px; }
  h1 { font-size: 14px; margin: 0 0 8px; }
</style>
</head>
<body>
<h1 id="title">ruvector — collection explorer</h1>
<table>
  <thead><tr><th>id</th><th>score</th><th></th><th>metadata</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
  // The Apps SDK runtime calls window.openai.toolOutput (or postMessage,
  // depending on host) with this tool's structured result. We read it
  // defensively since the widget may also be opened standalone for review.
  function render(data) {
    const hits = (data && data.hits) || [];
    document.getElementById('title').textContent =
      'ruvector — ' + hits.length + ' result(s)';
    const maxScore = Math.max(1e-9, ...hits.map(h => h.score));
    const rows = hits.map(h => {
      const pct = Math.max(2, 100 - (h.score / maxScore) * 100);
      return '<tr><td>' + h.id + '</td><td>' + h.score.toFixed(4) + '</td>' +
        '<td><div class="barwrap"><div class="bar" style="width:' + pct + '%"></div></div></td>' +
        '<td>' + (h.metadata ? JSON.stringify(h.metadata) : '') + '</td></tr>';
    }).join('');
    document.getElementById('rows').innerHTML = rows;
  }
  try {
    if (window.openai && window.openai.toolOutput) {
      render(window.openai.toolOutput);
    }
  } catch (e) { /* standalone preview — no host runtime present */ }
</script>
</body>
</html>
"""


@server.resource(
    _WIDGET_URI,
    name="ruvector explore widget",
    description="HTML widget rendering vector_search results as a scored table.",
    mime_type="text/html",
)
def explore_widget() -> str:
    return _WIDGET_HTML


@server.tool(
    name="vector_explore",
    title="Explore search results",
    description="Run a search and render it in the ruvector explore widget.",
    meta=_WIDGET_META,
    annotations=_RO,
)
def vector_explore(name: str, query: List[float], k: int = 10) -> Dict[str, Any]:
    return vector_search(name=name, query=query, k=k)


def run_stdio() -> None:
    server.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8420) -> None:
    server.run(transport="streamable-http", host=host, port=port)


__all__ = ["server", "run_stdio", "run_http"]
