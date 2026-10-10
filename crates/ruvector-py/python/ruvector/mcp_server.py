"""ruvector MCP server (ADR-352).

Built on the official ``mcp`` Python SDK (``mcp.server.mcpserver.MCPServer``,
mcp>=2.0 — note the v1->v2 rename from ``FastMCP``). Exposes the
``Collection`` surface as tools, plus one ChatGPT-Apps-SDK widget tool
(``vector_explore``) whose ``_meta`` matches the shape live-verified in
ADR-352 against ``web-based-chatgpt-mcp-starter.ruv.chatgpt.site/api/mcp``.

Security boundary #1 (ADR-352 "Security"): tool arguments that name a
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

Security boundary #2 — bearer auth on the ``--http`` transport (closes the
gap ADR-352 originally left open). **Policy, stated explicitly rather than
left implicit:**

- If ``RUVECTOR_MCP_TOKEN`` is set, every tool call over ``--http`` must
  carry ``Authorization: Bearer <token>`` matching it (constant-time
  compared via :func:`hmac.compare_digest`), enforced by the SDK's own
  ``token_verifier``/``AuthSettings`` machinery (not a hand-rolled ASGI
  middleware — see ``StaticTokenVerifier`` below). A request with no token,
  or the wrong one, is rejected by the SDK before it ever reaches a tool
  handler.
- The one token this design supports carries both ``read`` and ``write``
  scopes (a single shared secret for a single-tenant deployment — this
  slice does not implement separate reader/writer secrets). Every
  *mutating* tool (create/insert/insert_batch/delete) additionally calls
  :func:`_require_write_scope`, which checks the authenticated token's
  scopes via ``get_access_token()``. This is deliberate defense-in-depth,
  not redundant: if ``RUVECTOR_MCP_TOKEN`` is unset (no auth configured at
  all — e.g. local stdio use, or ``--http`` run without a token on
  purpose), ``get_access_token()`` returns ``None`` and the guard is a
  no-op, so local/dev use is unaffected either way.
- An **empty or whitespace-only** ``RUVECTOR_MCP_TOKEN`` counts as *unset*
  (:func:`_configured_token`), never as "the token is the empty string":
  a blank value must not silently turn auth off while looking configured.
- If no token is configured and ``--http`` binds a **loopback** address
  (``127.0.0.0/8``, ``::1``, ``localhost``), :func:`run_http` prints one
  prominent startup warning and serves. If no token is configured and the
  bind host is **not** loopback (``0.0.0.0``, a LAN/tailnet address, a
  hostname), :func:`run_http` raises :class:`UnsafeBindError` before
  binding — the CLI turns that into a non-zero exit. There is no override
  flag: set a token, or bind loopback.
- ``ruvector serve --read-only`` calls :func:`apply_read_only`, which
  unregisters every mutating tool (create/insert/insert_batch/delete) so
  they are not even listed, let alone callable.
- **`MCPServer.custom_route`-mounted endpoints do NOT get this protection**
  — the SDK's own docstring for that decorator says so explicitly
  ("Routes using this decorator will not require authorization"). This
  module does not currently mount any `custom_route` endpoints, but if a
  future Salesforce Agentforce action handler is added via that mechanism,
  it MUST call its own manual check (e.g. reusing
  ``StaticTokenVerifier.verify_token`` directly against the incoming
  request's `Authorization` header) rather than assume the MCP-level auth
  config covers it — confirmed by reading the SDK source, not assumed.
- ``stdio`` transport (the default) has no HTTP layer at all, so none of
  this applies there; `auth=`/`token_verifier=` are stored on the server
  object unconditionally but only consulted when the Starlette ASGI app is
  built for `--http` — passing them does not change stdio behaviour.
"""

from __future__ import annotations

import functools
import hmac
import ipaddress
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl

from ruvector._native import RuVectorError
from ruvector.collection import MAX_DIM, CollectionError

if TYPE_CHECKING:
    from ruvector.collection import Collection

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Per-request resource caps for tool arguments (the dimension / k / rerank
# bounds live in ``ruvector.collection`` and are enforced there). These
# limits are checked on the raw argument lists *before* any numpy
# conversion so an oversized call is rejected cheaply. They bound the work
# one call can demand, not the HTTP body size itself (the SDK parses the
# JSON body before a tool runs; cap that at a reverse proxy if exposed).
MAX_BATCH_ROWS = 10_000
MAX_BATCH_ELEMENTS = 2_000_000  # rows * dim floats per insert_batch call (~8 MB as float32)
MAX_METADATA_BYTES = 65_536  # serialized JSON, per row
# Total vectors a client may write into ONE collection (a cumulative cap, unlike
# the per-call caps above). Every write re-saves the whole collection, so an
# unbounded collection also makes each later write slower. Override with the
# RUVECTOR_MCP_MAX_VECTORS environment variable or `ruvector serve --max-vectors`.
DEFAULT_MAX_VECTORS = 1_000_000
_MAX_VECTORS_ENV_VAR = "RUVECTOR_MCP_MAX_VECTORS"
_max_vectors_override: Optional[int] = None

_TOKEN_ENV_VAR = "RUVECTOR_MCP_TOKEN"
# Self-referential placeholder — this design does not do real OAuth
# discovery or RFC 8707 resource-indicator validation (our tokens carry no
# meaningful `aud`/resource claim), so these just need to be syntactically
# valid AnyHttpUrl values for AuthSettings' schema; `validate_token_resource`
# is deliberately left at its default (`None`/falsy) so the SDK never tries
# to enforce a resource match against them.
_SELF_ISSUER_URL = "http://ruvector.local/"


class StaticTokenVerifier(TokenVerifier):
    """Single-shared-secret ``TokenVerifier``: compares the bearer token
    against ``RUVECTOR_MCP_TOKEN`` with :func:`hmac.compare_digest`
    (constant-time, so response timing can't leak how many leading bytes
    matched). The one valid token carries both ``read`` and ``write``
    scopes — see this module's docstring for why that's the deliberate
    scope model for this slice, not an oversight.
    """

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if not hmac.compare_digest(token, self._expected):
            return None
        return AccessToken(
            token=token,
            client_id="ruvector-static-token",
            scopes=["read", "write"],
            resource=_SELF_ISSUER_URL,
        )


def _require_write_scope() -> None:
    """Defense-in-depth guard called at the top of every mutating tool.

    A no-op when no auth is configured at all (``get_access_token()`` is
    ``None`` on stdio, and on ``--http`` with no ``RUVECTOR_MCP_TOKEN``
    set) — see this module's docstring for why that's the intended
    behaviour, not a bypass. When auth IS configured, the SDK's bearer
    middleware already rejected an unauthenticated or wrong-token request
    before this tool body ever ran; this check instead guards against a
    token that authenticated successfully but lacks the ``write`` scope
    (relevant once a non-``StaticTokenVerifier`` is ever swapped in with a
    real reader/writer scope split).
    """
    token = get_access_token()
    if token is None:
        return
    if "write" not in token.scopes:
        raise ValueError("this token does not have the 'write' scope required for this operation")


def _configured_token() -> Optional[str]:
    """The bearer token from ``RUVECTOR_MCP_TOKEN``, or ``None`` when unset,
    empty or whitespace-only (stripped: HTTP header parsing strips it too,
    so a padded value could otherwise never match)."""
    raw = os.environ.get(_TOKEN_ENV_VAR)
    if raw is None:
        return None
    token = raw.strip()
    return token or None


def _build_auth() -> "tuple[Optional[StaticTokenVerifier], Optional[AuthSettings]]":
    expected = _configured_token()
    if expected is None:
        return None, None
    verifier = StaticTokenVerifier(expected)
    self_url = AnyHttpUrl(_SELF_ISSUER_URL)
    settings = AuthSettings(
        issuer_url=self_url,
        resource_server_url=self_url,
        # Our StaticTokenVerifier always sets AccessToken.resource to the
        # same _SELF_ISSUER_URL, so this is safe to enforce (not just
        # silenced) — it makes the SDK actually check the token's resource
        # claim against resource_server_url instead of accepting any token
        # regardless of audience. Explicit per the SDK's own deprecation
        # warning when this is left unset.
        validate_token_resource=True,
    )
    return verifier, settings

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

# `ruvector serve --http` dispatches concurrent tool calls to worker
# threads (the same fact that made RabitqIndex's old `unsendable` pyclass
# panic — see ADR-352). Without this lock, two concurrent `vector_insert`
# calls on the same collection can both read the same `Collection._next_id`
# and both `add()` with the same id, or race on `_metadata`/`_tombstones`
# (plain dict/set, no internal locking). One process-wide `RLock` (not a
# plain `Lock`: `vector_explore` calls `vector_search` from inside the same
# thread, which must be able to re-acquire) around every tool body that
# touches `_cache` or a `Collection` serializes all of them — correct over
# concurrent, which is the right tradeoff for a single vector index, not a
# throughput-critical service. The stdio transport (the default) is already
# one request at a time, so this only changes behavior under `--http`.
_lock = threading.RLock()


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


def parse_max_vectors(raw: Union[str, int], source: str) -> int:
    """Validate a vectors-per-collection cap; raises ``ValueError`` if not a positive int."""
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{source} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{source} must be a positive integer, got {raw!r}")
    return value


def set_max_vectors(value: Optional[int]) -> None:
    """Set (or with ``None`` clear) the cap given on the command line; it wins over the environment."""
    global _max_vectors_override
    _max_vectors_override = None if value is None else parse_max_vectors(value, "max vectors")


def max_vectors() -> int:
    """Effective per-collection cap: ``--max-vectors``, else the env var, else the default.

    An empty env var counts as unset.
    """
    if _max_vectors_override is not None:
        return _max_vectors_override
    raw = os.environ.get(_MAX_VECTORS_ENV_VAR, "").strip()
    return parse_max_vectors(raw, _MAX_VECTORS_ENV_VAR) if raw else DEFAULT_MAX_VECTORS


def _check_capacity(coll: "Collection", incoming: int) -> None:
    """Reject a write that would take the collection past the cap.

    Counts tombstoned rows too: they still occupy the index and the saved
    files until ``vector_delete(..., vacuum=True)`` runs.
    """
    cap = max_vectors()
    stats = coll.stats()
    used = stats.count + stats.tombstoned
    if used + incoming > cap:
        raise ValueError(
            f"collection is at its vector limit: it holds {used} vectors"
            f"{f' ({stats.tombstoned} deleted, not yet vacuumed)' if stats.tombstoned else ''}, "
            f"adding {incoming} would exceed the maximum of {cap} per collection "
            f"(set {_MAX_VECTORS_ENV_VAR} or `ruvector serve --max-vectors` to change it)"
        )


def _check_vector_len(vec: "List[float]", what: str) -> None:
    if len(vec) > MAX_DIM:
        raise ValueError(f"{what} has {len(vec)} elements; the maximum dimension is {MAX_DIM}")


def _check_metadata(md: "Optional[Dict[str, Any]]") -> None:
    if md is None:
        return
    size = len(json.dumps(md, default=str).encode("utf-8"))
    if size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata is {size} bytes serialized; the maximum is {MAX_METADATA_BYTES}")


def _check_batch(vectors: "List[List[float]]", metadatas: "Optional[List[Optional[Dict[str, Any]]]]") -> None:
    if len(vectors) > MAX_BATCH_ROWS:
        raise ValueError(f"too many rows: {len(vectors)} (maximum {MAX_BATCH_ROWS} per call)")
    total = 0
    for row in vectors:
        _check_vector_len(row, "a vector row")
        total += len(row)
    if total > MAX_BATCH_ELEMENTS:
        raise ValueError(f"batch has {total} elements in total (maximum {MAX_BATCH_ELEMENTS} per call)")
    if metadatas is not None:
        if len(metadatas) != len(vectors):
            raise ValueError(f"metadatas length ({len(metadatas)}) must match the number of rows ({len(vectors)})")
        for md in metadatas:
            _check_metadata(md)


_token_verifier, _auth_settings = _build_auth()

server = MCPServer(
    name="ruvector",
    title="RuVector",
    version="0.1.2",
    instructions=(
        "ultra-low-latency vector search backed by a Rust RaBitQ core. "
        "Create a collection, insert vectors with optional metadata, search "
        "with optional exact-match filters, and explore results visually "
        "via vector_explore."
    ),
    token_verifier=_token_verifier,
    auth=_auth_settings,
)


# mcp 2.x hides the message of any exception that is not a ToolError and sends
# the client only "Error executing tool <name>". Our own validation errors
# (bad dimension, unknown collection, rejected name, non-finite input) are safe
# to show and tell an agent how to correct the call, so re-raise them as
# ToolError. Anything else (an OSError carrying a filesystem path, say) stays masked.
_raw_tool = server.tool


def _tool_with_messages(*targs: Any, **tkwargs: Any) -> Any:
    register = _raw_tool(*targs, **tkwargs)

    def wrap(fn: Any) -> Any:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except (ValueError, TypeError, KeyError, CollectionError, RuVectorError) as e:
                raise ToolError(str(e)) from e

        return register(inner)

    return wrap


server.tool = _tool_with_messages  # type: ignore[method-assign]


@server.tool(
    name="vector_create_collection",
    description="Create a new empty vector collection.",
    annotations=_WRITE,
)
def vector_create_collection(
    name: str, dim: int, rerank_factor: int = 20, seed: int = 42, backend: str = "hnsw"
) -> Dict[str, Any]:
    from ruvector.collection import Collection

    _require_write_scope()
    with _lock:
        path = _safe_path(name)
        if Collection.meta_path(path).exists():
            raise ValueError(f"collection {name!r} already exists")
        coll = Collection.create(dim=dim, backend=backend, rerank_factor=rerank_factor, seed=seed)
        _save(name, coll)
        return {"name": name, "dim": dim, "backend": backend, "rerank_factor": rerank_factor}


@server.tool(
    name="vector_insert",
    description="Insert one vector (with optional metadata) into a collection.",
    annotations=_WRITE,
)
def vector_insert(name: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import numpy as np

    _require_write_scope()
    _check_vector_len(vector, "vector")
    _check_metadata(metadata)
    with _lock:
        coll = _load(name)
        _check_capacity(coll, 1)
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

    _require_write_scope()
    _check_batch(vectors, metadatas)
    with _lock:
        coll = _load(name)
        _check_capacity(coll, len(vectors))
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

    _check_vector_len(query, "query")
    with _lock:
        coll = _load(name)
        hits = coll.search(np.asarray(query, dtype=np.float32), k, filter=filter, rerank_factor=rerank_factor)
        return {"hits": [{"id": h.id, "score": h.score, "metadata": h.metadata} for h in hits]}


@server.tool(
    name="vector_delete",
    description="Soft-delete one id from a collection; optionally vacuum immediately to reclaim space.",
    annotations=_DELETE,
)
def vector_delete(name: str, id: int, vacuum: bool = False) -> Dict[str, Any]:
    _require_write_scope()
    with _lock:
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

    with _lock:
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
  //
  // SECURITY: every server-supplied value (ids, scores, and above all the
  // user-controlled metadata) reaches the page ONLY through textContent /
  // style properties on elements we create. No HTML-parsing sink is used
  // (no markup-setting property, no write-to-document call), so metadata such as
  // <img src=x onerror=...> renders as inert text.
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = String(text);
    if (cls) e.className = cls;
    return e;
  }
  function render(data) {
    const hits = (data && Array.isArray(data.hits)) ? data.hits : [];
    document.getElementById('title').textContent =
      'ruvector — ' + hits.length + ' result(s)';
    const scores = hits.map(h => Number(h.score) || 0);
    const maxScore = Math.max(1e-9, ...scores);
    const tbody = document.getElementById('rows');
    tbody.textContent = '';
    hits.forEach((h, i) => {
      const score = scores[i];
      const pct = Math.max(2, 100 - (score / maxScore) * 100);
      const tr = el('tr');
      tr.appendChild(el('td', h.id));
      tr.appendChild(el('td', score.toFixed(4)));
      const barCell = el('td');
      const wrap = el('div', undefined, 'barwrap');
      const bar = el('div', undefined, 'bar');
      bar.style.width = pct + '%';
      wrap.appendChild(bar);
      barCell.appendChild(wrap);
      tr.appendChild(barCell);
      tr.appendChild(el('td', h.metadata ? JSON.stringify(h.metadata) : ''));
      tbody.appendChild(tr);
    });
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


_MUTATING_TOOLS = ("vector_create_collection", "vector_insert", "vector_insert_batch", "vector_delete")


def apply_read_only() -> None:
    """Unregister every mutating tool (``ruvector serve --read-only``).

    Removal rather than a runtime flag: a removed tool is neither listed
    nor callable, so there is no code path left that could write.
    Idempotent.
    """
    for tool_name in _MUTATING_TOOLS:
        try:
            server.remove_tool(tool_name)
        except Exception:  # already removed
            pass


class UnsafeBindError(RuntimeError):
    """Refusing to serve HTTP on a non-loopback address with no auth token."""


def _is_loopback_host(host: str) -> bool:
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:  # a hostname other than localhost, or empty (= all interfaces)
        return False


def check_http_exposure(host: str, *, token_configured: bool) -> None:
    """Raise :class:`UnsafeBindError` when ``host`` is reachable beyond this
    machine and no bearer token is configured."""
    if token_configured or _is_loopback_host(host):
        return
    raise UnsafeBindError(
        f"refusing to serve HTTP on non-loopback host {host!r} without authentication: "
        f"set {_TOKEN_ENV_VAR} to a non-empty secret, or bind 127.0.0.1"
    )


def run_stdio() -> None:
    server.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8420) -> None:
    check_http_exposure(host, token_configured=_token_verifier is not None)
    if _token_verifier is None:
        print(
            f"\n{'!' * 70}\n"
            f"WARNING: ruvector serve --http is starting WITHOUT authentication\n"
            f"(loopback only: {host}:{port}). Set the {_TOKEN_ENV_VAR} environment\n"
            f"variable to require a bearer token on every tool call (see\n"
            f"ruvector.mcp_server's module docstring for the exact policy).\n"
            f"Any local process can call every tool, including inserts and deletes.\n"
            f"{'!' * 70}\n",
            file=sys.stderr,
        )
    server.run(transport="streamable-http", host=host, port=port)


__all__ = [
    "server",
    "run_stdio",
    "run_http",
    "StaticTokenVerifier",
    "apply_read_only",
    "UnsafeBindError",
    "set_max_vectors",
    "max_vectors",
]
