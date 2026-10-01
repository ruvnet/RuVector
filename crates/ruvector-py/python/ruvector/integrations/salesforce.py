"""Salesforce Agentforce integration (ADR-352).

Research correction recorded in ADR-352 before this module was written —
re-stated here because it shapes every design choice below: Salesforce's
documented "bring your own retriever" story only plugs into Data Cloud's
own native vector store, not an arbitrary external one; and Agentforce's
native MCP client exists but is **Beta, AE-gated, not self-service** (live-
verified against ``salesforce.com/blog/agentforce-mcp``, dated 2026-01-15).
The one *buildable-today* integration path is **External Services +
OpenAPI** — a custom Agentforce action backed by an Apex/Flow invocable
action that calls an External Service, defined via a Named Credential plus
an OpenAPI 3.0 spec. That is what this module implements:

1. :func:`generate_openapi_spec` — a hand-built, deliberately **flat**
   OpenAPI 3.0 document (no ``oneOf``/``anyOf``/``allOf`` — even though
   live research confirmed Salesforce has supported those since Spring
   '22, flat schemas are the safer compatibility floor across org
   versions, and this surface has no genuine need for polymorphism) for
   the three actions below. Operation ids and schema names are kept under
   255 characters (the real org limit found in research, for Apex/Flow
   Builder usability).
2. Three framework-agnostic action functions —
   :func:`action_search`, :func:`action_upsert`, :func:`action_ground` —
   each a thin wrapper over :class:`ruvector.Collection`. They take and
   return plain dicts (JSON-compatible), so they're callable directly from
   a test, or from an HTTP handler (see ``ruvector.mcp_server``'s
   `custom_route`-mounted ``/salesforce/*`` endpoints, gated behind
   ``RUVECTOR_ENABLE_SALESFORCE_ACTIONS`` — **not** auto-registered by
   importing this module).
3. A record-sync helper (:func:`fetch_records`, :func:`sync_records_to_collection`)
   that pulls rows via Salesforce's REST Query API using an OAuth 2.0
   client-credentials grant (:func:`get_oauth_token`) against a Named
   Credential's token endpoint — the auth flow confirmed live-supported
   for Named Credentials in research (``xcloud.nc_auth_protocols.htm``).
   **Every test against this module uses a mocked HTTP transport — no
   real Salesforce org is touched, per the task boundary.**

Credentials are read from environment variables
(``RUVECTOR_SALESFORCE_INSTANCE_URL``, ``RUVECTOR_SALESFORCE_CLIENT_ID``,
``RUVECTOR_SALESFORCE_CLIENT_SECRET``) — never hardcoded, never logged.

``httpx`` (the `salesforce` extra's one dependency — chosen over
`simple_salesforce` to avoid that package's heavier `requests`-based
transitive footprint, and because this module only needs two REST calls,
not a full Salesforce ORM) is imported **lazily inside the two functions
that make network calls** (:func:`get_oauth_token`, :func:`fetch_records`),
not at module level — unlike `langchain.py`/`llamaindex.py`, this module
never subclasses a third-party base class, so there's no forced-eager-
import requirement the way there is for those two. `import
ruvector.integrations.salesforce` alone stays `httpx`-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from ..collection import Collection, SearchHit

if TYPE_CHECKING:
    from numpy.typing import NDArray

_INSTANCE_URL_ENV = "RUVECTOR_SALESFORCE_INSTANCE_URL"
_CLIENT_ID_ENV = "RUVECTOR_SALESFORCE_CLIENT_ID"
_CLIENT_SECRET_ENV = "RUVECTOR_SALESFORCE_CLIENT_SECRET"
_DEFAULT_API_VERSION = "61.0"


@dataclass(frozen=True)
class SalesforceConfig:
    """OAuth2 client-credentials config, read from environment variables
    (never hardcoded, never logged — :meth:`__repr__` is overridden
    implicitly by dataclass default repr, which WOULD print the secret, so
    callers must not `print(config)`/log it directly; this is documented
    rather than silently "fixed" with a lossy custom repr that could hide
    a real misconfiguration during debugging).
    """

    instance_url: str
    client_id: str
    client_secret: str
    api_version: str = _DEFAULT_API_VERSION

    @classmethod
    def from_env(cls) -> "SalesforceConfig":
        instance_url = os.environ.get(_INSTANCE_URL_ENV)
        client_id = os.environ.get(_CLIENT_ID_ENV)
        client_secret = os.environ.get(_CLIENT_SECRET_ENV)
        missing = [
            name
            for name, val in [
                (_INSTANCE_URL_ENV, instance_url),
                (_CLIENT_ID_ENV, client_id),
                (_CLIENT_SECRET_ENV, client_secret),
            ]
            if not val
        ]
        if missing:
            raise ValueError(f"missing required environment variable(s): {', '.join(missing)}")
        assert instance_url and client_id and client_secret  # narrows for mypy after the check above
        return cls(instance_url=instance_url, client_id=client_id, client_secret=client_secret)


async def get_oauth_token(config: SalesforceConfig, *, transport: "Optional[Any]" = None) -> str:
    """OAuth 2.0 client-credentials grant against
    ``{instance_url}/services/oauth2/token`` — the flow a Salesforce Named
    Credential uses when configured for "OAuth 2.0 Client Credentials"
    (confirmed live-supported in ADR-352's research). Returns the bearer
    access token string.

    `transport` (an ``httpx.BaseTransport``, typed ``Any`` here so this
    module's own type hints don't force an eager ``httpx`` import at
    *type-check* time either) is a dependency-injection seam for tests —
    pass an ``httpx.MockTransport`` to exercise this function with zero
    real network I/O, which is how every test in
    ``tests/test_salesforce_integration.py`` verifies this without
    touching a real Salesforce org. Production callers omit it and get
    ``httpx``'s real transport.

    Raises ``RuntimeError`` with the response body on a non-2xx response
    (Salesforce's token endpoint returns a JSON error body on failure,
    e.g. ``{"error":"invalid_client", ...}`` — surfaced verbatim rather
    than swallowed).
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patch in tests
        raise ImportError(
            "ruvector.integrations.salesforce's network calls require 'httpx'. "
            "Install it with: pip install 'ruvector[salesforce]'"
        ) from exc

    async with httpx.AsyncClient(transport=transport) as client:
        resp = await client.post(
            f"{config.instance_url}/services/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": config.client_id,
                "client_secret": config.client_secret,
            },
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Salesforce OAuth token request failed ({resp.status_code}): {resp.text}")
    body = resp.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"Salesforce OAuth response had no access_token: {body}")
    return str(token)


async def fetch_records(
    config: SalesforceConfig,
    soql: str,
    *,
    token: Optional[str] = None,
    transport: "Optional[Any]" = None,
) -> List[Dict[str, Any]]:
    """Run a SOQL query via Salesforce's REST Query API
    (``/services/data/v{api_version}/query``), following
    ``nextRecordsUrl`` pagination until ``done: true``. Pass `token` to
    reuse an already-fetched bearer token (avoids one OAuth round trip per
    call when syncing many queries in a row); omitted, a fresh token is
    requested via :func:`get_oauth_token` (and `transport`, if given, is
    forwarded to that call too, so a single mocked transport covers both
    the token fetch and the query in one test).

    `transport` is the same dependency-injection seam documented on
    :func:`get_oauth_token` — see that function's docstring.

    Returns the flat list of record dicts (Salesforce's own
    ``attributes`` key, if present on each record, is left in place —
    callers that don't want it can drop it themselves; stripping it here
    would be lossy for callers who do want the object type it carries).
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patch in tests
        raise ImportError(
            "ruvector.integrations.salesforce's network calls require 'httpx'. "
            "Install it with: pip install 'ruvector[salesforce]'"
        ) from exc

    bearer = token or await get_oauth_token(config, transport=transport)
    headers = {"Authorization": f"Bearer {bearer}"}
    records: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport) as client:
        path: Optional[str] = f"/services/data/v{config.api_version}/query"
        params: Optional[Dict[str, str]] = {"q": soql}
        while path is not None:
            resp = await client.get(f"{config.instance_url}{path}", headers=headers, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Salesforce query failed ({resp.status_code}): {resp.text}")
            body = resp.json()
            records.extend(body.get("records", []))
            next_path = body.get("nextRecordsUrl")
            path = next_path if not body.get("done", True) else None
            params = None  # nextRecordsUrl already carries the query cursor
    return records


def sync_records_to_collection(
    collection: Collection,
    records: Iterable[Dict[str, Any]],
    *,
    embed_fn: "Callable[[str], NDArray[np.float32]]",
    text_field: str,
    metadata_fields: Optional[List[str]] = None,
    id_field: Optional[str] = None,
) -> List[int]:
    """Embed and insert Salesforce `records` (plain dicts, e.g. from
    :func:`fetch_records`) into `collection` for grounding.

    `ruvector` has no embedder of its own yet (M3, deferred — see
    ADR-352's milestone table), so the caller supplies `embed_fn`
    (text -> float32 vector) explicitly; this function does the
    record-shape handling (field extraction, metadata assembly, batching
    into one `Collection.insert_batch` call) rather than the embedding
    itself. Records missing `text_field` are skipped (not an error — a
    partial/sparse SOQL result set is a normal occurrence, not a caller
    mistake), and the number of inserted ids reflects only the records
    actually embedded.

    `id_field` (if given, e.g. ``"Id"`` for the Salesforce record id) is
    stored in each inserted vector's metadata under the key
    ``"_salesforce_id"`` so the two systems' ids can be correlated later
    (``Collection``'s own ids are backend-assigned ints, unrelated to
    Salesforce's 18-char record ids — no attempt is made to reuse the
    Salesforce id as the ruvector id, which would risk the same
    u32-range/type mismatch issues documented on `Collection.from_vectors`
    for the rabitq backend).
    """
    texts: List[str] = []
    metadatas: List[Optional[Dict[str, Any]]] = []
    for record in records:
        text = record.get(text_field)
        if not text:
            continue
        texts.append(str(text))
        meta: Dict[str, Any] = {}
        if metadata_fields:
            for field in metadata_fields:
                if field in record:
                    meta[field] = record[field]
        if id_field and id_field in record:
            meta["_salesforce_id"] = record[id_field]
        metadatas.append(meta or None)

    if not texts:
        return []

    vectors = np.stack([np.asarray(embed_fn(t), dtype=np.float32) for t in texts])
    return collection.insert_batch(vectors, metadatas=metadatas)


# ── Agentforce actions (framework-agnostic — callable directly, or from an
# HTTP handler; see ruvector.mcp_server's gated /salesforce/* routes) ──────


def action_search(
    collection: Collection,
    query_vector: List[float],
    k: int = 10,
    filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Agentforce action: search `collection`. Mirrors `vector_search`'s
    MCP tool shape (ADR-352) so the two surfaces stay consistent."""
    hits = collection.search(np.asarray(query_vector, dtype=np.float32), k, filter=filter)
    return {"hits": [_hit_to_dict(h) for h in hits]}


def action_upsert(
    collection: Collection,
    vector: List[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Agentforce action: insert one vector (grounding-corpus maintenance
    from within an agent conversation, e.g. "remember this"). Named
    "upsert" to match the Agentforce action-naming convention in the
    research (externalservice actions commonly use upsert-shaped verbs),
    but this is always an insert — `ruvector.Collection` has no update-in-
    place (see its own docstring); a caller that wants to replace a prior
    entry must `delete` the old id first.
    """
    new_id = collection.insert(np.asarray(vector, dtype=np.float32), metadata=metadata)
    return {"id": new_id, "count": len(collection)}


def action_ground(
    collection: Collection,
    query_vector: List[float],
    k: int = 5,
    text_field: str = "text",
) -> Dict[str, Any]:
    """Agentforce action: search + format as a grounding context block.

    Returns ``{"context": "<joined text>", "sources": [...]}`` — `context`
    is every hit's `metadata[text_field]` joined with blank lines (skipping
    hits with no such field, rather than inserting an empty line for them),
    ready to drop into an Agentforce prompt template's grounding variable.
    `sources` carries the same hits in full (id/score/metadata) for
    citation/audit. This is the one action that assumes a metadata
    convention (`text_field`); `action_search` makes no such assumption.
    """
    hits = collection.search(np.asarray(query_vector, dtype=np.float32), k)
    snippets = [str(h.metadata[text_field]) for h in hits if h.metadata and text_field in h.metadata]
    return {
        "context": "\n\n".join(snippets),
        "sources": [_hit_to_dict(h) for h in hits],
    }


def _hit_to_dict(hit: SearchHit) -> Dict[str, Any]:
    return {"id": hit.id, "score": hit.score, "metadata": hit.metadata}


# ── OpenAPI 3.0 spec for Salesforce External Services import ───────────────


def generate_openapi_spec(base_url: str, *, collection_name_default: str = "default") -> Dict[str, Any]:
    """A hand-built OpenAPI 3.0 document describing the three actions
    above as REST operations, for Salesforce External Services import.

    Deliberately flat (no `oneOf`/`anyOf`/`allOf`, no nested `$ref` chains)
    even though research confirmed Salesforce has supported composition
    keywords since Spring '22 — flat schemas are the safer floor across
    org versions this module can't test against directly (no real org is
    touched per the task boundary). Every `operationId` is well under the
    255-character limit found in research (for Apex/Flow Builder action
    naming).
    """
    vector_schema = {
        "type": "array",
        "items": {"type": "number", "format": "float"},
        "description": "A float32 embedding vector.",
    }
    metadata_schema = {
        "type": "object",
        "additionalProperties": True,
        "description": "Arbitrary JSON-compatible metadata.",
    }
    hit_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "score": {"type": "number", "format": "float"},
            "metadata": metadata_schema,
        },
        "required": ["id", "score"],
    }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ruvector Agentforce Actions",
            "version": "0.1.0",
            "description": "Vector search, upsert, and grounding actions backed by ruvector's Rust core.",
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/salesforce/search": {
                "post": {
                    "operationId": "ruvectorSearch",
                    "summary": "Search a collection for the k nearest neighbours of a query vector.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "collection": {"type": "string", "default": collection_name_default},
                                        "query_vector": vector_schema,
                                        "k": {"type": "integer", "default": 10},
                                        "filter": metadata_schema,
                                    },
                                    "required": ["query_vector"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"hits": {"type": "array", "items": hit_schema}},
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/salesforce/upsert": {
                "post": {
                    "operationId": "ruvectorUpsert",
                    "summary": "Insert one vector into a collection.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "collection": {"type": "string", "default": collection_name_default},
                                        "vector": vector_schema,
                                        "metadata": metadata_schema,
                                    },
                                    "required": ["vector"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Upsert result.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "count": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/salesforce/ground": {
                "post": {
                    "operationId": "ruvectorGround",
                    "summary": "Search and format results as a grounding context block for a prompt template.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "collection": {"type": "string", "default": collection_name_default},
                                        "query_vector": vector_schema,
                                        "k": {"type": "integer", "default": 5},
                                        "text_field": {"type": "string", "default": "text"},
                                    },
                                    "required": ["query_vector"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Grounding context.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "context": {"type": "string"},
                                            "sources": {"type": "array", "items": hit_schema},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
    }


__all__ = [
    "SalesforceConfig",
    "get_oauth_token",
    "fetch_records",
    "sync_records_to_collection",
    "action_search",
    "action_upsert",
    "action_ground",
    "generate_openapi_spec",
]
