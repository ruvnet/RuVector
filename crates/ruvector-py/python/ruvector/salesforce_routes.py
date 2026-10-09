"""Salesforce Agentforce action routes (ADR-352) — mounted on the same
``MCPServer``/port as the MCP tools, via ``MCPServer.custom_route``.

**Not imported by ``ruvector.mcp_server`` automatically.** Importing this
module has the side effect of registering its routes onto
``ruvector.mcp_server.server`` (``custom_route`` is a decorator that
mutates the server object it's called on), so it is only imported when
explicitly enabled — see :func:`maybe_register` and ``cli.py``'s ``serve``
command, which calls it before starting the HTTP transport.

**Critical security fact, verified by reading the SDK source before
writing this file, not assumed**: ``MCPServer.custom_route``'s own
docstring states "Routes using this decorator will not require
authorization" — the `token_verifier`/`AuthSettings` auth wired up in
``mcp_server.py`` for the `/mcp` JSON-RPC endpoint does **not** cover these
routes at all. Every handler below therefore does its own manual bearer
check via :func:`_check_salesforce_auth` (a plain constant-time compare —
see that function's docstring for why it doesn't reuse
``mcp_server.StaticTokenVerifier`` directly) against a *separate* env var
(``RUVECTOR_SALESFORCE_ACTION_TOKEN`` — falls back to
``RUVECTOR_MCP_TOKEN`` if unset, so a single-secret deployment doesn't need
to configure two env vars, but a deployment that wants Salesforce's Named
Credential to carry a different secret than whatever drives MCP tool
calls can do so).

The three action routes operate on **named** collections, loaded via
``mcp_server._load``/``_safe_path`` — the exact same traversal-safe name
resolution the MCP tools use (no separate, possibly-weaker path-handling
logic introduced here).
"""

from __future__ import annotations

import hmac
import os
from typing import Any, Dict

from mcp.server.mcpserver import MCPServer

from .mcp_server import _lock, _load, _save, server

_ACTION_TOKEN_ENV = "RUVECTOR_SALESFORCE_ACTION_TOKEN"
_FALLBACK_TOKEN_ENV = "RUVECTOR_MCP_TOKEN"

_registered = False


class SalesforceAuthNotConfiguredError(RuntimeError):
    """The Salesforce action routes were enabled with no bearer token set."""


def _expected_action_token() -> "str | None":
    """The configured action token: ``RUVECTOR_SALESFORCE_ACTION_TOKEN``,
    else ``RUVECTOR_MCP_TOKEN``. Empty / whitespace-only values count as
    unset (and are skipped), mirroring ``mcp_server._configured_token``."""
    for var in (_ACTION_TOKEN_ENV, _FALLBACK_TOKEN_ENV):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return None


def _check_salesforce_auth(request: Any) -> "Dict[str, Any] | None":
    """Return a JSON-serializable error dict if the request's bearer token
    doesn't match, or ``None`` if the request may proceed. A missing
    `_expected_action_token()` (neither env var set) is NOT a free pass here:
    unlike the MCP tools (loopback-only without a token), these routes
    are a network REST surface, so `maybe_register` refuses to mount them
    without a token and this check fails closed if that ever changes.

    Plain ``hmac.compare_digest`` (constant-time), not
    ``mcp_server.StaticTokenVerifier`` — that class's `verify_token` is an
    async `TokenVerifier`-protocol method that builds a full `AccessToken`
    (scopes, resource, ...) for the SDK's own bearer middleware to consume;
    none of that machinery applies on a `custom_route`, which the SDK
    never routes through that middleware at all (see this module's
    docstring) — reimplementing just the comparison here is more honest
    than instantiating a class whose real purpose doesn't apply on this
    path.
    """
    expected = _expected_action_token()
    if expected is None:
        # Fail closed. `maybe_register` refuses to mount these routes with no
        # token, so this is only reachable if the env changed afterwards.
        return {"error": "invalid_token", "error_description": "server has no action token configured"}
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return {"error": "invalid_token", "error_description": "missing Authorization: Bearer header"}
    token = header[len("Bearer ") :]
    # Compare as bytes: compare_digest(str, str) raises TypeError on any
    # non-ASCII character, which an attacker-controlled header can contain
    # (that was an unhandled 500).
    if not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        return {"error": "invalid_token", "error_description": "token does not match"}
    return None


def maybe_register(*, read_only: bool = False) -> bool:
    """Register the Salesforce action routes onto `server`, if not already
    done. Idempotent (safe to call more than once — e.g. once from
    `cli.py`'s `serve` command and once from a test). Returns whether
    registration happened in *this* call (``False`` if already registered
    by an earlier call).

    Raises :class:`SalesforceAuthNotConfiguredError` when no action token
    is configured — these routes never run unauthenticated. With
    ``read_only=True`` the mutating ``/salesforce/upsert`` route is not
    mounted at all.
    """
    global _registered
    if _registered:
        return False
    if _expected_action_token() is None:
        raise SalesforceAuthNotConfiguredError(
            f"refusing to mount the Salesforce action routes without authentication: set "
            f"{_ACTION_TOKEN_ENV} (or {_FALLBACK_TOKEN_ENV}) to a non-empty secret"
        )
    _register_routes(server, read_only=read_only)
    _registered = True
    return True


def _error_response(exc: Exception) -> "Dict[str, Any]":
    """Map an exception from an action call to a clean 4xx-shaped body.

    Without this, the first live end-to-end test of these routes
    (ADR-352) returned a raw Starlette "Internal Server Error" / 500 for
    an unknown collection name or a malformed request body — an unhandled
    `CollectionError`/`KeyError`/`ValueError` propagating straight out of
    the handler. A REST action a Salesforce Flow calls needs a real error
    *body* it can branch on (Agentforce surfaces the HTTP status +ids body
    to the conversation), not an opaque 500 — so every caught exception
    type here is one the action functions / `Collection` are documented to
    raise for a caller mistake (unknown collection, missing required
    field, dimension mismatch), not a blanket swallow of anything.
    """
    return {"error": type(exc).__name__, "error_description": str(exc)}


async def _run_action(request: Any, handler: Any) -> Any:
    """Shared auth-check + error-handling wrapper for the three mutating/
    query actions (not `openapi_spec`, which needs neither). `handler` is
    called with the parsed JSON body and must do its own `_load`/`_save`
    under `_lock` and return the JSON-serializable result dict.

    The exception mapping is deliberately narrow: `KeyError` (missing
    required field), `ValueError`/`TypeError` (bad field value/shape), and
    `ruvector.RuVectorError` (the real base of `CollectionError` and every
    extension-raised error -- unknown collection, dimension mismatch) are
    the only types the action functions / `Collection` are documented to
    raise for a *caller* mistake, so only those get a clean 4xx body. A
    prior version caught bare `Exception` here, which would silently
    report an unrelated bug (e.g. an `AttributeError` from a real defect
    in this module) as a 404 with its internal message exposed in the
    response body -- anything else now propagates and surfaces as
    Starlette's normal 500, which is the honest outcome for "this is our
    bug, not the caller's".
    """
    from starlette.responses import JSONResponse

    from ruvector import RuVectorError

    auth_error = _check_salesforce_auth(request)
    if auth_error is not None:
        return JSONResponse(auth_error, status_code=401)
    # The routes are POST + JSON by contract. Requiring the media type keeps
    # a cross-site HTML form (text/plain, urlencoded, multipart: the
    # "simple" content types a browser may send without a CORS preflight)
    # from reaching a handler.
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return JSONResponse(
            {"error": "unsupported_media_type", "error_description": "Content-Type must be application/json"},
            status_code=415,
        )
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(_error_response(exc), status_code=400)
    try:
        result = await handler(body)
    except KeyError as exc:
        return JSONResponse({"error": "KeyError", "error_description": f"missing required field: {exc}"}, status_code=400)
    except (ValueError, TypeError) as exc:
        return JSONResponse(_error_response(exc), status_code=400)
    except RuVectorError as exc:  # CollectionError and other extension-raised caller errors
        return JSONResponse(_error_response(exc), status_code=404)
    return JSONResponse(result)


def _register_routes(srv: MCPServer, *, read_only: bool = False) -> None:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @srv.custom_route("/salesforce/openapi.json", methods=["GET"])  # type: ignore[untyped-decorator]
    async def openapi_spec(request: Request) -> JSONResponse:
        from .integrations.salesforce import generate_openapi_spec

        base_url = f"{request.url.scheme}://{request.url.netloc}"
        return JSONResponse(generate_openapi_spec(base_url))

    @srv.custom_route("/salesforce/search", methods=["POST"])  # type: ignore[untyped-decorator]
    async def search_action(request: Request) -> JSONResponse:
        from .integrations.salesforce import action_search

        async def handler(body: Dict[str, Any]) -> Dict[str, Any]:
            with _lock:
                coll = _load(body.get("collection", "default"))
                return action_search(coll, body["query_vector"], k=body.get("k", 10), filter=body.get("filter"))

        return await _run_action(request, handler)  # type: ignore[no-any-return]

    if not read_only:

        @srv.custom_route("/salesforce/upsert", methods=["POST"])  # type: ignore[untyped-decorator]
        async def upsert_action(request: Request) -> JSONResponse:
            from .integrations.salesforce import action_upsert

            async def handler(body: Dict[str, Any]) -> Dict[str, Any]:
                name = body.get("collection", "default")
                with _lock:
                    coll = _load(name)
                    result = action_upsert(coll, body["vector"], metadata=body.get("metadata"))
                    _save(name, coll)
                    return result

            return await _run_action(request, handler)  # type: ignore[no-any-return]

    @srv.custom_route("/salesforce/ground", methods=["POST"])  # type: ignore[untyped-decorator]
    async def ground_action(request: Request) -> JSONResponse:
        from .integrations.salesforce import action_ground

        async def handler(body: Dict[str, Any]) -> Dict[str, Any]:
            with _lock:
                coll = _load(body.get("collection", "default"))
                return action_ground(
                    coll, body["query_vector"], k=body.get("k", 5), text_field=body.get("text_field", "text")
                )

        return await _run_action(request, handler)  # type: ignore[no-any-return]


__all__ = ["maybe_register", "SalesforceAuthNotConfiguredError"]
