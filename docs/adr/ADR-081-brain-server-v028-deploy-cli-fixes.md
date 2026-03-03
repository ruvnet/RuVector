# ADR-081: Brain Server v0.2.8–0.2.10 Deploy + CLI/MCP Bug Fixes

**Status**: Accepted
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud Deployment), ADR-060 (Shared Brain Capabilities), ADR-064 (Pi Brain Infrastructure)

## 1. Context

v0.2.7 shipped proxy-aware fetch and new brain API types/routes in the Rust server (types.rs, store.rs, routes.rs), but the Cloud Run service at pi.ruv.io was serving a **stale pre-built binary** — the Dockerfile copies a pre-compiled `mcp-brain-server` ELF from the repo root rather than building from source. The binary at `./mcp-brain-server` predated the v0.2.7 Rust changes, so scored search, paginated list, `POST /v1/verify`, and enhanced transfer all returned old formats or 404.

Deep review uncovered six bugs across CLI, MCP, and deployment:

### 1.1 Stale Binary in Docker Image

The `crates/mcp-brain-server/Dockerfile` does `COPY mcp-brain-server /usr/local/bin/mcp-brain-server` — it copies a pre-built binary from the Docker build context, not from a Cargo build step. The binary at the repo root was compiled before the v0.2.7 Rust changes (`ScoredBrainMemory`, `ListResponse`, `/v1/verify`), so Cloud Run was running old code despite the source being updated.

### 1.2 `proxyFetch()` Curl Fallback Hardcodes Status 200

`proxyFetch()` (cli.js line ~174) provides a curl-based fallback when Node's `fetch()` cannot reach the proxy. The fallback constructs a fake Response object with `status: 200` and `headers: new Map()` regardless of actual HTTP status. This means:
- The 204 guard in `brainFetch()` (`resp.status === 204`) never triggers
- `resp.headers.get('content-length')` returns `undefined` (Map, not Headers)
- DELETE operations returning 204 with empty body crash on `JSON.parse('')`
- Non-2xx errors silently appear as success

### 1.3 `brainFetch()` 204 Guard (Initial Fix)

`brainFetch()` unconditionally called `resp.json()`. While the 204 guard was added in the initial v0.2.8 pass, it was insufficient alone because of the proxyFetch fallback (1.2 above).

### 1.4 `fetchBrainEndpoint()` Missing 204 Guard

The AGI subcommands (`brain agi status`, `brain agi sona`, etc.) use a separate `fetchBrainEndpoint()` function (line ~8276) that also unconditionally calls `resp.json()` without a 204 guard.

### 1.5 MCP `brain_list` Schema Missing Properties

The MCP tool schema for `brain_list` only declared `category` and `limit`, but the handler reads `args.offset`, `args.sort`, and `args.tags`. Claude (or any MCP client) could not discover or send these parameters.

### 1.6 MCP `brain_sync` Handler Ignores `direction` Parameter

The sync handler at MCP line ~3419 hardcoded `url = ${brainUrl}/v1/lora/latest` without appending the `direction` query parameter from `args.direction`. The `pull`/`push`/`both` parameter was silently dropped.

### 1.7 MCP Brain Handler Missing 204 Guard

The shared brain tool handler (MCP line ~3426) does `const result = await resp.json()` unconditionally. DELETE returning 204 crashes the same way as the CLI.

## 2. Decision

### 2.1 Rebuild and Redeploy Binary

Compile `mcp-brain-server` from source (`cargo build --release` in `crates/mcp-brain-server/`), copy the fresh binary to the repo root, and redeploy via Cloud Build + Cloud Run. This activates:

- `ScoredBrainMemory` with `score: f64` in search results
- `ListResponse { memories, total_count, offset }` paginated envelope
- `POST /v1/verify` endpoint for witness chain verification
- Enhanced transfer warnings with domain-level safety checks

### 2.2 Fix `proxyFetch()` Curl Fallback

Capture actual HTTP status from curl via `-w '\n%{http_code}'`, parse the status code from the last line, and construct the Response object with correct `ok`, `status`, and safe `json()` that returns `{}` for empty bodies:

```js
const args = ['-sS', '-L', '--max-time', '30', '-w', '\n%{http_code}'];
// ...
const lines = stdout.trimEnd().split('\n');
const statusCode = parseInt(lines.pop(), 10) || 200;
const body = lines.join('\n').trim();
const ok = statusCode >= 200 && statusCode < 300;
return {
  ok,
  status: statusCode,
  statusText: ok ? 'OK' : `HTTP ${statusCode}`,
  text: async () => body,
  json: async () => body ? JSON.parse(body) : {},
  headers: new Map(),
};
```

### 2.3 Fix `brainFetch()` and `fetchBrainEndpoint()` 204 Guards

Both functions now check for 204 or empty content-length before calling `resp.json()`:

```js
if (resp.status === 204 || resp.headers.get('content-length') === '0') return {};
return resp.json();
```

### 2.4 Add `--json` to 4 CLI Brain Commands

Added `.option('--json', 'Output as JSON')` and the standard JSON gate to `brain share`, `brain vote`, `brain delete`, and `brain sync`.

### 2.5 Fix MCP `brain_list` Schema

Added `offset`, `sort`, and `tags` properties to the `brain_list` tool `inputSchema`, matching the handler's usage.

### 2.6 Fix MCP `brain_sync` Handler

Changed the sync handler to append `?direction=...` from `args.direction`:

```js
case 'sync': {
  const p = new URLSearchParams();
  if (args.direction) p.set('direction', args.direction);
  url = `${brainUrl}/v1/lora/latest${p.toString() ? '?' + p : ''}`;
  break;
}
```

### 2.7 Fix MCP Brain Handler 204 Guard

Changed `const result = await resp.json()` to:

```js
const result = (resp.status === 204 || resp.headers.get('content-length') === '0') ? {} : await resp.json();
```

### 2.8 Add `GET /v1/pages` Route (List Pages)

The Rust server had `POST /v1/pages` and `GET /v1/pages/:id` but no `GET /v1/pages` to list all pages. The CLI `brain page list` command tried to call this endpoint and got 405 Method Not Allowed.

Added:
- `PageSummary` and `ListPagesResponse` types in `types.rs`
- `list_pages()` store method in `store.rs`
- `list_pages` route handler in `routes.rs` with pagination (`limit`, `offset`), `status` filter, and sort by `updated_at` descending
- Registered route: `.route("/v1/pages", get(list_pages).post(create_page))`

### 2.9 Add 9 Page/Node MCP Tools

The `brain page` and `brain node` CLI commands (Brainpedia ADR-062, WASM Nodes ADR-063) were only available via the Rust SSE MCP server, not in the Node.js stdio MCP server. This meant Claude Desktop (stdio transport) could not access page or node operations.

Added 9 new MCP tool definitions and handlers to `mcp-server.js`:

| Tool | Method | Endpoint |
|------|--------|----------|
| `brain_page_list` | GET | `/v1/pages` |
| `brain_page_get` | GET | `/v1/pages/:id` |
| `brain_page_create` | POST | `/v1/pages` |
| `brain_page_update` | PUT | `/v1/pages/:id` |
| `brain_page_delete` | DELETE | `/v1/pages/:id` |
| `brain_node_list` | GET | `/v1/nodes` |
| `brain_node_get` | GET | `/v1/nodes/:id` |
| `brain_node_publish` | POST | `/v1/nodes` |
| `brain_node_revoke` | POST | `/v1/nodes/:id/revoke` |

All handlers include the 204 guard pattern and use `proxyFetch` for proxy-aware connectivity.

### 2.10 Cosmetic Fixes (v0.2.10)

- **`brain delete` JSON output**: Changed `--json` / non-TTY output from bare `{}` to `{ "deleted": true, "id": "<id>" }` — meaningful for piped consumers
- **`brain page get` display**: Unwrap `.memory` wrapper from `PageDetailResponse` for human-readable output — shows title, status, category, quality score, tags, delta/evidence counts, and content instead of raw JSON dump
- **`brain page list` display**: Enhanced formatting with quality scores, status badges, and total count header

### 2.11 Version Bumps

- **0.2.7 → 0.2.8**: Initial bug fixes (proxyFetch, 204 guards, --json flags)
- **0.2.8 → 0.2.9**: GET /v1/pages route, 9 new MCP tools, fresh binary deploy
- **0.2.9 → 0.2.10**: Cosmetic fixes for delete JSON output and page display

Updated in:
- `npm/packages/ruvector/package.json`
- `npm/packages/ruvector/bin/mcp-server.js` (2 occurrences)

## 3. Files Modified

| File | Changes |
|------|---------|
| `npm/packages/ruvector/bin/cli.js` | Fix `proxyFetch()` curl fallback to capture real HTTP status; fix `brainFetch()` and `fetchBrainEndpoint()` 204 guards; add `--json` to 4 brain commands |
| `npm/packages/ruvector/bin/mcp-server.js` | Add `offset`/`sort`/`tags` to `brain_list` schema; fix `brain_sync` direction passthrough; add 204 guard to brain handler; add 9 page/node MCP tools; version bump x2 |
| `npm/packages/ruvector/package.json` | Version 0.2.7 → 0.2.9 |
| `npm/packages/ruvector/test/integration.js` | MCP tool count threshold updated from 103 to 112 |
| `crates/mcp-brain-server/src/types.rs` | Add `PageSummary`, `ListPagesResponse` types |
| `crates/mcp-brain-server/src/store.rs` | Add `list_pages()` method |
| `crates/mcp-brain-server/src/routes.rs` | Add `list_pages` handler, register `GET /v1/pages` route |
| `mcp-brain-server` (binary) | Rebuilt from source with `ScoredBrainMemory`, `ListResponse`, `/v1/verify`, `GET /v1/pages` |

## 4. Consequences

### Positive

- **Server-side features live**: Scored search, paginated list, verify endpoint, GET /v1/pages, and enhanced transfer are now served from a binary compiled from the current source.
- **CLI robustness**: `brain delete` and `brain vote` no longer crash. The proxy fallback correctly reports non-2xx errors instead of silently swallowing them.
- **MCP completeness**: 112 total MCP tools. `brain_list` schema exposes pagination/sort/tags. `brain_sync` direction parameter reaches the server. 9 new page/node tools available in stdio transport (previously SSE-only). DELETE operations return clean `{}`.
- **API consistency**: All 19 brain CLI commands + 6 AGI commands now support `--json`.
- **Full parity**: Every brain CLI command now has a corresponding Node.js MCP tool — no more SSE-only gaps.

### Negative

- The Dockerfile still uses a pre-built binary strategy. A future improvement would add a Cargo build stage to ensure the deployed binary always matches the source.

## 5. Audit: Brain CLI Commands vs Server Routes vs MCP Tools

### CLI Commands (19 total)

| CLI Command | Server Route | MCP Tool | --json | Notes |
|------------|--------------|----------|--------|-------|
| `brain search <query>` | `GET /v1/memories/search` | `brain_search` | Yes | Score field now present |
| `brain share <title>` | `POST /v1/memories` | `brain_share` | Yes | Fixed in v0.2.8 |
| `brain get <id>` | `GET /v1/memories/:id` | `brain_get` | Yes | |
| `brain vote <id> <dir>` | `POST /v1/memories/:id/vote` | `brain_vote` | Yes | Fixed in v0.2.8 |
| `brain list` | `GET /v1/memories/list` | `brain_list` | Yes | Paginated envelope now live |
| `brain delete <id>` | `DELETE /v1/memories/:id` | `brain_delete` | Yes | Fixed 204 crash |
| `brain status` | `GET /v1/status` | `brain_status` | Yes | |
| `brain drift` | `GET /v1/drift` | `brain_drift` | Yes | |
| `brain partition` | `GET /v1/partition` | `brain_partition` | Yes | |
| `brain transfer <s> <t>` | `POST /v1/transfer` | `brain_transfer` | Yes | |
| `brain sync [dir]` | `GET /v1/lora/latest` | `brain_sync` | Yes | Fixed direction passthrough |
| `brain page list` | `GET /v1/pages` | `brain_page_list` | Yes | Added in v0.2.9 |
| `brain page get <id>` | `GET /v1/pages/:id` | `brain_page_get` | Yes | Added in v0.2.9 |
| `brain page create` | `POST /v1/pages` | `brain_page_create` | Yes | Added in v0.2.9 |
| `brain page update <id>` | `PUT /v1/pages/:id` | `brain_page_update` | Yes | Added in v0.2.9 |
| `brain page delete <id>` | `DELETE /v1/pages/:id` | `brain_page_delete` | Yes | Added in v0.2.9 |
| `brain node list` | `GET /v1/nodes` | `brain_node_list` | Yes | Added in v0.2.9 |
| `brain node get <id>` | `GET /v1/nodes/:id` | `brain_node_get` | Yes | Added in v0.2.9 |
| `brain node publish` | `POST /v1/nodes` | `brain_node_publish` | Yes | Added in v0.2.9 |
| `brain node revoke <id>` | `POST /v1/nodes/:id/revoke` | `brain_node_revoke` | Yes | Added in v0.2.9 |
| `brain agi status` | `GET /v1/status` | `brain_agi_status` | Yes | AGI field extraction |
| `brain agi sona` | `GET /v1/sona/stats` | `brain_sona_stats` | Yes | |
| `brain agi temporal` | `GET /v1/temporal` | `brain_temporal` | Yes | |
| `brain agi explore` | `GET /v1/explore` | `brain_explore` | Yes | |
| `brain agi midstream` | `GET /v1/midstream` | `brain_midstream` | Yes | |
| `brain agi flags` | `GET /v1/status` | `brain_flags` | Yes | Flag field extraction |

### Server Routes Not Exposed in CLI/MCP

| Route | Description | Status |
|-------|-------------|--------|
| `GET /v1/health` | Health check | Used internally by midstream tools |
| `GET /v1/challenge` | Nonce for replay protection | Used by SSE MCP, not needed in CLI |
| `POST /v1/verify` | Witness chain verification | New in v0.2.8 — no CLI command yet |
| `POST /v1/lora/submit` | Submit LoRA weights | No CLI command |
| `GET /v1/training/preferences` | Training prefs | No CLI command |
| `GET /v1/pages/:id/deltas` | List page deltas | Accessible via `brain page` but not granular MCP tool |
| `POST /v1/pages/:id/evidence` | Add evidence | Not exposed as separate command |
| `POST /v1/pages/:id/promote` | Promote page | Not exposed as separate command |
| `GET /v1/nodes/:id/wasm` | Download WASM binary | Not exposed |

## 6. Verification

### v0.2.8 (initial fixes)

1. `node -c bin/cli.js && node -c bin/mcp-server.js` — syntax check passes
2. `npm test` — 69 tests pass
3. `cargo build --release` in `crates/mcp-brain-server/` — compiles with `ScoredBrainMemory`, `ListResponse`
4. Cloud Build + Cloud Run redeploy with fresh binary
5. `brain status --json` — confirms API responds
6. `brain search <query> --json` — confirms `score` field present in results
7. `brain list --sort quality --limit 5 --json` — confirms `{ memories, total_count, offset }` envelope
8. `brain delete <id> --json` — returns `{}` without crash

### v0.2.9 (page/node MCP tools + GET /v1/pages)

9. `GET /v1/pages?limit=2&status=canonical` — returns `{ pages, total_count, offset, limit }` envelope
10. `npm test` — 69 tests pass, MCP tool count 112
11. 9 new MCP tools in `mcp-server.js` — `brain_page_list/get/create/update/delete`, `brain_node_list/get/publish/revoke`
12. Cloud Build + Cloud Run redeploy (revision `ruvbrain-00075-m7w`)
13. Published `ruvector@0.2.9` to npm
