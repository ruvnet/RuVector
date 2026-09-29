# ADR-351: ruvector edge — Multi-Tenant Rust/wasm Services on Cloudflare Workers + Durable Objects, Behind an Edge Authorization Server Federated to cognitum-one OAuth

## Status

**Proposed.** 2026-09-28; reconciled 2026-09-29 with the edge-authorization-server decision and
**again with the shipped code at `52e2a5c55`** (Review log). M0–M5 are **deployed to workers.dev**
on account `501c77f5…`, which is on **Workers Free** (Workers Paid is a G2 precondition, pending
rUv): `ruvector-edge-auth` and `ruvector-edge-gateway` (DO migration tags `v1`, `v2-registry`,
`v-m4-quant-graph`). Hosted login waits on G1 (cognitum-one/console#605): `UPSTREAM_CLIENT_ID` and
`ACCEPTED_UPSTREAM_KIDS` are empty, `/authorize` answers `temporarily_unavailable`, so no
authenticated flow has run live. Deploy evidence lives in `edge/HORIZON.json`; every further
resource creation and provider touch stays gated (§13).

> **Numbering note.** `docs/adr/` stops at ADR-349; ADR-350 has four open claimants (PRs #936,
> #1030, #1056, one more); if 351 is taken before merge, renumber and keep this note.

### Verification legend

- **[V]** Verified by (a) a live fetch or read-only API call on 2026-09-28, (b) source at a
  **recorded commit** — console `origin/main` **`ce9ddca`** (re-read 2026-09-28 with `git show`),
  api `origin/main` **`9295444`** — (c) the ruvector-edge **working tree** on
  `feat/ruvector-edge-services` re-read at 2026-09-28 23:25 EDT ("[V] tree": the file, type or
  constant exists as stated; a concurrent workflow is still changing it), or (d) Cloudflare's docs.
  `.rs:NN` are line numbers at those commits.
- **[L]** Likely: inferred, or verified on an older checkout only.
- **[U]** Unverified: the named milestone must measure or confirm it.
- **[V] 52e2a5c55** Re-read in the edge worktree at that commit on 2026-09-29 (the "Shipped vs
  designed" table and every body line carrying the tag). **[Paid]** marks a limit that exists only
  because the account is on Workers Free (10 ms CPU per invocation, no `[limits] cpu_ms`: the API
  refuses it with code 100328) and that Workers Paid (30 s default CPU) would lift or relax.
  **[needs Paid]** marks the opposite case: a budget sized for Paid's 30 s default (VectorShard
  lazy-load `DECODE_BUDGET_MS` 300 ms + `REPLAY_BUDGET_MS` 600 ms, flushes, `REBUILD_BUDGET_MS`
  20 s, registry writes), which on Free is expected to exceed CPU for any non-trivial shard or blob.

### Shipped vs designed (as of `52e2a5c55`)

Where the table and a body section disagree, the table and the corrected body describe what ships;
the design text is kept where it is still the target. `file:line` are in the edge worktree.

| # | Area | Designed | Shipped, and why | Evidence |
|---|---|---|---|---|
| 1 | Idempotency (§6.1, §7, §16.3) | `422 idempotency_mismatch` | **`409 op_replayed`** on REST, `/v1/ops` and graph mutations: one code for one fact (a key reused with another body), shared with the §16.3 contract. The key is bound to sha256(route tag ‖ raw body bytes) (REST/graph, namespaced `rest:<key>`) or sha256(raw body) (`/v1/ops`), not a canonical JSON form. It is reserved in the lookup's DO turn (a concurrent twin gets `409 conflict`, with `retry_after_s: 1` on `/v1/ops`; reservation TTL 120 s); **only successful** mutating responses are remembered (≤ 64 KiB, 24 h); failures release the key. Honoured on create/upsert/delete, graph `…/edges` and mutating Cypher; on REST and graph, reads and dry runs ignore the header (a malformed key there is not rejected); on `/v1/ops` dry runs and read ops still **look up** `op_id` (a reuse with another body is `409 op_replayed`, a same-body hit can replay) but are never reserved or remembered, and the header must equal `op_id` (else 400). `idempotency_mismatch` survives only in the unused tenancy `ProblemCode`. | `edge/gateway/src/idem.rs:1-9,52,77`; `rest.rs:270-296,315-327`; `ops.rs:89-91,107,128-170`; `graph_routes.rs:18-21,285-311`; `crates/ruvector-edge-store/src/ledger/idem.rs:1-28`; `crates/ruvector-edge-store/src/error.rs:69`; `crates/ruvector-edge-tenancy/src/problem.rs:80` |
| 2 | rv-mincut modes (§3) | exact and approximate | `mode: approximate` (ε ∈ (0,1]) is **answered by the exact solver**; the report says `mode: "exact"`, `requested_mode: "approximate"`, echoes `epsilon`. `ruvector-mincut` `ApproxMinCut` returned ≈ 0.10–0.33× the exact cut (up to 10× below the true minimum, e.g. 0.333 vs exact 3.333; no cut of that weight exists), so serving it would break the ε contract. Upstream fix owed: RuVector#1085 (also `ruvector-mincut` wasm32 clippy `static_mut_refs`/`unused_unsafe`). | `crates/ruvector-edge-analytics/src/service.rs:55-74`; `tests/equivalence.rs:163-168`; `edge/gateway/src/mincut_core.rs:99-114,200-225` |
| 3 | Vector index kinds (§6.1, §7.2) | M1 `flat` = f32 slab; M2a `q8`; M2b `hnsw` | Kinds are **`flat` (default) \| `hnsw` \| `rabitq`**; `q8` is refused. `flat` = int8 codes resident + exact f32 rerank from SQLite (rerank `max(40, 4·top_k)` ≤ 1000); measured recall@10 **1.000** (20k × 384). `hnsw` is **opt-in** (built in-repo in `ruvector-edge-index`, not via G4a): m 16 / efc 128 default, per-metric default `ef` **cosine 1024, l2/dot 512** (max 2048) because 0.95 recall on random data needed them. Resident ≈ 384 B/vector flat, ≈ 522 B/vector HNSW m16 at 384-d, plus ~128 B row bookkeeping. HNSW is built only in alarms (`REBUILD_BUDGET_MS` 20 s ⇒ ≈ 18k nodes at 384-d m16) **[needs Paid]**: that budget and the lazy-load budgets (decode 300 ms + replay 600 ms; a full 14 MB shard decodes in ≈ 31 ms at 2.3 ms/MiB) are sized for Paid's 30 s, so on Free cold loads and rebuilds of non-trivial shards are expected to exceed CPU. | `crates/ruvector-edge-store/src/shard/codec.rs:150-164,216-222`; `shard/read.rs:38-46,173-178`; `shard/maintain.rs:59`; `shard/slab.rs:16`; `crates/ruvector-edge-index/src/memory.rs:31-42` |
| 4 | Memory caps (§6.1, §10) | 56 MB resident per isolate, per shard 3M floats (12 MB f32), 3–4 shards | 56 MB is **split per class** because every DO class of the script can share an isolate: `VectorShard` 28 MB (per-shard cap 14 MB, **2** full shards), `QuantShard` 16 MB, `GraphStore` 12 MB; plus one `AnalyticsJob` turn ≤ 32 MiB and one registry finalize part 2 × 16 MiB ⇒ ≈ 123 of 128 MB. Stored f32 per shard ≤ 16M floats (SQLite). `M1_SHARD_FLOAT_CAP` is unused by the gateway (f32 stays in SQLite). | `edge/gateway/src/shard_core.rs:20-39`; `crates/ruvector-edge-store/src/shard/mod.rs:69-73`; `shard/write.rs:34-39`; `resident.rs:18`; `edge/gateway/src/quant_shard.rs:36-40`; `graph_store.rs:51-56`; `mincut_core.rs:41-44` |
| 5 | Registry (§3, §6.3, §7.2) | `/v1/rvf/{name}/{version}*`, push by `upload_id`, D1 index, keys `rvf/{tenant}/{name}/{version}` | Names are **`@scope/name`**; a scope is a **global** namespace claimed by one tenant (`POST /v1/rvf/scopes/{scope}`, `ruvector:admin`); index = DOs `RegistryRoot` (`idFromName("v1\|registry\|root")`) + one `RegistryScope` per scope (`hex(sha256("v1\|registry\|" + scope))`), no D1. Push = registry-native upload sessions (`…/uploads`, `PUT …/parts/{n}` ≤ 16 MiB, `:finalize`), then `:publish`/`:yank`/`:unyank`. Blobs are content-addressed and **the key includes the scope**: `rvf/{tenant_key}/{scope}/blobs/sha256/{hex}` (public: `public/blobs/sha256/{hex}`, write-once). A manifest mirror at `rvf/{tenant_key}/@{scope}/{name}/{version}/manifest.json` is *(designed, not written)*: `keys::manifest_key` exists but only its unit test calls it; the gateway stores the blob plus the `RegistryScope` DO rows. `RESERVED_SCOPES`/`RESERVED_NAMES` are enforced on **deserialise** too, so adding a reserved word makes stored names unreadable: a list change is a **compatibility break** (migration needed). All registry writes hash MBs per request ⇒ **[needs Paid]** (expected 1102 on Free). | `edge/gateway/src/registry_routes.rs:1-19,87-118`; `registry_ports.rs:95`; `rvf_finalize.rs:249`; `crates/ruvector-edge-registry/src/keys.rs:7-16,52-66,70-96,126`; `name.rs:1-15,29,212-214,240-245`; `edge/gateway/wrangler.toml:110-127` |
| 6 | Workers Free limits (§6.3, §7.2, §10) — all **[Paid]** | 8 MiB inline import; streamed snapshots of any size; 200k-edge graphs; mincut ≤ 200k edges | Inline `:import` ≤ **512 KiB**; synchronous snapshot/export/restore **413** above **2^18 stored floats** (682 rows at 384-d), checked before any side effect; queued import segments ≤ **1 MiB** (rvf CLI needs `--batch-size` ≤ 680 at 384-d), 2 batches per delivery, stalled after 15 deliveries; quant shard ≤ **50k rows / 6M load units**, a cold 50k load answers **one retryable 503**; graph persisted state ≤ **256 KiB** (≈ 1.5–2k edges), ≤ 1k edges per bulk request, 20 graphs/tenant; mincut inline ≤ 5M work units, jobs ≤ **32 MiB** (413 above ≈ 80k edges), ≤ 20 live jobs/tenant, TTL **24 h**, 3 attempts — a 51k-edge solve is admitted but ends **413 `budget_exceeded`** on Free; registry writes fail on Free (**[needs Paid]**, row 5). | `edge/gateway/src/uploads.rs:38`; `sync_budget.rs:1-27`; `ingest.rs:71-84`; `quant_shard.rs:56-62`; `graph_store.rs:31-50`; `graph_catalog.rs:18-21`; `mincut_core.rs:33-58`; `mincut_job.rs:69`; `crates/ruvector-edge-analytics/src/job.rs:18` |
| 7 | Scopes (§5.3, §16.1) | — | As designed: per-resource vocabularies (`/v1`, `/v1/mcp`: `ruvector:*` + `offline_access`; `https://team.ruv.io/mcp`: `team:read/write/run`); exact `aud`; RFC 8693 map `team:read→ruvector:read`, `team:write→ruvector:write`, `team:run→∅`, a `ruvector:*` subject maps to nothing; `act` bound; adapter-audience tokens get 401 at `/v1` and `/v1/mcp`. **Gap:** the live `RESOURCE_ALLOWLIST` and PRM list no `ruvector:publish`, so `:publish` is unreachable until the operator adds it (M5 config). | `crates/ruvector-edge-authz/src/resource.rs:77-97,127-135`; `client.rs:37`; `crates/ruvector-edge-auth/src/prm.rs:13-21`; `edge/auth-worker/wrangler.toml:63` |
| 8 | rabitq collections (§7.2) | snapshot/export/restore for every index | Refused **400 `invalid_request`** before any charge: rows live in `QuantShard`, which does not serve the `/m3` side channel. Also no metadata `filter` (400) and no per-row `ops` log on `QuantShard`. Imports are served. | `edge/gateway/src/quant_route.rs:70-84`; `quant_shard.rs:14-16`; `quant_query.rs:44` |
| 9 | M3 resources (§6.3–§6.5, G3) | R2 + D1 `ruvector-edge-control` + Queues `ruvector-edge-jobs`(+DLQ), `ruvector-edge-audit` | **One** R2 binding `EDGE_DATA` → `ruvector-edge-data` for every prefix (M3 data and M5 registry unified); Queues **`ruvector-edge-ingest`** (DLQ `ruvector-edge-ingest-dlq`, 20 retries × 30 s) and **`ruvector-edge-audit`** (DLQ `ruvector-edge-audit-dlq`); Workers AI `AI` → `@cf/baai/bge-small-en-v1.5` (embedder `bge-small-en-v1.5`). **No D1:** no global deny (`DENY_GLOBAL` unset, `deny_check` is tenant-only), no `usage_daily`, `tenants`, `plans`, `packages`. | `edge/gateway/wrangler.toml:136-188`; `embed.rs:1-4,32`; `tenant_admin.rs:315-358` |
| 10 | Audit (§6.1, §6.5) | per-DO hash-chained `audit` tail | **One chain per tenant** in `TenantLedger`, fed by `AUDIT_QUEUE` (one message per write, sent via `wait_until`, best effort) → `audit/{tenant}/{yyyy}/{mm}/{dd}/{seq:012}.ndjson`. Plain reads are not audited. `GET /v1/audit` is not shipped. | `edge/gateway/src/audit.rs:1-20`; `m3_audit_ledger.rs:62` |
| 11 | Graph / job DO names (§4.3) | ledger-assigned random uid; recreate ⇒ new DO | Graph uid = `sha256("v1\|graph\|" + name)[0..16]`, job uid from `"v1\|mincut-job\|" + job_id`; a per-tenant catalog DO `$catalog`. Recreating a deleted graph **reuses** the DO name (the delete wipes rows first). | `edge/gateway/src/graph_wire.rs:15,26-50` |
| 12 | Authorization table (§5.3) | one runtime default-deny `ROUTE_TABLE` with roles | `ROUTE_TABLE` (M1 rows, no role column, `RouteSurface` = `Rest \| Mcp`) is **not consulted at runtime**; each handler authorizes: `/v1` ops via `ops::authz::authorize(op, scope caps, role, claimed)`, drop = write + owner, graph `Need::{Read,Write}`, registry caps = scope ∩ role. | `crates/ruvector-edge-auth/src/scopes.rs:80-85,194`; `edge/gateway/src/service.rs:70-78`; `crates/ruvector-edge-store/src/ops/authz.rs:47`; `tenant_admin.rs:255-261` |
| 13 | Edge AS open items (§5.6) | `localhost` loopback; 30 s refresh grace; `subjects` table; auth-worker trust roots compiled | Still open (shipped is stricter or var-based): `http://localhost` redirects are **refused** (only `127.0.0.1`/`[::1]`), so Claude Code / MCP Inspector must use `127.0.0.1`; any re-presented rotated refresh token **revokes the family immediately** (no grace); AuthStore has `redeemed_codes`, `dcr_rate`, `client_activity`, `assertion_jtis` but **no `subjects`**; auth-worker `ISSUER`/`UPSTREAM_*` are `[vars]` (gateway trust root is compiled, `trust_root.rs`). Stricter than designed at the callback: upstream `kid`s are pinned (`ACCEPTED_UPSTREAM_KIDS`) and a returned id_token is verified including `nonce` (designed: unpinned, id_token ignored). | `crates/ruvector-edge-authz/src/client.rs:166-172`; `src/tests/dcr.rs:136`; `refresh.rs:143-150`; `edge/auth-worker/src/sql_ports.rs:28-38`; `config.rs:139-158`; `edge/gateway/src/trust_root.rs:13-30` |
| 14 | Surfaces not shipped | `/v1/results`, `GET /v1/audit`, `GET …/vectors/{id}`, `GET /v1/rvf?prefix=`, `ruvector-edge-cli`, global deny, dry_run on every mutating tool | Absent. `dry_run` exists on collection create via `/v1/ops` and MCP only, on vector upsert/delete (REST, ops, MCP) and on MCP `graph_mutate`; **not** on REST `POST /v1/collections` (400 `dry_run not accepted`), tenant claim (REST and MCP), collection drop, member invite/remove, deny, graph REST create/delete/edges, `rvf_import`, M3, mincut jobs or registry routes. | `edge/gateway/src/rest.rs:141-175,239-250`; `rest_tests.rs:116`; `mcp_tests.rs:233-236`; `m3_api.rs:79-100`; `mcp.rs:110-135,415-419`; `rvf_mcp.rs:72-100` |
| 15 | CORS (§7) | explicit origin allowlist | `Access-Control-Allow-Origin: *` on every response and preflight: the API is bearer-only (no cookies), and browser MCP clients must read the 401/403 challenge. Exposed: `WWW-Authenticate`, `Retry-After`, `ETag`, `X-RVF-Yanked`, `Content-Disposition`. | `edge/gateway/src/respond.rs:1-30` |

Rate Limiting: both Workers declare `namespace_id` 35101/35102 (auth `DCR_RATE_LIMITER`/
`AUTHZ_RATE_LIMITER`, gateway `RL_READ_USER`/`RL_READ_ORG`) although the auth file says ids are
unique within the account **[U]**: if namespaces are account-scoped the counters are shared (key
spaces are disjoint: hashed IP vs `user:`/`org:`, but the limits differ); renumber under G2
(`edge/gateway/wrangler.toml:41,46`, `edge/auth-worker/wrangler.toml:28-40`).

## 1. Context

A **family of ruvector services** (vector search, quantised search, property graph, graph analytics,
RVF registry, MCP facade) on Cloudflare, multi-tenant, signing users in via `auth.cognitum.one`.

### 1.1 The upstream authorization server as it stands today

| Fact | Value | Tag |
|---|---|---|
| Issuer | `https://auth.cognitum.one` | [V] live RFC 8414 discovery |
| Endpoints | `/oauth/authorize`, `/oauth/token`, `/oauth/revoke`, `/userinfo`, `/oauth/register` (DCR), `/.well-known/jwks.json` | [V] |
| Grants | `authorization_code`, `refresh_token`; PKCE `S256` only; public clients (`none`) allowed. `/oauth/authorize` requires `state` and rejects an empty `scope`. | [V] |
| Signing | ES256 only. One P-256 key, `kid = _jQ62WD8cCiIGkKNQB8Hg4El2TNU5rHIITV4h_ba4YM` (RFC 7638 thumbprint). ID tokens are signed with the **same key**. | [V] live JWKS + `jwt.rs` |
| Introspection (RFC 7662) | **Does not exist.** Tokens are verified locally against the JWKS. | [V] |
| Access-token TTL | 15 min. A refresh token is **always** issued (90-day sliding family); `/oauth/revoke` accepts refresh tokens only (form-encoded) and revokes the whole family. | [V] `token.rs:429-560`, `revoke.rs:25` |
| **Audience binding** | `aud` is the **requesting client_id**. `AuthorizeQuery` has no `resource` field, so RFC 8707 is silently ignored (api-repo ADR-038 still open). | [V] `jwt.rs`, `oauth/authorize.rs` |
| Org / workspace selection | **None.** `handle_authorize` always takes `first_org_for_user` (oldest `invited_at`) and that org's default workspace; no selector, no workspace-membership check. A self-signup user's tokens carry their **personal** org. | [V] `authorize.rs:491-497`; [L] `members.rs:78`, `invited_at DEFAULT now()` |
| Role claims | `access_role` is set only for the `cognitum-sales\|legal\|university` application clients. DCR/OAuth access tokens carry no `access_role`, `email` or `sid`. **No org-role source exists.** | [V] `jwt.rs:221, 396-399` |
| **DCR redirect allowlist** | Public clients only. A redirect must **start with** one of `https://chatgpt.com/connector/oauth/`, `https://chatgpt.com/aip/`, `https://claude.ai/api/mcp/`, `https://claude.com/api/mcp/`, `http://127.0.0.1:`, `http://localhost:`, **or exactly equal** one of three named URLs (`mcp-factory.ruv.chatgpt.site`, `ruflo-federation.ruv.chatgpt.site`, `api.lovable.dev`). Anything else → `invalid_redirect_uri`. The source calls adding an exact entry "a deliberate act … it belongs in review". **`https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/callback` matches none of them.** | [V] `oauth/register.rs:50-86, 169-190` at `ce9ddca` |
| DCR other rules | Minted ids are prefixed `dcr-`. Omitting `scope` grants the **whole** registrable ceiling. Global cap `MAX_REGISTERED_CLIENTS = 500` across Cognitum. | [V] `register.rs:135, 224, 344` |
| `REGISTRABLE_SCOPES` | `openid email profile offline_access swarm:read swarm:publish mcp:read mcp:invoke mcp:account mcp:governance mcp:spaces`. `console`, `inference`, `federation:*`, `sensing:*`, `fleet`, `spaces:read`, `brains:*`, `namespaces:claim` are reserved. | [V] `metadata.rs:196-215` |

Upstream access tokens (`jwt.rs:178-260`) [V] carry `sub` (stable user UUID), `org_id`,
`workspace_id`, `client_id`, `scope`, `family_id`, `jti`, `iat`, `exp`, `iss`, `aud = client_id`,
claim `typ = "access"`, and false-by-default flags `setup`, `workload`, `exchanged`. ID tokens have
no `typ`; the edge AS ignores them (§5.6).

Constraints from other ADRs and live services:

- **ADR-124 (website repo, Proposed)** approves no provider change [V]; G1 (§13) is separate.
- **api-repo ADR-103 is merged**: `audienceAccepted()` is `aud === clientId ||
  aud.startsWith('dcr-')` (`mcp-oauth.ts:223`) and `mcp:*` is enforced per request [V]; the live PRM
  at `https://api.cognitum.one/.well-known/oauth-protected-resource/v1/mcp` names
  `auth.cognitum.one` [V]. **Consequence:** any upstream `dcr-*` token carrying `mcp:*` is a working
  credential at `api.cognitum.one/v1/mcp`. The edge AS's upstream client therefore requests
  **identity scopes only** (§5.6).

### 1.2 Platform limits that drive the design (Cloudflare docs, 2026-09-28)

| Limit | Value | Tag |
|---|---|---|
| Isolate memory | **128 MB per isolate**, shared by the JS heap and wasm. **DOs of the same class can share one isolate** (DO metrics: "Memory is measured per isolate, not per Durable Object"). With workers-rs, co-located DOs share **one wasm instance and one linear memory**, and wasm memory never shrinks. | [V] |
| Global-scope startup | **1 s** for module instantiate; lazy load on the first request counts against request CPU. | [V] |
| Bundle size | 64 MiB uncompressed on all plans (changelog 2026-09-04) | [V] |
| CPU per request | Free 10 ms; Paid 30 s default, `limits.cpu_ms` up to 300 s, **per script**. Whether DO request CPU follows it is unclear. **This account is on Free**: a `[limits] cpu_ms` block is refused (API code 100328), so the M3/M4/M5 request budgets are sized for 10 ms (**[Paid]** rows in the Shipped-vs-designed table), but the VectorShard lazy-load (decode 300 ms + replay 600 ms), flush and HNSW rebuild (20 s) budgets and all registry writes are sized for Paid's 30 s (**[needs Paid]**): on Free, cold loads and HNSW rebuilds of non-trivial shards are expected to exceed CPU. | [V] / **[U]** (M1); [V] 52e2a5c55 deploy log |
| DO SQLite | 10 GB per object; 30-day PITR. **2 MB** max string/BLOB/row, **100 KB** max statement, **100** bound parameters, 100 columns. | [V] |
| DO SQLite atomicity | No `txn` object; atomicity comes from **write coalescing** (synchronous `sql.exec` calls with no intervening `await`). Non-storage I/O lets requests interleave. No transaction spans two DOs. | [V] |
| DO throughput / connections / alarms | ~500–1,000 simple req/s per object; 6 simultaneous outgoing connections per request; exactly **one** alarm per object. | [V] |
| workers-rs | `Storage::transaction` needs a `'static` closure (`durable.rs:635-639`); `SqlCursor::to_array` materialises every row (`sql.rs:330-340`), `cursor.raw()` streams (`sql.rs:388`). The edge workspace pins `worker =0.8.5`, `wasm-bindgen =0.2.125` (§8). | [V] workers-rs `b57ba6e`; [V] tree |
| workers-rs panic recovery | Any `WebAssembly.RuntimeError` (OOM, `panic = "abort"`) **reinitialises the whole wasm instance**, recreating every co-located DO. `--panic-unwind` needs nightly + `-Zbuild-std`. | [V] |
| Rate Limiting binding | GA since 2025-09-19; global vs per-location counting not confirmed. | [V] / **[U]** (M1) |
| Cache API | `cache.put` is guaranteed only on custom domains; on `workers.dev` it may no-op. | [V] |
| Service Bindings | Worker-to-Worker calls within one account; not an authentication mechanism. | [L] |
| Workers AI `@cf/baai/bge-small-en-v1.5` | 384-dim embeddings (same as all-MiniLM-L6-v2) | [V] |
| D1 per-database size cap | Commonly cited as 10 GB. Moot as shipped: no D1 is bound (Shipped-vs-designed #9). | **[U]** (M3) |
| Vectorize namespaces | Logical partitions of **one shared index**; no hard isolation | [V] |

### 1.3 The target account

| Fact | Tag |
|---|---|
| Account `501c77f558c68152352b8ddfade9b854` also holds the `cognitum-consultant-email-staging-*` stack (7 Workers, 4 Queues, 2 R2 buckets, 1 DO namespace). An account-wide deploy token could modify it. | [V] |
| workers.dev subdomain **`cognitum-consulting-mail`**, so the hosts are `ruvector-edge-auth.cognitum-consulting-mail.workers.dev` and `ruvector-edge-gateway.cognitum-consulting-mail.workers.dev`. **No custom domains**: the only zone is `cognitum.consulting`; `cognitum.one` (so `api.cognitum.one`) and `ruv.io` are **not** in this account. A custom domain is future work (G6). | [V] |
| D1, R2, DO and Queues are reachable with wrangler OAuth for this account. Workers for Platforms is not enabled (10121). No Vectorize, KV or Secrets Store yet. | [V] (rUv, 2026-09-29) / [V] |
| D1 `ruvector-chatgpt` (`ee7fc4c4-…`) holds `memberships(issuer, subject, tenant_id, role)`, JSON-text vectors and a 500-vector trigger. It belongs to the ChatGPT adapter (`integrations/cloudflare-chatgpt`, PR #1062). This ADR never writes it; the boundary is §16. | [V] schema; [L] ownership (coordinator) |

### 1.4 ruvector crates that could back the services (wasm32 survey)

| Need | Crate | wasm32 status | Tag |
|---|---|---|---|
| HNSW | `crates/rvf/rvf-index` | **Not usable for a quantised in-memory index as is**: traversal reads `&[f32]` per visited node (`hnsw.rs:183-189`), `BTreeMap` adjacency (`hnsw.rs:98`), dense visited bitmap per search, positional codec (`codec.rs:243-296`), `IndexSegHeader` lacks `entry_point`/`max_layer`/`m0`/`alpha`; no wasm SIMD. | [V] |
| Exact baseline | `ruvector-core --no-default-features --features memory-only` | `FlatIndex` only; needs `getrandom` `js` on wasm32. | [V] |
| int8 codes | `crates/rvf/rvf-quant` | `cargo check` passes for wasm32; `ScalarQuantizer::train` derives per-dim min/max from a training set; `encode_vec` clamps. | [V] |
| RaBitQ | `ruvector-rabitq` | Seeded RNG; rayon compiled out on wasm32; `load_index` rebuilds a D×D rotation per load. | [V] |
| Cypher-lite | `rvlite::cypher` | Pure Rust, but `rvlite` links wasm-bindgen, web-sys and IndexedDB. | [V] |
| Min-cut | `ruvector-mincut` (`wasm,exact,approximate`) | `DynamicGraph` uses DashMap, not serialisable; rayon non-optional, single-thread fallback untested. | [V] / **[U]** |
| RVF codecs | `rvf-wire`, `rvf-types` (serde off) | `cargo check` passes; readers work on whole in-memory segments. `rvf-wasm` (global dlmalloc) and `rvf-runtime` (paths, `SystemTime`) cannot be linked. | [V] |
| Rejected | `micro-hnsw-wasm`, `ruvector-router-core` (redb/mmap), `ruvector-embed-core` wasm (tract-onnx too large) | — | [V] |

Build hazards [V]: the shell exports `RUSTFLAGS='-C link-arg=-fuse-ld=mold'`, which breaks every
wasm32 link (build with `env -u RUSTFLAGS`); getrandom 0.2 and 0.3 are both in the tree and each
needs a wasm32 backend.

## 2. Decision

Build ruvector edge as **OAuth 2.1 resource servers behind our own edge authorization server**
`ruvector-edge-auth`, which federates login to the unchanged `auth.cognitum.one` and mints
resource-bound tokens (rUv, 2026-09-29; the upstream AS mints `aud = client_id` and ignores RFC
8707, §1.1).

1. **Rust throughout.** Pure cores without `worker`/`wasm-bindgen` (`ruvector-edge-{auth,authz,
   tenancy}`, `-store` from M1), tested natively with London-school mocks, in thin workers-rs cdylibs.
2. **Edge AS.** `ruvector-edge-auth` is an ordinary OAuth client of `auth.cognitum.one` (code + PKCE
   S256, identity scopes only; upstream tokens verified, then discarded) and runs its own DCR,
   authorize, token and revoke endpoints, minting **its own ES256 tokens** with `aud` = the exact
   canonical resource URL (§5.2).
3. **Resource servers accept only edge tokens with the exact `aud`** — no prefix matching, no
   `dcr-*`, no second audience on any route (including `/v1/ops`: adapters obtain a `/v1` token by
   RFC 8693 exchange, §5.6). Upstream first-party tokens are an **optional mode, off by default**,
   REST only (§5.5).
4. **Browser login from M0.5, ChatGPT/Claude connectors end to end from M1** (not M6): the edge AS
   accepts https and loopback redirects under its own DCR policy, connectors can step up to
   `ruvector:write` on consent (§5.3), and tokens bound to `/v1/mcp` make mutating tools safe
   without provider changes.
5. **One SQLite Durable Object per `(tenant, service, collection uid, shard)`**, memory budgeted
   **per isolate** (§6.1, §10).
6. **Tenant from verified edge claims** `(upstream_iss, org_id, workspace_id)` (§4.1); no tenant
   parameter anywhere.
7. **Capability = route ∩ scope ∩ membership role** (§5.3), from a closed ruvector scope vocabulary.
8. **Compiled-in trust roots** in release builds for both Workers — edge and upstream issuers,
   JWKS URLs, upstream endpoints, public origin (§5.4, §5.6); the gateway's are compiled, the auth Worker's are still vars (§8 delta 6).
9. **Exactly one provider touch:** one exact upstream DCR redirect-allowlist entry for the edge
   callback (G1, [V] §1.1). Team tenancy still needs an upstream org selector or role claim (G5).
10. **Incremental delivery** (§15): M0 cores; **M0.5 both Workers live on workers.dev**; M1 flat
    scan + MCP; M2a int8; M2b HNSW (built in-repo, `ruvector-edge-index`); M3 durability; M4–M5 the wider family; M6 GA.

## 3. Services

| Service | Worker / routes | Backing crate(s) | Durable Object | Capability source | Milestone |
|---|---|---|---|---|---|
| **ruvector-edge-auth** (edge AS) | Worker `ruvector-edge-auth`: `/.well-known/oauth-authorization-server`, `/.well-known/jwks.json`, `/register`, `/authorize`, `/callback`, `/token`, `/revoke` | `ruvector-edge-authz` (core), `ruvector-edge-auth` (`ResourceUrl`, JWKS, upstream verify); glue `edge/auth-worker` | `AuthStore` (one global instance, §5.6) | — (it issues capabilities) | M0 core, M0.5 live |
| **rv-gateway** (public resource entrypoint) | Worker `ruvector-edge-gateway`: `/.well-known/oauth-protected-resource/v1[/mcp]`, `/v1/health`, `/v1/me`, `/v1/usage`, `/v1/tenant*`, dispatch | `ruvector-edge-auth`, `ruvector-edge-tenancy`; glue `edge/gateway` | — | none of its own | M0.5 (PRM, `/v1/me`, `/v1/mcp` 401 challenge), M1 |
| **rv-vector** | `/v1/collections*`, `index=flat\|hnsw` (`q8` not a kind) | `ruvector-edge-store` + `ruvector-edge-index`: `flat` (default) = int8 codes + f32 rerank from SQLite; `hnsw` opt-in over the same codes, in-repo (G4a not needed) [V] 52e2a5c55 | `VectorShard` | scope ∩ role | M1/M2a/M2b |
| **TenantLedger** | internal control plane for one tenant | plain Rust + serde | `TenantLedger` (one per tenant) | — | M1 |
| **rv-snapshot** | `POST/GET …/snapshots`, `POST …/snapshots/{id}:restore`, `POST …:export` → `GET /v1/exports/{id}` | `ruvector-edge-snapshot` (streaming RVF segments + witness chain) | `/m3` side channel of `VectorShard` + `TenantLedger` (no new class) | export → read; snapshot → write; restore → admin | M3 (sync, ≤ 2^18 floats **[Paid]**; not for `rabitq`) |
| **rv-embed** | `text` on upsert/query | Workers AI `bge-small-en-v1.5` (384-dim) | — | inherits route | M3 |
| **rv-ingest** | bulk upsert, `POST /v1/uploads` (+ parts, `:complete`), `…:import`, `GET /v1/jobs/{id}` | Queue `ruvector-edge-ingest` (+ DLQ) + R2 `staging/` (server-minted `upload_id`); inline ≤ 512 KiB **[Paid]** | — | submit → write; status → read | M3 |
| **rv-quant** | `index=rabitq` | `ruvector-edge-quant` over `ruvector-rabitq` with its own persist v2 (`rbqx0002`, no f32 re-encode on load; G4c done in-repo) | `QuantShard` | as rv-vector; no `filter`, no per-row `ops` log | M4 (≤ 50k rows/shard **[Paid]**) |
| **rv-graph** | `POST/GET /v1/graphs`, `GET/DELETE /v1/graphs/{g}`, `POST …/cypher`, `POST …/edges` | `rvlite::cypher` `PropertyGraph`, `rvlite` pinned to git **rev `c6ece785`** (head of `feat/rvlite-browser-feature-gate`, open PR RuVector#1084) with `default-features = false` (G4b realised as an unmerged upstream branch; pinned by rev so a squash-merge + branch deletion cannot break clean builds; move to `main`/a release after #1084 merges; `edge/Cargo.toml:47`) | `GraphStore` (one per graph + `$catalog`) | MATCH/RETURN/WITH → read; other → write + editor | M4 (state ≤ 256 KiB **[Paid]**) |
| **rv-mincut** | `POST /v1/mincut`, `GET /v1/mincut/jobs/{id}` | `ruvector-edge-analytics` over `ruvector-mincut` (exact solver; approximate answered exactly) | `AnalyticsJob` | inline compute → read; job → write + one write token | M4 |
| **rv-registry** | `/v1/rvf/scopes[/{scope}]`, `/v1/rvf/{scope}/{name}[/{version}[/blob\|/uploads…\|:publish\|:yank\|:unyank]]`, `POST /v1/collections/{c}:import-rvf` | `ruvector-edge-registry` (strict names, streaming `rvf-wire` validator) + R2 `EDGE_DATA` | `RegistryRoot` (global scope directory) + `RegistryScope` (one per scope) | pull → read; push → write + owned scope; scope claim → admin; public publish → `ruvector:publish` ∩ owner | M5 (writes **[needs Paid]**) |
| **rv-mcp** | `POST /v1/mcp` (Streamable HTTP, JSON-RPC 2.0; protocol 2025-11-25 / 2025-06-18) | Rust MCP framing calling the REST handlers | — | scope ∩ role; every tool carries `readOnlyHint`/`destructiveHint`, `destructiveHint: true` only on `vector_upsert`, `vector_delete`, `graph_mutate` and `rvf_import` (`collection_create` and `tenant_claim` mutate but are not destructive; `mcp.rs:106-135`, `rvf_mcp.rs:63-64`); `dry_run` on `collection_create`, `vector_upsert`/`vector_delete` and `graph_mutate` only (not `tenant_claim`, which refuses it; §7.2) | M1 (vector tools), M4 graph/mincut, M5 rvf |
| **ops** (private contract) | `POST /v1/ops` on the gateway, for Service-Binding callers holding a `/v1` token (exchanged, §5.6) | same handlers as REST | — | scope ∩ role (§16.3) | M1 |

"read", "write", "create", "admin", "publish" are internal **capabilities**; §5.3 defines how they
are derived.

## 4. Tenancy model

### 4.1 Deriving the tenant (after signature verification only)

The inputs are the edge-token claims defined in §5.2: `upstream_iss`, `org_id`, `workspace_id`,
`sub`, `family_id`, `jti`, `client_id`.

1. Require `upstream_iss` to equal the compiled-in upstream issuer `https://auth.cognitum.one`, and
   `org_id`, `workspace_id` to be strings matching `^[A-Za-z0-9_-]{1,64}$` (upstream values are
   UUIDs [L]). Otherwise `401 invalid_token`.
2. `tenant_key = base32lower(sha256("v1|" + upstream_iss + "|" + org_id + "|" +
   workspace_id))[0..26]`. Keying on the **upstream** issuer gives the §5.5 mode the **same** tenant
   and keeps a second IdP from colliding; hashing keeps org UUIDs out of DO names, R2 keys and logs.
3. **What the tenant really is.** Upstream picks the org and default workspace with no selector or
   membership check (§1.1) [V], carried forward verbatim: for a self-signup user the tenant is **one
   user**; a shared SSO first org maps several people to one tenant, hence §4.2 is default-deny.
   Nobody can switch org or workspace; team tenancy needs G5.
4. `sub` (the edge subject, §5.2) is the actor for audit rows, memberships and per-user rate limits.
5. Never tenant keys: `client_id`/`aud` (identify the calling app and resource), `family_id`
   (logged, deny-listable), `jti`.

`TenantContext { tenant_key, upstream_iss, org_id, workspace_id, sub, client_id, family_id, jti,
token_kind, scopes, role }` is built once per request. Its only constructor is
`TenantContext::from_verified(VerifiedClaims, Option<Role>, RouteSurface)` [V] tree (scope ∩ role,
edge-subject shape checked, upstream-first-party refused off REST); no storage API accepts a raw
tenant string.

### 4.2 Membership (default-deny, stored in `TenantLedger`)

`TenantLedger` holds `memberships(sub PRIMARY KEY, role ∈ {owner, editor, viewer}, invited_by,
created_at)` for its own tenant only; there is no shared membership table.

- **Explicit claim.** `POST /v1/tenant:claim` or the MCP tool `tenant_claim` (scope `ruvector:write` or
  `ruvector:admin`, so a connector-only user can bootstrap) is one coalesced write
  (`INSERT … WHERE NOT EXISTS (SELECT 1 FROM memberships)`); the winner becomes `owner`. A claim
  grants authority only over data created afterwards, so in a shared SSO org the first claimant owns
  an empty container.
- **Everyone else gets `403 role_required`** until the owner invites their `sub` (`POST
  /v1/tenant/members {sub, role}`; default `viewer`). A user reads their own `sub` from `/v1/me`.
  Invitation is by `sub` only (tokens carry no email).
- **Removal** is `DELETE /v1/tenant/members/{sub}` plus a tenant `sub` deny entry (§5.8); upstream
  workspace removal does nothing here.
- **Ownership transfer / recovery** is an audited operator break-glass action.
- Roles: `viewer` → read; `editor` → read, write, create; `owner` → all of those plus admin
  (members, deny entries, client blocks, restore, audit), **dropping collections** and publish.

### 4.3 Durable Object naming and in-object assertion

- Collection and graph names match `^[a-z0-9][a-z0-9_-]{0,62}$`; vector ids are 1–256 bytes of UTF-8
  with no control characters.
- Each collection gets a **`collection_uid`** = `sha256("v2|collection_uid|" ‖ salt[32] ‖ seq_be[8]
  ‖ nonce[16])[0..16]` (ledger salt + seq + a fresh CSPRNG nonce, so a ledger rollback cannot
  reproduce a deleted uid; collisions with any catalog row are retried) [V] tree `uid.rs`.
  Recreating a deleted name gets a new uid, so a new DO and new R2 keys. **Shipped exception
  ([V] 52e2a5c55):** graphs and min-cut jobs are not in the ledger catalog; their uid is
  `sha256("v1|graph|" + name)[0..16]` / `sha256("v1|mincut-job|" + job_id)[0..16]` with shard 0,
  plus one per-tenant catalog `GraphStore` named for `$catalog` (`graph_wire.rs:15,26-50`). A
  recreated graph name therefore reuses its DO, which `DELETE /v1/graphs/{g}` wipes first. The
  registry is not tenant-named at all (§3, Shipped-vs-designed #5).
- DO name = `hex(sha256("v1|" + tenant_key + "|" + service + "|" + collection_uid + "|" + shard))`
  with `collection_uid` as **32 lowercase hex chars** and `shard` as a **decimal** index (0..5), via
  `idFromName`; the ledger is `hex(sha256("v1|" + tenant_key + "|ledger"))`. Never `newUniqueId`.
  [V] tree `do_name(&TenantKey, Service, &CollectionUid, ShardIndex)`, `ledger_do_name`.
- Each DO stores `(tenant_key, service, collection_uid, shard)` in `meta` on first write and checks
  the caller's `TenantContext` on **every** call; a mismatch returns 404. DOs never call another
  tenant's DOs.
- **Foreign-tenant resources return `404 not_found`**, indistinguishable from absent ones.

### 4.4 Sharding

A collection starts with `shard_count = 1`; splitting is a background copy-then-cutover task driven
by alarms and the Queue, recorded in the ledger. Ids route by `hash(id) mod N`; queries fan out and
the gateway merges top-k. **`shard_count ≤ 6` through M3** (6 outgoing connections per request);
more needs a two-level merge (M4+). Rate and admission are charged by `shards_queried`. A shard
splits when its resident cap is exceeded, its SQLite nears 1 GB, or it sustains > ~500 req/s
**[U]**.

## 5. Authentication and authorization

### 5.1 Topology and login flow

```
client (CLI / browser app / ChatGPT / Claude)
  │ 1. POST gateway /v1/mcp (no token) → 401 WWW-Authenticate: Bearer
  │         resource_metadata=…/oauth-protected-resource/v1/mcp, scope="ruvector:read offline_access"
  │ 2. GET PRM              → authorization_servers = [EDGE_ISSUER]
  │ 3. GET EDGE_ISSUER/.well-known/oauth-authorization-server
  │ 4. POST /register (RFC 7591, public client)
  │ 5. GET /authorize?resource=<canonical URL>&code_challenge=…(S256)&scope&state
  ▼
ruvector-edge-auth ── stores flow, renders consent page, sets __Host-eaf-<state'[0..16]> cookie
  │ 6. user clicks Continue → auth.cognitum.one/oauth/authorize (our upstream client,
  │         PKCE S256, state', nonce, scope "openid profile email")
  │ 7. user logs in upstream → 302 /callback?code&state'
  │ 8. check cookie + one-time flow; exchange code (server side); verify upstream
  │    access token (upstream JWKS); revoke + discard upstream tokens
  │ 9. 302 redirect_uri?code&state&iss   (code: 60 s, one-time, bound)
  │10. POST /token → edge access token (aud = resource, 15 min) + refresh token
  ▼
gateway verifies: iss = EDGE_ISSUER, typ at+jwt, ES256 via edge JWKS, aud == its own resource
```

The first-party CLI uses the same flow with a loopback DCR registration at the **edge** AS; it no
longer needs its own upstream client. Clients that only implement MCP 2025-03-26 discovery (AS
metadata at the MCP server's origin, default `/authorize`) are unsupported: the gateway answers 404
for `/.well-known/oauth-authorization-server` and `/.well-known/openid-configuration`.

### 5.2 The edge access token

JOSE header: `{"alg":"ES256","typ":"at+jwt","kid":"<RFC 7638 thumbprint>"}` (RFC 9068). Payload —
every claim below is **required** and typed, except `act`, which appears only on exchanged tokens
(§5.6); unknown claims are ignored by verifiers:

| Claim | Type | Value |
|---|---|---|
| `iss` | string | exactly `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev` |
| `aud` | string (never an array) | exactly one allowlisted canonical resource URL: `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1` or `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp`, or an adapter resource added per §16.1 (verified only by that adapter, never accepted by the gateway) |
| `sub` | string | **edge subject** `"es1_" + base32lower(sha256("ruvector-edge/sub/v1\|" + upstream_iss + "\|" + upstream_sub))[0..26]` (30 chars). Stable across clients, sessions, refreshes and key rotation; changes only if the upstream `sub` does. One function, `ruvector-edge-auth` `subject::edge_subject` [V] tree, used by the AS and by the §5.5 mode. |
| `client_id` | string | the edge DCR client id (`edc-` + random) |
| `scope` | string | space-delimited subset of the `aud` resource's scopes (§5.3: `ruvector:*` for the gateway resources, `team:*` for team.ruv.io; plus `offline_access`) |
| `jti` | string | 16 random bytes, base64url |
| `iat` | integer | issue time |
| `exp` | integer | `iat + 900` (15 min) |
| `family_id` | string | grant id: one per code redemption, shared by every refresh rotation of that grant (also for clients without a refresh grant); 16 random bytes, base64url |
| `upstream_iss` | string | `https://auth.cognitum.one` — the tenant namespace (§4.1) |
| `org_id` | string | upstream `org_id`, verbatim |
| `workspace_id` | string | upstream `workspace_id`, verbatim |
| `act` | object, optional | exchanged tokens only: `{"sub": "<adapter client_id>"}` (RFC 8693 §4.1) |

Not carried: the raw upstream `sub` (kept only in the AS `subjects` table), email, `account_id`,
upstream `jti`/`family_id`, and any `typ` claim. ([V] tree: `AccessTokenClaims` has exactly the
required set above, `sub` minted by `subject.rs` `edge_subject`; `act` arrives with the exchange
grant, M1.)

### 5.3 Scope vocabulary and capability derivation

The edge AS mints **only** the scopes below, and **each resource has its own vocabulary**: every
`RESOURCE_ALLOWLIST` entry declares its scopes from exactly one family plus `offline_access` —
`ruvector:*` for the two gateway resources, `team:*` for the RuFlo AI Team adapter resource
`https://team.ruv.io/mcp` (§16.1) — and a mixed or unknown entry fails the config load [V] tree
(`authz/resource.rs` `Vocabulary`, `ResourceEntry::new`). Which family a URL may carry is
**compiled** (`resource::RESOURCE_VOCABULARIES`: `…/v1` and `…/v1/mcp` → `ruvector:*`,
`https://team.ruv.io/mcp` → `team:*`) and enforced at load [V] tree: an uncompiled URL,
`ruvector:*` for team.ruv.io or `team:*` for a gateway resource fails the load, so the mutable
`RESOURCE_ALLOWLIST` var can narrow a resource's scopes but can neither add a resource nor mint an
adapter-audience token carrying gateway scope names. The AS `scopes_supported` (RFC 8414 and
the DCR registrable set) is the **union** of every resource's scopes, derived from the allowlist.
`openid`, `profile` and `email` in a registration or authorization request are accepted and
**dropped** (the edge AS issues no ID token); any other unknown scope is `invalid_scope`.

| Scope | Grants (always ∩ role) | Role needed | Registrable by default (DCR without `scope`) | Granted by default (`/authorize` without `scope`) | Resource (PRM `scopes_supported`) |
|---|---|---|---|---|---|
| `ruvector:read` | read: list/get collections, query, fetch, usage, MCP read tools | viewer | yes | yes | `/v1`, `/v1/mcp` |
| `ruvector:write` | write + create: upsert/delete vectors, create collections (drop: owner only), snapshots/imports, MCP mutating tools | editor (drop: owner) | yes | **no** — granted only when requested and shown on consent | `/v1`, `/v1/mcp` |
| `ruvector:admin` | admin: members, tenant deny entries, client blocks, restore, audit (and `tenant:claim`, like `ruvector:write`) | owner (claim: unclaimed tenant) | no (request at DCR) | no | `/v1` only |
| `ruvector:publish` | public registry publish | owner | no; not minted before M5 | no | `/v1` from M5 |
| `team:read` | defined by the team.ruv.io adapter (read its MCP tools); nothing on the gateway | adapter-defined | **no** (request at DCR) | yes, on team.ruv.io | team.ruv.io only |
| `team:write` | adapter-defined (mutating team tools) | adapter-defined | no (request at DCR) | no | team.ruv.io only |
| `team:run` | adapter-defined (run team agents/jobs) | adapter-defined | no (request at DCR) | no | team.ruv.io only |
| `offline_access` | accepted and echoed; refresh tokens follow the client's registered `refresh_token` grant, not this scope | — | yes | yes | all |

- **Grant rule.** At `/authorize` the grant is requested ∩ client ceiling ∩ the resource's scopes
  [V] tree (`resource::grant_scopes`), applied in this order: (1) a scope of **no** vocabulary →
  `invalid_scope` ("unknown scope requested"); (2) a scope of **another resource's** vocabulary —
  `ruvector:*` for team.ruv.io, `team:*` for `/v1` or `/v1/mcp` — is **dropped** (never minted:
  the compiled binding keeps it out of every entry of the other family), so a client that requests
  the AS-metadata `scopes_supported` union still gets its resource's share (Q16(b), closed);
  (3) a scope of this resource's vocabulary that the resource or the client ceiling does not offer
  (`ruvector:admin` for `/v1/mcp`, `ruvector:publish` before M5, `team:run` outside the ceiling)
  is **dropped**, not refused, and the token response `scope` reports what was granted (RFC 6749
  §3.3); (4) a request left with nothing but `offline_access` is `invalid_scope` ("no requested
  scope can be granted for this resource"), e.g. `ruvector:read` alone for team.ruv.io. An omitted `scope` asks for the
  resource's first scope plus `offline_access` (`ruvector:read`, or `team:read` on team.ruv.io).
  Refresh may only narrow within the family's granted scopes, so it can never cross vocabularies.
  Consent plus role, not the DCR ceiling, is the control.
- **Client ceiling.** A DCR registration without `scope` gets `ruvector:read ruvector:write
  offline_access` (compiled `DEFAULT_CLIENT_SCOPE`), i.e. the gateway only; `ruvector:admin`,
  `ruvector:publish` and every `team:*` scope must be registered explicitly. A registration naming
  scopes of several vocabularies gets their **union** as its ceiling (e.g. `ruvector:read team:read`
  may authorise both `/v1/mcp` and team.ruv.io, one resource per grant). Consequence: a connector
  that registers without `scope` cannot obtain a team.ruv.io token (Q16).
- **Step-up.** A mutating call without `ruvector:write` gets HTTP **403** `WWW-Authenticate: Bearer
  error="insufficient_scope", scope="ruvector:read ruvector:write offline_access",
  resource_metadata="…"` — also for `tools/call` on `/v1/mcp` (an HTTP status, not a JSON-RPC
  error), so MCP clients re-authorize with the wider scope.
- **Capability = route requirement ∩ scope ∩ role.** One versioned default-deny table in
  `ruvector-edge-auth` (`scopes.rs` `ROUTE_TABLE`) has rows `(method, pattern, capability,
  min_role ∈ {viewer, editor, owner, unclaimed})`, with per-op/per-tool rows for `/v1/ops` and
  `/v1/mcp`; a unit test asserts every route is covered and unknown routes are denied. A missing
  role → `403 role_required`/`not_claimed`. **As shipped ([V] 52e2a5c55)** the vocabulary is
  `ruvector:*`, but `ROUTE_TABLE` keeps M1 rows with no role column and is **not consulted at
  runtime**; each handler authorizes instead (`ops::authz::authorize(op, scope caps, role,
  claimed)` for REST/ops/MCP ops, `need_owner` for drop, graph `Need::{Read, Write}`, registry
  caps = scope ∩ role). The single-table gap test is therefore owed (§8 delta 1).
- **`ruvector:publish` is not live:** it is in the scope table but absent from the deployed
  `RESOURCE_ALLOWLIST` and the `/v1` PRM (`prm.rs:13-21`), so no token carries it and
  `…:publish` is unreachable until the operator adds it (reviewed AS config, M5).
- `/v1/me` needs any valid token and returns only token- and ledger-derived fields. Upstream
  product scopes (`mcp:*`, `swarm:*`, `console`, `brains:*`, …) are never minted and grant nothing.

### 5.4 Resource-server verification (`ruvector-edge-auth`, no JS)

1. **Transport.** Only `Authorization: Bearer`. **Pre-auth throttle** before signature work: Rate
   Limiting `ip:{hash(cf-connecting-ip)}` and a 60 s in-isolate cache of `sha256(token)` failures.
2. **Shape.** ≤ 8 KiB, exactly 3 segments, strict base64url.
3. **Header.** `alg` exactly `ES256` (`none`, `HS*`, `RS*` rejected before key lookup); `kid`
   required; header `typ` **`at+jwt`** for edge tokens (absent or `JWT` only in the §5.5 mode);
   `jku`, `x5u`, `x5c`, `jwk` rejected. `AudiencePolicy::classify` picks the kind from `iss` +
   header `typ` [V] tree.
4. **Signature.** `p256` over fixed 64-byte `r‖s`; DER rejected.
5. **Trust root (compile-time).** In release builds `PUBLIC_ORIGIN` (which fixes both audiences),
   `EDGE_ISSUER`, `EDGE_JWKS_URL = EDGE_ISSUER + "/.well-known/jwks.json"`, `UPSTREAM_ISSUER`,
   `UPSTREAM_JWKS_URL` and `FIRST_PARTY_AUDS` are Rust `const`s; wrangler `vars` supply them **only**
   under `#[cfg(feature = "dev-issuer")]`, and a release build that sees contradicting vars answers
   **503 `trust_root_mismatch`**. Edge `kid`s are **not** pinned — both sides are ours and pinning
   would only block rotation (§5.6) — but each JWK's RFC 7638 thumbprint must equal its `kid`. The
   edge JWKS is fetched through the `EDGE_AUTH` Service Binding [V] tree, because same-account
   workers.dev → workers.dev subrequests are refused (Cloudflare 1042) [L]. ([V] 52e2a5c55: all six are compiled
   consts in `edge/gateway/src/trust_root.rs`, with `FIRST_PARTY_AUDS` and
   `ACCEPTED_UPSTREAM_KIDS` empty; §8 delta 6 is closed for the gateway.)
6. **JWKS.** `kty=EC`, `crv=P-256` only. In-isolate TTL map (10 min) is the primary cache (the Cache
   API may no-op on workers.dev); every successful fetch **replaces** the set (a removed `kid` stops
   verifying within ≤ 10 min); an unknown `kid` triggers at most one single-flight refetch per 30 s
   per isolate; on fetch failure the last good set serves up to 24 h; never fetched → `503`.
7. **Claims.** Every §5.2 claim present and typed; `iss == EDGE_ISSUER`; `aud` a string **byte-equal
   to this resource's canonical URL**, on every route including `/v1/ops`. Any other `aud` (the
   other gateway resource, an adapter resource, an array) → **401** `WWW-Authenticate: Bearer
   error="invalid_token", error_description="audience mismatch", resource_metadata="<this route's
   PRM>"` so MCP clients re-run discovery; `audience_not_allowed` is only the log/metric reason
   (closed: 401 as shipped, §8 delta 10). `exp > now − 60 s`, `iat ≤ now + 60 s`, **`exp − iat ≤ 900`**; `nbf`
   honoured if present; `upstream_iss` equals the compiled upstream issuer; deny-list (§5.8) on
   `jti`, `family_id`, `sub`, `client_id`, `kid`.
8. Clock and network only through the `Clock` and `HttpFetch`/`KeySource` traits; the crate never
   calls `std::time`.

### 5.5 Optional upstream-first-party mode (flagged, off by default)

For a first-party CLI that already holds `auth.cognitum.one` tokens. Enabled only by the var
`UPSTREAM_FIRST_PARTY = "true"` (default `"false"`) [V] tree; `FIRST_PARTY_AUDS` is a compiled const
like the other trust roots (§5.4 item 5; compiled, empty, [V] 52e2a5c55), so a var edit cannot widen it.
When on, and **only on `/v1` REST, never `/v1/mcp` or `/v1/ops`**:

- `iss == https://auth.cognitum.one`, `kid ∈` compiled `ACCEPTED_UPSTREAM_KIDS` (today exactly
  `_jQ62WD8…`; upstream rotation needs a redeploy; as shipped the compiled const is **empty**, so the mode
  fails closed even with the var set to `"true"`), header `typ` absent or `JWT`, **claim** `typ == "access"`
  (rejects ID and `inference` tokens), `exp − iat ≤ 3600`, `setup`/`workload`/`exchanged` rejected;
- `aud ∈ FIRST_PARTY_AUDS`, exact ids only — never a `dcr-` prefix match;
- the verifier normalises `sub` with `edge_subject` and sets `upstream_iss = iss`, so tenant and
  actor match the edge path; capabilities are the fixed set `read write admin` ∩ role (upstream
  tokens carry no ruvector scopes).

Such a client is registered at the upstream DCR with a loopback redirect (allowed today [V]) and
**identity scopes only**. Nothing in M0.5–M5 depends on this mode.

### 5.6 The edge authorization server (`ruvector-edge-auth`)

**Configuration.** `ISSUER` and the upstream issuer, JWKS URL, authorize and token endpoints are
release consts, as in §5.4 item 5 (still `[vars]` in the auth Worker at 52e2a5c55, §8 delta 6); `UPSTREAM_CLIENT_ID`,
`RESOURCE_ALLOWLIST` and `MAX_CLIENTS` stay reviewed deploy config (`RESOURCE_ALLOWLIST` only
within the compiled URL → vocabulary bindings, §5.3).

**Storage.** One global `AuthStore` DO (`idFromName("auth-store-v1")`): `meta`, `clients`, `codes`,
`flows`, `refresh_tokens`, `refresh_families` [V] tree, plus `subjects(sub, upstream_iss,
upstream_sub, first_seen, last_seen)`, `dcr_rate` and redeemed-code tombstones (§8 delta 3: `redeemed_codes`, `dcr_rate`, `client_activity`,
`assertion_jtis` ship; `subjects` does not). Codes
and refresh tokens are stored hashed; one-time takes are one `DELETE … RETURNING`. Metadata and
JWKS need no DO hop; only login, token and DCR traffic reaches it (~500–1,000 req/s [V]).

**DCR (`POST /register`, RFC 7591)** [V] tree `client.rs` contract: public clients (`none`);
`response_types ["code"]`; `grant_types` ⊆ {`authorization_code`, `refresh_token`} incl. the first,
**omitted → both** (deliberately not the RFC 7591 default, so connectors get refresh tokens);
1–8 redirect URIs (≤ 512 bytes, no fragment/userinfo): `https` with a host, or `http` on
`127.0.0.1`, `[::1]` or **`localhost`**, port-agnostic with path equality (RFC 8252 §7.3/§8.3;
Claude Code and MCP Inspector use `http://localhost:<port>/…`); `scope` ⊆ the union of the §5.3
vocabularies, **omitted → `ruvector:read ruvector:write offline_access`** (admin/publish and any
`team:*` scope only if requested here; the ceiling is the union of what was requested, §5.3);
`client_name` ≤ 128 chars; body ≤ 16 KiB; ids `edc-` + random; cap `MAX_CLIENTS` (10,000) [V] tree;
per-IP rate limit and 30-day idle expiry **[U]** (M0.5). Today: §8 delta 4 — **`localhost` is
still refused** ([V] 52e2a5c55 `client.rs:166-172`), so loopback clients must register
`http://127.0.0.1:<port>/…` until it closes. Connector-prefix
(`chatgpt.com/connector/oauth/`, `chatgpt.com/aip/`, `claude.ai/api/mcp/`, `claude.com/api/mcp/`)
and loopback redirects show as **verified** on consent, others **unverified** (shipped, `redirect_is_verified`). AS
metadata omits `client_id_metadata_document_supported`, so MCP 2025-11-25 clients use DCR (CIMD:
Q15). **Adapter clients** (M1) are operator-registered **confidential** clients (`private_key_jwt`,
RFC 7523, per-adapter JWK), allowed only the exchange grant to `…/v1`.

**Authorization (`GET /authorize`)** [V] tree `endpoints/{authorize,consent}.rs`:

- Until `client_id` and `redirect_uri` are validated, errors are shown, **never redirected**; after
  that they redirect with `error`, `state` and `iss`.
- `response_type=code`; PKCE `S256` only (43-char challenge) [V] tree `pkce.rs`; `state` ≤ 512;
  scope per the §5.3 grant rule (§8 delta 4).
- **`resource` is required**, once, byte-equal to a canonical `RESOURCE_ALLOWLIST` entry, else
  `invalid_target`; no default audience; one resource per grant (`/v1` and `/v1/mcp` authorise
  separately).
- A valid request **stores the flow** (upstream `state`, `nonce`, PKCE verifier, sha256 of a fresh
  browser secret; `FLOW_TTL_SECS` 600) and renders the **consent page** (client name, redirect
  host, resource, scopes; for team.ruv.io, what each granted `team:read`/`team:write` also permits
  on the user's ruvector data through the compiled exchange map, §16.1), which
  sets `__Host-eaf-{state[0..16]}` = the browser secret (`Secure;
  HttpOnly; SameSite=Lax; Path=/; Max-Age=600`; one cookie per flow, so parallel logins do not
  collide). **Continue** links to the upstream authorize URL; Cancel returns `access_denied`. AS
  HTML responses send `X-Frame-Options: DENY`, CSP `default-src 'none'; frame-ancestors 'none';
  form-action 'none'; base-uri 'none'` and `Referrer-Policy: no-referrer` (clickjacking, RFC 9700
  §4.16; no Referer leak). Consent is never skipped.

**Upstream leg.** Client `UPSTREAM_CLIENT_ID` at `auth.cognitum.one` (G1), redirect `<ISSUER>/callback`,
scope **`openid profile email` only** — never `mcp:*` (§1.1; enforced at load as shipped).

**Callback (`GET /callback`).**

1. `take_flow(state)` is one-time (missing/expired → `access_denied`); `sha256` of the
   `__Host-eaf-{state[0..16]}` cookie must equal the stored binding (constant time), defeating
   upstream-code injection and login CSRF. An RFC 9207 `iss` parameter, if sent, must equal the
   upstream issuer (**[U]** whether it is).
2. Exchange the upstream code server-side (form-encoded, PKCE verifier).
3. Verify the upstream **access token** against the compiled upstream JWKS URL, accepting any P-256
   key whose RFC 7638 thumbprint equals its `kid` (unknown `kid` → one single-flight refetch under
   the §5.4 limits; a new `kid` is alerted, not refused — no pinning on this path); `iss ==
   https://auth.cognitum.one`, `aud == UPSTREAM_CLIENT_ID`, claim `typ == "access"`, `sub`,
   `org_id`, `workspace_id` present, flags false. The id_token is ignored; `nonce` is compared only
   if the access token carries one [V] tree `federation.rs`, `upstream.rs`. **Shipped ([V]
   52e2a5c55), stricter:** the upstream `kid` must be in the pinned var `ACCEPTED_UPSTREAM_KIDS`
   (empty until G1 ⇒ `temporarily_unavailable`), so an upstream rotation needs a redeploy; and an
   id_token, **if returned**, is verified (same pinned keys, `iss`, `aud == UPSTREAM_CLIENT_ID`,
   expiry) and its `nonce` must equal the flow's (`upstream.rs:138-216`, `callback.rs:122-130`).
4. Revoke the upstream refresh token (best-effort) and **discard** every upstream token — never
   stored, logged or forwarded (shipped: `callback.rs:98-100`).
5. Upsert `subjects`; issue the edge code (32 random bytes, 60 s, hashed, bound to `client_id`,
   `redirect_uri`, `resource`, PKCE challenge, scopes, identity); redirect
   `redirect_uri?code&state&iss` [V] tree contract.

**Token (`POST /token`, form-encoded).** Repeated parameters → `invalid_request`; `client_secret` →
`invalid_client`.

- `authorization_code`: `take_code` is one-time and **consumed even if a later check fails**;
  `client_id`, `redirect_uri` equal the stored values, `resource` (if sent) the bound one, PKCE S256
  verifies (constant time); starts a new `family_id`. A redeemed code leaves a `(code_hash →
  family_id)` tombstone for TTL + 600 s; a replay gets `invalid_grant` and **revokes that family**
  [V] tree `code.rs` (RFC 6749 §4.1.2).
- `refresh_token` (whenever the client registered that grant): rotated on every use, bound to
  `client_id` and `resource`, `scope` may only narrow. A rotated token re-presented **within
  `REFRESH_REUSE_GRACE_SECS = 30`** by the same client and resource (concurrent refresh, retry)
  gets `invalid_grant` without revocation; after the window it **revokes the family** (RFC 9700
  §4.14.2; §8 delta 11 — **open**: shipped code has no grace window, so any re-presentation of a
  rotated token, including a concurrent refresh, revokes the family, [V] 52e2a5c55
  `refresh.rs:143-150`). `REFRESH_TTL_SECS` 30 d per token, `FAMILY_MAX_LIFETIME_SECS` 90 d per
  family [V] tree. Refresh re-mints from the stored identity, so upstream membership changes and
  deprovisioning are seen only at the next login (§12).
- `urn:ietf:params:oauth:grant-type:token-exchange` (RFC 8693, M1, adapter clients only):
  `subject_token` = the user's edge token for that adapter's own resource, `resource = …/v1`. The
  result keeps `sub`, `upstream_iss`, `org_id`, `workspace_id`, `family_id`; `exp ≤` the subject
  token's; `scope` = the adapter resource's compiled **exchange map** applied to the subject
  token's scopes, ∩ the adapter client's ceiling ∩ the `…/v1` scopes (an adapter vocabulary never
  appears on a `…/v1` token; team.ruv.io: `team:read → ruvector:read`, `team:write →
  ruvector:write`, `team:run → ∅`; nothing mapped → `invalid_scope`; the consent page
  discloses the map for every mapped `team:*` grant it shows, so no consent predates it, §16.1);
  `act = {sub: <adapter client_id>}`, audited; no refresh token.
- Response: `access_token`, `token_type=Bearer`, `expires_in=900`, rotated `refresh_token` if the
  grant allows, `scope` (as granted); `Cache-Control: no-store`.

**Revocation (`POST /revoke`, RFC 7009).** A known refresh token of the calling client revokes its
family; anything else gets a silent 200 [V] tree contract. Outstanding access tokens live ≤ 15 min
unless denied by `family_id` (§5.8); from M3 the AS feeds revoked families to the global deny
table, and an operator action revokes every family of one edge `sub` (deprovisioning).

**CORS.** Metadata, JWKS, `/register`, `/token`, `/revoke`: `Access-Control-Allow-Origin: *`,
preflight, no credentials [V] tree `http.rs`; never on `/authorize` or `/callback`.

**Signing keys.** Secret `EDGE_AUTH_SIGNING_JWK` [V] tree `signer.rs`: a private EC P-256 JWK, or a
JWK Set whose **first** key is the active private key and whose other entries (≤ 3, `MAX_KEYS = 4`)
are only published; `x`/`y`/`kid`, if present, must match `d`; `kid` = RFC 7638 thumbprint.
Generated offline, never in vars, logs or the repository; absent → JWKS 503, `/token` fails closed.

- **Planned rotation:** add the next key as the second entry; after ≥ 20 min (RS JWKS TTL 10 min +
  skew + margin) move it first; drop the old key ≥ 16 min after it stops signing.
- **Emergency:** replace the secret without the old key and deny-list its `kid` (≤ 60 s at the
  gateway; adapter resource servers at their next JWKS fetch, ≤ 10 min). Refresh tokens are
  unaffected (not signed).

### 5.7 RFC 9728 protected-resource metadata

RFC 9728 §3.1: the metadata URL inserts `/.well-known/oauth-protected-resource` before the resource
path, and `resource` MUST equal that identifier [V] tree (`prm::metadata_url`).

| Document URL (gateway host) | `resource` | `scopes_supported` |
|---|---|---|
| `/.well-known/oauth-protected-resource/v1` | `https://<gw>/v1` | `ruvector:read ruvector:write ruvector:admin offline_access` |
| `/.well-known/oauth-protected-resource/v1/mcp` | `https://<gw>/v1/mcp` | `ruvector:read ruvector:write offline_access` |

Both carry `authorization_servers:
["https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev"]` and `bearer_methods_supported:
["header"]`, and are served with `Access-Control-Allow-Origin: *`. The **bare-origin** path returns
**404**: no token is minted for `https://<gw>`, so a document naming it would send fallback clients
to `invalid_target`. (§8 delta 5 closed: live-verified 404, per-resource scopes, [V] 52e2a5c55.)

- 401 challenges are route-specific: `/v1/mcp*` → `Bearer
  resource_metadata="…/oauth-protected-resource/v1/mcp", scope="ruvector:read offline_access"`
  (plus `error="invalid_token"` when a token was presented, none when absent, RFC 6750 §3.1); other
  `/v1/*` → the `/v1` document. 401 and 403 responses carry `Access-Control-Expose-Headers:
  WWW-Authenticate`.
- 403 `insufficient_scope` is the §5.3 step-up challenge.
- Only `/v1/health` (`{ok:true}`) and the metadata documents are anonymous.

### 5.8 Revocation gap and deny-list

No introspection, so a stolen access token works for ≤ 15 min. Mitigation: a two-scope deny-list
`deny(scope ∈ {global, tenant}, tenant_key NULL, kind, value, expires_at, created_by, reason)`.

- **Tenant entries** live in `TenantLedger`, owner-only via `POST /v1/tenant/deny` (server forces
  `scope = tenant`, `tenant_key = caller`). Kinds: `jti` (TTL ≤ 900 s), `family_id` (≤ 7 d), `sub`
  (≤ 30 d, renewable, members only) and `client_id` (≤ 365 d — the owner's way to block an app for
  their tenant; replaces connector approval). They apply only to requests for that tenant.
- **Global entries** are operator-only (break-glass, audited), kinds as above plus `org` and `kid`.
  Before M3 a Worker secret (`DENY_GLOBAL`, small JSON); from M3 D1, fed also by AS family
  revocations. **Not shipped** ([V] 52e2a5c55): no `DENY_GLOBAL` secret, no D1; `deny_check`
  reads only the caller's `TenantLedger` (`tenant_admin.rs:315-358`, kinds `jti`, `family_id`,
  `sub`, `client_id`; a hit is `401 invalid_token`, cached 30 s per isolate). Operator break-glass
  is key rotation (§5.6) or per-tenant entries until the D1 control plane lands (§6.4).
- Cache: in-isolate map (30 s) keyed by `(tenant_key, kind, value)`; exposure after a write ≤ 30 s
  (tenant) / ≤ 60 s (global).
- The deny-list is enforced by ruvector resource servers only. Adapter-hosted resource servers
  (§16.1) see revocation only through `kid` removal (≤ 10 min) and token expiry (≤ 15 min); every
  call they forward to `/v1/ops` is still denied here.

### 5.9 Internal hops

Only the gateway and the auth Worker are public. Services split out later (M4) set `workers_dev =
false` and no routes, receive the **original `Authorization` header** over a Service Binding and
**re-verify** the token with the same crate (cost measured at M1 **[U]**). No shared-secret context
header (§14.7). The same rule governs adapter callers of `/v1/ops` (§16.3). A CI check fails any
internal environment that sets `workers_dev = true` or declares `routes`.

## 6. Storage and persistence

### 6.1 `VectorShard` (DO SQLite)

```sql
meta(k TEXT PRIMARY KEY, v TEXT)   -- identity (tenant_key, service, collection_uid, shard), dim, metric,
                                   -- index_kind, embedder, schema_ver, write_seq, snapshot_seq, index_epoch,
                                   -- index_sha256, M2b entry_point/max_layer/m0/alpha, quant_kind/epoch/params
vectors(id TEXT PRIMARY KEY, iid INTEGER UNIQUE, f32 BLOB, q8 BLOB, q8_epoch INTEGER,
        metadata TEXT, updated_at INTEGER, deleted INTEGER DEFAULT 0)
ops(seq INTEGER PRIMARY KEY, op TEXT, id TEXT, ts INTEGER, actor_sub TEXT, jti TEXT, family_id TEXT)
index_chunks(epoch INTEGER, part INTEGER, bytes BLOB, PRIMARY KEY(epoch, part))   -- M2b, parts ≤ 1.9 MB
filter_idx(key TEXT, value TEXT, id TEXT)                                         -- ≤ 8 declared keys
idempotency(sub TEXT, key TEXT, body_sha256 BLOB, response TEXT, bytes INTEGER, expires_at INTEGER,
            PRIMARY KEY(sub, key))
audit(seq INTEGER PRIMARY KEY, ts, sub, client_id, act_sub, family_id, route, outcome, bytes,
      approval_ref, hash_prev, hash, shipped INTEGER DEFAULT 0)
timers(kind TEXT PRIMARY KEY, due_at INTEGER, state TEXT)                          -- alarm multiplexing
```

**Shipped schema ([V] 52e2a5c55, `crates/ruvector-edge-store/src/schema.rs:1-33`):** `VectorShard`
has `meta`, `vectors` (`q8` column unused: codes persist as `index_chunks`), `ops` (+ `body`,
`act_sub`), `filter_idx`, `index_chunks`; `idempotency` lives in `TenantLedger` (one table for
`op_id`s and REST keys, §7), audit is one chain per tenant in the ledger (§6.5), and alarms are
driven by shard state (`maintenance_due`), not a `timers` table.

**Write path** (no `transaction()`, no `blockConcurrencyWhile`). *Designed:* (1) hold an unexpired
**batch quota lease** from `TenantLedger`; (2) validate in memory; (3) coalesce row, `ops`,
`idempotency` and `audit` writes; (4) then mutate the slab; (5) alarm reconciles lease usage.
*Shipped* (`edge/gateway/src/vectors.rs:1-14`): no leases — every executor **admits** `ops: 1` +
base work units through the ledger before the first shard call; upsert is `Plan` (validate, report
the delta; `dry_run` stops here, §17) → ledger `Admit` of the growth → `Apply` (the shard re-plans
atomically and answers `409` if the fresh delta exceeds what was admitted); uncharged remainder is
refunded, an unknown-effect write stays charged. Row and `ops` writes still coalesce with no
`await` (≤ 100 bound parameters; deletes ≤ 100 per batch). Idempotency is checked in the gateway
against the ledger **before** the op (§7: reuse with another body is **`409 op_replayed`**).

**Isolate-wide memory budget.** Resident data is capped at **56 MB per isolate [U]**, LRU-evicting
cold shards. *Shipped* ([V] 52e2a5c55): because every DO class of the script can share an isolate,
the 56 MB is **split per class** — `VectorShard` 28 MB (per-shard cap **14 MB** of codes/links plus
~128 B/row bookkeeping, so **two** full shards; ≈ 25k × 384 flat, **[needs Paid]**: a cold load of a full shard exceeds Free's 10 ms), `QuantShard` 16 MB,
`GraphStore` 12 MB — and the rest of the 128 MB is budgeted for one `AnalyticsJob` turn (≤ 32 MiB)
and one registry finalize part (2 × 16 MiB), ≈ 123 MB total (`shard_core.rs:20-39`,
`shard/mod.rs:69-73`). The M1 "3,000,000 floats" cap was never used (f32 stays in SQLite). A panic or OOM reinitialises the **whole instance**: no-unwrap lint on the request path,
fuzzing, and the Q12 decision.

**Index kinds.** *Shipped* ([V] 52e2a5c55): `flat` (default) is the int8 scan below with f32 rerank
(`max(40, 4·top_k)` ≤ 1000 candidates, recall@10 1.000 measured), `hnsw` is opt-in (default
`ef` cosine 1024, l2/dot 512, ≤ 2048), there is no `q8` kind and no f32 slab; HNSW was built
in-repo (`ruvector-edge-index`), so G4a no longer gates it, and it is rebuilt only in alarms within
`REBUILD_BUDGET_MS` 20 s **[needs Paid]** (sized under Paid's 30 s default, `maintain.rs:50-60`; not reachable on Free). *Designed:* **M1 `flat`**: f32 slab, exact scan, no `q8`. **M2a `q8`**: u8 flat scan with
asymmetric distance and f32 rerank from SQLite; cosine uses the fixed range `[-1, 1]`, l2/dot a
1k-reservoir quantizer with `quant_epoch`, per-row `q8_epoch` and sliced re-encode. **M2b `hnsw`**
after G4a: `DistanceOracle` traversal over u8 codes, CSR `u32` adjacency, dense iids renumbered on
compaction (`node_count == max_iid + 1` asserted), working set ≈ `n·d + n·m0·4` (100k × 384, m0 = 32
→ ≈ 51 MB) **[U]**; flush at 200 ops or 60 s through a streaming encoder into ≤ 1.9 MB
`index_chunks` with `index_sha256`; ≤ 64 synchronous inserts per request; compaction and rebuilds in
alarm slices or off-platform; wasm32 simd128. Randomness is `splitmix64(seq ^ collection_salt)` — no
`getrandom`, no clock.

**Cold start (lazy).** Nothing at module top level; the first request loads `meta` and streams rows
with `cursor.raw()` paged by `WHERE iid > ? ORDER BY iid LIMIT 1024` (never `to_array`); M2b decodes
the latest chunks, verifies sha256 (else previous epoch + replay) and replays newer `ops`.
**Alarms** multiplex `flush`, `snapshot`, `compact_slice`, `split_step`, `lease_reconcile`,
`audit_ship`, `requantize` through `timers` (`min(due_at)`). **Bounded growth:** idempotency rows
(24 h) and the audit tail count toward bytes stored; before M3 the tail is capped at 10k rows. The
request path never calls `std::time`; `rvf-runtime`, rvlite storage and rayon are not linked into
M1–M3.

### 6.2 `TenantLedger` (one DO per tenant)

Holds, for its own tenant only: memberships (§4.2), tenant deny entries (including client blocks,
§5.8), the collection catalog `(name, collection_uid, service, kind, dim, metric, embedder,
shard_count, state, origin, created_by, created_at)`, quota counters and **work-unit** usage (§17),
outstanding leases, and jobs. Admission is lease-based; from M3 an alarm rolls usage into D1
`usage_daily` every 5 minutes. *Shipped* ([V] 52e2a5c55): no leases (per-op admission, §6.1) and
no D1 roll-up; the ledger also holds the idempotency table (§7), the per-tenant audit chain and
witness chain (§6.5, `m3_audit_ledger.rs`, `m3_ledger.rs`), upload and import-job state (M3).

### 6.3 R2 layout (M3: bucket `ruvector-edge-data`, one binding `EDGE_DATA`)

Every key component is server-derived or a validated slug. One R2 binding, `EDGE_DATA`, serves M3
and M5 (`edge/gateway/wrangler.toml:136-144`); lifecycle rules (per the `wrangler.toml` comment and HORIZON, not in repo) **[L]** expire `exports/` after
1 day and abort incomplete multipart uploads after 7 days; `staging/` has no rule (shared with the M5 sweep).

| Prefix | Use |
|---|---|
| `snapshots/{tenant_key}/{service}/{collection_uid}/{shard}/{epoch:020}-{sha256_12}/seg-{n}.rvf` (+ `manifest.rvf`) | snapshots |
| `exports/{tenant_key}/{export_id}.rvf` | exports ([V] 52e2a5c55 `export.rs:136`; download `GET /v1/exports/{id}` within 15 min) |
| `staging/{tenant_key}/{upload_id}` | uploads (`upload_id` server-minted by `POST /v1/uploads`, or by a registry upload session) |
| `rvf/{tenant_key}/{scope}/blobs/sha256/{hex}` | private / tenant registry blobs, content-addressed, deduplicated per scope ([V] 52e2a5c55 `registry/src/keys.rs:7-16`) |
| `public/blobs/sha256/{hex}` | public registry blobs (global, write-once) |
| `rvf/{tenant_key}/@{scope}/{name}/{version}/manifest.json`, `public/{tenant_key}/@{scope}/{name}/{version}/manifest.json` | listable manifest mirrors *(designed, not written: `keys::manifest_key` has no caller outside its unit test; the index is the `RegistryScope` DO)* |
| `results/{tenant_key}/{result_id}` | signed result pages (§17) — **not shipped** |
| `audit/{tenant_key}/{yyyy}/{mm}/{dd}/{seq:012}.ndjson` | shipped audit |

- RVF segments via `rvf-wire`: ≤ 4 MB `VEC_SEG`s from a SQLite cursor, index chunks (M2b), metadata,
  and a manifest with per-segment sha256, the `tenant_key`, `collection_uid` and audit-chain head.
  Streamed (multipart / `ReadableStream`), never assembled whole. Hourly when dirty; 24 hourly + 7
  daily retained. *Shipped* ([V] 52e2a5c55): snapshots are on request only (no hourly alarm or
  retention policy yet); snapshot, export and restore each run synchronously in one request and
  are refused **413** above 2^18 stored floats before any side effect (`sync_budget.rs`) **[Paid]**;
  export segments are sized to the 1 MiB queued-import segment cap; `index = rabitq` collections are
  refused **400 `invalid_request`** (`quant_route.rs:70-84`).
- **Restore** checks in order: size vs quota before allocation, manifest `tenant_key`, per-segment
  sha256, dimension and quota; segments are read one at a time by range.
- **Import** takes only an `upload_id` resolving inside the caller's tenant, or an inline body
  ≤ **512 KiB** **[Paid]** (designed 8 MiB); queued deliveries process ≤ 1 MiB segments and 2
  batches each, and a job with 15 deliveries without progress fails before its DLQ
  (`uploads.rs:38`, `ingest.rs:63-84`).
- DO point-in-time recovery (30 d) is an operator-only second layer.

### 6.4 Control plane (M3, gated: D1 `ruvector-edge-control`) — not shipped

**Not built as of `52e2a5c55`:** the gateway binds no D1. Per-tenant state (quota, usage, deny,
catalog) lives in `TenantLedger`; the registry index lives in DOs (§3); `tenants`/`plans`,
`deny_global`, `usage_daily` and `audit_index` do not exist, so M3's "global deny within 60 s" and
"usage reconciles within 1%" acceptance items are open. The design below stays the target.

D1 holds only global state: `tenants(tenant_key, plan, status)`, `plans`, `deny_global`,
`usage_daily`, `packages`, `audit_index`. Every access goes through a typed repository taking
`&TenantContext` (or an explicit `OperatorContext`) that adds `tenant_key = ?` itself; raw SQL in
handlers is banned by a CI grep and a clippy `disallowed_methods` entry. Cache keys include
`tenant_key` under `https://cache.ruvector-edge.invalid/…`. The adapter's `ruvector-chatgpt` D1 is
**not** part of the backend (§16.4).

### 6.5 Audit

Each DO keeps a hash-chained append-only `audit` tail; from M3 a Queue (`ruvector-edge-audit`) ships
batches to R2 NDJSON. *Shipped* ([V] 52e2a5c55, `audit.rs:1-20`): no per-DO tail; each mutating or
privileged request (and each finished import job) sends one event to `AUDIT_QUEUE` via
`wait_until` (best effort, no added latency); the consumer hands each tenant's batch to its
`TenantLedger`, which deduplicates by message id and chains it onto the **one** per-tenant chain,
then writes `audit/{tenant}/{yyyy}/{mm}/{dd}/{seq:012}.ndjson`. Plain reads, job/snapshot-list polls
and 429s are not audited. DLQ `ruvector-edge-audit-dlq`. Logged: `tenant_key`, `sub`, `client_id`, `act.sub` (exchanged tokens), `jti`,
`family_id`, route/op, scope and role used, outcome, byte/row counts, `approval_ref`, ray id. **Never logged:** tokens, upstream
identifiers, vectors, metadata payloads, IPs in the clear (IPs are hashed with the Worker secret
`IP_HASH_SALT`, rotated monthly). **[V] 52e2a5c55:** `IP_HASH_SALT` is **not set** on the live
`ruvector-edge-auth`, and the code falls back to an **empty salt** (`auth-worker/src/lib.rs:123-127`,
`store.rs:131-136`), so the per-IP DCR/authorize buckets (`dcr_rate` rows and Rate Limiting keys)
are unsalted truncated sha256 of `CF-Connecting-IP` — brute-forceable for IPv4. Only operator IPs
reach `/register`/`/authorize` before G1; **setting the secret is a G1 precondition** (`wrangler
secret put IP_HASH_SALT`, no code deploy). Failing closed on an empty salt in release builds is a
follow-up.

## 7. API specification

**Conventions (gateway `/v1`).** JSON (UTF-8) with enforced `Content-Type`; `Content-Length` checked
before parsing; strict schemas (unknown fields rejected); RFC 9457 `application/problem+json` errors
with a stable `code`; `x-request-id` on every response; `Cache-Control: no-store` on credentialed
routes; `Idempotency-Key` (≤ 255 bytes, 24 h, scoped `(tenant, sub, key)`, body-hash bound) on
mutating routes; cursor pagination `limit ≤ 100`; CORS default-deny with an explicit origin
allowlist and no credentials mode, except the public PRM documents (`*`) and
`Access-Control-Expose-Headers: WWW-Authenticate` on 401/403.

*Shipped* ([V] 52e2a5c55): **`Idempotency-Key`** is 1–255 visible ASCII bytes, honoured
on collection create, vector upsert/delete, graph `…/edges` and mutating Cypher (not on dry runs,
reads, tenant admin routes or drop, which are naturally idempotent); the 400 for a malformed key
applies only where the key is honoured (on a read or dry run it is ignored, not rejected); it is bound to sha256 of the
route tag and the **raw body bytes** (so a re-serialised but equal JSON body is a different
request), stored in the `TenantLedger` table shared with `/v1/ops` `op_id`s under the `rest:`
prefix (a `text` request is keyed on its original body under `m3text:`). A reuse with another body
is **`409 op_replayed`** (designed `422 idempotency_mismatch`, changed so REST and §16.3 share one
code); a concurrent twin while the first is in flight is `409 conflict` (`retry_after_s` 1 on `/v1/ops` only);
only a successful response is remembered and replayed (failures release the key, so a retry
re-executes). **CORS** is `Access-Control-Allow-Origin: *` on every response (bearer-only API, no
cookies), exposing `WWW-Authenticate, Retry-After, ETag, X-RVF-Yanked, Content-Disposition`
(`respond.rs:1-30`). Data-route bodies are capped at 1 MiB while streaming (declared larger →
413 before auth); registry parts ≤ 16 MiB, upload parts ≤ 8 MiB.

### 7.1 Edge authorization server (`https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev`)

| Method & path | Request | Response | Milestone |
|---|---|---|---|
| `GET /.well-known/oauth-authorization-server` | — | RFC 8414: `issuer`, `authorization_endpoint`, `token_endpoint`, `registration_endpoint`, `revocation_endpoint`, `jwks_uri`, `scopes_supported` (§5.3), `response_types_supported ["code"]`, `grant_types_supported ["authorization_code","refresh_token"]` (+ token-exchange from M1), `token_endpoint_auth_methods_supported ["none"]` (+ `private_key_jwt` from M1), `revocation_endpoint_auth_methods_supported ["none"]`, `code_challenge_methods_supported ["S256"]`, `authorization_response_iss_parameter_supported true`; no `client_id_metadata_document_supported` [V] tree `metadata.rs` | M0.5 |
| `GET /.well-known/jwks.json` | — | `{keys:[…]}` every key in `EDGE_AUTH_SIGNING_JWK` (public parts), `kid` = thumbprint | M0.5 |
| `POST /register` | RFC 7591 JSON ≤ 16 KiB (§5.6) | 201 `{client_id, client_id_issued_at, redirect_uris, token_endpoint_auth_method:"none", grant_types, response_types:["code"], scope, client_name?}` | M0.5 |
| `GET /authorize` | `response_type=code, client_id, redirect_uri, scope, state, code_challenge, code_challenge_method=S256, resource` | consent page (stores the flow, sets `__Host-eaf-{state[0..16]}`; Continue → upstream authorize, Cancel → `access_denied`), or error page / error redirect | M0.5 |
| `GET /callback` | upstream `code, state[, iss]` + cookie | 302 `redirect_uri?code&state&iss` | M0.5 |
| `POST /token` | form: `grant_type=authorization_code` (`code, redirect_uri, client_id, code_verifier[, resource]`) or `refresh_token` (`refresh_token, client_id[, scope][, resource]`); M1: token-exchange (`subject_token, subject_token_type=…:access_token, resource=…/v1[, scope]`, `client_assertion` private_key_jwt) | `{access_token, token_type:"Bearer", expires_in:900, refresh_token?, scope}` (exchange: `issued_token_type`, no refresh) | M0.5 (exchange M1) |
| `POST /revoke` | form: `token, client_id[, token_type_hint]` | 200 (always, except storage failure) | M0.5 |

Errors are RFC 6749 JSON `{error, error_description}` with a static description (never echoing
input) and `Cache-Control: no-store`: `400` `invalid_request`, `invalid_grant`,
`unauthorized_client`, `unsupported_grant_type`, `unsupported_response_type`, `invalid_scope`,
`access_denied`, `invalid_target` (RFC 8707), `invalid_redirect_uri`, `invalid_client_metadata` (RFC
7591); `401 invalid_client`; `500 server_error`; `503 temporarily_unavailable` [V] tree
`authz/error.rs`.

### 7.2 Gateway (`https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev`)

As shipped at `52e2a5c55` ([V]; routers `routes.rs`, `rest.rs:141-175`, `m3_api.rs:79-100`,
`graph_routes.rs`, `mincut_routes.rs`, `registry_routes.rs:87-118`, `mcp.rs`). Rows marked
*(designed)* are not shipped.

| Method & path | Body / params | Scope + role | Milestone |
|---|---|---|---|
| `GET /.well-known/oauth-protected-resource/v1[/mcp]` | — (bare origin, `/.well-known/oauth-authorization-server`, `/.well-known/openid-configuration` → 404) | anonymous | M0.5 |
| `GET /v1/health` | → `{ok:true}` | anonymous | M0.5 |
| `GET /v1/me` | → `{tenant_key, org_id, workspace_id, sub, client_id, scopes, role \| null, claimed}` | any valid token | M0.5 (role/claimed from M1) |
| `POST /v1/tenant:claim` (alias `/v1/claim`; MCP `tenant_claim`) | → 201 `{role:"owner"}` or 409 | `ruvector:write` or `ruvector:admin`; tenant unclaimed | M1 |
| `GET/POST /v1/tenant/members`, `DELETE /v1/tenant/members/{sub}` | `{sub, role: viewer\|editor}` | `ruvector:admin` + owner | M1 |
| `POST /v1/tenant/deny` | `{kind: jti\|family_id\|sub\|client_id, value, ttl_s}` (caps §5.8) | `ruvector:admin` + owner | M1 |
| `GET /v1/usage` | → quotas, usage, work units, rate headroom | `ruvector:read` + viewer | M1 |
| `POST /v1/collections` | `{name, dim: 1..1536, metric: cosine\|l2\|dot, index?: {kind: flat\|hnsw\|rabitq, m?: 8..48, ef_construction?: 32..200}, embedder?: "bge-small-en-v1.5", filterable_keys?: [≤8], shards?: 1..6}` → 201 (`q8` refused) | `ruvector:write` + editor | M1 (`flat`), M2, M3 (`embedder`), M4 (`rabitq`) |
| `GET /v1/collections`, `GET /v1/collections/{c}` | cursor / → config + counts | `ruvector:read` + viewer | M1 |
| `DELETE /v1/collections/{c}` | tombstone-first (`Deleting`), wipe every shard of that uid, then purge; the uid is never reused, the name is free at once (designed: 7-day tombstone); 409 while a restore is live | `ruvector:write` + **owner** (§5.3) | M1/M2 |
| `POST /v1/collections/{c}/vectors:upsert` (alias `…/vectors`) | `{vectors: [{id ≤256B, values?, text?, metadata? ≤4KiB}] (1..500), dry_run?}` → `{upserted, write_seq}`; > 500 rows (or > 64 per `hnsw` shard) → **413 `payload_too_large`** (`vectors.rs:79-82`, `shard/write.rs:150-155`) (designed: 202 job; bulk goes through `:import`) | `ruvector:write` + editor | M1 (`text` M3) |
| `POST /v1/collections/{c}/query` | `{vector?\|text?, top_k: 1..100, ef?: ≤2048, rerank?: top_k..1000, filter?: ≤8 clauses (not on `rabitq`), include?}` → `{matches, shards_queried, took_ms}` | `ruvector:read` + viewer | M1 |
| `POST …/vectors:fetch` (alias `…/fetch`) `{ids ≤100}` | — (`GET …/vectors/{id}` *(designed)*) | `ruvector:read` + viewer | M1 |
| `POST …/vectors:delete` (alias `DELETE …/vectors`) | `{ids ≤1000, dry_run?}` (batches of ≤ 100) | `ruvector:write` + editor | M1 |
| `POST /v1/ops` | §16.3 envelope (Service-Binding callers, `aud = …/v1` only) | per op | M1 |
| `POST …/snapshots` / `GET …/snapshots` | synchronous, ≤ 2^18 floats **[Paid]** / list | write+editor / read+viewer | M3 |
| `POST …/snapshots/{id}:restore` | synchronous, ≤ 2^18 floats **[Paid]** | `ruvector:admin` + owner | M3 |
| `POST /v1/collections/{c}:export` → `GET /v1/exports/{id}` | `redact` keys (≤ 32) → `.rvf` with witness manifest, downloadable 15 min; ≤ 2^18 floats **[Paid]** | `ruvector:read` + viewer | M3 |
| `POST /v1/uploads`, `POST /v1/uploads/{id}/parts/{n}` (≤ 8 MiB), `POST /v1/uploads/{id}:complete`, `POST …:import`, `GET /v1/jobs/{id}` | `{size, sha256}` / part bytes / — / `{upload_id}` or ≤ **512 KiB** inline **[Paid]** / — | write+editor (uploads, import) / read+viewer (job) | M3 |
| `GET /v1/audit?cursor=&limit≤500` *(designed)* | own tenant only | `ruvector:admin` + owner | M3 |
| `POST/GET /v1/results[/{id}]` *(designed)* | publish a signed, redacted result page (§17) / read it | write+editor / read, or public if opted in | M3/M5 |
| `POST/GET /v1/graphs`, `GET/DELETE /v1/graphs/{g}`, `POST …/edges` (≤ 1k per request), `POST …/cypher` (≤ 4 KiB, ≤ 10k steps, ≤ 1k rows) | state ≤ 256 KiB **[Paid]**; `Idempotency-Key` on edges and mutating Cypher | read / write + editor (mutating Cypher, edges, create, delete) | M4 |
| `POST /v1/mincut`, `GET /v1/mincut/jobs/{id}` | `{graph \| edges, mode: exact\|approximate, epsilon?}` → 200 inline, 202 job (write + editor + one write token; ≤ 20 live jobs, 24 h), 413 beyond | read / write | M4 |
| `POST /v1/rvf/scopes/{scope}`, `GET /v1/rvf/scopes` | claim a global `@scope` / list own scopes | `ruvector:admin` / `ruvector:read` | M5 |
| `POST /v1/rvf/{scope}/{name}/{version}/uploads`, `PUT …/uploads/{id}/parts/{n}` (≤ 16 MiB), `POST …/uploads/{id}:finalize` | immutable SemVer; registry-native upload session (202 per step until 201) **[Paid]** | `ruvector:write` + owned scope | M5 |
| `GET /v1/rvf/{scope}/{name}`, `GET …/{version}`, `GET …/{version}/blob` | versions (paginated) / manifest / streamed bytes (`ETag`, `X-RVF-Yanked`) | `ruvector:read` | M5 |
| `POST …/{version}:yank` / `:unyank` | — | `ruvector:write` (uploader or admin) | M5 |
| `POST …/{version}:publish` | public publish | `ruvector:publish` + owner (not grantable until the allowlist lists it, §5.3) | M5 |
| `POST /v1/collections/{c}:import-rvf` | `{package, version, offset?}` → windows with `next_offset` | read on the package, write + editor on the collection | M5 |
| `GET /v1/rvf?prefix=` *(designed)* | — | `ruvector:read` | M5 |
| `POST /v1/mcp` | JSON-RPC 2.0 `initialize` (protocol `2025-11-25`, `2025-06-18`; `MCP-Protocol-Version` checked), `tools/list`, `tools/call`. Tools: `tenant_me`, `tenant_claim`, `collection_list`, `collection_get`, `vector_query`, `vector_fetch`; mutating with `dry_run`: `collection_create`, `vector_upsert`, `vector_delete`; M4 `graph_list`, `graph_query` (read-only Cypher), `graph_mutate` (write, `dry_run`), `mincut` (read, inline only); M5 `rvf_list`, `rvf_get`, `rvf_import` (write, **no** `dry_run`). Missing write → HTTP 403 step-up challenge (§5.3) | scope ∩ role | M0.5 401 challenge; M1; M4; M5 |

**Status codes** (`crates/ruvector-edge-store/src/error.rs:63-103`): `400` `invalid_request`,
`dimension_mismatch`, `non_finite_value`, `unknown_op`, `target_mismatch`; `401` `invalid_token`
(route-specific `resource_metadata`; also any `aud` mismatch, logged as `audience_not_allowed`, and a
tenant deny hit); `403` `insufficient_scope` (step-up challenge), `role_required`, `not_claimed`,
`tenant_mismatch`; `404` `not_found` (also foreign tenants); `409` **`op_replayed`** (idempotency key
reused with another body — designed as `422 idempotency_mismatch`, §7) and `conflict` (a concurrent
twin of an in-flight key, `retry_after_s` 1; a shard re-plan that outgrew its admission); `413`
`payload_too_large`, `quota_exceeded`, `budget_exceeded`; `429` `rate_limited` + `Retry-After`; `503`
`jwks_unavailable`, `shard_unavailable` (retryable, e.g. a cold 50k-row quant load or a graph load
over 128 KiB), `trust_root_mismatch`; `500` `server_error`. `tenant_suspended` is designed but not
shipped (no plan/status source without D1). Retryable outcomes are never stored as idempotent
replies.

## 8. Crate and repository layout (as built, `52e2a5c55`)

```
crates/ruvector-edge-auth/    [V] RS verification: jws, jwks, claims (TokenKind EdgeIssued |
                              UpstreamFirstParty), audience (exact aud), resource, subject (edge_subject),
                              scopes (Capability, SCOPE_TABLE, M1 ROUTE_TABLE — not used at runtime),
                              prm (REST_SCOPES, MCP_SCOPES), verifier, clock, error.
  fuzz/                       [V] cargo-fuzz crate (own [workspace]): jws_compact, jwks_parse, resource_url.
crates/ruvector-edge-authz/   [V] AS core over sync ports: client (DCR, DEFAULT_CLIENT_SCOPE), authorize,
                              pkce, code, federation, token, grant (incl. RFC 8693 exchange), params,
                              refresh, revoke, metadata, resource (vocabularies, TEAM_EXCHANGE_MAP), error.
crates/ruvector-edge-tenancy/ [V] context, names (do_name, Service vector|quant|graph|mincut), uid, shard,
                              meta, membership, lease (unused by the gateway), validate, quota, problem.
crates/ruvector-edge-store/   [V] SqlStore port + schema, VectorShard (flat/hnsw, ops log, index_chunks,
                              alarms), TenantLedger (catalog, members, deny, usage, idempotency), ops
                              dispatch/authz, ResidentRegistry.
crates/ruvector-edge-index/   [V] int8 flat + HNSW over codes, rerank, chunked persist, memory model (M2).
crates/ruvector-edge-snapshot/ [V] RVF snapshot/export/import writer + reader, witness chain (M3).
crates/ruvector-edge-quant/   [V] RaBitQ shard over ruvector-rabitq, persist v2 rbqx0002, budgets (M4).
crates/ruvector-edge-analytics/ [V] min-cut over ruvector-mincut (exact), job descriptor (M4).
crates/ruvector-edge-registry/ [V] names/scopes, SemVer, manifests, upload sessions, keys (M5).
crates/ruvector-edge-cli/     planned (not built): edge-AS loopback login, logout, claim, commands.
edge/                         [V] standalone workspace (root `exclude`, own Cargo.lock): worker =0.8.5,
                              wasm-bindgen =0.2.125; release opt-level z, lto, 1 CGU, panic = abort;
                              rvlite from git branch feat/rvlite-browser-feature-gate, no default features.
  auth-worker/                → Worker `ruvector-edge-auth`: DO AUTH_STORE (migration v1); rate limits
                              DCR_RATE_LIMITER / AUTHZ_RATE_LIMITER; vars ISSUER, RESOURCE_ALLOWLIST,
                              UPSTREAM_* (incl. UPSTREAM_CLIENT_ID, ACCEPTED_UPSTREAM_KIDS — empty until
                              G1), CONFIDENTIAL_CLIENTS (empty); secrets EDGE_AUTH_SIGNING_JWK,
                              IP_HASH_SALT (optional).
  gateway/                    → Worker `ruvector-edge-gateway`, one script for every data-plane class:
                              DOs TenantLedger + VectorShard (tag v1), RegistryRoot + RegistryScope
                              (tag v2-registry), QuantShard + GraphStore + AnalyticsJob (tag
                              v-m4-quant-graph); Service Binding EDGE_AUTH (JWKS); 8 Rate Limiting
                              bindings RL_{READ,WRITE,OPS,MCP}_{USER,ORG}; R2 EDGE_DATA →
                              ruvector-edge-data; Queues ruvector-edge-ingest (+ -dlq) and
                              ruvector-edge-audit (+ -dlq), producer and consumer; Workers AI AI;
                              compiled trust root (trust_root.rs), only UPSTREAM_FIRST_PARTY a var;
                              no D1, no [limits] (Free plan).
```

**Known deltas between the code and this ADR** (re-checked at `52e2a5c55`; "closed" = the code now
matches the body):

1. **Scopes and capabilities.** Vocabulary **closed** (`ruvector:*`, `Admin`, `PublishPublic` →
   `ruvector:publish`). **Open:** `ROUTE_TABLE` has no role column and no `Ops` surface and is not
   used at runtime (authorization is per handler, §5.3); the route-coverage gap test is owed.
2. **Closed:** `UPSTREAM_SCOPES = "openid profile email"`, enforced at load.
3. **Partly open:** `redeemed_codes`, `dcr_rate`, `client_activity`, `assertion_jtis` exist; the
   `subjects` table does not (`sql_ports.rs:28-38`).
4. **Partly open:** DCR default `ruvector:read ruvector:write offline_access`, `/authorize` default
   first scope, drop-not-refuse and omitted `grant_types` → both are closed; **`localhost` is still
   refused** (`client.rs:166-172`, `tests/dcr.rs:136`).
5. **Closed:** bare-origin PRM 404, per-resource PRM scopes, `/v1/mcp` route and challenge, exposed
   `WWW-Authenticate` (live-verified, HORIZON M0.5).
6. **Partly open:** the gateway trust root is compiled (`trust_root.rs`, 503
   `trust_root_mismatch`), and `ACCEPTED_UPSTREAM_KIDS`/`FIRST_PARTY_AUDS` are compiled (empty) there;
   the **auth-worker** `ISSUER` and `UPSTREAM_*` are still `[vars]` (`config.rs:139-158`), and its
   `ACCEPTED_UPSTREAM_KIDS` is a var pinned at the callback (§5.6 says "not pinned").
7. **Closed:** consent verified/unverified marker (`redirect_is_verified`), upstream refresh
   revoked at `/callback` (`callback.rs:98-100`).
8. **Recorded choice:** pins stay 0.8.5 / 0.2.125.
9. **Closed:** `wrangler.toml` builds with `env -u RUSTFLAGS`.
10. **Closed:** `aud` mismatch → 401 `invalid_token` + `resource_metadata`.
11. **Open:** no `REFRESH_REUSE_GRACE_SECS`; any re-presented rotated refresh token revokes the
    family (`refresh.rs:143-150`). **Closed (M1):** exchange grant, `private_key_jwt` confidential
    clients (`CONFIDENTIAL_CLIENTS`, empty), `act`.
12. **New (M1–M5 drift):** see the Shipped-vs-designed table; the open ones that need code are
    `dry_run` on the remaining mutating surfaces (REST collection create, tenant claim REST/MCP,
    drop, member invite/remove, deny, graph REST create/delete/edges, `rvf_import`, M3, mincut
    jobs, registry), global deny + D1 (§6.4), `/v1/results`, `GET /v1/audit`, resumable
    snapshot/export/restore, `ruvector:publish` in the allowlist, the registry manifest mirror
    (§6.3), fail-closed `IP_HASH_SALT`, and distinct Rate Limiting `namespace_id`s.

**Rules.** Every source file < 500 lines. Release builds never enable `dev-issuer` or `test-util`.
getrandom: the auth Worker needs a CSPRNG (`getrandom` `js` on wasm32 only, [V] tree); the gateway
needs none at runtime through M3. **CI:** refuse `wrangler deploy --env dev`; assert each
`wrangler.toml` binds only `ruvector-edge-*` resources; assert the release wasm contains no
`dev-issuer`/test-issuer strings; grep-ban raw D1 SQL; fail on `workers_dev = true`/`routes` in
internal envs; fail if any secret name appears under `[vars]`.

## 9. Test strategy

1. **Unit tests, London-school, native** (`cargo test -p ruvector-edge-auth -p ruvector-edge-authz
   -p ruvector-edge-tenancy`), ports mocked with mockall; ES256 keypairs generated in-test, none
   committed.
   - **RS:** `alg=none`, HS256 confusion, RS256, `jku`/`jwk`, DER, thumbprint ≠ `kid`, unknown `kid`
     (rate-limited refetch), expiry/`iat`/`exp − iat > 900`, each §5.2 claim missing or mistyped,
     array `aud`, the other resource's or an adapter's `aud` (401 + `resource_metadata`), header
     `typ` ≠ `at+jwt`, wrong `upstream_iss`, a removed `kid` after refetch (replace, not merge),
     an upstream token with the mode off; in the mode, ID token, `typ=inference`, `dcr-` prefix,
     flags; each deny kind; trust-root var mismatch → 503.
   - **AS:** DCR redirects (`localhost`/`127.0.0.1`/`[::1]` port-agnostic; other `http`, fragment,
     userinfo refused); DCR and `/authorize` scope defaults, dropped scopes, no admin for `/v1/mcp`,
     `grant_types` default; no CIMD flag; no redirect before client validation; `resource`
     missing/unknown/repeated; `plain` PKCE; consent headers; `/callback` cookie missing, wrong, for
     another state, two parallel flows, replayed `state`, upstream `aud`/`iss`/`typ` wrong, unknown
     upstream `kid` (refetch, accept, alert); code one-time, bound, consumed on failure, replay
     revokes; refresh inside grace → no revocation, after → family revoked; 90-day cap; revoke
     silent 200; CORS scope; exchange (M1: adapter-only, `…/v1` only, scope mapped via the adapter's compiled exchange map
     ∩ adapter ceiling ∩ `…/v1` scopes, `exp` narrowed, `act`); minted-token
     golden test; edge subject stable and collision-free (proptest).
   - **Capability / tenancy / store:** every route covered; scope without role and role without
     scope denied; concurrent claim → one owner; a tenant deny never crosses tenants; `tenant_key`
     equal on the edge path and §5.5 mode; recreate → new uid; codec round trips; lease crash; timer
     multiplexing.
   - **RFC 8414 / 9728:** documents match §7.1/§5.7 and each PRM `resource` equals its URL's
     identifier, the bare-origin PRM and AS well-knowns on the gateway are 404, 401/403 expose
     `WWW-Authenticate` — the **conformance checker** reused by MCP Launch Doctor (§17).
2. **Recall oracle:** `ruvector-core` `FlatIndex` (M2a/M2b recall@10). **wasm runtime test** per index
   kind with a peak linear-memory assertion. **CLI:** concurrent refresh → one call; logout revokes.
3. **Integration (`wrangler dev`, `dev-issuer`, stub upstream IdP):** 401 → PRM → AS metadata → DCR
   → consent → callback → token → `/v1/me` → claim (step-up to write) → CRUD; exchange → `/v1/ops`;
   cross-tenant negatives; DO restart; 429; 413; N shards per isolate.
4. **Fuzzing:** JWS, JWKS, DCR JSON, `/token` form, `ResourceUrl`, index chunks (M2b), `rvf-wire`
   (M3), `rbpx0001` (M4). [V] tree: `crates/ruvector-edge-auth/fuzz` targets `jws_compact` (raw
   `Authorization` values through bearer → compact parse → payload → claims JSON → classify →
   ES256 verify → audience → claims validation, plus fuzz-derived header/payload **validly
   signed** with a fixed key so the post-signature checks see arbitrary JSON; accepted tokens must
   satisfy the §5.2 `iss`/`aud`/lifetime invariants) and `jwks_parse` (`JwkSet::parse_usable`,
   thumbprint and key round trip), run with `env -u RUSTFLAGS cargo +nightly fuzz run <t> --
   -max_total_time=600`. `crates/ruvector-edge-registry/fuzz` adds `rvf_validate` (streaming
   `rvf-wire` validator) and `rvf_payloads` (payload decoders). [V] `14e3f77e3` adds
   `resource_url` (auth), `dcr_json` + `token_form` (`ruvector-edge-authz/fuzz`), `index_chunks`
   (`ruvector-edge-index/fuzz`), `rbqx_decode` (`ruvector-edge-quant/fuzz`; `rbqx0002`, which
   superseded the designed `rbpx0001`) and `m3_reader` (`ruvector-edge-snapshot/fuzz`: sealed
   manifest, restore session, RVF import). **Run evidence (2026-09-29, nightly 1.100.0, cargo-fuzz
   0.13.1, `-rss_limit_mb=2048`, 10 targets in parallel, all exit 0, no crash/OOM/timeout):** 600 s
   each for `jws_compact` (687,634 runs), `jwks_parse` (45.8M), `resource_url` (49.2M), `dcr_json`
   (6.05M), `token_form` (11.4M), `index_chunks` (9.74M), `rbqx_decode` (506k); 3600 s each for
   `rvf_validate` (80.3M), `rvf_payloads` (2.47B), `m3_reader` (102M). This meets the 10 min (M0)
   and 1 h (M3) criteria; full table in `edge/HORIZON.json` (`fuzz_table_2026-09-29`).
   **Live:** M0.5 acceptance.

## 10. Quotas and abuse controls

Only layer 4 is a hard, consistent limit; the others are shields.

1. **Pre-auth:** `Content-Length` caps (1 MiB; 8 MiB inline import); `ip:{hash}` limit before
   signature work; failed-token cache; AS per-IP limits on `/register`, `/authorize`, `/token`. WAF
   only with a custom domain (G6).
2. **Rate:** `org:{tenant_key}` (100 / 10 s reads, lower for writes) and `user:{tenant_key}:{sub}`
   (20 / 10 s); a query costs `shards_queried` tokens. Locality **[U]** (M1).
3. **Per-operation budgets:** step/row counters return `413 budget_exceeded` before `cpu_ms`.
4. **Admission:** `TenantLedger` leases (§6.1).

*Shipped* ([V] 52e2a5c55): layer 1 is the 1 MiB streamed body cap (declared larger → 413 before
auth; inline `:import` 512 KiB) and the AS per-IP limits (`DCR_RATE_LIMITER` 5/60 s,
`AUTHZ_RATE_LIMITER` 60/60 s, plus a durable `DCR_RATE_PER_HOUR`); the gateway has **no** pre-auth
`ip:` limit and **no** failed-token cache yet. Layer 2 is eight Workers Rate Limiting bindings,
`user:` then `org:` per class (read 20/100, write 10/50, ops 20/100, mcp 20/100 per 10 s), failing
open (`ratelimit.rs:1-20`, `edge/gateway/wrangler.toml:34-76`). Layer 3 budgets are sized to the
Free 10 ms CPU (**[Paid]**, Shipped-vs-designed #6). Layer 4 is per-op ledger admission, not leases
(§6.1).

| Limit | M1 default (per plan) |
|---|---|
| Collections per tenant | 20 |
| Shards per collection | ≤ 6 through M3 |
| Dimension | 1..=1536, fixed per collection |
| Resident data per isolate | 56 MB, LRU eviction **[U]**; shipped split `VectorShard` 28 / `QuantShard` 16 / `GraphStore` 12 MB, plus ≤ 32 MiB job + 2 × 16 MiB registry part |
| Per-shard cap | designed M1 3,000,000 floats (12 MB); **shipped** 14 MB resident codes/links + row bookkeeping (≈ 25k × 384 flat, 2 per isolate), ≤ 16M stored floats — **[needs Paid]**: a cold load of a full shard does not fit Free's 10 ms, so the cap is a design ceiling, not reachable on Free; HNSW ≤ `REBUILD_BUDGET_MS` 20 s of nodes (≈ 18k at 384-d m16) **[needs Paid]**; `rabitq` ≤ 50k rows / 6M load units **[Paid]** |
| Vectors per tenant | 250k free / 2M paid **[U]**, applied only once the milestone ceiling reaches them |
| Upsert batch | ≤ 500 vectors, ≤ 1 MiB (≤ 64 per shard for `hnsw`); over → 413 |
| `top_k` / metadata | ≤ 100 / ≤ 4 KiB per vector |
| Cypher / mincut inline (M4) | designed ≤ 10k steps, ≤ 1k rows, ≤ 6 hops / ≤ 200k edges, ≤ 50k nodes; **shipped** Cypher ≤ 4 KiB, 10k steps, 1k rows, 20 graphs/tenant, graph state ≤ 256 KiB (≈ 1.5–2k edges), 1k edges/request **[Paid]**; mincut inline ≤ 5M work units, job ≤ 32 MiB (≈ 80k edges), 20 live jobs/tenant, 24 h, 3 attempts (51k edges ⇒ 413 on Free) **[Paid]** |
| M3 synchronous passes | snapshot/export/restore ≤ 2^18 stored floats (682 rows at 384-d) **[Paid]**; queued import segments ≤ 1 MiB |
| Registry (M5) | parts ≤ 16 MiB, ≤ 1000 parts; every write needs Workers Paid (MB-scale hashing) **[Paid]** |
| `limits.cpu_ms` | one script-level value (M1: 30 s) **[U]** DO CPU source; **shipped: none** — the Free plan refuses the block (code 100328) |
| Edge DCR clients | cap `MAX_CLIENTS` (default 10,000) [V] tree; per-IP rate limit and 30-day idle expiry **[U]** (M0.5) |

**Derived per-collection ceiling at 384 dims** (6 shards × per-shard cap): M1 ≈ 47k vectors, M2a ≈
190k if the measured cap lands near 12 MB of codes; per tenant at 20 collections, M1 ≈ 940k.
*Shipped:* ≈ 6 × 25k ≈ 150k `flat` vectors per collection, but at most two full `VectorShard`s stay
resident per isolate, so a full 6-shard collection relies on its shards landing in different isolates **[U]**; `rabitq` 6 × 50k = 300k.

The edge AS consumes **one** slot of the upstream `MAX_REGISTERED_CLIENTS = 500` cap; connectors
register at the edge AS, not upstream. Workers AI calls and analytics jobs are charged to per-tenant
work units (§17); binary inputs are checked against quota before allocation.

## 11. Observability

- **Logs:** `request_id`, `tenant_key`, route/op, status, `took_ms`, `shard`, `aud`, `client_id`,
  `family_id` — never tokens, codes, upstream ids, vectors or metadata.
- **Metrics:** auth failures by reason and `client_id`; JWKS refetches; instantiate time, bundle
  size, lazy-load latency; resident bytes, evictions; leases, quota rejections, work units. **AS:**
  registrations (verified/unverified), consent outcomes, callback failures by reason, grants,
  refresh reuse detections (inside / after grace), token exchanges by adapter, active `kid`.
- **Alerts:** auth-failure or refresh-reuse spike; callback cookie mismatches; DCR bursts; JWKS
  unavailable > 5 min; a new upstream `kid` (accepted, §5.6) and upstream verification failures;
  resident set > 80%; instantiate > 600 ms; any `trust_root_mismatch`. Operator cross-tenant views only via a
  break-glass Worker behind Cloudflare Access (M6).

## 12. Security considerations and threat model (STRIDE)

| Threat | Vector | Mitigation | Residual |
|---|---|---|---|
| **S**poofing: cross-resource token reuse | A token minted for another Cognitum app or resource | RS accepts only `iss = EDGE_ISSUER` and **exact** `aud` = its own resource on every route (no second audience, not even on `/v1/ops`; adapter-audience tokens never reach the gateway's trust); mismatch → 401 re-discovery; edge tokens carry no upstream scopes, and api.cognitum.one rejects them (foreign `iss`) | — |
| Spoofing: upstream token over-privilege | Edge's upstream client holding `mcp:*` would hold a live api.cognitum.one credential | Upstream client registered with identity scopes only; upstream tokens verified, refresh revoked, all discarded, never stored or forwarded | — (§8 delta 2 closed; `UPSTREAM_SCOPES` enforced at load) |
| Spoofing: trust-root substitution | Var edit or dev deploy pointing an issuer, JWKS URL, upstream endpoint, `PUBLIC_ORIGIN` or `FIRST_PARTY_AUDS` elsewhere, on either Worker | Compile-time consts on both Workers; vars only under `dev-issuer`; 503 on mismatch; CI refuses dev deploys | Gateway trust root compiled ([V] 52e2a5c55 `trust_root.rs`); auth-worker issuer and upstream endpoints still vars (§8 delta 6) |
| Spoofing: algorithm confusion / key injection | `alg=none`, HS256, `jku`/`jwk`, DER | Exact ES256, header allowlist, compiled JWKS URL, thumbprint = `kid`, fixed r‖s | — |
| **AS: open redirect** | Crafted `redirect_uri` or error path used to bounce users | Errors before client + redirect validation rendered, never redirected; exact (loopback: port-only) redirect match; `/callback` redirects only to the stored flow's validated URI | — |
| **AS: code injection / login CSRF** | Attacker injects their upstream code or edge code into a victim's browser; stolen edge code replayed | Per-flow `__Host-eaf-{state[0..16]}` cookie whose hash is stored with the flow [V] tree; one-time `state` and codes; 60 s code TTL; code bound to client, redirect, resource and PKCE S256 (both legs); code consumed on any failure; a replayed code revokes its family | — |
| **AS: mix-up with upstream** | Response from another AS accepted as the upstream's, or a client confusing our AS with another | Single upstream with compiled issuer and JWKS URL (keys accepted by thumbprint = `kid`, new `kid` alerted; `ACCEPTED_UPSTREAM_KIDS` pinning only in the §5.5 mode); upstream token `iss` and `aud == UPSTREAM_CLIENT_ID` checked; upstream `iss` parameter checked when sent; our responses carry RFC 9207 `iss`; distinct redirect per AS | Upstream `iss` parameter support **[U]**; auth-worker upstream trust root and pinned kids are vars (§8 delta 6) |
| **AS: consent phishing via open DCR** | Anyone registers an https client and lures a user | Consent always shown; unverified-host marker (§8 delta 7); write/admin never granted without an explicit scope request shown on the consent page, and still ∩ role; clickjacking blocked (`X-Frame-Options: DENY`, `frame-ancestors 'none'`) [V] tree; DCR rate limit, cap, idle expiry; owner `client_id` block (§5.8); CIMD-verified client ids later (Q15) | A user can still approve a malicious app for their own data (as with any OAuth AS) |
| **AS: refresh-token theft** | Stolen refresh token | Rotation with reuse detection revoking the family after a 30 s grace; bound to `client_id` and `resource`; 30 d per token, 90 d family cap; `/revoke`; RS deny by `family_id` | A thief who refreshes first keeps access until the victim's next refresh (after the grace window) triggers reuse detection, or ≤ 90 d; a replay inside the 30 s window gets `invalid_grant` without revocation. Shipped has no grace window (§8 delta 11): every re-presentation revokes, so a benign concurrent refresh also logs the user out |
| **AS: signing-key compromise / rotation** | Leaked `EDGE_AUTH_SIGNING_JWK` mints arbitrary tokens | Worker secret only, generated offline; staged rotation by key order (§5.6); emergency rotation removes the `kid` from JWKS and deny-lists it (≤ 60 s at the gateway) | Tokens minted before detection work until the deny entry lands; adapter resource servers until their next JWKS fetch (≤ 10 min) |
| **AS: availability** | Flooding `/register`, `/authorize`, `/token` at one `AuthStore` DO | Per-IP rate limits; stateless metadata/JWKS; login-rate traffic only | Single DO throughput ceiling; shard at M6 if measured |
| **AS: upstream deprovisioning** | A user disabled or removed upstream keeps refreshing at the edge | Families capped at 90 d; operator action revokes every family of an edge `sub` plus a global `sub` deny (M3); M6 pentest item | Not propagated automatically: up to 90 d of edge access without an operator action |
| **Adapter resource servers** | Adapter-hosted MCP endpoints (§16.1) cannot see the ruvector deny-list | JWKS replaced on every fetch (removed `kid` dead ≤ 10 min), last-good fallback ≤ 1 h there; everything they forward to `/v1/ops` is re-checked against the deny-list | Tenant/family/`sub` denies reach them only via ≤ 15 min token expiry |
| **T**ampering: cross-tenant write | Crafted names/ids; routing bug; D1 query missing a filter | No tenant parameter; hashed DO names with `collection_uid`; in-DO assertion; per-tenant state in `TenantLedger`; typed D1 repository | — |
| Tampering: import / restore injection | Another tenant's R2 object; malformed RVF | Server-minted `upload_id`; manifest `tenant_key`; per-segment sha256; fuzzed decoders; size checks first | — |
| Tampering: `/v1/ops` misuse / token passthrough | An adapter forwards the user's adapter-audience token, or a captured op is replayed or retargeted | `/v1/ops` accepts only `aud = …/v1`; adapters get it by RFC 8693 exchange as confidential `private_key_jwt` clients, scope mapped via the adapter's compiled exchange map ∩ adapter ceiling ∩ `…/v1` scopes, `exp` narrowed, `act` audited; `target` and `tenant_key` echo checked; `op_id` idempotency; scope ∩ role (§16.3) | A compromised adapter holding its private key can exchange tokens it receives into the mapped `ruvector:*` scopes (`team:write` implies `ruvector:write`) within their lifetime; the consent page discloses this mapping (§16.1) |
| **R**epudiation | "I didn't delete that" | `ops` + hash-chained audit with `sub`, `jti`, `family_id`, `approval_ref`; claims, invites, deny writes audited | Audit is one chain per tenant fed by a best-effort Queue send ([V] 52e2a5c55 `audit.rs`): an outage past the DLQ loses events; reads are not audited |
| **I**nformation disclosure | Existence probing; log leakage; cache cross-hits | 404 for foreign resources; no tokens, upstream ids, vectors or clear IPs in logs; cache keys include `tenant_key`; redacted exports | — |
| **D**oS: cross-tenant memory | One tenant's OOM/panic reinitialises the shared instance | Resident-set registry and caps; pre-sized slabs; no-unwrap lint; fuzzing; panic-unwind evaluation | A panic can cold-restart co-located shards until Q12 is resolved |
| DoS: floods / fan-out | Unauthenticated floods, kid-spraying, fan-out amplification | Pre-auth IP limit; failed-token cache; single-flight JWKS refetch; `shards_queried` charging; ≤ 6 shards; step budgets | RL may be per location; no WAF on workers.dev; gateway pre-auth IP limit and failed-token cache not shipped ([V] 52e2a5c55), so kid-spraying costs one signature check per request |
| **E**levation of privilege | Viewer mutating; first claimant in a shared org; scope inflation | Scope ∩ role ∩ route; default-deny route table; explicit audited claim granting nothing pre-existing | Team tenancy needs G5 |
| Supply chain and deploy | Account-wide token; mold `RUSTFLAGS`; pin drift; a git dependency on an unmerged branch | Scoped deploy credentials; CI binding checks; `env -u RUSTFLAGS`; pinned lockfile; `rvlite` pinned by **rev** `c6ece785` (RuVector#1084 still open) | Account shared with the consultant-email stack until G6; move `rvlite` to `main`/a release once #1084 merges |
| M4 residuals ([V] 52e2a5c55, from `edge/HORIZON.json` M4) | Unbounded per-tenant bytes; data surviving removal; shared-isolate blast radius | Per-graph 256 KiB state cap, 20 graphs/tenant, ≤ 20 live jobs/tenant, 24 h job TTL | Graph and job bytes are **not counted in the tenant quota**; there is **no graph/job wipe** on tenant removal (and no tenant deletion at all); the isolate-kill blast radius on Free (a `GraphStore`/`AnalyticsJob` OOM restarting co-located `VectorShard`s) is **unverified** — all M6 items |

Run `npx @claude-flow/cli@latest security scan` and an external review before exposure beyond
staging; a pentest gates GA (M6).

## 13. Gates: out of scope until separately approved

| Gate | What it needs | Earliest milestone |
|---|---|---|
| G1 | **Register the edge AS as an upstream client** at `auth.cognitum.one` via the public DCR endpoint (use authorised by rUv, 2026-09-29): public client, PKCE, `scope = "openid profile email"` (explicit — omitting it grants `mcp:*`), `redirect_uris = ["https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/callback"]`. **Precondition [V]:** at console `ce9ddca` that redirect is not on `ALLOWED_REDIRECT_PREFIXES`/`ALLOWED_EXACT_REDIRECTS`, so DCR returns `invalid_redirect_uri`. G1 therefore includes **one exact-match line** in `ALLOWED_EXACT_REDIRECTS` (console repo review; ADR-124 does not pre-approve it) — or an equivalent hand-seeded client row. No scope, claim or protocol change. | M0.5 |
| G2 | Cloudflare resources in account `501c77f5…`: Workers `ruvector-edge-auth` and `ruvector-edge-gateway` on workers.dev, DO namespaces `AuthStore` (M0.5), `VectorShard`, `TenantLedger` (M1), Rate Limiting bindings, secrets `EDGE_AUTH_SIGNING_JWK`, `IP_HASH_SALT`, `DENY_GLOBAL`; the gateway → auth `EDGE_AUTH` Service Binding. Workers Paid plan. Operator deploys via wrangler OAuth; CI uses a dedicated token scoped to Workers Scripts, DO, D1, R2 and Queues, plus the binding check. **Status ([V] 52e2a5c55):** exercised — both Workers, `AuthStore`, `TenantLedger`, `VectorShard` and (M4/M5) `QuantShard`, `GraphStore`, `AnalyticsJob`, `RegistryRoot`, `RegistryScope`, 8 + 2 Rate Limiting bindings, `EDGE_AUTH_SIGNING_JWK` deployed; `IP_HASH_SALT` and `DENY_GLOBAL` **not set**; the account is still on **Workers Free** (the Paid precondition is open, pending rUv); the gateway and auth rate-limit `namespace_id`s 35101/35102 collide [U]. | M0.5 (auth + gateway), M1 (DOs) |
| G3 | R2 `ruvector-edge-data`, D1 `ruvector-edge-control`, Queues `ruvector-edge-jobs` (+ DLQ) and `ruvector-edge-audit`, Workers AI binding. **Status ([V] 52e2a5c55):** exercised without D1 — one R2 binding `EDGE_DATA` → `ruvector-edge-data` (also M5), Queues `ruvector-edge-ingest` (+ `ruvector-edge-ingest-dlq`) and `ruvector-edge-audit` (+ `ruvector-edge-audit-dlq`), Workers AI `AI`; D1 `ruvector-edge-control` not created (§6.4). Created by the M3 deploy (`f1959f942`, gateway version `07de745c`); **no G3 approval line was recorded before the run** (as for G2). | M3 |
| G4 | Upstream crate PRs: (a) `rvf-index` DistanceOracle traversal, CSR `u32` adjacency, dense ids, header fields, streaming encoder, simd128; (b) `rvlite-core` split / `browser` gate; (c) `ruvector-rabitq` persist v2 **Status:** (a) not needed — HNSW built in-repo (`ruvector-edge-index`); (b) consumed from the unmerged branch `feat/rvlite-browser-feature-gate` of **RuVector#1084**, pinned by rev `c6ece785` (merge still owed); (c) not needed — persist v2 (`rbqx0002`) is in `ruvector-edge-quant`. | M2b, M4 |
| G5 | **Team tenancy only:** an upstream org selector on `/oauth/authorize` and/or an org-role claim (console repo). The former G5a (first-party web client), G5b (RFC 8707 at the upstream AS) and G5c (`ruvector:*` upstream scopes) are **withdrawn** — the edge AS provides all three. | M6 |
| G6 | Production hostname: a zone in this account, a `cognitum.one` delegation (e.g. a future `api.cognitum.one` path — not in this account today), or preferably a separate Cloudflare account. Changing hosts changes every canonical resource URL and the edge issuer (a re-login for all clients). | M6 |

## 14. Alternatives considered

1. **Resource servers directly against `auth.cognitum.one` (previous revision).** Rejected: `aud =
   client_id` without RFC 8707 pins REST to one client id, matches connectors only by id (their
   tokens also work at api.cognitum.one), and blocked browser login and mutating MCP tools until G5b.
2. **Pass upstream tokens through the edge AS** — still `aud = client_id`, cross-valid upstream.
3. **RFC 8693 at the upstream AS (`EXCHANGE_PROFILES`)** — only for a Google-SA actor in
   `mcp_exchange_actors`, which a Worker lacks.
4. **Adapters forwarding their own-audience tokens to `/v1/ops` (`OPS_AUDS`)** — token passthrough
   (MCP security best practices) and a confused deputy; replaced by the edge exchange grant (§5.6).
5. **Vectorize as the tenancy primitive** — one shared index, `topK` ≤ 50 with metadata [V]; kept as
   a possible `index=vectorize`. **Workers for Platforms** — not enabled [V].
6. **D1 as data plane or shared per-tenant tables** — isolation would hinge on every `WHERE`.
7. **An HMAC-signed internal context header** — adds nothing over re-verifying the user token.
8. **One global DO, or one DO per tenant** — ~500–1,000 req/s per object, one 128 MB isolate [V].
9. **tract-onnx in wasm; linking `rvf-wasm`/`rvf-runtime`; HNSW on today's `rvf-index`** — size,
   own allocator/paths/`SystemTime`, and resident f32 per visited node; hence M2a, then G4a.
10. **`transaction()` spanning a ledger reservation** — no transaction spans two DOs; leases instead.
11. **Tenant-admin approval of connector ids** — unnecessary with resource-bound tokens and consent;
    owners keep a per-tenant `client_id` block (§5.8).

## 15. Phased implementation plan

**Status at `52e2a5c55`** (tracker `edge/HORIZON.json`; checkboxes below are the original plan and
are not ticked individually). M0 done (fuzz criteria met at `14e3f77e3`, §9 item 4). M0.5–M5 **deployed to
workers.dev pending G1**: every acceptance item that needs a real login (claim, CRUD, connectors,
refresh/revoke, 413/429 live, latency) is unrun, and these items are **not met by the code**
regardless of G1: M0.5/M1 `http://localhost` loopback and the 30 s refresh grace (§8 deltas 4, 11);
M1 pre-auth IP limit, failed-token cache and `ruvector-edge-cli`; "four 12 MB shards resident"
(shipped two 14 MB `VectorShard`s per isolate); M3 global deny and usage reconciliation (no D1),
signed result pages, and 100k-vector imports / snapshots on Free (**[Paid]**); M4 the 51k-edge
min-cut ends `413` on Free (**[Paid]**) and graphs cap at 256 KiB, not 200k edges; M5 `dry_run` on
`rvf_import`, `ruvector:publish` absent from the allowlist, rvf CLI imports only with
`--batch-size` ≤ 680 at 384-d, and registry writes need Paid. M2b and M4 no longer wait on G4a/G4c
(built in-repo); G4b is consumed as an unmerged branch.

### M0: Cores including authz (no deploy, no Cloudflare or provider changes)

- [ ] `ruvector-edge-auth`: JWS, JWKS cache, typed claims for both kinds, exact-`aud` policy, edge
      subject, compile-time trust root, route table with the §5.3 vocabulary, PRM documents and
      challenges.
- [ ] `ruvector-edge-authz`: DCR (§5.3 defaults), authorize (resource required, grant rule), PKCE,
      codes (replay revokes), federation, §5.2 minting, refresh rotation with grace and reuse
      detection, family cap, revocation, RFC 8414 + JWKS.
- [ ] `ruvector-edge-tenancy`: `from_verified(claims, role, surface)`, `tenant_key` from
      `upstream_iss`, `collection_uid` DO names, leases (largely [V] tree).
- [ ] Close every §8 delta; both Workers build for wasm32 (`env -u RUSTFLAGS`); §9.1 matrix; fuzz
      targets for JWS, JWKS, DCR JSON, token form, `ResourceUrl`.

**Acceptance:** native tests pass (minted-token golden test, refresh grace and reuse detection,
cookie binding, scope grant rule, 401 on `aud` mismatch, upstream-mode ID-token and
`typ=inference` cases, trust-root mismatch on both Workers); the route-table gap test
passes; every file < 500 lines; fuzzers run 10 min with no panic.

### M0.5: Deploy `ruvector-edge-auth` + gateway to workers.dev (gated: G1, G2)

- [ ] G1: land the exact redirect-allowlist entry, register the upstream client with identity
      scopes, record `UPSTREAM_CLIENT_ID`.
- [ ] Generate the ES256 key offline; `wrangler secret put EDGE_AUTH_SIGNING_JWK`.
- [ ] `AuthStore` ports and all §7.1 endpoints; gateway PRM, `/v1/health`, `/v1/me`, the `/v1/mcp`
      401 challenge; deploy both Workers to `*.cognitum-consulting-mail.workers.dev`.
- [ ] Resolve the M0.5 **[U]**s: upstream `iss` parameter, scopes and `grant_types` Claude,
      ChatGPT, Claude Code and MCP Inspector send at DCR/authorize, whether they follow the 403
      step-up challenge, DCR limits.

**Acceptance (live):** AS metadata, JWKS and both PRM documents name the edge issuer (bare origin
404); a browser user completes DCR → consent → upstream login → callback → token and `GET /v1/me`
returns the edge `sub` and `tenant_key`; the token carries exactly the §5.2 claims and is refused by
the other gateway resource (401 + `resource_metadata`) and at `api.cognitum.one/v1/mcp`; replayed
code, replayed `state` and a mismatched flow cookie are refused; refresh rotates, a concurrent
refresh inside the grace window does not revoke, reuse after it does, `/revoke` works; Claude.ai,
ChatGPT, Claude Code (`http://localhost`) and MCP Inspector go from the `/v1/mcp` 401 through DCR
and consent to a `/v1/mcp` token with a refresh token; no upstream token in storage or logs.

### M1: rv-vector MVP, exact flat scan, connectors end to end

- [ ] Router with pre-auth IP limit and failed-token cache; `/v1/usage`.
- [ ] `TenantLedger`: claim, members, tenant deny (incl. `client_id`), catalog with
      `collection_uid`, leases, work-unit counters.
- [ ] `VectorShard`: lease + coalesced writes with `dry_run`, get/fetch, batched delete, exact flat
      scan with post-filter, `ops`, idempotency, audit, `timers`, paged cold load, resident-set
      registry with eviction.
- [ ] `/v1/ops` (§16.3) and the RFC 8693 exchange grant for confidential adapter clients (§5.6);
      `/v1/mcp` with `initialize`, `tools/list` and the collection and vector tools.
- [ ] Rate Limiting (locality **[U]**), DO CPU source **[U]**, panic policy (Q12);
      `ruvector-edge-cli`.

**Acceptance:** locally and live: login → claim → create → upsert 1k × 384 → exact top-10 equals
native brute force; tenant B gets 404 on tenant A's names; a non-member gets 403; no upsert without
`ruvector:write`, even as owner; foreign-`iss`, wrong-`aud`, upstream (mode off) and ID tokens
rejected; 413 at the shard cap, 429 at the rate limit; four 12 MB shards resident, a fifth evicts;
instantiate < 600 ms, lazy-load p95 < 1 s; Claude.ai and ChatGPT connectors run `vector_query`,
step up to `ruvector:write` on consent and run a `dry_run` `vector_upsert`; `/v1/ops` takes only
exchanged `/v1` tokens (adapter audience → 401), refuses a wrong `target` or `tenant_key`, never
applies an `op_id` twice, and audits `act.sub`.

### M2a: int8 flat scan with f32 rerank (today's crates)

- [ ] Quantizer policy (fixed-range cosine; reservoir l2/dot with epochs and sliced re-encode); code
      arena; u8 distance; paged rerank; measured cap.

**Acceptance:** recall@10 ≥ 0.95 vs `FlatIndex` on 20k × 384 random vectors and one real-embedding
set; ≥ 3 shards per isolate at the measured cap; a quantizer change re-encodes without blocking
queries for more than one slice.

### M2b: HNSW (gated on G4a)

- [ ] Land G4a; streaming chunked flush with sha256 and epoch fallback; dense-iid compaction; ≤ 64
      sync inserts; measure insert cost, rebuild CPU and peak memory; fuzz chunk decoding.

**Acceptance:** recall@10 ≥ 0.95; warm query p50 < 10 ms **[U]**; ≥ 2 full shards per isolate;
lazy-load p95 < 1 s with ≤ 200-op replay; a corrupted chunk falls back with identical results; a
gapped-iid encode fails loudly.

### M3: Durability, ingest, embeddings, control plane, result pages (gated on G3)

- [ ] Streamed snapshots/exports with witness manifests and `redact`; range-read restore; multipart
      uploads; import by `upload_id`; signed result pages (§17), private by default.
- [ ] D1 control plane (typed repository; AS family revocations feed global deny); Queue ingest with
      DLQ; Workers AI; audit shipping; fuzz rvf-wire; confirm D1 limits **[U]**.

**Acceptance:** a kill-and-restore drill returns byte-identical results; cross-tenant restore, a
tampered snapshot and a foreign `upload_id` are refused; a 100k-vector import stays within quota and
the isolate cap; a global deny takes effect within 60 s; usage reconciles within 1%; a result page's
witness verifies offline and a redacted key is absent from it; fuzzers run 1 h clean.

### M4: rv-quant, rv-graph, rv-mincut (gated on G4b/G4c)

- [ ] `QuantShard` (50k × 384 cap until persist v2); `GraphStore` with Cypher budgets; mincut inline
      and as jobs; two-level fan-out if needed; rayon wasm test **[U]**; split into per-service
      Workers if instantiate time or bundle size requires (Q8).

**Acceptance:** a 50k × 384 rabitq cold load fits CPU and memory; a viewer cannot run a mutating
Cypher query; mincut on 50k edges equals the native crate; an exhausted budget returns 413, not 500.

### M5: rv-registry and the full rv-mcp tool set

- [ ] Registry (immutable versions, fuzzed streaming validator, sha256 manifest, import;
      `ruvector:publish` for public packages); `rvf_*` MCP tools; MCP App-friendly outputs for the
      control room (§17).

**Acceptance:** every mutating MCP tool needs `ruvector:write` + editor and honours `dry_run`; a
`/v1/mcp` connector token gets 401 (`audience mismatch`) on `/v1` REST; an RVF from the `rvf` CLI imports via `upload_id`
and queries correctly; the Launch Doctor checker (§9.1) passes against the live hosts.

### M6: Hardening and GA (gated on G5, G6)

- [ ] Team tenancy (G5); production hostname (G6), workers.dev denied in production, WAF;
      `AuthStore` sharding if measured necessary.
- [ ] Break-glass operator Worker behind Cloudflare Access; SLOs; 30-day hard-purge SLA.
- [ ] External pentest: foreign-app and upstream tokens, consent phishing, code injection, refresh
      theft, token-exchange abuse, upstream deprovisioning, key rotation, trust-root substitution,
      kid-spraying, oversized RVF, Cypher bomb, lease races, cross-tenant OOM.

**Acceptance:** no open high or critical findings; p50 query < 30 ms and p99 < 150 ms at the edge
for single-shard 384-dim collections **[U]**; this ADR moves to Accepted.

## 16. Adapter contract (for `integrations/cloudflare-chatgpt` and RuFlo AI Team)

Audience: the ChatGPT coordinator (`integrations/cloudflare-chatgpt`, PR #1062, D1
`ruvector-chatgpt`) and any RuFlo front end that fronts ruvector.

**Adoption status ([V] 52e2a5c55, 2026-09-29).** No coordinator acceptance or receipt of this
contract is recorded, and it changed at reconciliation (`409 op_replayed` replaces `422`; raw-body
idempotency binding; `Idempotency-Key` must equal `op_id`). PR #1062 does not yet reference the edge
issuer, `/v1/ops`, `op_replayed` or the token exchange. `CONFIDENTIAL_CLIENTS` is empty
(`edge/auth-worker/wrangler.toml:106`), so no adapter can use the RFC 8693 exchange until its
`private_key_jwt` client is registered there. Owed: send this §16 delta to the coordinator (Slack
`#swarm` + x.ruv.io) and record the receipt in `edge/HORIZON.json`. RuFlo front ends also depend on
RuVector#1063/#1064 (`@ruvector/core` `VectorDb` sharing `./ruvector.db` across tenants), which
must be fixed adapter-side before they front multi-tenant data.

### 16.1 Identity contract

| Item | Value |
|---|---|
| Issuer (`iss`) | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev` |
| AS metadata | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/.well-known/oauth-authorization-server` |
| JWKS URL | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/.well-known/jwks.json` (ES256, P-256, `kid` = RFC 7638 thumbprint). A Worker in the same account fetches it through a **Service Binding** to `ruvector-edge-auth` (workers.dev → workers.dev subrequests are refused, Cloudflare 1042 [L]; the gateway does this [V] tree). Cache ≤ 10 min; **replace** the set on every successful fetch; refetch on unknown `kid` at most once per 30 s; last-good fallback ≤ 1 h |
| Token format | JWS, header `alg=ES256`, `typ=at+jwt`; claims exactly §5.2 (`act` only on exchanged tokens) |
| Stable subject | `sub` = `"es1_" + base32lower(sha256("ruvector-edge/sub/v1\|" + upstream_iss + "\|" + upstream_sub))[0..26]` (`subject::edge_subject`). Key memberships on **(`iss`, `sub`)**; never on `client_id`, `jti` or email |
| Tenant claims | `upstream_iss`, `org_id`, `workspace_id` → `tenant_key` (§4.1, formula §16.2); also returned by `GET /v1/me` |
| Scope vocabulary | **Per resource** (§5.3). Gateway (`…/v1`, `…/v1/mcp`): `ruvector:read`, `ruvector:write`, `ruvector:admin` (`/v1` only), `ruvector:publish` (M5), `offline_access`; default grant `ruvector:read`, write only when requested on consent; step-up via 403 `insufficient_scope`. Adapter resource: its **own** vocabulary, never `ruvector:*` — `https://team.ruv.io/mcp` has `team:read`, `team:write`, `team:run`, `offline_access`, default grant `team:read` (+ `offline_access`); what each `team:*` scope permits is defined and enforced by the adapter. A scope of the other vocabulary is dropped, never minted (nothing left is `invalid_scope`); `team:*` must be registered explicitly at DCR (the default ceiling is `ruvector:*` only). A new adapter vocabulary or resource is a reviewed code change (`authz/resource.rs` `Vocabulary` and the compiled `RESOURCE_VOCABULARIES` URL binding, enforced at load) plus its allowlist entry |
| Canonical audiences | MCP `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp`; REST and `/v1/ops` `…/v1`. An adapter that serves its **own** MCP endpoint gets its canonical URL compiled into `resource::RESOURCE_VOCABULARIES` with its own vocabulary (reviewed code change) and added to `RESOURCE_ALLOWLIST` (reviewed AS config), and verifies inbound tokens exactly as §5.4 with that URL as `aud` and that vocabulary as `scope`; the gateway never accepts that audience (401 `invalid_token`, decided on `aud` before any scope). Registered today: `https://team.ruv.io/mcp` (RuFlo AI Team) |
| Calling ruvector | Exchange (§5.6, M1): the adapter is an operator-registered **confidential** client (`private_key_jwt`, per-adapter public JWK, exchange grant only). `POST /token` `grant_type=urn:ietf:params:oauth:grant-type:token-exchange`, `subject_token=<user's token, aud = adapter resource>`, `subject_token_type=urn:ietf:params:oauth:token-type:access_token`, `resource=…/v1`, optional narrower `scope` → a `…/v1` token with the same `sub`/tenant claims, `exp ≤` the subject token's, `act.sub` = adapter `client_id`, and `scope` = the adapter's compiled **exchange map** of the subject token's scopes ∩ the adapter client's ceiling ∩ the `…/v1` scopes (∩ the requested `scope`, if sent). team.ruv.io map: `team:read → ruvector:read`, `team:write → ruvector:write`, `team:run → ∅` (run is an adapter capability; its jobs need `team:write` to write vectors), `offline_access` dropped; an empty result is `invalid_scope`. Because the subject token was consented as `team:*`, the consent page for team.ruv.io says, whenever it shows a `team:read`/`team:write` grant, that it also lets the team.ruv.io service read (`ruvector:read`) / add, change and delete (`ruvector:write`) the user's ruvector data, derived from the compiled map (`resource::TEAM_EXCHANGE_MAP`, `consent::ruvector_data_notice`) [V] tree. The disclosure ships in the same release that first allows `team:*` for team.ruv.io, so no `team:*` family predates it (until then team.ruv.io listed `ruvector:*`, and `/authorize` answers `temporarily_unavailable` until G1); the M1 exchange therefore needs no consent-version check on the family, and any later change to the map that widens it must add one (or re-consent). An adapter that is only an OAuth client (no own resource) requests `resource=…/v1` in its own flow instead |
| Revocation | Adapter resource servers do not see the ruvector deny-list: `kid` removal (≤ 10 min) and token expiry (≤ 15 min) only (§5.8, §12) |
| Host change | G6 changes the issuer and every audience; adapters pin them at build/deploy review (as §5.4), so a host change is a reviewed redeploy plus one re-login |

### 16.2 Tenant → Durable Object mapping

The backend resolves tenants itself; an adapter never addresses a DO and never binds a ruvector DO
namespace.

- `tenant_key = base32lower(sha256("v1|" + upstream_iss + "|" + org_id + "|" +
  workspace_id))[0..26]` (no padding).
- `TenantLedger` DO name = `hex(sha256("v1|" + tenant_key + "|ledger"))`.
- `VectorShard` DO name = `hex(sha256("v1|" + tenant_key + "|vector|" + collection_uid + "|" +
  shard))`, with `collection_uid` = **32 lowercase hex chars** assigned by the ledger (v2 formula,
  §4.3) and `shard` = **decimal** index (0..5). Other services use their own service token
  (`quant`, `graph`, …) in the same position. As shipped ([V] 52e2a5c55): a `rabitq` collection's
  `QuantShard` uses `quant` with the collection's own uid; `GraphStore`/`AnalyticsJob` use `graph` /
  `mincut` with a **name-derived** uid (§4.3); the registry DOs are keyed by `@scope`, not tenant.

### 16.3 Private operation contract (`POST /v1/ops`)

Transport is a Service Binding from the adapter Worker to `ruvector-edge-gateway` (required inside
this account, 1042 [L]; public HTTPS only from outside it). **Trust never comes from the
transport:** every call carries a user token with `aud = …/v1` (exchanged, §16.1) and is fully
re-verified (§5.4); an adapter-audience token gets 401. Scope ∩ role applies exactly as on REST.

```
POST /v1/ops
Authorization: Bearer <user's …/v1 edge token>
Content-Type: application/json
Idempotency-Key: <op_id>

OpRequest  { "v": 1,
             "op_id": "<26 chars: ULID, or a deterministic id for re-runnable jobs>",
             "target": "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/ops",
             "tenant_key": "<26 chars, the adapter's view of the caller's tenant>",
             "op": "tenant_me" | "collection_list" | "collection_create" | "vector_upsert"
                   | "vector_query" | "vector_fetch" | "vector_delete" | "usage_get",
             "args": { …same schema as the matching REST body… },
             "dry_run": false,
             "approval_ref": "<optional, ≤ 128 chars, recorded in audit>" }

OpResponse { "v": 1, "op_id": "…", "ok": true,
             "result": { … }, "usage": { "work_units": n, "rows": n, "bytes": n } }
         | { "v": 1, "op_id": "…", "ok": false,
             "error": { "code": "<below>", "status": 4xx|5xx, "retry_after_s": n? } }
```

| Error `code` | HTTP | Meaning |
|---|---|---|
| `invalid_token` | 401 | token failed §5.4, including any `aud` ≠ `…/v1` (`WWW-Authenticate` names the `/v1` PRM) |
| `insufficient_scope` | 403 | scope missing; `WWW-Authenticate` carries the needed `scope` |
| `role_required` / `not_claimed` | 403 | role check |
| `target_mismatch` | 400 | `target` ≠ this route's canonical URL |
| `tenant_mismatch` | 403 | `tenant_key` ≠ the key derived from the token |
| `op_replayed` | 409 | `op_id` seen with a different body (sha256 of the **raw** request bytes); same body returns the stored response. Only successful ops are remembered, so a failed op re-executes on retry |
| `conflict` | 409 | the same `op_id` is still in flight (reserved in the lookup's DO turn, 120 s); `retry_after_s` 1 ([V] 52e2a5c55 `ops.rs:128-150`) |
| `unknown_op` / `invalid_request` / `dimension_mismatch` | 400 | request shape |
| `not_found` | 404 | absent or foreign resource |
| `quota_exceeded` / `budget_exceeded` / `payload_too_large` | 413 | limits |
| `rate_limited` | 429 | with `retry_after_s` |
| `jwks_unavailable` / `shard_unavailable` | 503 | retryable |

**Replay and target verification.** Tokens live ≤ 15 min (exchanged ones never outlive their
subject token) and are checked against `exp` and the deny-list; `op_id` is remembered 24 h per
`(tenant, sub)` (the idempotency table); `target` binds the call to this endpoint so a captured
request cannot be redirected to another service; `tenant_key` echo catches an adapter-side mapping
error; `act.sub` records which adapter acted. Calls with no user token (adapter cron jobs) are not
supported yet (Q13).

*Shipped* ([V] 52e2a5c55, `ops.rs:80-170`, `ops/types.rs:12,97-101`): the check order is shape
(`v` 1, `op_id` exactly 26 ASCII alphanumerics, `approval_ref` ≤ 128) → `Idempotency-Key` header,
if sent, must equal `op_id` (400 `invalid_request`) → `target` → `tenant_key` → known op → scope
→ role → `op_id` reservation (dry runs and read ops are not reserved or remembered) → execute. The
body cap is the 1 MiB upsert cap. No adapter is registered yet (`CONFIDENTIAL_CLIENTS` empty).

### 16.4 Storage boundary, migration and rollback

- **Boundary.** The adapter keeps D1 `ruvector-chatgpt` and `memberships(issuer, subject,
  tenant_id, role)`; the backend never reads or writes it. Backend tenancy and roles are §4 (ledger,
  keyed by edge `sub`); the adapter's `role` is presentation only (mirror it from `GET /v1/me`) and
  can never exceed the backend's, which checks every call.
- **Mapping.** Adapter `issuer` = edge `iss`, `subject` = edge `sub`, `tenant_id` = `tenant_key`
  (§16.2 or `GET /v1/me`) — pure functions of the token, so no join table.
- **Migrating adapter rows** keyed on upstream identities: `(auth.cognitum.one, upstream_sub)` →
  `(edge iss, edge_subject(…))` with the exported `ruvector-edge-auth` function; `tenant_id` becomes
  the `tenant_key` at the user's next sign-in.
- **Migrating vectors** from the D1 JSON-text table: batched `vector_upsert` ops with deterministic
  `op_id`s (safe to re-run) into collections with `origin = "ruvector-chatgpt"`; M3 adds bulk
  `:import`. Dual-read until query parity is shown on a sample.
- **Rollback.** The adapter keeps its D1 data read-only for 30 days and flips its read path back;
  backend collections with that `origin` are deleted in bulk. Backend schema migrations are additive
  and versioned (`schema_ver` in `meta`, applied lazily; DO classes via `[[migrations]]` tags), so
  rolling back a Worker version leaves data readable. Applied gateway tags, append-only and never
  edited or reordered: `v1` (`TenantLedger`, `VectorShard`), `v2-registry` (`RegistryRoot`,
  `RegistryScope`), `v-m4-quant-graph` (`QuantShard`, `GraphStore`, `AnalyticsJob`); auth `v1`
  (`AuthStore`) ([V] 52e2a5c55 `edge/gateway/wrangler.toml:106-134`). Rolling the gateway back
  past a tag would orphan that tag's classes; stored registry names are re-parsed on read, so a
  reserved-word list change is itself a schema migration (Shipped-vs-designed #5).

## 17. Downstream products

The brief
[`docs/research/ruvector-edge-services/product-brief-ruflo-ai-team.md`](../research/ruvector-edge-services/product-brief-ruflo-ai-team.md)
(author-reported marketplace data) names **RuFlo AI Team** as the flagship — a packaged specialist
team with an inline MCP App control room — and **MCP Launch Doctor** as the fastest wedge: turning
an API or MCP repo into a directory-ready Claude plugin, including OAuth and RFC 9728 validation.

| Product need | Edge service |
|---|---|
| Accurate OAuth + RFC 9728 challenges on every MCP surface | `ruvector-edge-auth` (resource-bound tokens, RFC 8414 + PRM), §5 |
| Launch Doctor's OAuth/9728 validator | the §9.1 RFC 8414/9728/7591/PKCE conformance suite, packaged as a checker |
| Durable team memory, recall, evidence search | rv-vector / rv-quant collections, one DO per tenant collection shard |
| Signed, shareable, forkable result pages | RVF export with witness manifest (`/v1/results`), read-only per tenant, public only by opt-in |
| Work-unit metering and free-tier caps | per-tenant quota and work-unit counters (`TenantLedger`, `/v1/usage`, `OpResponse.usage`) |
| Control-room MCP App | `/v1/mcp` behind the edge AS, tenant-scoped |

Design constraints the edge services must support for approval-safe clients:

- **Approval-gated writes:** `ruvector:write` (never granted without an explicit scope request shown
  on the consent page) plus editor; every mutating route, op and MCP tool supports `dry_run` and
  carries `destructiveHint`; `approval_ref` is recorded in the audit chain.
- **Redaction:** exports and result pages take an explicit key redaction list with a preview;
  nothing public by default; logs never carry payloads.
- **Work-unit metering:** rows, bytes, `shards_queried`, embedding calls and job CPU roll into one
  per-tenant work-unit counter, so free allowances are set from measured p50/p95 cost.
- **Signed result pages:** the RVF manifest (per-segment sha256 + audit-chain head) is signed with a
  dedicated result-signing key (not the token key) and verifiable offline.
- No money movement, no advertising, and third-party content fenced as untrusted data — enforced by
  the products; the edge services expose only narrow, factual tools.

**Shipped support ([V] 52e2a5c55).** The edge AS, per-resource PRMs and the team.ruv.io adapter
resource (`team:*` vocabulary, exchange map) are live; memory is `flat`/`hnsw`/`rabitq`
collections; `destructiveHint`/`readOnlyHint` annotations are on every MCP tool; per-tenant usage
and work units are in `TenantLedger` (`/v1/usage`, `OpResponse.usage`); exports carry the witness
manifest and a `redact` key list (≤ 32). **Not yet:** `/v1/results` and the result-signing key,
`dry_run` on REST collection create, tenant claim (REST and MCP), collection drop, member/deny
admin routes, graph REST create/delete/edges, `rvf_import`, M3 and registry routes (so "every
mutating tool supports `dry_run`" is not met), `approval_ref` outside `/v1/ops`, the packaged Launch Doctor checker, and
the Workers Paid plan the free-tier caps assume (**[Paid]** budgets throughout).

## 18. Open questions

1. **Q1 DO SQLite limits**, **Q2 loopback redirects**, **Q6 `ruvector-chatgpt` D1** — resolved
   (§1.2, §5.6, §16.4). **Q3 role source** — closed negative [V]; roles are ruvector-local (§4.2).
2. **Q4 Tenant granularity** — needs G5; ask the console team whether invited team orgs are ever
   reachable given `first_org_for_user`.
3. **Q5 Production hostname** — separate account preferred (G6).
4. **Q7 Plan tiers and work-unit pricing** — consistent with 6 shards × cap.
5. **Q8 Split the data-plane script** when instantiate time > 600 ms, the bundle nears the limit, or
   an M4 crate is added.
6. **Q9 G1 form** — one exact `ALLOWED_EXACT_REDIRECTS` line or a hand-seeded client row, and who
   reviews it in the console repo.
7. **Q10 MCP transport** — is Streamable HTTP POST-only enough for ChatGPT and Claude, or is SSE on
   `GET /v1/mcp` needed?
8. **Q11 Connector scope and `grant_types` requests, step-up behaviour** and **Q14 upstream RFC
   9207 `iss` support** — measured at M0.5 **[U]** (ID-token contents no longer matter: it is
   ignored, §5.6).
9. **Q12 Panic mode** — `panic = "abort"` vs nightly `--panic-unwind` (M1).
10. **Q13 Userless adapter calls** — scheduled adapter work without a user token; the natural
    extension is a client-credentials grant for the same confidential adapter clients (§5.6).
11. **Q15 Client ID Metadata Documents** (MCP 2025-11-25) — support URL `client_id`s (https only,
    SSRF guard, size cap, `redirect_uris` check, caching) and show "verified" only for CIMD ids on
    `chatgpt.com`, `claude.ai`, `claude.com`; narrows the open-DCR phishing surface (M5/M6).
12. **Q16 Connectors on team.ruv.io and scope requests across vocabularies** **[U]** (M0.5 connector
    checks). (a) A connector that registers **without** `scope` gets the `ruvector:*` default ceiling
    and therefore `invalid_scope` on team.ruv.io; either team.ruv.io's docs tell connectors to
    register `team:*`, or the default ceiling grows a per-resource default (reviewed change). (b) **Closed:** a
    client that requests the AS-metadata `scopes_supported` (the union) instead of the PRM or
    challenge `scope` gets its resource's share: rule (2) of the §5.3 grant rule drops the other
    family's scopes instead of refusing them [V] tree (`team_vocabulary.rs`
    `as_metadata_union_request_gets_each_resource_share`); only (a) remains open.

13. **Q17 Workers Paid (G2 precondition).** Every **[Paid]** limit above (inline import 512 KiB,
    2^18-float sync passes, 1 MiB queued segments, 50k-row quant shards, 256 KiB graphs, 32 MiB /
    51k-edge min-cut jobs) was sized to Workers Free's 10 ms. Conversely every **[needs Paid]**
    budget (VectorShard lazy load, flush, the 20 s HNSW rebuild budget, the 14 MB per-shard cap,
    all registry writes) needs Paid **before live M2 load** or any registry write. After the upgrade, which are raised by config and which need the resumable-job designs
    (snapshot/export/restore as queued jobs, delta-persisted graphs)?
14. **Q18 D1 control plane (§6.4)** — build it (global deny, `usage_daily`, plans) or record that
    per-tenant ledgers plus key rotation are enough through GA; M3 acceptance depends on it.
15. **Q19 Registry scope namespace** — scopes are global and first-claimer-wins (with brand and
    look-alike refusals); who arbitrates disputes, and is the reserved list frozen (a change breaks
    stored names)?
16. **Q20 Idempotency body binding** — keys bind raw bytes; confirm clients (CLI, adapters) resend
    byte-identical bodies on retry, or move to a canonical JSON hash (a compatibility change for
    stored keys, ≤ 24 h).

## References

- Upstream: live `https://auth.cognitum.one/.well-known/{oauth-authorization-server,jwks.json}`;
  console `ce9ddca` `services/identity/src/{jwt.rs, identity.rs, members.rs, oauth/*.rs}`; api
  `9295444` ADR-103, `src/gateway/mcp-oauth.ts`; website `plans/ADR-124-…`.
- RFC 6749, 6750, 7009, 7523, 7591, 7636, 7638, 8252, 8414, 8693, 8707, 9068, 9207, 9457, 9700,
  9728; OAuth 2.1 draft; MCP authorization spec 2025-06-18 / 2025-11-25 and its security best
  practices.
- workers-rs `b57ba6e`; Cloudflare docs (Workers and DO limits, SQLite API, Rate Limiting, Cache
  API, Vectorize).
- In-repo: `crates/ruvector-edge-{auth,authz,tenancy}`, `edge/{gateway,auth-worker}`,
  `crates/rvf/*`, `ruvector-{rabitq,mincut,core}`, `rvlite`, and
  `docs/research/ruvector-edge-services/product-brief-ruflo-ai-team.md`.

## Review log

**2026-09-28, first revision.** Resolved: trust root changeable at deploy; RFC 9728 shape; org
picked with no role source (audited claim, default-deny invitations); tenant-writable global deny;
shared D1 control state; per-isolate memory; HNSW on `rvf-index`; cross-DO transactions; streaming
flushes and exports; quota/DoS gaps.

**2026-09-29, reconciliation with the edge-AS decision (rUv).** `auth.cognitum.one` mints `aud =
client_id` and ignores RFC 8707, so ruvector runs its own AS (§2, §5 rewritten; new M0.5, §16,
§17; G5 reduced to team tenancy). Removed: REST pinned to `FIRST_PARTY_AUDS`, "`aud` must equal
`client_id`", approved `dcr-*` ids, read-only MCP until G5b, G5a/b/c, "no browser login before M6".
**Conflict recorded:** upstream DCR refuses the edge callback at `ce9ddca`, so G1 carries one exact
redirect-allowlist line (Q9).

**2026-09-29, second review pass (spec + code re-read).** Resolved: token passthrough (`OPS_AUDS`
removed; `/v1/ops` takes only `…/v1` tokens from an RFC 8693 exchange with `act`); `aud` mismatch
→ 401 re-discovery; connector scopes (DCR ceiling read+write+offline, `/authorize` default read,
dropped-not-refused scopes, `grant_types` default both, HTTP 403 step-up, `localhost` loopback);
30 s refresh grace; code-replay family revocation; clickjacking/Referer headers; per-flow cookie;
unpinned upstream keys on the login path; id_token ignored; deprovisioning residual; trust roots of
both Workers as consts; `EDGE_AUTH_SIGNING_JWK` rotation by key order; bare PRM 404; CORS; `/v1/mcp`
challenge at M0.5; drop collection = owner; CIMD (Q15). Now **[V] tree**: `collection_uid` DO
names (v2 uid), `upstream_iss` and edge `sub` in minted tokens, scope ∩ role in `from_verified`,
consent + cookie binding, `MAX_CLIENTS`, AS CORS, code-replay revocation. **Open code deltas:**
the eleven in §8.

**2026-09-29, reconciliation with shipped code (`52e2a5c55`, M0–M5 deployed, Free plan).** Added
the Shipped-vs-designed table and **[Paid]** markers; corrected §1.2, §3, §4.3, §5.3, §5.6, §5.8,
§6.1–§6.5, §7, §7.2, §8, §10, §12, §13, §15–§18 in place. Recorded rationale: `409 op_replayed`
replaces `422 idempotency_mismatch` (one code for REST and `/v1/ops`; raw-body binding; only
successes remembered); approximate min-cut answered exactly (`ApproxMinCut` ≈ 0.10–0.33× exact, RuVector#1085);
`flat` = int8 + f32 rerank as the default (recall 1.000) with HNSW opt-in at `ef` 512/1024;
per-class split of the 56 MB isolate cap; scope-keyed registry blobs and a global scope namespace;
one R2 binding, `ruvector-edge-ingest`/`-audit` queues, no D1; rabitq refused for snapshot/export/
restore; every 10 ms-driven cap listed as **[Paid]**. Recorded as still open in code: `localhost`
loopback, refresh grace, `subjects`, auth-worker trust-root consts, runtime route table, global
deny/D1, pre-auth IP limit and failed-token cache, `dry_run` on the remaining mutating tools,
`/v1/results`, `GET /v1/audit`, `ruvector:publish` in the allowlist, the rate-limit
`namespace_id` collision [U], and the CLI. No code, config or deployment changed in this pass.

**2026-09-29, review of the reconciliation (finalize pass).** Corrected against code: the registry
manifest mirror is *designed, not written* (§6.3 keys repaired to `…/@{scope}/{name}/…`); a new
**[needs Paid]** marker separates budgets sized for Paid's 30 s (VectorShard lazy load, flush,
`REBUILD_BUDGET_MS`, the 14 MB shard cap, registry writes) from the Free-sized **[Paid]** ones, and
§1.2/Q17 no longer call every budget Free-sized; `dry_run` is absent on REST collection create,
tenant claim and the admin routes; only `vector_upsert`/`vector_delete`/`graph_mutate`/`rvf_import`
are `destructiveHint: true`; `/v1/ops` dry runs and reads still look up `op_id`; ApproxMinCut error
is ≈ 0.10–0.33× (RuVector#1085). Added: the unset `IP_HASH_SALT` (empty-salt fallback) as a G1
precondition, M4 residual risks (§12), G3 provenance, §16 adoption status (no coordinator receipt;
`CONFIDENTIAL_CLIENTS` empty; RuVector#1063/#1064), and fuzz evidence (§9: ten targets, 10 min /
1 h clean at `14e3f77e3`). One build change: `rvlite` is now pinned by **rev** `c6ece785` instead of the PR
#1084 branch (resolved commit unchanged, `edge/Cargo.lock` source strings only); the
`keys.rs` doc no longer claims the mirror is written. Nothing deployed.
