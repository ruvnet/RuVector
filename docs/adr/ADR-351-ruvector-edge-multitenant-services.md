# ADR-351: ruvector edge — Multi-Tenant Rust/wasm Services on Cloudflare Workers + Durable Objects, Behind an Edge Authorization Server Federated to cognitum-one OAuth

## Status

**Proposed.** 2026-09-28; **reconciled 2026-09-29 with the edge-authorization-server decision**
(Review log). Nothing is deployed or changed; every resource creation and provider touch is **gated
on a separate, explicit approval** (§13).

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
| CPU per request | Free 10 ms; Paid 30 s default, `limits.cpu_ms` up to 300 s, **per script**. Whether DO request CPU follows it is unclear. | [V] / **[U]** (M1) |
| DO SQLite | 10 GB per object; 30-day PITR. **2 MB** max string/BLOB/row, **100 KB** max statement, **100** bound parameters, 100 columns. | [V] |
| DO SQLite atomicity | No `txn` object; atomicity comes from **write coalescing** (synchronous `sql.exec` calls with no intervening `await`). Non-storage I/O lets requests interleave. No transaction spans two DOs. | [V] |
| DO throughput / connections / alarms | ~500–1,000 simple req/s per object; 6 simultaneous outgoing connections per request; exactly **one** alarm per object. | [V] |
| workers-rs | `Storage::transaction` needs a `'static` closure (`durable.rs:635-639`); `SqlCursor::to_array` materialises every row (`sql.rs:330-340`), `cursor.raw()` streams (`sql.rs:388`). The edge workspace pins `worker =0.8.5`, `wasm-bindgen =0.2.125` (§8). | [V] workers-rs `b57ba6e`; [V] tree |
| workers-rs panic recovery | Any `WebAssembly.RuntimeError` (OOM, `panic = "abort"`) **reinitialises the whole wasm instance**, recreating every co-located DO. `--panic-unwind` needs nightly + `-Zbuild-std`. | [V] |
| Rate Limiting binding | GA since 2025-09-19; global vs per-location counting not confirmed. | [V] / **[U]** (M1) |
| Cache API | `cache.put` is guaranteed only on custom domains; on `workers.dev` it may no-op. | [V] |
| Service Bindings | Worker-to-Worker calls within one account; not an authentication mechanism. | [L] |
| Workers AI `@cf/baai/bge-small-en-v1.5` | 384-dim embeddings (same as all-MiniLM-L6-v2) | [V] |
| D1 per-database size cap | Commonly cited as 10 GB | **[U]** (M3) |
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
   JWKS URLs, upstream endpoints, public origin (§5.4, §5.6); today they are vars (§8 delta 6).
9. **Exactly one provider touch:** one exact upstream DCR redirect-allowlist entry for the edge
   callback (G1, [V] §1.1). Team tenancy still needs an upstream org selector or role claim (G5).
10. **Incremental delivery** (§15): M0 cores; **M0.5 both Workers live on workers.dev**; M1 flat
    scan + MCP; M2a int8; M2b HNSW (G4a); M3 durability; M4–M5 the wider family; M6 GA.

## 3. Services

| Service | Worker / routes | Backing crate(s) | Durable Object | Capability source | Milestone |
|---|---|---|---|---|---|
| **ruvector-edge-auth** (edge AS) | Worker `ruvector-edge-auth`: `/.well-known/oauth-authorization-server`, `/.well-known/jwks.json`, `/register`, `/authorize`, `/callback`, `/token`, `/revoke` | `ruvector-edge-authz` (core), `ruvector-edge-auth` (`ResourceUrl`, JWKS, upstream verify); glue `edge/auth-worker` | `AuthStore` (one global instance, §5.6) | — (it issues capabilities) | M0 core, M0.5 live |
| **rv-gateway** (public resource entrypoint) | Worker `ruvector-edge-gateway`: `/.well-known/oauth-protected-resource/v1[/mcp]`, `/v1/health`, `/v1/me`, `/v1/usage`, `/v1/tenant*`, dispatch | `ruvector-edge-auth`, `ruvector-edge-tenancy`; glue `edge/gateway` | — | none of its own | M0.5 (PRM, `/v1/me`, `/v1/mcp` 401 challenge), M1 |
| **rv-vector** | `/v1/collections*`, `index=flat\|q8\|hnsw` | M1 exact f32 flat scan; M2a u8 flat scan + f32 rerank; M2b HNSW via revised `rvf-index` (G4a) | `VectorShard` | scope ∩ role | M1/M2a/M2b |
| **TenantLedger** | internal control plane for one tenant | plain Rust + serde | `TenantLedger` (one per tenant) | — | M1 |
| **rv-snapshot** | `…/snapshots`, `:restore`, `:export` | `rvf-wire`, `rvf-types`, streaming segment writer | task on `VectorShard` | export → read; snapshot → write; restore → admin | M3 |
| **rv-embed** | `text` on upsert/query | Workers AI `bge-small-en-v1.5` (384-dim) | — | inherits route | M3 |
| **rv-ingest** | bulk upsert, `:import`, `/v1/jobs/{id}` | Queues + R2 staging (server-minted `upload_id`) | — | submit → write; status → read | M3 |
| **rv-quant** | `index=rabitq` | pure `ruvector-rabitq`, `rbpx0001` | `QuantShard` | as rv-vector | M4 (large shards need rabitq persist v2) |
| **rv-graph** | `/v1/graphs*`, `…/cypher` | `rvlite::cypher` after `rvlite-core` split (G4b) | `GraphStore` | MATCH → read; mutating → write | M4 |
| **rv-mincut** | `/v1/mincut`, `/v1/mincut/jobs*` | `ruvector-mincut` rebuilt from a persisted edge list | `AnalyticsJob` | compute → read; job → write | M4 |
| **rv-registry** | `/v1/rvf/{name}/{version}*` | fuzzed streaming `rvf-wire` validator + R2 | — (D1 index) | pull → read; push → write; public publish → `ruvector:publish` ∩ owner | M5 |
| **rv-mcp** | `POST /v1/mcp` (Streamable HTTP, JSON-RPC 2.0) | Rust MCP framing calling the REST handlers | — | scope ∩ role; mutating tools carry `destructiveHint` and support `dry_run` | M1 (vector tools), each later tool with its service (M4/M5) |
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
  Recreating a deleted name gets a new uid, so a new DO and new R2 keys.
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
| `scope` | string | space-delimited subset of the §5.3 vocabulary |
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

The edge AS mints **only** these scopes (`scopes_supported` in RFC 8414 and DCR). `openid`,
`profile` and `email` in a registration or authorization request are accepted and **dropped** (the
edge AS issues no ID token); any other unknown scope is `invalid_scope`.

| Scope | Grants (always ∩ role) | Role needed | Registrable by default (DCR without `scope`) | Granted by default (`/authorize` without `scope`) | PRM `scopes_supported` |
|---|---|---|---|---|---|
| `ruvector:read` | read: list/get collections, query, fetch, usage, MCP read tools | viewer | yes | yes | `/v1`, `/v1/mcp` |
| `ruvector:write` | write + create: upsert/delete vectors, create collections (drop: owner only), snapshots/imports, MCP mutating tools | editor (drop: owner) | yes | **no** — granted only when requested and shown on consent | `/v1`, `/v1/mcp` |
| `ruvector:admin` | admin: members, tenant deny entries, client blocks, restore, audit (and `tenant:claim`, like `ruvector:write`) | owner (claim: unclaimed tenant) | no (request at DCR) | no | `/v1` only |
| `ruvector:publish` | public registry publish | owner | no; not minted before M5 | no | `/v1` from M5 |
| `offline_access` | accepted and echoed; refresh tokens follow the client's registered `refresh_token` grant, not this scope | — | yes | yes | all |

- **Grant rule.** At `/authorize` the grant is requested ∩ client ceiling ∩ the resource's PRM
  `scopes_supported` (so `ruvector:admin` is never minted for `/v1/mcp`); vocabulary scopes outside
  it are **dropped**, not refused, and the token response `scope` reports what was granted (RFC
  6749 §3.3). Only a request left with no `ruvector:*` scope is `invalid_scope`. Consent plus role,
  not the DCR ceiling, is the control.
- **Step-up.** A mutating call without `ruvector:write` gets HTTP **403** `WWW-Authenticate: Bearer
  error="insufficient_scope", scope="ruvector:read ruvector:write offline_access",
  resource_metadata="…"` — also for `tools/call` on `/v1/mcp` (an HTTP status, not a JSON-RPC
  error), so MCP clients re-authorize with the wider scope.
- **Capability = route requirement ∩ scope ∩ role.** One versioned default-deny table in
  `ruvector-edge-auth` (`scopes.rs` `ROUTE_TABLE`) has rows `(method, pattern, capability,
  min_role ∈ {viewer, editor, owner, unclaimed})`, with per-op/per-tool rows for `/v1/ops` and
  `/v1/mcp`; a unit test asserts every route is covered and unknown routes are denied. A missing
  role → `403 role_required`/`not_claimed`. ([V] tree: the tables still say `mcp:read mcp:invoke`
  and lack roles; §8 delta 1.)
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
   workers.dev → workers.dev subrequests are refused (Cloudflare 1042) [L]. ([V] tree: all six are
   `[vars]` today; §8 delta 6.)
6. **JWKS.** `kty=EC`, `crv=P-256` only. In-isolate TTL map (10 min) is the primary cache (the Cache
   API may no-op on workers.dev); every successful fetch **replaces** the set (a removed `kid` stops
   verifying within ≤ 10 min); an unknown `kid` triggers at most one single-flight refetch per 30 s
   per isolate; on fetch failure the last good set serves up to 24 h; never fetched → `503`.
7. **Claims.** Every §5.2 claim present and typed; `iss == EDGE_ISSUER`; `aud` a string **byte-equal
   to this resource's canonical URL**, on every route including `/v1/ops`. Any other `aud` (the
   other gateway resource, an adapter resource, an array) → **401** `WWW-Authenticate: Bearer
   error="invalid_token", error_description="audience mismatch", resource_metadata="<this route's
   PRM>"` so MCP clients re-run discovery; `audience_not_allowed` is only the log/metric reason
   (today 403, §8 delta 10). `exp > now − 60 s`, `iat ≤ now + 60 s`, **`exp − iat ≤ 900`**; `nbf`
   honoured if present; `upstream_iss` equals the compiled upstream issuer; deny-list (§5.8) on
   `jti`, `family_id`, `sub`, `client_id`, `kid`.
8. Clock and network only through the `Clock` and `HttpFetch`/`KeySource` traits; the crate never
   calls `std::time`.

### 5.5 Optional upstream-first-party mode (flagged, off by default)

For a first-party CLI that already holds `auth.cognitum.one` tokens. Enabled only by the var
`UPSTREAM_FIRST_PARTY = "true"` (default `"false"`) [V] tree; `FIRST_PARTY_AUDS` is a compiled const
like the other trust roots (§5.4 item 5; a var today, §8 delta 6), so a var edit cannot widen it.
When on, and **only on `/v1` REST, never `/v1/mcp` or `/v1/ops`**:

- `iss == https://auth.cognitum.one`, `kid ∈` compiled `ACCEPTED_UPSTREAM_KIDS` (today exactly
  `_jQ62WD8…`; upstream rotation needs a redeploy; the const does not exist yet, so the mode stays
  off until M0 adds it, §8 delta 6), header `typ` absent or `JWT`, **claim** `typ == "access"`
  (rejects ID and `inference` tokens), `exp − iat ≤ 3600`, `setup`/`workload`/`exchanged` rejected;
- `aud ∈ FIRST_PARTY_AUDS`, exact ids only — never a `dcr-` prefix match;
- the verifier normalises `sub` with `edge_subject` and sets `upstream_iss = iss`, so tenant and
  actor match the edge path; capabilities are the fixed set `read write admin` ∩ role (upstream
  tokens carry no ruvector scopes).

Such a client is registered at the upstream DCR with a loopback redirect (allowed today [V]) and
**identity scopes only**. Nothing in M0.5–M5 depends on this mode.

### 5.6 The edge authorization server (`ruvector-edge-auth`)

**Configuration.** `ISSUER` and the upstream issuer, JWKS URL, authorize and token endpoints are
release consts, as in §5.4 item 5 (vars today, §8 delta 6); `UPSTREAM_CLIENT_ID`,
`RESOURCE_ALLOWLIST` and `MAX_CLIENTS` stay reviewed deploy config.

**Storage.** One global `AuthStore` DO (`idFromName("auth-store-v1")`): `meta`, `clients`, `codes`,
`flows`, `refresh_tokens`, `refresh_families` [V] tree, plus `subjects(sub, upstream_iss,
upstream_sub, first_seen, last_seen)`, `dcr_rate` and redeemed-code tombstones (§8 delta 3). Codes
and refresh tokens are stored hashed; one-time takes are one `DELETE … RETURNING`. Metadata and
JWKS need no DO hop; only login, token and DCR traffic reaches it (~500–1,000 req/s [V]).

**DCR (`POST /register`, RFC 7591)** [V] tree `client.rs` contract: public clients (`none`);
`response_types ["code"]`; `grant_types` ⊆ {`authorization_code`, `refresh_token`} incl. the first,
**omitted → both** (deliberately not the RFC 7591 default, so connectors get refresh tokens);
1–8 redirect URIs (≤ 512 bytes, no fragment/userinfo): `https` with a host, or `http` on
`127.0.0.1`, `[::1]` or **`localhost`**, port-agnostic with path equality (RFC 8252 §7.3/§8.3;
Claude Code and MCP Inspector use `http://localhost:<port>/…`); `scope` ⊆ vocabulary, **omitted →
`ruvector:read ruvector:write offline_access`** (admin/publish only if requested here);
`client_name` ≤ 128 chars; body ≤ 16 KiB; ids `edc-` + random; cap `MAX_CLIENTS` (10,000) [V] tree;
per-IP rate limit and 30-day idle expiry **[U]** (M0.5). Today: §8 delta 4. Connector-prefix
(`chatgpt.com/connector/oauth/`, `chatgpt.com/aip/`, `claude.ai/api/mcp/`, `claude.com/api/mcp/`)
and loopback redirects show as **verified** on consent, others **unverified** (§8 delta 7). AS
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
  host, resource, scopes), which sets `__Host-eaf-{state[0..16]}` = the browser secret (`Secure;
  HttpOnly; SameSite=Lax; Path=/; Max-Age=600`; one cookie per flow, so parallel logins do not
  collide). **Continue** links to the upstream authorize URL; Cancel returns `access_denied`. AS
  HTML responses send `X-Frame-Options: DENY`, CSP `default-src 'none'; frame-ancestors 'none';
  form-action 'none'; base-uri 'none'` and `Referrer-Policy: no-referrer` (clickjacking, RFC 9700
  §4.16; no Referer leak). Consent is never skipped.

**Upstream leg.** Client `UPSTREAM_CLIENT_ID` at `auth.cognitum.one` (G1), redirect `<ISSUER>/callback`,
scope **`openid profile email` only** — never `mcp:*` (§1.1; §8 delta 2).

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
   if the access token carries one [V] tree `federation.rs`, `upstream.rs`.
4. Revoke the upstream refresh token (best-effort) and **discard** every upstream token — never
   stored, logged or forwarded (§8 delta 7).
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
  §4.14.2; §8 delta 11). `REFRESH_TTL_SECS` 30 d per token, `FAMILY_MAX_LIFETIME_SECS` 90 d per
  family [V] tree. Refresh re-mints from the stored identity, so upstream membership changes and
  deprovisioning are seen only at the next login (§12).
- `urn:ietf:params:oauth:grant-type:token-exchange` (RFC 8693, M1, adapter clients only):
  `subject_token` = the user's edge token for that adapter's own resource, `resource = …/v1`. The
  result keeps `sub`, `upstream_iss`, `org_id`, `workspace_id`, `family_id`; `scope ⊆` and `exp ≤`
  the subject token's; `act = {sub: <adapter client_id>}`, audited; no refresh token.
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
to `invalid_target`. (§8 delta 5: routes.rs serves the `/v1` document there, and every document
uses one `DEFAULT_SCOPES`.)

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
  revocations.
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

**Write path** (no `transaction()`, no `blockConcurrencyWhile`): (1) hold an unexpired **batch quota
lease** from `TenantLedger` (e.g. 10k rows / 4M floats / 8 MiB, 10 min), awaiting a new one only
when exhausted — no per-write DO hop; (2) **validate** in memory (dimension, finite values, norm > 0
for cosine, limits, idempotency `(sub, key)` with `422 idempotency_mismatch`) — `dry_run` stops here
and returns the would-be effect (§17); (3) issue every `sql.exec` for row, `ops`, `idempotency` and
`audit` **back to back with no `await`** so they coalesce into one commit (one statement per row, ≤
100 bound parameters; deletes in batches of ≤ 100); (4) only then mutate the in-memory slab or
index; (5) the alarm reconciles lease usage (crash-tested, §9).

**Isolate-wide memory budget.** A process-wide resident-set registry caps resident vector data at
**56 MB per isolate [U]**, evicting cold shards first. **Per-shard cap, M1: 3,000,000 floats (12 MB
f32, ≈ 7.8k × 384)**, so 3–4 fit; slabs are pre-sized from `meta`, M2a codes use a fixed reusable
arena. A panic or OOM reinitialises the **whole instance**: no-unwrap lint on the request path,
fuzzing, and the Q12 decision.

**Index kinds.** **M1 `flat`**: f32 slab, exact scan, no `q8`. **M2a `q8`**: u8 flat scan with
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
`usage_daily` every 5 minutes.

### 6.3 R2 layout (M3, gated: bucket `ruvector-edge-data`)

Every key component is server-derived or a validated slug.

| Prefix | Use |
|---|---|
| `snapshots/{tenant_key}/{service}/{collection_uid}/{shard}/{epoch:020}-{sha256_12}/seg-{n}.rvf` | snapshots |
| `staging/{tenant_key}/{upload_id}` | uploads (`upload_id` server-minted by `POST /v1/uploads`) |
| `rvf/{tenant_key}/{name}/{version}` | private registry packages |
| `public/{publisher_tenant_key}/{name}/{version}` | public packages (after M5) |
| `results/{tenant_key}/{result_id}` | signed result pages (§17) |
| `audit/{tenant_key}/{yyyy}/{mm}/{dd}/{seq}.ndjson` | shipped audit |

- RVF segments via `rvf-wire`: ≤ 4 MB `VEC_SEG`s from a SQLite cursor, index chunks (M2b), metadata,
  and a manifest with per-segment sha256, the `tenant_key`, `collection_uid` and audit-chain head.
  Streamed (multipart / `ReadableStream`), never assembled whole. Hourly when dirty; 24 hourly + 7
  daily retained.
- **Restore** checks in order: size vs quota before allocation, manifest `tenant_key`, per-segment
  sha256, dimension and quota; segments are read one at a time by range.
- **Import** takes only an `upload_id` resolving inside the caller's tenant.
- DO point-in-time recovery (30 d) is an operator-only second layer.

### 6.4 Control plane (M3, gated: D1 `ruvector-edge-control`)

D1 holds only global state: `tenants(tenant_key, plan, status)`, `plans`, `deny_global`,
`usage_daily`, `packages`, `audit_index`. Every access goes through a typed repository taking
`&TenantContext` (or an explicit `OperatorContext`) that adds `tenant_key = ?` itself; raw SQL in
handlers is banned by a CI grep and a clippy `disallowed_methods` entry. Cache keys include
`tenant_key` under `https://cache.ruvector-edge.invalid/…`. The adapter's `ruvector-chatgpt` D1 is
**not** part of the backend (§16.4).

### 6.5 Audit

Each DO keeps a hash-chained append-only `audit` tail; from M3 a Queue (`ruvector-edge-audit`) ships
batches to R2 NDJSON. Logged: `tenant_key`, `sub`, `client_id`, `act.sub` (exchanged tokens), `jti`,
`family_id`, route/op, scope and role used, outcome, byte/row counts, `approval_ref`, ray id. **Never logged:** tokens, upstream
identifiers, vectors, metadata payloads, IPs in the clear (IPs are hashed with the Worker secret
`IP_HASH_SALT`, rotated monthly).

## 7. API specification

**Conventions (gateway `/v1`).** JSON (UTF-8) with enforced `Content-Type`; `Content-Length` checked
before parsing; strict schemas (unknown fields rejected); RFC 9457 `application/problem+json` errors
with a stable `code`; `x-request-id` on every response; `Cache-Control: no-store` on credentialed
routes; `Idempotency-Key` (≤ 255 bytes, 24 h, scoped `(tenant, sub, key)`, body-hash bound) on
mutating routes; cursor pagination `limit ≤ 100`; CORS default-deny with an explicit origin
allowlist and no credentials mode, except the public PRM documents (`*`) and
`Access-Control-Expose-Headers: WWW-Authenticate` on 401/403.

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

| Method & path | Body / params | Scope + role | Milestone |
|---|---|---|---|
| `GET /.well-known/oauth-protected-resource/v1[/mcp]` | — (bare origin, `/.well-known/oauth-authorization-server`, `/.well-known/openid-configuration` → 404) | anonymous | M0.5 |
| `GET /v1/health` | → `{ok:true}` | anonymous | M0.5 |
| `GET /v1/me` | → `{tenant_key, org_id, workspace_id, sub, client_id, scopes, role \| null, claimed}` | any valid token | M0.5 (role/claimed from M1) |
| `POST /v1/tenant:claim` (MCP `tenant_claim`) | → 201 `{role:"owner"}` or 409 | `ruvector:write` or `ruvector:admin`; tenant unclaimed | M1 |
| `GET/POST/DELETE /v1/tenant/members[/{sub}]` | `{sub, role: viewer\|editor}` | `ruvector:admin` + owner | M1 |
| `POST /v1/tenant/deny` | `{kind: jti\|family_id\|sub\|client_id, value, ttl_s}` (caps §5.8) | `ruvector:admin` + owner | M1 |
| `GET /v1/usage` | → quotas, usage, work units, rate headroom | `ruvector:read` + viewer | M1 |
| `POST /v1/collections` | `{name, dim: 1..1536, metric: cosine\|l2\|dot, index?: flat\|q8\|hnsw\|rabitq, embedder?, filterable_keys?: [≤8], shards?: 1..6, hnsw?: {m: 8..48, ef_construction: 32..200}}` → 201 | `ruvector:write` + editor | M1 (`flat`), M2a, M2b, M4 |
| `GET /v1/collections`, `GET /v1/collections/{c}` | cursor / → config + `{count, resident_bytes, shards, snapshot_seq}` | `ruvector:read` + viewer | M1 |
| `DELETE /v1/collections/{c}` | soft delete, 7-day tombstone, then purge; name reusable with a new uid | `ruvector:write` + **owner** (§5.3) | M1 |
| `POST /v1/collections/{c}/vectors:upsert` | `{vectors: [{id ≤256B, values?, text?, metadata? ≤4KiB}] (1..500), dry_run?}` → `{upserted, write_seq}`; > 500 (> 64 for `hnsw`) → 202 job | `ruvector:write` + editor | M1 (`text` M3) |
| `POST /v1/collections/{c}/query` | `{vector?\|text?, top_k: 1..100, ef?, filter?: ≤8 clauses, include?}` → `{matches, shards_queried, took_ms}` | `ruvector:read` + viewer | M1 |
| `GET …/vectors/{id}`, `POST …/vectors:fetch` `{ids ≤100}` | — | `ruvector:read` + viewer | M1 |
| `POST …/vectors:delete` | `{ids ≤1000, dry_run?}` (batches of ≤ 100) | `ruvector:write` + editor | M1 |
| `POST /v1/ops` | §16.3 envelope (Service-Binding callers, `aud = …/v1` only) | per op | M1 |
| `POST …/snapshots` / `GET …/snapshots` | → 202 `{snapshot_id}` / list | write+editor / read+viewer | M3 |
| `POST …/snapshots/{id}:restore` | — | `ruvector:admin` + owner | M3 |
| `GET /v1/collections/{c}:export` | `?redact=<keys>` → streamed `.rvf` with witness manifest | `ruvector:read` + viewer | M3 |
| `POST /v1/uploads`, `POST …:import`, `GET /v1/jobs/{id}` | `{size, sha256}` / `{upload_id}` or ≤ 8 MiB inline / — | write+editor / write+editor / read+viewer | M3 |
| `GET /v1/audit?cursor=&limit≤500` | own tenant only | `ruvector:admin` + owner | M3 |
| `POST/GET /v1/results[/{id}]` | publish a signed, redacted result page (§17) / read it | write+editor / read, or public if opted in | M3/M5 |
| `/v1/graphs*`, `…/cypher`, `/v1/mincut*` | as in §3; Cypher ≤ 10k steps, ≤ 1k rows | read / write | M4 |
| `PUT/GET /v1/rvf/{name}/{version}*`, `GET /v1/rvf?prefix=` | immutable semver; push by `upload_id` | read / write | M5 |
| `POST /v1/rvf/{name}/{version}:publish` | public publish | `ruvector:publish` + owner | M5 |
| `POST /v1/mcp` | JSON-RPC 2.0 `initialize`, `tools/list`, `tools/call`. Bootstrap: `tenant_claim`. Read tools: `collection_list`, `vector_query`, `vector_fetch`, `graph_cypher` (MATCH), `rvf_list`. Mutating (`destructiveHint`, `dry_run` arg): `vector_upsert`, `vector_delete`, `collection_create`, `mincut_compute`, `rvf_import`. Missing write → HTTP 403 step-up challenge (§5.3) | scope ∩ role | **M0.5: 401 challenge with the `/v1/mcp` `resource_metadata`**; M1 `initialize`, `tools/list`, `tools/call` (`collection_*`, `vector_*`); M4 graph/mincut; M5 rvf |

**Status codes:** `400` `invalid_request`, `dimension_mismatch`, `non_finite_value`; `401`
`invalid_token` (route-specific `resource_metadata`; also any `aud` mismatch, logged as
`audience_not_allowed`);
`403` `insufficient_scope` (step-up challenge), `role_required`, `not_claimed`, `tenant_suspended`;
`404` `not_found` (also foreign tenants); `409` `conflict`; `413` `payload_too_large`,
`quota_exceeded`, `budget_exceeded`; `422` `idempotency_mismatch`; `429` `rate_limited` +
`Retry-After`; `503` `jwks_unavailable`, `shard_unavailable`, `trust_root_mismatch`.

## 8. Crate and repository layout (as built, working tree 2026-09-28 23:25)

```
crates/ruvector-edge-auth/    [V] root member. RS verification: jws, jwks, claims (TokenKind EdgeIssued |
                              UpstreamFirstParty), audience (exact aud), resource (ResourceUrl, shared with
                              the AS), subject (edge_subject), scopes (Capability, ROUTE_TABLE), prm,
                              verifier, clock, error. M0 adds trust_root.rs.
crates/ruvector-edge-authz/   [V] root member. AS core over sync ports (Clock, Rng, Signer, Client/Code/
                              Refresh/FederationStore): client (DCR), authorize, pkce, code, federation,
                              token (RFC 9068 mint), grant (TokenEndpoint), params, refresh, revoke,
                              metadata (RFC 8414, JWKS, paths), resource (ResourceAllowlist), error.
crates/ruvector-edge-tenancy/ [V] root member. context, names, uid, shard, meta (DoMeta/LedgerMeta identity
                              check), membership, lease, validate, quota, problem, error.
crates/ruvector-edge-store/   planned M1: SqlStore (paged raw cursor), codecs, ops log, resident-set registry.
crates/ruvector-edge-cli/     planned M1: edge-AS loopback login, logout, claim, commands.
edge/                         [V] standalone workspace (root `exclude`, own Cargo.lock): worker =0.8.5,
                              wasm-bindgen =0.2.125; release opt-level z, lto, 1 CGU, panic = abort.
  auth-worker/                → Worker `ruvector-edge-auth`: lib.rs, config.rs, platform.rs (clock, RNG,
                              fetch), signer.rs (EnvSigner), store.rs (#[durable_object] AuthStore), sql.rs
                              (SqlExec port) + sql_ports.rs (schema v1), upstream.rs (exchange + upstream
                              verify), http.rs, endpoints/{authorize,consent,callback,register,token}.rs.
                              wrangler: DO AUTH_STORE (sqlite migration v1), vars ISSUER, RESOURCE_ALLOWLIST,
                              SCOPES_SUPPORTED, UPSTREAM_*, MAX_CLIENTS (optional, default 10,000); secret
                              EDGE_AUTH_SIGNING_JWK.
  gateway/                    → Worker `ruvector-edge-gateway`: lib.rs, config.rs, routes.rs, auth.rs,
                              keys.rs (isolate JWKS caches), platform.rs, respond.rs; Service Binding
                              EDGE_AUTH (JWKS). M1 adds the VectorShard and TenantLedger DO classes (Q8).
```

**Known deltas between the working tree and this ADR** ([V] tree; M0 closes each unless marked M1,
or amends this section with a reason):

1. **Scopes and capabilities.** `SCOPE_TABLE`, `DEFAULT_SCOPES`, `SCOPES_SUPPORTED` use
   `mcp:read`/`mcp:invoke` → §5.3. `Capability` `{Read, Write, CreateCollection, PublishPublic}`
   → add `Admin`, rename to `Publish`. `ROUTE_TABLE` `(Method, pattern, RouteRequirement)` →
   `(method, pattern, capability, min_role)` with rows for `tenant:claim` (write|admin, unclaimed),
   `tenant/members`, `tenant/deny` (admin, owner), `DELETE /v1/collections/{c}` (write, owner;
   today `CreateCollection`), and per-op/tool rows for `/v1/ops`, `/v1/mcp`. `RouteSurface` gains
   `Ops`; `capabilities_for` ignores `kind` → §5.5 fixed set, `UpstreamFirstParty` refused off REST.
2. `UPSTREAM_SCOPES` includes `mcp:read` → `openid profile email`.
3. `AuthStore` schema v1 lacks `subjects`, `dcr_rate` and the redeemed-code store `code.rs` expects.
4. **Scope defaults.** `AuthConfig::dcr_policy` sets `default_scope = scopes_supported` (a test
   asserts it) → const `ruvector:read ruvector:write offline_access`; `/authorize` without `scope`
   grants the ceiling → `ruvector:read`; `ensure_subset` refuses → intersect and report; omitted
   `grant_types` → `[authorization_code]` → both; `localhost` refused → accepted.
5. **PRM.** `routes.rs` serves the `/v1` document at the bare path (RFC 9728 §3.3) → 404; one
   `DEFAULT_SCOPES` → per resource; no `/v1/mcp` route (falls to `OtherV1`, `/v1` challenge) → M0.5
   route with its own challenge; no `Access-Control-Expose-Headers: WWW-Authenticate`.
6. **Trust roots `[vars]` → release consts** (vars only under `dev-issuer`, 503 on contradiction):
   gateway `PUBLIC_ORIGIN`, `EDGE_ISSUER`, `EDGE_JWKS_URL`, `UPSTREAM_ISSUER`, `UPSTREAM_JWKS_URL`,
   `FIRST_PARTY_AUDS` (only the `UPSTREAM_FIRST_PARTY` flag stays a var); auth-worker `ISSUER`,
   `UPSTREAM_ISSUER`, `UPSTREAM_JWKS_URL`, `UPSTREAM_AUTHORIZATION_ENDPOINT`,
   `UPSTREAM_TOKEN_ENDPOINT`. `ACCEPTED_UPSTREAM_KIDS` does not exist → add it on the §5.5 path
   only. `ClaimsPolicy` default `max_lifetime_secs = 3600` → 900 for edge tokens.
7. Consent page lacks the verified/unverified marker; `/callback` does not revoke the upstream
   refresh token (`UpstreamTokens` drops it).
8. Pins 0.8.5 / 0.2.125 (shared root wasm-bindgen) instead of the planned 0.8.7 / 0.2.129; M0
   records the choice after a wasm32 build of both Workers.
9. `wrangler.toml` build commands → a wrapper running `env -u RUSTFLAGS` with
   `CARGO_TARGET_DIR=/data/scratch/ruvector-edge-target/edge`.
10. `AuthError::AudienceNotAllowed` → 403 → 401 `invalid_token` + `resource_metadata` (§5.4).
11. `refresh.rs` revokes on any re-presentation → `REFRESH_REUSE_GRACE_SECS = 30`. M1: exchange
    grant, `private_key_jwt` adapter clients, `act`.

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
     silent 200; CORS scope; exchange (M1: adapter-only, `…/v1` only, narrowed, `act`); minted-token
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
   (M3), `rbpx0001` (M4). **Live:** M0.5 acceptance.

## 10. Quotas and abuse controls

Only layer 4 is a hard, consistent limit; the others are shields.

1. **Pre-auth:** `Content-Length` caps (1 MiB; 8 MiB inline import); `ip:{hash}` limit before
   signature work; failed-token cache; AS per-IP limits on `/register`, `/authorize`, `/token`. WAF
   only with a custom domain (G6).
2. **Rate:** `org:{tenant_key}` (100 / 10 s reads, lower for writes) and `user:{tenant_key}:{sub}`
   (20 / 10 s); a query costs `shards_queried` tokens. Locality **[U]** (M1).
3. **Per-operation budgets:** step/row counters return `413 budget_exceeded` before `cpu_ms`.
4. **Admission:** `TenantLedger` leases (§6.1).

| Limit | M1 default (per plan) |
|---|---|
| Collections per tenant | 20 |
| Shards per collection | ≤ 6 through M3 |
| Dimension | 1..=1536, fixed per collection |
| Resident vector data per isolate | 56 MB, LRU eviction **[U]** |
| Per-shard cap | M1 3,000,000 floats (12 MB); M2a/M2b set from measurement (≥ 3 / ≥ 2 shards per isolate) **[U]** |
| Vectors per tenant | 250k free / 2M paid **[U]**, applied only once the milestone ceiling reaches them |
| Upsert batch | ≤ 500 vectors, ≤ 1 MiB (≤ 64 sync for `hnsw`) |
| `top_k` / metadata | ≤ 100 / ≤ 4 KiB per vector |
| Cypher / mincut inline (M4) | ≤ 10k steps, ≤ 1k rows, ≤ 6 hops / ≤ 200k edges, ≤ 50k nodes |
| `limits.cpu_ms` | one script-level value (M1: 30 s) **[U]** DO CPU source |
| Edge DCR clients | cap `MAX_CLIENTS` (default 10,000) [V] tree; per-IP rate limit and 30-day idle expiry **[U]** (M0.5) |

**Derived per-collection ceiling at 384 dims** (6 shards × per-shard cap): M1 ≈ 47k vectors, M2a ≈
190k if the measured cap lands near 12 MB of codes; per tenant at 20 collections, M1 ≈ 940k.

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
| Spoofing: upstream token over-privilege | Edge's upstream client holding `mcp:*` would hold a live api.cognitum.one credential | Upstream client registered with identity scopes only; upstream tokens verified, refresh revoked, all discarded, never stored or forwarded | §8 delta 2 until M0 |
| Spoofing: trust-root substitution | Var edit or dev deploy pointing an issuer, JWKS URL, upstream endpoint, `PUBLIC_ORIGIN` or `FIRST_PARTY_AUDS` elsewhere, on either Worker | Compile-time consts on both Workers; vars only under `dev-issuer`; 503 on mismatch; CI refuses dev deploys | All are vars until M0 (§8 delta 6) |
| Spoofing: algorithm confusion / key injection | `alg=none`, HS256, `jku`/`jwk`, DER | Exact ES256, header allowlist, compiled JWKS URL, thumbprint = `kid`, fixed r‖s | — |
| **AS: open redirect** | Crafted `redirect_uri` or error path used to bounce users | Errors before client + redirect validation rendered, never redirected; exact (loopback: port-only) redirect match; `/callback` redirects only to the stored flow's validated URI | — |
| **AS: code injection / login CSRF** | Attacker injects their upstream code or edge code into a victim's browser; stolen edge code replayed | Per-flow `__Host-eaf-{state[0..16]}` cookie whose hash is stored with the flow [V] tree; one-time `state` and codes; 60 s code TTL; code bound to client, redirect, resource and PKCE S256 (both legs); code consumed on any failure; a replayed code revokes its family | — |
| **AS: mix-up with upstream** | Response from another AS accepted as the upstream's, or a client confusing our AS with another | Single upstream with compiled issuer and JWKS URL (keys accepted by thumbprint = `kid`, new `kid` alerted; `ACCEPTED_UPSTREAM_KIDS` pinning only in the §5.5 mode); upstream token `iss` and `aud == UPSTREAM_CLIENT_ID` checked; upstream `iss` parameter checked when sent; our responses carry RFC 9207 `iss`; distinct redirect per AS | Upstream `iss` parameter support **[U]**; upstream trust root in vars until M0 (§8 delta 6) |
| **AS: consent phishing via open DCR** | Anyone registers an https client and lures a user | Consent always shown; unverified-host marker (§8 delta 7); write/admin never granted without an explicit scope request shown on the consent page, and still ∩ role; clickjacking blocked (`X-Frame-Options: DENY`, `frame-ancestors 'none'`) [V] tree; DCR rate limit, cap, idle expiry; owner `client_id` block (§5.8); CIMD-verified client ids later (Q15) | A user can still approve a malicious app for their own data (as with any OAuth AS) |
| **AS: refresh-token theft** | Stolen refresh token | Rotation with reuse detection revoking the family after a 30 s grace; bound to `client_id` and `resource`; 30 d per token, 90 d family cap; `/revoke`; RS deny by `family_id` | A thief who refreshes first keeps access until the victim's next refresh (after the grace window) triggers reuse detection, or ≤ 90 d; a replay inside the 30 s window gets `invalid_grant` without revocation |
| **AS: signing-key compromise / rotation** | Leaked `EDGE_AUTH_SIGNING_JWK` mints arbitrary tokens | Worker secret only, generated offline; staged rotation by key order (§5.6); emergency rotation removes the `kid` from JWKS and deny-lists it (≤ 60 s at the gateway) | Tokens minted before detection work until the deny entry lands; adapter resource servers until their next JWKS fetch (≤ 10 min) |
| **AS: availability** | Flooding `/register`, `/authorize`, `/token` at one `AuthStore` DO | Per-IP rate limits; stateless metadata/JWKS; login-rate traffic only | Single DO throughput ceiling; shard at M6 if measured |
| **AS: upstream deprovisioning** | A user disabled or removed upstream keeps refreshing at the edge | Families capped at 90 d; operator action revokes every family of an edge `sub` plus a global `sub` deny (M3); M6 pentest item | Not propagated automatically: up to 90 d of edge access without an operator action |
| **Adapter resource servers** | Adapter-hosted MCP endpoints (§16.1) cannot see the ruvector deny-list | JWKS replaced on every fetch (removed `kid` dead ≤ 10 min), last-good fallback ≤ 1 h there; everything they forward to `/v1/ops` is re-checked against the deny-list | Tenant/family/`sub` denies reach them only via ≤ 15 min token expiry |
| **T**ampering: cross-tenant write | Crafted names/ids; routing bug; D1 query missing a filter | No tenant parameter; hashed DO names with `collection_uid`; in-DO assertion; per-tenant state in `TenantLedger`; typed D1 repository | — |
| Tampering: import / restore injection | Another tenant's R2 object; malformed RVF | Server-minted `upload_id`; manifest `tenant_key`; per-segment sha256; fuzzed decoders; size checks first | — |
| Tampering: `/v1/ops` misuse / token passthrough | An adapter forwards the user's adapter-audience token, or a captured op is replayed or retargeted | `/v1/ops` accepts only `aud = …/v1`; adapters get it by RFC 8693 exchange as confidential `private_key_jwt` clients, scope and `exp` narrowed, `act` audited; `target` and `tenant_key` echo checked; `op_id` idempotency; scope ∩ role (§16.3) | A compromised adapter holding its private key can exchange tokens it receives, within their scope and lifetime |
| **R**epudiation | "I didn't delete that" | `ops` + hash-chained audit with `sub`, `jti`, `family_id`, `approval_ref`; claims, invites, deny writes audited | Audit is per DO |
| **I**nformation disclosure | Existence probing; log leakage; cache cross-hits | 404 for foreign resources; no tokens, upstream ids, vectors or clear IPs in logs; cache keys include `tenant_key`; redacted exports | — |
| **D**oS: cross-tenant memory | One tenant's OOM/panic reinitialises the shared instance | Resident-set registry and caps; pre-sized slabs; no-unwrap lint; fuzzing; panic-unwind evaluation | A panic can cold-restart co-located shards until Q12 is resolved |
| DoS: floods / fan-out | Unauthenticated floods, kid-spraying, fan-out amplification | Pre-auth IP limit; failed-token cache; single-flight JWKS refetch; `shards_queried` charging; ≤ 6 shards; step budgets | RL may be per location; no WAF on workers.dev |
| **E**levation of privilege | Viewer mutating; first claimant in a shared org; scope inflation | Scope ∩ role ∩ route; default-deny route table; explicit audited claim granting nothing pre-existing | Team tenancy needs G5 |
| Supply chain and deploy | Account-wide token; mold `RUSTFLAGS`; pin drift | Scoped deploy credentials; CI binding checks; `env -u RUSTFLAGS`; pinned lockfile | Account shared with the consultant-email stack until G6 |

Run `npx @claude-flow/cli@latest security scan` and an external review before exposure beyond
staging; a pentest gates GA (M6).

## 13. Gates: out of scope until separately approved

| Gate | What it needs | Earliest milestone |
|---|---|---|
| G1 | **Register the edge AS as an upstream client** at `auth.cognitum.one` via the public DCR endpoint (use authorised by rUv, 2026-09-29): public client, PKCE, `scope = "openid profile email"` (explicit — omitting it grants `mcp:*`), `redirect_uris = ["https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/callback"]`. **Precondition [V]:** at console `ce9ddca` that redirect is not on `ALLOWED_REDIRECT_PREFIXES`/`ALLOWED_EXACT_REDIRECTS`, so DCR returns `invalid_redirect_uri`. G1 therefore includes **one exact-match line** in `ALLOWED_EXACT_REDIRECTS` (console repo review; ADR-124 does not pre-approve it) — or an equivalent hand-seeded client row. No scope, claim or protocol change. | M0.5 |
| G2 | Cloudflare resources in account `501c77f5…`: Workers `ruvector-edge-auth` and `ruvector-edge-gateway` on workers.dev, DO namespaces `AuthStore` (M0.5), `VectorShard`, `TenantLedger` (M1), Rate Limiting bindings, secrets `EDGE_AUTH_SIGNING_JWK`, `IP_HASH_SALT`, `DENY_GLOBAL`; the gateway → auth `EDGE_AUTH` Service Binding. Workers Paid plan. Operator deploys via wrangler OAuth; CI uses a dedicated token scoped to Workers Scripts, DO, D1, R2 and Queues, plus the binding check. | M0.5 (auth + gateway), M1 (DOs) |
| G3 | R2 `ruvector-edge-data`, D1 `ruvector-edge-control`, Queues `ruvector-edge-jobs` (+ DLQ) and `ruvector-edge-audit`, Workers AI binding | M3 |
| G4 | Upstream crate PRs: (a) `rvf-index` DistanceOracle traversal, CSR `u32` adjacency, dense ids, header fields, streaming encoder, simd128; (b) `rvlite-core` split / `browser` gate; (c) `ruvector-rabitq` persist v2 | M2b, M4 |
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

### 16.1 Identity contract

| Item | Value |
|---|---|
| Issuer (`iss`) | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev` |
| AS metadata | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/.well-known/oauth-authorization-server` |
| JWKS URL | `https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/.well-known/jwks.json` (ES256, P-256, `kid` = RFC 7638 thumbprint). A Worker in the same account fetches it through a **Service Binding** to `ruvector-edge-auth` (workers.dev → workers.dev subrequests are refused, Cloudflare 1042 [L]; the gateway does this [V] tree). Cache ≤ 10 min; **replace** the set on every successful fetch; refetch on unknown `kid` at most once per 30 s; last-good fallback ≤ 1 h |
| Token format | JWS, header `alg=ES256`, `typ=at+jwt`; claims exactly §5.2 (`act` only on exchanged tokens) |
| Stable subject | `sub` = `"es1_" + base32lower(sha256("ruvector-edge/sub/v1\|" + upstream_iss + "\|" + upstream_sub))[0..26]` (`subject::edge_subject`). Key memberships on **(`iss`, `sub`)**; never on `client_id`, `jti` or email |
| Tenant claims | `upstream_iss`, `org_id`, `workspace_id` → `tenant_key` (§4.1, formula §16.2); also returned by `GET /v1/me` |
| Scope vocabulary | `ruvector:read`, `ruvector:write`, `ruvector:admin` (`/v1` only), `ruvector:publish` (M5), `offline_access`; default grant `ruvector:read`, write only when requested on consent; step-up via 403 `insufficient_scope` (§5.3) |
| Canonical audiences | MCP `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp`; REST and `/v1/ops` `…/v1`. An adapter that serves its **own** MCP endpoint gets its canonical URL added to `RESOURCE_ALLOWLIST` (reviewed AS config) and verifies inbound tokens exactly as §5.4 with that URL as `aud`; the gateway never accepts that audience |
| Calling ruvector | Exchange (§5.6, M1): the adapter is an operator-registered **confidential** client (`private_key_jwt`, per-adapter public JWK, exchange grant only). `POST /token` `grant_type=urn:ietf:params:oauth:grant-type:token-exchange`, `subject_token=<user's token, aud = adapter resource>`, `subject_token_type=urn:ietf:params:oauth:token-type:access_token`, `resource=…/v1`, optional narrower `scope` → a `…/v1` token with the same `sub`/tenant claims, `scope ⊆`, `exp ≤` the subject token's, `act.sub` = adapter `client_id`. An adapter that is only an OAuth client (no own resource) requests `resource=…/v1` in its own flow instead |
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
  (`quant`, `graph`, …) in the same position.

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
| `op_replayed` | 409 | `op_id` seen with a different body; same body returns the stored response |
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
  rolling back a Worker version leaves data readable.

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
