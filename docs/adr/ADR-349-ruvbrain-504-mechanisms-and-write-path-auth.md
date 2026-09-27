# ADR-349: pi.ruv.io — Three Independent 504 Mechanisms, a UTF-8 Crash, and the Unauthenticated Write Path

## Status

Accepted (code landed; item 6 fixed by P5 in a follow-up). Three items are deliberately **deferred with reasons**
rather than guessed at — see "Deliberately not decided here".

> **Numbering note.** Drafted as ADR-348 on 2026-09-19; renumbered to 349 at
> merge because 347 (PR #989) and 348 (PR #946, TwinKV) were claimed first.

## Context

`pi.ruv.io` is Cloud Run service `ruvbrain` in project `ruv-dev`, region
`us-central1`, built from `crates/mcp-brain-server`.

On 2026-09-19 production logged, within minutes of each other:

- `/v1/status` — 504 at exactly 300.00s, twice
- `/v1/memories/list` — 504
- `/v1/memories/search?q=*` — 504
- `/v1/pipeline/inject` — 504, five times in four minutes
- `/v1/pipeline/optimize` — 504
- a panic in `routes.rs`, followed by graceful shutdown and restart

The 300.00s is not a coincidence with the 5-minute cognitive loop. It is
`timeoutSeconds: 300` on the Cloud Run service — the request deadline. Every
504 is "the handler never returned in time", not "the loop took 5 minutes".
The cognitive loop was cleared: it is `spawn_blocking`-wrapped and takes
`graph.read()`, never `.write()`.

### Deployment reality (verified read-only against `ruv-dev`)

| Fact | Value |
|---|---|
| Serving revision | `ruvbrain-00205-5fv`, image built **2026-06-17** |
| Gap to `main` | ~3 months |
| Resources | `cpu=2`, `memory=4Gi` |
| `containerConcurrency` | 80 |
| `timeoutSeconds` | 300 |
| `minScale` | unset (scale-to-zero; every cold request pays startup) |
| `maxScale` | 15 |
| IAM | `roles/run.invoker` granted to **`allUsers`** |
| `READ_ONLY` | **not set** |
| `BRAIN_SYSTEM_KEY` | set (Secret Manager) |

Two consequences do most of the work in this ADR:

1. **`#[tokio::main]` on a 2-CPU machine gives 2 worker threads serving up to
   80 concurrent requests.** Anything that blocks a worker blocks 40 requests'
   worth of capacity. Anything that holds a lock blocks all of them.
2. **`allUsers` can invoke.** Cloud Run performs no authentication for this
   service. Every in-handler auth check is the *only* auth check.

### The three 504 mechanisms are independent

They are not one bug seen three ways. Each produces 504s on its own, and
fixing any one leaves the others.

**M1 — search takes a write lock on the whole graph.**
`ranked_search` required `&mut self` for exactly one reason: `ensure_csr()`.
`pagerank_scores` was already `&self`; everything else it touches is
read-only. So the search route took `state.graph.write()`, under a comment
claiming the lock was held "briefly". Every inject sets `csr_dirty = true`, so
the next search rebuilt the CSR — ~1.2M edges in production — while excluding
every concurrent reader. Injects and searches therefore amplify each other:
traffic that mixes the two serialises completely.

**M2 — the cold-start sparsifier holds a write lock for its entire build.**
At t+60s, if `100_000 < edges <= 5_000_000`, `main.rs` rebuilt the spectral
sparsifier. The live graph is **1,198,117 edges**, squarely inside the band,
so this ran on every cold start — and with `minScale` unset, cold starts are
routine.

**M3 — `list_memories` did work proportional to the corpus.**
It cloned every matching `BrainMemory` (each carrying a 384-dim embedding plus
several `String`s), sorted all of them, then returned `limit` of them.
`limit=5` bounded the response and nothing else.

### Why `spawn_blocking` on M2 is actively misleading

The code already wrapped the sparsifier build in `spawn_blocking`, with a
comment directly above it *predicting this exact 504*. A reader skims that and
concludes the problem is handled. It is not.

`spawn_blocking` moves CPU work off the tokio worker threads. That is a real
fix for **runtime starvation** — with only 2 workers, a long synchronous
compute would otherwise stall the reactor. It does nothing about **lock
contention**, because the write guard was acquired *inside* the closure and
held for the whole build. The blocking pool thread holds the lock; every
request thread still queues behind it.

Two distinct failure modes, one of which had been fixed. This ADR fixes the
other. Anyone auditing this file later should treat "it's in `spawn_blocking`"
as evidence about the runtime, never about locks.

Related: `Cargo.toml` sets `clippy::await_holding_lock = "allow"`. That lint
exists precisely to catch this shape. It is not re-enabled here — doing so on
a 25k-line crate is its own change — but it is worth knowing it is switched
off.

### The write path was unauthenticated

`pipeline_inject` and `pipeline_inject_batch` took only `State` and `Json`. No
`AuthenticatedContributor`, while sibling routes have used it all along. The
only gate was `check_read_only`, and `READ_ONLY` is unset in production. With
`allUsers` holding `run.invoker`, **anyone on the internet could write up to
100 memories per request.**

Worse, injected rows were **undeletable through the API by anyone**.
`process_inject` stores `contributor_id = format!("pipeline:{}", req.source)`
— `source` comes from the caller's own body — and `delete_memory` only removes
rows where `contributor_id == caller_pseudonym`. Pseudonyms are 32 hex
characters, so no caller can ever present one equal to `pipeline:anything`.
There was no system or admin override.

The same missing-extractor shape appeared on `/v1/pipeline/pubsub`,
`/internal/queue/push`, `/internal/queue/drain`, `/internal/session/create`,
`/internal/session/:id`, `/v1/email/inbound` and `/v1/chat/google`.
`internal_session_create` additionally inserted into two `DashMap`s and
created an mpsc channel per call with no cap; the cleanup task only runs once
the sender drops, and the sender lives in `state.sessions`, so repeated calls
leaked memory without bound.

### The UTF-8 panic

`str` indexing is by **byte**. `&s[..N]` panics when byte `N` is not a `char`
boundary. Every place the crate bounded a user-supplied string by a fixed byte
budget was a latent panic — a title whose byte `N` lands mid-emoji, mid-CJK,
or mid-accented-letter aborts the request task. The digest row builder
(`notify_digest`) is where production actually hit it.

The audit found **more sites than the nine originally reported**, because
`&s[..s.len().min(N)]` is the same bug: when `len > N` it still slices at byte
`N`. **22 unsafe slice expressions** were removed across four files —
`routes.rs` 15, `gist.rs` 5, `trainer.rs` 1, `notify.rs` 1 — counted by
grepping the removed lines of the P0 diff, not by hand.

(The P0 commit message says 21; the verified count is 22. The diff is
authoritative.)

Notably, `web_ingest::truncate` already implemented the correct rule — the fix
existed in the codebase and had simply never propagated.

## Decision

### P0 — one truncation helper, every site routed through it

`str::floor_char_boundary` is unstable, so `text::truncate_at_char_boundary`
provides a stable equivalent: round the byte budget **down** to the nearest
char boundary.

**Semantics chosen: byte budget, not character count.** The helper never
returns more than `max_bytes`, so it preserves the existing size-bound
intention exactly and differs from the old code *only* where the old code
panicked. The alternative — truncating to N *characters* — would change
output length at every one of those 22 sites, and nothing in the code says whether
those budgets are display widths or downstream field limits. A hotfix should
not silently answer that question. Hence the name says `at_char_boundary`,
not `_chars`.

`web_ingest::truncate` now delegates, so there is one implementation.

**Left alone, safe by construction:**

| Site | Why safe |
|---|---|
| `auth.rs:32` `api_key[..8]` | `HeaderValue::to_str` succeeds only on visible ASCII |
| `bin/local.rs` `hash[..2]` | hex digest |
| `routes.rs` wasm magic `[..4]` | `Vec<u8>`, not `str` |
| `graph.rs` `emb[..prefix]` | `f64` slices |
| `web_ingest.rs:73` `pages[..N]` | `Vec<Page>` |
| every `&s[..pos]` from `.find()` | ASCII needle ⇒ `pos` is always a boundary (verified individually at `routes.rs` ×2, `gist.rs`, `pipeline.rs` ×3) |
| `notify.rs` `inject_tracking` | guarded by `ends_with("</div>")`; rewritten as `strip_suffix` for clarity |

### P1 — authenticate per endpoint, by caller, not uniformly

Applying one extractor everywhere would have broken live integrations. The
caller inventory was established from Cloud Scheduler jobs, Pub/Sub push
subscriptions and IAM bindings (all read-only), and each endpoint decided on
its own evidence:

| Endpoint | Real caller | Decision |
|---|---|---|
| `/v1/pipeline/inject/batch` | `brain-pubmed-daily` scheduler, **already sends `Authorization: Bearer`** | `AuthenticatedContributor` + write rate limit. No behaviour change for the live caller. |
| `/v1/pipeline/inject` | `curl/7.81.0`, an ad-hoc script | `AuthenticatedContributor` + write rate limit. **Risk accepted:** whether that script sends a header is not observable from logs. It is an unauthenticated public write endpoint producing undeletable rows; the security case outweighs the chance of a 401 that is fixed by adding one header. |

**Write rate limiting is an addition beyond "add authentication".** It was
included for consistency with every sibling write route, and because an
authenticated-but-unlimited public write endpoint is only marginally better
than an unauthenticated one. The limit is 500 writes/hour per pseudonym
(`rate_limit.rs`, `default_limits`), and a batch of 100 items consumes one
token. The observed `/v1/pipeline/inject` caller runs at roughly 2/min
(~120/hour), comfortably inside the limit, so this should not 429 it.
| `/internal/*` (4 routes) | `ruvbrain-sse` proxy | `verify_system_key` (`BRAIN_SYSTEM_KEY`) — the mechanism `/v1/notify/digest` already uses. Proxy updated to send it. **Deploy dependency, see below.** |
| `/internal/session/create` | as above | additionally capped at `MAX_SSE_SESSIONS = 1024` |

`internal_queue_drain` returns an empty array rather than a 401 on auth
failure, to keep its response shape. That is a **debuggability trade**: a
misconfigured proxy polls forever seeing "nothing yet" instead of logging a
401. Chosen because the drain loop polls every 100ms and a hard failure there
is noisier than a quiet one; revisit if it ever masks a real outage.
| `/v1/pipeline/pubsub` | `brain-inject-push`, OIDC as `ruvbrain-scheduler@` | **unchanged** — see below |
| `/v1/email/inbound` | Resend webhook | **unchanged** — see below |
| `/v1/chat/google` | Google Chat add-on | **unchanged** — see below |

`delete_memory_as(.., system)` lets a `BRAIN_SYSTEM_KEY` holder delete any
row, so pipeline injections are recoverable. This was chosen over changing
`process_inject`'s owner derivation: making injected rows belong to the
authenticated pseudonym would be cleaner, but it silently rewrites attribution
and reputation accounting for the existing scheduler jobs, and the set of
dashboards that filter on the `pipeline:` prefix is not enumerable from here.

> **DEPLOY DEPENDENCY — blocking.** `ruvbrain-sse` has only `BRAIN_API_URL` in
> its environment today. It must be given `BRAIN_SYSTEM_KEY` **before or with**
> the API deploy, or the SSE transport breaks.
>
> The asymmetry matters: `verify_system_key` **fails open** when
> `BRAIN_SYSTEM_KEY` is unset — inherited behaviour, unchanged here, but the
> SSE deploy story now depends on it. The API service *has* the key, so it
> starts enforcing the moment it deploys. The SSE proxy, lacking it, sends no
> header and is rejected. The gate is therefore on the *caller's* config, not
> the server's.
>
> A consequence of that same fail-open rule: in any environment where
> `BRAIN_SYSTEM_KEY` is unset, `/internal/*` stays fully open. That is the
> pre-existing convention for dev, not a new hole, but it means this fix is
> only load-bearing where the secret is configured.

### P2 — decouple the CSR cache from the graph lock

`csr_cache` becomes `RwLock<Option<Arc<CsrMatrix<f64>>>>` with an
`AtomicBool` dirty flag and a `csr_build` mutex for a double-checked rebuild,
so a burst of concurrent searches after an inject performs one rebuild instead
of one per thread. `ensure_csr`, `rebuild_csr`, `ranked_search` and
`pagerank_search` all take `&self`; the route takes `state.graph.read()`.
`rebuild_csr` builds the matrix before taking the cache write lock, so readers
wait only for a pointer swap.

**On staleness — the trade-off was surfaced and then declined.**

The framing anticipated that decoupling means accepting searches that run one
inject behind. It does not, as implemented. `ensure_csr` still rebuilds
*synchronously* with respect to the search that triggers it, so **there is no
staleness window at all**. What changed is only *which* lock is held: the
rebuild no longer excludes other readers.

Rebuilding in the background — letting searches proceed against a
one-inject-stale CSR — was considered and rejected. It trades a correctness
property for latency that has not been measured, and the contention problem is
already solved without it. If a future benchmark shows the synchronous rebuild
still spikes first-search latency unacceptably, that is the moment to revisit;
it should not be pre-paid for.

Writers still wait for in-flight readers. That interaction remains and is much
smaller.

**Not done, flagged for the reader:** the search route holds its guard
synchronously on one of only two tokio workers. Wrapping the whole search in
`spawn_blocking` is a small change that addresses the 2-worker starvation
directly, and is orthogonal to the lock fix. Left out to keep this change
reviewable.

### P3 — build the sparsifier off-lock

`rebuild_sparsifier` split into:

- `sparsifier_snapshot(&self)` — COO entries plus node/edge counts, under a read lock
- `build_sparsifier_from(entries, n)` — an associated fn over plain data; no `self`, so no lock can be held across it; this is what runs in `spawn_blocking`
- `install_sparsifier(&mut self, ..)` — a brief write lock

Making the build concurrent with mutation creates failure modes that did not
exist while it ran under a write lock. The install is guarded on all of them:

- **Edges appended during the build are replayed.** `add_memory` only feeds
  the sparsifier while it is `Some`, and it is `None` for the whole build
  window, so those edges would otherwise be silently dropped.
- **Reassigned node positions abort the install** (`index_generation`).
  A sparsifier is built against node *positions*. Comparing node/edge counts
  is **not sufficient**, which was the first version of this guard and was
  wrong: a `remove_memory` followed by any `add_memory` restores both counts
  while every index past the removed node has shifted, and
  `rebuild_from_batch` reassigns *every* position even for an identical
  memory set, because it iterates a `DashMap` whose order is arbitrary.
  `GRAPH_AUTO_REBUILD=true` in production, so that second path is live.
  A `u64` generation counter, bumped wherever `node_index` is rebuilt rather
  than appended to, is compared for equality on install.
- **A shrinking graph aborts the install** — a cheap check retained on top of
  the generation counter.
- **An already-installed sparsifier is never overwritten.**
  `rebuild_from_batch` is followed by an inline `rebuild_sparsifier` on the
  small-graph path (`routes.rs`), so without this the background build could
  clobber a freshly correct sparsifier with an older one.

Better to have no sparsifier than one that silently describes the wrong graph:
it is an analytics accelerator, and every consumer already handles `None`.

The `> 5_000_000` skip is left in place. Its stated reason — a write lock held
across the build — no longer applies, but CPU cost at that size has not been
measured, so widening the band would be a guess.

### P4 — paginate before cloning

`list_memories` collects 24-byte `(key, Uuid)` pairs, partitions with
`select_nth_unstable_by` at `offset + limit`, sorts only that prefix, and
clones only the returned rows.

Ordering changes in two ways, both deliberate:

- **`UpdatedAt` now compares microseconds, not nanoseconds.** The key is an
  `f64` (exactly representable at microsecond resolution for any plausible
  timestamp); the previous comparator used `DateTime::cmp`, i.e. nanoseconds.
  Rows written within the same microsecond now tie-break on id rather than on
  sub-microsecond time.
- **Ties are now deterministic.** Equal sort keys were previously ordered by
  `DashMap` iteration order, which can differ between two identical requests.

The equivalence test uses the *new* key in its reference implementation, so it
proves the partition-and-paginate logic is correct — it does **not** prove
byte-for-byte equivalence with the old nanosecond ordering.

## Deliberately not decided here

Each of these is a real gap. None is guessed at, because a wrong choice breaks
a live integration or invents a configuration value.

**1. `/v1/pipeline/pubsub` — needs OIDC verification, not a contributor key.**
The in-code comment claimed "Cloud Run validates Pub/Sub OIDC tokens
automatically". That is **false for this service**: Cloud Run validates the
token only when the service *requires* authentication, and `ruvbrain` grants
`run.invoker` to `allUsers` because it serves the public page from the same
service. The comment is corrected in place, because left standing it tells the
next reader the endpoint is protected.

The live `brain-inject-push` subscription *does* attach an OIDC token for
`ruvbrain-scheduler@ruv-dev.iam.gserviceaccount.com`. The fix is to verify it
in-handler (issuer, signature against Google's JWKS, and the expected service
account). Not done here: it needs a new dependency and cannot be exercised against a
real token from a unit test.

The audience is no longer an unknown — `brain-inject-push` sets no explicit
`audience`, which means Google signs the token with the push endpoint URL
(`https://pi.ruv.io/v1/pipeline/pubsub`) as the audience. So a future
implementation should expect exactly that, with
`ruvbrain-scheduler@ruv-dev.iam.gserviceaccount.com` as the verified email.

Note that `AuthenticatedContributor` is the **wrong** mechanism here: a Google
OIDC JWT is far longer than the extractor's 256-byte ceiling, so applying it
would reject every legitimate push. This is the concrete case where
uniform application of one extractor would have broken production.

**2. `/v1/email/inbound` and `/v1/chat/google` — machine callers whose
verification material does not exist in the service config.** Resend signs
webhooks with Svix headers against a webhook secret; Google Chat sends a
JWT that must be verified against Google's keys with the project number as
audience. Neither a webhook secret nor an audience is present in the
service's environment. The right value cannot be determined from here, so
nothing was changed.

**3. The API key check is length-only.** `auth.rs` accepts any string of
8..=256 bytes and derives a pseudonym via SHAKE-256. There is no registry and
no allowlist, so **any 8-character string authenticates**. `brain-ui` — the
hardcoded default in the page — is exactly 8 bytes. The live scheduler jobs
rely on this: they send static strings like `ruvector-swarm` and
`ruvector-crawl-2026`.

This means P1 raises the bar from "nothing" to "send any 8-character string".
That is a genuine improvement — it closes drive-by and unauthenticated
automated writes, brings the endpoints under the write rate limiter (which is
keyed on pseudonym), and makes writes attributable. It is **not** an access
control system. Introducing a real key registry would break every scheduler
job simultaneously and is a separate design decision.

**4. `clippy::await_holding_lock` is allowed crate-wide.** Re-enabling it
would likely have caught M1 and M2. Not changed here.

**5. The three-month deploy gap.** The serving image predates `main` by ~3
months. Every fix in this ADR is inert until a deploy, and the gap itself is a
risk: the larger it grows, the more a deploy changes at once and the harder a
regression is to attribute. Deploys are human-authorized; this ADR does not
perform one.

**6. M4 — the full graph rebuild held the graph write lock for O(n²) work.
Fixed in a follow-up (P5 below); this was the dominant 504 signature as of
2026-09-26.**
`KnowledgeGraph::rebuild_from_batch` computes every pairwise cosine
(~1.8B pairs at the live 59,758 nodes) and ran with `graph.write()` held at
two sites:

- `rebuild_graph` in `/v1/pipeline/optimize`, inside the async handler (not
  `spawn_blocking`), followed by an inline `rebuild_sparsifier`. Triggered by
  the ENABLED Cloud Scheduler jobs `brain-graph` (`0 */6 * * *`) and
  `brain-full-optimize` (`0 3 * * *`).
- The post-hydration rebuild in `create_router`'s background task, which
  runs in `tokio::spawn` on a worker thread while the server is already
  serving. Every cold start pays it.

Production logs for 2026-09-23..26 show `Graph rebuilt from batch (ADR-149
P3)` at exactly 03:00/06:00/18:00 UTC (the scheduler jobs) and at ~hh:12–:18
on each fresh instance id (cold starts), each followed by 504 bursts. P2 and
P3 do not touch this path.

#### P5 — build the graph off-lock (fix for item 6)

Same shape as P3, in `src/graph/rebuild.rs`:

1. **`begin_rebuild(&mut self)`** — brief write lock. Returns a ticket, or
   `None` if a rebuild is already in flight. The marker lives *in the graph*,
   so it is single-flight across both sites (a `brain-graph` tick landing
   while a fresh instance is still on its post-hydration rebuild is refused,
   not doubled). It also arms a **mutation log**.
2. **`build_batch(memories, threshold, threads)`** — an associated fn over
   plain data (no `self`, so no lock can be held across it), run in
   `spawn_blocking`. The memory snapshot (`store.all_memories()`, a clone of
   ~60k memories) is taken *inside* the blocking closure too, and *after* the
   log is armed. Readers keep using the old graph; writers keep mutating it.
3. **`install_batch(&mut self, ticket, build)`** — brief write lock. Swaps
   the new adjacency in, clears mincut/sparsifier, marks the CSR dirty, bumps
   `index_generation` (so an in-flight P3 sparsifier build refuses to
   install), then **replays the log in order**: an add whose id the snapshot
   already has is skipped; any other add is inserted exactly as
   `add_memory` would; a remove deletes the node if the snapshot still has
   it. The replaced structures are returned and dropped *after* the lock is
   released. A stale ticket (e.g. an in-place `rebuild_from_batch` ran
   meanwhile, which cancels the in-flight rebuild) installs nothing.

The delta replay is not optional: every write path calls `graph.add_memory`
*before* the store write, so without it any share landing during a
50-second build would silently vanish from the graph at swap time.

Both serving sites call it: the optimize action through `spawn_rebuild`, a
**detached** `tokio::spawn` the handler merely awaits — Cloud Scheduler's
HTTP deadline is shorter than a 60k build, and a disconnect drops the
handler future; the rebuild must neither be wasted nor wedge the marker. An
`AbortOnDrop` guard releases the marker on panic or cancellation. The
sparsifier now follows through the P3 off-lock path at both sites (the
optimize action previously ran `rebuild_sparsifier` inline under the write
lock with no size gate — a second heavy op at the same site). The
post-hydration path caps it at the same `<= 5_000_000` edges as `main.rs`'s
background task; the optimize action keeps its old "always" behaviour.
`rebuild_from_batch` survives as a thin build-then-swap wrapper for the
offline worker binary and tests, documented as not for serving paths. The
log line `Graph rebuilt from batch (ADR-149 P3)` is kept verbatim (with
`build_ms`, `threads`, `install_ms`, `replayed_*` fields added) because ops
correlate 504 bursts against it.

**Semantics are exact, not approximate.** Same nodes, same positions (input
order), same edges in the same `(i, j)` order, bitwise-identical `f64`
weights. The edge pass is cheaper only because each vector's norm is
computed once, with *exactly* the accumulation order `cosine_similarity`
uses, so `dot / (|a|·|b|)` is the same double (Rust does not contract to
FMA). Tests assert edge-for-edge bit equality against the verbatim old loop,
sequential and parallel. The pass is split into contiguous, pair-balanced
row ranges over `std::thread::scope` (no new dependency); concatenating them
in order reproduces the sequential edge order. Threads default to
`available_parallelism() - 1` (overridable via `GRAPH_REBUILD_THREADS`) so
one core is always left for serving — **on Cloud Run's `--cpu=2` that is one
build thread**, so the production wall time is the `threads=1` row below.

**Approximate kNN via HNSW was considered and not implemented.** It would
turn a similarity-*threshold* graph into a *k-nearest* graph — a different
edge set and different PPR/partition results — to buy wall time. Once the
build is off-lock, wall time is no longer what causes 504s; request latency
is. So no approximate variant exists and no recall figure is claimed. If
the corpus grows until even the off-lock build is a CPU problem on 2 vCPU
(~O(n²): 60k → 50 s single-threaded; 200k would be ~9 min), that is the
point to revisit, with a recall check against this exact builder.

**Measurements** (release, `rebuild_benchmark`, `#[ignore]`d; synthetic
clustered L2-normalised embeddings at the live `EMBED_DIM = 128`, cluster
size 40 so the 60k graph has 1.15M edges ≈ the live ~1.2M; 2 OS reader
threads each doing `graph.read()` + `ranked_search(q, 10)` in a loop; 32-core
dev box under load average ~8–17 from other jobs, so treat absolute numbers
as indicative):

| n | path | rebuild wall | read p50 | read p99 | read max | reads served |
|---|---|---|---|---|---|---|
| 10k | idle (no rebuild) | — | 1.8 ms | 2.6 ms | 3.2 ms | 1541 / 3 s |
| 10k | **old** (under write lock) | 2.58 s | — | 2.58 s | 2.58 s | 54 |
| 10k | new, 1 thread | 1.26 s | 1.9 ms | 2.9 ms | 8.0 ms | 684 (634 during build) |
| 10k | new, 31 threads | 0.12 s | 1.9 ms | 7.9 ms | 8.2 ms | 89 |
| 60k | idle (no rebuild) | — | 24.5 ms | 29.2 ms | 29.4 ms | 228 / 3 s |
| 60k | **old** (under write lock) | **96.4 s** | — | **96.4 s** | **96.4 s** | 12 |
| 60k | **new, 1 thread (= Cloud Run)** | **50.7 s** | 28.1 ms | **31.1 ms** | **55.2 ms** | 3378 (3370 during build) |
| 60k | new, 31 threads | 3.9 s | 74.8 ms | 108.8 ms | 110.9 ms | 109 |

- With the old path every read that arrived during the rebuild waited for
  all of it: at 60k that is 96 s, over Cloud Run's request timeout — the
  504 storm. With P5, reads during the rebuild are within noise of idle; the
  60k max of 55 ms is the first search after the swap synchronously
  rebuilding the 1.15M-edge CSR (P2's documented, deliberate behaviour).
- The write lock is held for the swap only: 0.3–0.8 µs measured inside
  `install_batch` (no concurrent mutations to replay in the bench; each
  replayed add costs one O(n) `add_memory`).
- The 31-thread rows show the CPU-contention trade-off the `-1` default
  guards against: reads slow (not block) when the build saturates cores.
- Wall time at 1 thread is ~1.9x faster than old from the hoisted norms.

Every new test was confirmed to fail against the code it protects:

| Test | Verified by |
|---|---|
| `reads_stay_responsive_during_rebuild_and_result_is_exact` (3k nodes, 2 reader threads, each read < 200 ms, reads overlap the build, final graph = exact rebuild + the mid-build add) | taking `graph.write()` inside the build closure (the old behaviour) → fails |
| `mutations_during_build_are_replayed` | removing the replay loop → fails (so does the test above) |
| `cancelled_rebuild_releases_the_marker` | disabling `AbortOnDrop` → fails |
| `begin_rebuild_is_single_flight`, `concurrent_rebuilds_are_single_flight` | removing the in-flight check → both fail |
| `install_batch_invalidates_an_in_flight_sparsifier_build` | removing the `index_generation` bump from the swap → fails |
| `build_batch_matches_reference_exactly_sequential_and_parallel`, `prenormed_cosine_is_bitwise_identical` | bit-equality (weights via `to_bits`) against the verbatim old loop at 1/2/3/8 threads, incl. a zero vector and a wrong-dimension vector |

**Single-flight vs. cold start is safe to skip.** `/v1/pipeline/optimize` sits behind `check_read_only`, which (since #1022) refuses while `store.is_hydrated()` is false, so a scheduler-triggered rebuild can only begin after hydration completed and therefore snapshots the complete store. If the post-hydration rebuild then gets `AlreadyRunning`, the in-flight rebuild is already building from a full snapshot; nothing is lost. Conversely, when the optimize action finds the post-hydration rebuild in flight it reports the action as succeeded-and-skipped (`true`), not as a failure. Optimize runs themselves were already serialised by `optimize_semaphore` (a second concurrent run gets 429), so the simultaneous 03:00 firing of `brain-graph` and `brain-full-optimize` behaves as before.

**Transient memory.** The old code cleared the graph before rebuilding; P5 keeps the old graph serving while the snapshot and the new build exist, so all three coexist for the build duration — roughly +80 MB at 60k nodes / 1.2M edges (memory clones, second node map with embeddings, second edge Vec) on a 4 GiB instance. Estimated from sizes, not measured.

**Pre-existing race, not widened:** because writes hit the graph before the
store, a memory whose `graph.add_memory` lands *before* `begin_rebuild` and
whose store write lands *after* the snapshot is absent from the new graph.
The old code had the same window (between `all_memories()` and acquiring
`graph.write()`); P5's is no wider. Removals are ordered store-first, so a
removal cannot resurrect a node through this path.

## Consequences

- Production stops crashing on non-ASCII titles. The panic → graceful
  shutdown → restart cycle ends.
- The public write endpoints require a token. See caveat 3 for exactly how
  much that is worth.
- Searches no longer serialise behind each other after an inject; cold starts
  no longer stall behind the sparsifier build; `list` stops scaling with the
  corpus.
- `ruvbrain-sse` cannot be deployed independently of this change without
  first receiving `BRAIN_SYSTEM_KEY`.
- `/v1/pipeline/inject` may 401 for its unidentified `curl` caller until a
  header is added.

**No performance figure is claimed for P0–P4.** (P5, item 6, carries its own measurements.) No benchmark was
run. Every claim is about lock structure or asymptotic shape, both readable
from the code. Anything phrased as a speedup would be invented.

## Verification

Baseline before any edit, so pre-existing and introduced failures could not be
confused: `cargo fmt --check` clean, 146 tests passing,
`cargo clippy --all-targets --all-features -- -D warnings` clean. After:
**166 unit tests plus 1 doctest passing**, fmt and clippy still clean on the
final commit. After P5 (item 6): **181 unit tests plus 1 doctest passing**, fmt clean,
`cargo clippy -p mcp-brain-server --no-deps --all-targets -- -D warnings` clean.

Every new test was confirmed to **fail** against the code it protects:

| Test | Verified by |
|---|---|
| helper boundary cases | naive `&s[..max]` body → panics `byte index 120 is not a char boundary; it is inside 'é'` |
| digest rows survive multibyte titles | reverting `format_digest_rows` to byte slicing → both tests panic |
| `ranked_search` under concurrent read locks | reverting to `&mut self` → **fails to compile** (`cannot borrow ... as mutable`) |
| pipeline rows deletable by system | removing the `system \|\|` clause → fails |
| sparsifier edge replay | removing the replay loop → test fails |
| sparsifier shrink guard | removing it → test fails |
| sparsifier `index_generation` guard | removing it → the remove-plus-add and `rebuild_from_batch` tests fail, while the shrink and overwrite tests still pass (they are covered by the other guards) |
| `list_memories` pagination | off-by-one in the partition point → fails |

The `ranked_search` case is worth calling out: it is enforced by the type
system, so the regression cannot be reintroduced without the build breaking.

### Not verified

- **Nothing was exercised against production.** No deploy, no request to
  `pi.ruv.io`. All cloud inspection was read-only.
- **No load or latency measurement.** The 504s are explained by lock
  structure, not reproduced.
- The `/v1/pipeline/inject` `curl` caller's headers are not observable from
  request logs, so it is unknown whether P1 will 401 it.
- OIDC verification for pubsub is described, not implemented or tested.
- `brain-reclassify-daily` is `PAUSED` with status `-1`. It sends an OIDC
  token to `/v1/reclassify`, which uses `AuthenticatedContributor` — and a
  Google OIDC JWT exceeds that extractor's 256-byte ceiling, so it would be
  rejected. Whether that is *why* it was paused is **not established**; the
  timing was not checked. Recorded as an observation, not a diagnosis.
- Local build is stable `rustc 1.98.1`; the Dockerfile pins
  `nightly-2026-03-20` (for a nalgebra ICE). The container build was not run.
