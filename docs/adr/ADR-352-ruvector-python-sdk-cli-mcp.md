# ADR-352: ruvector Python SDK, CLI, and MCP Server (PyO3 core, SOTA scope)

## Status

**Proposed, with M1 implemented and verified.** 2026-10-01. Supersedes the scope (not the
binding strategy) of the existing planning set at `docs/sdk/01-survey.md` through
`06-decision-record.md` (written 2026-04-25 against `main@2e68f0c9f`). Implementation happens on
branch `feat/python-sdk` in worktree `/home/ruvultra/projects/ruvector-python`, created from
`origin/main@5a93328f2`.

### Verification legend

- **[V]** Verified in this session by running the command shown and reading its output.
- **[D]** Design decision carried over unchanged from `docs/sdk/02-strategy.md` — reference only,
  not re-argued here.
- **[U]** Unverified / deferred — stated as such, not fabricated.

## Context

`docs/sdk/01-06` already did the hard strategic work of picking a binding approach (PyO3 +
maturin + abi3, §"Decision") and scoped a 4-milestone plan: M1 RaBitQ, M2 ruLake, M3 embeddings,
M4 A2A client (`docs/sdk/04-milestones.md`). That plan was **deliberately narrow**:
`01-survey.md` §"What we are deliberately ignoring" explicitly excludes MCP ("MCP is a
coordination protocol; if a Python user wants MCP they use the official MCP SDK") and says
"the Python SDK does not need to chase parity" with the npm `ruvector` package's "everything
bagel" (`01-survey.md` §"What the JS/TS SDK actually covers").

The actual request this ADR answers is broader than that plan on three axes the prior docs
excluded by design:

1. **CLI.** A polished Python CLI, not just a library.
2. **MCP server.** First-party stdio + streamable-HTTP MCP server, with a ChatGPT Apps SDK
   `ui://` widget — the exact pattern live-verified against
   `https://web-based-chatgpt-mcp-starter.ruv.chatgpt.site/api/mcp` in this session (below).
3. **"All major capabilities."** Not just RaBitQ/ruLake/Embed/A2A — the npm `ruvector` package's
   keyword list (`npm/packages/ruvector/package.json`) names HNSW, hybrid search, Graph RAG,
   FlashAttention-3, ColBERT, hyperbolic geometry, SONA/LoRA/EWC, MCP, as the capability bar.

This ADR keeps `02-strategy.md`'s binding decision **[D]** and `05-risks-and-tradeoffs.md`'s risk
list **[D]** verbatim, updates the one piece of that strategy that had already drifted (PyO3
version), and adds the scope the prior docs excluded.

### What's already true on disk (this session, verified)

- **[V]** A stale branch `origin/feature/python-sdk-m1` (926 commits behind `origin/main`, 1
  ahead) implemented M1's RaBitQ crate. Salvaged via `git checkout origin/feature/python-sdk-m1
  -- crates/ruvector-py` into this worktree, added to workspace `members` in `Cargo.toml`.
- **[V]** Compiled clean against current `main` with zero API drift in `ruvector-rabitq` itself
  (`from_vectors_parallel`, `search_with_rerank`, `persist::{save_index,load_index,MAGIC}`,
  `export_items` all unchanged since the M1 branch was written 2026-04-25).
- **[V]** `pyo3` was pinned to `0.22`/`numpy 0.22` (6 months stale; current is `pyo3 0.29.3`,
  `numpy 0.29.0` per `cargo search`). Bumped. Two mechanical API renames fixed:
  `Python::allow_threads` → `Python::detach`, `Python::get_type_bound` → `Python::get_type`
  (both confirmed by reading `pyo3-0.29.3/src/marker.rs`).
- **[V]** `cargo build -p ruvector-py` clean; `cargo clippy -p ruvector-py --all-targets --no-deps
  -- -D warnings` clean; `maturin develop --release` builds and installs an
  `abi3-py39-cp39-abi3-linux_x86_64` wheel; `pytest tests/` → **7/7 pass**
  (`test_smoke.py`: version, build+search, repr, dim-mismatch error, dtype error, save/load
  roundtrip, per-call rerank).
- **[V]** `uv tool install maturin` (1.15.0) was needed — the sandbox the M1 commit was written in
  did not have it, so none of this had actually been run before now.
- **[V]** `pyo3-async-runtimes` (successor to the dead `pyo3-asyncio` that `02-strategy.md`
  already flagged as "should track") exists at `0.29.0` on crates.io — confirms `02-strategy.md`'s
  async plan without changes needed, just the concrete crate name for M2+.

### The `ui://` widget convention (live-verified, not from memory)

Per advisor guidance, this was fetched live rather than assumed. `POST
https://web-based-chatgpt-mcp-starter.ruv.chatgpt.site/api/mcp` with
`{"method":"tools/list"}` (no auth needed for this call; `resources/list` on the same server
*does* require auth — "Cognitum authentication required") returns tools where a
ChatGPT-widget-backed tool carries:

```json
"_meta": {
  "ui": { "resourceUri": "ui://starter/plugin-launchpad-v4.html" },
  "openai/outputTemplate": "ui://starter/plugin-launchpad-v4.html",
  "openai/widgetAccessible": true,
  "ui/resourceUri": "ui://starter/plugin-launchpad-v4.html"
}
```

i.e. the tool definition's `_meta` carries **three redundant spellings** of the same pointer
(`ui.resourceUri`, `openai/outputTemplate`, `ui/resourceUri`) plus `openai/widgetAccessible:
true`, and the actual widget is a separate MCP **resource** at that `ui://` URI (fetched via
`resources/read`, which this server gates behind auth — consistent with the Apps SDK docs'
"widget resource" being a normal MCP resource whose URI scheme happens to be `ui://`, serving an
HTML document). `initialize` reports `protocolVersion: "2024-11-05"`, server name `mcp-studio
2.2.1`, capabilities `{tools:{listChanged:true}, resources:{listChanged:true}, prompts:{...},
completions:{}}` over streamable HTTP (plain `POST` + `Accept: application/json,
text/event-stream`; CORS headers advertise `Mcp-Session-Id` but none was issued for this
stateless flow).

The sibling site `signal-to-swarm.ruv.chatgpt.site` (both its landing page and `/live.html`,
fetched separately) is unrelated to vector search (an ESP32/Wi-Fi-sensing field guide and a
live radar/Wi-Fi presence-fusion dashboard respectively) that name-drops RuVector as a *future*
tool for sensor-disagreement analysis — confirms the brand context, not a technical pattern to
copy. All four URLs in the original task were fetched; these two contributed no MCP/`ui://`
pattern beyond what the `web-based-chatgpt-mcp-starter` endpoint already gave.

### What the npm CLI/MCP actually ship (anchor for "major capabilities", not a parity target)

`npm/packages/ruvector/bin/cli.js` is 10,546 lines with 100+ subcommands
(`agi`, `attractor`, `decompile`, `darwin`, `gate`, ... alongside `create`, `search`, `delete`,
`benchmark`, `export`, `embed`, `cluster`, `graph`, `attention`, `gnn`). `bin/mcp-server.js` is
4,381 lines. Per `01-survey.md`'s already-correct judgment, the Python CLI/MCP target the
**data-plane core** of this list (create/insert/search/delete/export/import/benchmark/serve),
not the research/experimental tail (`agi`, `attractor`, `decompile`, `darwin`, `gate` are
out of scope for v1 — they are not stable surfaces even in the JS package).

## Decision

**Keep `02-strategy.md`'s binding strategy unchanged [D]: PyO3 + maturin, single extension module
`ruvector._native`, `abi3-py39`, `pyo3-async-runtimes` for async, zero-copy NumPy via
`py.detach(...)` (renamed from `allow_threads` as of pyo3 0.23) around every call over ~50µs,
hand-written `.pyi` stubs, monorepo crate at `crates/ruvector-py/`.** Pin bumped to `pyo3 0.29`,
`numpy 0.29` (current as of 2026-10-01; re-check at each future milestone — R9 in
`05-risks-and-tradeoffs.md` already prices in ~1 person-day per PyO3 major bump).

**Add three in-tree siblings the prior plan excluded, as first-class ADR-352 scope:**

1. **`crates/ruvector-py` grows a generic vector-DB surface**, not RaBitQ-only: a `Collection`
   abstraction that can be backed by `ruvector-rabitq` (today, M1) and by the workspace's generic
   HNSW core (M2, binding `ruvector-core`'s index trait the same way `ruvector-diskann-node`
   does) with metadata filters, batch insert/delete, and persistence to the workspace's RVF
   container format (`ruvector-rvf` / `rvf_*` MCP tools already in this session's toolset are
   the Rust-side precedent). GNN/attention rerank and clustering (`ruvector-gnn`,
   `ruvector-attention`, `ruvector-cluster`) are **optional M3+ reranking hooks on `Collection`**,
   not separate top-level classes — this keeps the "G1: RAG in 5 lines" gate from
   `06-decision-record.md` intact while adding capability depth.
2. **`python/ruvector_cli/`** — a `click`/`rich`-based CLI (`ruvector` console-script entry
   point), covering: `create`, `insert`, `insert-batch`, `search`, `delete`, `export`, `import`,
   `benchmark`, `serve` (launches the MCP server), `info`. Fast startup is a hard requirement:
   lazy-import every subcommand module so `ruvector --help` doesn't pay for numpy/mcp import
   (measured via `python -X importtime`, M-series milestone gate below).
3. **`python/ruvector_mcp/`** — an MCP server (stdio default, `--http` for streamable-HTTP) built
   on the official `mcp` Python SDK, wrapping the same `Collection` surface as tools
   (`vector_search`, `vector_insert`, `vector_create_collection`, `vector_stats`, ...) plus one
   ChatGPT-Apps-SDK widget tool (`vector_explore`) whose `_meta` matches the live-verified shape
   above, backed by a `ui://ruvector/explore.html` resource (a small static HTML/JS scatter-plot
   + result-list widget, no build step, consistent with the "no server round trip for the widget
   shell" pattern the starter site uses).

**Reverses two explicit non-goals from the prior docs**, recorded here for anyone diffing against
them later: `01-survey.md`'s "if a Python user wants MCP they use the official MCP SDK" (now:
we ship one, built *on* the official SDK, not instead of it — not a contradiction, a
clarification) and its implicit CLI exclusion (the prior plan has no CLI milestone at all).

## Revised milestones

| M | Scope | Status |
|---|---|---|
| **M1** | RaBitQ index, 4 variants, persistence, PyO3 wheel. | **Done, verified this session** (see above). |
| **M1.5** *(new)* | `Collection` generic wrapper over M1's `RabitqPlusIndex` + metadata filter dict + JSON-sidecar persistence (stopgap before RVF in M2). CLI `create/insert/search/delete/export/import/benchmark/info`. MCP server (stdio) over the same surface, incl. the `ui://` widget tool. | This session, below. |
| **M2** | ruLake bindings (unchanged from `04-milestones.md`) + generic HNSW `Collection` backend + RVF persistence + streamable-HTTP MCP transport. | Deferred — scoped, not started. |
| **M3** | Embeddings (unchanged) + GNN/attention optional rerank hook on `Collection`. | Deferred. |
| **M4** | A2A client (unchanged). | Deferred. |

M1.5 is the pragmatic answer to "all major capabilities in one session": it buys the CLI + MCP +
widget + a *usable* (if not yet HNSW-backed) generic `Collection` API now, without blocking on
the RVF/ruLake integration work that M2 correctly scopes as multi-week.

## Capability landing — graph / GNN / attention / clustering / SONA

Per rUv's follow-up asking for the remaining "major capabilities" from the npm package's surface
where practical, rather than silently deferring all of them to M3/M4. Each landed as a standalone
PyO3 binding in `crates/ruvector-py/src/{graph,gnn,cluster,sona}.rs`, verified by this session —
not just by the implementing pass — with a hands-on smoke test of every new class/function after
integration (see commits on `feat/python-sdk` after `cebb38bc4`).

| Capability | Class/fn | Scope landed | Scope deliberately NOT landed |
|---|---|---|---|
| Graph (raw CRUD) | `ruvector.GraphDB` | create/get node, create/get edge, outgoing-edge traversal, in-memory only (no `storage` feature — `default-features=false` on `ruvector-graph`) | persistent graph storage, distributed/sharded graphs |
| Graph (Cypher) | `GraphDB.query_cypher` | `MATCH`/`WHERE` execution, ported from `ruvector-graph-node`'s `cypher_exec.rs` (confirmed NAPI-free, a clean copy-port) | `RETURN` as a real projection (parsed, not applied — matches the upstream contract exactly, not a new limitation), `CREATE`/`MERGE`/`SET`/`DELETE`/`REMOVE` (rejected or silently no-op, matching upstream), variable-length relationships, cross-`MATCH` joins |
| GNN forward-pass rerank | `ruvector.GnnLayer` | `forward()` on `ruvector_gnn::layer::RuvectorLayer` (message passing + multi-head attention + GRU + layer norm), `to_json`/`from_json` | training (no `backward()` wired to this type at all — an untrained layer's `forward()` is a Xavier/Glorot **random projection**, not a quality improvement; stated plainly in the binding's own doc comment, not buried) |
| Attention rerank | `ruvector.AttentionReranker` | `softmax(QK^T/√d)V` over a query + candidate set, returns both the blended vector and the raw per-candidate weights (the weights are what a RAG reranker actually wants to re-sort by) | multi-head variant (scalar-dot-product alone judged a complete, honest deliverable) |
| k-means clustering | `ruvector.kmeans()` | `ruvector_cluster_rag::cluster::kmeans` (Lloyd's), returns assignments/centroids/cohesion/cluster_sizes | — (small, complete surface) |
| SONA (inference) | `ruvector.SonaEngine` | `apply_micro_lora`/`apply_base_lora`/`stats`/`save_state`/`load_state` | the online-learning API (`begin_trajectory`/`tick`/`force_learn`/`find_patterns`) — real but only does something with a genuine reward signal a test can't fabricate; **on a fresh engine both LoRA forward passes are an *exact* identity transform** (zero-init projections, residual forward pass), stated as fact, not "close to one" |
| Quantization (Turbo4) | — | not evaluated this session | — |
| `ruvector-cluster` (the OTHER "cluster" crate) | — | **deliberately not bound** — it's distributed-sharding/consensus infrastructure (gossip discovery, consistent hashing, Raft-like consensus), not an ML clustering algorithm; binding it would need a running multi-node cluster, not a Python process. If "clustering" meant this crate specifically, say so and this gets revisited — the capability-table row above binds the actual k-means algorithm instead. |
| RVF persistence | — | **not practical this session** — ~30-file subsystem (COW pages, witness/crypto log, eBPF, federation); neither `ruvector-core` nor `ruvector-collections` depend on it today either, so adopting it for `Collection` persistence is a separate project, not a one-session add-on. |

Two real bugs this slice's integration pass found and fixed (both in shared infrastructure, not
in the new bindings themselves):

1. **`patches/hnsw_rs`'s stdout-corrupting `println!`** — found and fixed in the HNSW-backend
   benchmark pass (see above), before this capability slice started; mentioned again here
   because every new binding's `cargo test`/`pytest` run depended on it already being fixed.
2. **Three parallel forks all needed to append to `__init__.py`/`__init__.pyi`/`_native.pyi`'s
   `__all__` lists** — refactored `__init__.py`'s `__getattr__` to dispatch dynamically
   (`hasattr(_native, name)`) against the compiled module instead of a hardcoded name set,
   *before* forking, specifically to eliminate the bug class that made `ruvector.HnswIndex`
   unreachable earlier in this session (addable in Rust, forgotten in the dispatch set). This
   also pre-empted what would otherwise have been a 3-way merge conflict on the exact same
   set literal — the `__all__` list conflicts that did occur (append-only, different physical
   lines) were trivial 2-minute resolutions, not a design failure needing a redo.

## Security (run this session, results below — not projected)

- **Input validation at the PyO3 boundary**: dimension checks before any NumPy buffer read
  (`rabitq.rs` `build`/`search`/`add`/`add_batch`), C-contiguity enforced (non-contig raises
  `TypeError` rather than silently copying).
- **Path handling has two different rules for two different trust levels**, not one blanket
  rule: the CLI's `--path` runs with the caller's own filesystem authority (like `sqlite3
  file.db` — no traversal check needed, confirmed appropriate since the caller already has
  arbitrary filesystem access by definition). The MCP server is the actual trust boundary (a
  remote client sends a `name` string): `ruvector.mcp_server._safe_path` restricts collection
  *names* to `^[A-Za-z0-9_-]{1,128}$` — no `/`, no `..`, no absolute paths, no null bytes — then
  resolves under `RUVECTOR_MCP_DATA_DIR` and asserts `.relative_to(root)`. Verified by
  `tests/test_mcp_server.py::test_collection_name_traversal_rejected` against
  `"../etc/passwd"`, `"/etc/passwd"`, `"a/b"`, a null byte, and the empty string — all raise.
  Resource reads (the `ui://` widget) get the `mcp` SDK's own
  `ResourceSecurity(reject_path_traversal=True, reject_absolute_paths=True,
  reject_null_bytes=True)`, on by default in `MCPServer.__init__` — not something this module
  implements itself.
- No pickle anywhere. Collection metadata sidecar is JSON; the index itself is the existing
  `.rbpx` binary format (magic-byte header, `persist.rs`).
- **`cargo audit --file Cargo.lock`** [V]: ran against the full workspace lockfile (hundreds of
  crates). `ruvector-py`'s own dependency tree (`cargo tree -p ruvector-py`) does **not** include
  either of the two findings (`rustls-pemfile` RUSTSEC-2025-0134 unmaintained,
  `lru` RUSTSEC-2026-0253 unsound `pop()`) — both belong to unrelated workspace crates
  (networking/server crates elsewhere in the monorepo). Zero findings in `ruvector-py`'s own
  tree: pyo3, numpy (rust-numpy), ruvector-rabitq, rand/rand_distr/rayon/serde/thiserror.
- **`pip-audit`** [V]: run as `python3 -m pip_audit` inside the dev venv (the standalone
  `~/.local/bin/pip-audit` binary audits the *system* Python, not the active venv — caught this
  by seeing Ubuntu system packages like `cloud-init`/`ufw` in its output and re-running
  correctly). Result: **"No known vulnerabilities found"** across numpy, click, rich, mcp,
  pytest, mypy, and their transitive deps. `ruvector` itself is skipped (not yet on PyPI, can't
  be looked up — correct, not a failure).
- **`mypy --strict`** [V]: clean across `collection.py`, `cli.py`, `mcp_server.py`, `__init__.py`,
  and all 4 test files, after fixing two real type-narrowing bugs this surfaced (not just
  annotation noise) — see the commit history for specifics (a `callable()` branch that doesn't
  narrow a `Dict | Callable` union the way `isinstance(filter, dict)` does; a missing
  `_native.pyi` that silently typed every `RuVectorError` subclass as `Any`).
- **`npx @claude-flow/cli@latest security scan`** [V], run from `crates/ruvector-py/`: "No
  security issues found!" — Critical 0, High 0, Medium 0, Low 0, Total 0.
- **Concurrency** [V, fixed]: a second review pass found that `_cache`, `Collection._next_id`,
  `_metadata`, and `_tombstones` had no locking, and `ruvector serve --http` dispatches
  concurrent tool calls to worker threads — the same fact that made `RabitqIndex`'s old
  `unsendable` pyclass panic. Two concurrent `vector_insert` calls on one collection could read
  the same `_next_id` and both `add()` with the same id. Fixed with one process-wide
  `threading.RLock()` around every tool body that touches shared state (`RLock`, not `Lock`,
  because `vector_explore` calls `vector_search` from the same thread). Verified the fix is
  real, not cosmetic: temporarily neutered the lock and reproduced **3 duplicate ids out of 160**
  under 16 concurrent threads × 10 inserts each
  (`tests/test_mcp_server.py::test_concurrent_inserts_do_not_collide`); restoring the lock
  eliminates the collisions. This serializes all tool calls under `--http` (correctness over
  throughput — the right tradeoff for a single vector index, not a high-QPS service); `stdio`
  (the default transport) was never affected, since it's one request at a time regardless.
- MCP HTTP transport auth (bearer token, matching the live-probed starter site's convention):
  **not implemented this session** — `ruvector serve --http` currently has no auth. This is a
  real gap for internet-facing deployment; fine for the localhost/stdio default. Flagged as a
  concrete M2 follow-up, not silently left implicit.

## Benchmark — M1.5, RabitqPlus backend, random-Gaussian data (superseded as the default; kept for the record)

Host: 32-core x86_64, 123 GiB RAM (`nproc` / `free -h`). `hnswlib==0.8.0` installed fresh via
`uv pip install hnswlib` (needed `sudo apt-get install -y libomp-dev` first — no prebuilt wheel
for this platform, source build failed on a missing `omp` shared lib without it) as the real
comparator; `faiss`/`chromadb`/`qdrant-client` were not installed (time budget) — stated here
rather than silently substituted. Workload: i.i.d. standard-normal (Gaussian) float32 vectors,
seed 0, L2 distance, k=10. Ground truth is exact brute-force L2, computed per query.

**[V] Random-Gaussian data is close to worst-case for both algorithms** — all pairwise angles
concentrate near 90° in high dimensions, which is exactly the structure RaBitQ's rotation-based
binary quantization and HNSW's graph proximity both rely on *less* of than they would on real
embedding clusters. The recall numbers below are real but should not be read as "ruvector gets
64% recall" in general — `ruvector-rabitq/BENCHMARK.md`'s "100% recall@10" figure was measured on
a different (unspecified in this session) workload/host and is not reproduced here; this is an
honest discrepancy to flag, not a regression.

| n | dim | rerank_factor | ruvector p50/p99 (ms) | ruvector recall@10 | ruvector QPS | hnswlib p50/p99 (ms) (ef=50) | hnswlib recall@10 | hnswlib QPS |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 128 | 20 | 0.171 / 0.290 | 0.901 | 5,546 | 0.030 / 0.039 | 0.736 | 32,757 |
| 100,000 | 128 | 20 | 0.491 / 0.587 | 0.636 | 1,982 | 0.059 / 0.130 | 0.398 | 15,780 |

Recall climbs with `rerank_factor` as expected (same n=100,000, dim=128, 100 queries):

| rerank_factor | ruvector recall@10 | ruvector p50 (ms) |
|---|---|---|
| 20 | 0.63 | 0.495 |
| 50 | 0.789 | 0.848 |
| 100 | 0.888 | 1.464 |
| 200 | 0.96 | 2.426 |

hnswlib's `ef` has the same shape (n=100,000, dim=128, 100 queries):

| ef | hnswlib recall@10 | hnswlib p50 (ms) |
|---|---|---|
| 50 | 0.391 | 0.064 |
| 100 | 0.539 | 0.099 |
| 200 | 0.699 | 0.204 |
| 400 | 0.818 | 0.413 |

**Honest read:** at roughly matched recall (~0.82-0.89) on this synthetic worst-case workload,
hnswlib's p50 (≤0.413ms at ef=400) beats ruvector's RabitqPlus p50 (1.464ms at rerank_factor=100)
by roughly 3-4x. This is a real result on real-but-adversarial data, not a fabricated number, and
not necessarily representative of real embedding workloads (where RaBitQ's published benchmarks
claim parity or better — unverified in this session, no real-embedding dataset was on hand).
Build time favors ruvector at these sizes (0.053s vs 1.449s at n=100,000 — RabitqPlus's
`from_vectors_parallel` has no graph-construction cost). Memory is comparable (both ≈512-537
bytes/vector at dim=128, dominated by the stored f32 originals in both).

**Follow-up for a future session, not done here:** re-run this comparison on a real embedding
dataset (e.g. SIFT1M or a MiniLM-embedded text corpus once M3 ships) where RaBitQ's quantization
has actual angular structure to exploit, and extend to recall@10 ≥ 0.95 for both to find the
genuine crossover point rather than reading two unmatched curves.

## Benchmark — M2, HnswIndex backend (now the default), real embeddings

Per rUv's follow-up asking for both the HNSW-backed default *and* a real-embedding re-benchmark
(the random-Gaussian section above flagged its own worst-case-workload caveat). Dataset: 10,000
documents from `sklearn.datasets.fetch_20newsgroups` (train split, headers/footers/quotes
stripped, real English text), embedded with `all-MiniLM-L6-v2` (CPU, `sentence-transformers`) to
384-dim — 64.4 s one-time embedding cost, not counted in any latency number below. k=10, 200
queries, cosine distance, exact-cosine brute force as ground truth. `m=16, ef_construction=200`
fixed; `ef_search` swept at 50/100/200 on both sides via identical construction parameters.

**Two real implementation bugs found and fixed while investigating why the first run of this
benchmark looked suspiciously slow, per a second advisor review — not hypothetical, both
verified against the actual `ruvector-core`/`hnsw_rs` source:**

1. **A per-call `ef_search` override on `HnswIndex.search()` silently did nothing.**
   `SearchQuery.ef_search` exists on the Rust struct, but `VectorDB::search` (`vector_db.rs`)
   never reads it — it calls the generic `VectorIndex::search(&self, query, k)` trait method,
   which has no `ef` parameter at all; `HnswIndex`'s trait impl always uses
   `self.config.ef_search`, fixed at construction. Removed the misleading parameter from
   `HnswIndex.search()` rather than ship a kwarg that does nothing — `ef_search` is
   construction-time-only (`HnswIndex.create(..., ef_search=...)`), documented in both the Rust
   doc comment and the `.pyi` stub. (The benchmark's three `ef_search` rows were still valid —
   each used a freshly-constructed index with that `ef_search`, not the dead per-call path —
   but the parameter needed to go.)
2. **`Collection.search()` over-fetched `k * overfetch` (4x) candidates from Rust on every
   unfiltered search**, even with `filter=None` — the widen-on-undersupply loop has no reason to
   run at all when there's nothing to widen for. Added a fast path: `filter is None` now calls
   `HnswIndex.search(qvec, k, filter=None)` for exactly `k`, matching what `hnswlib` is asked
   for. Measured effect at `ef_search=50`: p50 0.304 ms → 0.252 ms (~17% faster) — real, but not
   the dominant cost.
3. **(Found, not fixed — a real bug in a vendored dependency, fixed instead):**
   `patches/hnsw_rs/src/hnsw.rs` had a hardcoded `println!` to stdout every 50,000 points
   inserted. This corrupts any consumer that frames a protocol over stdout — exactly what
   `ruvector serve` (stdio MCP transport) does. Removed the `println!` (an adjacent `trace!`
   already logs the identical message through the `log` facade, which a caller can opt into via
   a subscriber instead of inheriting a hardcoded stdout write). Verified with a 60,000-vector
   `insert_batch` producing zero stray stdout lines before the fix would have printed one.

| ef_search | ruvector p50 (ms) | ruvector QPS | hnswlib p50 (ms) | hnswlib QPS | recall@10 (both) | gap |
|---|---|---|---|---|---|---|
| 50 | 0.252 | 3,854 | 0.035 | 28,104 | 0.999 / 0.999 | ~7.2x |
| 100 | 0.391 | 2,388 | 0.059 | 16,758 | 1.0 / 1.0 | ~6.6x |
| 200 | 0.611 | 1,514 | 0.108 | 9,057 | 1.0 / 1.0 | ~5.7x |

**Honest read.** On real text embeddings (not the adversarial random-Gaussian workload above),
recall is essentially identical between the two at matched `m`/`ef_construction`/`ef_search` —
the algorithm is doing the same job either way. The latency gap narrowed (8.6x → ~5.7-7.2x) after
fixing bug #2 but did **not** close, and that residual is structural, not a bug: `HnswIndex.search`
copies the query into an owned `Vec<f32>` (`SearchQuery.vector` is `Vec<f32>`, not a borrow) and
constructs a Python dict per hit (`json_map_to_py`) even when the caller discards the metadata —
costs `hnswlib`'s bare-`labels`-array return doesn't pay. Separating "how much is PyO3/marshaling
overhead" from "how much is `hnsw_rs` itself being slower than `hnswlib`'s tuned C++" needs
profiling that wasn't done this session — stated as a gap, not guessed at. A batched `search_many`
entry point (amortizing the per-call marshaling over N queries) is the obvious next lever and is
not implemented.

Build time still favors ruvector at this n (0.28-0.32 s vs hnswlib's 0.08 s is actually a
**loss** here, inverted from the random-Gaussian section above — small-n HNSW construction is
cheap for both, and `VectorDB`'s per-insert storage-layer overhead (the in-memory `MemoryStorage`
path, still exercised even though nothing is written to disk) shows up at this n where graph
construction itself is not the bottleneck).

## Publishing prep (verified; not executed — out of scope per task boundary)

- **[V]** `maturin build --release --out dist` succeeds standalone (not just `maturin develop`):
  produces `ruvector-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl`, **345 KiB** (well under the
  8 MiB M1 budget).
- **[V]** Installed that wheel into a **completely fresh venv** (`uv venv
  /data/scratch/ruvector-freshwheel-venv`, no editable install, no dev-venv carryover) and ran
  the full test suite against it: **38/38 pass**.
- **[V]** `.github/workflows/python-wheels.yml` created: Linux x86_64/aarch64 (manylinux_2_28),
  macOS x86_64/aarch64, Windows x86_64 — 5 wheels via `PyO3/maturin-action` (see the workflow's
  header comment for why maturin-action over the literal "cibuildwheel" in `02-strategy.md`).
  Trusted publishing (PyPI OIDC) on the publish job, gated on a `python-v*` tag or an explicit
  `workflow_dispatch` input — **this workflow has not been run in CI and nothing has been
  published**; PyPI trusted-publisher registration for this repo/workflow is a one-time setup
  step that hasn't happened either (open question O1 from `06-decision-record.md` still stands).
- **Not done**: aarch64/cross-platform wheels are configured but unbuilt/untested locally (this
  host is x86_64 Linux only) — CI is the first real test of the cross-compile paths.

## Open questions carried over unchanged from `06-decision-record.md`

O1 (PyPI name), O2 (Python floor — now leaning abi3-py39 still, since 3.9 is EOL-but-deployed and
abi3-py39 costs nothing to keep wide), O3 (tokio sizing), O4 (`ort` coupling), O5 (A2A server
location), O6 (abi3 commitment) all still stand, unresolved, owner unchanged.

## Source pointers

- Strategy (unchanged): `docs/sdk/02-strategy.md`.
- Risks (unchanged): `docs/sdk/05-risks-and-tradeoffs.md`.
- Milestones baseline: `docs/sdk/04-milestones.md` (M2–M4 still apply as written).
- Salvaged + fixed M1 code: `crates/ruvector-py/` (this worktree, branch `feat/python-sdk`).
- Loop state / resume point: `docs/sdk/LOOP-STATE.md`.
- `ui://` widget live evidence: captured in this ADR above; raw JSON in session transcript.
