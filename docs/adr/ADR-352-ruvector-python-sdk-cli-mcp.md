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

The sibling site `signal-to-swarm.ruv.chatgpt.site` is unrelated to vector search (an ESP32/
Wi-Fi-sensing field guide that name-drops RuVector as a *future* tool for sensor-disagreement
analysis — confirms the brand context, not a technical pattern to copy).

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

## Security

- Input validation at every PyO3 boundary: dimension checks before any unsafe-adjacent NumPy
  buffer read (already present in salvaged `rabitq.rs`); CLI/MCP path arguments resolved and
  checked against directory traversal (`os.path.realpath` + prefix check) before touching disk.
- No pickle. Collection metadata sidecar is JSON; vectors are raw float32 buffers with an
  explicit header (mirrors the existing `.rbpx` format's magic-byte approach).
- MCP HTTP transport (M2+): bearer-token auth, same shape as the live-probed starter site's
  `Authorization` header convention.
- `cargo audit` and `pip-audit` to run before publish; results recorded in the loop-state file,
  not fabricated here before they're run.

## Benchmark methodology

No fabricated numbers. Real comparator: this session confirms no `hnswlib`/`faiss`/`chromadb`/
`qdrant-client` is pre-installed (`pip list` empty for all four) — one will be `uv pip install`ed
specifically to get an honest side-by-side, or the absence will be stated explicitly if install
is skipped. Metrics: p50/p99 single-query latency, QPS, recall@10, build time, wheel size — same
shape as `04-milestones.md`'s acceptance tests, measured on this host (32-core, per
`ruv_system_info`), not assumed from `ruvector-rabitq/BENCHMARK.md`'s numbers (those were a
different host).

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
