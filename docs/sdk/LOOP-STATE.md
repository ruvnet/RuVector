# Python SDK/CLI/MCP — loop state (resume point)

Updated: 2026-10-01 (full-scope checkpoint — MCP auth, Salesforce Agentforce, CI gap fix, docs
all landed). Owner: claude-flow agent on branch `feat/python-sdk`. PR:
https://github.com/ruvnet/RuVector/pull/1117 (draft).

## Where things live

- Worktree: `/home/ruvultra/projects/ruvector-python` (branch `feat/python-sdk`, tracks
  `origin/main`).
- ADR: `docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md` — read this first. Has both benchmark
  sections, security findings, publishing verification, and the capability-landing table, all
  with real commands/numbers.
- Rust crate: `crates/ruvector-py/src/{lib,rabitq,hnsw,graph,gnn,cluster,sona,error}.rs` (+
  `src/graph/{convert,cypher_bridge,cypher_eval,cypher_exec}.rs`).
- Python package: `crates/ruvector-py/python/ruvector/` (`__init__.py` — dynamic `__getattr__`
  dispatch, see below —, `collection.py`, `cli.py`, `mcp_server.py`, `integrations/{langchain,
  llamaindex}.py`, `.pyi` stubs). `collection.py` has two backends (`hnsw` default, `rabitq`).
- Tests: `crates/ruvector-py/tests/` — **201/201 passing** (smoke, collection [parametrized over
  both backends], cli, mcp_server [incl. auth], graph, graph_cypher, gnn, cluster, sona,
  langchain_integration, llamaindex_integration, salesforce_integration, salesforce_routes).
  Plus **49/49 `cargo test -p ruvector-py`** — now actually running in CI (see below; it wasn't
  before this checkpoint despite a comment claiming otherwise).
- Vendored patch: `patches/hnsw_rs/` (workspace `[patch.crates-io]`) — fixed a stdout-corrupting
  `println!` in it this session (would have corrupted the stdio MCP transport).
- CI: `.github/workflows/python-wheels.yml` — first real CI run already caught and fixed a
  rustfmt failure and an aarch64 maturin-action container-selection bug (see git log). Also
  patched `.github/workflows/release.yml` to exclude `ruvector-py` from 4 workspace-wide
  build/test steps (bare ubuntu-22.04, no Python setup — libpython linking unverified there).
  **Gap found and fixed this checkpoint**: `release.yml`'s own comment claimed the 49 Rust unit
  tests "run in python-wheels.yml instead" — checked the actual workflow file and they didn't
  (it only builds wheels, no `cargo test` step anywhere). Added a `rust-tests` job to
  `python-wheels.yml` that does; confirmed working locally first (`cargo test -p ruvector-py
  --release` with a venv python on PATH, no `extension-module` feature needed).
- Cargo target dir (NOT the default — root disk pressure, see `~/CLAUDE.local.md`):
  `/data/scratch/ruvector-python-target`.
- Python venvs: `/data/scratch/ruvector-py-venv` (dev, editable install, has every extra incl.
  langchain-core/llama-index-core/sentence-transformers/hnswlib/scikit-learn/torch);
  `/data/scratch/ruvector-freshwheel-venv` (clean wheel-install check).
- Benchmark scripts (scratchpad, not committed): `.../scratchpad/bench_compare.py` (M1.5
  RabitqPlus vs hnswlib, random-Gaussian), `.../scratchpad/bench_hnsw_real.py` (M2 HnswIndex vs
  hnswlib, real MiniLM/20newsgroups embeddings).
- Dependency-compile trial scratch: `.../scratchpad/trial_deps_check/` — used to verify
  ruvector-graph/gnn/attention/cluster-rag/sona compile together before forking; not needed
  again unless adding a 6th new dependency.

## Capability coverage (vs the npm `ruvector` package's keyword list). Update at every milestone.

| Capability | Status | Where | Notes |
|---|---|---|---|
| RaBitQ quantized index | ✅ done | `RabitqIndex`, `Collection(backend="rabitq")` | M1/M1.5 |
| HNSW index | ✅ done | `HnswIndex`, `Collection(backend="hnsw", default)` | M2 |
| Metadata filtering | ✅ done (partial) | `Collection.search(filter=dict)` | dict-filter in Rust for hnsw backend; callable predicate always Python-side (structural); rabitq backend always Python-side |
| Collections (named, CRUD) | ✅ done | `Collection` class | both backends |
| Persistence | ✅ done (own format, not RVF) | `Collection.save/load` | `.rbpx`+sidecar (rabitq) / `.npy`+sidecar (hnsw) |
| CLI | ✅ done, polished | `ruvector` console script | rich tables, colored/ranked output, progress spinner, `--no-color`; importtime unchanged |
| MCP server (stdio+HTTP) | ✅ done | `ruvector serve` | bearer auth via `RUVECTOR_MCP_TOKEN` + SDK `TokenVerifier`/`AuthSettings`, read/write scopes, live-verified with real curl |
| ChatGPT `ui://` widget | ✅ done | `vector_explore` tool | live-verified `_meta` shape |
| Graph (raw CRUD) | ✅ done | `ruvector.GraphDB` | create/get node+edge, outgoing-edge traversal; in-memory only |
| Graph (Cypher) | ✅ done (partial) | `GraphDB.query_cypher` | `MATCH`/`WHERE` only; `RETURN` parsed not projected, `CREATE`/etc rejected or no-op — matches upstream `ruvector-graph-node` contract exactly |
| GNN forward-pass rerank | ✅ done | `ruvector.GnnLayer` | untrained = random projection, stated explicitly, not sold as a quality win |
| Attention rerank | ✅ done | `ruvector.AttentionReranker` | trainless, returns blended vector + raw weights |
| k-means clustering | ✅ done | `ruvector.kmeans()` | `ruvector_cluster_rag`, NOT `ruvector-cluster` (premise mismatch, see below) |
| SONA (inference-only) | ✅ done | `ruvector.SonaEngine` | fresh engine = exact identity transform (zero-init + residual), stated as fact |
| RVF persistence | ❌ not practical this session | — | ~30-file subsystem; neither `ruvector-core` nor `ruvector-collections` depend on it today either |
| Embeddings (M3) | ❌ deferred | — | needs ONNX/`ort` + model download, separate milestone by design |
| `ruvector-cluster` (distributed) | ❌ premise mismatch | — | sharding/consensus infra, not ML clustering — the real k-means (above) is the correct binding target |
| Quantization (Turbo4) | ⬜ unexplored | — | exists in `ruvector-core`'s `QuantizationConfig`, not evaluated |
| LangChain VectorStore | ✅ done | `ruvector.integrations.langchain.RuVectorStore` | verified vs langchain-core 1.6.6 |
| LlamaIndex VectorStore | ✅ done | `ruvector.integrations.llamaindex.RuVectorStore` | verified vs llama-index-core 0.14.25; found+fixed a real distance-vs-similarity bug |
| Salesforce Agentforce | ✅ done (External Services + OpenAPI path) | `ruvector.integrations.salesforce`, `ruvector.salesforce_routes` | OAuth2 client-credentials, paginated SOQL, record sync, 3 actions (search/upsert/ground), generated OpenAPI 3.0 doc, own bearer auth (custom_route has no SDK auth). 28 tests, all mocked — no real org touched. Agentforce MCP: beta/AE-gated, documented not built. |
| Rust unit tests | ✅ done, now in CI | `#[cfg(test)]` in every `src/*.rs` | 49 tests, was 0; added a CI job to actually run them (wasn't running anywhere before this checkpoint) |

## Done this session (full list — see git log on `feat/python-sdk` after `5a93328f2` for commits)

M1 salvage → M1.5 Collection → M2 HnswIndex-default + real-embedding re-benchmark → CLI/LangChain/
LlamaIndex/Rust-tests (3 parallel forks, integration-tested, 1 real bug fixed: LlamaIndex
distance-vs-similarity) → pre-wiring (dynamic `__getattr__` dispatch + 5 new Cargo deps verified
by standalone trial compile) → graph/GNN/attention/cluster/SONA (3 more parallel forks,
integration-tested, 0 new bugs found beyond trivial `__all__`-list merge conflicts) → first real
CI run (rustfmt + aarch64 maturin-action container-selection bugs found and fixed).

**Every fork's claims were independently re-verified after cherry-picking**, not just trusted:
rebuilt the wheel, reran the full pytest+cargo test suite, ran `mypy --strict`, and hand-smoke-
tested every new public class/function directly against the rebuilt wheel before considering a
batch done.

## Next (priority order)

Everything from the previous checkpoint's list is now done: MCP HTTP bearer auth (implemented,
live-verified), Salesforce Agentforce (External Services + OpenAPI path, 28 tests, mocked only),
the Python user guide, the separate flagged README commit, and the CI gap (Rust tests weren't
running anywhere — now they are). Remaining, in priority order:

- [ ] CI: a full run with today's commits (Salesforce, CI-gap fix, docs) hasn't happened yet —
      check `gh pr checks 1117` once pushed, for the Linux aarch64/macOS/Windows legs and the new
      `rust-tests` job specifically.
- [ ] Final Slack summary + `ClaimReleased` block in the #swarm thread (ts 1790885023.580889),
      cc Dragan/Martin, plus a one-line pointer in #development — not yet sent as of this
      checkpoint (4 progress updates posted so far).
- [ ] PR description (#1117) needs the "land only after `ruvector` is live on PyPI" flag on the
      `docs(readme):` commit called out explicitly, plus the final capability-coverage table.
- [ ] Not done, lower priority: the 3 `known_limitations` bugs in `hnsw.rs`'s JSON converter
      (large-int precision loss, NaN→null, lone-surrogate error message) — characterized by
      tests, not fixed. The large-int one is a genuine ~5-line fix (raise instead of the f64
      fallback) worth doing before publish.
- [ ] Not started, explicitly deferred per the ADR: RVF persistence, Embeddings (M3), Turbo4
      quantization evaluation. Each has a stated reason in the capability table above, not a
      silent gap.
- [ ] Still no PyPI publish, no merge, no tags — boundaries unchanged throughout.

## Gotchas hit this session (don't rediscover)

- `ruvector-core`'s *default* Cargo features pull in `api-embeddings` (reqwest+rustls) for
  nothing this crate uses — depend with `default-features = false, features = ["storage",
  "hnsw", "simd", "parallel"]`. Same pattern needed for `ruvector-graph` (its default "full"
  feature pulls in tokio/moka/zstd/lz4/redb) — `default-features = false, features = ["simd"]`.
- **The opposite trap**: `ruvector-sona` (crate `ruvector-sona`, dir `crates/sona`) must NOT get
  `default-features = false` — its `serde`/`serde_json` are Cargo-"optional" but several modules
  use them with no cfg-gate at all, so disabling the default `serde-support` feature fails to
  compile. Confirmed by a real trial build in `trial_deps_check/`, not assumed from the feature
  table. Always do a standalone trial compile before committing to a feature selection.
- `ruvector-core::types::VectorId` is `String` everywhere, not an integer — `HnswIndex` stores
  ids as strings; `Collection`'s external contract stays `int` (stringified internally).
- `VectorDB::new`'s default `storage_path` is `"./ruvector.db"` (a real relative-path redb file)
  — always pass `"memory://..."` explicitly unless you want a stray file in cwd.
- **A per-call kwarg that silently does nothing is worse than not having one.**
  `SearchQuery.ef_search` exists on the struct but `VectorDB::search` never reads it — found via
  a suspicious benchmark number, not by reading the code first. Removed rather than kept.
- **`#[pyclass(unsendable)]`-class bug generalizes**: any vendored dependency that writes to
  stdout unconditionally (found: `patches/hnsw_rs`'s `println!` every 50k inserts) will corrupt
  a stdio-framed protocol server. Check for this in any new Rust dependency before trusting a
  stdio MCP transport with it.
- **Dynamic `__getattr__` dispatch (fixed this session) eliminates a whole bug class.**
  `__init__.py` used to check a hardcoded `_NATIVE_NAMES` set before dispatching to the compiled
  `_native` module — which is exactly how `ruvector.HnswIndex` went unreachable (added to
  `_native` in Rust, forgotten in the Python-side set). Now it's `hasattr(_native, name)`
  dynamically. Also pre-empted a guaranteed 3-way merge conflict when three parallel forks each
  needed to register a new class. Do this kind of registry-eliminating refactor *before*
  parallelizing work that would otherwise all touch the same hardcoded list.
- `cargo test -p ruvector-py` needs `extension-module` OFF in `pyo3`'s feature list (pyo3 FAQ: it
  disables linking against libpython, which a `cargo test` binary needs) — but maturin re-adds
  it for real wheel builds via `pyproject.toml`'s own `[tool.maturin] features=[...]`, so this is
  safe. **Side effect discovered via the first real CI run**: this means `cargo build`/`cargo
  test` on `ruvector-py` now needs libpython available on *any* bare runner that builds the
  whole workspace — excluded `ruvector-py` from 4 steps in `release.yml` rather than risk it.
- `cargo fmt --all -- --check` will fail CI if you forget to run `cargo fmt -p <crate>` before
  committing — happened once this session (cherry-picked code from a fork that hadn't run fmt).
  Always run `cargo fmt -p ruvector-py` + `cargo fmt --check -p ruvector-py` before every push.
- **`PyO3/maturin-action` picked the wrong manylinux container for a native ARM64 GitHub runner**
  (`ubuntu-24.04-arm`) — tried to cross-compile for aarch64 using an x86_64 cross-container
  needing `aarch64-linux-gnu-gcc`, which doesn't exist on that image. Fix: `container: 'off'` for
  that matrix leg (maturin-action's own documented escape hatch — "disable manylinux docker
  build and build on the host instead"), at the cost of the wheel losing its `manylinux_2_28`
  platform tag until someone can iterate against a real aarch64 runner.
- `mypy --strict` reusing a variable name across two branches of the same function (where only
  one branch is reachable after the other's early `return`) still produces a type conflict —
  rename the variable per branch instead of arguing with mypy about it.
- `sentence-transformers` installs cleanly into an isolated venv even with system-wide torch
  already present. `sklearn.datasets.fetch_20newsgroups` is a convenient real-text corpus for an
  embedding benchmark fixture when no other real dataset is on hand.
- `MCPServer.custom_route`'s routes do **not** get the `token_verifier=`/`auth=` protection —
  its own docstring says so explicitly. Don't assume "mount it on the same server" means "same
  auth for free" — found while researching the MCP-HTTP-auth design, before implementing it.
