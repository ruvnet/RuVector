# Python SDK/CLI/MCP — loop state (resume point)

Updated: 2026-10-01 (mid-session, scope expanded by rUv's follow-up). Owner: claude-flow agent
on branch `feat/python-sdk`. PR: https://github.com/ruvnet/RuVector/pull/1117 (draft).

## Where things live

- Worktree: `/home/ruvultra/projects/ruvector-python` (branch `feat/python-sdk`, tracks
  `origin/main`).
- ADR: `docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md` — read this first. Has both benchmark
  sections (M1.5 RabitqPlus/random-Gaussian, M2 HnswIndex/real-embeddings), security findings,
  publishing verification, all with real commands/numbers.
- Rust crate: `crates/ruvector-py/src/{lib,rabitq,hnsw,error}.rs`.
- Python package: `crates/ruvector-py/python/ruvector/` (`__init__.py`, `collection.py`,
  `cli.py`, `mcp_server.py`, `.pyi` stubs). `collection.py` has two backends now (`hnsw` default,
  `rabitq`) — see its module docstring.
- Tests: `crates/ruvector-py/tests/` — 63/63 passing (`test_smoke`, `test_collection` [most
  tests parametrized over both backends], `test_cli`, `test_mcp_server`).
- Vendored patch: `patches/hnsw_rs/` (workspace `[patch.crates-io]`, also used by
  `ruvector-core`/`ruvector-graph`) — fixed a stdout-corrupting `println!` in it this session.
- CI: `.github/workflows/python-wheels.yml` — pushed, not yet run (PR triggers it).
- Cargo target dir (NOT the default — root disk pressure, see `~/CLAUDE.local.md`):
  `/data/scratch/ruvector-python-target`.
- Python venvs: `/data/scratch/ruvector-py-venv` (dev, editable install — has numpy/pytest/mypy/
  click/rich/mcp/hnswlib/pyyaml/pip-audit/sentence-transformers/scikit-learn/torch);
  `/data/scratch/ruvector-freshwheel-venv` (clean wheel-install check, rerun after each Rust
  change if you want to re-verify "fresh install passes").
- Benchmark scripts (scratchpad, not committed — rerun from scratch in a new session):
  `.../scratchpad/bench_compare.py` (M1.5 RabitqPlus vs hnswlib, random-Gaussian),
  `.../scratchpad/bench_hnsw_real.py` (M2 HnswIndex vs hnswlib, real MiniLM/20newsgroups
  embeddings — results cached at `/tmp/bench_hnsw_real_results.json`, host-local, not committed).

## Capability coverage (vs the npm `ruvector` package's keyword list — the "all major
capabilities" bar rUv set). Update this table at every milestone, not just at the end.

| Capability | Status | Where | Notes |
|---|---|---|---|
| RaBitQ quantized index | ✅ done | `RabitqIndex`, `Collection(backend="rabitq")` | M1/M1.5 |
| HNSW index | ✅ done | `HnswIndex`, `Collection(backend="hnsw", default)` | M2, this session |
| Metadata filtering | ✅ done (partial) | `Collection.search(filter=dict)` | dict-filter in Rust for hnsw backend; callable predicate always Python-side (structural — can't ship a Python fn into Rust); rabitq backend always Python-side |
| Collections (named, CRUD) | ✅ done | `Collection` class | both backends |
| Persistence | ✅ done (own format, not RVF) | `Collection.save/load` | `.rbpx`+sidecar (rabitq) / `.npy`+sidecar (hnsw) — see "RVF" row for why not RVF |
| CLI | ✅ done, polish in progress | `ruvector` console script | forked for rich-table/progress-bar polish — see below |
| MCP server (stdio+HTTP) | ✅ done, auth gap | `ruvector serve` | no bearer/OAuth on `--http` yet — queued |
| ChatGPT `ui://` widget | ✅ done | `vector_explore` tool | live-verified `_meta` shape |
| Graph (raw CRUD) | ⬜ not started | — | `ruvector-graph::GraphDB` confirmed Send+Sync, pub API ready to bind (ADR-352 inventory) |
| Graph (Cypher-lite) | ⬜ not started | — | needs porting `cypher_exec.rs` (759 lines, non-napi-specific, cdylib-only crate today) out of `ruvector-graph-node` |
| GNN forward-pass rerank | ⬜ not started | — | `ruvector-gnn::RuvectorLayer::forward` works on a random-init layer, no training needed first |
| k-means clustering | ⬜ not started | — | real algorithm is `ruvector-cluster-rag::cluster::kmeans`, NOT `ruvector-cluster` (that's distributed sharding infra, premise mismatch — see its row) |
| SONA (inference-only) | ⬜ not started | — | `SonaEngine::apply_micro_lora` usable immediately (untrained ≈ identity); training loop (`begin_trajectory`/...) needs a real reward signal, lower priority |
| RVF persistence | ❌ not practical this session | — | ~30-file subsystem (COW, witness/crypto, eBPF, federation); neither `ruvector-core` nor `ruvector-collections` depend on it today either — documented honestly in ADR-352, not silently skipped |
| Embeddings (M3) | ❌ deferred | — | needs ONNX/`ort` + model download, a separate milestone by design (docs/sdk/04-milestones.md M3), unchanged |
| `ruvector-cluster` (distributed) | ❌ premise mismatch | — | it's node-coordination/sharding infra, not ML clustering — binding it needs a running multi-node cluster, not a Python process |
| Quantization (Turbo4) | ⬜ unexplored | — | exists in `ruvector-core`'s `QuantizationConfig`, not evaluated this session |
| LangChain VectorStore | 🔄 forked, in progress | `python/ruvector/integrations/langchain.py` | against the stable `Collection` API |
| LlamaIndex VectorStore | 🔄 forked, in progress | `python/ruvector/integrations/llamaindex.py` | same |
| Salesforce Agentforce | 🔄 scoped, corrected | — | sent rUv a correction: no real "BYO retriever" extension point; Agentforce MCP is Beta/AE-gated. Proceeding with External Services + OpenAPI as the primary path pending any reply. |
| Rust unit tests | 🔄 forked, in progress | `src/*.rs` test modules | `cargo test -p ruvector-py` ran 0 before this |

## Next (in flight this message / immediately after)

- [ ] 3 forked agents in flight: LangChain+LlamaIndex adapters, Rust unit tests, CLI rich-UI
      polish. Check their results before continuing other Rust/CLI work on the same files.
- [ ] Graph raw-CRUD + GNN-forward + k-means + SONA-inference bindings (doing myself while forks run).
- [ ] MCP HTTP bearer/OAuth auth (independent, pick up after the forks land).
- [ ] Salesforce Agentforce integration once/if rUv confirms the corrected design.
- [ ] Python user guide `docs/python/README.md` (per rUv — install+extras, quick start, SDK,
      CLI, MCP+ChatGPT ui://, each integration incl. Agentforce, perf tips, benchmark table).
- [ ] Separate commit `docs(readme): add Python install + user guide link` — minimal root
      README.md edit, flagged in the PR body as "land only after `ruvector` is live on PyPI".
- [ ] First CI run on the PR (untested non-Linux-x86_64 cross-compile paths).
- [ ] Slack: posted M1/M1.5 progress to #swarm thread (ts 1790885023.580889). Post again at
      HNSW-landing milestone (done — not yet posted, do next), integrations-landing, and final.

## Gotchas hit this session (don't rediscover) — M1/M1.5 ones omitted here, see git log; M2 additions below

- `ruvector-core`'s *default* Cargo features pull in `api-embeddings` (reqwest+rustls) for
  nothing this crate uses — depend on it with `default-features = false, features = ["storage",
  "hnsw", "simd", "parallel"]`.
- `ruvector-core::types::VectorId` is `String` everywhere, not an integer — `HnswIndex` stores
  ids as strings; `Collection`'s external contract stays `int` (stringified internally) to keep
  the Python API identical across backends.
- `VectorDB::new`'s default `storage_path` is `"./ruvector.db"` (a real relative-path redb file)
  — always pass `"memory://..."` explicitly unless you want a stray file in cwd.
- **A per-call kwarg that silently does nothing is worse than not having one.**
  `SearchQuery.ef_search` exists on the struct but `VectorDB::search` never reads it — found via
  a suspicious benchmark number, not by reading the code first. Removed rather than kept.
- **`#[pyclass(unsendable)]`-class bug from M1 generalizes**: any vendored dependency that writes
  to stdout unconditionally (found: `patches/hnsw_rs`'s `println!` every 50k inserts) will
  corrupt a stdio-framed protocol server. Check for this in any new Rust dependency before
  trusting a stdio MCP transport with it.
- `mypy --strict` reusing a variable name (`items`) across two branches of the same function,
  where only one branch is reachable after the other's early `return`, still produces a type
  conflict — mypy's control-flow narrowing doesn't extend to "this name's type from branch A
  can't appear in branch B because A always returns." Rename the variable per branch instead of
  arguing with mypy about it.
- `sentence-transformers` installs cleanly into an isolated venv even with system-wide torch
  already present (it pulls its own compatible torch) — no special `--system-site-packages` dance
  needed. `sklearn.datasets.fetch_20newsgroups` is a convenient real-text corpus for an
  embedding benchmark fixture when no other real dataset is on hand.
