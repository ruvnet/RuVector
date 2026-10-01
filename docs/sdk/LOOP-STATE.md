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
| CLI | ✅ done, polished | `ruvector` console script | rich tables, colored/ranked output, progress spinner, `--no-color`; importtime unchanged |
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
| LangChain VectorStore | ✅ done | `python/ruvector/integrations/langchain.py` | verified vs langchain-core 1.6.6; lazy module (not imported by plain `import ruvector`) |
| LlamaIndex VectorStore | ✅ done | `python/ruvector/integrations/llamaindex.py` | verified vs llama-index-core 0.14.25; found+fixed a real distance-vs-similarity bug in review |
| Salesforce Agentforce | 🔄 scoped, corrected | — | sent rUv a correction: no real "BYO retriever" extension point; Agentforce MCP is Beta/AE-gated. Proceeding with External Services + OpenAPI as the primary path pending any reply. |
| Rust unit tests | ✅ done | `src/hnsw.rs`, `src/error.rs` `#[cfg(test)]` modules | 26 tests, was 0; required a Cargo.toml feature change, fixed a CI risk that introduced in `release.yml` |

## Done, continued (3 forked agents landed + integration-tested, this checkpoint)

- [x] LangChain (`ruvector.integrations.langchain.RuVectorStore`, verified against
      `langchain-core==1.6.6`) and LlamaIndex (`ruvector.integrations.llamaindex.RuVectorStore`,
      verified against `llama-index-core==0.14.25`) adapters. `ruvector[langchain]`/
      `ruvector[llamaindex]` extras. Cherry-picked from forked agents, then **integration-tested
      for real** (not just trusted) — found and fixed a real correctness bug: the LlamaIndex
      adapter returned raw *distance* in `VectorStoreQueryResult.similarities` (should be
      *similarity*, higher=closer) — would have silently inverted ranking for any
      `SimilarityPostprocessor` cutoff. Fixed via a new `Collection.metric` property +
      `_distance_to_similarity()` (exact for cosine, documented-approximate otherwise).
- [x] 26 Rust unit tests for `ruvector-py` (was 0) — `parse_metric`, JSON round-trip converters,
      error mappers. Required dropping `extension-module` from `ruvector-py`'s pyo3 feature list
      so `cargo test -p ruvector-py` can link against libpython at all (maturin re-adds the
      feature for real wheel builds via `pyproject.toml`, unaffected). **Fixed a CI risk this
      introduced**: `.github/workflows/release.yml`'s `validate`/`build-crates` jobs run
      workspace-wide `cargo build`/`cargo test` on bare `ubuntu-22.04` with no Python setup —
      excluded `ruvector-py` from all 4 of those invocations (`--exclude ruvector-py`) rather
      than risk breaking shared CI on an unverifiable libpython-availability assumption.
      3 real bugs found in `hnsw.rs`'s JSON converters, documented as `known_limitations` tests
      (not fixed — out of scope, reported honestly): large-int (`>i64::MAX`) precision loss,
      NaN→null silent coercion, misleading error message for a lone UTF-16 surrogate string.
- [x] CLI rich-UI polish: colored/ranked `search` table (+ `--no-color`, verified NO_COLOR/TTY
      detection), `info` as a Rich panel over real `CollectionStats` fields, indeterminate
      spinner on `insert-batch`/`import` (chunking was considered and rejected — would change
      the rabitq rotation's fit quality on partial data). Importtime unchanged (~16-21ms).
- [x] Fixed a pre-existing bug found while integration-testing: `ruvector.HnswIndex` was
      unreachable via the public API (`__init__.py`'s lazy `__getattr__` dispatch set was never
      updated when `HnswIndex` was added to the compiled module — `Collection` worked fine since
      it imports `HnswIndex` directly, masking the gap).
- [x] All of the above cherry-picked onto `feat/python-sdk` as individual commits (not merged
      branches — each fork's worktree had drifted onto `feat/python-sdk`'s tip on its own, so
      cherry-picking the single commit was clean with zero conflicts). Full suite re-verified
      after every cherry-pick, not just at the end: 93/93 pytest, mypy --strict clean (14 files),
      clippy clean, 26/26 cargo test. Pushed.

## Next

- [ ] Graph raw-CRUD + GNN-forward + k-means + SONA-inference bindings.
- [ ] MCP HTTP bearer/OAuth auth.
- [ ] Salesforce Agentforce integration — corrected design sent to rUv (External Services +
      OpenAPI primary path; MCP registration documented as beta/AE-gated, not built), awaiting
      any reply before building.
- [ ] Python user guide `docs/python/README.md` (install+extras incl. langchain/llamaindex/
      salesforce, quick start, SDK, CLI, MCP+ChatGPT ui://, each integration, perf tips,
      benchmark table — now has real content for langchain/llamaindex to document).
- [ ] Separate commit `docs(readme): add Python install + user guide link` — minimal root
      README.md edit, flagged in the PR body as "land only after `ruvector` is live on PyPI".
- [ ] First CI run on the PR (untested non-Linux-x86_64 cross-compile paths).
- [ ] Slack: posted M1/M1.5 progress to #swarm thread (ts 1790885023.580889). Post the
      HNSW+integrations-landing milestone next, then final summary.

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
