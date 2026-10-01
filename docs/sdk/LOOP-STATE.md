# Python SDK/CLI/MCP — loop state (resume point)

Updated: 2026-10-01. Owner: claude-flow agent on branch `feat/python-sdk`.

## Where things live

- Worktree: `/home/ruvultra/projects/ruvector-python` (branch `feat/python-sdk`, tracks
  `origin/main`, created from `5a93328f2`).
- ADR: `docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`.
- Prior planning (scope baseline, still valid for binding strategy): `docs/sdk/01-06*.md`.
- Rust crate: `crates/ruvector-py/`.
- Cargo target dir (do NOT use default — root disk pressure, see `~/CLAUDE.local.md`):
  `/data/scratch/ruvector-python-target`.
- Python venv for dev/test: `/data/scratch/ruvector-py-venv` (`python3.12`, has `numpy pytest
  mypy maturin`).

## Done, verified

- [x] Worktree + branch created.
- [x] Salvaged `crates/ruvector-py` from stale `origin/feature/python-sdk-m1` (926 commits
      behind main; cherry-picked by path, not merge).
- [x] Added `crates/ruvector-py` to workspace `members` in root `Cargo.toml`.
- [x] Bumped pyo3 0.22→0.29.3, numpy 0.22→0.29.0. Fixed `allow_threads`→`detach`,
      `get_type_bound`→`get_type`.
- [x] `cargo build -p ruvector-py` clean, `cargo clippy --all-targets --no-deps -D warnings`
      clean, `maturin develop --release` installs, `pytest tests/` 7/7 pass.
- [x] Committed: `f1829ef44` "feat(ruvector-py): salvage Python SDK M1 (RaBitQ) onto current main".
- [x] ADR-352 written (supersedes scope, not strategy, of docs/sdk/01-06).
- [x] Live-verified the ChatGPT `ui://` widget `_meta` convention against
      `web-based-chatgpt-mcp-starter.ruv.chatgpt.site/api/mcp` (see ADR-352 body).
- [x] Confirmed no hnswlib/faiss/chromadb/qdrant-client pre-installed (benchmark comparator
      must be installed fresh, not assumed).

## In progress / next (M1.5 per ADR-352)

- [ ] `Collection` generic wrapper in `crates/ruvector-py/src/collection.rs` over
      `RabitqPlusIndex` + JSON metadata sidecar + dict filters.
- [ ] `python/ruvector_cli/` — click/rich CLI: create/insert/search/delete/export/import/
      benchmark/serve/info. Lazy imports, measure `python -X importtime`.
- [ ] `python/ruvector_mcp/` — MCP server (official `mcp` SDK), stdio first, tools wrapping
      `Collection`, one `vector_explore` widget tool with the live-verified `_meta` shape,
      resource at `ui://ruvector/explore.html`.
- [ ] pytest e2e for CLI + MCP (incl. a round-trip `tools/call` test and a `resources/read` on
      the widget URI).
- [ ] `cargo audit`, `pip-audit`, `npx @claude-flow/cli@latest security scan`.
- [ ] Benchmark vs one real installed comparator (hnswlib preferred — pure C++, no GPU/compile
      surprises). Record real numbers, methodology, host spec.
- [ ] `pyproject.toml` / CI wheel matrix review (cibuildwheel config exists from M1 salvage —
      verify it still matches `02-strategy.md`'s 5-platform matrix).
- [ ] Push branch, open draft PR to `main`. **Do not** merge, tag, or publish to PyPI — out of
      scope per the task boundary (user approval required).

## Deferred to later sessions (M2–M4, per ADR-352 — scoped, not started)

- ruLake bindings, generic HNSW-backed `Collection` (vs RabitqPlus-only today), RVF persistence,
  streamable-HTTP MCP transport.
- Embeddings (`Embedder.from_pretrained`), GNN/attention optional rerank hook.
- A2A client bindings.
- LangChain / LlamaIndex VectorStore adapters (`ruvector[langchain]`, `ruvector[llamaindex]`
  extras) — not started; natural fit once `Collection` is stable post-M2.

## Gotchas hit this session (don't rediscover)

- `feature/python-sdk-m1` branch is 926 commits behind main — do not try to rebase/merge it,
  cherry-pick the crate directory by path into a fresh worktree instead.
- `git ls-tree -r` (not plain `git ls-tree`) is needed to list files inside `docs/adr/` — a
  non-recursive call silently returns just the directory name.
- The MCP endpoint's `resources/list` requires auth ("Cognitum authentication required") even
  though `tools/list` doesn't — don't assume the whole server is open.
- `uv tool install maturin` was required; it was not present despite the salvaged M1 commit
  message claiming tests "will run as soon as maturin is present" — it wasn't, until now.
