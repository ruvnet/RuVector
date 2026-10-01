# Python SDK/CLI/MCP — loop state (resume point)

Updated: 2026-10-01 (end of session). Owner: claude-flow agent on branch `feat/python-sdk`.

## Where things live

- Worktree: `/home/ruvultra/projects/ruvector-python` (branch `feat/python-sdk`, tracks
  `origin/main`, created from `5a93328f2`).
- ADR: `docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md` — read this first, it has the real
  benchmark table, security findings, and publishing verification with commands/numbers.
- Prior planning (scope baseline, binding strategy still valid): `docs/sdk/01-06*.md`.
- Rust crate: `crates/ruvector-py/` (Cargo.toml, src/{lib,rabitq,error}.rs).
- Python package: `crates/ruvector-py/python/ruvector/` (`__init__.py`, `collection.py`,
  `cli.py`, `mcp_server.py`, `.pyi` stubs).
- Tests: `crates/ruvector-py/tests/` (`test_smoke.py`, `test_collection.py`, `test_cli.py`,
  `test_mcp_server.py`) — 38/38 passing.
- CI: `.github/workflows/python-wheels.yml` (not yet run — first PR push will trigger it).
- Cargo target dir (do NOT use default — root disk pressure, see `~/CLAUDE.local.md`):
  `/data/scratch/ruvector-python-target`.
- Python venvs: `/data/scratch/ruvector-py-venv` (dev, editable install, has numpy/pytest/mypy/
  click/rich/mcp/hnswlib/pyyaml/pip-audit); `/data/scratch/ruvector-freshwheel-venv` (clean
  install of the built wheel, used once to verify a fresh-install test pass).
- Benchmark script: `/home/ruvultra/.cache/claude-code/tmp/claude-1000/
  -home-ruvultra-projects-ruvector/f20ef550-d4f6-4d45-bbef-133c084f2dd7/scratchpad/
  bench_compare.py` (scratchpad, not committed — session-specific path, rerun from scratch if
  needed in a new session).

## Done, verified (commands + results in ADR-352 and commit messages — not re-summarized here)

- [x] Worktree + branch; salvaged + modernized M1 RaBitQ crate (pyo3 0.22→0.29.3).
- [x] Fixed `unsendable` pyclass bug (would panic any multi-threaded host, e.g. the MCP server).
- [x] Added `RabitqIndex.add`/`add_batch`/`export_items` (true incremental insert + rebuild path).
- [x] `Collection` (ids, metadata, filters, soft delete, vacuum, save/load) — pure Python over
      the M1 index, per CLAUDE.md's "core logic in Rust" rule.
- [x] `ruvector` CLI (click): create/insert-batch/search/delete/export/import/info/benchmark/
      serve. Fast-startup fix (PEP 562 lazy `__getattr__` in `__init__.py`): 72.8ms → 16.1ms.
- [x] MCP server (official `mcp` SDK v2, `MCPServer` not `FastMCP`): 7 tools + 1 widget tool
      (`vector_explore`) + 1 `ui://` resource. Verified live over real stdio AND real
      streamable-HTTP (curl against a running uvicorn process), not just in-process tests.
- [x] Fixed a real existence-check bug (checked the `.rbpx` path instead of the `.meta.json`
      sidecar — broke on empty collections) in 3 places (collection.py, cli.py, mcp_server.py).
- [x] `mypy --strict` clean on the whole package + all 4 test files (fixed 2 real narrowing
      bugs along the way, not just annotation noise).
- [x] `cargo audit`, `pip-audit` (note: the standalone binary audits the *system* Python, not
      an active venv — use `python3 -m pip_audit` instead), `npx @claude-flow/cli@latest
      security scan` — all clean; results and exact commands in ADR-352 "Security".
- [x] Real benchmark vs `hnswlib` (installed fresh, needed `libomp-dev`) — table + honest
      "random-Gaussian is adversarial, don't over-read this" caveat in ADR-352.
- [x] `maturin build --release` (not just `develop`) + fresh-venv install + full test pass.
- [x] `.github/workflows/python-wheels.yml` (5-platform matrix, maturin-action, trusted
      publishing gated on tag/dispatch — not run yet).

## Next (not done this session — pick up here)

- [ ] Push branch, open **draft** PR to `main`. (Boundary: do not merge, tag, or publish.)
- [ ] First CI run on the PR — this is the first real test of the aarch64/macOS/Windows
      cross-compile paths; nothing here has been verified on non-Linux-x86_64.
- [ ] MCP HTTP transport has no auth yet (ADR-352 flags this explicitly as a real gap, not
      silently deferred) — bearer-token auth is unimplemented.
- [ ] `--http` MCP transport error handling/reconnect behavior under load: untested beyond one
      manual curl request.

## Deferred to later sessions (M2–M4, per ADR-352 — scoped, not started)

- ruLake bindings, generic HNSW-backed `Collection` (vs RabitqPlus-only today), RVF persistence,
  streamable-HTTP MCP transport hardening.
- Embeddings (`Embedder.from_pretrained`), GNN/attention optional rerank hook.
- A2A client bindings.
- LangChain / LlamaIndex VectorStore adapters (`ruvector[langchain]`, `ruvector[llamaindex]`
  extras) — not started at all this session; natural fit once `Collection` is stable post-M2.
- Re-benchmark on a real embedding dataset (SIFT1M or MiniLM-embedded text) once M3 ships — the
  random-Gaussian numbers in ADR-352 are real but adversarial; see that section's "Follow-up".

## Gotchas hit this session (don't rediscover)

- `feature/python-sdk-m1` branch is 926 commits behind main — cherry-pick the crate directory
  by path into a fresh worktree, don't try to rebase/merge it.
- `git ls-tree -r` (not plain `git ls-tree`) to list files inside a directory recursively.
- pyo3 0.22→0.29: `Python::allow_threads` → `Python::detach`, `Python::get_type_bound` →
  `Python::get_type`. `pyo3-async-runtimes` (not `pyo3-asyncio`, confirmed dead) is the current
  crate name for M2's async work.
- **`#[pyclass(unsendable)]` panics the moment the object crosses an OS thread** — not a
  theoretical risk, hit it for real the first time an MCP tool handler ran on a worker thread
  different from the one that built the index. If the wrapped Rust type is actually `Send +
  Sync` (check its trait bounds), just remove `unsendable` — pyo3 auto-derives correctly.
- mcp>=2.0 renamed `FastMCP` → `MCPServer` (`mcp.server.mcpserver`), with a helpful
  `ModuleNotFoundError` pointing at the migration guide — read it before guessing at the new API.
- `ToolAnnotations`/tool `meta=` fields are snake_case in pydantic (`read_only_hint`) but
  camelCase on the wire (`readOnlyHint`) via pydantic aliases — don't guess the Python-side name
  from the JSON-RPC shape.
- `CallToolResult.is_error`, not `.isError` (pydantic snake_case again).
- `~/.local/bin/pip-audit` audits the **system** Python even inside an activated venv if it's a
  pipx-style standalone install — use `python3 -m pip_audit` after activating instead, or you'll
  get Ubuntu system-package noise (`cloud-init`, `ufw`, `bcc`, ...) instead of your deps.
- `hnswlib` has no prebuilt wheel on this platform; building from source needs
  `sudo apt-get install -y libomp-dev` first (missing `omp` shared lib otherwise).
- An empty `Collection.create()`'s index is `None` until the first `insert` — `Collection.save()`
  therefore only writes the `.meta.json` sidecar, not the `.rbpx` file. Any "does this collection
  exist" check must test `Collection.meta_path(path)`, never the raw index path.
