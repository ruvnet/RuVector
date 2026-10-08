# ruvector Python SDK — Planning Index

## VectorDB implementation

An experimental native VectorDB interface now lives in
[`crates/ruvector-py`](../../crates/ruvector-py/README.md). It wraps the existing
core's HNSW, storage and metadata APIs, with real-native tests and dedicated
Linux/macOS/Windows CI. This is a narrower scope than the RaBitQ/ruLake roadmap below; those
milestones remain future work. The survey statements below describe the original
2026-04-25 planning snapshot.


This directory contains the design review for a first-party Python SDK over the
ruvector workspace. It is a planning artifact, not source code. No `pyproject.toml`,
`*-py` crate, or PyO3 dependency exists in the workspace today (verified
2026-04-25 by searching for `pyo3`/`maturin` in every `Cargo.toml` and for
`pyproject.toml`/`*.pyi` outside `target/` and `node_modules/`). Everything
below is greenfield.

## Documents

- **[01-survey.md](./01-survey.md)** — What ruvector ships today: which crates
  are realistic SDK targets vs internal-only, what FFI surfaces already exist
  (NAPI-RS templates, wasm-bindgen modules, raw cbindgen consumers), the
  shape of the JS/TS distribution, and which `examples/` are good Python
  notebook material.
- **[02-strategy.md](./02-strategy.md)** — The binding-approach decision.
  Reviews PyO3 + maturin, CFFI, ctypes-over-cbindgen, wasmtime-py over the
  WASM crates, and gRPC-server-with-Python-client. Picks PyO3 + maturin and
  defends the choice. Covers the asyncio story, the GIL story, the wheel
  matrix, and the type-stub plan.
- **[03-api-surface.md](./03-api-surface.md)** — A concrete sketch of the
  Python API the user types: `ruvector.RabitqIndex.build(...)`,
  `ruvector.RuLake.builder()...build()`, `ruvector.A2aClient(...)`. Locks
  in the error hierarchy, sync-vs-async signatures per call, NumPy interop,
  and the Pythonic conveniences (`len(idx)`, `idx[i]`, context managers).
- **[04-milestones.md](./04-milestones.md)** — Four buildable milestones
  with explicit scope, file lists, LoC budgets, and acceptance tests in
  the same shape as ADR-159's milestone plan. M1 is RaBitQ-only. M2 adds
  ruLake. M3 adds embeddings. M4 wraps `rvagent-a2a`.
- **[05-risks-and-tradeoffs.md](./05-risks-and-tradeoffs.md)** — The honest
  reservations: tokio runtime in a PyO3 extension, GIL for batched ops,
  wheel size, NEON/AVX-512 build-time-vs-runtime detection, abi3 vs
  version-specific wheels, the `ruvector` PyPI squat question, and where
  this code lives in the repo (a new `crates/ruvector-py/` member, not a
  separate repo).
- **[06-decision-record.md](./06-decision-record.md)** — One-page summary
  with the chosen strategy, the 4-milestone roadmap, three acceptance
  gates that gate the whole effort, and the open questions for stakeholders
  to answer before M1 starts.

## How to read this

Read `06` first if you want the call-to-action. Read `02` first if you want
to argue with the binding strategy. Read `01` first if you've never opened
this codebase before.
