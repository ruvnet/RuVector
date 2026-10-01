# Verification evidence

Initial validation completed 2026-10-01 on Ubuntu/WSL2 in an isolated directory.
The integrated binding uses direct upstream dependencies in `crates/ruvector-py`.
The 41 native tests pass after integration with the pinned upstream HNSW patch;
Ruff, strict mypy, Rust formatting and all-target Clippy pass. Standard PEP 517
source packaging and an out-of-checkout rebuild pass using cached dependencies.
Dedicated CI adds Python 3.10 and 3.13 checks; those remote results remain pending
until the workflow runs.

## Tested stack

- Ubuntu/WSL2 Linux 6.18.33.2, x86_64, glibc 2.39.
- CPython 3.13.12; Rust/Cargo 1.95.0.
- PyO3 0.23.5, abi3-py310; maturin 1.15.0.
- pytest 9.1.1, Ruff 0.16.9, mypy 2.3.1.
- Real RuVector core 2.3.1, commit
  `5a93328f2fceb0307c25929ed38cd7a0911fdf00`.
- Core feature selection: storage, hnsw, parallel; default features disabled.
  HNSW's upstream runtime CPU kernels remain native; no CPU-specific compiler
  flags are used. No embedding provider, paid service or credentials were used.

## Results

| Check | Result |
| --- | --- |
| Native extension release build/install | Passed |
| pytest against editable native package | 41 passed, no skips |
| Wheel installed in fresh virtual environment, pytest | 41 passed, no skips |
| Both executable examples | Passed |
| Ruff lint and formatting | Passed |
| mypy strict | Passed, 2 source files |
| cargo fmt --check | Passed |
| cargo clippy --locked --lib -- -D warnings | Passed |
| Wheel and source distribution builds | Passed |
| Source archive rebuilt using cached Cargo dependencies, offline Cargo | Passed |

All tests exercise the compiled Rust backend. There are **no mocked-search tests**.
Coverage includes insertion/get/delete/count/keys, missing IDs, UUID generation,
three metric rankings, exact JSON and AND filters, post-top-k filter shortage,
empty batches, 96-entry native batch insertion (parallel path), batch query
order, duplicate rejection, validation before mutation at the native boundary,
nonfinite/oversized vectors, invalid metadata/options, Unicode and zero vectors,
deterministic close, native storage errors, ID reuse after deletion, and threaded
operations on one handle.

The persistence test closes the original file, starts a fresh Python process,
reopens with deliberately different constructor configuration, verifies the
persisted dimension/metric/HNSW options and metadata, checks deletion visibility,
performs filtered search and insertion, then reopens again in the original process.
This proves file-backed persistence and index reconstruction across process lifetime.

Initial native tests revealed a dot-product ranking defect: in
`vendor/ruvector-core/src/index/hnsw.rs`, the native distance callback computes
`max(-dot, 0)`. Positive similarities tie at zero. The public interface excludes
this metric and the binding rejects restored dot-product configurations.
The final suite has no expected failures; the original failure is documented
instead of weakening the expected ranking.

The integration/packaging workflow repeats these checks in CI.
No generated logs or binaries are committed.

## Artifacts and practical limits

The wheel is `cp310-abi3-manylinux_2_34_x86_64`: CPython 3.10+ on Linux x86_64
with glibc >=2.34. Only CPython 3.13 on the stack above was executed.
The source archive includes the lockfile, required local Cargo dependencies and licenses, typed
Python files, binding, tests, examples, and design/provenance documentation.
No generated wheel, shared library, bytecode or cache is bundled in the source archive.

Source builds from scratch need internet access for registry/build dependencies
and Rust plus linker tooling. The offline archive rebuild reused downloaded Cargo
dependencies; it is not a claim of dependency-free/offline installation on a new
machine. The PyO3 ABI promise does not substitute for cross-version CI.

Not run: macOS, Windows-native, ARM, other CPython versions, PyPy/free-threaded
Python, performance/recall benchmarks, high-volume deletion churn, native crash
recovery, cross-process concurrent writes or multi-handle coherence. The API
intentionally supplies one-handle serialized operations, not distributed locking.

Filters, approximate search, tombstones, batching limits, and unsupported upstream
modules are documented in README.md. No PyPI upload or service deployment is part of this contribution.
The change is prepared as a draft upstream PR.
