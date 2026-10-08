# Verification evidence

Initial validation completed 2026-10-01 on Ubuntu/WSL2 in an isolated directory.
The integrated binding uses direct upstream dependencies in `crates/ruvector-py`.
The 41 native tests pass after integration with the pinned upstream HNSW patch;
Ruff, strict mypy, Rust formatting and all-target Clippy pass. Standard PEP 517
source packaging and an out-of-checkout rebuild pass using cached dependencies.
Linux CI completed on CPython 3.10.21 and 3.13.15 at
`80212abd8823f03fbc3178ab2825c657e4d199ec`:
[run 36911712198](https://github.com/ruvnet/RuVector/actions/runs/36911712198).
The macOS follow-up adds one persistence-path case (42 tests total) and expands
CI to Linux x86_64, macOS ARM64/Intel and Windows x86_64.

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
`crates/ruvector-core/src/index/hnsw.rs`, the native distance callback computes
`max(-dot, 0)`. Positive similarities tie at zero. The public interface excludes
this metric and the binding rejects restored dot-product configurations.
The final suite has no expected failures; the original failure is documented
instead of weakening the expected ranking.

The integration/packaging workflow repeats these checks in CI.
No generated logs or binaries are committed.

## Artifacts and practical limits

The wheel is `cp310-abi3-manylinux_2_34_x86_64`: CPython 3.10+ on Linux x86_64
with glibc >=2.34. Local Linux validation used CPython 3.13; remote Linux
CI also executed CPython 3.10, wheel tests, examples and sdist rebuild/tests.
The source archive includes the lockfile, required local Cargo dependencies and licenses, typed
Python files, binding, tests, examples, and design/provenance documentation.
No generated wheel, shared library, bytecode or cache is bundled in the source archive.

Source builds from scratch need internet access for registry/build dependencies
and Rust plus linker tooling. The offline archive rebuild reused downloaded Cargo
dependencies; it is not a claim of dependency-free/offline installation on a new
machine. The PyO3 ABI promise does not substitute for cross-version CI.

Additional local validation on macOS ARM64 is recorded below. Native Windows
and macOS Intel execution depend on the expanded CI; no local run is claimed.
Not run locally: Linux ARM, PyPy/free-threaded Python, performance/recall
benchmarks, high-volume deletion churn, native crash
recovery, cross-process concurrent writes or multi-handle coherence. The API
intentionally supplies one-handle serialized operations, not distributed locking.

Filters, approximate search, tombstones, batching limits, and unsupported upstream
modules are documented in README.md. No PyPI upload or service deployment is part of this contribution.
The change is prepared as a draft upstream PR.


## Native macOS follow-up (2026-10-01)

- Apple Silicon ARM64; macOS 27.0 (26A428), Darwin 27.0.0.
- Rust/Cargo 1.94.1, host `aarch64-apple-darwin`; Xcode linker ld-27037.1.
- CPython 3.10.21, 3.13.15 and 3.14.7 (standard GIL builds).
- maturin 1.15.0, pytest 9.1.1, Ruff 0.16.10, mypy 2.3.1.
- Wheel: `ruvector_python-0.1.0-cp310-abi3-macosx_11_0_arm64.whl`.
  The wheel's minimum deployment tag does not establish execution on macOS 11.

| Check | Result |
| --- | --- |
| Release wheel imported/installed on all three Python versions | Passed |
| Native pytest on CPython 3.10 / 3.13 / 3.14 | 42 passed each, no skips |
| Both examples on all three Python versions | Passed |
| Ruff lint/format, strict mypy, cargo fmt | Passed |
| cargo clippy --locked --all-targets -- -D warnings | Passed |
| Standard PEP 517 sdist | Passed |
| Offline Cargo rebuild outside checkout, fresh Cargo target directory | Passed |
| Source-rebuilt wheel on CPython 3.13, native tests/examples | 42 passed; both examples passed |
| Sdist inspection: patch/license sources present, generated binaries/caches absent | Passed |
| actionlint 1.7.12 on expanded workflow; git diff --check | Passed |

The additional persistence parameter uses a nested directory containing spaces
and Japanese characters, including the fresh-process reopen and file-lock release.
All three Python versions loaded the same abi3 wheel; these are runtime checks,
not separate version-specific wheel builds.

The initial release wheel compiled but could not import: macOS 27's loader rejected
`mis-aligned LINKEDIT string pool`. The raw Cargo dylib also failed, isolating this
from wheel installation. Rebuilding with release stripping disabled passed the
same tests. `Cargo.toml` now sets `strip = "none"` for release builds, avoiding
implicit debug-info stripping. The failure matches
[rust-lang/rust#157750](https://github.com/rust-lang/rust/issues/157750).
This does not modify any upstream native source or bypass loader/signature checks.
The source rebuild passed without an environment override for stripping.

The portable source rebuild used an empty Cargo target directory and downloaded
registry caches, with `CARGO_NET_OFFLINE=true`. Source creation itself fetched
additional platform metadata dependencies. This is not an offline-first install
claim. No wheel, cache, build log or generated extension is committed.

## Upstream CI failures outside the Python contribution

At the prior head, the dedicated Python workflow passed. Existing workspace
strict Clippy failed with five `double_must_use` diagnostics from `async_trait`
expansions in `rvagent-core` graph/models/subagent traits under Rust 1.99:
[failed job](https://github.com/ruvnet/RuVector/actions/runs/36911712258/job/110535691904).
Core/rest tests failed to link the ONNX Runtime dependency of
`ruvector-typesafe-train`: unresolved `__isoc23_strtol` and C++ `_M_replace_cold`
symbols in test binaries:
[failed job](https://github.com/ruvnet/RuVector/actions/runs/36911712214/job/110536213635).
Both logs were inspected again on the Mac. These sources, manifests, existing
workflows and root lockfile are unchanged from baseline `5a93328f`; only the root
manifest's exclusion of the standalone Python crate differs. No baseline rerun
or broad workspace repair is claimed. Current PR check status is reported in the
PR description after the expanded workflow finishes.
