# Testing Guide

How to run unit tests, integration tests, and the Criterion benchmark suite,
and what each bench measures.

## Test Layout

```
ruvLLM/
├── src/                     # unit tests live next to the code they test
│   └── **/*.rs             # `#[cfg(test)] mod tests { ... }`
├── tests/                   # integration tests
│   ├── integration.rs
│   └── sona_integration.rs
└── benches/                 # Criterion benches
    ├── pipeline.rs
    ├── router.rs
    ├── memory.rs
    ├── attention.rs
    └── sona_bench.rs
```

The convention is documented in [Code Standards](code-standards.md): unit
tests are colocated and small, integration tests are async and exercise the
full orchestrator, benches are reproducible and tracked as a contract.

## Unit Tests

Unit tests live inside the modules they cover, gated by `#[cfg(test)]`. They
exercise pure functions in isolation — distance kernels, tokenizer wrappers,
HNSW navigation, FastGRNN forward, LoRA forward, etc.

Run all unit tests:

```sh
cargo test --lib
```

Run a specific module's tests:

```sh
cargo test --lib router::
cargo test --lib sona::lora::
```

Filter by test name:

```sh
cargo test --lib forward_dim_128
```

Use `-- --nocapture` to see `println!` output:

```sh
cargo test --lib -- --nocapture
```

## Integration Tests

Two integration test files in `tests/`:

| File | What it covers |
|---|---|
| `tests/integration.rs` | Async pipeline end-to-end: query, context, confidence-threshold branch, latency budget. |
| `tests/sona_integration.rs` | The SONA learning flow: trajectory → ReasoningBank → LoRA adapter, concurrent safety, instant-loop latency under load. |

Both use `#[tokio::test]` and the multi-thread runtime (matching the
production `tokio` configuration). Run all integration tests:

```sh
cargo test --test integration
cargo test --test sona_integration
```

Run all tests including doctests:

```sh
cargo test
```

### Feature-Gated Tests

Some tests need optional features:

```sh
# With real inference (Candle backend)
cargo test --features real-inference

# With the HTTP server stack (some tests build the Axum router)
cargo test --features server

# Everything
cargo test --features full
```

If you're adding a test that depends on a feature, gate it with
`#[cfg(feature = "...")]` at the top of the module and document the
requirement in the test's doc comment.

## Benchmarks

The `benches/` directory uses Criterion 0.5 with `async_tokio` and the
HTML report generator. Every bench is a contract: regressions on the
documented numbers are release-blocking. See
[Project Overview](project-overview-pdr.md) for the headline targets.

### Run All Benches

```sh
cargo bench
```

Each bench takes minutes (Criterion needs many samples for tight
confidence intervals). Output goes to stdout and to
`target/criterion/`.

### Run a Single Bench File

```sh
cargo bench --bench pipeline
cargo bench --bench router
cargo bench --bench memory
cargo bench --bench attention
cargo bench --bench sona_bench
```

### Filter Within a Bench

Criterion accepts a regex on the bench-id:

```sh
cargo bench --bench router -- "forward_dim_128"
cargo bench --bench memory -- "search_768d_batch_500"
```

### What Each Bench Measures

| Bench | Scope | Key dimensions |
|---|---|---|
| `pipeline.rs` | End-to-end query latency through the full orchestrator | Input length |
| `router.rs` | FastGRNN forward and training | Hidden dim 64, 128, 256, 512 |
| `memory.rs` | HNSW insert and search | 768-D vectors, batch 10 / 50 / 100 / 500 |
| `attention.rs` | Multi-head graph attention on variable-size subgraphs | 768-D, varying node counts |
| `sona_bench.rs` | SONA hot path: MicroLoRA, trajectory append, ReasoningBank, InstantLoop, EWC++ | Targets MicroLoRA <100 µs, trajectory <1 µs/step, InstantLoop <1 ms |

Together they exercise every hot-path module from
[System Architecture](system-architecture.md).

### HTML Reports

After `cargo bench`, open the consolidated report:

```sh
open target/criterion/report/index.html      # macOS
xdg-open target/criterion/report/index.html  # Linux
```

Each individual benchmark also has its own `target/criterion/<bench-id>/report/index.html`
with violin plots, regression-comparison vs. the prior run, and raw sample
data. Criterion automatically diffs against the last run, which makes it
easy to spot performance changes as you iterate.

### Comparing Against a Baseline

```sh
# Save the current result as 'before'
cargo bench -- --save-baseline before

# Make changes...

# Compare against the saved baseline
cargo bench -- --baseline before
```

Use this when refactoring a hot-path module — you want a clean before/after
comparison, not just a noisy run-over-run delta.

## Quick Bench: `ruvllm-bench`

The `ruvllm-bench` binary is a thin wrapper that runs a fast latency
probe. Useful as a CI smoke test — it finishes in seconds and emits a
single-line summary that is easy to assert on:

```sh
cargo run --release --bin ruvllm-bench
```

For the full-fidelity suite use `ruvllm-benchmark-suite`, which wraps the
Criterion benches into one reproducible invocation.

```sh
cargo run --release --bin ruvllm-benchmark-suite
```

## SIMD Detection Smoke Test

`ruvllm-simd-demo` prints which SIMD path was selected at runtime
(AVX2 / SSE4.1 / NEON / scalar). Run it on every new deployment target
to confirm the right kernel is active:

```sh
cargo run --release --bin ruvllm-simd-demo
```

## CI Recipe

A minimal CI matrix:

```yaml
- name: Unit + integration (default features)
  run: cargo test --workspace

- name: Tests with full features
  run: cargo test --workspace --features full

- name: Build server release
  run: cargo build --release --bin ruvllm-server --features "server,real-inference,parallel,metrics,storage"

- name: Smoke bench
  run: cargo run --release --bin ruvllm-bench

- name: Criterion suite (nightly only)
  run: cargo bench --bench pipeline --bench router --bench memory --bench attention --bench sona_bench
```

The Criterion suite belongs in a nightly job, not on every PR — it takes
long enough that gating PRs on it slows iteration without enough signal.
The smoke bench (`ruvllm-bench`) is fast enough for per-PR.

## Writing a New Test

1. **Unit test?** Add to `#[cfg(test)] mod tests` in the same `.rs` file.
2. **Integration test?** Add a function to one of the existing files in
   `tests/` if it fits a current theme; otherwise create a new `tests/foo.rs`.
3. **Async?** Use `#[tokio::test]` and the multi-thread flavor matching
   production: `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]`.
4. **Touches a hot path?** Add or update a Criterion bench too. See
   [Code Standards](code-standards.md): "Latency claims must be benched."

## Debugging Test Failures

- **Increase verbosity:** `cargo test -- --nocapture --test-threads=1`.
- **Filter to one test:** `cargo test path::to::test_name`.
- **Race conditions in async tests:** add a `tokio::time::timeout` so a
  hang shows as a failure rather than a CI timeout.
- **Flakiness on benches:** run with `--baseline` to compare; Criterion's
  noise model surfaces real regressions but tolerates jitter.

## See also

- [Code Standards](code-standards.md)
- [System Architecture](system-architecture.md)
- [Project Overview & PDR](project-overview-pdr.md)
- [Codebase Summary](codebase-summary.md)
