# Code Standards

Conventions used throughout the `ruvllm` crate, the `esp32/` sub-crate, and
the `esp32-flash/` firmware.

## Rust Edition and Toolchain

- The crate is on a current stable edition. New code uses 2021-edition idioms
  (let-else, GATs where they help, `Result` on every fallible path).
- The hot path forbids `unwrap()` and `expect()` outside of tests, benches, and
  `main.rs` initialization.
- `async fn` in traits is acceptable now that the crate targets stable
  toolchains that support it natively.

## Error Handling — `thiserror` Pattern

`src/error.rs` defines a single `thiserror`-derived enum that is the canonical
error type for the library. Every public fallible function returns
`Result<T, ruvllm::Error>` (or a domain-specific variant that converts via
`#[from]`).

Rules:

1. **Library code never panics.** Anything that could fail at runtime returns
   a typed error.
2. **`#[from]` for layer crossings.** When wrapping an underlying error
   (`io::Error`, `serde_json::Error`, Candle errors, HNSW errors), add a
   variant with `#[from]` rather than calling `.map_err`.
3. **Errors carry context, not strings.** Variants name the failed operation,
   e.g. `MemoryWriteFailed { path }` rather than a generic `IoError`.
4. **`anyhow` is allowed only in binaries.** The five `ruvllm-*` binaries may
   use `anyhow::Result` for top-level error reporting; library code never
   does.

## Feature Flag Discipline

Cargo features are a contract, not a toggle. Rules:

- **Default features stay minimal.** Only `storage` and `metrics` are on by
  default; everything else is explicit. See
  [Codebase Summary](codebase-summary.md) for the full table.
- **`#[cfg(feature = "x")]` at the smallest viable scope.** Prefer gating a
  function or `mod` rather than gating a whole file.
- **No silent fallbacks.** If `real-inference` is off, `inference_real.rs` is
  not compiled; it does not silently fall back to mock — the user must opt in.
- **No feature-flag combinations that produce a non-compiling crate.** Every
  feature must compile in isolation (`cargo build --no-default-features
  --features X`) and in combination with the documented sets (`server`,
  `real-inference`, `full`).
- **`full` is a real test target.** CI builds with `full` to catch
  flag-combination bugs.

## `no_std` for ESP32

The `esp32/` library sub-crate is `no_std` by default. The `esp32-std` feature
re-enables the standard library when running on a host (e.g. for unit tests
on a workstation).

`no_std` rules in the ESP32 codebase:

- Use `heapless::Vec`, `heapless::String`, `heapless::FnvIndexMap` instead of
  `alloc::vec::Vec` / `String` / `HashMap`.
- All math goes through `libm` (no `f32::sin` etc., which require `std`).
- Fixed-point arithmetic via the `fixed` crate where determinism matters more
  than dynamic range.
- Wire formats use `postcard` rather than `serde_json` to avoid heap.
- No `println!` — diagnostic output goes through whatever logger the firmware
  binds (defmt or similar in `esp32-flash/`).

The host-side `ruvllm` crate is **always** `std`. There is no expectation of
sharing a `no_std` boundary with the ESP32 sub-crate; they share concepts and
quantization formats, not code.

## Async Patterns — Tokio

The runtime is `tokio` 1.41 configured for `multi-thread`, `sync`, and
`macros`. Async conventions:

- **Hot-path tasks use `tokio::spawn`.** Background loops (the hourly pattern
  extraction in `sona/loops/background.rs` and the weekly coordinator in
  `coordinator.rs`) are spawned at startup and live for the process lifetime.
- **No blocking calls inside `async fn`.** CPU-bound numeric kernels go
  through `tokio::task::spawn_blocking` when they cannot be made fast enough
  to run inline on the executor.
- **Cancellation is opt-in.** Long-running tasks accept a
  `tokio_util::sync::CancellationToken` or equivalent; they do not rely on
  task abort.
- **Channels: `tokio::sync::mpsc` for fan-in, `dashmap` for shared state.**
  We avoid `Arc<Mutex<HashMap<...>>>` on the hot path because `dashmap`
  removes the global lock.
- **`#[tokio::test]` for async tests.** The integration tests under `tests/`
  follow this pattern uniformly.

## Concurrency Primitives

- `dashmap` 6.1 for any concurrent map that sees high read/write contention
  (embedding cache, session table).
- `parking_lot` 0.12 for the few read-mostly mutexes; `parking_lot::RwLock`
  is preferred over `std::sync::RwLock` for shorter critical sections.
- Per-shard structures rather than one big lock whenever possible.

## Naming Conventions

- **Crate name: `ruvllm`** (lowercase, no hyphen). The capitalized form
  `RuvLLM` appears only in prose, never in code identifiers.
- **Binary names: `ruvllm-*`** (lowercase, hyphenated). The seven binaries
  follow this without exception. See [Codebase Summary](codebase-summary.md).
- **Modules: short, lowercase, no underscores when avoidable.**
  `inference_real.rs` is one of the few exceptions, intentionally signaling
  "this is the real-inference variant of `inference.rs`."
- **Types: `UpperCamelCase`.** Acronyms collapsed: `Lora`, not `LoRA`, in
  identifiers (the prose form remains "LoRA").
- **Errors end in `Error`** when they are the top-level enum, e.g. `Error`
  in `error.rs` is intentionally short because it is always namespaced.

## File Size Limits

A file that grows past ~800 lines is a candidate for splitting. The
`sona/` submodule is the canonical example: it was a single file and was
split when it crossed that threshold. New files should aim for <500 lines and
single-responsibility.

## Testing Convention

- **Unit tests live next to the code** in `#[cfg(test)] mod tests { ... }`
  inside the same file. They are small and exercise pure functions.
- **Integration tests live under `tests/`.** They are async, use `#[tokio::test]`,
  and exercise the full orchestrator. See `tests/integration.rs` and
  `tests/sona_integration.rs`.
- **Benches live under `benches/`** and use Criterion 0.5 with `async_tokio`
  and `html_reports`. See [Testing Guide](testing-guide.md) for the full list.
- **Latency claims must be benched.** Any change that touches a hot-path
  module (`embedding`, `memory`, `router`, `attention`, `inference`,
  `simd_inference`, anything in `sona/loops/`) must be accompanied by a
  before/after Criterion run.

## SIMD and Platform Code

- Runtime detection only — never compile-time `#[cfg(target_feature = "...")]`
  on hot-path code, because the deployed binary may run on a different CPU
  than the build host. `simsimd` and `simd_inference.rs` both follow this.
- The `simd_inference.rs` dispatcher checks AVX2, SSE4.1, then NEON, then
  falls through to scalar.
- `ruvllm-simd-demo` exists specifically to print which path was selected, so
  deployments can verify the right kernel got picked.

## Public API Stability

- The library exposes a small public surface (`RuvLLM` struct, request/response
  types, error enum). See [API Reference](api-reference.md).
- Internal modules are `pub(crate)` unless they need to be re-exported.
- HTTP endpoints are versioned by path prefix when they change shape.

## Documentation

- **rustdoc on every public item.** Internal items are documented when their
  invariants are non-obvious.
- **`/// # Examples` blocks compile.** Doctests are part of `cargo test`.
- **Architectural docs live in `docs/`** and are referenced from rustdoc when
  a function is part of a documented subsystem (e.g. SONA).

## See also

- [Testing Guide](testing-guide.md)
- [System Architecture](system-architecture.md)
- [Codebase Summary](codebase-summary.md)
- [Configuration Guide](configuration-guide.md)
