# Patches Directory

**CRITICAL: Do not delete this directory or its contents!**

This directory contains patched versions of external crates that are necessary for building RuVector.

## hnsw_rs

The `hnsw_rs` directory contains a patched version of the [hnsw_rs](https://crates.io/crates/hnsw_rs) crate.

### Why this patch exists

The official hnsw_rs crate uses `rand 0.9` which is **incompatible with WebAssembly (WASM)** builds. This patched version:

1. Uses `rand 0.8` instead of `rand 0.9` for WASM compatibility
2. Uses Rust edition 2021 (not 2024) for stable Rust toolchain compatibility

### How it's used

The patch is applied via `Cargo.toml` at the workspace root:

```toml
[patch.crates-io]
hnsw_rs = { path = "./patches/hnsw_rs" }
```

This ensures all workspace crates that depend on `hnsw_rs` use this patched version.

### What depends on it

- `ruvector-core` (with `hnsw` feature enabled by default)
- `ruvector-graph` (with `hnsw_rs` feature)
- All native builds (Node.js bindings, CLI tools)

### Consequences of deletion

If this directory is deleted:
- **All CI builds will fail** (Build Native Modules, PostgreSQL Extension CI, etc.)
- `cargo build` will fail with "failed to load source for dependency `hnsw_rs`"
- The project cannot be compiled

### Updating the patch

If you need to update hnsw_rs:
1. Download the new version from crates.io
2. Apply the rand 0.8 compatibility changes from the current patch
3. Test WASM and native builds before committing

## candle-transformers

The `candle-transformers` directory contains a patched version of the
[candle-transformers](https://crates.io/crates/candle-transformers) crate (0.9.2), the
same version otherwise resolved from crates.io.

### Why this patch exists

`quantized_llama::ModelWeights::forward` always slices its output down to the
last sequence position (`x.i((.., seq_len - 1, ..))`) before the output
projection, so a multi-token forward call only ever returns one token's worth
of logits no matter how many new tokens were fed in. `ruvllm`'s speculative
decoding needs per-position logits to verify several draft tokens against the
main model in a single batched forward pass (the whole point of speculative
decoding is amortizing that forward pass over K tokens instead of paying for
K separate decode steps). This patch adds one new method,
`forward_all_positions`, that shares `forward`'s entire layer loop and only
skips the final last-position slice. `forward` itself is untouched.

Three more additions in the same file, all needed by the same feature:

- `causal_mask_with_offset`: `ModelWeights::mask` only ever builds a square
  `[t, t]` causal mask, correct only when the KV cache is empty. A
  multi-token forward continuing an existing cache produces attention
  scores of shape `[b, heads, t, index_pos + t]`, which a `[t, t]` mask
  fails to broadcast against — this combination (`index_pos > 0` and
  `t > 1`) never arose before `forward_all_positions` existed, since every
  other caller fed either the whole (empty-cache) prompt or one token at a
  time. `forward_all_positions` uses this instead of `self.mask(...)`.
- `ModelWeights::snapshot_kv_cache` / `restore_kv_cache`: cheap (O(num
  layers), shared-storage `Tensor` clones, not data copies) capture/restore
  of every layer's KV cache. Speculative decoding uses this to roll back a
  rejected draft token without resetting to position 0 and replaying the
  entire context — `LayerWeights.kv_cache` is a private field, so this can
  only be added from inside the same module.

### How it's used

```toml
[patch.crates-io]
candle-transformers = { path = "./patches/candle-transformers" }
```

Cargo's `[patch]` only overrides sources for versions that satisfy the
patched crate's own declared version (`0.9.2` here), so this only affects
workspace members that depend on `candle-transformers = "0.9"` (currently
`ruvllm`). Other workspace crates pinned to `candle-transformers = "0.8"`
keep resolving to the unpatched crates.io release.

### What depends on it

- `ruvllm` (`candle` feature) — `src/backends/candle_backend.rs` calls
  `forward_all_positions` from its batched-logits API used by
  `src/speculative.rs`.

### Consequences of deletion

- `ruvllm`'s speculative decoding batched-verify path will fail to compile
  (`forward_all_positions` not found).
- Removing the `[patch.crates-io]` entry silently falls back to the
  unpatched crate and produces the same compile error.

### Updating the patch

If you need to update candle-transformers:
1. Download the new version from crates.io (`curl -A "<contact>" https://crates.io/api/v1/crates/candle-transformers/<version>/download`)
2. Re-apply `forward_all_positions` next to `forward` in `src/models/quantized_llama.rs`
3. Bump the version in this patch's `Cargo.toml` to match
4. Bump `candle-transformers = "..."` in `crates/ruvllm/Cargo.toml` if needed
5. `cargo build -p ruvllm --features candle` before committing
