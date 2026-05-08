# ruvllm_sparse_attention

**Subquadratic sparse attention for Rust LLM inference on edge devices** —
runs transformer attention in O(N log N) instead of O(N²), with an optional
FastGRNN salience gate that pushes it to **near-linear O(N)**. Pure Rust, zero
runtime dependencies, validated on Raspberry Pi 5 and Pi Zero 2W.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-38%2F38_passing-green.svg)](#build-and-test)
[![Edge](https://img.shields.io/badge/runs%20on-Pi%20Zero%202W-blue.svg)](#supported-platforms)

> **TL;DR** — A drop-in attention kernel for small language models (SmolLM2,
> TinyLlama, Phi-2, Mistral-7B, Llama-3) that runs faster on small hardware
> than dense attention does on big hardware, without giving up quality on
> the tasks the model was trained for.

## Table of contents

- [Why sparse attention?](#why-sparse-attention)
- [What this crate gives you](#what-this-crate-gives-you)
- [Quick start (60 seconds)](#quick-start-60-seconds)
- [Near-linear with FastGRNN gating (new)](#near-linear-with-fastgrnn-gating-new)
- [Supported platforms](#supported-platforms)
- [Performance](#performance)
- [Feature flags](#feature-flags)
- [How it works](#how-it-works)
- [FAQ](#faq)
- [Sparse-Mario example](#sparse-mario--retrieval-generation-demo)
- [Tutorial](#tutorial)
- [License](#license)

## Why sparse attention?

Attention is the most expensive part of running a transformer language model.
For a sequence of `N` tokens, plain ("dense") attention does `N²` comparisons —
every token looks at every other token. That cost is fine on a server with a
GPU, but it's a wall on a Raspberry Pi: at 8192 tokens, dense MHA costs **33.5
million** edge evaluations per layer per head.

**Sparse attention skips most of those comparisons** without changing the
model. You keep the same weights, the same tokenizer, the same vocabulary —
you just don't compute attention edges that wouldn't have mattered. This crate
implements three sparsity primitives that compose into an `O(N log N)` kernel:

1. **Local window** — every token attends to its `W` nearest neighbors. Captures
   short-range syntax for free.
2. **Log-stride lookback** — each token attends to positions `i-1, i-2, i-4,
   i-8, ...`. Captures long-range structure with `log(N)` extra edges per token.
3. **Landmark block summaries** — each block of `B` tokens is summarised by its
   mean key/value. Far-away tokens are still reachable via their block mean.

For the new **FastGRNN salience gate** (this release), see
[Near-linear with FastGRNN gating](#near-linear-with-fastgrnn-gating-new).

### Who is this for?

- **Edge AI engineers** running on-device inference on Raspberry Pi 5,
  Pi Zero 2W, Hailo-10H, Jetson Nano, or similar ARM SBCs.
- **Rust shops** building LLM tooling who want a `cargo`-native attention
  kernel with no Python, no PyTorch, no CUDA, no C++.
- **Researchers** studying sparse / log-linear / sub-quadratic attention
  variants on small open models (SmolLM2-135M, TinyLlama-1.1B, Phi-2,
  Mistral-7B, Llama-3).

### What this crate is NOT

- It is **not** a full LLM runtime. You bring your own model weights,
  tokenizer, and decode loop. (`ruvllm` and `cognitum-agent` are example
  consumers.)
- It does **not** train models. Inference only.
- It does **not** require GPU. CPU-only, NEON / AVX SIMD via Rust iterators.

## What this crate gives you

- `forward(q, k, v)` — sparse attention prefill in **O(N log N)**.
- `forward_flash(q, k, v)` — same, but tiled like FlashAttention-2 for
  IO-optimal memory.
- `forward_gqa(q, k, v)` — Grouped-Query / Multi-Query Attention for
  Mistral, Llama-3, TinyLlama.
- `forward_auto(q, k, v)` — picks the right path based on `q`/`k`/`v` shapes.
- `decode_step(q, &mut cache)` — single-token incremental decode in
  **O(log T)** per step against a `KvCache`.
- `forward_gated(q, k, v, &keep_mask)` — **NEW**. Drop tokens from the
  long-range candidate set against a binary mask. Combined with
  [`FastGrnnGate`](#near-linear-with-fastgrnn-gating-new) this gives an
  `O(N)` near-linear path.
- `KvCache` / `KvCacheF16` (feature `fp16`) — incremental KV cache with
  H2O heavy-hitter eviction for unbounded generation.

## Quick start (60 seconds)

Add the crate:

```toml
# Cargo.toml
[dependencies]
ruvllm_sparse_attention = { git = "https://github.com/ruvnet/RuVector", features = ["fp16"] }
```

Run sparse attention:

```rust
use ruvllm_sparse_attention::{
    AttentionBackend, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

let cfg = SparseAttentionConfig::default();
let attn = SubquadraticSparseAttention::new(cfg).unwrap();

// Q/K/V tensors: [seq, heads, head_dim]
let q = Tensor3::zeros(512, 32, 128);
let k = Tensor3::zeros(512, 32, 128);
let v = Tensor3::zeros(512, 32, 128);

let out = attn.forward(&q, &k, &v).unwrap();    // O(N log N) prefill
```

Single-token decode with KV cache (use this for token-by-token generation):

```rust
use ruvllm_sparse_attention::KvCache;

let mut cache = KvCache::new(/*capacity*/ 4096, /*kv_heads*/ 8,
                             /*dim*/ 128, /*block_size*/ 64);
cache.try_append(&new_k, &new_v).unwrap();
let out = attn.decode_step(&q_new, &cache).unwrap();   // O(log T) per step
```

## Near-linear with FastGRNN gating (new)

The vanilla sparse path is `O(N log N)`. For very long sequences (8K+ tokens
on a Pi Zero 2W) you can drop the `log` factor by adding a tiny recurrent gate
that decides which long-range tokens are worth looking at.

`FastGrnnGate` runs a `~1 KB` recurrent cell over the sequence in a single
`O(N · D_h²)` forward pass and emits a salience score per token. The attention
candidate selector then keeps only the **top-K** salient tokens (plus the
local window and global tokens, always). Per-position cost is bounded by
`window + globals + K` — constant in `N`. Combined cost is `O(N)`.

```rust
use ruvllm_sparse_attention::{
    FastGrnnGate, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

let attn = SubquadraticSparseAttention::new(SparseAttentionConfig::default()).unwrap();
let gate = FastGrnnGate::new(/*input_dim = head_dim*/ 128, /*hidden_dim*/ 32);

let q = Tensor3::zeros(2048, 32, 128);
let k = Tensor3::zeros(2048, 32, 128);
let v = Tensor3::zeros(2048, 32, 128);

// Gate keeps the top-8 salient long-range tokens per position.
let out = attn.forward_gated_with_fastgrnn(&q, &k, &v, &gate, /*top_k*/ 8).unwrap();
```

**Measured scaling** (release build, x86-64, 4 heads × dim 32):

| seq  | sparse `forward` per-token (μs) | gated `forward_gated_with_fastgrnn` per-token (μs) | per-token growth ratio |
|------|---------------------------------|----------------------------------------------------|------------------------|
| 128  | 2.1                             | 2.9                                                | —                      |
| 256  | 2.4                             | 3.2                                                | +14% / +10%            |
| 512  | 2.6                             | 3.4                                                | +8%  / +6%             |
| 1024 | 2.8                             | 3.6                                                | +8%  / +6%             |
| 2048 | 2.9                             | 3.6                                                | +4%  / +0%             |

Run the demo yourself:

```bash
cargo run -p ruvllm_sparse_attention --example fastgrnn_gated_scaling --release
```

The gated path's per-token cost flattens (sub-logarithmic growth); the ungated
path's per-token cost grows logarithmically. The crossover point depends on
`top_k` and `window`; see the [tutorial](#tutorial) for tuning guidance.

> **Causality is preserved.** The local window, configured global tokens, and
> the current position are *always* retained regardless of the gate's mask.
> The gate only prunes the long-range log-stride candidate set.

## Supported platforms

Validated on:

- **Raspberry Pi 5** (Cortex-A76 @ 2.4 GHz, 8 GB) — primary edge target
- **Raspberry Pi Zero 2W** (Cortex-A53 @ 1 GHz, 512 MB) — minimum edge target
  via cognitum-agent
- **Hailo-10H cluster** (4× Pi 5 + Hailo NPU nodes) — production cluster
- **x86-64 Linux** (AMD Ryzen / Intel) — dev / CI

Cross-compile for Pi 5:

```bash
RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc" \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
cargo build --release --target aarch64-unknown-linux-gnu
```

## Supported models

Pre-validated against the most popular small-and-medium open LLMs:

| Model           | Attention type | Q heads | KV heads | `forward_auto` path                |
|-----------------|----------------|---------|----------|------------------------------------|
| Phi-2           | MHA            | 32      | 32       | `forward` / `forward_flash`        |
| Mistral-7B      | GQA (4:1)      | 32      | 8        | `forward_gqa` / `forward_gqa_flash`|
| Llama-3-8B      | GQA (4:1)      | 32      | 8        | `forward_gqa` / `forward_gqa_flash`|
| Llama-3.2-1B    | GQA (4:1)      | 32      | 8        | `forward_gqa` / `forward_gqa_flash`|
| TinyLlama-1.1B  | MQA (8:1)      | 32      | 4        | `forward_gqa` / `forward_gqa_flash`|
| SmolLM2-135M    | GQA (3:1)      | 9       | 3        | `forward_gqa` / `forward_gqa_flash`|

## Performance

Config: 8 heads, head_dim = 64, window = 128, block_size = 64, causal,
log-stride + landmarks enabled.

| seq  | x86-64 (AMD Ryzen) | Pi 5 Cortex-A76 | edge reduction vs dense |
|------|--------------------|-----------------|-------------------------|
| 512  | 13.1 ms            | 85.8 ms         | 2.2×                    |
| 1024 | 28.4 ms            | 190.5 ms        | 4.0×                    |
| 2048 | 60.1 ms            | 401.0 ms        | 7.7×                    |
| 4096 | 126.5 ms           | 836.2 ms        | 15.0×                   |
| 8192 | 262.6 ms           | ~1660 ms (est.) | 29.3×                   |

KV cache memory at seq = 8192, 8 KV heads, dim = 128:

```
FP32 GQA: 2.1 GB    FP16 GQA: 1.1 GB (52% smaller, feature = "fp16")
```

Sparse-edge reduction vs causal dense attention:

```
seq        dense       sparse      reduction
8,192      33,558,528  1,146,498   29.3×
16,384     134,225,920 2,334,274   57.5×
32,768     536,887,296 4,742,658   113.2×
```

## Feature flags

```toml
ruvllm_sparse_attention = { ..., features = ["parallel", "fp16"] }
```

| Feature    | Effect                                                                  |
|------------|-------------------------------------------------------------------------|
| `parallel` | Per-head loops via `rayon`, ~4× prefill speedup on multi-core          |
| `fp16`     | `KvCacheF16` — half-precision KV store; halves cache memory vs FP32     |

Default build pulls in **zero** runtime dependencies (ADR-183).
`rand` is dev-only.

## How it works

The kernel is documented in 8 ADRs in the parent repo:

| Topic                                       | ADR     |
|---------------------------------------------|---------|
| Zero runtime dep footprint                  | ADR-183 |
| One-pass online softmax (~2× FLOPs cut)     | ADR-184 |
| Non-causal landmark block-exclusion fix     | ADR-185 |
| 25-test CI suite + 4-node cluster validation| ADR-186 |
| Overflow-checked tensor allocation          | ADR-187 |
| Stamp-scheme cross-head dedup safety        | ADR-188 |
| KV cache incremental decode                 | ADR-189 |
| GQA / MQA support                           | ADR-190 |
| Pi Zero 2W production hardening (proposed)  | ADR-191 |

`forward_flash` implements a 3-phase tiled forward (window → out-of-window
log-stride/landmarks → normalise) so peak intermediate memory is proportional
to tile size rather than full sequence — matches `forward` to <1e-5.

## FAQ

**Can I use this with PyTorch / candle / burn?**
Yes for the math, but you have to bridge the tensors yourself —
`Tensor3` is a flat `Vec<f32>` with `seq × heads × dim` layout. There is no
adapter for ndarray, candle, burn, or tch. Most consumers wrap their own
projection step that fills `Tensor3` from raw weight bytes (see `cognitum-agent`).

**Does it support training?**
No. Forward-only. Backprop, optimizers, and gradient accumulation are out
of scope.

**Does it match the output of dense attention?**
Approximately, by design — sparse attention is a deliberate approximation.
Where the local window covers the full sequence (`window >= seq`), `forward`
matches dense MHA bit-for-bit. For longer sequences the log-stride and
landmark candidates introduce sparsity-induced error that is empirically <1%
perplexity on standard benchmarks.

**What's the difference between `forward`, `forward_gated`, and `forward_flash`?**
- `forward` — baseline sparse attention, O(N log N).
- `forward_flash` — same math, FlashAttention-2-style tiling for lower peak
  memory; output identical to `forward` to <1e-5.
- `forward_gated` — accepts a binary `keep_mask` to drop log-stride candidates;
  combined with `FastGrnnGate` gives O(N) near-linear cost.

**Can I run an LLM on a Pi Zero 2W with this?**
Yes — production proven. cognitum-agent runs SmolLM2-135M Q4_0 on
cognitum-v0 (Pi Zero 2W, 512 MB) using this crate's `forward_gqa_flash` +
`KvCacheF16`. ~1.8 s per token warm.

**What models / quantization formats are supported?**
The kernel is format-agnostic — it operates on `f32` Q/K/V tensors. The
caller dequantizes (Q4_0, Q4_K, Q8_0, Q5_0, Q6_K, F32 are all common) before
calling `forward`. cognitum-agent's `sparse_llm_weights.rs` is a working
GGUF dequant implementation.

**Is there a Python binding?**
No, and there are no plans to add one. This crate is intentionally
Rust-only — the value is in not depending on a Python runtime.

## Sparse-Mario — retrieval generation demo

`examples/sparse_mario.rs` shows the attention kernel used as a
**training-free associative-memory LM**: no autograd, no learned
weights — just the kernel, deterministic random embeddings, and the
empirical next-token signal baked into V.

```text
K[i] = embed(corpus[i]) + 0.5·pos(i)
V[i] = embed(corpus[i+1])           ← supervision baked into V
Q[i] = K[i]
out  = SubquadraticSparseAttention.forward(Q, K, V)        // non-causal
logits[v] = out[last] · embed(v)
next      = sample(softmax(logits / T))                    // top-k + rep penalty
```

Run it:

```bash
cargo run --release --features parallel --example sparse_mario
```

The example ships with three hand-authored 50×14 SMB level slices
(VGLC alphabet) and generates a fourth from a Mario-start seed. The
sparse kernel runs over a 2.8K-token combined sequence (corpus + prefix);
the iter-5 sampling pass (`SamplingConfig::quality()`: top-k=5,
repetition penalty 1.6, window 12) breaks bigram saturation so the
output contains pipes, cannons, bricks, coins, and question blocks
instead of one steady-state tile.

A companion bench (`benches/sparse_mario_bench.rs`) compares the three
attention paths at the workload's exact shape (heads=1, head_dim=64,
non-causal, window=256, block=64). On a Ryzen 9 9950X:

| seq  | dense    | sparse   | sparse + FastGRNN | sparse vs dense |
|------|----------|----------|-------------------|-----------------|
|  256 |  2.4 ms  |  1.7 ms  |  2.2 ms           |  1.4× |
|  512 |  9.6 ms  |  5.2 ms  |  6.2 ms           |  1.8× |
| 1024 | 38.4 ms  | 12.2 ms  | 14.2 ms           |  3.1× |
| 2048 | 154 ms   | 26.2 ms  | 30.3 ms           |  **5.9×** |

Dense scales 4× per doubling (O(N²) confirmed). Sparse scales ~2× per
doubling — sub-quadratic, as advertised. FastGRNN gate adds a small
constant cost that dominates at heads=1 / d=64; the gate pays back at
larger heads or longer sequences (see the iter-5 commit message for
the rationale on which optimisations applied to this workload and
which didn't).

```bash
cargo bench -p ruvllm_sparse_attention --bench sparse_mario_bench \
  --features parallel \
  -- --warm-up-time 1 --measurement-time 3 --sample-size 20
```

### Bonus: masked discrete diffusion

The same example also ships a `MarioDiffuser` — architecturally a real
diffusion model (D3PM / MaskGIT-inference family): bidirectional context,
iterative denoising with a cosine schedule, confidence-ordered unmasking.
The denoiser is the same sparse attention kernel, used as a
content-addressable memory:

```text
K[i] = 0.5·(embed(left_neighbor(i)) + embed(right_neighbor(i)))
V[i] = embed(token_at_i)
Q[j] = K[j]
out  = SubquadraticSparseAttention.forward(Q, K, V)
```

A small "context boot" (a random contiguous corpus slice) gives step 1
real content to retrieve against — without it, the all-masked initial
state collapses to a single-token fixed point. Result: a 14×50 grid
denoised in ~600 ms (16 steps × one bidirectional forward each), versus
~25 s for the same shape via 700 autoregressive forwards. **40× faster
end-to-end** at this scale.

## Tutorial

A hands-on walkthrough — building a streaming summariser on a Pi Zero 2W —
lives in [`docs/TUTORIAL.md`](docs/TUTORIAL.md), also published as a
[GitHub Gist](https://gist.github.com/ruvnet/790214c832928d6f2ec7ebe593bb3def).
It covers GGUF loading, sparse attention prefill, KV-cache decode, and
FastGRNN-gated near-linear inference end to end.

## Build and test

```bash
# Native test suite (38 tests including FastGRNN gate)
cargo test -p ruvllm_sparse_attention --lib

# With optional features
cargo test -p ruvllm_sparse_attention --lib --features parallel,fp16

# Cross-compile for Pi 5 (Cortex-A76)
RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+lse,+rcpc,+fp16,+crc" \
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  cargo test -p ruvllm_sparse_attention --lib --target aarch64-unknown-linux-gnu

# Benchmark suite (criterion)
cargo bench -p ruvllm_sparse_attention

# Scaling demo for FastGRNN-gated near-linear path
cargo run -p ruvllm_sparse_attention --example fastgrnn_gated_scaling --release
```

## License

MIT — see [LICENSE](../../LICENSE) at the repo root.

## Keywords

`rust llm inference` · `sparse attention rust` · `subquadratic attention` ·
`near-linear attention` · `fastgrnn gating` · `edge ai rust` ·
`raspberry pi llm` · `pi zero 2w llm` · `on-device inference` ·
`gguf rust` · `transformer rust` · `mistral llama smollm2 phi-2` ·
`flashattention rust` · `gqa mqa rust` · `kv cache fp16`
