# ruvllm_retrieval_diffusion

**Training-free retrieval LM and masked discrete diffusion that work on any
small-vocab token domain — game levels, drum patterns, configs, MIDI loops,
visual tokens.** Built on the
[`ruvllm_sparse_attention`](https://crates.io/crates/ruvllm_sparse_attention)
kernel; no autograd, no learned weights, no Python in the loop.

This is the corpus-agnostic generalisation of the
[Sparse-Mario gist](https://gist.github.com/ruvnet/d3e0aaa7af2745b678a9eecddf610301).

## What you get

Two pipelines from one kernel, parameterised by a runtime `RetrievalConfig`
(vocab size, embedding dim, mask sentinel, sampling controls):

- **`Retriever::generate_fast`** — autoregressive next-token retrieval via
  `KvCache` + `decode_step`. O(log T) per generated token. ~3,000× faster
  than the reference full-forward path on the Mario benchmark.
- **`Diffuser::diffuse`** — bidirectional masked discrete diffusion with a
  MaskGIT cosine schedule and a corpus-slice context boot. Beats AR by
  6.9× on the Mario aggregate metric; SOTA-on-this-artifact for training-
  free PCG.

## Plug-in checklist

```rust
use ruvllm_retrieval_diffusion::{Retriever, Diffuser, RetrievalConfig, SamplingConfig};

// 1. Pick a vocab. Each token is a u8 index < vocab_size.
let cfg = RetrievalConfig {
    vocab_size: 5,        // your domain's atomic tokens
    head_dim: 64,         // 64 works well for vocab ≤ 32
    pos_scale: 0.0,       // 0.0 if domain is shape-invariant; 0.5 for grids
    mask_sentinel: 255,   // any byte ≥ vocab_size
    ..RetrievalConfig::default()
};

// 2. Encode your corpus into u8 tokens.
let corpus: Vec<u8> = my_encoder("examples and structure");

// 3. Build the retriever (one-time cost).
let retriever = Retriever::new(corpus, cfg, /* embedding seed */ 0xCAFE_BABE);

// 4. Generate token-by-token (AR) or fill a fixed-shape grid (Diffusion).
let cont = retriever.generate_fast(&seed, 256, &SamplingConfig::quality(), 0xC0FFEE);
let grid = Diffuser::new(&retriever).diffuse(700, 24, &SamplingConfig::quality(), 0xD1FFCAFE);
```

## Two examples ship in this crate

```bash
# Drum-pattern generator — 5-token vocab, 4-bar loops
cargo run --release -p ruvllm_retrieval_diffusion --example drum_patterns
```

The Mario example lives in the parent crate
(`crates/ruvllm_sparse_attention/examples/sparse_mario.rs`) — it predates
this generalisation but uses the same algorithmic approach.

## When to pick which pipeline

| Use case | Best path |
|---|---|
| Token-by-token streaming output | `Retriever::generate_fast` |
| Fixed-shape grid you'll fill all at once | `Diffuser::diffuse` |
| Inpainting / repair (some tokens already known) | `Diffuser::diffuse` after pre-filling known positions in the working buffer |
| Latency-critical, low-end hardware | `Retriever::generate_fast` (single KvCache, single decode per token) |

## Domain-specific knobs

`pos_scale` is the single most important config. **0.0** makes the AR
retriever purely content-based (good for cyclic / shape-invariant domains
like drum patterns). **0.5** lets retrieval inherit some absolute-position
structure (good for grid-shaped domains where row/column index matters,
like Mario levels).

`diffusion_context_weights` controls the bidirectional radius. Default
`[0.5, 0.10]` pulls token identity from the immediate neighbour with a
small contribution from offset-2; bump or extend the array for larger
context windows (with diminishing returns past radius 2).

`SamplingConfig::quality()` returns the Mario-validated recipe; tune
`no_repeat_window` to your domain's meaningful local span.

## What this is NOT

- Not a trained model. There is no learning step. The corpus *is* the model.
- Not a substitute for a real LLM. Outputs are bigram-grade; no long-range
  structure, no syntax awareness, no counting.
- Not specific to any one domain. The Mario application is a worked example;
  the kernel-as-memory pattern is the artifact.

## Where this came from

The `sparse-mario` branch of the parent repository chronicles the 13
iterations that built and validated this approach end-to-end on Super Mario
Bros levels:

- Iter 1-7: corpus, AR LM, ASCII output, masked discrete diffusion.
- Iter 8: KvCache + decode_step for AR (2,880× speedup).
- Iter 9-10: top-p sampling, multi-token bidirectional context.
- Iter 11-13: PCG metrics, hyperparameter sweep, cross-baseline comparison
  (SOTA on this artifact: 3.8× lower L2 distance to corpus than a 1st-order
  Markov bigram baseline).

See the [Sparse-Mario gist](https://gist.github.com/ruvnet/d3e0aaa7af2745b678a9eecddf610301)
for the full iteration log, benchmarks, and SOTA comparison table. This crate
is the generalisation step — same code, packaged corpus-agnostic.

## License

MIT, same as the underlying kernel.
