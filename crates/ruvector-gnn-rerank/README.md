# ruvector-gnn-rerank

**GNN score-diffusion reranking for approximate ANN search** — recovers recall lost to
quantization by smoothing candidate scores over a k-NN graph of the (full-precision) candidate
vectors. Part of the [ruvector](https://github.com/ruvnet/ruvector) ecosystem.

> Measured: **recall@10 28.0% → 38.4% (+10.4pp)** on a clustered synthetic benchmark (N=5K, D=128),
> as a cheap second-stage rerank over the top candidates from a noisy first-stage index.

## Why

A quantized / coarse ANN index returns approximately-correct candidates but mis-ranks them near the
top-k boundary. `GnnDiffusion` builds a k-NN graph over the candidates' full-precision vectors and
diffuses each candidate's score across its graph neighbours — pulling true neighbours (which got an
unluckily-low noisy score) back up.

## Rerankers

| variant | what it does |
|---|---|
| `NoisyScoreReranker` | passthrough baseline (sort by the first-stage score) |
| `GnnDiffusionReranker` | 1-hop score diffusion over the candidate k-NN graph — **the +10.4pp win** |
| `GnnMincutReranker` | coherence-gated diffusion (propagate only across structurally-similar edges) |
| `ExactL2Reranker` | exact L2 re-scoring (quality ceiling) |

## Usage

```rust
use ruvector_gnn_rerank::{Candidate, CandidateReranker, GnnDiffusionReranker};

let candidates = vec![
    Candidate { id: 0, vector: vec![/* full-precision */], noisy_score: 0.81 },
    // … the top candidates from your ANN index
];
let reranker = GnnDiffusionReranker::default(); // alpha=0.6, hops=1, k_graph=8
let top = reranker.rerank(&query, &candidates, 10)?;
```

## Performance & honesty

This is a **recall/latency tradeoff**, not free throughput. On the same benchmark:

| variant | latency | throughput |
|---|---|---|
| NoisyScore (no rerank) | ~0.15 µs/q | ~7 M QPS |
| GnnDiffusion (+10.4pp) | ~300 µs/q | ~2.5 K QPS |

Right for a rerank stage over a small candidate set; not a replacement for the first-stage index.

## Robustness

Inputs are validated fail-fast: non-finite scores/vectors and mixed candidate dimensions are
rejected (`RerankerError`) rather than silently producing a corrupted ranking — relevant to the
poisoned-first-stage (MemoryGraft) threat model.

## Test & benchmark

```bash
cargo test -p ruvector-gnn-rerank                                   # unit + recall regression + security
cargo test -p ruvector-gnn-rerank --release --test perf_benchmark -- --ignored --nocapture   # latency
```

## License

MIT © Ruvector Team. See ADR-194 for design notes.
