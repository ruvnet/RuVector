# ADR-305: Coherence-Adaptive Quantization (Mutual-kNN Bit-Width Allocation)

**Date**: 2026-08-15
**Status**: Closed — negative result (documented; not recommended for production)
**Deciders**: Nightly research agent
**Tags**: ann, quantization, coherence, mincut, ruvector-coherence-quant, negative-result

---

## Context

RuVector owns multiple uniform quantization codecs (`ruvector-rabitq`, `ruvector-turboquant`,
`ruvector-pq-search`, `ruvector-matryoshka`) and a real dynamic min-cut / graph conductance engine
(`ruvector-mincut`). No prior work combined the two: every existing quantization scheme applies
the same bit budget to every stored vector, regardless of whether that vector is locally
redundant (deep in a dense cluster, cheaply reconstructable from neighbours) or locally sensitive
(near a boundary/bridge, where quantization noise is more likely to flip a nearest-neighbour
ranking).

Agent-memory corpora grow faster than operators want to pay for full-precision storage. A working
content-aware bit-allocation policy would let an index spend precision where it is needed and save
it where it is not — the same principle behind the (also-rejected) entropy-adaptive beam search
work in ADR-303, applied to storage instead of search.

---

## Hypothesis

```text
Given a 4,000-vector corpus at dimension 32, 12 semantic clusters, noise=0.18,

when per-vector scalar quantization bit-width is chosen by mutual-kNN coherence
(>= 0.5 -> 4-bit; < 0.5 -> 8-bit) instead of a uniform 4-bit budget,

then recall@10 on held-out queries should recover to within 1.5 percentage points of the
uniform 8-bit baseline,

subject to total memory remaining <= 65% of the 8-bit baseline, and subject to uniform 4-bit
itself showing a real (>1.5pp) recall gap versus baseline.
```

Fixed before benchmarking; not changed after seeing results.

---

## Decision

**Do not adopt mutual-kNN coherence as a quantization bit-allocation signal.** The PoC computes a
mutual-kNN coherence score (fraction of a vector's k nearest neighbours that are mutually
nearest-neighbours) as a lightweight, ground-truth-free structural proxy for local graph
conductance, and uses it to allocate 4-bit vs 8-bit scalar quantization per vector. It was
measured against uniform 8-bit and uniform 4-bit baselines and **fails both pre-registered
acceptance gates**. The crate is merged as a documented negative result and a reusable benchmark
harness (deterministic dataset generation, real bit-packed quantization, brute-force k-NN
coherence scoring) for future quantization-signal experiments.

### Measured result

```
baseline_uniform_8bit           recall=0.9887  mem=156KB  8.00 bits/dim
candidate_A_uniform_4bit        recall=0.8583  mem= 93KB  4.00 bits/dim
candidate_B_coherence_adaptive  recall=0.8770  mem=116KB  5.47 bits/dim  (63.3% core @4bit, 36.7% boundary @8bit)

Gate 1 (uniform 4-bit shows real signal, gap > 1.5pp): gap = 13.03pp -> PASS
Gate 2 (candidate B recall within 1.5pp of baseline):  gap = 11.17pp -> FAIL
Gate 3 (candidate B memory <= 65% of baseline):        74.7%        -> FAIL

VERDICT: REJECT
```

Reproducible: two independent runs produced bit-identical recall and memory values (latency
varies a few percent between runs, as expected for wall-clock CPU timing).

Reproduce with:

```bash
cargo test --release -p ruvector-coherence-quant
cargo run --release -p ruvector-coherence-quant --bin benchmark
```

---

## Why the signal fails (measured)

The hypothesis was that mutual-kNN coherence encodes quantization sensitivity:

- High mutuality (dense, locally-agreed-upon neighbourhood) → redundant → safe to compress hard.
- Low mutuality (boundary/bridge point) → few redundant neighbours to fall back on → keep precision.

The measurements refute this on the PoC data:

1. **Mutuality is real but measures the wrong quantity.** A dedicated unit test
   (`coherence::tests::clustered_corpus_has_higher_mean_coherence_than_random`) confirms mutual-kNN
   coherence is genuinely higher on clustered data than on random point clouds — the signal is not
   noise. But quantization-induced recall loss under min-max scalar quantization is driven by a
   vector's own coordinate-range shape (how much a `range/15` quantization step distorts its
   components) and its absolute distance to its true top-k, not by whether its neighbours agree it
   is nearby.
2. **Weak transfer, not zero transfer.** Candidate B did recover some recall (87.70% vs 85.83% for
   uniform 4-bit) — the direction is not reversed, unlike the ADR-303 entropy sign flip. But giving
   36.7% of the corpus a full doubling of precision closed only 14% of the recall gap while adding
   25% more memory than uniform 4-bit, which is not a favourable trade against either gate.
3. **Threshold choice (0.5) was not the failure mode.** The gap between candidate B and the 1.5pp
   target (11.17pp vs 1.5pp) is large enough that no reasonable coherence-threshold retuning would
   close it without abandoning the memory-budget gate — this is a signal-quality problem, not a
   hyperparameter problem.

---

## Alternatives Considered

| Alternative | Notes |
|---|---|
| Uniform quantization (status quo) | Remains the recommendation |
| Quantization-error-derived signal (per-dimension coordinate range) | Untested; more directly measures what breaks under quantization |
| Direct `ruvector-mincut` conductance query | Untested; production-grade signal, more expensive than the mutual-kNN proxy tested here |
| Product quantization with mixed codebook sizes | Untested; different quantization error model than min-max scalar |

---

## Consequences

### What the merge provides

- A self-contained, zero-dependency Rust harness (deterministic clustered dataset generation,
  real bit-packed 4/8-bit scalar quantization, brute-force mutual-kNN coherence scoring,
  recall/memory/latency benchmark) usable as a baseline for future quantization-signal experiments.
- A validated (but non-transferable) structural signal: mutual-kNN coherence reliably separates
  clustered from random data, documented as a building block for other uses even though it fails
  this specific application.
- Documented failure mode (weak signal transfer from neighbourhood structure to quantization
  sensitivity) that future work can avoid by testing quantization-error-derived signals directly.

### Costs / trade-offs measured

- Coherence graph build (k=12, brute-force k-NN over N=4,000): ~800ms one-time cost, amortized at
  index build time, excluded from per-query latency.
- Candidate B search latency (354µs mean) sits between baseline (258µs) and candidate A (399µs) —
  expected, since 4-bit dequantization is marginally cheaper per vector than 8-bit, and candidate B
  is a majority-4-bit mix.

### If this is ever revisited

1. Replace the neighbourhood-mutuality signal with a per-vector quantization-error estimate
   (coordinate range, local intrinsic dimensionality) that more directly predicts rank-order
   sensitivity.
2. If a working signal is found, validate it against the real `ruvector-mincut` conductance API
   rather than the standalone mutual-kNN proxy used here.
3. Retest at production embedding dimensionality (384+), where per-dimension quantization noise
   averages differently than at dim=32.
4. Always report the uniform-4-bit and uniform-8-bit bracket as permanent benchmark columns, as
   done here, so any future "adaptive" claim is falsifiable against both ends.

---

## Implementation Status

**PoC**: `crates/ruvector-coherence-quant` v0.1.0 — merged as negative result
**Tests**: 16 assertions, all pass (`cargo test --release -p ruvector-coherence-quant`)
**Clippy**: clean (`cargo clippy --release -p ruvector-coherence-quant --all-targets`)
**Benchmark**: `cargo run --release -p ruvector-coherence-quant --bin benchmark`

No production integration is planned.

---

## Security

No new attack surface: the crate is a standalone benchmark harness with no network I/O, no
external input parsing beyond in-process generated data, and is not wired into any production
index or MCP surface.

## Governance

No mutation authority, no write-path changes, no witness/provenance implications — this is a
read-only research benchmark.

## Migration / Rollback

N/A — not integrated into any production path. Rollback is deleting the crate and its workspace
member entry, which the harness has not recommended.

## Rejection Criteria

Already applied: this ADR documents the rejection itself, per the pre-registered acceptance gates
in the Hypothesis section.

## Open Questions

1. Does a quantization-error-derived signal (rather than a neighbourhood-structure signal) predict
   recall sensitivity well enough to clear the same gates?
2. Does the production `ruvector-mincut` conductance API behave differently from the mutual-kNN
   proxy tested here, given it captures more than 1-hop neighbourhood structure?
3. Does the relationship change at production embedding dimensionality (384+) where distance
   concentration effects differ from dim=32?

---

## References

- Crate: `crates/ruvector-coherence-quant`
- `ruvector-mincut` — subpolynomial dynamic min-cut / conductance engine (untested integration path)
- ADR-303, `docs/research/nightly/2026-08-13-entropy-adaptive-ann` — prior negative result with a
  structurally similar lesson (a real, measurable signal that does not transfer to the specific
  downstream metric it was hypothesised to predict)
- Research README: `docs/research/nightly/2026-08-15-coherence-adaptive-quant/README.md`
