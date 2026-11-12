# ADR-303: Entropy-Adaptive Beam Search for ANN Graph Traversal

**Date**: 2026-08-13  
**Status**: Closed — negative result (documented; not recommended for production)  
**Deciders**: Nightly research agent (revised after measured review)  
**Tags**: ann, hnsw, entropy, beam-search, ruvector-entropy-ann, negative-result

---

## Context

RuVector's agent memory subsystem uses approximate nearest-neighbour (ANN) search to retrieve
semantically relevant memories during inference. Current production HNSW uses a fixed `ef_search`
parameter. This creates a systematic tension:

- Easy queries (near a stored cluster centroid) over-search: the result heap fills correctly after
  ~10 expansion steps but HNSW continues for the full `ef` budget.
- Hard queries (between memory clusters, or genuinely novel) under-search: the budget runs out
  before all true nearest neighbours are reached.

The correct `ef` value is per-query, yet HNSW provides no runtime signal to choose it.

---

## Decision

**Do not adopt entropy-gated beam width in its PoC form.** The PoC implemented a
Shannon-entropy gate on the candidate-heap distance distribution as a live beam-width control
signal, in two strategies. Both were measured and neither delivers per-query adaptivity on the
benchmark data. The crate is merged as a documented negative result and a baseline harness for
future work.

### Strategy A — EntropyThresholdBeam (reactive)

At each beam-expansion step, compute the entropy *H* of the result-heap distances using softmin
at temperature *T*. Stop traversal when *H* drops below a threshold *h_stop*:

```
p_i = exp(-d_i / T) / Z
H   = -Σ p_i ln(p_i)
if H < h_stop: stop
```

**Measured**: the gate never fires. With brute-force entry the heap distribution is near-uniform
for every query (H ≈ ln(heap_size)), so the variant adds per-step entropy overhead with recall
identical to FixedEf.

### Strategy B — EntropyScaledEf (predictive)

Run a fixed `probe_depth` expansion steps at `base_ef`. Compute *H* of the probe result set once.
Scale `ef` for the remainder of the search:

```
scale     = clamp(1 + α · H / ln(|results|), ef_min_factor, ef_max_factor)
ef_actual = base_ef * scale
```

**Measured**: because *H* ≈ ln(|results|) for every query, the scale saturates at
`ef_max_factor` and `ef_actual` = 122–124 for **every** query (base_ef=50, alpha=1.5,
ef_max_factor=2.5). There is no per-query adaptivity. A plain `FixedEf` baseline at the matched
budget (ef=124) reproduces EntropyScaledEf's recall to four decimal places on all three query
sets (easy/hard/mixed). The earlier reported +1.6–3.9 pp recall gain over FixedEf(50) is
entirely explained by the ~2.5× larger ef budget, not by entropy. EDEN — the cited theoretical
basis — specifically claims entropy-based branching beats fixed branching *within the same
budget*, which is exactly the property this PoC does not exhibit.

---

## Why the signal fails (measured)

The hypothesis was that heap-distance entropy encodes routing ambiguity:

- Uniform distribution (high *H*): candidates equidistant → query at a cluster boundary → expand.
- Peaked distribution (low *H*): one cluster dominates → converged → stop.

The measurements refute this on the PoC data:

1. **Wrong quantity**: softmin entropy over *already-retrieved* neighbour distances measures the
   local density of the neighbourhood the search has landed in, not the ambiguity of routing to it.
2. **Wrong sign**: at every temperature tested, easy/hard separation is *negative* — hard
   (out-of-distribution) queries produce **lower** entropy than easy ones, because their retrieved
   neighbourhoods are locally denser relative to *T*. A controller built on this signal would
   shrink the beam exactly when it should grow it.
3. **Temperature range**: T=0.1 is effectively infinite temperature on this data (all
   probabilities near-uniform, H ≈ ln(n) for every query; observed spread ≈ 0.003 nats). The
   usable range is roughly T ∈ [0.001, 0.01], and even there the separation remains negative.

The idea that multi-layer HNSW (approximate entry points) would restore a usable, correctly-signed
entropy signal is **untested conjecture** — it is a hypothesis for future work, not a conclusion
supported by any measurement in this PoC.

---

## Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| Fixed ef (status quo) | Remains the recommendation; matched-budget FixedEf equals EntropyScaledEf |
| Scalar distance threshold (DAB, arXiv:2505.15636) | Untested here; scalar signal, but at least directionally validated in its paper |
| Offline-predicted ef (Ada-ef, arXiv:2512.06636) | Requires training pipeline, drifts with index |
| Per-step ef adjustment | Too noisy; adds overhead on every step |

---

## Consequences

### What the merge provides

- A self-contained, zero-dependency Rust harness (flat k-NN graph, three search variants,
  deterministic synthetic datasets, recall/latency benchmark) usable as a baseline for future
  adaptive-ef experiments.
- A falsifiable benchmark: the matched-budget `FixedEf(124)` control is a permanent column in the
  benchmark output, so any future "adaptive" claim must beat it.
- Documented failure modes of heap-distance entropy (wrong quantity, wrong sign, temperature
  sensitivity) that future work can avoid.

### Costs / trade-offs measured

- **Latency**: with ground-truth computation correctly excluded from timing, EntropyScaledEf costs
  ~50 µs/query vs ~33 µs for FixedEf(50) on this dataset — the recall difference is purchasable
  more honestly by simply raising ef.
- EntropyThresholdBeam adds ~10–13 µs/query of per-step entropy overhead with zero behavioural
  difference from FixedEf(50).

### If this is ever revisited

1. Test entropy of *candidate frontier* distances (pre-retrieval), not retrieved results.
2. Use multi-layer HNSW with approximate entries — and treat the "signal will emerge" claim as the
   hypothesis under test, since it is currently unvalidated.
3. Calibrate temperature to the dataset's distance scale (T ∈ [0.001, 0.01] on data like this).
4. Always report matched-budget FixedEf controls.

---

## Implementation Status

**PoC**: `crates/ruvector-entropy-ann` v0.1.0 — merged as negative result  
**Tests**: 15 assertions, all pass (`cargo test -p ruvector-entropy-ann`)  
**Benchmark**: `cargo run --release -p ruvector-entropy-ann --bin benchmark` — includes the
matched-budget FixedEf control  

No production integration is planned.

---

## References

- arXiv:2605.09745 — EDEN: Entropy-Driven Efficient Decoding with Adaptive LLM Beam Search
  (ICML 2026) — note: EDEN's claim is *same-budget* improvement, not reproduced here
- arXiv:2505.15636 — Distance Adaptive Beam for HNSW (NeurIPS 2025)
- arXiv:2512.06636 — Ada-ef: Adaptive ef-search for HNSW
- Vespa HNSW adaptive beam (2024): heuristic candidate-list-size adaptation
- Research README: `docs/research/nightly/2026-08-13-entropy-adaptive-ann/README.md`
