# ADR-353: Frontier-Cut Conductance as an Adaptive ANN Beam-Width Signal

**Date**: 2026-10-18
**Status**: Closed — negative result (documented; not recommended for production)
**Deciders**: Nightly research agent (autonomous run, reviewed against measured output)
**Tags**: ann, hnsw, mincut, conductance, beam-search, ruvector-frontier-cut-ann, negative-result

---

## Context

The 2026-08-13 nightly (ADR-303, `ruvector-entropy-ann`) tested whether the Shannon entropy of the
result-heap distance distribution could drive per-query adaptive `ef_search` for HNSW-style beam
search. It was refuted: softmin entropy over already-retrieved neighbour distances tracks local
point density, not routing ambiguity, and has the wrong sign on the benchmark data (hard queries
showed *lower* entropy than easy ones).

That failure is specific to a **distance-based** signal computed over the result heap. RuVector
already has graph-cut/conductance machinery elsewhere in the repository — `ruvector-mincut`'s
local k-cut and sparsify tooling computes exactly this family of object (sparsest cut /
conductance) for graph partitioning. This nightly asks: does a **topological** signal — the
fraction of a search frontier's edges that cross into unvisited territory, a live conductance
proxy — avoid the entropy signal's wrong-sign failure and provide genuine per-query adaptivity?

---

## Decision

**Do not adopt frontier-cut conductance gating in this PoC form.** Two variants were implemented
and measured on the same dataset, graph, baseline, and matched-budget methodology as ADR-303 so
the two signals are directly comparable. Neither delivers a production-worthy result. The crate is
merged as a second documented negative result plus a reusable conductance-probe utility.

### Signal definition

During a `probe_depth`-step beam expansion from the brute-force entry point, every k-NN edge
examined is classified as it is encountered:

```
cut_edges      += 1  if the neighbour has not yet been visited (crosses the frontier)
internal_edges += 1  if the neighbour is already in the visited set

phi = cut_edges / (cut_edges + internal_edges)
```

Hypothesis: inside a tight cluster, repeated k-NN edges fold back onto already-visited nodes as
the frontier grows (`phi` low → "easy", nearly closed-off region). Near a cluster boundary, the
frontier keeps discovering new territory (`phi` high → "hard", needs a larger beam).

### Strategy A — `CutThresholdBeam` (reactive)

Stop expanding once a sliding-window `phi` (recomputed every `window_steps` expansions, not
cumulative-since-start, to avoid a monotonic-decay confound) falls below `phi_stop`.

**Measured**: `recall@10 = 0.799` on easy queries, **below** the `0.80` acceptance threshold, in
all three repeated runs. The gate fires too early relative to the recall bar on this dataset. The
acceptance test for this variant **fails**.

### Strategy B — `FrontierCutScaledEf` (predictive)

Probe `phi` for `probe_depth=5` steps at `base_ef`, then scale:

```
ef_actual = base_ef * clamp(1 + alpha * phi, ef_min_factor, ef_max_factor)
```

**Measured** (mixed query set, `base_ef=50`, `alpha=1.5`):

| Query set | mean `ef_actual` | min | max |
|-----------|------------------:|----:|----:|
| easy      | 97.2 | 94  | 103 |
| hard      | 97.1 | 84  | 109 |
| mixed     | 96.7 | 85  | 110 |

Easy and hard queries receive **statistically indistinguishable** mean budgets (97.2 vs 97.1,
difference < 2). The computed matched-budget control, `FixedEf(97)`, reproduces
`FrontierCutScaledEf`'s mixed-set recall almost exactly (0.7202 vs 0.7220, Δ = +0.0018 — within
measurement noise, and nowhere near the ADR-303 `h_stop`/entropy gap). The recall gain over
`FixedEf(50)` (+0.0248) is explained by the larger average budget, not by the signal.

---

## Why the signal fails (measured) — and why it fails *differently* than entropy

The raw `phi` distribution itself is nearly flat across difficulty classes:

| Query set | phi mean | phi p50 | phi p95 |
|-----------|---------:|--------:|--------:|
| easy      | 0.639 | 0.637 | 0.700 |
| hard      | 0.634 | 0.637 | 0.738 |
| mixed     | 0.630 | 0.637 | 0.750 |

This is a **different failure mode than ADR-303's entropy signal**:

- Entropy saturated to a *constant maximum* after clamping (wrong sign, but non-trivial variance
  before the clamp) — a *miscalibrated* signal.
- Frontier-cut `phi` over a short probe window is close to **uninformative** before any clamping
  is even applied: a `probe_depth=5` / `k_neighbours=16` window is dominated by local graph degree
  statistics that are roughly constant everywhere in this flat k-NN graph, regardless of whether
  the query sits near a cluster centroid or at a boundary. Five expansion steps is not enough
  traversal distance for cluster-boundary structure to show up as a measurably different
  cut-to-internal edge ratio — the window closes on roughly the same fraction of "fresh" neighbours
  whether or not the local region is a tight cluster or a sparse boundary.

In short: entropy had the wrong sign; frontier-cut conductance (at this probe depth, on this
graph) carries almost no signal at all. A deeper probe might reveal real structure, but a deeper
probe spends exactly the query-time budget this technique is trying to save, which is why this ADR
does not recommend simply increasing `probe_depth` without re-running the falsifiability check.

---

## Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| Fixed ef (status quo) | Remains the recommendation; matched-budget FixedEf(97) equals FrontierCutScaledEf |
| Heap-distance entropy (ADR-303) | Already refuted — wrong sign |
| Offline precomputed spectral coherence (`ruvector-coherence::spectral`, `HnswHealthMonitor`) | Not tested here; moves the cost from query-time probing to build-time/background computation — plausible next step, see "Future Work" |
| Deeper online probe (`probe_depth` > 5) | Would spend the latency budget being optimized away; not attempted without re-validating falsifiability |
| Per-step phi re-evaluation (no fixed window) | Rejected a priori: adds overhead on every step for a signal already shown to be near-flat in a 5-step window |

---

## Consequences

### What the merge provides

- A second documented negative result using a genuinely different (topological, not distance-based)
  signal family, directly comparable to ADR-303 via the same dataset/graph/baseline/matched-budget
  methodology.
- A reusable `probe_phi` conductance-probe utility (`crates/ruvector-frontier-cut-ann/src/cut.rs`)
  and a `predicted_ef` inspection method for future adaptive-ef work to measure adaptivity *before*
  benchmarking recall/latency, catching saturation early.
- A connection, made explicit and then empirically closed off, between `ruvector-mincut`'s
  local-cut/conductance tooling and live ANN traversal control — this nightly's measurement is
  evidence *against* applying raw local conductance at short horizons, which future work on
  mincut-assisted retrieval should not have to rediscover.

### Costs / trade-offs measured

- `FrontierCutScaledEf` costs ~80–95 µs/query vs ~65–70 µs for `FixedEf(50)` on this dataset (the
  larger average ef budget, not the signal, is responsible).
- `CutThresholdBeam` is faster (~57–72 µs/query) but at a recall cost that fails the easy-query
  acceptance threshold — not a safe trade on this benchmark.

### If this is ever revisited

1. Use a precomputed, offline structural coherence score (e.g. `ruvector-coherence`'s spectral gap
   / Fiedler estimate, already implemented behind the `spectral` feature) instead of a live,
   shallow online probe — paying the cost once at build/background time rather than per query.
2. If staying online, measure `phi` at a probe depth deep enough to show measurable separation
   between easy/hard query sets *before* committing to a benchmark — this ADR's adaptivity check
   (`easy`/`hard` mean `ef_actual` differing by < 2) is the exact failure signature future work
   should screen for first.
3. Always report the computed matched-budget `FixedEf` control, as both this ADR and ADR-303 do.

---

## Implementation Status

**PoC**: `crates/ruvector-frontier-cut-ann` v0.1.0 — merged as negative result
**Depends on**: `ruvector-entropy-ann` (path dependency — reuses its `FlatGraph`, dataset
generators, `FixedEfSearch` baseline, `Searcher` trait, and `recall_at_k` so both nightlies'
numbers are measured under identical conditions)
**Tests**: 9 assertions, all pass (`cargo test --release -p ruvector-frontier-cut-ann`)
**Benchmark**: `cargo run --release -p ruvector-frontier-cut-ann --bin benchmark` — computes the
matched-budget `FixedEf` control from the measured mean `ef_actual` rather than a hardcoded
constant

No production integration is planned.

---

## References

- ADR-303 — Entropy-Adaptive Beam Search for ANN Graph Traversal (refuted heap-entropy signal)
- `crates/ruvector-mincut` — local k-cut / sparsify / dynamic mincut tooling this nightly borrows
  the conductance concept from
- `crates/ruvector-coherence` (`spectral` feature) — `estimate_fiedler`, `estimate_spectral_gap`,
  `HnswHealthMonitor` — the precomputed-signal alternative identified in "Future Work"
- Research README: `docs/research/nightly/2026-10-18-frontier-cut-ann/README.md`
