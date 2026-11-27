# A graph-conductance signal for adaptive ANN beam search — and why it also fails

## Problem

HNSW-style approximate nearest-neighbour (ANN) search uses a fixed `ef_search` budget, chosen
offline, that controls how many candidates the beam search considers. This is a blunt instrument:
queries near a tight cluster centroid are "easy" and get over-searched; queries near a cluster
boundary or genuinely out-of-distribution are "hard" and get under-searched. A per-query,
zero-calibration runtime signal that tells the search how hard the current query is would let
`ef_search` scale itself. Finding that signal is an open problem.

A previous experiment in this codebase tried using the Shannon entropy of the result heap's
distance distribution as that signal and found it refuted: entropy over *retrieved* neighbour
distances tracks local point density, not query difficulty, and had the wrong sign on the test
data (hard queries showed lower entropy than easy ones).

## Hypothesis

Entropy is a *distance*-based signal. This experiment asks whether a *topological* signal fares
better: **frontier-cut conductance**, the fraction of a search frontier's edges that cross into
unvisited graph territory versus fold back into already-visited nodes. This is the same
mathematical object (sparsest cut / conductance) used in graph partitioning, computed instead as a
live, incremental probe during beam search:

```
phi = cut_edges / (cut_edges + internal_edges)
```

Hypothesis: inside a tight cluster, a growing frontier quickly starts folding back onto itself
(low `phi`, "easy"). Near a boundary, the frontier keeps finding new territory (high `phi`,
"hard"). Unlike entropy, this signal doesn't depend on distances at all, so it shouldn't inherit
entropy's wrong-sign failure.

## Technical design

Two variants, implemented in Rust (`crates/ruvector-frontier-cut-ann`), reusing the exact same
synthetic dataset generator, flat k-NN graph, `FixedEf` baseline, and benchmark methodology as the
prior entropy experiment so the two signals are directly comparable:

- **`CutThresholdBeam`** (reactive): stop expanding once a sliding-window `phi` — recomputed every
  few steps, not cumulative, to avoid a trivial decay-over-time artifact — drops below a threshold.
- **`FrontierCutScaledEf`** (predictive): probe `phi` for 5 expansion steps, then scale
  `ef_actual = base_ef * clamp(1 + alpha * phi, min, max)` for the rest of the search.

The benchmark computes a **matched-budget control**: a plain `FixedEf` run at the exact mean
budget `FrontierCutScaledEf` actually chooses, computed from the live measurement rather than
hardcoded. If the adaptive variant can't beat that control, its apparent gain is just "bigger
budget," not "smarter budget" — the same test that refuted the entropy signal.

## Actual implementation and benchmark evidence

Real `cargo run --release` output, three repeated runs, N=2000 vectors, D=16, 10 clusters:

```
easy     ef_actual — mean=97.2  min=94  max=103
hard     ef_actual — mean=97.1  min=84  max=109
mixed    ef_actual — mean=96.7  min=85  max=110

easy     phi — mean=0.639  p50=0.637  p95=0.700
hard     phi — mean=0.634  p50=0.637  p95=0.738
mixed    phi — mean=0.630  p50=0.637  p95=0.750

FrontierCutScaledEf recall=0.7220 vs matched-budget FixedEf(97) recall=0.7202  (delta=+0.0018)
CutThreshold  easy  recall=0.799  [FAIL — below 0.80 acceptance threshold, reproduced in all 3 runs]
```

Easy and hard queries get statistically indistinguishable mean budgets (97.2 vs 97.1) and
indistinguishable `phi` (0.639 vs 0.634). The predictive variant's apparent recall edge over the
plain `ef=50` baseline (+0.025) evaporates against the matched-budget control (+0.002, noise). The
reactive variant is faster but fails its own recall bar.

## Limitations

- Tested only at PoC scale (N=2000, D=16) with a flat single-layer graph and brute-force entry —
  not multi-layer HNSW with approximate entry routing.
- A single probe depth (5 steps) and window size (3 steps) were tested. The signal's
  near-zero separation at this depth is itself informative (see below) but a depth sweep was not
  performed in this run, to avoid tuning parameters until the result looked better — that would
  have violated the "don't change the hypothesis after seeing results" discipline this project
  follows.

## Why it fails — and why that's informative

A 5-step probe on a graph with k=16 neighbours per node is dominated by local vertex-degree
statistics that are roughly constant everywhere in this graph, independent of whether the query
sits near a cluster centroid or at a boundary. Cluster-level topology — the thing that should make
`phi` differ by query difficulty — doesn't show up in such a short, local window. A deeper probe
might reveal it, but a deeper probe spends exactly the query-time latency this technique is trying
to save, so simply increasing probe depth isn't a free fix.

This is a *different* failure mode than the entropy signal's wrong sign: here the raw signal
itself carries almost no information before any clamping or scaling is applied.

## Production relevance

None directly — the status quo (fixed, offline-tuned `ef_search`) remains the recommendation. The
production relevance is negative-knowledge: a plausible, principled next idea for adaptive ANN
control has now been implemented and measured, not just speculated about, and the measurement
rules it out at this probe depth. Future engineers (human or agentic) working on this codebase's
ANN layer do not need to re-discover this; the ADR and this gist record it.

## RuVector ecosystem implications

This repository already has a local k-cut / sparsify / dynamic-mincut crate
(`ruvector-mincut`) used for offline graph partitioning, and a spectral-coherence crate
(`ruvector-coherence`, behind a `spectral` feature) that computes Fiedler-vector-based structural
scores — also offline/precomputed. This experiment's result is a specific, measured argument for
pursuing the *precomputed* structural-signal direction next, rather than a second live-probing
variant: a short, live, per-query conductance probe doesn't carry enough information, but a
background-computed, amortized structural score might, since it can afford to look much further
than 5 steps without charging that cost to every query.

## Future direction

1. Evaluate `ruvector-coherence`'s existing spectral/Fiedler tooling as a precomputed, per-region
   adaptive-ef signal instead of a live probe.
2. If live probing is revisited, run the adaptivity pre-check (does the signal differ between easy
   and hard queries at all?) across a depth sweep *before* running a full latency/recall benchmark,
   to avoid spending benchmark cycles on a signal that hasn't cleared the first bar.
3. Test real local k-cut computation (not this proxy) computed offline at index-build time.

## References

- Prior experiment (same codebase): entropy-based adaptive beam width, refuted for the same class
  of reason (budget, not signal) via a different root cause (wrong sign vs. no signal).
- `crates/ruvector-mincut`, `crates/ruvector-coherence` — the two existing pieces of machinery this
  experiment connects to and the one it points toward next.
