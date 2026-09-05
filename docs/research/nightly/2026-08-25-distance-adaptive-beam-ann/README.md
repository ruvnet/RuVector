# Distance-Adaptive Beam Search: A Real Per-Query Signal That Still Loses to a Fixed Budget

**150-char summary:** A (1+γ)·d_k relative-distance stopping rule genuinely varies its per-query
cost — but adapts to local density, not query difficulty, and misses its own matched-budget bar.

**Date:** 2026-08-25
**Crate:** `crates/ruvector-dab-search`
**ADR:** [ADR-340](../../../adr/ADR-340-distance-adaptive-beam-search.md)
**Follows:** [2026-08-13-entropy-adaptive-ann](../2026-08-13-entropy-adaptive-ann/README.md) (ADR-303)

---

## Abstract

ADR-303 asked whether a live, per-query signal derived from the search itself could replace HNSW's
fixed `ef_search` budget, and measured a clean negative result: Shannon entropy of the candidate
heap saturates to a constant for every query on that PoC's data, so the "adaptive" variant's recall
gain was just a bigger budget in disguise. That work's own prior-art table cited "Distance Adaptive
Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search" (arXiv:2505.15636, a real,
theoretically-grounded 2025 result with a proved approximation guarantee on navigable graphs) as an
alternative — but never implemented it.

This nightly does. It implements the paper's exact stopping inequality,
`d(q,x) >= (1+γ)·d(q,x_k)`, on the same dataset and graph construction ADR-303 used, and subjects it
to the same discipline that caught the entropy signal's flaw: a mandatory matched-budget control.

**Result: REJECT**, but with three pieces of genuinely useful evidence:

1. Unlike entropy, this signal **does** vary substantially per query (distance-computation stddev of
   96–172, vs FixedEf's 19–61 at comparable means) — it is a real per-query control, not a disguised
   constant.
2. It varies in the **wrong direction** for the intended use: hard (out-of-distribution) queries cost
   *less* work than easy (cluster-core) queries — ratio 0.915, not the hypothesized >=1.15. It tracks
   local point density, not task difficulty — a different mechanism from ADR-303's entropy signal,
   but landing on the same kind of confound.
3. It **narrowly misses** its own pre-registered matched-budget bar (+0.014 recall advantage on hard
   queries at matched cost, vs a required +0.02) and **loses** on the headline cost-at-matched-recall
   metric (-6.6%, the opposite of the source paper's reported 10–50% *reduction* on real embedding
   benchmarks).

All numbers are from `cargo run --release -p ruvector-dab-search --bin benchmark` on the hardware
below. Raw output is reproduced verbatim in [Benchmark Results](#benchmark-results).

**Hardware:** x86-64, 4 logical CPUs, Linux 6.18.44, `rustc` release build.

---

## Hypothesis

```text
Given a 2,000-vector synthetic corpus at dimension 16, clustered into 10 groups (identical
construction to ADR-303's benchmark), indexed by a single-layer k-NN proximity graph
(k=16 neighbours/node) with query-time entry routing through 40 deterministic seed nodes,

when beam-search traversal uses the distance-adaptive stopping rule
d(q,x) >= (1+gamma) * d(q,x_k) (gamma=0.5, pre-registered before this dataset was benchmarked)
instead of a fixed ef_search budget,

then (1) the rule's per-query work should vary measurably more on hard queries than easy queries
(hard/easy distance-computation ratio >= 1.15),

and (2) recall@10 should stay within 3 points of a FixedEf(100) high-recall reference on every
query set,

and (3) on hard queries specifically, it should beat a FixedEf baseline whose ef is calibrated (on
the disjoint mixed-query set) to match its own average distance-computation budget, by >= 2 recall
points.

ACCEPT requires all three. REJECT if (1) or (2) fails. INCONCLUSIVE if only (3) fails.
```

**Result: REJECT** — (1) failed (ratio 0.915, wrong direction), (2) passed, (3) failed narrowly
(+0.014 vs +0.02 required). Because (1) failed outright, not just (3), the pre-registered logic
calls this REJECT rather than INCONCLUSIVE — see [Acceptance Result](#acceptance-result).

**What this does NOT claim:** that arXiv:2505.15636's method fails in general. It is validated on
SIFT1M/DEEP/MNIST/GloVe/GIST with real, incrementally-built navigable graphs (HNSW/Vamana/NSG/
EFANNA). This PoC's graph is a flat exact-k-NN graph over a small synthetic corpus, not proven
navigable — see [Why This Result May Not Transfer](#why-this-result-may-not-transfer-to-production-hnsw).

---

## Why This Matters for RuVector

RuVector's agent-memory retrieval path uses ANN search with a fixed `ef_search`, the same tension
ADR-303 named: easy queries over-search, hard queries under-search, and there is no free per-query
signal to fix it without calibration. This nightly connects:

1. **Vector search** — the graph-traversal stopping rule under test.
2. **Agent memory** — the intended consumer (semantically ambiguous memory queries mixing easy and
   hard cases in the same workload).
3. **Prior nightly research (Flywheel-style evidence retention)** — this experiment exists only
   because ADR-303's prior-art table recorded an untested citation; the negative result here is now
   itself retained evidence for whoever attempts adaptive stopping next (see
   [Lessons for Future Attempts](#lessons-for-future-attempts)).
4. **MetaHarness / nightly research process** — a second consecutive rejection of a per-query
   stopping signal on the same dataset is a stronger, more specific finding than either rejection
   alone: it suggests the synthetic dataset generator itself may not separate "hard" from "dense" in
   a way any purely local, distance-based signal can exploit (see Open Questions in the ADR).
5. **Darwin-style bounded evolution (conceptual)** — γ was swept over {0.2, 0.5, 1.0} as exploratory
   context; the pre-registered γ=0.5 result is what is reported as the finding, not the best
   post-hoc value across the sweep (see [Gamma Sweep](#gamma-sweep-exploratory)), to avoid exactly
   the kind of cherry-picking a bounded-evolution promotion gate must reject.

---

## Architecture

```mermaid
flowchart TD
    Q[Query vector] --> ROUTE[Route via 40 entry seeds<br/>O(seeds), not O(n)]
    ROUTE --> ENTRY[Entry node]
    ENTRY --> FRONTIER[Min-heap frontier<br/>closest-first]
    FRONTIER -->|pop closest| CHECK{Stopping rule}
    CHECK -->|FixedEf: results.len&gt;=ef<br/>and current.dist &gt; worst| STOP[Stop]
    CHECK -->|AdaptiveGamma: results.len&gt;=k<br/>and current.dist &gt;= (1+gamma)*d_k| STOP
    CHECK -->|continue| EXPAND[Expand neighbours<br/>update top-k result heap]
    EXPAND --> FRONTIER
    STOP --> RESULT[Top-k hits + dist_computations count]
```

The graph (`FlatGraph`) is an exact per-node k-NN graph — the same construction ADR-303 used — built
once (`O(n^2 * dim)`), with entry routing (`entry_seeds`) as the one deliberate departure from
ADR-303's design, explained next.

### Why not ADR-303's brute-force entry point

ADR-303 finds each query's entry node by an O(n) brute-force scan, explicitly to remove
entry-quality as a variable while studying beam width. That is wrong for *this* experiment: at
N=2,000 the O(n) entry scan is an order of magnitude larger than any traversal-cost difference this
crate measures (tens to low hundreds of distance computations), so it would swamp exactly the signal
under test.

### Why not a single fixed entry point (and how that failure was caught)

The first implementation of this crate used a single fixed entry point (the node nearest the corpus
centroid, computed once at build time) instead. It measured ~19% recall across every variant and
every γ — because an exact k-NN graph over well-separated clusters has few or no edges *between*
clusters, so one fixed entry point can only reach the fraction of the corpus in its own cluster (with
10 clusters, ~19% recall is consistent with reaching roughly one cluster plus adjacent-cluster
noise-overlap). This was caught by the crate's own test suite
(`loose_gamma_achieves_high_recall_on_majority_of_self_queries` failed at 19% against an 80%
threshold) before any benchmark numbers were trusted — exactly the kind of thing an attack pass is
supposed to catch. The fix, `entry_seeds`, is documented in [graph.rs](../../../../crates/ruvector-dab-search/src/graph.rs)
and kept in ADR-340 as a reusable pitfall for future flat-graph PoCs in this repository.

---

## Implementation

Three `Searcher` implementations, matching the repository's required baseline/candidate-A/candidate-B
shape:

| Variant | Role | Stopping rule | Result-heap capacity |
|---|---|---|---|
| `FixedEf` | Baseline | `results.len() >= ef && current.dist > worst_result` | `ef_search` (tunable, `>= k`) |
| `AdaptiveGamma` (uncapped) | Candidate A | `results.len() >= k && current.dist >= (1+γ)·d_k` | `k` (no separate ef) |
| `AdaptiveGamma` (capped) | Candidate B | Same rule, plus a hard `max_expansions` safety bound | `k` |

Every `search()` call returns a `SearchOutcome{ hits, dist_computations, expansions }` — the crate
counts real `l2sq` calls, not a proxy, so "distance computations" in every table below is an exact
count, not an estimate.

No external dependencies (matches ADR-303's convention): the deterministic dataset generator uses
the same fixed-seed LCG.

---

## Benchmark Methodology

- **Release build**, `opt-level = 3`, `lto = "thin"`.
- **Deterministic seeds** throughout: corpus seed 42, easy-query seed 101, hard-query seed 202,
  mixed-query seed 303, entry-seed sampling seed fixed in `graph.rs` — reruns reproduce identical
  numbers (verified: two independent runs in this nightly produced identical dist_comp/recall
  figures to the printed precision).
- **Ground-truth computation excluded from timed sections** — the brute-force scan used to compute
  recall is not part of the measured search latency.
- **Query sets**: `easy` (tight clusters, noise=0.02), `hard` (uniform random unit vectors — maximally
  out-of-distribution), `mixed` (same noise as the corpus, different seed) — identical construction
  to ADR-303's three-way split.
- **Matched-budget calibration** (Test 3) is done via linear scan of `FixedEf`'s `ef` on the *mixed*
  query set only, then applied to the *hard* set for the actual test — deliberately avoiding
  calibrating on the same set the test measures, to prevent the calibration procedure itself from
  leaking into the result it's supposed to control for.
- **γ=0.5 was pre-registered** as the primary value (paper's valid range is `(0, 2]`; 0.5 was chosen
  as roughly the paper's own the mid-low working range before any run on this dataset). The
  {0.2, 1.0} sweep is reported as exploratory context and does not change which number is reported
  as the finding.

---

## Benchmark Results

Verbatim output from `cargo run --release -p ruvector-dab-search --bin benchmark`:

```text
=== Distance-Adaptive Beam (DAB) Search Benchmark ===

OS:           linux / x86_64
Rust:         (see: rustc --version)
CPU threads:  4

Dataset (identical construction to ADR-303):
  N (corpus) : 2000
  Dimensions : 16
  Clusters   : 10  noise=0.2
  k (recall) : 10
  Graph K    : 16
  gamma      : primary=0.5, sweep=[0.2, 1.0]

Building corpus...
  corpus built in 0ms
Building flat graph (k=16)...
  graph built in 236ms, entry_seeds=40

─── Recall / Work / Latency by variant and query set ───

  FixedEf(50)              easy     n=200   recall=0.811  dist_comp(mean= 215.3 sd= 33.1 min= 192 max= 337)  lat_mean=  34.3us  29145 qps
  FixedEf(50)              hard     n=200   recall=0.706  dist_comp(mean= 218.1 sd= 37.0 min= 188 max= 443)  lat_mean=  36.9us  27056 qps
  FixedEf(50)              mixed    n=400   recall=0.635  dist_comp(mean= 215.9 sd= 28.5 min= 185 max= 384)  lat_mean=  41.1us  24315 qps
  FixedEf(100)             easy     n=200   recall=0.846  dist_comp(mean= 276.8 sd= 60.0 min= 226 max= 374)  lat_mean=  65.4us  15280 qps
  FixedEf(100)             hard     n=200   recall=0.722  dist_comp(mean= 259.6 sd= 58.7 min= 222 max= 546)  lat_mean=  62.4us  16008 qps
  FixedEf(100)             mixed    n=400   recall=0.663  dist_comp(mean= 256.3 sd= 53.8 min= 222 max= 530)  lat_mean=  59.4us  16822 qps
  Adaptive(g=0.5)          easy     n=200   recall=0.903  dist_comp(mean= 346.7 sd=153.6 min= 228 max= 608)  lat_mean=  84.4us  11834 qps
  Adaptive(g=0.5)          hard     n=200   recall=0.756  dist_comp(mean= 317.3 sd=130.4 min= 214 max= 635)  lat_mean=  80.6us  12402 qps
  Adaptive(g=0.5)          mixed    n=400   recall=0.678  dist_comp(mean= 291.5 sd= 95.7 min= 191 max= 634)  lat_mean=  68.3us  14632 qps
  Adaptive(g=0.5,cap=40)   easy     n=200   recall=0.801  dist_comp(mean= 198.0 sd= 19.0 min= 176 max= 274)  lat_mean=  26.7us  37344 qps
  Adaptive(g=0.5,cap=40)   hard     n=200   recall=0.693  dist_comp(mean= 202.2 sd= 23.2 min= 175 max= 332)  lat_mean=  33.9us  29474 qps
  Adaptive(g=0.5,cap=40)   mixed    n=400   recall=0.625  dist_comp(mean= 200.6 sd= 18.5 min= 174 max= 312)  lat_mean=  33.4us  29848 qps
  Adaptive(g=0.2)          easy     n=200   recall=0.809  dist_comp(mean= 208.8 sd= 39.0 min= 168 max= 351)  lat_mean=  32.4us  30769 qps
  Adaptive(g=0.2)          hard     n=200   recall=0.714  dist_comp(mean= 219.4 sd= 51.3 min= 164 max= 555)  lat_mean=  37.2us  26857 qps
  Adaptive(g=0.2)          mixed    n=400   recall=0.632  dist_comp(mean= 211.4 sd= 37.6 min= 142 max= 420)  lat_mean=  30.7us  32564 qps
  Adaptive(g=1.0)          easy     n=200   recall=0.917  dist_comp(mean= 447.6 sd=171.3 min= 238 max= 637)  lat_mean= 172.9us  5780 qps
  Adaptive(g=1.0)          hard     n=200   recall=0.767  dist_comp(mean= 401.2 sd=172.7 min= 238 max= 637)  lat_mean= 136.5us  7323 qps
  Adaptive(g=1.0)          mixed    n=400   recall=0.690  dist_comp(mean= 382.8 sd=161.9 min= 237 max= 637)  lat_mean= 147.8us  6764 qps

  Index memory: 375 KB

─── Test 1: Does the stopping rule actually adapt per query? ───
  Adaptive(g=0.5): mean dist_comp easy=346.7 hard=317.3 ratio(hard/easy)=0.915 (threshold >= 1.15)
  Contrast — ADR-303 measured EntropyScaledEf's ef_actual at 122-124 for EVERY query (ratio ~= 1.00), which is why it was rejected. This test is the same question asked of a different signal.
  [FAIL]

─── Test 2: Recall floor vs FixedEf(100) reference ───
  easy   reference=0.846 adaptive=0.903 delta=+0.057 (floor: adaptive >= reference - 0.03) [PASS]
  hard   reference=0.722 adaptive=0.756 delta=+0.033 (floor: adaptive >= reference - 0.03) [PASS]
  mixed  reference=0.663 adaptive=0.678 delta=+0.015 (floor: adaptive >= reference - 0.03) [PASS]

─── Test 3: Matched-budget control (crux test) ───
  Calibrated on MIXED set only: FixedEf(ef=150) has mean dist_comp=291.0 (target from Adaptive(g=0.5) on mixed = 291.5)
  On HARD queries at ~matched average budget: Adaptive(g=0.5) recall=0.756 vs FixedEf(150,matched) recall=0.741  advantage=+0.014 (threshold >= 0.02)
  This is the test ADR-303 could not pass: does adaptively reallocating budget toward harder queries beat a flat allocation at the same average cost?
  [FAIL]

─── Headline: cost at matched recall (arXiv:2505.15636's own metric) ───
  On MIXED queries at matched recall (0.678): FixedEf(ef=122) needs 273.4 dist_comp/query vs Adaptive(g=0.5)'s 291.5 (-6.6% change)

─── Acceptance Result ───
  Test 1 (adapts per query):        FAIL
  Test 2 (recall floor):             PASS
  Test 3 (beats matched budget):     FAIL
  VERDICT: REJECT
```

Reproduced twice; both runs produced identical figures to the printed precision (deterministic
seeds, no floating-point-order nondeterminism observed at this scale).

### Gamma Sweep (exploratory)

Not used to select the reported result — γ=0.5 was fixed in advance. Included because a Pareto view
is informative: larger γ trades more cost for more recall roughly monotonically (γ=0.2: 208.8–219.4
dist_comp, recall 0.632–0.714; γ=1.0: 382.8–447.6 dist_comp, recall 0.690–0.917), and the
hard/easy adaptivity-ratio problem (Test 1) persists at every γ tested — it is not an artifact of the
particular γ=0.5 choice.

---

## Why This Result May Not Transfer to Production HNSW

arXiv:2505.15636's theorem, and its reported 10–50% distance-computation reduction, is stated for
*navigable* graphs (formally: a graph where a greedy walk from any start node monotonically
approaches any target). This PoC's graph is an exact per-node k-NN graph, not an incrementally
constructed HNSW/Vamana/NSG graph, and it is not proven navigable — indeed, the entry-routing
incident above is direct evidence it is not even fully *connected* in the relevant sense without
seed-based routing. A production HNSW graph, built by the standard heuristic-pruned insertion
algorithm, has different degree and connectivity properties by construction. This nightly's result
is therefore evidence about *this specific graph construction*, not a refutation of the source
paper's own reported numbers on real navigable graphs and real embedding datasets — see Open
Questions in [ADR-340](../../../adr/ADR-340-distance-adaptive-beam-search.md#open-questions) for the
concrete next experiment this implies.

---

## Lessons for Future Attempts

1. **A signal having real per-query variance (unlike ADR-303's constant) is necessary but not
   sufficient.** This nightly's signal clears that first bar and still loses on the metric that
   matters (matched-budget recall). Report both, always.
2. **Two different local signals (heap entropy, relative-distance ratio) have now both been observed
   to track local point density rather than task difficulty** on this repository's synthetic
   cluster-plus-noise dataset generator. That is either a property of relative/local signals in
   general on this kind of data, or an artifact of the generator itself (uniform noise within
   clusters, uniform random "hard" queries) — worth testing with a difficulty axis that is
   *independent* of density (e.g. queries at a fixed distance from the nearest cluster centroid,
   rather than uniform random) before trying a third local signal.
3. **Calibrate matched-budget controls on a disjoint query set from the one under test.** This
   nightly calibrated on `mixed` and tested on `hard`, specifically to avoid the calibration itself
   absorbing the effect being measured.

---

## MCP / RVF / RVM / ruFlo / Edge Implications

Given the REJECT verdict, none of these are recommended for integration from this specific PoC. For
completeness, briefly:

- **MCP**: not applicable — no capability is being promoted.
- **RVF/RVM**: not applicable — no portable index format or coherence-domain change is proposed.
- **ruFlo**: the one transferable piece is the *process*: a ruFlo workflow role that runs "test the
  next cited-but-unimplemented alternative from the last rejected nightly" is a well-defined,
  boundable, valuable autonomous task — this nightly is itself an instance of exactly that pattern
  (ADR-303's citation → this nightly's implementation).
- **Edge/WASM**: not evaluated; moot given the REJECT verdict.

---

## Security / Governance

No security-relevant surface is introduced: this is a benchmark-only crate with no I/O, no network
access, and no production integration path. `cargo test` and the benchmark binary are the only
executables. No secrets, credentials, or external data are used.

---

## Practical and Long-Horizon Applications

Not applicable in the standard sense — this is a rejected research direction. The transferable value
is methodological (see [Lessons for Future Attempts](#lessons-for-future-attempts)), not a
capability to deploy.

---

## Falsification Criteria (met)

The hypothesis was falsifiable and was falsified: pre-registered Test 1 (adaptivity direction) and
Test 3 (matched-budget advantage) were specified with numeric thresholds before the benchmark was
run, and both failed as measured. Test 2 (recall floor) passed but is not sufficient alone for
ACCEPT under the pre-registered logic.

---

## Limitations

- Single synthetic dataset (N=2,000, dim=16, 10 clusters); no real embedding dataset was used this
  run (see [Why This Result May Not Transfer](#why-this-result-may-not-transfer-to-production-hnsw)).
- Flat exact-k-NN graph, not an incrementally constructed HNSW/Vamana graph — the source paper's
  navigability assumption is not verified to hold here.
- Single hardware configuration (4 logical CPUs); no multi-thread or SIMD path measured.
- γ sweep limited to 3 values; a finer sweep or an offline-optimal γ was not attempted (would itself
  require an evaluation-leakage-free protocol to avoid p-hacking the reported number).

---

## Next Research

Per [ADR-340](../../../adr/ADR-340-distance-adaptive-beam-search.md#if-this-is-ever-revisited):
test on a real incrementally-built HNSW graph and a real embedding dataset before concluding the
method itself (as opposed to this PoC's graph construction) is not viable for RuVector; and/or test
a minimum-expansion-floor + gamma hybrid to address the under-search-on-sparse-regions direction
found here.

---

## References

- arXiv:2505.15636 — Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest
  Neighbor Search (source of the stopping rule implemented here; NeurIPS 2025)
- ADR-303 / `docs/research/nightly/2026-08-13-entropy-adaptive-ann/README.md` — the prior nightly
  this one directly follows up on
- VBASE (OSDI 2023) — relaxed monotonicity as a related but distinct termination-relaxation idea,
  not implemented here
- Li, Zhang, Andersen, He — "Improving Approximate Nearest Neighbor Search through Learned Adaptive
  Early Termination" (SIGMOD 2020) — a learned-regressor approach to the same problem, not attempted
  this run
