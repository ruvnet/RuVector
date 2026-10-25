# Nightly Research: Sampled Betweenness Centrality as a Fast Alternative to Exact Min-Cut for Agent-Memory Bridge Protection

**Date:** 2026-10-25
**Slug:** `approx-bridge-forgetting`
**ADR:** [ADR-353](../../../adr/ADR-353-approximate-betweenness-bridge-forgetting.md)
**Crate:** `ruvector-agent-memory` (new `approx_bridge` module, always-on; no new feature dependency)
**Acceptance:** **REJECT** (effectiveness hypothesis) — see [Acceptance result](#acceptance-result)

## Summary

The 2026-09-05 nightly (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`,
[ADR-345](../../../adr/ADR-345-mincut-gated-forgetting.md)) tried using
`ruvector-mincut`'s exact global min-cut (`RuVectorGraphAnalyzer`) as a
structural "protect the bridges" signal for agent-memory compaction, and
rejected it on two independent axes: it was 1,800-2,700x slower than the
scalar baseline at only 84 vertices, *and* it produced zero measured
bridge-survival benefit on that corpus. The 2026-09-11 nightly
([ADR-346](../../../adr/ADR-346-deterministic-mincut-witness-partition.md))
fixed a real non-determinism bug in the same code path but explicitly left
both the latency problem and the effectiveness question open, naming
"root-cause the `partition()` latency scaling problem" as its own
"Next Research" item 2.

This run attacks that latency bottleneck from a different angle than the
suggested `PolylogConnectivity` swap: instead of trying to make exact
min-cut faster, it replaces the *signal itself* with **sampled betweenness
centrality** (Brandes 2001; approximated via pivot sampling per Bader,
Kintali, Madduri & Mihail 2007) — a vertex that lies on many shortest paths
between other vertex pairs is a structural bridge candidate by construction,
and unlike global min-cut this is `O(pivots * (V + E))`, with `pivots`
chosen independently of `V` (see `approx_bridge::default_pivot_count`). The
result: `ApproxBridgeForgetting`, a new always-on `CompactionPolicy` in
`crates/ruvector-agent-memory/src/approx_bridge.rs` requiring **no**
`ruvector-mincut` dependency at all.

**Measured result, matched to the exact n=84 corpus and access pattern
ADR-345 used:**

| Policy | Bridge Surv. | Recall@5 | Compaction |
|---|---|---|---|
| CoherenceWeighted (baseline) | 66.7% | 100.0% | 123 us |
| MincutGatedForgetting-Hard (exact min-cut) | 66.7% | 100.0% | 157,246 us |
| **ApproxBridgeForgetting-Hard (this run)** | **66.7%** | 100.0% | **1,470 us** |

The speed hypothesis is **confirmed**: 107x faster than exact min-cut at
matched n=84, and a full compaction pass at n=4,000 still completes in
3.38s (well under the 5s practicality bound), where exact min-cut was
already established as impractical at a two-order-of-magnitude smaller
size. But the effectiveness hypothesis is **falsified again, this time by
a second, independent algorithm**: betweenness-gated protection ties the
plain scalar baseline exactly (+0.0pp bridge-survival gap), and so —
measured in the same run, same corpus, same seed — does exact min-cut. Two
structurally different algorithms (global min-cut and sampled betweenness)
agreeing that there is no detectable graph-topological signal distinguishing
these particular synthetic "bridge" memories from an ordinary cluster
member is stronger evidence against the underlying hypothesis than either
result alone, and is consistent with "Selective Forgetting: A Graph-Based
Memory Framework for Long-Term LLM Agents" (arXiv:2608.28978), whose
own degree-centrality-based graph memory pruning also did not beat a flat
vector baseline on LongMemEval.

**Hardware**: x86-64, Linux `6.18.44-fc-v153`, `rustc 1.97.0`, `cargo
1.97.0`, release build.

## Why This Matters Now and Longer Term

Agent memory systems that grow without bound eventually need compaction,
and every compaction policy risks silently discarding the few memories
that connect otherwise-disparate topic clusters — exactly the failure mode
RuVector's own `fusion::CausalEpisodicGraph` and cross-cluster retrieval
depend on not happening. Two prior nightlies (2026-06-14, 2026-09-05)
approached this from the scalar and exact-graph ends respectively; this run
closes the "is a *cheap* graph signal the answer?" question with real
evidence rather than leaving it as an assumed-but-untested possibility
(the literal state the 2026-09-11 nightly left it in). Longer term, the
honest answer accumulating across three nightlies now is that *this
particular corpus construction* (Gaussian clusters + cosine-midpoint
"bridges") does not produce a bridge that any graph-topological signal this
repo has tried can distinguish from noise — which is itself a useful,
specific, falsifiable finding for whoever next tries a fourth algorithm, or
whoever next tries real (non-synthetic) agent-memory embeddings instead.

## RuVector Ecosystem Fit

This run connects (at least) three RuVector ecosystem capabilities:

1. **Vector search** — the k-NN cosine-similarity graph construction over
   compaction candidates (shared logic pattern with
   `graph_forget::MincutGatedForgetting`, independently reimplemented here
   so this module has no `ruvector-mincut` dependency).
2. **Graph intelligence** — Brandes' betweenness-centrality algorithm and
   its sampled-pivot approximation, evaluated directly against
   `ruvector-mincut`'s exact global min-cut on the identical graphs.
3. **Agent memory** — `ruvector-agent-memory`'s `CompactionPolicy` trait and
   compaction pipeline (`compact()`, `MemoryStore`), the same integration
   surface ADR-345/ADR-346 used.

## Architecture

```mermaid
flowchart LR
    A[MemoryStore entries] --> B[build_knn_adjacency<br/>cosine sim, k=8, min_sim=0.05]
    B --> C{BridgeMode}
    C -->|pivots: default_pivot_count n| D[select_pivots<br/>seeded StdRng, sorted]
    D --> E[approx_betweenness<br/>Brandes BFS per pivot]
    E --> F[rank vertices,<br/>top boundary_fraction]
    F --> G[Soft: additive bonus<br/>Hard: reserved budget]
    G --> H[weighted_importance<br/>scalar baseline score]
    H --> I[select_survivors]
```

`ApproxBridgeForgetting` sits at the same integration point as
`MincutGatedForgetting` (both implement `CompactionPolicy` and layer a
boundary-vertex bonus/reservation on top of the same
`compaction::weighted_importance` scalar score) but swaps the boundary
computation for a self-contained, dependency-free algorithm.

## Implementation

- `crates/ruvector-agent-memory/src/approx_bridge.rs` (492 lines): `BridgeMode`
  (`Soft`/`Hard`, independent of `graph_forget::ForgetMode` so this module
  never needs the `mincut-forget` feature), `ApproxBridgeForgetting`, the
  k-NN adjacency builder, deterministic pivot sampling
  (`rand::seq::index::sample` from a seeded `StdRng`, sorted before use so
  accumulation order never depends on the RNG's draw sequence), and Brandes'
  algorithm (`approx_betweenness`) over `Vec<BTreeSet<usize>>` adjacency —
  deliberately never a hash-keyed structure, given `graph_forget`'s own
  documented `DashMap`-iteration-order non-determinism bug (fixed by
  ADR-346) as a direct cautionary precedent.
- `crates/ruvector-agent-memory/examples/approx_bridge_bench.rs` (465 lines):
  the benchmark below. Requires `--features mincut-forget` only because it
  *also* runs `MincutGatedForgetting` for the matched-n=84 reference row —
  `ApproxBridgeForgetting` itself has no such requirement.
- 7 unit tests in `approx_bridge.rs`: the same hand-built two-clique +
  bridge fixture `graph_forget`'s tests use (Soft/Hard protect the bridge),
  a graceful-fallback-below-minimum-size test, a 30-repetition determinism
  test (the direct analog of the bug category ADR-346 fixed elsewhere), and
  three algorithm-correctness tests (`default_pivot_count` bounds, zero
  betweenness on a clique, peak betweenness at a path's midpoint).

## Benchmark Methodology

Dataset generator (seed `341`, fixed and documented, identical family to
the 2026-09-05 nightly's generator, reimplemented standalone in
`approx_bridge_bench.rs` for the same "no cross-example coupling" reason
`mincut_gated_forgetting_bench.rs`'s own header comment gives):

- `N_CLUSTERS` Gaussian topic-cluster centroids (unit vectors), `PER_CLUSTER`
  members each perturbed by Gaussian noise (0.35) and re-normalized.
- `N_BRIDGES` additional "bridge" vectors: the cosine midpoint of two
  randomly chosen distinct cluster centroids, perturbed by noise 0.15.
- An access-simulation pass with a "cold" era (uniform random accesses over
  the whole corpus) and a "hot" era (90% of accesses concentrated on
  `N_HOT_CLUSTERS=2` clusters), producing a context window (last 10
  accessed cluster centroids) for `weighted_importance`'s coherence term.
- `N_QUERIES=20` recall-probe queries (perturbed cluster centroids, K=5),
  with ground truth computed by brute-force search *before* compaction.

Two parts, both compacting to 50% of corpus size:

- **Part 1 (gated):** n=84 — `N_CLUSTERS=6`, `PER_CLUSTER=12`,
  `N_BRIDGES=12`, dims=32, cold/hot accesses = 40/80 (the literal ADR-345
  constants, not a scaled approximation, for a true apples-to-apples
  reproduction of its access pattern). Three policies: `CoherencePolicy`
  (baseline), `MincutGatedForgetting::hard(.., 0.2)` with `mincut_trials=1`
  (identical settings to ADR-345's production run), and
  `ApproxBridgeForgetting::hard(.., 0.2)` with default pivot count.
- **Part 2 (exploratory):** n=1,000 (`N_CLUSTERS=10`, `PER_CLUSTER=88`,
  `N_BRIDGES=120`, dims=64) and n=4,000 (`N_CLUSTERS=16`, `PER_CLUSTER=220`,
  `N_BRIDGES=480`, dims=64), cold/hot access counts scaled by the same
  ratio as ADR-345's 40/84 and 80/84. Only `CoherencePolicy` and
  `ApproxBridgeForgetting` run — `MincutGatedForgetting` is not attempted at
  these sizes, because ADR-345/ADR-346 already measured
  `RuVectorGraphAnalyzer::partition()` at 76ms-11.4s per call for graphs of
  only 50-400 *vertices*, two orders of magnitude smaller than 1,000-4,000.

Command:

```bash
cargo run --release -p ruvector-agent-memory --example approx_bridge_bench --features mincut-forget
```

## Benchmark Results (Raw)

Full raw output of the exact command above, one run, this session:

```text
╔══════════════════════════════════════════════════════════════════╗
║  ruvector-agent-memory — Approximate Bridge Forgetting            ║
║  (sampled betweenness vs exact min-cut; follow-up to ADR-345/346) ║
╚══════════════════════════════════════════════════════════════════╝

Platform  : linux
Arch      : x86_64
Seed      : 341

## Part 1 — n=84 (matched to ADR-345/ADR-346) ##

  Clusters: 6, per-cluster: 12, bridges: 12, dims: 32, target: 50% compaction

Policy                             Bridge Surv.     Recall@5    Compaction (us)
--------------------------------------------------------------------------------
CoherenceWeighted                         66.7%       100.0%                123
MincutGatedForgetting-Hard                66.7%       100.0%             157246
ApproxBridgeForgetting-Hard                66.7%       100.0%               1470

Part 1 acceptance checks
  T1 speedup vs exact mincut (107.0x) >= 20x        : PASS
  T2 bridge-survival gap vs baseline (+0.0pp) >= 15pp : FAIL
  T3 |recall delta| (0.00pp) <= 2pp              : PASS
  (reference) exact mincut's own survival gap vs baseline: +0.0pp

## Part 2 — exploratory scale probe (baseline vs approx only) ##

  n=1,000 (exploratory) — clusters: 10, per-cluster: 88, bridges: 120, dims: 64
Policy                             Bridge Surv.     Recall@5    Compaction (us)
--------------------------------------------------------------------------------
CoherenceWeighted                          65.8%       100.0%               1698
ApproxBridgeForgetting-Hard                69.2%       100.0%             209028

  n=4,000 (exploratory) — clusters: 16, per-cluster: 220, bridges: 480, dims: 64
Policy                             Bridge Surv.     Recall@5    Compaction (us)
--------------------------------------------------------------------------------
CoherenceWeighted                          54.2%       100.0%               6938
ApproxBridgeForgetting-Hard                59.4%       100.0%            3381956

  n=1,000 (exploratory): approx bridge-survival gap vs baseline = +3.3pp, |recall delta| = 0.00pp, compaction = 209.028ms
  n=4,000 (exploratory): approx bridge-survival gap vs baseline = +5.2pp, |recall delta| = 0.00pp, compaction = 3381.956ms

Part 2 acceptance check (practicality bound only — effectiveness at scale is exploratory/descriptive, no pre-existing bar)
  T4 compaction wall-clock at n=4,000 (3.382s) <= 5s : PASS

=> REJECT: one or more mandatory (T1-T4) acceptance thresholds failed (see above).
```

## Benchmark Results (Summarized)

| Check | Threshold | Measured | Result |
|---|---|---|---|
| T1 — speedup vs exact min-cut @ n=84 | >= 20x | 107.0x | **PASS** |
| T2 — bridge-survival gap vs baseline @ n=84 | >= 15pp | +0.0pp | **FAIL** |
| T3 — \|recall delta\| @ n=84 | <= 2pp | 0.00pp | PASS |
| T4 — compaction wall-clock @ n=4,000 | <= 5s | 3.382s | PASS |

Exploratory (not gated, no pre-existing bar): bridge-survival gap +3.3pp
(n=1,000) and +5.2pp (n=4,000) — present, consistently positive, but far
below the 15pp bar and not independently validated against a hand-labeled
ground truth at that scale (see Limitations).

## Failure Modes

1. **No detectable structural signal from either algorithm at n=84.**
   `MincutGatedForgetting-Hard` (exact min-cut) and
   `ApproxBridgeForgetting-Hard` (sampled betweenness) both produced
   *exactly* the baseline's 66.7% bridge-survival rate on this run's corpus
   and seed — not approximately equal, identical. Inspecting the mechanism:
   in `Hard` mode, the protect-budget reservation only changes the outcome
   when it forces in a boundary vertex that scalar ranking alone would have
   evicted; if the designed "bridge" vertices are never flagged as boundary
   by either algorithm's graph signal (plausible on Gaussian-cluster data
   with continuous-noise midpoint interpolation, per ADR-345's own
   "may instead isolate an ordinary noise-outlier vertex" finding), the
   reservation protects different, unrelated vertices and the measured
   bridge-survival rate is identical to plain scalar ranking by
   construction, not by chance.
2. **Betweenness is a real algorithm reimplementation, not a renamed
   min-cut — validated independently of the main corpus.** Two unit tests
   (`betweenness_zero_on_a_clique`, `betweenness_peaks_on_a_path_midpoint`)
   confirm the core `approx_betweenness` function behaves correctly on
   graphs with known, hand-computable betweenness values, and
   `determinism_is_stable` confirms 30 repeated calls on an unchanged graph
   return byte-identical boundary sets — so the T2 failure is not
   attributable to an implementation bug in this run's new code, consistent
   with it replicating ADR-345's own finding via an independent mechanism.
3. **Exploratory scale-probe gaps are small, positive, but unvalidated.**
   +3.3pp and +5.2pp at n=1,000/4,000 are in the "might be real, might be
   noise" range — no hand-labeled ground-truth bridge set exists at that
   scale (unlike the n=84 fixture, which was hand-verified in prior
   nightlies), and no repeated-trial variance estimate was computed this
   run (single seed per size, to keep the n=4,000 run's 3.4s cost from
   multiplying). Reported descriptively, not as a secondary ACCEPT.

## Rejected Alternatives

- **Trying to make exact min-cut itself faster** (the 2026-09-11 nightly's
  own suggested next step: swap in `connectivity::polylog::PolylogConnectivity`,
  already in-tree but not wired into `BoundedInstance`). Not attempted this
  run: it requires modifying `ruvector-mincut` internals (a 44,000-line
  crate), a materially larger and riskier change for one session than
  adding a new, independent, dependency-free signal in
  `ruvector-agent-memory` itself. Left as a live option for a future
  nightly, now with betweenness as a second negative data point motivating
  it less than it might have looked after ADR-345 alone.
- **Karger-Stein approximate min-cut** (named as an untried alternative in
  ADR-345's own "Alternatives Considered"). Not attempted: betweenness was
  chosen instead because "high betweenness correlates with
  cut/bridge-vertex-ness" is independently well-established (Brandes 2001;
  Bader et al. 2007) and does not require re-deriving an approximate cut
  value, only a ranking — a better fit for this module's "boundary set"
  consumption pattern than an approximate cut *value* would be.
- **Plain degree centrality** (what "Selective Forgetting", arXiv:2608.28978,
  actually uses). Considered but not implemented as a fourth candidate this
  run: degree centrality is `O(V+E)` with no sampling at all (even cheaper
  than betweenness) but is a strictly weaker signal — a vertex's own edge
  count says nothing about whether those edges happen to be the only path
  between two clusters, which is exactly the case betweenness is designed
  to catch and degree centrality is not. Reported here as a concrete
  next-research candidate rather than silently assumed inferior.
- **A statistical-outlier boundary threshold (mean + k*stddev) instead of a
  fixed top-`boundary_fraction`.** Not implemented: adds a free parameter
  with no principled way to set it without exactly the kind of
  post-hoc-tuning this nightly process exists to avoid; the fixed-fraction
  approach is simpler and auditable, at the cost of not adapting to graphs
  where the "true" boundary set is much smaller or larger than 15%.

## Security and Governance Notes

No new cryptographic primitive, no new untrusted input path, no change to
`ruvector-agent-memory`'s existing "no witness, no mutation" invariant
(`ApproxBridgeForgetting` is a plain `CompactionPolicy`, consumed by
`compact()` or `compact_witnessed()` exactly like any other policy — the
latter still certifies every eviction regardless of which policy chose it).
The betweenness computation is read-only over the candidate set and cannot
itself corrupt the witness chain or violate `target_size`. Unlike
`graph_forget`, this module has **no optional dependency** (`rand` is
already a direct dependency of this crate), so it carries no new supply-chain
surface.

## MCP / WASM / Edge / RVF Applicability

`approx_betweenness`'s core loop (BFS + back-propagation over a bounded
adjacency list, `f64` arithmetic only, no threading, no unsafe) compiles
cleanly to `wasm32-unknown-unknown` as written (not smoke-tested against an
actual wasm target this run — flagged as a next step), which the exact
min-cut path's `ruvector-mincut` dependency tree does not guarantee as
cheaply given its `DashMap`/`rayon` dependencies. This makes the approach a
plausible fit for an edge/WASM agent-memory compaction pass if the
effectiveness question is ever answered positively on a different corpus —
not claimed as validated here.

## Practical Applications

1. A drop-in, dependency-light structural-signal *option* for any
   `ruvector-agent-memory` deployment wanting to experiment with bridge
   protection without pulling in `ruvector-mincut` at all.
2. A reusable `approx_betweenness` primitive (Brandes' algorithm) for any
   future RuVector graph-analysis task needing cheap centrality, independent
   of this specific compaction use case.
3. A second, independently-measured data point — alongside ADR-345 — against
   "any cheap graph signal will obviously help agent-memory compaction,"
   directly useful to whoever next evaluates a third algorithm or a
   real-embedding corpus instead of synthetic Gaussian clusters.

## Acceptance Result

```text
REJECT (effectiveness hypothesis, T2)
```

T1 (speed, 107.0x >= 20x), T3 (recall, 0.00pp <= 2pp), and T4 (practicality
at n=4,000, 3.382s <= 5s) all pass — sampled betweenness centrality is
confirmed to be a dramatically faster, still-practical-at-scale signal,
exactly as hypothesized. But T2, the core effectiveness question this
module exists to answer, fails: measured bridge-survival gap over baseline
is +0.0pp against a pre-registered 15pp bar, identical (not merely similar)
to exact min-cut's own result on the same corpus and seed. Per this
project's own rule ("if the result fails the threshold, that's a valid
REJECT outcome — report it honestly rather than lowering the bar"), this is
reported as REJECT for production use, not reframed around the (real, but
secondary) speed win.

## Limitations

- Single corpus family (Gaussian clusters + cosine-midpoint bridges) and a
  single seed (341) per size — no variance estimate across seeds.
- The n=1,000/4,000 "exploratory" gaps have no hand-labeled ground truth
  (the n=84 fixture's bridges were spot-checked in prior nightlies; at
  larger n this run trusts the generator's own `bridge_indices` bookkeeping
  without independent verification that those vectors remain
  topologically bridge-like at scale).
- `wasm32-unknown-unknown` compilation was reasoned about, not tested.
- `BridgeMode::Soft` was implemented and unit-tested but not included in
  the main comparison table (only `Hard` was benchmarked at scale, matching
  ADR-345's own choice to report `Hard` as its primary candidate in later
  nightlies' cross-references).
- `boundary_fraction=0.15` and `default_pivot_count`'s `4*sqrt(n)` scaling
  are engineering defaults, not swept or tuned against the effectiveness
  metric — consistent with the stated no-post-hoc-tuning rule, but it also
  means a different, untried choice might (or might not) change the T2
  result.

## Next Research

1. Hand-verify whether the n=1,000/4,000 "bridge" vectors this run's
   generator produces are still topologically distinguishable from ordinary
   cluster members at all (e.g. check their actual k-NN degree and
   component membership), rather than trusting the generator's own
   bookkeeping — directly addresses this run's own Limitations item 2.
2. Try plain degree centrality (the signal "Selective Forgetting",
   arXiv:2608.28978, actually used) as a third, even-cheaper candidate on
   this exact benchmark harness, to check whether betweenness's extra cost
   over degree centrality buys anything on this specific corpus family.
3. Re-run Part 1 and Part 2 against real (non-synthetic) agent-memory
   embeddings instead of Gaussian clusters, to test ADR-345's own open
   question 3 (whether the "no signal" finding is corpus-specific) with a
   different, more realistic generator this time.
4. Only if (1) or (3) finds a real signal: revisit
   `connectivity::polylog::PolylogConnectivity` as a way to make *exact*
   min-cut itself fast enough to use as a higher-fidelity fallback or
   cross-check against the approximate signal.

## References

- Brandes, U. (2001). "A faster algorithm for betweenness centrality."
  *Journal of Mathematical Sociology*, 25(2).
- Bader, D., Kintali, S., Madduri, K., & Mihail, M. (2007). "Approximating
  Betweenness Centrality." *WAW 2007*.
- "Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM
  Agents" (arXiv:2608.28978).
- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md`,
  [ADR-345](../../../adr/ADR-345-mincut-gated-forgetting.md).
- `docs/research/nightly/2026-09-11-mincut-partition-determinism/README.md`,
  [ADR-346](../../../adr/ADR-346-deterministic-mincut-witness-partition.md).
- `docs/research/nightly/2026-06-14-agent-memory-compaction/README.md`
  (origin of `CoherencePolicy` and the scalar-baseline methodology reused
  here).
