# Sampled Betweenness Centrality vs Exact Min-Cut for Agent-Memory Bridge Protection

A standalone technical write-up. Repo: `ruvnet/ruvector`, crate
`ruvector-agent-memory`, module `src/approx_bridge.rs`.

## Problem

Agent-memory compaction policies that score memories independently
(recency, frequency, semantic coherence with a context window — RuVector's
own `CoherencePolicy`, introduced 2026-06-14) can evict a "bridge" memory:
the sole semantic link between two otherwise-disjoint topic clusters. Doing
so silently fragments the surviving store's connectivity, which matters for
anything downstream that walks cross-cluster relationships
(`fusion::CausalEpisodicGraph`, multi-hop retrieval).

A 2026-09-05 experiment (ADR-345) tried the obvious fix: build a k-NN
similarity graph over compaction candidates, find its exact global minimum
cut with `ruvector-mincut`'s `RuVectorGraphAnalyzer`, and protect the
vertices straddling that cut. It was rejected on two independent grounds:
1,800-2,700x slower than the scalar baseline even at 84 vertices
(76ms-11.4s per `partition()` call for 50-400 vertices), *and* zero
measured bridge-survival benefit at the 84-vertex size the latency problem
forced the experiment down to. A 2026-09-11 follow-up (ADR-346) found and
fixed a genuine non-determinism bug in the same `partition()` call path
(traced to `DashMap`'s randomly-seeded iteration order affecting
tie-breaking) but explicitly left the latency and effectiveness questions
open, naming "root-cause the latency scaling problem" as its own next step.

## Hypothesis

This run asks a different question than "can exact min-cut be made
faster?": **can a cheaper, different graph signal — sampled betweenness
centrality — give the same or better bridge-protection outcome, fast
enough to be practical at a corpus size exact min-cut cannot reach?**

Given synthetic agent-memory corpora of Gaussian topic clusters plus
cosine-midpoint-interpolated "bridge" memories (same generator family as
ADR-345, reimplemented standalone), a k-NN (k=8, cosine >= 0.05) similarity
graph, release build, seed 341:

- **T1 (speed):** at matched n=84, the new candidate's compaction
  wall-clock is >= 20x faster than exact min-cut's.
- **T2 (effectiveness):** at matched n=84, the new candidate's
  bridge-survival-rate gap over the scalar baseline is >= 15 percentage
  points (ADR-345's own pre-registered bar).
- **T3 (quality):** |recall@5 delta| vs baseline is <= 2 percentage points.
- **T4 (practicality, exploratory corpus sizes):** compaction wall-clock at
  n=4,000 is <= 5 seconds.

## Design

**Betweenness centrality** (Brandes 2001) measures, for each vertex `v`,
the fraction of shortest paths between all other vertex pairs that pass
through `v`. A vertex that is the *sole* connector between two clusters
necessarily carries every shortest path between them — maximal betweenness
by construction. Computing it exactly for all vertices and all
source-target pairs is `O(V * E)` via Brandes' single-source algorithm run
from every vertex; this run instead samples a bounded number of source
("pivot") vertices and accumulates only their contribution — the standard
approximation family surveyed in Bader, Kintali, Madduri & Mihail (2007)
and later refined by KADABRA (Borassi & Natale, 2016) and ABRA (Riondato &
Upfal). With `pivots` chosen independently of `V` (here, `4*sqrt(V)`
clamped to `[16, V]` — an engineering choice, not a formal error bound from
either paper), total cost is `O(pivots * (V + E))`: linear in graph size for
fixed-degree k-NN graphs, not the apparent cubic-or-worse behavior ADR-345
measured for `RuVectorGraphAnalyzer::partition()`.

Implementation (`approx_bridge.rs`, 492 lines):

1. Build a symmetric k-NN cosine-similarity adjacency list
   (`Vec<BTreeSet<usize>>` — sorted iteration, never a hash-keyed structure,
   a deliberate choice given `graph_forget`'s own documented
   `DashMap`-order non-determinism bug as precedent).
2. Deterministically sample `pivots` source vertices from a seeded
   `StdRng` (`rand::seq::index::sample`), sorted ascending before use.
3. Run Brandes' algorithm (BFS + back-propagated dependency accumulation)
   from each pivot; sum contributions into one betweenness score per
   vertex.
4. Rank vertices by score (ties broken by index), take the top
   `boundary_fraction` (default 15%) as the "boundary" set.
5. Combine with the existing scalar `weighted_importance` score exactly as
   `MincutGatedForgetting` does: `Soft` (additive bonus) or `Hard` (reserved
   eviction-immune budget).

`ApproxBridgeForgetting` implements the crate's existing `CompactionPolicy`
trait and has **no dependency on `ruvector-mincut`** — `rand` is already a
direct dependency of `ruvector-agent-memory`, so this is a zero-new-dependency
addition.

## Real Measured Evidence

Command: `cargo run --release -p ruvector-agent-memory --example
approx_bridge_bench --features mincut-forget` (the feature is only needed
for the reference `MincutGatedForgetting` row; the policy itself does not
require it). Hardware: x86-64, Linux `6.18.44-fc-v153`, `rustc 1.97.0`,
release build, seed 341.

**Part 1 — matched n=84** (identical corpus size and access pattern to
ADR-345: 6 clusters x 12 + 12 bridges, 32-dim, 40 cold-era + 80 hot-era
accesses):

| Policy | Bridge Surv. | Recall@5 | Compaction |
|---|---|---|---|
| CoherenceWeighted (baseline) | 66.7% | 100.0% | 123 us |
| MincutGatedForgetting-Hard (exact) | 66.7% | 100.0% | 157,246 us |
| ApproxBridgeForgetting-Hard (approx) | 66.7% | 100.0% | 1,470 us |

- T1 (speed): 157,246 / 1,470 = **107.0x** faster. >= 20x bar: **PASS**.
- T2 (effectiveness): +0.0pp gap over baseline. >= 15pp bar: **FAIL**.
- T3 (recall): 0.00pp delta. <= 2pp bar: **PASS**.

**Part 2 — exploratory scale probe** (no exact min-cut: already established
impractical at 2 orders of magnitude smaller):

| Corpus | Baseline survival | Approx survival | Gap | Approx compaction |
|---|---|---|---|---|
| n=1,000 | 65.8% | 69.2% | +3.3pp | 209.0 ms |
| n=4,000 | 54.2% | 59.4% | +5.2pp | 3,382.0 ms |

- T4 (practicality): 3.382s <= 5s at n=4,000: **PASS**.

Full raw console output is in `README.md`'s "Benchmark Results (Raw)"
section, verbatim from the actual run.

## Interpretation

T1, T3, T4 all pass — the speed hypothesis holds completely: sampled
betweenness is two orders of magnitude faster than exact min-cut at matched
size and stays comfortably practical (3.4s, not the 11+s/400-vertices or
worse exact min-cut already showed) three orders of magnitude further out
in corpus size.

T2 fails, and the way it fails is informative: `ApproxBridgeForgetting`
does not merely under-perform the 15pp bar, it reproduces baseline's
66.7% survival *exactly* — and so, independently, does
`MincutGatedForgetting` in this same run (+0.0pp, not the "zero measured
difference" ADR-345 reported from its own separate run, but a fresh
independent confirmation of the same qualitative result). Two structurally
unrelated graph algorithms — one exact and global, one approximate and
local-sampling-based — agreeing that there's no detectable signal here is
stronger evidence against "a cheap graph signal fixes agent-memory
bridge-forgetting on this corpus" than either algorithm's result alone.
This is also consistent with external evidence: "Selective Forgetting: A
Graph-Based Memory Framework for Long-Term LLM Agents" (arXiv:2608.28978)
reports its own degree-centrality-based graph-memory pruning did not beat a
flat vector baseline on LongMemEval either.

The most likely mechanism (not independently confirmed this run, flagged as
next research): Gaussian-cluster-plus-cosine-midpoint "bridges" may not
reliably become low-connectivity cut vertices or high-betweenness vertices
in the resulting k-NN graph at all — the interpolation is exact in
embedding space but the resulting k-NN graph topology is a noisy function
of it, and nothing in this run's generator enforces that the bridge ends
up topologically load-bearing rather than just semantically interesting.

## Limitations

- One corpus family, one seed per size — no cross-seed variance estimate.
- The n=1,000/4,000 bridge-survival gaps (+3.3pp, +5.2pp) have no
  hand-verified ground truth the way the n=84 fixture's bridges were
  spot-checked in prior nightlies; they are suggestive, not validated.
- `wasm32-unknown-unknown` compatibility was reasoned about from the code's
  shape (no unsafe, no threading, bounded `f64` arithmetic) but not
  actually built or tested against a wasm target.
- `boundary_fraction=0.15` and the `4*sqrt(n)` pivot-count heuristic are
  engineering defaults, not swept against the effectiveness metric.
- Only `BridgeMode::Hard` was benchmarked at scale (both modes are
  unit-tested).

## Production Relevance

`ApproxBridgeForgetting` is not recommended for production bridge
protection as evaluated: it is fast and practical, but the underlying
"graph structure predicts which memories are safe to evict" premise remains
unsupported by two independent algorithms across two nightlies now. Its
production-relevant contribution is negative-but-useful: it rules out "the
min-cut approach would have worked if only it were faster" as an
explanation for ADR-345's rejection, narrowing the live hypothesis space for
whoever investigates this again. The `approx_betweenness` primitive itself
(Brandes' algorithm with pivot sampling) is reusable, dependency-light, and
independently correct (confirmed on hand-computable graphs: zero on a
clique, peak at a path's midpoint) — a candidate building block for other
RuVector graph-analysis tasks regardless of this specific application's
outcome.

## Future Direction

See the README's "Next Research" for the four concrete follow-ups:
hand-verifying bridge topology at scale, trying plain degree centrality as
an even-cheaper third candidate, re-running against real (non-synthetic)
embeddings, and only then revisiting exact min-cut's own latency via
`connectivity::polylog::PolylogConnectivity` if either prior step finds a
real signal worth a higher-fidelity cross-check.

## References

- Brandes, U. (2001). "A faster algorithm for betweenness centrality."
  *Journal of Mathematical Sociology*, 25(2), 163-177.
- Bader, D., Kintali, S., Madduri, K., & Mihail, M. (2007). "Approximating
  Betweenness Centrality." *Algorithms and Models for the Web-Graph (WAW)*.
- Borassi, M., & Natale, E. (2016). "KADABRA is an Adaptive Algorithm for
  Betweenness via Random Approximation." arXiv:1604.08553.
- "Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM
  Agents." arXiv:2608.28978.
- ADR-345, ADR-346 and their corresponding nightly READMEs (this repo).
