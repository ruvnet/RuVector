# Global Min-Cut Gated Streaming Memory Admission

**A write-time admission gate for streaming agent memory, using a terminal-free global minimum cut instead of a fixed similarity threshold — plus the calibration bug that nearly hid the result.**

## Problem

Agent memory systems cluster incoming vectors into topics/sessions as they
stream in. The standard mechanism (sequential k-means / leader-follower,
still the production default) is a fixed threshold test: merge the new
vector into its nearest cluster if cosine similarity clears a bar, else
spawn a new cluster. That test looks at exactly one edge of the similarity
graph — the new point to its single nearest centroid — and ignores
everything else in the graph. It cannot distinguish a point that is a
legitimate, distant member of a naturally spread-out cluster from a point
that is only weakly attached to *everything*, including its nearest
centroid.

`ruvector-namespace-merge` (this workspace, 2026-08-08) showed that
looking at the *whole* graph beats a threshold for the dual read-time
question — which namespaces to search for a given query — using S-T
max-flow/min-cut. But that formulation needs a source and a sink, and a
query naturally supplies both ("relevant" vs. "irrelevant" per
namespace). Write-time admission has no query, and so no natural
source/sink.

## Hypothesis

Does a **terminal-free global minimum cut** — the single weakest link in
the whole graph, not a fixed two-way split — work as a write-time
admission gate, and does it beat the threshold baseline at the *same*
downstream maintenance cost (same final cluster count)?

## Technical Design

Three admission policies, one shared trait:

```rust
pub trait AdmissionPolicy {
    fn decide(&self, point: &[f32]) -> Decision;              // read-only
    fn commit(&mut self, point: &[f32], decision: &Decision);  // mutating
    fn admit(&mut self, point: &[f32]) -> Decision { /* decide + commit */ }
}
```

1. **`NearestCentroidThreshold`** (baseline) — cosine-to-nearest-centroid
   threshold test.
2. **`MincutGatedAdmission`** (candidate A) — build a graph of `(existing
   cluster centroids) + (candidate point)`, edge weight = clamped cosine
   similarity, run **Stoer-Wagner global minimum cut** (self-contained
   O(V³) implementation, `src/mincut.rs`), and gate on the cut's average
   crossing-edge weight against a threshold `tau`. With exactly one
   existing cluster the graph has only 2 nodes, and *any* cut of a 2-node
   graph trivially separates them — this is a real bug the crate's own
   test suite caught before any benchmark ran (see below), fixed by
   special-casing that degenerate case.
3. **`AdaptiveMincutAdmission`** (candidate B) — identical mechanism, but
   `tau` is set online via Welford's running mean/std of previously
   observed cut weights instead of a hand-tuned constant.

## The Calibration Trap

The first benchmark run used hand-picked constants (threshold=0.55,
tau=0.35) chosen before any diagnostic. The baseline degenerated: 3,289 of
4,000 points spawned their own cluster, scoring a trivially "pure" 0.9988
— because a policy that spawns a cluster per point is definitionally
"pure" — with recall@10 of just 0.0603. **Purity alone is gameable by
over-fragmentation.** This is exactly the trap the mission's adversarial-
review pass exists to catch, and here it surfaced from simply running the
numbers, not from catching it in advance.

The fix: compare policies at a **matched final cluster count** — the same
downstream reindex/memory budget — rather than at independently
hand-picked thresholds. The benchmark binary-searches the baseline's
threshold (25 bisection steps) to match candidate A's own cluster count
under a fixed `tau`, then reports quality at that matched budget. `tau =
0.005` was chosen from a documented plateau (0.005, 0.002, and 0.001 all
gave identical results), not a single cherry-picked point.

## Actual Implementation

Zero-dependency Rust crate, `crates/ruvector-memory-admission`:

- `src/mincut.rs` — Stoer-Wagner global min cut on a dense weighted graph.
  Correctness checked against a hand-solved 4-node graph (all 7
  bipartitions verified by hand), a closed-form uniform-weight complete
  graph (min cut = n−1), and a planted-outlier graph.
- `src/dataset.rs` — deterministic synthetic streaming dataset (LCG64 +
  Box-Muller, no external RNG dependency), 8 ground-truth clusters, 20%
  high-noise "drift" points, Fisher-Yates-shuffled arrival order.
- `src/policy.rs` — the three policies above.
- `src/bin/benchmark.rs` — matched-budget calibration + the benchmark.

20/20 tests pass; `cargo clippy --all-targets` is clean.

## Actual Benchmark Evidence

4,000 synthetic vectors, 8 ground-truth clusters, 64 dimensions, 300
held-out queries, matched at 17 clusters:

| Variant | Clusters | Purity | Recall@10 | Mean insert |
|---|---|---|---|---|
| NearestCentroidThreshold (calibrated) | 17 | 0.8285 | 0.7840 | 0.01µs |
| **MincutGatedAdmission** | 17 | **0.8735** | **0.8623** | 19.28µs |
| AdaptiveMincutAdmission | 48 | 0.8615 | 0.6610 | 37.14µs |

At matched cluster budget: **+4.50pp purity, +7.83pp recall@10** over the
threshold baseline, for ~19µs mean insertion latency (well under a 500µs
write-path ceiling). This is not a free lunch — it's an O(C³)-vs-O(C) cost
trade, fully disclosed and bounded by a safety valve.

Candidate B's self-calibrating threshold does **not** replicate this — it
drifts to the safety-valve cap (48 clusters) and loses 12.3pp of recall.
A genuine negative result: a plain running mean/std of the *global* cut
weight does not track the *local* admission-relevant threshold as cluster
count grows. `k_std` and `min_observations` were fixed before this run,
not swept afterward to chase a pass.

Full raw output (including the original degenerate run, the threshold and
tau sweeps, and the exact reproduction command) is preserved in the
nightly research doc's Raw Evidence section — nothing here was hand-edited
after the fact.

## Limitations

Synthetic data only; single run per configuration (no variance
characterisation); no concurrent-writer, delete, or scale-beyond-4,000
testing; candidate B's negative result is about one specific estimator
design, not a claim that no self-calibrating threshold could work.

## Production Relevance

Not wired into any production write path. `ruvector-agent-memory` already
owns memory *eviction* (compaction); this is the complementary *admission*
mechanism it currently lacks. A concrete production path exists — see the
ADR — gated on concurrent-writer support, delete interaction, and
larger-scale validation, none of which this PoC attempted.

## RuVector Ecosystem Implications

Extends the "graph cuts for agent-memory lifecycle decisions" pattern
`ruvector-namespace-merge` started, to a second, structurally distinct
lifecycle stage and a different cut algorithm (terminal-free global min
cut vs. S-T max-flow). Candidate cluster-admission state (centroids +
assignment history) is a natural fit for RVF's portable, replayable memory
shards; a `n_clusters`/purity health signal is a natural ruFlo trigger for
background reclustering when self-calibration (as measured here) isn't
enough on its own.

## Future Direction

1. Locate the O(C³) practical cost ceiling empirically.
2. A cluster-count-conditioned or local-similarity-conditioned
   self-calibrating threshold, addressing candidate B's documented
   specific failure mode.
3. Concurrent-writer and delete-interaction hardening.
4. A real (non-synthetic) agent-memory corpus benchmark.

## References

- Stoer, M., Wagner, F. "A Simple Min-Cut Algorithm." *Journal of the ACM*, 1997.
- Hartigan, J. A. "Clustering Algorithms." Wiley, 1975.
- Zhang, T., Ramakrishnan, R., Livny, M. "BIRCH." *SIGMOD*, 1996.
- `ruvector-namespace-merge`, ADR-299 (this workspace, 2026-08-08).
- `ruvector-agent-memory` nightly research, 2026-06-14 (this workspace).

Full methodology, benchmark hygiene notes, and the complete raw evidence
trail: `docs/research/nightly/2026-09-02-mincut-streaming-memory-admission/README.md`.
ADR: `docs/adr/ADR-341-mincut-gated-streaming-memory-admission.md`.
