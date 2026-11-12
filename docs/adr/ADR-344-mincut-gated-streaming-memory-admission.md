# ADR-344: Global-Min-Cut Gated Streaming Memory Admission

## Status

Proposed. Experimental crate (`ruvector-memory-admission`), not wired into
`ruvector-agent-memory` or any production write path. Standalone research
PoC, evaluated at synthetic-benchmark scale only.

## Context

`ruvector-agent-memory` (2026-06-14 nightly) gave RuVector a principled
*eviction* mechanism: coherence-weighted compaction decides which existing
memories to keep under a size budget. It did not address the dual
question at the other end of the memory lifecycle: as new memories stream
in, which cluster should each one join, or does it need a new one?

`ruvector-namespace-merge` (ADR-299, 2026-08-08) answered a structurally
similar-looking but actually different question — at *query* time, which
namespaces should be searched — using S-T max-flow/min-cut. That
formulation needs a source and a sink; a query naturally provides
"relevant" (source-side) and "irrelevant" (sink-side) affinities per
namespace. Write-time admission has no such terminals: there is no query,
only a candidate point and a growing set of existing clusters. Framing
admission with a fixed cosine-similarity threshold against the nearest
centroid (the standard sequential-k-means / leader-follower approach, and
still the default in most production streaming-clustering systems) uses
only one edge of the similarity graph and cannot distinguish "this point
is a legitimate but distant member of a naturally spread cluster" from
"this point is only weakly attached to everything, including its nearest
centroid."

**Product claim to earn**: at the same downstream maintenance budget (same
final cluster count, i.e. the same reindex/memory cost any admission
policy imposes on the rest of the system), does looking at the whole
cluster graph — not just the nearest edge — produce measurably better
clusters?

## Hypothesis

```text
Given a stream of 4,000 synthetic agent-memory vectors drawn from 8
ground-truth semantic clusters (64 dimensions, 20% high-noise boundary
points, randomly interleaved arrival order),

when MincutGatedAdmission (global Stoer-Wagner min-cut over existing
cluster centroids + the candidate point, gated on the cut's average
crossing-edge weight against a fixed tau) is used for online cluster
admission,

compared to NearestCentroidThreshold calibrated via binary search to
produce the SAME final cluster count (matched maintenance budget, not an
independently hand-picked threshold),

then purity and held-out recall@10 should both improve over the matched
baseline,

subject to: purity not regressing (>= 0pp), recall@10 improving by
>= 2 percentage points, mean insertion latency staying under 500µs
(write-path, not hot-query-path budget), and final cluster count staying
within 3x ground truth (24) as a fragmentation guard.
```

A secondary hypothesis tested whether a self-calibrating threshold
(running mean/std of observed cut weights, SONA-inspired in spirit, not a
`sona` crate integration) could replace the hand-tuned `tau` without
hand-tuning, evaluated at whatever cluster count it lands on unassisted
(deliberately not matched to the baseline).

## Decision

Implement three admission policies behind one `AdmissionPolicy` trait in a
new crate, `ruvector-memory-admission`:

1. **`NearestCentroidThreshold`** — baseline. Merge into nearest centroid
   if cosine >= threshold, else spawn. Not a straw man: this is the
   standard production approach (sequential k-means / leader-follower).
2. **`MincutGatedAdmission`** — candidate A. Build a `(clusters + 1)`-node
   weighted graph (edge weight = clamped cosine similarity between
   centroids, and between each centroid and the candidate), run Stoer-
   Wagner global minimum cut (`src/mincut.rs`, self-contained, O(V^3), no
   dependency on the existing `ruvector-mincut` crate — chosen to keep this
   PoC single-file-auditable; a dependency edge to `ruvector-mincut` is a
   fair follow-up if this is promoted), and gate on the cut's average
   crossing-edge weight against `tau`. A computational safety valve
   (`max_clusters`, deliberately set well above the acceptance bound so
   the bound is a real measurement) caps the O(C^3) per-insertion cost.
3. **`AdaptiveMincutAdmission`** — candidate B. Same mechanism, `tau` set
   online via Welford's running mean/std of previously observed cut
   weights, using only the graph's own structure (no ground-truth labels,
   no evaluation leakage).

Policies are compared at a **matched final cluster count**: the benchmark
binary-searches the baseline's threshold (25 bisection iterations) to
match candidate A's natural cluster count under a fixed `tau`, then
reports purity/recall@10 at that matched operating point. This exists
because an *uncalibrated* first run (hand-picked constants, not swept)
produced a degenerate baseline — 3,289 of 4,000 points spawning their own
cluster, trivially "pure" (0.9988) but useless (recall@10 = 0.0603) —
demonstrating that purity alone is gameable by over-fragmentation and that
comparing policies at independently chosen thresholds is not a fair test.
That run and the calibration sweep that replaced it are both preserved
verbatim in the nightly research doc's Raw Evidence section, not silently
overwritten.

## Evidence

Matched-budget run (both baseline and candidate A at 17 clusters, `tau =
0.005`, chosen from a documented plateau where 0.005/0.002/0.001 all gave
identical results — not a single lucky point):

| Variant | Clusters | Purity | Recall@10 | Mean insert (µs) |
|---|---|---|---|---|
| NearestCentroidThreshold (calibrated) | 17 | 0.8285 | 0.7840 | 0.01 |
| **MincutGatedAdmission** | 17 | **0.8735** (+4.50pp) | **0.8623** (+7.83pp) | 19.28 |
| AdaptiveMincutAdmission | 48 (uncalibrated) | 0.8615 | 0.6610 (−12.30pp vs. matched baseline) | 37.14 |

**Candidate A: all four pre-registered criteria PASS.** Purity gain
+4.50pp (>= 0pp required), recall gain +7.83pp (>= 2pp required), latency
19.28µs (<= 500µs required), 17 final clusters (<= 24 required).

**Candidate B: FAIL on 2 of 3 criteria.** Its self-calibrating threshold
drifted to the 48-cluster safety-valve cap instead of settling near 17,
losing 12.30pp of recall versus the matched baseline (> 2pp tolerance) and
exceeding the cluster-count bound. This is a genuine negative result about
this specific estimator (a plain running mean/std of the *global* cut
weight does not track the *local* admission-relevant threshold as cluster
count grows), not a tuning failure — `k_std` and `min_observations` were
fixed before this run and not swept afterward to chase a pass. Full
methodology, raw sweep data, and the negative-result analysis are in
`docs/research/nightly/2026-09-02-mincut-streaming-memory-admission/README.md`.

**Correctness**: 20/20 tests pass (`cargo test --release -p
ruvector-memory-admission`) — 4 hand-verified/closed-form min-cut
correctness tests, policy-level admission tests including a regression
test for a real bug this PoC's own test suite caught (see below), and
integration tests for no-lost-vectors / bounded cluster count / read-only
`decide()`. `cargo clippy --release -p ruvector-memory-admission
--all-targets`: clean.

**A bug the process caught before any benchmark number was reported**: the
first implementation of candidate A's spawn/merge decision treated "the
candidate is alone on its side of the global cut" as always meaning
"spawn" — true when >= 2 other clusters exist (a genuine structural-
outlier signal), but with exactly 1 existing cluster the graph has only 2
nodes and *any* cut trivially separates them regardless of similarity.
This made two near-identical vectors (cosine ~0.998) route to separate
clusters, failing `mincut_admission_merges_close_points`. Fixed by special-
casing `c == 1` to rely solely on the cut-weight threshold
(`src/policy.rs::should_spawn`).

## Consequences

**Positive**:
- A validated, disclosed-cost technique for a lifecycle stage
  (write-time admission) RuVector's agent-memory substrate did not
  previously have a principled answer for.
- Establishes a second graph-cut-based primitive (terminal-free global
  min cut) alongside `ruvector-namespace-merge`'s S-T max-flow, extending
  the "graph cuts for agent-memory lifecycle decisions" pattern started
  there to a second, structurally distinct problem.
- A documented, reusable matched-budget benchmark methodology (calibrate
  the baseline to the candidate's own cluster count) that the next
  online-clustering nightly in this workspace can reuse directly, avoiding
  a repeat of this run's original purity-gaming baseline mistake.
- A preserved negative result (candidate B) that saves a future run from
  re-attempting the same self-calibration design blind.

**Negative / costs**:
- O(C^3) per-insertion cost is real and measured (~1,600x the baseline's
  latency at matched budget) — an explicit trade, not hidden, bounded by
  the `max_clusters` safety valve and the 500µs acceptance ceiling, but a
  real cost a production integration must budget for.
- No concurrent-writer, delete/eviction-interaction, or scale-beyond-4,000
  testing was performed — all named explicitly as prerequisites in
  "Implementation Plan" below, not silently assumed safe.
- Self-calibration (candidate B) is not production-viable in its current
  form; any near-term production use of this pattern requires the
  hand-tuned `tau` of candidate A, with the operational cost of tuning
  that implies.

## Alternatives

- **S-T max-flow/min-cut** (as `ruvector-namespace-merge` uses): rejected
  for this problem because write-time admission has no natural source/sink
  to fix in advance — there is no query defining relevance at insertion
  time. This is the core reason a *different* cut algorithm was chosen
  here rather than reusing ADR-299's flow graph code.
- **Raw cosine with an adaptive percentile threshold** (no graph cut at
  all, just an online-adjusted scalar threshold): a simpler alternative to
  both candidates here; not implemented, because it would not test this
  ADR's actual claim (whether whole-graph structure, not just threshold
  adaptation, adds value). Left as an explicitly open, fair comparison
  target for a future nightly — not claimed to have been beaten.
- **Hierarchical CF-tree (BIRCH-style)**: bounded-cost merge/split with
  materially more implementation complexity than a one-night PoC scope
  allows; named as a production-hardening direction, not attempted.

## Implementation Plan

Not scheduled; contingent on promotion interest. If pursued:

1. Concurrent-writer support (today's `&mut self` `commit` assumes a
   single writer per memory store).
2. Define delete/eviction interaction with `ruvector-agent-memory`'s
   existing compaction path (currently undefined).
3. Scale testing past 4,000 points / 48 clusters to empirically locate the
   O(C^3) cost ceiling.
4. Cross-platform floating-point determinism check (prerequisite for any
   RVF-replay claim).
5. A read-only `memory_admission_stats` MCP tool before any write-authority
   MCP surface (see nightly doc's MCP Implications).
6. Only after 1–5: an opt-in, feature-flagged `AdmissionPolicy` in
   `ruvector-agent-memory`, defaulting to existing behavior.

## API Shape

```rust
pub trait AdmissionPolicy {
    fn name(&self) -> &str;
    fn decide(&self, point: &[f32]) -> Decision;             // read-only
    fn commit(&mut self, point: &[f32], decision: &Decision); // mutating
    fn n_clusters(&self) -> usize;
    fn centroid(&self, cluster_id: usize) -> &[f32];
    fn admit(&mut self, point: &[f32]) -> Decision { /* decide + commit */ }
}
```

No production API surface is proposed by this ADR; the trait above is the
PoC's internal shape, listed for continuity if a future ADR promotes it.

## Feature Flags

None proposed. Not wired into any existing crate.

## Benchmark Evidence

`cargo run --release -p ruvector-memory-admission --bin benchmark`
(env: `TAU=0.005`, all other parameters default). Full raw output for both
the matched-budget run and the original uncalibrated run, plus the
threshold/tau sweeps that motivated the final parameter choices, are
preserved in
`docs/research/nightly/2026-09-02-mincut-streaming-memory-admission/README.md`
under "Raw Evidence."

## Security

No untrusted deserialization, no network or filesystem I/O beyond the
benchmark binary's diagnostics, no secrets. `max_clusters` bounds this
crate's own worst-case cost; a caller streaming adversarial "always spawn"
input could still drive every insertion to the `O(max_clusters)`
safety-valve fallback — bounded, not unbounded, but worth rate-limiting at
the caller if ever exposed to untrusted write volume. See nightly doc's
"Security" section for the full disclosure.

## Governance

Research PoC only; changes no existing crate's behavior. No governance
action required at this status.

## Failure Modes

See nightly doc's "Failure Modes" section: cold start, the c==1 graph-size
degeneracy (fixed, regression-tested), runaway fragmentation (bounded by
the safety valve, separately checked by the 3x-K acceptance bound),
candidate B's self-calibration divergence (documented negative result),
and two disclosed-not-silently-assumed gaps (concurrency, deletes).

## Migration

None; no existing behavior changes.

## Rollback

None; nothing is wired in to roll back. Deleting `crates/ruvector-memory-
admission` and its workspace-member entry fully removes this ADR's
footprint.

## Rejection Criteria

This ADR's candidate A would be rejected from any future promotion
attempt if: (a) concurrent-writer testing reveals correctness issues
without a redesign, (b) the O(C^3) cost ceiling falls within realistic
production cluster-count ranges, or (c) a real (non-synthetic) agent-
memory corpus fails to replicate the matched-budget purity/recall gain
measured here.

## Open Questions

1. Does the matched-budget purity/recall gain replicate on a real (not
   synthetic) agent-memory corpus?
2. Where exactly does the O(C^3) cost cross the 500µs (or a tighter,
   production-relevant) latency budget as cluster count grows past 48?
3. Can candidate B's self-calibration be fixed by conditioning the running
   statistic on cluster count or local (not whole-graph) similarity,
   addressing its documented specific failure mode?
4. Does this technique's determinism survive cross-platform floating-point
   non-associativity well enough for an RVF-replay use case?
