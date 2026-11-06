# ADR-353: Sampled Betweenness Centrality as a Fast Alternative Signal for Agent-Memory Bridge Protection

## Status

Rejected (for production use, as designed — same effectiveness conclusion
as ADR-345, confirmed by a second, independent algorithm). New crate
addition (`ruvector-agent-memory::approx_bridge`), always-on (no feature
flag, no new dependency), retained as evidence and as a reusable
dependency-light betweenness-centrality primitive, not promoted as a
recommended compaction policy.

## Context

ADR-345 (2026-09-05 nightly) added `ruvector-agent-memory::graph_forget`,
layering `ruvector-mincut`'s exact global min-cut
(`RuVectorGraphAnalyzer::partition()`) on top of the crate's existing
scalar `CoherencePolicy` as a structural "protect bridge memories" signal.
It was rejected on two independent grounds, both measured: 1,800-2,700x
slower than the scalar baseline at only 84 vertices (76ms-11.4s per
`partition()` call for graphs of 50-400 vertices), and zero measured
bridge-survival benefit at the 84-vertex size the latency forced the
experiment down to.

ADR-346 (2026-09-11 nightly) found and fixed two real bugs in the same
code path — `DashMap`'s randomly-seeded iteration order affecting
tie-breaking non-deterministically, and an incorrect vertex-range inference
in partition materialization — making `RuVectorGraphAnalyzer::partition()`
deterministic. It explicitly left the latency problem open (naming it "the
remaining blocker" in its own Next Research) and did not re-attempt the
bridge-protection effectiveness question.

This ADR asks a different question than "can exact min-cut be made
faster": **is global min-cut even the right family of algorithm for this
signal**, or does a cheaper, different graph-centrality measure —
**sampled betweenness centrality** (Brandes 2001; approximated via pivot
sampling per Bader, Kintali, Madduri & Mihail 2007) — give comparable or
better bridge-protection, fast enough to use at corpus sizes exact min-cut
cannot reach at all?

## Hypothesis

```text
Given synthetic agent-memory corpora of Gaussian topic clusters plus
cosine-midpoint-interpolated "bridge" memories (same generator family as
ADR-345), a k-NN (k=8, cosine >= 0.05) similarity graph over compaction
candidates, release build, seed=341 (fixed, documented):

T1 (speed): at matched n=84, ApproxBridgeForgetting-Hard's compaction
wall-clock is >= 20x faster than MincutGatedForgetting-Hard's
(mincut_trials=1, identical settings to ADR-345's production run).

T2 (effectiveness): at matched n=84, ApproxBridgeForgetting-Hard's
bridge-survival-rate gap over the CoherencePolicy baseline is >= 15
percentage points (ADR-345's own pre-registered bar, reused unchanged).

T3 (quality): |recall@5 delta| vs baseline is <= 2 percentage points.

T4 (practicality, exploratory corpus sizes n=1,000 / n=4,000, where exact
min-cut is not run — already established impractical two orders of
magnitude smaller): compaction wall-clock at n=4,000 is <= 5 seconds.
```

Full methodology and raw output:
`docs/research/nightly/2026-10-25-approx-bridge-forgetting/README.md`.

## Decision

1. Add `ruvector-agent-memory::approx_bridge::ApproxBridgeForgetting`
   (`BridgeMode::Soft` / `Hard`), a `CompactionPolicy` implementing sampled
   betweenness centrality (Brandes' algorithm, pivot-sampled) as a
   structural boundary signal, combined with the existing
   `compaction::weighted_importance` scalar score exactly as
   `graph_forget::MincutGatedForgetting` combines its exact-min-cut signal.
2. **No feature flag, no new dependency.** Unlike `graph_forget`
   (feature-gated behind `mincut-forget`, optional `ruvector-mincut`
   dependency), `approx_bridge` is always compiled: it has no dependency on
   `ruvector-mincut` at all, and `rand` is already a direct dependency of
   `ruvector-agent-memory`.
3. **Do not promote `ApproxBridgeForgetting` to a recommended or default
   compaction policy.** T2 (the core effectiveness question) fails,
   confirming — via a second, independent algorithm — ADR-345's own
   "zero measured difference" finding rather than overturning it.
4. Keep the module in-tree as (a) retained negative evidence per the
   nightly process's evidence-retention rule, and (b) a reusable,
   independently-tested `approx_betweenness` (Brandes' algorithm) primitive
   for future graph-analysis work in this repo, regardless of this specific
   application's outcome.

## Evidence

Full raw benchmark output is in the nightly README (linked above);
summarized here (one run, seed 341, release build, this session):

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| T1 — speedup vs exact min-cut @ n=84 | >= 20x | 107.0x | **PASS** |
| T2 — bridge-survival gap vs baseline @ n=84 | >= 15pp | +0.0pp | **FAIL** |
| T3 — \|recall@5 delta\| @ n=84 | <= 2pp | 0.00pp | PASS |
| T4 — compaction wall-clock @ n=4,000 | <= 5s | 3.382s | PASS |

Raw n=84 comparison table:

| Policy | Bridge Surv. | Recall@5 | Compaction |
|---|---|---|---|
| CoherenceWeighted (baseline) | 66.7% | 100.0% | 123 us |
| MincutGatedForgetting-Hard (exact) | 66.7% | 100.0% | 157,246 us |
| ApproxBridgeForgetting-Hard (approx) | 66.7% | 100.0% | 1,470 us |

Exploratory (n=1,000 / n=4,000, no pre-existing bar, descriptive only):
bridge-survival gap +3.3pp / +5.2pp, |recall delta| 0.00pp / 0.00pp,
compaction 209.0ms / 3,382.0ms.

Root cause (see nightly README "Failure modes" for the full account): in
this run, `MincutGatedForgetting-Hard` (exact min-cut) *also* measured a
+0.0pp gap on the identical corpus and seed — not merely a similar small
number, but the identical 66.7% survival rate as the unmodified baseline.
Two structurally unrelated algorithms (global min-cut; local pivot-sampled
betweenness) agreeing that no detectable graph signal distinguishes this
corpus's labeled "bridge" memories from an ordinary cluster member is
stronger evidence against the underlying premise than either algorithm's
result in isolation. Two correctness unit tests
(`betweenness_zero_on_a_clique`, `betweenness_peaks_on_a_path_midpoint`)
independently confirm `approx_betweenness`'s core computation is correct
on graphs with known values, ruling out "the new code is just buggy" as an
explanation for T2's failure.

## Consequences

- `ruvector-agent-memory` gains a dependency-light, always-on
  `CompactionPolicy` option (`ApproxBridgeForgetting`) that is 107x faster
  than the existing `mincut-forget`-gated alternative at matched size and
  remains practical three orders of magnitude further out (3.4s at
  n=4,000) — useful as a building block even though its specific
  bridge-protection premise is unproven on this corpus.
- `ruvector-agent-memory` gains a reusable, independently-tested
  `approx_betweenness` (Brandes' algorithm, pivot-sampled) primitive with
  no new crate dependency, usable by future graph-analysis work regardless
  of this ADR's rejection.
- No existing behavior changes: `CoherencePolicy`, `LruPolicy`, `LfuPolicy`,
  `MincutGatedForgetting`, and the plain `compact()` function are untouched.
  `approx_bridge` is additive and off by default in the sense that no
  caller is switched to it automatically.
- Narrows the live hypothesis space for the open question ADR-345 left:
  "does the global-min-cut-isolates-a-noise-outlier finding hold on real
  (non-synthetic) embeddings, or is it corpus-specific?" remains open, but
  "would a different, cheaper algorithm have found the signal min-cut
  missed, on this same synthetic corpus?" is now answered: no.

## Alternatives Considered

- **Fixing exact min-cut's latency directly** (ADR-346's own suggested next
  step: swap in `connectivity::polylog::PolylogConnectivity`, already
  in-tree but not wired into `BoundedInstance`). Not attempted this run:
  requires modifying `ruvector-mincut` internals (44,000 lines), materially
  riskier for one session than adding an independent, self-contained
  module. Remains a live option, now motivated less strongly given a
  second negative effectiveness data point.
- **Karger-Stein approximate min-cut** (named as untried in ADR-345's own
  "Alternatives Considered"). Not attempted: betweenness was chosen
  instead because it produces a ranking directly (this module's
  consumption pattern) rather than requiring a re-derived approximate cut
  *value*, and "high betweenness correlates with bridge-vertex-ness" is
  independently well-established in the cited literature.
- **Plain degree centrality** (the actual signal used by "Selective
  Forgetting," arXiv:2608.28978). Not implemented as a fourth candidate
  this run — flagged as next research rather than assumed inferior; it is
  `O(V+E)` with zero sampling, strictly cheaper than betweenness, but
  structurally weaker (a vertex's edge count says nothing about whether
  those edges are the unique path between two clusters).
- **A statistical-outlier boundary threshold instead of a fixed top-15%.**
  Not implemented: adds a free, untuned parameter with no principled
  default, which this project's no-post-hoc-tuning rule argues against
  introducing without separate justification.

## Implementation Plan

Already implemented in this change:

- `crates/ruvector-agent-memory/src/approx_bridge.rs` (492 lines, always
  compiled, no new dependency): `BridgeMode`, `ApproxBridgeForgetting`,
  `default_pivot_count`, the k-NN adjacency builder, deterministic pivot
  sampling, and `approx_betweenness` (Brandes' algorithm).
- `crates/ruvector-agent-memory/src/lib.rs`: `pub mod approx_bridge;` and
  re-exports (`ApproxBridgeForgetting`, `BridgeMode`, `default_pivot_count`).
- `crates/ruvector-agent-memory/examples/approx_bridge_bench.rs` (465
  lines): the two-part benchmark (matched n=84 three-way comparison;
  exploratory n=1,000/4,000 scale probe), registered in `Cargo.toml` with
  `required-features = ["mincut-forget"]` (needed only for the reference
  `MincutGatedForgetting` row).
- 7 unit tests in `approx_bridge.rs`: Soft/Hard bridge protection on the
  shared hand-built fixture, graceful fallback below minimum size, a
  30-repetition determinism test, `default_pivot_count` bounds, and two
  betweenness-correctness tests on graphs with known values.

No further implementation is planned under this ADR; see "Next Research"
in the nightly README for follow-up scope.

## API Shape

```rust
// Always available (no feature flag, no ruvector-mincut dependency):
pub enum BridgeMode { Soft, Hard }

pub struct ApproxBridgeForgetting {
    pub weights: CoherenceWeights,
    pub mode: BridgeMode,
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,   // Soft
    pub protect_fraction: f32,   // Hard
    pub n_pivots: usize,         // 0 => default_pivot_count(n)
    pub boundary_fraction: f32,
    pub seed: u64,
}

impl ApproxBridgeForgetting {
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self;
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self;
}
impl CompactionPolicy for ApproxBridgeForgetting { /* .. */ }

pub fn default_pivot_count(n: usize) -> usize;
```

## Feature Flags

None added. `approx_bridge` is unconditionally compiled as part of
`ruvector-agent-memory`'s default build — it introduces zero new crate
dependencies (`rand` is already direct), unlike `graph_forget`'s
`mincut-forget` flag.

## Benchmark Evidence

See "Evidence" above and the linked nightly README for full raw console
output, methodology, and the generator's exact parameters.

## Security

No new cryptographic primitive, no new dependency, no new untrusted input
path. `ApproxBridgeForgetting` is a plain, read-only (over the candidate
set) `CompactionPolicy`; it is consumed identically to any other policy by
`compact()` or the witness-emitting `compact_witnessed()`, so the existing
"no witness, no mutation" invariant is unaffected by this change. The
betweenness computation cannot itself corrupt state or bypass
`target_size`.

## Governance

None beyond the existing invariants `compaction.rs` and
`witnessed_compaction.rs` already enforce; this ADR adds no new governance
surface.

## Failure Modes

See the nightly README's "Failure modes" section: (1) both exact min-cut
and sampled betweenness measured the identical 66.7% baseline
bridge-survival rate on this run's corpus/seed — a tie, not an
approximation error; (2) two unit tests independently confirm
`approx_betweenness`'s correctness on graphs with hand-computable values,
ruling out an implementation bug as the explanation; (3) the exploratory
n=1,000/4,000 gaps (+3.3pp, +5.2pp) are small, positive, but have no
hand-verified ground truth and no cross-seed variance estimate.

## Migration

None: `approx_bridge` is new and additive. No existing caller or default
behavior changes.

## Rollback

Remove the `approx_bridge` module and its `lib.rs` re-exports; nothing else
in the crate depends on it (it is a leaf module that depends on
`compaction` and `memory`/`scoring`, not the reverse).

## Rejection Criteria

The hypothesis in this ADR is treated as rejected because:

1. The primary effectiveness gate (T2: bridge-survival gap >= 15pp)
   measured +0.0pp — identical to the scalar baseline, and identical to
   exact min-cut's own result on the same corpus and seed in this run.

T1, T3, and T4 (speed, recall-neutrality, and practicality-at-scale) all
pass; per this project's rule that any single mandatory gate's failure is
sufficient for rejection regardless of other gates passing, this is
reported as REJECT for production use, not reframed as a partial accept
around the speed result.

## Open Questions

1. Does the "no detectable bridge signal" finding hold on real
   (non-synthetic) agent-memory embeddings, or is it an artifact of this
   (and ADR-345's) Gaussian-cluster-plus-cosine-midpoint generator?
   (ADR-345's own open question 3, still unanswered by this run.)
2. Would plain degree centrality (cheaper still, and the signal "Selective
   Forgetting" actually uses) perform any differently on this exact
   benchmark harness, or does the corpus itself simply contain no
   extractable graph-topological bridge signal regardless of which
   centrality measure is tried?
3. Are the n=1,000/4,000 exploratory gaps (+3.3pp, +5.2pp) real, or noise?
   No cross-seed replication was run this session to distinguish them.
