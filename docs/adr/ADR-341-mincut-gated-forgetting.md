# ADR-341: Mincut-Gated Forgetting — Structural Eviction Signal and Eviction Witnesses for Agent Memory

## Status

Rejected (for production use as designed). Experimental crate addition
(`ruvector-agent-memory::graph_forget`, feature-gated behind `mincut-forget`,
off by default) retained as evidence and reference implementation, not
promoted. The companion `witnessed_compaction` module (no mincut dependency)
is independently sound and left enabled by default.

## Context

`ruvector-agent-memory`'s `CoherencePolicy` (introduced by the
2026-06-14 nightly, `docs/research/nightly/2026-06-14-agent-memory-compaction`)
scores every memory independently: `I = alpha*recency + beta*frequency +
gamma*coherence`. It has no notion of graph structure. A "bridge" memory —
the sole semantic link between two otherwise-disjoint topic clusters — can
score low on every scalar term and be evicted, silently fragmenting the
surviving store's connectivity even though downstream consumers
(`fusion::CausalEpisodicGraph`, cross-cluster retrieval) depend on such
bridges surviving.

Separately, `ruvector-agent-memory` witnesses *admission* (`ledger.rs`'s
`TransactionalLedger`, ADR-307) and the sibling `ruvector-retrieval-receipt`
crate witnesses *retrieval* (ADR-304, ADR-340). Neither covers *deletion*: a
compaction pass drops entries with no auditable record of which ids were
evicted, when, or by which policy.

`ruvector-mincut` already implements a general dynamic minimum-cut engine
with a purpose-built vector-graph integration layer,
`RuVectorGraphAnalyzer` (`crates/ruvector-mincut/src/integration/mod.rs`),
including `from_knn` construction and `.partition()` / `.find_bridges()`
convenience methods. This ADR's question: can that existing engine, used
as-is, give `ruvector-agent-memory` the structural signal `CoherencePolicy`
lacks?

## Hypothesis

```text
Given a synthetic corpus of 6 topic clusters (12 memories each = 72) plus 12
"bridge" memories interpolated 50/50 between two randomly paired clusters,
32-dim, with a hot-cluster access simulation pattern, and a k-NN (k=5, cosine
>= 0.05) similarity graph feeding ruvector-mincut's RuVectorGraphAnalyzer,

when the 84-entry store is compacted to 50% (42 entries) using
MincutGatedForgetting-Soft (structural bonus delta=0.5) and
MincutGatedForgetting-Hard (20% of the retained budget reserved for boundary
vertices) versus the existing CoherencePolicy baseline,

then both candidates retain a bridge-memory survival rate at least 15
percentage points higher than baseline, while Recall@10 stays within 2
percentage points of baseline,

subject to: (a) compaction wall-clock under 100x baseline's, and (b) 100%
tamper-detection across 20 single-byte-flip trials against the eviction
witness chain.
```

Full methodology, raw output, and the two supporting feasibility probes
(`examples/mincut_scaling_probe.rs`, `examples/mincut_determinism_probe.rs`)
are in
`docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md`.

## Decision

1. Add `ruvector-agent-memory::graph_forget::MincutGatedForgetting`
   (`ForgetMode::Soft` / `Hard`), a `CompactionPolicy` implementation that
   layers a `ruvector-mincut`-derived structural boundary signal on top of
   the existing `weighted_importance` scalar score (factored out of
   `CoherencePolicy` for reuse). Feature-gated behind `mincut-forget`
   (optional path dependency on `ruvector-mincut`), off by default.
2. Add `ruvector-agent-memory::witnessed_compaction::compact_witnessed` +
   `EvictionWitnessChain`: certifies every evicted id with a chained
   `LedgerWitnessRecord` (ADR-134 schema, `crate::ops`, one new
   `action_kind` constant `LEDGER_COMPACT_EVICT = 0xA7`), enforcing "no
   witness, no mutation" identically to `ledger.rs`'s admission path. Not
   feature-gated; has no mincut dependency and works with any
   `CompactionPolicy`.
3. **Do not promote `MincutGatedForgetting` to a recommended or default
   policy.** The measured evidence (below) falsifies the hypothesis on both
   the effectiveness and performance axes.
4. Keep the module in-tree, behind its feature flag, as a working
   integration reference and as retained negative evidence per the nightly
   process's evidence-retention rule.

## Evidence

Full raw benchmark output and the performance-scaling / determinism tables
are in the nightly README (linked above); summarized here:

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Bridge-survival gap (Soft) | >= 15pp | +0.0pp | FAIL |
| Bridge-survival gap (Hard) | >= 15pp | +0.0pp | FAIL |
| Recall@10 delta (Soft) | <= 2pp | 0.00pp | PASS |
| Recall@10 delta (Hard) | <= 2pp | 0.00pp | PASS |
| Compaction slowdown (Soft) | <= 100x | ~1,800-2,700x | FAIL |
| Compaction slowdown (Hard) | <= 100x | ~1,800-2,700x | FAIL |
| Tamper detection | 100% / 20 trials | 20/20 | PASS |

Root causes, both independently measured (see README "Failure modes"):

- **Latency.** `RuVectorGraphAnalyzer::partition()` scales from ~77ms (n=50)
  to ~11.4s (n=400) on a synthetic k-NN graph, with an outlier 69s at n=19 on
  a small regular topology. This alone makes any corpus above a few hundred
  memories impractical for a per-compaction call.
- **Non-determinism.** On a hand-built 19-vertex graph with a *provably
  unique* weakest link, 30 repeated `partition()` calls on byte-identical
  input returned an empty/unusable result in 15/30 (50%) of calls, at an
  average 841ms/call. No direct RNG usage was found in `ruvector-mincut`'s
  relevant modules, so this is more consistent with hash-map
  iteration-order-dependent tie-breaking than an intentional randomized
  algorithm.
- **No measured effect at the size the above forced.** At the resulting
  84-memory corpus, `MincutGatedForgetting` computed real, non-empty
  boundary sets (2-10 vertices out of 84, confirmed via an instrumented
  run), but the specific vertices flagged did not overlap with the
  labeled ground-truth bridge memories in any run — the global minimum cut
  of noisy Gaussian-cluster data is not guaranteed to isolate the
  human-intended "bridges" rather than an arbitrary low-connectivity
  outlier.

## Consequences

- `ruvector-agent-memory` gains a working, tested (if unpromoted) reference
  for how a downstream crate would integrate `RuVectorGraphAnalyzer`, useful
  to whoever next attempts to fix its performance/determinism.
- `ruvector-agent-memory` gains a real, working, default-on capability
  independent of the rejected hypothesis: `compact_witnessed` closes the
  admission/retrieval/deletion witness-coverage gap for any existing
  compaction policy, at negligible cost (64 bytes/evicted entry).
- No existing behavior changes: `CoherencePolicy`/`LruPolicy`/`LfuPolicy`
  and the plain `compact()` function are untouched; `graph_forget` is
  opt-in and off by default.
- A follow-up hardening item against `ruvector-mincut` is documented (see
  the nightly README's "Next Research") but not filed as a tracked issue in
  this ADR's scope — this ADR's job is to record the evidence, not fix the
  dependency.

## Alternatives Considered

- **`RuVectorGraphAnalyzer::find_bridges()` instead of `.partition()`.**
  Rejected: recomputes a full `MinCutWrapper::query()` per edge (O(E) full
  min-cut recomputations per call), strictly worse than the already-too-slow
  one-shot `.partition()`.
- **`ruvector_mincut::DynamicMinCut` / `ClusterHierarchy` directly, bypassing
  `RuVectorGraphAnalyzer`.** Not implemented in this pass — flagged as the
  primary next-research direction, since it might avoid the specific
  overhead measured here without invalidating the underlying idea.
- **Ship without the acceptance benchmark, on the grounds that any
  structural signal is "probably fine."** Rejected outright: this is
  precisely the unsupported-claim failure mode the nightly process exists to
  prevent.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-agent-memory/src/graph_forget.rs` (feature-gated)
- `crates/ruvector-agent-memory/src/witnessed_compaction.rs` (always on)
- `crates/ruvector-agent-memory/src/compaction.rs`: extracted
  `weighted_importance()`
- `crates/ruvector-agent-memory/src/ops.rs`: added
  `action_kind::LEDGER_COMPACT_EVICT`
- `crates/ruvector-agent-memory/examples/mincut_gated_forgetting_bench.rs`,
  `mincut_scaling_probe.rs`, `mincut_determinism_probe.rs`
- Unit tests in `graph_forget.rs` (2, using `mincut_trials = 10` to keep
  flake probability negligible against the documented non-determinism) and
  `witnessed_compaction.rs` (3)

No further implementation is planned under this ADR; see "Next Research" in
the nightly README for the follow-up scope (out of this ADR).

## API Shape

```rust
// Always available (no mincut dependency):
pub struct EvictionWitnessChain { /* .. */ }
pub fn compact_witnessed(
    store: &mut MemoryStore,
    policy: &dyn CompactionPolicy,
    target_size: usize,
    context_window: &[Vec<f32>],
    actor_id: &str,
    now_ns: u64,
    chain: &mut EvictionWitnessChain,
    sink: &mut dyn WitnessSink,
) -> Result<Vec<LedgerWitnessRecord>, LedgerError>;

// Behind `mincut-forget`:
pub enum ForgetMode { Soft, Hard }
pub struct MincutGatedForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode,
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,
    pub protect_fraction: f32,
    pub mincut_trials: usize,
}
impl CompactionPolicy for MincutGatedForgetting { /* .. */ }
```

## Feature Flags

- `mincut-forget` (new): gates `graph_forget` and its `ruvector-mincut` path
  dependency. Off by default; the crate's default build is unaffected.
- `witnessed_compaction` is not feature-gated — it adds no new dependency
  and reuses machinery already always-compiled in `crate::ops`.

## Benchmark Evidence

See "Evidence" above and the linked nightly README for full raw output,
methodology, and the two supporting probe scripts.

## Security

No new cryptographic primitive. `compact_witnessed` reuses
`crate::ops`'s existing keyless-FNV-1a witness chain verbatim (same
tamper-evidence scope already documented there: naive-edit detection, not
adversary-resistant — see that module's docs). The mincut structural signal
is advisory-only and cannot itself corrupt the witness chain or bypass
`target_size`.

## Governance

None beyond the existing "no witness, no mutation" invariant, which
`compact_witnessed` enforces identically to `ledger.rs`.

## Failure Modes

See the nightly README's "Failure modes" section for the full, measured
account (latency scaling, non-determinism reproduction, and the
no-measured-effect finding).

## Migration

None: `graph_forget` is new and feature-gated; `compact_witnessed` is
additive (existing `compact()` callers are unaffected).

## Rollback

Remove the `mincut-forget` feature and its `graph_forget` module entirely
with no impact on any existing caller — nothing in the crate's default
build path depends on it. `witnessed_compaction` could similarly be reverted
in isolation if a problem were found, independent of the mincut question.

## Rejection Criteria

The hypothesis in this ADR is treated as rejected because:

1. The primary comparison (bridge-survival gap >= 15pp) measured 0.0pp.
2. The performance gate (<=100x baseline) measured ~1,800-2,700x.

Either failure alone would have been sufficient for rejection; both were
observed independently.

## Open Questions

1. Does `ruvector_mincut::DynamicMinCut`/`ClusterHierarchy`, used directly,
   avoid the measured `RuVectorGraphAnalyzer::partition()` overhead?
   (Next-research item 1.)
2. What specifically causes the measured non-determinism inside
   `ruvector-mincut` — instance construction order, witness materialization,
   or something else in the `MinCutWrapper::process_instances` path?
   (Next-research item 2; out of this ADR's scope to answer.)
3. Does the "global min-cut isolates an outlier, not the intended bridge"
   finding hold on real (non-synthetic) agent-memory embeddings, or is it an
   artifact of this experiment's Gaussian-cluster generator?
