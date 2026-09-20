# ADR-346: Canonical (Pseudo-Deterministic) Mincut Backend Option for Agent-Memory Forgetting

## Status

Accepted (as an additive, opt-in backend — default behavior unchanged).
`MincutGatedForgetting` itself remains **not promoted** to a recommended or
default compaction policy; this ADR does not change ADR-345's rejection of
that policy, it closes two of its three open findings.

## Context

ADR-345 added `ruvector-agent-memory::graph_forget::MincutGatedForgetting`,
a structural (min-cut-derived) eviction signal for agent-memory compaction,
built on `ruvector_mincut::RuVectorGraphAnalyzer`. It was rejected for
production use on two measured, mandatory-gate failures:

1. **Non-determinism**: `RuVectorGraphAnalyzer::partition()` returned
   different results across repeated calls on byte-identical input (up to
   50% degenerate/empty in ADR-345's measurement).
2. **Performance**: compaction using the signal was 1,800-2,700x slower
   than the scalar baseline, over the pre-registered 100x threshold.

ADR-345's "Next Research" left both as open follow-ups, plus a third: if
either finding changed under a different backend, re-run the exact original
benchmark (same corpus, seed, thresholds) rather than a new one, per the
nightly research process's "don't move the goalposts" rule.

`ruvector-mincut` already ships a second, unused engine for this purpose:
`canonical::source_anchored::SourceAnchoredMinCut` (Cargo feature
`canonical`), an implementation of ADR-117's pseudo-deterministic canonical
minimum cut (Kenneth-Mordoch, "Faster Pseudo-Deterministic Minimum Cut",
2026) — deterministic by construction via a fixed vertex ordering and
lexicographic tie-break, rather than dependent on the iteration order of
`RuVectorGraphAnalyzer`'s underlying `DashMap`/`HashSet`-backed graph
storage.

## Hypothesis

```text
Given the exact ADR-345 corpus, seed, and acceptance thresholds,

when MincutGatedForgetting-Soft/-Hard use SourceAnchoredMinCut in place of
RuVectorGraphAnalyzer for boundary-vertex detection,

then repeated calls on identical input return an identical partition, and
compaction wall-clock stays under the original 100x-vs-baseline threshold,

subject to: the bridge-survival gap (>=15pp) and recall delta (<=2pp)
thresholds from ADR-345 apply unmodified, and tamper-detection stays at
100%/20 trials.
```

Full methodology, raw benchmark output, root-cause confirmation, and
ecosystem/long-horizon analysis are in
`docs/research/nightly/2026-09-20-canonical-mincut-forgetting/README.md`.

## Decision

1. Add `ruvector_agent_memory::graph_forget::MincutBackend` (`Legacy` |
   `Canonical`) and a `backend` field on `MincutGatedForgetting`, plus
   `soft_canonical`/`hard_canonical` constructors selecting the canonical
   backend. `soft`/`hard` (and the default `MincutBackend::Legacy`)
   are **unchanged** — this is additive, not a behavior change to any
   existing caller.
2. Enable the `canonical` feature on `ruvector-mincut` whenever
   `ruvector-agent-memory`'s existing `mincut-forget` feature is enabled
   (both already off by default; no change to any default build).
3. Add `examples/mincut_canonical_probe.rs` (determinism + scaling probes
   against the canonical backend, mirroring ADR-345's
   `mincut_determinism_probe.rs`/`mincut_scaling_probe.rs` exactly) and
   `examples/mincut_gated_forgetting_bench_canonical.rs` (a line-for-line
   copy of ADR-345's benchmark with only the backend swapped).
4. **Do not change `MincutGatedForgetting`'s default backend, and do not
   promote it to a recommended policy.** The bridge-survival acceptance
   gate still fails under the canonical backend, identically to ADR-345
   under the legacy backend (+0.0pp vs. a required ≥15pp on both). The
   corrected backend rules out "the legacy engine's bug was suppressing a
   real effect" as an explanation, it does not supply one.
5. Document, as confirmed (not new) root cause: `RuVectorGraphAnalyzer`'s
   graph storage (`DynamicGraph`'s `adjacency: DashMap<VertexId,
   HashSet<(VertexId, EdgeId)>>`) has no fixed vertex/edge visitation
   order, so ties in its min-cut search resolve by map/set iteration order
   rather than any graph property — consistent with ADR-345's suspicion
   and with the measured non-determinism. No change was made to
   `DynamicGraph`, `MinCutWrapper`, or `RuVectorGraphAnalyzer` themselves;
   this ADR routes around the issue via an already-shipped alternative
   rather than fixing the legacy path.

## Evidence

Same-environment, same-run comparison (Linux x86_64, release build, seed
341, this crate's ADR-345 corpus unmodified):

| Gate | Threshold | Legacy backend | Canonical backend |
|---|---|---|---|
| Partition determinism (30 calls) | 100% identical | 40% agreement / 60% degenerate | 100% identical (30/30) |
| Bridge-survival gap (Soft) | ≥ 15pp | +0.0pp — FAIL | +0.0pp — FAIL |
| Bridge-survival gap (Hard) | ≥ 15pp | +0.0pp — FAIL | +0.0pp — FAIL |
| Recall@10 delta (Soft) | ≤ 2pp | 0.00pp — PASS | 0.00pp — PASS |
| Recall@10 delta (Hard) | ≤ 2pp | 0.00pp — PASS | 0.00pp — PASS |
| Compaction slowdown (Soft) | ≤ 100x | 2003.2x — FAIL | 67.4x — PASS |
| Compaction slowdown (Hard) | ≤ 100x | 1989.7x — FAIL | 62.4x — PASS |
| Tamper detection | 100%/20 | 20/20 — PASS | 20/20 — PASS |

Scaling probe (ring k-NN graphs, single build+cut call per size):

| n | Legacy total (build+partition) | Canonical total (build+cut) | Speedup |
|---:|---:|---:|---:|
| 50 | 109.51 ms | 0.954 ms | ~114.8x |
| 100 | 656.14 ms | 3.45 ms | ~190.2x |
| 200 | 3,499.98 ms | 17.55 ms | ~199.4x |
| 400 | 14,309.68 ms | 109.74 ms | ~130.4x |

Raw command output and the n=19 outlier (a 91.4s legacy-backend call,
reproducing ADR-345's "over a minute" degenerate case) are in the nightly
README.

## Consequences

- Any future `ruvector-mincut` consumer needing a repeatable partition
  should prefer `SourceAnchoredMinCut`/`canonical_mincut` over
  `RuVectorGraphAnalyzer`; this ADR is the first concrete evidence of that
  preference inside this codebase, not a new rule imposed on
  `ruvector-mincut` itself.
- `MincutGatedForgetting` remains available only behind the
  `mincut-forget` feature, off by default, unpromoted, exactly as ADR-345
  left it — this ADR changes what backend it *can* use, not whether it is
  recommended.
- The most promising concrete next step (not implemented here): a *local*
  per-cluster-pair cut signal (`ruvector-mincut`'s existing
  `GomoryHuTree`/`ClusterHierarchy::boundary_size`, both unused by
  `graph_forget.rs` today) in place of a single *global* min-cut, motivated
  directly by this ADR's effectiveness finding (see nightly README
  "Interpretation" §2 and "Next research" §1).

## Alternatives considered

- **Fix `RuVectorGraphAnalyzer`/`MinCutWrapper` directly** (e.g. sort
  vertices before iteration) instead of routing around it. Rejected for
  this run: it would touch a widely-used, general-purpose integration
  layer (`CommunityDetector`, `GraphPartitioner` also depend on it) for the
  benefit of one caller, when an already-shipped, already-tested
  alternative exists. Left as a possible future `ruvector-mincut` hardening
  item, not attempted here.
- **Promote `MincutGatedForgetting-Canonical` as the new default** despite
  the unchanged bridge-survival failure. Rejected: violates the nightly
  process's explicit rule against promoting on partial evidence — the
  mandatory effectiveness gate still fails.
- **Widen the corpus to the originally-planned ~1,950-memory size**, now
  that it may be affordable, to get a "final" answer on effectiveness in
  one run. Rejected for this run specifically to keep the comparison
  strictly apples-to-apples with ADR-345 (same corpus); recorded as
  "Next research" item 2 instead.

## Implementation plan / API shape

Additive only, already implemented in this change:

```rust
pub enum MincutBackend { Legacy, Canonical } // Legacy is Default

pub struct MincutGatedForgetting {
    // ...existing fields unchanged...
    pub backend: MincutBackend,
}

impl MincutGatedForgetting {
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self; // unchanged, backend=Legacy
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self; // unchanged, backend=Legacy
    pub fn soft_canonical(weights: CoherenceWeights, structural_bonus: f32) -> Self; // new, backend=Canonical
    pub fn hard_canonical(weights: CoherenceWeights, protect_fraction: f32) -> Self; // new, backend=Canonical
}
```

## Feature flags

- `ruvector-agent-memory/mincut-forget` (existing, off by default):
  unchanged trigger; now additionally enables `ruvector-mincut`'s
  `canonical` feature (small compile-time cost, zero behavior change when
  `mincut-forget` itself is off).
- No new feature flag introduced.

## Security

No change to the security posture documented in ADR-345:
`witnessed_compaction`'s eviction witness chain is backend-agnostic and was
re-verified at 100%/20 tamper-detection trials under the canonical
backend. `SourceAnchoredMinCut` performs no I/O, no new external
dependencies, and (per `ruvector-mincut`'s existing `canonical` feature
scope) no unsafe code beyond what the crate already ships.

## Governance

Same governance posture as ADR-345: this policy remains
feature-gated, off by default, and unpromoted. This ADR does not authorize
default-on behavior for `MincutGatedForgetting` in any consuming
application.

## Failure modes

See nightly README "Failure modes considered" and "Limitations". In
summary: this ADR's claims are scoped to the exact corpus/seed/thresholds
tested; larger corpora, different topologies, or a local (rather than
global) cut signal are explicitly out of scope and flagged as follow-ups,
not claimed as covered.

## Migration

None required — fully additive, default behavior unchanged for all
existing callers of `MincutGatedForgetting::soft`/`hard`.

## Rollback

Revert this commit; `MincutBackend`/`soft_canonical`/`hard_canonical` and
the two new example binaries are the only additions, with no changes to
existing default-path behavior to unwind.

## Rejection criteria

This ADR's canonical-backend recommendation would need revisiting if a
future, larger-scale run of `mincut_canonical_probe`/
`mincut_gated_forgetting_bench_canonical` found the canonical backend's
build-time regression (observed growing with `n` in the scaling probe)
eventually overtakes its cut-computation advantage at some untested corpus
size — flagged as "Next research" item 2, not observed within the n<=400
range tested here.

## Open questions

1. Does a local (Gomory-Hu-tree or per-cluster) structural signal clear
   the bridge-survival gate where the global min-cut did not? (Next
   nightly topic, not answered here.)
2. Does the canonical backend's advantage hold at corpora beyond n=400
   vertices?
3. What is the memory (RSS) delta between the two backends at matched
   corpus sizes? Not measured this run.
4. Does the `canonical` feature build and perform acceptably under
   `wasm32-unknown-unknown`? Not exercised this run.
