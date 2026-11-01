# ADR-350: Content-Correlation-Augmented Memory Arbitration

- **Status**: Proposed
- **Date**: 2026-11-01
- **Extends**: ADR-330 (CAMA-pattern correlation-aware memory arbitration)
- **Related**: `crates/ruvector-agent-memory/src/arbitration.rs`,
  `crates/ruvector-agent-memory/src/content_correlation.rs`,
  `docs/research/nightly/2026-11-01-content-correlation-arbitration/README.md`
- **External anchor**: CAMA, "Beyond Memory Majority: Latent-Source
  Reasoning for Multi-Agent Memory Arbitration" (arXiv:2608.19701)

## Context

ADR-330 implemented declared-ancestry memory arbitration
(`crate::arbitration::arbitrate`): a retrieved memory set is clustered into
independent evidence lineages by the connected components of the
`causal_parents` ancestry graph, and effective confidence is scored
per-lineage rather than per-memory, so repeated derivation from one source
counts once.

ADR-330's own module documentation names a deliberate, acknowledged
limitation:

> "Correlation is detected from declared `causal_parents`, never from
> content. A memory that copies another memory's content while declaring no
> parent is, to this module, a fresh root ... twenty agents that each
> re-assert one rumor's content without citing it therefore arbitrate to
> twenty lineages with no correlation downgrade ... Deployments that need
> adversarial robustness must pair this with content-level near-duplicate
> detection or authenticated citation at ingest."

This is locked into the test suite
(`tests/arbitration.rs::content_copying_yields_no_correlation_downgrade_under_default_reliability`):
20 parentless copies of one claim score naive `18.0` / arbitrated `16.2`
and mint 20 lineages — the arbitration module cannot, by design, tell this
apart from 20 genuinely independent witnesses.

No prior work in this repository ships the named pairing. This ADR does.

## Hypothesis

```text
Given a retrieved memory set containing an astroturf cluster of N
parentless memories whose content is near-identical, with no declared
causal parent,

when content-similarity-based near-duplicate detection (cosine similarity
over caller-supplied embeddings, threshold 0.95) is layered on top of
declared-ancestry arbitration, either by hard-merging near-duplicate
lineages or by discounting a flagged lineage's weight,

then the astroturf cluster's lineage count drops by at least 70% relative
to declared-ancestry-only arbitration, while a disjoint set of genuinely
independent memories (orthogonal embeddings) sees 0% false merges,

subject to: the added pass costing under 50ms at N=500 memories
(release build) and the downgrade-only invariant (arbitrated ≤ naive)
holding throughout.
```

## Decision

1. **Add `crates/ruvector-agent-memory/src/content_correlation.rs`** as a
   new, purely additive module. It does not modify `arbitration.rs`,
   `fusion.rs`, or `observation.rs` in any way — it calls
   `arbitration::arbitrate` as an opaque first pass and transforms its
   output.
2. **Two modes, both downgrade-only by construction**:
   - `CorrelationMode::HardMerge` — any two declared lineages with a
     cross-pair cosine similarity ≥ `similarity_threshold` are merged
     (transitively, via union-find). Merged weight = `max` of constituent
     lineage weights (itself already a max over members), so the merged
     lineage's weight can never exceed either input — merging only ever
     reduces `Σ lineage weights`.
   - `CorrelationMode::SoftDiscount { strength }` — lineages are not
     merged; a lineage with a detected near-duplicate elsewhere has its
     weight multiplied by `1 - strength` (`strength ∈ [0,1]`), preserving
     the lineage count (the documented "real diagnostic") while still
     reducing the score.
3. **Embeddings are caller-supplied**, keyed by `NodeRef` in a
   `BTreeMap<NodeRef, Vec<f32>>`. `AtomicObservation` carries no embedding
   field (by ADR-320 design — it is content-addressed over an opaque
   `payload`), so bridging content to a vector is explicitly the
   integrator's responsibility, matching how `crate::scoring` /
   `CoherencePolicy` and the causal-fusion layer are already two separate
   subsystems in this crate.
4. **Fail loud on missing or invalid input**: a memory in the arbitrated
   verdict with no corresponding embedding is a hard error
   (`ContentCorrelationError::MissingEmbedding`), never silently treated as
   "no correlation". An out-of-range `similarity_threshold` or
   `SoftDiscount` `strength` is rejected before any scoring
   (`ContentCorrelationError::InvalidConfig`). The downgrade-only invariant
   is re-checked at runtime after the transform and fails loud
   (`DowngradeInvariantViolated`) rather than clamping, matching
   `arbitration.rs`'s existing posture.
5. **No Darwin/MetaHarness/Flywheel infrastructure was built or invoked**
   for this experiment. This repository already has deep, working
   reward-hack-defense and promotion-gate infrastructure
   (`crates/sona/src/darwin_guard.rs`, ADR-271's EWC/weightAdapter Darwin
   line, ADR-335's per-repo optimization manifest). This module's design
   space (two hand-specified variants, no tunable continuous parameter
   search over a labeled dataset) did not warrant invoking it; the
   hypothesize → implement → benchmark → adversarially test → accept
   loop was executed by hand instead.

## Evidence

Benchmark: `cargo run --release -p ruvector-agent-memory --example
content_correlation_bench` (rustc 1.97.0, 4-core x86_64 Linux 6.18.44 VM).
Full methodology and raw output in the nightly README (linked above).
Headline results:

| Scenario | Baseline | HardMerge | Acceptance |
|---|---|---|---|
| Astroturf (N=20, no parent) | 20 lineages, arbitrated 16.2 | 1 lineage, arbitrated 0.81 (95% reduction) | PASS (≥70% required) |
| Independent (N=20, orthogonal) | 20 lineages | 20 lineages, 0 false merges | PASS (0% required) |
| Mixed (10 astroturf + 10 independent) | 20 lineages | 11 lineages (exactly 1 + 10) | PASS |
| Latency, N=500, worst-case L≈N | — | p95 ≈ 20.7ms | PASS (<50ms required) |

Test evidence: `cargo test -p ruvector-agent-memory --lib --tests` — 84
passed, 0 failed (45 pre-existing lib tests + 1 bench-bin test + 10
pre-existing arbitration integration tests + 8 pre-existing fusion
integration tests + 7 new content-correlation integration tests + 13
pre-existing ledger tests). Zero regressions to any pre-existing test.
`cargo clippy -p ruvector-agent-memory --all-targets`: zero warnings
attributable to this change.

Acceptance result: **ACCEPT**.

## Consequences

**Positive**: closes a documented, acknowledged gap with a small
(~340-line), purely additive, already-tested module; zero regression risk
to the carefully invariant-checked `arbitration.rs`; zero new dependencies;
both variants provably downgrade-only.

**Negative / risks**: paraphrase/translation-blind (only catches
near-verbatim content in embedding space); single threshold (`0.95`)
validated against one synthetic astroturf shape and one synthetic
independent shape, not real topically-related-but-independent content;
`O(N²)` worst-case cost, validated only to `N=500`; embedding-quality-bound
(garbage embedding in, garbage detection out).

## Alternatives

1. **Modify `arbitration.rs` directly** to add an embedding field and
   inline correlation step. Rejected — that file's invariants
   (downgrade-only, non-finite choke points) and locked test suite make an
   orthogonal-concern change there higher-risk than an additive layer, for
   no capability the layer doesn't already provide.
2. **A general Darwin-evolved promotion-gate crate** for this and future
   nightly experiments. Rejected after finding `crates/sona/src/
   darwin_guard.rs` and ADR-271/335 already implement this in depth;
   building a second would duplicate working infrastructure.
3. **Fix ADR-335's `#920` TypeScript `hard_regression` gate bug.** Out of
   scope — TypeScript harness tooling, and this nightly process's
   constraints require Rust production code.

## Implementation plan

Implemented in full in this change: `src/content_correlation.rs`,
`tests/content_correlation.rs`, `examples/content_correlation_bench.rs`,
`lib.rs` wiring. No further phased rollout needed — the module is
feature-complete for its stated scope and already exported.

## API shape

```rust
pub enum CorrelationMode {
    HardMerge,
    SoftDiscount { strength: f32 },
}

pub struct ContentCorrelationConfig {
    pub similarity_threshold: f32, // finite, [-1, 1]
    pub mode: CorrelationMode,
}

pub fn arbitrate_with_content_correlation(
    graph: &CausalEpisodicGraph,
    memories: &[NodeRef],
    embeddings: &BTreeMap<NodeRef, Vec<f32>>,
    config: &ArbitrationConfig,
    content: &ContentCorrelationConfig,
) -> Result<ArbitrationOutcome, ContentCorrelationError>;
```

## Feature flags

None added. The module has no optional dependency and is always compiled
(unlike `mincut-forget` / `proof-gate`, which gate on external crates this
module does not need).

## Benchmark evidence

See "Evidence" above and the nightly README for full raw output,
methodology, and hardware/software details.

## Security

No new trust boundary. `embeddings` is treated as untrusted caller input
exactly like `memories` — a missing entry is a hard error, never a silent
"assume independent". Both transforms are provably downgrade-only and
re-verified at runtime. Does not claim to fully close the astroturfing
threat model (see Consequences/Negative) — it is the additional signal
ADR-330's own docs call for, not a replacement for authenticated citation
at ingest.

## Governance

No change to `ArbitrationOutcome`'s shape or the sufficiency-threshold
semantics; this module changes the lineage count and weights the existing
decision is computed from, not the decision structure itself.

## Failure modes

See the nightly README's "Failure modes" section: paraphrase blindness,
embedding-quality dependence, quadratic cost ceiling past `N=500`,
single-threshold validation.

## Migration

None required — purely additive, opt-in API. No existing caller of
`arbitrate()` is affected.

## Rollback

Trivial: the module is self-contained and additively exported; removing
`pub mod content_correlation;` and its re-exports from `lib.rs` (and
deleting the three new files) fully reverts this change with no effect on
any other module.

## Rejection criteria

Would reject promotion to default-on behavior if: a production deployment
shows a materially nonzero false-merge rate on real independent content at
`similarity_threshold = 0.95`; or the `O(N²)` cost becomes a measured
bottleneck at realistic `N` without an ANN/LSH prefilter in place. Neither
is a reason to reject this ADR's additive, opt-in API as currently scoped.

## Open questions

1. Does a single global `similarity_threshold` generalize across embedding
   models and content domains, or does it need to be workload-specific?
2. Should `SoftDiscount`'s `strength` itself be learned/evolved once a
   labeled correlated/independent dataset exists (candidate for this
   repository's existing Darwin machinery, train/eval-disjoint per
   `crates/sona/src/darwin_guard.rs`'s contamination guard)?
3. Is an ANN/LSH-prefiltered near-duplicate step (composing with
   `ruvector-mincut`'s k-NN construction) the right path past the `O(N²)`
   ceiling, and at what `N` does it become necessary in practice?

## What should become stable API

`ContentCorrelationConfig`, `CorrelationMode`, and
`arbitrate_with_content_correlation` as currently shaped — they are
additive, tested, and require no further design iteration to be useful as
opt-in API.

## What should remain experimental

The specific `similarity_threshold` default and the choice between
`HardMerge`/`SoftDiscount` as a deployment default — both are
workload-dependent per the Open Questions above, and no default is
asserted by this ADR (callers must choose explicitly; there is no
`Default` impl for `ContentCorrelationConfig`).

## What evidence supports promotion

The benchmark table above, run against the exact pre-existing locked
astroturf fixture in `tests/arbitration.rs`, plus a disjoint false-merge
fixture and a mixed-scenario fixture, all with zero regressions to the 77
pre-existing tests in the crate.

## What evidence would cause rejection

See Rejection criteria.

## Why does this belong in RuVector?

`ruvector-agent-memory` is the one crate in the workspace with both the
vector/embedding primitives and the causal-fusion/provenance primitives in
one place; this module is the natural bridge between them that ADR-330
explicitly left for later.

## Why is this more than an experiment?

It is a complete, tested, zero-regression, already-exported public API
addition with measured evidence against the exact documented gap it
targets — not a prototype requiring further work to be usable.
