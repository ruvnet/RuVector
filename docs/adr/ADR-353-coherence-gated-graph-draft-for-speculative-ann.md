# ADR-353: Coherence-Gated Graph Draft for Speculative ANN — Negative Result

## Status

Rejected (with retained evidence). 2026-10-21.

The graph-draft half of this proposal (replacing a linear O(n) draft scan
with bounded graph beam traversal) is validated and recommended for
follow-up integration work. The coherence-gating half (pruning that
traversal with a traversal-direction threshold) is falsified for the tested
configuration and is not promoted.

## Context

`ruvector-speculative-ann` (nightly research, 2026-07-27) implements a
speculative-decoding-style ANN search: a cheap draft proposes a candidate
pool, an exact f32 verify re-ranks it. Its own "What to Improve Next" section
named an open item: replace the draft's linear O(n) u8 scan with
coherence-gated HNSW traversal (`ruvector-coherence-hnsw`'s technique) for
O(log n) draft cost.

This ADR documents the result of implementing and measuring that proposal in
`crates/ruvector-coherence-speculative-ann`.

## Hypothesis

```
Given a 4,000-vector, 64-dim clustered corpus with a navigable small-world
draft graph (M=16 local + 6 long-jump edges per node, u8-quantized ranking
distances) and a fixed distant entry point,

when the draft stage applies a traversal-coherence expansion gate
(ruvector-coherence-hnsw's technique) instead of expanding every popped
candidate's neighbors unconditionally,

then average per-query draft cost (graph-traversal expansions) should fall by
at least 10% relative to the ungated graph draft,

subject to recall@10 remaining within 2 percentage points of the ungated
graph draft's recall@10, and overall speculative-pipeline latency remaining
below the LinearFull exact-scan baseline.
```

## Decision

Reject coherence gating as implemented. Accept the graph-draft substitution
on its own.

## Evidence

Benchmark: `cargo run --release -p ruvector-coherence-speculative-ann --bin
benchmark`. Dataset: 20 clusters x 200 vectors, dims=64, std=0.15, 300
queries, k=10, seed=0xC0DECAFEBABE7777. Hardware: x86_64 Linux container,
release build, `rustc` per workspace `rust-version = "1.77"` floor (actual
toolchain reported by `rustc -V` at run time; see research README for the
exact string).

| Variant | Recall@10 | Mean (us) | Draft cost (expansions/query) |
|---|---|---|---|
| LinearFull (exact) | 1.000 | 324.5 | n/a (O(n)=4000) |
| GraphDraft (ungated) | 0.998 | 73.8 | 54.2 |
| GraphDraft + coherence t=0.00..0.45 | 0.998 | 59.9-96.1 | 54.0-54.2 (noise-level) |
| GraphDraft + coherence t=0.52 | 0.985 | 82.8 | 53.2 (-1.8%) |
| GraphDraft + coherence t=0.55 | **0.875** | 57.0 | **47.2 (-12.9%)** |
| GraphDraft + coherence t=0.60 | 0.623 | 40.7 | 33.0 (-39%) |
| GraphDraft + coherence t=0.80 | 0.080 | 12.8 | 4.3 (-92%) |

Full sweep (13 thresholds) and methodology in
`docs/research/nightly/2026_10_21_coherence_graph_draft_speculative_ann/README.md`.

Finding: there is a sharp recall cliff between t=0.52 and t=0.60. No
threshold delivers both a reduction large enough to be distinguishable from
per-query pop-order noise (>=10%) and a recall within 2pp of the ungated
graph draft. The only configuration clearing the 10% cost-reduction bar
(t=0.55, -12.9%) drops recall@10 by 12.3 percentage points — a REJECT against
the acceptance threshold declared before benchmarking.

Root cause (traced in `src/bin/diag.rs`): the draft's priority-queue beam
already pops near-query candidates first (ranked by u8 distance) and
terminates as soon as its bounded pool fills and overflows — by construction,
nodes already close to the query tend to have high traversal coherence
relative to a distant entry, so low-to-moderate thresholds gate nothing. Once
a threshold is high enough to gate some of those near-query nodes too, it
starts rejecting exactly the expansions the search needs to converge, and
recall collapses abruptly rather than degrading gracefully.

The graph-draft substitution itself (no gating) is validated independent of
the gating question: 54.2 average distance evaluations vs. 4,000 for a
linear scan (~74x fewer), 0.998 recall@10 against the exact baseline, and
~3-4x lower mean wall-clock latency than `LinearFull` on this dataset size.

## Consequences

- Do not add a traversal-coherence gate to `ruvector-speculative-ann`'s draft
  stage in its current single-fixed-threshold form.
- Do pursue graph-based (vs. linear-scan) drafting for `ruvector-speculative-ann`
  — this ADR's evidence supports it as a real latency win with no measured
  recall cost at this scale.
- The specific mechanism that would need to change before coherence gating is
  worth re-testing: either (a) a per-node adaptive threshold conditioned on
  beam convergence state (as `ruvector-coherence-hnsw`'s `AdaptiveCoherenceSearch`
  already implements, not evaluated here), or (b) gating based on something
  other than direction-from-a-single-fixed-entry, since a single entry point
  makes "coherent" and "close to query" nearly synonymous once the beam has
  partially converged.

## Alternatives Considered

1. **AdaptiveCoherenceSearch-style dynamic threshold** — not implemented in
   this pass; flagged as the most promising next experiment (see research
   README, "Next Research").
2. **Multiple randomized entry points** — would break the "entry defines a
   single direction" degeneracy that causes the cliff; not implemented this
   pass (adds a parameter dimension outside the declared scope).
3. **Keep the linear u8 draft (status quo in `ruvector-speculative-ann`)** —
   still valid; this ADR does not touch that crate's production code path.

## Implementation Plan

None — no production integration is recommended from this ADR. The new crate
(`ruvector-coherence-speculative-ann`) remains a standalone research PoC
under `crates/`, not wired into `ruvector-agent-memory` or any production
retrieval path.

## API Shape

n/a (research PoC; no public production API changes).

## Feature Flags

n/a (new standalone crate; no existing crate's default behavior changed).

## Benchmark Evidence

See Evidence section above and the research README's full 13-point threshold
sweep with raw benchmark output.

## Security

No new attack surface: the crate reads only its own deterministically
generated in-memory corpus, no network or filesystem I/O, no secrets.

## Governance

Research-only; not promoted to production. Rejection evidence retained under
`docs/research/nightly/2026_10_21_coherence_graph_draft_speculative_ann/`
and this ADR so a future nightly run does not have to rediscover the same
cliff from scratch.

## Failure Modes

The one found: the hard recall cliff described above. Documented so a future
attempt knows to test adaptive thresholds or multi-entry variants rather than
a single fixed threshold.

## Migration

n/a — nothing to migrate; no existing crate was changed.

## Rollback

n/a — the new crate can simply be removed (it has no dependents) if its
research value is judged not worth maintaining.

## Rejection Criteria

Already applied: coherence gating is rejected because the measured minimum
meaningful cost reduction (>=10%) correlates with a recall drop that fails
the declared acceptance band (within 2pp of ungated graph draft).

## Open Questions

1. Does an adaptive (beam-progress-conditioned) threshold, as already
   implemented in `ruvector-coherence-hnsw::AdaptiveCoherenceSearch` but not
   evaluated in this draft-graph context, find a usable middle ground?
2. Does the cliff persist at larger n (this PoC used n=4,000; the original
   `ruvector-speculative-ann` benchmark used n=10,000)?
3. Does randomizing or diversifying the entry point remove the
   "near-query implies coherent" degeneracy that produced the cliff?
