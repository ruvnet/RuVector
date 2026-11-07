# ADR-353: Coherence-Gated Mixed-Precision Quantization — Rejected

## Status

Rejected. Research-tier experiment, not merged into any production path.
`crates/ruvector-coherence-quant` is retained as a standalone, tested
experiment harness and negative-result record; it is not a dependency of
any other crate and nothing in `ruvector-attn-mincut` or
`ruvector-turboquant` changed.

## Context

`ruvector-attn-mincut` provides a dynamic min-cut gate (`dynamic_min_cut`)
originally built as an alternative to softmax attention: it builds a
weighted graph from attention logits and decides, per call, whether a
cheap separating cut exists and is worth applying. `ruvector-turboquant`
provides Turbo4 (4-bit) and 1-bit quantization codecs for vector storage
(ADR-296/297). Agent memory stores accumulate highly uneven local
redundancy — near-duplicate observations cluster tightly, boundary
observations don't — which uniform quantization doesn't exploit.

The 2026-10-24 nightly research run
(`docs/research/nightly/2026-10-24-coherence-gated-quant/README.md`) tested
whether `dynamic_min_cut`'s existing gate could supply a per-vector
coherence score to drive Turbo4-vs-1-bit precision allocation, reusing
both crates' public APIs unmodified. It connects three ecosystem
capabilities (attention-gating infrastructure, vector quantization, agent
memory workload shape) without inventing a new coherence metric, and
deliberately avoids duplicating the existing mincut-based nightly topics
(2026-09-02 memory admission, 2026-09-05 forgetting, 2026-09-11 partition
determinism), which use mincut for edge-level gating/admission, not
continuous per-vector precision scoring.

## Hypothesis

```text
Given a clustered agent-memory corpus (40 Gaussian clusters x 150 vectors,
dim 128, seed 42),

when per-vector Turbo4-vs-1-bit quantization precision is allocated by a
mincut-derived local coherence score (candidate_A) instead of uniform
Turbo4 (baseline),

then recall@10-per-byte should improve by at least 15% relative to
baseline,

subject to recall@10 remaining >= 0.9x baseline recall@10 and mean query
latency remaining <= 2x baseline.
```

Coherence score: for each cluster, build a directed cosine-similarity
self-graph (diagonal removed), order nodes nearest-to-farthest from the
cluster centroid, call the unmodified `ruvector_attn_mincut::dynamic_min_cut`
with source = nearest node, sink = farthest node, and define
`coherence(i) = kept_incident_edges(i) / total_incident_edges(i)` from the
returned `GatingResult.keep_mask`.

## Decision

**Do not integrate `dynamic_min_cut`-derived coherence into quantization
precision allocation.** The measured result is REJECT:

```
variant                                     bytes/vec  recall@10   mean_us
baseline (uniform Turbo4)                        72.0      0.657     252.5
candidate_A (coherence-gated)                    38.4      0.301     487.9
candidate_B (coherence-gated + residual)         65.7      0.280     534.7

recall@10 per byte: baseline=0.009118 candidate_A=0.007839 (-14.0%)
ACCEPTANCE RESULT: REJECT
```

`mean coherence score: 1.0000` (exactly uniform over all 6000 vectors) is
the signature of total signal collapse, confirmed by two controlled toy
graphs run directly against the unmodified `dynamic_min_cut`:

| Toy graph | `cut_cost` returned | Gate fired? |
|---|---|---|
| 6-node complete digraph, uniform weight 1.0 | `0` | No |
| Two 3-cliques + one weight-0.02 bridge edge | `0.02` (exactly the bridge) | Yes |

The bridge case proves the mechanism correctly finds and reports a true
min-cut when one exists cheaply. The dense case proves the limitation:
`GatingResult.cut_cost` reports `0` specifically when the λ-gate declines
to cut, which happens whenever the graph's true min-cut exceeds
`λ · mean_weight` — exactly the case for a dense, single-community,
unimodal-cluster similarity graph (confirmed on the real corpus: one
150-point cluster has 22,350/22,500 possible edges present at mean cosine
weight 0.91, with true min-cut above the λ=0.5 threshold of 0.45). This
held under a Gaussian-RBF edge-weight reparameterization and a λ sweep
from 0.1 to 0.8 — it is structural (min-cut detects *between-community*
boundaries; a center-to-fringe density gradient inside one unimodal blob
is not a community boundary), not a tuning problem.

## Evidence

Full methodology, raw benchmark stdout, memory/performance accounting, and
the toy-graph root-cause diagnostics are in
`docs/research/nightly/2026-10-24-coherence-gated-quant/README.md`.
Reproduce with:

```bash
cargo test -p ruvector-coherence-quant
cargo run --release -p ruvector-coherence-quant --example benchmark
```

Both the Turbo4 baseline encoding and the brute-force ground truth use
exact, unmodified library code (`ruvector_turboquant::Turbo4Codec`,
plain f32 L2) — no mocks on the measured path.

## Consequences

- No production code changed. `ruvector-turboquant`'s uniform Turbo4
  remains the shipped default for vector storage; `ruvector-attn-mincut`
  is unchanged.
- `crates/ruvector-coherence-quant` stays in the workspace as a tested,
  reproducible negative-result artifact (5 unit tests + 3 integration
  tests, `cargo clippy` clean) so a future attempt at this idea does not
  need to re-derive the root cause from scratch.
- Flags a real documentation gap in `ruvector-attn-mincut`:
  `GatingResult.cut_cost` is "cost of the cut that was actually applied
  (0 if the gate declined to cut)", not "the graph's true min-cut value" —
  worth a doc-only fix there independent of this ADR's outcome.
- An earlier implementation of the precision-allocation code used a
  percentile *value* threshold instead of exact-rank selection and
  silently selected 100% of vectors under the (common, as it turned out)
  case of tied coherence scores; this was caught by
  `tests/correctness.rs` before the accepted benchmark ran and fixed
  (exact-rank selection with an index tiebreak). This is itself a small,
  reusable lesson: percentile-value thresholds are unsafe on any scoring
  function that can produce heavy ties.

## Alternatives

- **Bypass the λ-gate via `DinicSolver` directly** for the raw,
  unconditional min-cut: not possible — `DinicSolver::new` is private in
  `ruvector-attn-mincut`; only the gated `dynamic_min_cut` is public. Not
  worked around with a hand-rolled max-flow reimplementation, since that
  would no longer be testing reuse of the actual ecosystem primitive.
- **Tune λ instead of root-causing**: swept 0.1–0.8 against both cosine
  and RBF edge weights; `cut_cost` stayed `0` throughout, confirming the
  collapse is structural, not a parameter-search problem.
- **Multi-modal sub-clusters per topic** (which would give min-cut a real
  community boundary, per the bridge toy case) was considered but
  deliberately not substituted into the registered hypothesis after
  seeing the unimodal result — that would validate a different, narrower
  claim than the one formalized before benchmarking. Recorded as the
  immediate next experiment instead (see research README §35).

## Implementation plan

None — rejected. Future implementation plan if a fix is found (e.g. a
bimodal-corpus retest in README §35 succeeds): land as an opt-in
`ruvector-turboquant` encoding mode behind its own feature flag, never as
a default-on change to the existing Turbo4 path, with the same
brute-force-verified recall testing this harness already performs.

## API shape

None shipped. `ruvector-coherence-quant`'s current API
(`dataset::generate_corpus/generate_queries`, `coherence::corpus_coherence`,
`allocator::encode_corpus/estimate_distance`, `metrics::recall_at_k/percentile_ns`)
is experiment-harness-only and explicitly not stable.

## Feature flags

None added to any production crate.

## Benchmark evidence

See Evidence section above and the linked research README §14 for full
raw output, §16–17 for memory/performance accounting.

## Security

No new attack surface (in-process deterministic arithmetic, no I/O, no
secrets, not wired into any serving path). See research README §20 for
the one documentation-adjacent note.

## Governance

Blocks (not merely cautions against) wiring `dynamic_min_cut` into any
future agent-memory precision/admission/forgetting decision on
unimodal-cluster-shaped data without first addressing the gate-collapse
finding here. Does not contradict or require revisiting the existing
mincut-based nightly topics (ADR-344/345/346), which use the primitive for
a different purpose (edge-level gating, not continuous per-vector scoring)
and are not shown to share this failure mode.

## Failure modes

Documented in research README §18 (mixed-precision ranking calibration
bias between Turbo4/1-bit/exact distance estimates — unquantified, not the
primary cause of this rejection but a real limitation of the harness as
built) and §15 (the root cause itself).

## Migration

None — no production code changed.

## Rollback

None needed — nothing was promoted.

## Open questions

1. Do real embedding corpora (as opposed to this run's synthetic unimodal
   Gaussian clusters) exhibit enough bimodal/multimodal sub-structure per
   topic for `dynamic_min_cut`'s gate to fire usefully? Untested here.
2. Would a plain local-density coherence score (e.g. k-NN average
   distance), decoupled from min-cut entirely, succeed where this
   mincut-based approach failed? Untested here; proposed as independent
   follow-up in research README §35.
3. Should `ruvector-attn-mincut` document (or change) `GatingResult.cut_cost`'s
   "0 means ungated, not 0-cost" semantics? Filed as a doc-gap observation,
   not acted on in this ADR.
