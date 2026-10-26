# ADR-353: Reject Fisher-Weighted ANN for EWC++ Policy-Entry Lookup

## Status

Rejected. This ADR records a negative result and closes an unresolved
assumption in ADR-002, rather than proposing new production code.

## Context

ADR-002 (`ruvllm-integration`) specifies a `PolicyEntry` schema for
RuvLLM's "Policy Memory Store": a single Ruvector HNSW index holding
quantization, router, pattern-bank, and EWC-type policies, each carrying a
semantic `embedding: Vec<f32>` for the shared "HNSW INDEX LAYER (Unified
Search)" and, for EWC-type entries, `fisher_diagonal: Option<Vec<f32>>`.
ADR-002 does not specify whether that Fisher diagonal should change how
the shared index *searches* for EWC-type entries, or whether it is purely
descriptive payload alongside an ordinary cosine/L2 index.

Separately, `crates/sona`'s `EwcPlusPlus` (EWC++) already performs
Fisher-weighted scoring in production today: `apply_constraints` and
`apply_anchor` loop over `task_memory`, a `VecDeque<TaskFisher>` bounded by
`EwcConfig::max_tasks` (default 10). That is a correct, cheap linear scan
at that scale. ADR-002's own stated motivations ("cross-session knowledge
transfer", "federated learning") imply this task bank could grow far past
10 entries for a long-running or federated agent, which raises the
question this ADR answers: does an approximate (HNSW-style) index, built
using the same Fisher-weighted scoring EWC++ already trusts, reproduce the
linear scan's answer closely enough, fast enough, to be worth building?

Full methodology, implementation, and raw benchmark output:
`docs/research/nightly/2026-10-26-fisher-weighted-ewc-ann/README.md` and
`bench_output.txt` in the same directory. New crate:
`crates/ruvector-fisher-ann`.

## Hypothesis

```text
Given a bank of N synthetic EWC task snapshots (weights + per-task Fisher
diagonal + importance) across 40 task families with sparse, family-
specific important dimensions,

when a Fisher-weighted HNSW index (candidate A) is used for nearest-prior-
task lookup instead of (a) a Fisher-weighted brute-force linear scan at
scale (baseline, generalizing EwcPlusPlus's own production task_memory
scan) or (b) a plain-L2 HNSW over the same points (candidate B, an
ablation),

then candidate A's mean query latency should be at least 5x lower than
baseline at N=10,000,

subject to candidate A's recall@10 against the Fisher-weighted ground
truth staying >= 0.90, its mean single-task EWC regularization-loss gap
staying below 5% of the mean ground-truth loss, and candidate A
measurably beating candidate B on recall@10 and/or loss gap (otherwise
the Fisher weighting itself is not what is providing any value measured).
```

## Decision

**Do not build a Fisher-weighted ANN index for `PolicyEntry`.** Concretely,
for any future implementation of ADR-002's Policy Memory Store:

1. Index `PolicyEntry.embedding` with an ordinary symmetric metric
   (cosine or L2), identically to every other policy type in the shared
   HNSW layer — no EWC-specific metric branch.
2. Store `fisher_diagonal` as inert payload on the record, fetched by id
   after a plain semantic lookup, not folded into the index's distance
   function.
3. Keep EWC++'s existing Fisher-weighted task selection
   (`apply_constraints`, `apply_anchor`, `regularization_loss`) exactly as
   the bounded linear scan it is today. This run's baseline measurements
   (261 µs – 1.5 ms mean, N up to 10,000, dim=128, on a 4-vCPU container)
   show that scan stays cheap for a long time past the current
   `max_tasks` default of 10.
4. If `max_tasks` is ever raised by orders of magnitude, the recommended
   design is a **shortlist-then-exact** pipeline — coarse unweighted ANN
   to retrieve a shortlist (e.g. top ~50), then exact Fisher-weighted
   scoring only within that shortlist — not a Fisher-weighted graph index.
   This is listed as future work (Next Research in the nightly report),
   not implemented here.

## Evidence

Full benchmark table in the nightly report; summary at N=10,000, dim=128,
40 families, 300 queries, release build:

| Variant | mean query latency | recall@10 (vs Fisher-weighted ground truth) | mean EWC loss gap |
|---|---|---|---|
| baseline (Fisher-weighted brute force) | 1,424.5 µs | 1.000 (is ground truth) | 0 |
| candidate A (Fisher-weighted HNSW) | 79.9 µs (17.8x faster) | **0.354** | **50,120.3** |
| candidate B (plain-L2 HNSW, ablation) | 337.3 µs | **0.388** | **0.0** |

Candidate A passes the speedup threshold (17.8x >= 5x) but fails recall
(0.354 vs required 0.90), fails the loss-gap threshold (50,120.3 vs. a
188.7 ceiling), and — decisively — loses the ablation: candidate B, which
*ignores* Fisher information entirely, beats candidate A on both quality
metrics. A diagnostic sweep of `ef_search` from 100 to 2,000 (equal to the
entire N=2,000 dataset) left candidate A's recall@10 completely flat at
0.352, ruling out a search-budget explanation.

**Root cause:** the Fisher-weighted scoring function
(`fisher[task] · (query - task.weights)² · importance[task]`) is
intentionally one-sided — it uses the *candidate's* Fisher diagonal, to
match what `EwcPlusPlus::regularization_loss` actually computes — which
makes it asymmetric and non-metric. HNSW's greedy construction and search
implicitly rely on an approximate triangle inequality to be navigable; a
one-sided score that lets a near-zero-Fisher candidate appear "close" to
almost any query regardless of raw distance breaks that assumption and
produces a poorly connected graph. Plain L2 (candidate B) is a genuine
metric, so standard HNSW theory applies to it, and in this dataset raw L2
still weakly correlates with family membership (every dimension, not just
the "important" ones, is built from a family-specific center), which is
why candidate B's recall against the *harder, more informative* Fisher-
weighted ground truth is nonetheless high.

## Consequences

### Positive

1. Closes ADR-002's open "how does `fisher_diagonal` interact with
   search" question with a measured answer instead of leaving it
   ambiguous for whoever implements the Policy Memory Store.
2. Prevents building and maintaining a Fisher-weighted ANN index that
   this evidence shows would silently misrank EWC anchor candidates at
   scale — a correctness risk for EWC's own forgetting-prevention
   guarantee, not just a performance footnote.
3. Establishes a reusable harness (`ruvector-fisher-ann`) and a
   production-cross-checked quality metric
   (`single_task_regularization_loss`, tested against the real
   `EwcPlusPlus::regularization_loss`) for testing the shortlist-then-
   exact design in a future run.

### Negative

1. The "cross-session"/"federated" scale ADR-002 anticipates (far beyond
   `max_tasks=10`) still has no validated scalable lookup design; this
   ADR only rules out one candidate.
2. The dataset is synthetic (no real fine-tuning loop was available in
   this sandbox); absolute recall/latency numbers are specific to this
   distribution, though the qualitative mechanism (one-sided score breaks
   graph navigability) is a structural property, not a dataset artifact.

## Alternatives Considered

1. **Symmetrize the Fisher-weighted metric** (average or min/max of both
   sides' Fisher diagonals) before building the graph — not attempted
   here because it changes the quantity being optimized away from
   `regularization_loss`'s actual semantics for either side; listed as
   Next Research in the nightly report.
2. **Add a public `TaskFisher`-injection constructor to `EwcPlusPlus`**
   so the benchmark could drive the real struct end-to-end — rejected in
   favor of a smaller, explicitly-tested equivalence function
   (`single_task_regularization_loss`), to avoid widening another crate's
   public API for a benchmark that turned out to falsify its own premise.
3. **Weight the semantic `embedding` dimension by a projected Fisher
   importance** instead of the raw weight space — rejected before
   implementation: `fisher_diagonal` is sized to the model's parameter
   space, with no principled correspondence to the embedding's dimensions.

## Implementation Plan / API Shape

None. This ADR's decision is "do not build X"; no new production API is
introduced. `crates/ruvector-fisher-ann` is a research-only crate (no
`sona` or `ruvector-agent-memory` changes), kept in the workspace as
reusable scaffolding for the Next Research items below.

## Feature Flags

None added.

## Benchmark Evidence

`docs/research/nightly/2026-10-26-fisher-weighted-ewc-ann/bench_output.txt`
— raw, unedited `cargo run --release -p ruvector-fisher-ann --bin
fisher_ann_bench` output, including the `ef_search` diagnostic sweep.

## Security

No secrets, credentials, or `.env` content touched. No existing test was
weakened, skipped, or deleted. No acceptance threshold was changed after
seeing results. The benchmark's ANN candidates have no access to the
ground-truth top-k during construction or search, so there is no
evaluation-information leak into the "solving path".

## Governance

This ADR is advisory to any future implementer of ADR-002's Policy Memory
Store; it does not change any currently-shipping public API.

## Failure Modes

If a future implementation ignores this ADR and builds a Fisher-weighted
ANN index anyway, the measured failure mode is silent: the index returns
*a* result quickly, with no error, that is frequently the wrong anchor
task — a correctness problem for EWC's forgetting-prevention guarantee
that would not surface as a crash or an obvious latency regression.

## Migration

Not applicable (no existing production code is changed).

## Rollback

Not applicable (nothing is shipped).

## Rejection Criteria

This ADR itself documents a rejection. Its own decision would be revisited
if Next Research item 1 (a symmetrized Fisher metric) or item 2 (the
shortlist-then-exact design) produces a passing result in a future
nightly run against the same acceptance thresholds.

## Open Questions

1. Does a symmetrized Fisher-weighted metric preserve enough of
   `regularization_loss`'s semantics to remain a meaningful ranking
   signal while restoring graph navigability?
2. At what `max_tasks` scale (if any, within realistic deployment
   horizons) does the current bounded linear scan actually become a
   measured bottleneck, justifying the shortlist-then-exact design's
   added complexity?
