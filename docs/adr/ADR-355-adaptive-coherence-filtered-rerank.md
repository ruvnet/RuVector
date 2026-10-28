# ADR-355: Adaptive Per-Node Coherence Gating for GNN Candidate Reranking (Rejected)

## Status

**Rejected** (hypothesis falsified by measurement). The implementation is retained as an
additive, opt-in reranker variant because it is harmless and marginally positive in some
conditions, but it is **not** promoted as a solution to the filtered-retrieval problem it
was designed to address.

## Context

`crates/ruvector-gnn-rerank` (ADR-194, nightly 2026-05-21) implements `CandidateReranker`
variants that post-process a noisy first-stage ANN candidate set by diffusing scores over
a candidate k-NN graph. `GnnMincutReranker` gates diffusion edges with one **global, fixed**
cosine-similarity threshold (`coherence_threshold`, default 0.50).

Nightly research 2026-10-28 investigated whether that fixed global threshold degrades when
a metadata-style filter (tenant scoping, ACLs, freshness windows, …) is applied upstream and
shrinks the candidate pool: a global bar calibrated for an unfiltered pool could leave many
filtered nodes with zero or few edges clearing it, collapsing the reranker toward the noisy
baseline exactly when reranking should matter most.

## Hypothesis

```
Given a synthetic clustered-vector candidate pool (seeded, deterministic) subjected to a
metadata-style filter at selectivity s ∈ {1.00, 0.50, 0.25, 0.10, 0.05},

when candidates are reranked with AdaptiveCoherenceReranker (per-node top-fraction edge
gating) instead of GnnMincutReranker (global fixed-threshold edge gating),

then mean recall@10 at low selectivity (s <= 0.10) should exceed GnnMincutReranker's by
>= 3 percentage points, averaged over >= 200 trials per selectivity level,

subject to:
  - recall@10 at s = 1.00 not regressing by more than 0.5 pp vs GnnMincutReranker,
  - mean latency not exceeding 2x GnnMincutReranker's,
  - `cargo test` / `cargo clippy -- -D warnings` staying green for the crate.
```

## Decision

Implement `AdaptiveCoherenceReranker` as an additive fourth `CandidateReranker` variant
(does not replace or modify `GnnMincutReranker`), ship it with tests, and run the fixed
benchmark described below. Based on the measured result, **do not** recommend
`AdaptiveCoherenceReranker` as a replacement or default for filtered-retrieval reranking;
keep it available as an alternative with no specific claimed advantage, and record the
falsification so future nightly runs do not re-litigate this exact framing without new
information.

## Evidence

Benchmark: `cargo run --release -p ruvector-gnn-rerank --bin filtered_rerank_bench`
(raw output: `docs/research/nightly/2026-10-28-adaptive-coherence-filtered-rerank/raw_benchmark_output.txt`).
200 successful trials per selectivity level, seeded (`base_seed=777`, `corpus_seed=42`),
5,000-vector / 128-dim / 20-cluster synthetic corpus, `k_graph=8`, `k=10`.

| s | NoisyScore recall@10 | GnnMincut recall@10 | AdaptiveCoherence recall@10 | Adaptive − Mincut |
|---|---|---|---|---|
| 1.00 | 29.80% ± 0.78% | 38.55% ± 0.95% | 40.05% ± 1.02% | **+1.50 pp** |
| 0.50 | 33.00% ± 0.93% | 38.75% ± 1.10% | 38.85% ± 1.11% | +0.10 pp |
| 0.25 | 40.40% ± 0.96% | 43.90% ± 1.05% | 42.85% ± 1.04% | −1.05 pp |
| 0.10 | 60.25% ± 0.91% | 61.00% ± 0.93% | 60.80% ± 0.93% | −0.20 pp |
| 0.05 | 84.75% ± 0.92% | 84.60% ± 0.89% | 85.15% ± 0.90% | +0.55 pp |

Mean gain at low selectivity (s ∈ {0.10, 0.05}): **+0.18 pp**, 95% CI ≈ [−1.6, +2.0] pp —
entirely below the required +3.0 pp threshold. **Hypothesis condition rejected.**

Subject-to conditions (all independently satisfied, but moot given the primary rejection):
unfiltered regression = **−1.50 pp** (Adaptive is *better*, so the ≤0.5 pp regression budget
holds); latency ratio ≤ 1.00x at every selectivity level (Adaptive is never slower, often
slightly faster); `cargo test -p ruvector-gnn-rerank` (24 tests) and
`cargo clippy -p ruvector-gnn-rerank --all-targets -- -D warnings` both green.

Bounded Darwin-style sweep (4 candidates, 1 generation, 150 trials each, fitness per
ADR research doc §Darwin): best mutation (`keep_fraction=0.75, alpha=0.50`) scored fitness
0.989 vs parent's normalized 1.000 — **no mutation beat the parent**. Parent defaults
(`alpha=0.60, keep_fraction=0.50, min_keep=1`) retained unchanged.

**Overall determination: REJECT.** See the nightly research doc for the likely root cause
(intra-cluster cosine similarity in this synthetic workload stays well above the 0.50 global
bar even after uniform random filtering, so `GnnMincutReranker`'s global threshold does not
actually collapse the way the hypothesis predicted for this filter model).

## Consequences

- No change to any existing reranker's default behaviour. `GnnMincutReranker` remains the
  crate's coherence-gated recommendation.
- `AdaptiveCoherenceReranker` is available for callers who want it, with its real,
  unremarkable (statistically indistinguishable-from-`GnnMincutReranker`) performance
  profile documented, not oversold.
- Future nightly work targeting the filtered-retrieval sparsification problem should either
  (a) use a filter model that is correlated with cluster membership (e.g. filtering out
  whole sub-clusters or tenants, which is more realistic for ACL/tenant-scoped filters than
  uniform-random thinning) and re-test, since uniform-random filtering preserves local
  cluster structure almost by construction, or (b) target pool sizes small enough that
  `k_graph=8` itself becomes the binding constraint rather than the similarity threshold.

## Alternatives

- Lowering `GnnMincutReranker`'s default `coherence_threshold` for filtered contexts instead
  of adding a new reranker — not pursued because it is a global parameter change with no
  principled per-pool justification, and the measured results indicate the proposed
  mechanism was not the bottleneck anyway.
- Multi-hop diffusion before measuring per-node sparsification effects — not pursued per
  the hypothesis's fixed single-hop scope (kept comparable to `GnnMincutReranker`'s structure).

## Implementation Plan

Already implemented in this change set:
- `crates/ruvector-gnn-rerank/src/reranker.rs`: `AdaptiveCoherenceReranker` struct + `CandidateReranker` impl, 3 new unit tests.
- `crates/ruvector-gnn-rerank/src/lib.rs`: re-export + 1 additional unit test.
- `crates/ruvector-gnn-rerank/src/bin/filtered_rerank_bench.rs`: the fixed benchmark described above.

## API Shape

```rust
pub struct AdaptiveCoherenceReranker {
    pub alpha: f32,          // default 0.60
    pub keep_fraction: f32,  // default 0.50
    pub min_keep: usize,     // default 1
    pub k_graph: usize,      // default 8
}
impl CandidateReranker for AdaptiveCoherenceReranker { /* ... */ }
```

## Feature Flags

None needed — this is a pure additive library type behind no `cfg`, consistent with the
rest of the crate's reranker variants.

## Benchmark Evidence

See Evidence section above; raw stdout preserved at
`docs/research/nightly/2026-10-28-adaptive-coherence-filtered-rerank/raw_benchmark_output.txt`.

## Security

No new attack surface: reranking operates read-only over an already-retrieved, in-memory
candidate set. `validate()` (shared with all variants) still rejects non-finite scores/
vectors and dimension mismatches before this code runs.

## Governance

Additive library change; no API removed or altered. Promotion gate (per nightly harness
spec) evaluated honestly: `beats_parent = false` on the primary hypothesis → **not promoted**
as a recommended default. Retained in the lineage as a measured, rejected variant.

## Failure Modes

- Degenerate zero-degree nodes (e.g. singleton filtered pools): falls back to the original
  `noisy_score` unchanged — covered by `adaptive_coherence_single_candidate_is_unchanged`.
- `min_keep` larger than available degree is clamped to the node's actual degree.

## Migration

None required. Purely additive; no existing call site is affected.

## Rollback

Trivial: simply stop constructing `AdaptiveCoherenceReranker`. No data format, schema, or
persisted state is touched by this crate.

## Rejection Criteria

This ADR's own subject documents the rejection: the primary measured effect (+0.18 pp mean
gain at low selectivity, 95% CI excluding the +3 pp threshold) falls far short of the
hypothesis. No further criteria apply since the work is already closed out as rejected.

## Open Questions

- Would a filter model correlated with cluster/tenant membership (rather than uniform-random
  thinning) produce the predicted sparsification collapse for `GnnMincutReranker`? Untested
  here — flagged as the next experiment in the nightly research doc.
- Is `k_graph=8` itself the dominant constraint on reranking quality under heavy filtering,
  independent of the gating rule? Not isolated by this experiment.
