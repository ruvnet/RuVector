# When "Reuse the Fisher Information" Breaks Your Graph Index

## Problem

Continual-learning systems that use Elastic Weight Consolidation (EWC) —
RuVector's `sona` crate implements EWC++ for its SONA learning loops —
already compute a per-task Fisher information diagonal: a vector saying
"how much would it hurt to change this parameter". It's tempting to reuse
that signal everywhere once you have it, including as a distance metric
for retrieval: "find the most similar previously-learned task" by
weighting the comparison with the stored task's own Fisher diagonal,
instead of plain Euclidean distance. RuVector's own ADR-002
(`ruvllm-integration`) leaves this exact question open — it sketches a
`fisher_diagonal: Option<Vec<f32>>` field sitting next to a semantic
embedding in a shared HNSW index, without saying whether the Fisher
diagonal should drive the index's search or just ride along as metadata.

This matters because EWC's own task-selection today (`apply_constraints`,
`apply_anchor` in `sona::ewc::EwcPlusPlus`) is a linear scan over a
bounded task buffer (`max_tasks`, default 10). A long-running or
federated agent accumulating hundreds or thousands of task-specific
adapters would need something faster than a linear scan — and "build an
ANN index with the same Fisher-weighted metric" is the obvious first
thing to try.

## Hypothesis

```text
Given a bank of N synthetic EWC task snapshots (weights + per-task Fisher
diagonal + importance) across 40 task families with sparse, family-
specific important dimensions,

when a Fisher-weighted HNSW index is used for nearest-prior-task lookup
instead of (a) a Fisher-weighted brute-force scan at scale, or (b) a
plain-L2 HNSW over the same points,

then the Fisher-weighted index's query latency should be at least 5x
faster than brute force at N=10,000, subject to recall@10 >= 0.90 against
the Fisher-weighted ground truth, a bounded EWC regularization-loss
quality gap, and the Fisher weighting measurably beating the plain-L2
ablation on quality.
```

## Technical Design

Three variants, same underlying from-scratch, deterministic, multi-layer
HNSW implementation (`crates/ruvector-fisher-ann/src/hnsw.rs`),
parameterized only by which distance function it's built and searched
with:

- **Baseline** — brute-force linear scan under the Fisher-weighted
  distance `fisher[task] · (query - task.weights)² · importance[task]` —
  the literal generalization of `EwcPlusPlus`'s existing production scan,
  and also this experiment's ground truth.
- **Candidate A** — HNSW built and searched with that same Fisher-weighted
  distance.
- **Candidate B** — HNSW built and searched with plain squared L2,
  ignoring Fisher entirely (the ablation).

The synthetic dataset (`dataset.rs`) reproduces one real structural
property of Fisher diagonals: importance concentrates on a small subset
of dimensions per task. 40 "families" each get a random center vector and
a random ~10%-dimension "important" mask; important dimensions get tight
clustering and high Fisher values, the rest get large nuisance noise and
near-zero Fisher.

Crucially, the quality metric isn't an invented number: `single_task_
regularization_loss` implements the identical closed-form EWC loss
formula `EwcPlusPlus::regularization_loss` uses, and a unit test builds a
real `EwcPlusPlus` through its public API and checks the two agree.

## Actual Implementation

`crates/ruvector-fisher-ann` — new workspace crate, ~700 lines across
`types.rs`, `dataset.rs`, `baseline.rs`, `hnsw.rs`, `main.rs`. 15 unit
tests (determinism, distance-function correctness, HNSW recall sanity,
edge cases, and the production-equivalence check above), all passing.
`cargo clippy --all-targets` clean.

## Actual Benchmark Evidence

Release build, dim=128, 40 families, 300 queries, 4-vCPU container,
`rustc 1.97.0`. Full raw output in `bench_output.txt` in this directory.

```text
variant                             n   build_ms    mean_us  recall@10     loss_gap
baseline_fisher_bruteforce      10000       0.00    1424.54      1.000       0.0000
candidate_A_fisher_hnsw         10000    2471.55      79.85      0.354   50120.3477
candidate_B_plain_l2_hnsw       10000    4550.78     337.31      0.388       0.0000
```

Candidate A is 17.8x faster than baseline — comfortably clearing the
speedup bar. It also fails recall by a wide margin (0.354 vs. a required
0.90), fails the loss-gap bound by two orders of magnitude, and loses the
ablation outright: candidate B, which throws Fisher information away
entirely, has *better* recall and a *zero* loss gap.

A follow-up diagnostic swept `ef_search` (HNSW's search-effort knob) from
100 up to 2,000 — equal to the entire dataset at N=2,000 — and recall@10
stayed flat at 0.352 the whole way. That rules out "just search harder":
the problem is structural, not a tuning knob.

**Why:** the Fisher-weighted distance is intentionally one-sided — it
weights by the *candidate* task's own Fisher diagonal, because that's what
the real EWC loss formula computes. That makes it asymmetric and not a
proper metric. HNSW's construction and search both lean on an implicit
triangle-inequality assumption to stay navigable; a scoring function where
a near-zero-Fisher candidate can look "close" to almost anything,
regardless of actual distance, breaks that assumption and produces a
poorly connected graph. Plain L2 doesn't have this problem — it's a real
metric — and in this dataset it still weakly tracks family membership
(every dimension, important or not, carries a bit of family signal),
which is why the "wrong" metric's index outperforms the "right" metric's
index here.

## Limitations

Fisher diagonals here are synthetic — no live fine-tuning loop was
available to capture real ones. The dataset is built to carry the one
structural property (sparse importance + nuisance noise) this hypothesis
needs, and the quality metric is checked against real production code,
but the exact 0.354 recall figure is a property of this distribution, not
a universal constant. N was capped at 10,000 to fit this session's time
budget. The HNSW implementation is a compact, from-scratch one built for
this ablation, not RuVector's production index — the navigability failure
traces to the metric, not an implementation bug (confirmed by a
determinism test, a small-scale recall sanity check, and the `ef_search`
sweep), but a from-scratch implementation can't fully rule out
contributing to the exact magnitude.

## Production Relevance

This closes an open design question in ADR-002 with a measured answer:
don't build a Fisher-weighted index for the EWC policy-memory slot. Index
the semantic embedding with a plain symmetric metric, carry `fisher_
diagonal` as inert payload, and keep EWC's own task selection as the
bounded linear scan it already is — which this run's baseline numbers
show stays cheap (sub-millisecond to low-single-millisecond at N=10,000)
well past the current default `max_tasks` of 10. If that bound ever needs
to grow by orders of magnitude, the indicated design is a shortlist-then-
exact pipeline (coarse unweighted ANN for a shortlist, exact Fisher
scoring only within it) — not a Fisher-weighted graph, which this result
shows degrades silently rather than failing loudly.

## RuVector Ecosystem Implications

Full ADR: `docs/adr/ADR-353-fisher-weighted-ewc-ann-rejected.md`. Connects
`sona` (EWC++, the real regularization-loss formula), `ruvector-core`'s
HNSW/ANN design space, ADR-002's RuvLLM policy-memory architecture, and
(by the production recommendation) RVF's natural fit for packaging
individual task snapshots as portable artifacts even though indexing them
by Fisher weight is rejected here.

## Future Direction

1. Test a symmetrized Fisher-weighted metric (e.g., averaging both sides'
   diagonals) — does restoring the metric property restore navigability
   without losing the signal's connection to `regularization_loss`?
2. Build and benchmark the shortlist-then-exact design directly.
3. Repeat at N=100,000+ on non-containerized hardware.

## References

- `docs/adr/ADR-002-ruvllm-integration.md` (the open question this run
  answers)
- `crates/sona/src/ewc.rs` (`EwcPlusPlus`, the production code this run's
  quality metric is checked against)
- Malkov, Y. A., & Yashunin, D. A. (2016). *Efficient and robust
  approximate nearest neighbor search using Hierarchical Navigable Small
  World graphs.* (the HNSW construction/search algorithm this run's
  `hnsw.rs` implements, generically over distance metric)
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in
  neural networks* (the original EWC formulation `EwcPlusPlus` extends)
