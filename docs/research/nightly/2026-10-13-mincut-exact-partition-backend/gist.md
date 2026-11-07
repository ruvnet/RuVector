# Finding an unused exact solver one layer below a slow API

*A RuVector nightly research note on `ruvector-mincut`.*

## The problem

Two previous research runs on `ruvector-mincut` — RuVector's from-scratch
implementation of a subpolynomial dynamic minimum-cut algorithm
(arXiv:2512.13105) — left an open, well-characterized performance problem.

A compaction policy for agent memory
(`MincutGatedForgetting`, built on `RuVectorGraphAnalyzer::partition()`) was
rejected because the mincut call was 1,800-2,700x slower than the scalar
baseline it was measured against, on a corpus of only 84 memories. A
follow-up run fixed a separate correctness bug — the same call was also
occasionally non-deterministic and returned empty results — but explicitly
left the latency problem untouched, naming the crate's lower-level
`DynamicMinCut` type as something worth trying directly.

## The hypothesis

Before writing any code, the question was: is this latency a property of
the *algorithm* `ruvector-mincut` implements, or a property of *which*
already-implemented code path a convenience API happens to call?

Tracing the call graph answered it. `RuVectorGraphAnalyzer::partition()`
always routes through `MinCutWrapper`, which dispatches to `BoundedInstance`
— the crate's actual subpolynomial-dynamic-algorithm implementation. That
algorithm is designed to *amortize* maintaining a minimum cut across many
incremental edge insertions and deletions on a large graph. For graphs
under 20 vertices it falls back to brute force; above that, it runs a
multi-seed, multi-budget heuristic local search that returns *a* cut within
a configured value range — not provably the smallest one.

Separately, the same crate already contains `algorithm::exact::minimum_cut`
— a complete, from-scratch sparse Stoer-Wagner implementation, exact and
polynomial-time, with its own doc comment stating as much
("Full recomputation is polynomial, not subpolynomial"). It's already used
internally by `DynamicMinCut::recompute_min_cut()` on every update. But
`RuVectorGraphAnalyzer` — the type all three in-tree consumers build on —
never calls it.

That's the mismatch: an algorithm built for amortized incremental
maintenance across many updates, used instead to answer one-shot queries on
a static graph, where it has none of its intended advantage and all of its
intended overhead.

## The fix

Two new, purely additive methods:

```rust
impl RuVectorGraphAnalyzer {
    pub fn min_cut_exact(&self) -> f64;
    pub fn partition_exact(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)>;
}
```

Both clone the analyzer's current graph into a fresh `DynamicMinCut` and
delegate to its already-existing, already-used `min_cut_value()`/
`partition()`. No existing method's signature or behavior changes.

`MincutGatedForgetting` (the agent-memory policy) gets a matching opt-in:

```rust
let policy = MincutGatedForgetting::soft(weights, bonus).exact();
```

## The evidence

Same ring-topology scaling probe, same five sizes, as the original rejection:

| n | old (`BoundedInstance`) | new (exact) | speedup |
|---:|---:|---:|---:|
| 19  | 69,269.9 ms (pathological outlier) | 0.472 ms  | ~146,800x |
| 50  | 76.8 ms   | 0.789 ms  | ~97x |
| 100 | 481.3 ms  | 1.846 ms  | ~261x |
| 200 | 2,712.9 ms | 4.801 ms | ~565x |
| 400 | 11,415.0 ms | 16.174 ms | ~706x |

Same 84-memory acceptance benchmark, same seed, same thresholds as the
original rejection, unmodified:

| Policy | Compaction time | Slowdown vs. baseline | Gate (≤100x) |
|---|---:|---:|---|
| `Soft` (old backend) | 116,147 µs | 3,226.3x | FAIL |
| `Soft-Exact` (new) | 1,993 µs | **52.4x** | **PASS** |
| `Hard` (old backend) | 118,103 µs | 3,280.6x | FAIL |
| `Hard-Exact` (new) | 5,038 µs | 132.6x | FAIL (24.7x closer) |

And the part that *didn't* move: bridge-memory survival gap versus baseline
was 0.0 percentage points for both the old and new backend, against a
required ≥15pp. The original rejection had two independent reasons — too
slow, and no measured effect. This run fixes the first and leaves the
second exactly where it was, which is itself informative: a 2-3
order-of-magnitude speed change moving the effectiveness number by exactly
0.0pp is stronger evidence that the "no effect" finding is architectural
(a global minimum cut doesn't reliably coincide with human-labeled semantic
bridges on continuous, noisy data) rather than a side effect of the slow,
non-exact old search.

Correctness of the new path was checked independently — against a
brute-force `O(2^n)` enumeration, not against anything `ruvector-mincut`
itself computes — on five random weighted graphs and one unambiguous
bridge construction. All matched to within `1e-9`. Determinism was checked
across 20 repeated calls on an identical graph: byte-identical every time.

The honest limit: an extended probe up to n=6,400 shows the new path's own
polynomial cost climbing to ~3.9 seconds there — it is not claimed to beat
a true incremental algorithm at every scale, only at the scale this
benchmark (and the original rejection) actually tested.

## What shipped vs. what didn't

**Shipped, promoted:** the two new `RuVectorGraphAnalyzer` methods — general
purpose, correct, deterministic, zero migration cost for existing callers.

**Not promoted:** `MincutGatedForgetting` for production use, on either
backend. The mandatory bridge-survival gate is still unmet. What changed is
*why* someone would reject it today — not "too slow to evaluate," but
"evaluated properly, and the answer is no" — which is a more useful
rejection to leave behind for whoever looks at this next.

## Why this is worth more than the latency number alone

Nightly research runs accumulate: the value here isn't just "52x faster,"
it's that two previously-entangled rejection reasons from an earlier run are
now shown to be genuinely independent, with two separate pieces of
measured evidence (a correct-by-construction exact solver, and a
brute-force-verified test suite) rather than one. The next person who looks
at `MincutGatedForgetting` doesn't need to re-litigate whether the backend
was the problem — it's answered, with numbers, and the actual remaining
question (does *any* global-cut-based signal track human-labeled bridges on
noisy data?) is isolated and ready to be attacked on its own.

## References

- `docs/adr/ADR-360-mincut-exact-partition-backend.md` — the architecture
  decision record for this change.
- `docs/adr/ADR-345-mincut-gated-forgetting.md`,
  `docs/adr/ADR-346-deterministic-mincut-witness-partition.md` — the two
  prior nightlies this one responds to.
- `crates/ruvector-mincut/src/algorithm/exact.rs` — the pre-existing
  Stoer-Wagner implementation this work connects to a new consumer.
