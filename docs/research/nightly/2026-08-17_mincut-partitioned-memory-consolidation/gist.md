# Partitioning agent memory before compaction: a negative result, and a bug it uncovered

## Problem

Agent memory systems that compact by a single global importance score
(recency + frequency + relevance to current context) can evict an entire
topic in one pass, even at a compaction ratio that looks fine on average.
A topic the agent isn't currently working on has no defense against a
score built for the topic it is working on.

## Hypothesis

Partition the memory similarity graph into topic clusters first, then
give each partition a guaranteed minimum retention share, so a topic only
competes with itself for its floor allocation instead of the whole
corpus. Tested on a synthetic 4,000-memory corpus with 6 unequal-size
semantic clusters (down to 2% of the corpus), against
`ruvector-agent-memory`'s existing `CoherencePolicy` baseline, at 50%
compaction.

Pre-declared bar: the best partition-aware candidate's **worst-cluster**
recall@10 must beat the baseline's by ≥15 percentage points.

## What happened

It didn't clear the bar. Overall recall improved (+6.8pp) and 4 of 6
clusters individually gained 15–23pp — the mechanism clearly does
something. But the specific cluster that was worst under the baseline was
*also* worst under the partitioned candidate, at the identical recall
value, because the partitioner left it merged with the majority cluster
instead of separating it out. A follow-up sweep of the retention floor
(1, 3, 8, 15) left the worst-cluster number flat across every value —
proof the gap is in *where the graph gets cut*, not *how the budget gets
split afterward*.

```text
per_cluster_recall GlobalTopScore  = [0.996, 0.152, 0.216, 0.316, 0.396, 0.440]
per_cluster_recall MincutAdaptive  = [0.792, 0.380, 0.504, 0.152, 0.556, 0.540]
                                              ^^^^^ improved a lot   ^^^^^ untouched
```

## The bug along the way

Before any of the above could be measured, `ruvector_mincut::DynamicMinCut::partition()`
turned out to be untrustworthy. Minimal repro: two triangles joined by a
single weak-weight bridge edge — a graph whose true minimum cut is
unique and easy to verify by hand.

```rust
let mincut = MinCutBuilder::new().exact().with_edges(edges).build().unwrap();
mincut.min_cut_value()  // 0.05, every single run — correct
mincut.partition()      // sometimes the correct split, sometimes a
                         // degenerate "isolate one vertex" split whose
                         // actual crossing weight is 40x the reported value
```

At 100 vertices the degenerate split happened on every run, not just
some. `GraphPartitioner` (built on the same machinery) separately dropped
vertices outright, fabricated vertex ids for non-contiguous id spaces,
and took 8.4 seconds to partition 500 vertices — with no sign of
finishing at 4,000 after nearly six minutes.

None of that is this crate's algorithm — it's the *value* computation
that was correct, only the *partition materialization* that wasn't. The
workaround was a from-scratch, tested, deterministic weighted
Stoer–Wagner implementation (`mincut_exact.rs`, ~250 lines, zero
non-`ruvector_mincut`-type dependencies), used as the sole source of
partition vertex sets, with `ruvector_mincut`'s value still queried
purely as an independent cross-check.

## Why report a rejected hypothesis

Because the measurement is real and the mechanism partially works. A
future variant with a smarter stopping rule — one that doesn't let a
single global threshold decide every split — is a legitimate next
experiment, and now has a concrete, per-cluster reason to exist instead
of a hunch. And because the correctness bug this candidate ran into would
have silently produced wrong partitions for anyone else building on
`GraphPartitioner` or `DynamicMinCut::partition()` today, whether or not
this particular hypothesis had panned out.

## Limitations

Single run per configuration, one synthetic corpus, one seed family — no
variance characterization. The `ruvector-mincut` defects are documented
with repros but not filed upstream from within this run; that needs the
owning maintainer's independent verification.

## Production relevance

None yet — this is a rejected hypothesis. If a per-branch stopping
criterion clears the bar in a follow-up run, the natural production path
is a scheduled ruFlo memory-consolidation workflow, not an inline
write-path operation (the measured ~11s partitioning time at n=4000 rules
that out regardless).

## RuVector ecosystem implications

`mincut_exact.rs` is a reusable, correctness-tested min-cut
implementation independent of this ADR's own rejected hypothesis — a
better foundation for any future RuVector graph-partitioning work than
`DynamicMinCut::partition()` as it stands today.

## Future direction

Test a per-branch/size-weighted stopping criterion (attempt at least one
more split on the largest remaining partition before accepting a global
threshold's verdict) against the same corpus and the same 15pp bar, as a
new, separately pre-declared hypothesis.

## References

- Nightly 2026-06-14, `ruvector-agent-memory` — `CoherencePolicy`, the
  baseline this experiment measured against and reused as a dependency.
- Nightly 2026-08-13, `ruvector-retrieval-receipt` — witness-chain design
  precedent.
- Stoer & Wagner, "A Simple Min-Cut Algorithm" (1997).
- Jin, Sun & Thorup, "Fully Dynamic Exact Minimum Cut in Subpolynomial
  Time" (SODA 2024) — the algorithm `ruvector-mincut` implements.

Full write-up, ADR, and raw benchmark output:
`docs/research/nightly/2026-08-17_mincut-partitioned-memory-consolidation/`
in [ruvnet/ruvector](https://github.com/ruvnet/ruvector).
