# Two bugs behind a "non-deterministic" minimum-cut algorithm

## Problem

A prior benchmark of a Rust dynamic minimum-cut crate (`ruvector-mincut`)
found that calling `partition()` repeatedly on the exact same, unchanged
graph returned different answers — sometimes a valid cut, sometimes an
empty ("no signal") result — about half the time. The benchmark also
measured it as slow. The investigation was filed as a follow-up rather than
solved on the spot: "consistent with internal tie-breaking that depends on
hash-map iteration order... filed as a follow-up hardening item."

This is that follow-up: read the actual code, find the root cause(s), fix
them, and re-run the original benchmark unmodified to see what changes.

## Hypothesis

Fixing whatever makes `partition()` non-deterministic should make repeated
calls on identical input return byte-identical results, without changing
the crate's existing test suite or (unless the non-determinism was itself
the reason) the original benchmark's pass/fail verdict.

## Technical Design

Two independent bugs, both found by reading the call graph from
`partition()` down to its actual solver, not by guessing:

**Bug 1 — tie-breaking depends on `HashSet` iteration order.**

```rust
// instance/bounded.rs, brute_force_min_cut()
let vertex_vec: Vec<_> = self.vertices.iter().copied().collect(); // self.vertices: HashSet<VertexId>
// ... vertex_vec[i] gets bitmask position i in an exhaustive subset search;
// the FIRST subset to reach the minimum boundary wins ties.
```

`self.vertices` is a `HashSet<VertexId>`. Rust's default hasher
(`RandomState`) is reseeded every time a new `HashSet` is constructed —
which happens on every call, since each `partition()` call builds a fresh
instance. So on a graph with more than one equally-cheap minimum cut
(common — a graph with a single "bridge" vertex between two clusters has
at least two, depending on which side the bridge joins), *which* tied cut
wins was a function of instance-construction entropy, not the graph. The
same pattern existed in the crate's other search path
(`search_for_cuts`, used for larger graphs), which tries seed vertices in
`HashSet` order and returns the first in-range result it finds.

**Bug 2 — the witness's "other side" computation is wrong by construction.**

```rust
// instance/witness.rs, materialize_partition()
let max_vertex = self.inner.membership.max().unwrap_or(0);
let v_minus_u: HashSet<VertexId> = (0..=max_vertex)
    .filter(|v| !membership.contains(v))
    .collect();
```

This assumes the graph's highest vertex ID is the highest ID *inside the
winning side* — i.e., it treats `max(membership)` as a proxy for the
graph's total size. On a 19-vertex test graph (IDs 0–18) where the winning
side happened to be the low-numbered half (`{0..=8}`), this produced
`{0..=8} − {0..=8} = ∅` instead of the real 10-vertex remainder. The
caller's own "empty side ⇒ no signal" guard then (correctly, given the
corrupted input) discarded the result. This bug requires no randomness to
trigger — only that the winning side, however it was chosen, excludes the
graph's true max-ID vertex.

**The fix.** Sort both order-sensitive vectors in bug 1 before they're
used (`sort_unstable()` — same values, deterministic order). For bug 2,
recompute the complement at the one production call site using the
caller's own known vertex set (`self.graph.vertices()`) instead of the
witness's self-reported, incomplete range.

## Actual Implementation

Three files changed, no new crate, no new dependency:

- `crates/ruvector-mincut/src/instance/bounded.rs` — two `sort_unstable()`
  calls.
- `crates/ruvector-mincut/src/integration/mod.rs` — `partition()`'s match
  arm recomputes the complement against `self.graph.vertices()`.
- `crates/ruvector-agent-memory/src/graph_forget.rs` — doc updates
  recording the root cause and fix; the caller's `mincut_trials`
  majority-vote workaround (previously up to 10 repeated calls, unioning
  results, to work around the bug) drops to a default of `1`.

## Actual Benchmark Evidence

Reused the original, unmodified benchmark artifacts. 200 repeated
`partition()` calls on a fixed 19-vertex graph with a known unique-weakest-
link structure:

| Stage | empty/wrong results | correct results |
|---|---|---|
| Before | 88/200 (44%) | 112/200 (56%) |
| Bug 1 fixed only | 200/200 (100%) | 0/200 (0%) |
| Both bugs fixed | **0/200 (0%)** | **200/200 (100%)** |

The middle row is the interesting one: fixing only the tie-breaking bug
made the result *fully deterministic and fully wrong* — every call
converged on the same tie-broken answer, which reliably tripped the second
bug. That's strong independent confirmation both bugs are real and both
needed fixing; it isn't a coincidence that fixing determinism alone made
things look worse before they got better.

Latency was flat across all three stages (~1.0–1.06 seconds per call on
this test graph) — both bugs are pure correctness bugs; neither touches
the underlying algorithm's actual computational cost, which remains a
separate, open question.

The original 84-memory production-scale acceptance benchmark's verdict
(REJECT — a proposed "protect structurally important memories from
eviction" policy built on this primitive didn't clear its bar) was
re-confirmed, unchanged, at every stage. That benchmark's own prior
analysis had already identified an independent reason for the rejection
(the graph's global minimum cut doesn't reliably correspond to what a
human would call "the important bridge memories" in this data), unrelated
to either bug — so an unchanged verdict here is the expected, and
falsifiable-if-wrong, result, not a null finding.

512 existing crate tests and 31 dependent-crate tests pass, unchanged
counts from before the fix. Clippy clean on both crates.

## Limitations

- Only the call path reachable from the one production caller's use of
  `partition()` was fixed and verified; the underlying witness method
  (`materialize_partition()`) is still wrong for any future direct caller
  until a deeper fix (adding a total-vertex-count field to its type)
  lands.
- Other modules in the same crate with similar `HashMap`/`HashSet`-heavy
  code were not audited for the same bug class.
- The separately-identified latency cost (tens of milliseconds to over a
  second per call, depending on graph size) is unaffected by this fix and
  remains open.

## Production Relevance

Determinism is a precondition, not a nice-to-have, for anything built on
top of a graph algorithm that needs to be replayed, audited, or verified —
witness chains, proof-gated mutations, reproducible benchmarking all
assume identical input produces identical output. This fix doesn't make
the specific feature it was discovered through (memory-eviction protection
based on graph structure) production-ready — that feature's own separate
problem (the structural signal doesn't line up with human judgment on this
data) still stands. What it does is remove a confound: any future attempt
at a similar idea, or any other consumer of this crate's minimum-cut
primitive, now inherits a correct, deterministic building block instead of
an intermittently broken one.

## Ecosystem Implications

The fixed code path (`RuVectorGraphAnalyzer::partition()`) also backs a
community-detection and a graph-partitioning utility in the same crate that
weren't exercised by the benchmark that found the bug — they inherit the
fix automatically. The witness type involved (`WitnessHandle`) is shared
infrastructure across the crate's certificate/audit machinery, so a
correct complement computation is also a precondition for any future
feature that needs to prove which side of a cut a vertex was on, not just
compute it once and move on.

## Future Direction

1. Test whether calling the crate's lower-level, already-sorted exact
   solver directly (bypassing the higher-level wrapper this fix touched)
   avoids the remaining latency cost while keeping today's correctness.
2. Push the complement fix down into the witness type's constructor so
   every future caller gets it for free, not just today's one caller.
3. Audit the crate's other graph-decomposition modules for the same class
   of "unordered-collection iteration feeds a first-match decision" bug.

## References

- The original benchmark and its filed follow-up items (this repository's
  research history), which this work directly answers.
- Stoer, M., Wagner, F., "A Simple Min-Cut Algorithm," Journal of the ACM,
  1997 — the classical algorithm family underlying the crate's exact
  solver (unchanged by this work; noted because it already sorts its
  inputs, unlike the buggy code path fixed here — a useful internal
  contrast for future audits).
