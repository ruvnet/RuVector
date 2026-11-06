# Incremental Boundary Tracking for `ruvector-mincut`'s Local K-Cut Oracle

## Problem

`ruvector-mincut`'s `RuVectorGraphAnalyzer::partition()` — the entry point
used to find a graph's minimum cut and its two sides — has a known,
measured latency scaling problem: ADR-345 (2026-09-05) measured 76ms at
50 vertices growing to 11.4 seconds at 400 vertices on a fixed-degree k-NN
graph, which blocked a candidate agent-memory compaction feature
(`MincutGatedForgetting`) from shipping. A follow-up nightly (ADR-346,
2026-09-11) fixed two unrelated non-determinism bugs in the same code path
but explicitly deferred the latency problem, suggesting (without checking)
that `PolylogConnectivity` — an already-implemented dynamic connectivity
structure in the same crate — might be a drop-in faster backend.

## What was actually wrong

Reading the implementation (`DeterministicLocalKCut::deterministic_bfs`,
`crates/ruvector-mincut/src/localkcut/paper_impl.rs`) line by line surfaced
two distinct inefficiencies, neither of which `PolylogConnectivity` could
have fixed (it answers "are these two vertices connected?", not "how many
edges cross this cut?" — a different primitive entirely):

1. **Per-layer boundary recompute.** The BFS that searches for a small cut
   expands outward layer by layer, and at *every* layer it recomputes the
   number of edges crossing the current visited/unvisited boundary by
   rescanning every visited vertex's edges from scratch. For a visited set
   that grows to `v` vertices over `d` layers, that's `O(d * v)` work where
   `O(v)` suffices: each edge only needs to be classified once, exactly
   when one of its endpoints joins the visited set.
2. **A budget loop that loops over nothing.** The calling code tries every
   integer "budget" value in a range, re-running the *identical* BFS for
   each one — but the budget parameter never affects which vertices get
   visited, only whether a layer's result counts as a hit. One pass at the
   most permissive budget already contains the answer for every smaller
   one.

## Fix

Two new, purely additive code paths (the existing default behavior is
untouched, so every existing caller of `ruvector-mincut` is unaffected
unless it explicitly opts in):

- `IncrementalLocalKCut` (`localkcut/incremental.rs`): the same
  deterministic BFS, but it classifies each edge once as it's absorbed
  into the visited set and reads the current boundary size as an O(1)
  counter instead of a full rescan.
- `BoundedInstance::with_strategy(..., CutSearchStrategy::IncrementalOracleCollapsedBudget)`:
  uses the above oracle *and* collapses the redundant budget loop into one
  pass per candidate starting point, picking the smallest valid cut found.

A short proof sketch (and an empirical differential test suite comparing
outputs against the unmodified code on dozens of graphs) establishes that
collapsing the budget loop changes nothing about which cut gets returned —
it only stops re-deriving the same answer repeatedly.

## Evidence

Real, `Instant::now()`-timed runs through the production
`RuVectorGraphAnalyzer::partition()` entry point, on the exact graph shape
and sizes the original ADR-345 measurement used (a ring k-NN graph, degree
8, at 19/50/100/200/400 vertices):

| n | Legacy (ms) | Collapsed-budget candidate (ms) | Speedup |
|---|---|---|---|
| 19 (brute-force path, unfixed) | 79,475.8 | 78,305.8 | 1.01x |
| 50 | 86.6 | 38.1 | 2.27x |
| 100 | 552.1 | 254.7 | 2.17x |
| 200 | 2,995.4 | 1,009.1 | 2.97x |
| 400 | 12,748.7 | 3,232.2 | 3.94x |

Every strategy returns the same cut value and the same single-vertex
witness at every size — the speedup comes from how fast the identical
answer is found, not from finding a different (wrong) one. A 204-second
differential test suite comparing the old and new code on dozens of
generated graphs (brute-force-path and oracle-path, several densities and
sizes) found zero mismatches.

## Verdict

**REJECT** the headline hypothesis (>=10x speedup at 400 vertices) as
written — the measured speedup is 3.94x, well short of the 10x bar fixed
before the benchmark ran. The fix is real, the direction is right (speedup
grows with graph size: 2.27x → 2.17x → 2.97x → 3.94x), and correctness is
fully preserved, but the magnitude predicted from reading the oracle in
isolation didn't hold up once measured end-to-end. The reason, once
measured: a real `partition()` call doesn't run one slow search — it tries
~15-20 progressively wider candidate ranges before landing on the right
one, and most of those have little redundant work for this fix to remove
in the first place. The per-call fix is real; the end-to-end win is capped
by how many calls still have to happen regardless.

## Limitations

- Only the `>=20`-vertex code path (the one with the measured scaling
  problem) was touched. A *separate*, independently-discovered
  inefficiency in the `<20`-vertex brute-force path — each of the up to
  100 bounded-range instances tried before finding the right one re-runs
  the entire exponential enumeration from scratch, with no sharing across
  instances — is documented but not fixed here.
- Does not re-run the `MincutGatedForgetting` feature's own acceptance
  benchmark with this fix wired in; whether it changes that feature's
  prior rejection is an open, named next step, not assumed.
- Two pre-existing, orthogonal correctness quirks were found while building
  the differential test suite (a silent multigraph-to-simple-graph
  collapse in the unmodified code's temporary graph rebuild, and a
  non-deterministic choice of isolated vertex on disconnected graphs) and
  are documented, not fixed, since fixing them is out of this run's scope
  and carries its own risk to 14+ downstream crates.
