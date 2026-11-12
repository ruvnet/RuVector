# ADR-357: Fix `ApproxMinCut`'s Stub Partition and Add It as a `MincutGatedForgetting` Backend

## Status

Accepted (bug fix in `ruvector-mincut`, superseded — see Update below),
Rejected (the `MincutBackend::Approx` boundary-detection use case).

## Update (2026-11-12): main independently fixed the same bug more rigorously; re-validating against it reverses this ADR's original ACCEPT

A landing review on PR #1286 found that `main` had, independently and in
parallel, fixed the exact same `ApproxMinCut.partition` stub bug this ADR
originally addresses — commits `d4a1092db` ("report cuts weighed on the
original graph") and `e2c33478a` ("report exact results and qualify
release claims"). Main's fix is more rigorous than this run's original
one: it reuses the crate's existing `exact::minimum_cut` solver instead of
a bespoke Stoer-Wagner reimplementation, always re-weighs every candidate
partition's crossing weight on the *original* graph (not the sparsifier's
rescaled weights — a real consistency gap in this ADR's original fix that
its own tests never caught, because they only exercised the `<=50`-edge
exact path), and removes this crate's previously-false "(1+ε) certified
approximation" framing in favor of an honestly-labeled heuristic with only
a trivial upper-bound guarantee.

This run's `ruvector-mincut` algorithm change (Decision item 1, the
`stoer_wagner_with_partition` addition and its 5 tests) is therefore
**dropped** in favor of main's version — adopted via merge, zero diff
against `main`'s `crates/ruvector-mincut/src/algorithm/approximate.rs`.

Re-running this ADR's own benchmark against main's fixed algorithm,
however, surfaced a second, more serious problem: **main's heuristic
reliably fails to find anything close to the true minimum cut on this
ADR's own two-dense-cliques-plus-weak-bridge correctness topology once the
graph exceeds the 50-edge exact-path threshold.** On the 19-vertex, 74-edge
bridge dataset (two ~9-clique topic clusters joined by one weak bridge —
the realistic shape of a k-NN graph over a real embedding, and the exact
shape `MincutGatedForgetting` needs the signal for), `ApproxMinCut` reports
a cut of weight ~150,000 — the true minimum is ~3.6 (the bridge's own two
edges) — and flags the bridge as a boundary vertex in **0% of 50 trials**,
down from 100% under this ADR's original (now-superseded) fix. Raising
`num_samples` from 3 to 50 reduces the reported cut to ~70,000 but does
not fix the qualitative failure (still ~19,000x too high, still 0%
bridge-detection) — this is not an under-sampling bug, it is a structural
weakness of "sample edges proportional to weight × resistance, take the
best of N candidates weighed on the original graph" when edge weights are
highly skewed (the cliques' edges carry weight ~10,000 from near-identical
embeddings; the bridge's carry ~1.8): the sampling procedure is heavily
biased against ever constructing the "isolate the bridge" candidate in
the first place, so no amount of "best of N" selection over the candidates
it does construct can recover it.

**Net effect on this ADR's hypothesis:** REJECT. The opt-in
`MincutBackend::Approx` wiring in `ruvector-agent-memory::graph_forget`
is retained (it is correct, tested, additive, and defaults off — see
[Production path](#production-path)), but is **not recommended** for
`MincutGatedForgetting`'s actual boundary-detection use case: it is fast,
but not reliably correct for the realistic topology this policy exists to
protect. The [Evidence](#evidence) section below is retained with its
original (now-stale, pre-main-fix) numbers for lineage, followed by the
corrected post-merge benchmark in a new **Evidence (post-merge,
2026-11-12)** section.

## Context

The 2026-09-05 nightly (ADR-345) rejected `MincutGatedForgetting` for
production use: its only backend, `RuVectorGraphAnalyzer::partition()` (an
exact global min-cut solver), measured 76ms-11.4s per call on 50-400
vertex k-NN graphs — 1,800-2,700x slower than the scalar baseline even at
an 84-memory corpus. Its Next Research item 1 asked whether a lower-level
API (`DynamicMinCut`, `ClusterHierarchy::boundary_size`) would avoid the
overhead. The 2026-09-11 nightly (ADR-346) fixed two independent
non-determinism bugs in the same exact code path but left the latency
question open, additionally naming `connectivity::polylog::PolylogConnectivity`
as a candidate.

Investigating all three named candidates this run (2026-10-29) found:

- `DynamicMinCut::recompute_min_cut()` always calls the exact
  `exact::minimum_cut` solver — its own `MinCutConfig.approximate` flag is
  never read. Same cost profile as `RuVectorGraphAnalyzer`.
- `ClusterHierarchy` builds its levels by arbitrary vertex-insertion-order
  pairing (`chunks(2)` over whatever order `DynamicGraph::vertices()`
  returns), not by minimum cut. Its `boundary_size` is a real, cheap,
  O(|E|) edge-crossing count — for an arbitrary split, not a minimum one.
  Not a valid correctness signal.
- `PolylogConnectivity` (`connectivity::polylog`) answers "are u and v
  still connected," not "what is the min cut" — no cut value, no
  partition. Not applicable to this use case at all.

A fourth candidate not named by either prior nightly,
`algorithm::approximate::ApproxMinCut` (SODA 2025 spectral
sparsification, arXiv:2412.15069, `pub use`d from the crate root with 9
passing unit tests, claimed O(m log² n / ε²) preprocessing / O(n polylog n
/ ε²) query), had the right complexity class — but its `partition` field
was a stub. `compute_partition` ignored the cut value it received and
returned an arbitrary BFS-order bisection unrelated to which edges the
min-cut computation actually selected. None of the 9 existing tests
checked `.partition` at all, so this was never caught.

## Hypothesis

```text
Given the same two-clique-plus-bridge 19-vertex correctness topology and
ring-shaped k-NN scaling probe (19/50/100/200/400 vertices, k=8) used by
the 2026-09-05/09-11 nightlies,

when MincutGatedForgetting's boundary detection uses a fixed
ApproxMinCut (MincutBackend::Approx{epsilon}) instead of
RuVectorGraphAnalyzer (MincutBackend::Exact, baseline),

then boundary-detection latency at 50-400 vertices decreases
substantially relative to Exact, while bridge-vertex detection and
determinism (empty/degenerate rate) stay at parity or better.
```

## Decision

1. ~~Fix `ApproxMinCut`'s partition bug in `ruvector-mincut`
   (`src/algorithm/approximate.rs`): add `stoer_wagner_with_partition`,
   which performs the same Stoer-Wagner contraction as the pre-existing
   value-only `stoer_wagner` while tracking supervertex membership, so the
   cut-of-the-phase partition can be read back in original-vertex terms.
   Wire it into both the small-graph exact path and the sparsifier path
   (re-run once on the specific sparsifier whose value was selected as the
   median — not on the full graph, which would defeat sparsification's
   purpose). Cache the partition in lockstep with the cached value so
   repeated `min_cut()` calls on an unchanged graph return it too (the old
   code returned `None` unconditionally on the cache-hit path, even though
   a value was already cached). Remove the stub `compute_partition` and
   the now-redundant `compute_exact_min_cut` outright.~~ **Superseded (see
   Update above):** `main` independently shipped a more rigorous fix to
   the same file (commits `d4a1092db`, `e2c33478a`) while this PR was
   open. This item's implementation was dropped in favor of main's on
   merge; `crates/ruvector-mincut/src/algorithm/approximate.rs` in this
   PR is now byte-identical to main's version.
2. Add `MincutBackend { Exact, Approx { epsilon: f64 } }` to
   `ruvector-agent-memory::graph_forget`, with `MincutGatedForgetting::backend`
   defaulting to `Exact` (zero behavior change for existing callers).
   Extract the per-trial boundary-detection dispatch into a public free
   function, `boundary_from_knn`, so it can be benchmarked directly.
3. Add 5 regression tests to `ruvector-mincut` (triangle/path partition
   sanity, a two-6-cliques-plus-weak-bridge isolation test built
   specifically to catch the original bug, a cache-consistency test, and a
   disconnected-graph partition test) and a new benchmark,
   `examples/mincut_backend_bench.rs`, comparing `Exact` against
   `Approx{0.1}`/`Approx{0.3}` on the prior nightlies' own topologies.

## Evidence

Release build, single-threaded, one process
(`cargo run --release -p ruvector-agent-memory --example mincut_backend_bench --features mincut-forget`):

**Latency scaling (ring k-NN, k=8):**

| n   | Exact (ms) | Approx(0.1) (ms) | Approx(0.3) (ms) | Speedup (vs 0.1) |
|-----|-----------:|------------------:|------------------:|-----------------:|
| 19  | 91,961.2   | 0.345             | 0.311             | ~266,000x        |
| 50  | 139.4      | 2.6               | 3.1               | ~54x             |
| 100 | 678.5      | 10.5              | 10.7              | ~65x             |
| 200 | 3,850.9    | 57.6              | 50.9              | ~67x             |
| 400 | 14,803.6   | 396.3             | 311.0             | ~37x             |

**Determinism + bridge-boundary correctness** (19-vertex two-clique-plus-bridge
topology, 50 trials/backend):

| backend        | empty/degenerate | bridge detected as boundary | avg latency/call |
|----------------|------------------:|------------------------------:|------------------:|
| Exact          | 0%                | 100%                          | 1,139.5ms         |
| Approx(0.1)    | 0%                | 100%                          | 0.311ms           |
| Approx(0.3)    | 0%                | 100%                          | 0.297ms           |

Both approximate configurations match `Exact`'s correctness (100%
bridge-detection, 0% degenerate results — the determinism fix from
ADR-346 holds for `Exact` here too) at 37x-266,000x lower latency,
confirming the hypothesis. The `n=19` ring-topology outlier (92 seconds
for `Exact`) shows `RuVectorGraphAnalyzer`'s cost is highly
topology-dependent, not just size-dependent — worth noting for anyone
sizing future `Exact` usage by vertex count alone.

All 9 pre-existing `ApproxMinCut` tests pass unmodified; all 5 new tests
pass; all 3 pre-existing `graph_forget` tests pass unmodified with the new
`backend` field defaulting to `Exact`.

## Evidence (post-merge, 2026-11-12, against main's `ApproxMinCut`)

Same benchmark, same command, release build, after merging main's
independent fix (this run's own algorithm change dropped):

**Latency scaling (ring k-NN, k=8):** even faster than before — main's
version skips the Stoer-Wagner-on-median-sparsifier step this ADR
originally added and instead runs the crate's existing exact solver
directly on each (small) sampled sparsifier:

| n   | Exact (ms) | Approx(0.1) (ms) | Approx(0.3) (ms) |
|-----|-----------:|------------------:|------------------:|
| 19  | 97,565.8   | 0.145             | 0.109             |
| 50  | 80.4       | 0.279             | 0.240             |
| 100 | 506.5      | 0.543             | 0.497             |
| 200 | 1,659.7    | 1.690             | 1.589             |
| 400 | 6,056.3    | 3.137             | 3.280             |

**Determinism + bridge-boundary correctness** (same 19-vertex
two-clique-plus-bridge topology, 50 trials/backend):

| backend     | empty/degenerate | bridge detected as boundary | avg latency/call |
|-------------|------------------:|------------------------------:|------------------:|
| Exact       | 0%                | 100%                          | 1,250.7ms         |
| Approx(0.1) | 0%                | **0%**                        | 0.071ms           |
| Approx(0.3) | 0%                | **0%**                        | 0.082ms           |

Faster, but no longer correct for this topology. A targeted diagnostic
(19 vertices, 74 edges after dedup — above the 50-edge exact-path
threshold) confirms why: `ApproxMinCut::min_cut()` reports value
**150,047.36** at the default `num_samples: 3`, and **70,066.31** at
`num_samples: 50` — both many orders of magnitude above the true global
minimum (**~3.6**, the bridge's own two edges, weight 1.809 each). The
reported partition splits off an arbitrary 5- or 7-vertex group instead of
isolating the bridge; the bridge and most of its own cluster land on the
*same* side, so `MincutGatedForgetting`'s crossing-edge check correctly
reports it as not a boundary vertex at all.

## Consequences

- `ApproxMinCut.partition` is now trustworthy *as a value/partition-consistency
  guarantee* for every in-tree and downstream consumer of `ruvector-mincut`
  (WASM/Node bindings, `ruvector-graph-condense`, `prime-radiant`,
  `cognitum-gate-kernel`, `mcp-brain-server`, this crate) — main's fix
  (adopted here) guarantees `value` always equals the returned partition's
  actual crossing weight on the original graph. It is **not** a guarantee
  that the returned cut is close to the true minimum — see
  [Evidence (post-merge)](#evidence-post-merge-2026-11-12-against-mains-approxmincut).
  Any caller treating a low `ApproxMinCut.value` as evidence of a
  near-minimum cut, on a graph with skewed edge weights, should re-check
  that assumption against this ADR's bridge-topology finding.
- `MincutGatedForgetting` gains a backend that is fast but **not** reliable
  for its own boundary-detection purpose on realistic (skewed-weight)
  topologies — see [Production path](#production-path). It does not close
  ADR-345's latency blocker in any usable way; `Exact` remains the only
  backend with validated correctness, at its original cost.
- No migration needed: both changes are additive/backward-compatible.

## Alternatives

- **`DynamicMinCut`**: rejected, ignores its own approximate config (see
  Context); flagged as a separate latent bug, not fixed here (out of
  scope — unused by this policy).
- **`ClusterHierarchy::boundary_size`**: rejected as a correctness signal,
  fast but not cut-based.
- **`PolylogConnectivity`**: rejected, wrong problem (connectivity, not
  cut value/partition).
- **Recomputing the exact solver on the full graph to fix `compute_partition`**:
  rejected at design time — defeats sparsification's purpose, silently
  reintroducing `Exact`'s cost.

## Implementation plan

Already implemented this run (see [Decision](#decision)); no phased
rollout needed given the additive, default-preserving API.

## API shape

```rust
// ruvector-mincut (unchanged signatures, corrected semantics):
pub struct ApproxMinCutResult {
    pub value: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub partition: Option<(Vec<VertexId>, Vec<VertexId>)>, // now correct
    pub epsilon: f64,
}

// ruvector-agent-memory (new):
pub enum MincutBackend {
    Exact,
    Approx { epsilon: f64 },
}
pub struct MincutGatedForgetting {
    // ...
    pub backend: MincutBackend, // defaults to Exact
}
pub fn boundary_from_knn(
    backend: MincutBackend,
    neighbors: &[(usize, Vec<(usize, f64)>)],
) -> std::collections::HashSet<usize>;
```

## Feature flags

No new feature flag. Both changes live behind `ruvector-agent-memory`'s
existing `mincut-forget` feature; `ruvector-mincut`'s `approximate` module
is already enabled by default (per its crate-level doc comment).

## Benchmark evidence

See [Evidence](#evidence) above; raw output preserved in
`docs/research/nightly/2026-10-29-approx-mincut-boundary-backend/README.md`.

## Security

No new cryptographic or authority surface. The fix is strictly more
correct than the code it replaces (new tests assert the partition always
matches its reported cut value, which the old stub had no such property
for at all) — cannot introduce a new integrity-weakening behavior.

## Governance

Promotion of `Approx` as `MincutGatedForgetting`'s default backend is
withdrawn, not deferred: the post-merge evidence above is a direct
correctness failure on this ADR's own topology, not an open question
pending more data. Re-opening this would require a *different* algorithm
or heuristic, not more benchmarking of this one.

## Production path

1. Ship the opt-in `MincutBackend::Approx` as-is: it is correct in the
   narrow sense main's fix guarantees (value matches partition, always),
   builds, and passes every test written against it. Document it,
   per this ADR, as **not recommended** for `MincutGatedForgetting` or any
   other structural-boundary-detection use on real (skewed-weight) graphs.
2. `MincutGatedForgetting` itself gains nothing from this run toward
   production-readiness: `Exact` is still the only backend with validated
   correctness, still at ADR-345's original cost. The latency blocker
   ADR-345/346 opened remains open.
3. If a future run wants to retry the "fast + correct" goal, it should
   evaluate algorithms that don't depend on uniform-ish edge weights for
   their sampling to work — e.g. `DynamicMinCut`/`min_cut_exact` (ADR-360,
   closed-but-evidence-worth-reusing per the 7-PR consolidation on
   PR #1286) already clears both latency and correctness on this same
   topology, at polynomial (not near-linear) cost. That tradeoff — exact
   and correct at higher-than-`ApproxMinCut` cost, versus `ApproxMinCut`'s
   speed without a usable correctness guarantee on this workload's typical
   weight distribution — is the real open choice, not a parameter to tune
   within `ApproxMinCut` itself.

## Failure modes

- Approximate backend's error bound is unverified past this run's 400-vertex
  ceiling; a much larger corpus could behave differently.
- Partition quality (not just the cut *value*) at looser epsilon is not
  separately quantified beyond the bridge-isolation test passing at the
  epsilons benchmarked (0.1, 0.3) — this was true before the merge, and
  the post-merge evidence confirms the *current* algorithm fails this
  question outright at both epsilons on the bridge topology, not merely
  "unquantified."

## Migration

None required — both changes are additive and default-preserving.

## Rollback

Revert both commits; `MincutBackend` has no persisted state and no other
crate depends on it yet, so rollback has no downstream migration cost.

## Rejection criteria

Already met, not merely pending: do **not** promote `MincutBackend::Approx`
to `MincutGatedForgetting`'s default, and do not claim
`MincutGatedForgetting` is production-ready on this backend. The
post-merge evidence (0% bridge-detection, cut value ~40,000x the true
minimum) is itself the rejecting measurement this section originally
asked a future nightly to go find — no further benchmarking at larger
corpus scale is needed to reach this verdict, since the backend already
fails this ADR's own small-scale correctness check outright.

## Open questions

1. ~~Does the full-scale re-run clear ADR-345's original thresholds?~~
   Moot for `Approx`: it fails correctness before latency/scale are even
   relevant. Still open for `Exact` (unchanged from ADR-345) and for
   `DynamicMinCut`/`min_cut_exact` (ADR-360), neither touched by this run.
2. Does any existing downstream consumer of `ApproxMinCut.partition`
   depend on it being close to the true minimum cut (not just internally
   consistent) on a skewed-weight graph? This run's finding suggests
   auditing `ruvector-graph-condense`, `prime-radiant`,
   `cognitum-gate-kernel`, and `mcp-brain-server` for exactly this
   assumption, not just for "did it use the old stub" (ADR-357's original
   open question).
3. Should `ApproxMinCut`'s own doc comment/tests gain a skewed-weight
   correctness check (this ADR's bridge topology, or similar), so this
   failure mode is caught by `ruvector-mincut`'s own test suite rather
   than by a downstream consumer's benchmark? Not fixed here — this run
   does not own `ApproxMinCut`'s algorithm, only this consumer's use of it.
4. Should `ClusterHierarchy` be renamed or rebuilt to match its
   documented intent? (Unchanged from this ADR's original text; unrelated
   to the update above.)
