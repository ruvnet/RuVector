# ADR-355: Fix `ApproxMinCut`'s Stub Partition and Add It as a `MincutGatedForgetting` Backend

## Status

Accepted (bug fix in `ruvector-mincut`, non-breaking). The new
`MincutBackend::Approx` option in `ruvector-agent-memory` is additive and
defaults off (`Exact`); promoting it to the default is deferred to a
follow-up nightly per the [Rejection criteria](#rejection-criteria) below.

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

1. Fix `ApproxMinCut`'s partition bug in `ruvector-mincut`
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
   the now-redundant `compute_exact_min_cut` outright.
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

## Consequences

- `ApproxMinCut.partition` is now trustworthy for every in-tree and
  downstream consumer of `ruvector-mincut` (WASM/Node bindings,
  `ruvector-graph-condense`, `prime-radiant`, `cognitum-gate-kernel`,
  `mcp-brain-server`, this crate) — previously, any caller relying on it
  structurally would have gotten a silently wrong answer past the
  `<=50`-edge exact fallback.
- `MincutGatedForgetting` now has a backend that is fast enough to
  consider for the production path ADR-345 closed off, pending the
  full-scale re-run named in [Next steps](#open-questions).
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

Promotion of `Approx` as `MincutGatedForgetting`'s *default* backend is a
separate decision, deferred to a follow-up nightly that re-runs the full
2026-09-05 bridge-retention/Recall@10 benchmark at the originally-intended
corpus scale (see [Rejection criteria](#rejection-criteria)).

## Failure modes

- Approximate backend's error bound is unverified past this run's 400-vertex
  ceiling; a much larger corpus could behave differently.
- Partition quality (not just the cut *value*) at looser epsilon is not
  separately quantified beyond the bridge-isolation test passing at the
  epsilons benchmarked (0.1, 0.3).

## Migration

None required — both changes are additive and default-preserving.

## Rollback

Revert both commits; `MincutBackend` has no persisted state and no other
crate depends on it yet, so rollback has no downstream migration cost.

## Rejection criteria

Do **not** promote `MincutBackend::Approx` to `MincutGatedForgetting`'s
default, and do not claim `MincutGatedForgetting` is production-ready,
until a follow-up nightly re-runs the original 2026-09-05
`mincut_gated_forgetting_bench` hypothesis (bridge-survival gap >= 15pp,
Recall@10 within 2pp of baseline, wall-clock bound) with
`backend: MincutBackend::Approx` substituted in at the originally-intended
~2,000-memory corpus scale `Exact`'s cost made infeasible, and it clears
those thresholds. This run's evidence is about boundary-detection
latency/correctness in isolation, not the full compaction-policy claim.

## Open questions

1. Does the full-scale re-run (above) clear ADR-345's original thresholds?
2. Does any existing downstream consumer of `ApproxMinCut.partition`
   depend on the old (wrong) behavior in a way this fix could break? (Not
   audited this run — see Next Research item 5 in the nightly README.)
3. Should `ClusterHierarchy` be renamed or rebuilt to match its
   documented intent?
