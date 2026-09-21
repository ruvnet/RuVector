# ADR-346: `ruvector-mincut` Partition Determinism — `BoundedInstance` Tie-Breaking and `WitnessHandle` Complement Fix

## Status

Accepted. Bug fix, landed directly (no Darwin/promotion-gate process —
see the parent nightly report's "Evolution Results" section for why).

## Context

ADR-345 (Mincut-Gated Forgetting) measured
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` returning different
results across repeated calls on byte-identical input, and filed two open
questions: what causes it, and does the crate's lower-level API avoid it.
ADR-345 explicitly declined to investigate further, filing it as a
follow-up hardening item against `ruvector-mincut` itself rather than
working around it in `ruvector-agent-memory`. This ADR is that follow-up.

## Hypothesis

```text
Given RuVectorGraphAnalyzer::partition(), called repeatedly on an
unchanged graph with more than one globally-minimum cut,

when the two root causes of its measured non-determinism and
empty/degenerate results are found by reading the implicated code paths
and fixed with minimal, targeted changes,

then repeated calls should return byte-identical, correct results,

subject to: zero regressions in the crate's existing 512-test suite, and
the 2026-09-05 acceptance benchmark's verdict changing only if the
underlying bugs — not the algorithm's fitness for that specific
task — explain its prior REJECT.
```

## Decision

Fix two independent bugs in `ruvector-mincut`, in place, without changing
any public type signature:

1. **`crates/ruvector-mincut/src/instance/bounded.rs`** —
   `BoundedInstance::brute_force_min_cut()` and
   `BoundedInstance::search_for_cuts()` each built a `Vec<VertexId>` from
   `self.vertices: HashSet<VertexId>` iteration order to drive,
   respectively, bitmask-position assignment in an exhaustive subset search
   and try-order in a first-match budget/seed search. `HashSet`'s
   iteration order depends on its `RandomState` hasher, reseeded on every
   `BoundedInstance::new()`. **Fix:** `sort_unstable()` both vectors
   immediately after collecting them, so enumeration order is a function of
   vertex ID, not construction-time entropy.
2. **`crates/ruvector-mincut/src/integration/mod.rs`** —
   `RuVectorGraphAnalyzer::partition()` called
   `witness.materialize_partition()` and trusted both halves of its return
   value. That method's second half (`v_minus_u`, in
   `instance/witness.rs`) is computed as `0..=max(membership)` minus
   `membership` — an approximation of "the rest of the graph" that is
   silently wrong (frequently empty) whenever the winning side does not
   happen to contain the graph's actual highest vertex ID. **Fix:** keep
   `materialize_partition()`'s first half (`side_a`, the witness's own
   membership — never buggy) and recompute the complement as
   `self.graph.vertices()` (the analyzer's real vertex universe) minus
   `side_a`.

## Evidence

Three benchmark stages, each using the unmodified 2026-09-05 artifacts
(`examples/mincut_determinism_probe.rs`,
`examples/mincut_gated_forgetting_bench.rs` in
`crates/ruvector-agent-memory`), full raw output in the parent nightly
report (`docs/research/nightly/2026-09-21-mincut-determinism-fix/README.md`):

| Stage | Determinism probe (200 trials) | 84-vertex bench verdict |
|---|---|---|
| Baseline | 44% empty, 56% correct | REJECT |
| Fix 1 only (`bounded.rs`) | 100% empty (now *consistently* wrong) | REJECT |
| Fix 1 + 2 | **0% empty, 100% correct** | REJECT (unchanged, expected) |

- `cargo test --release -p ruvector-mincut --lib`: 512 passed, 0 failed, 5
  ignored — identical to pre-fix baseline.
- `cargo test --release -p ruvector-agent-memory --features mincut-forget
  --lib`: 31 passed, 0 failed (was 2 failing at the fix-1-only intermediate
  stage; 0 failing at baseline only because that stage's non-determinism
  happened not to be sampled during that particular `cargo test`
  invocation — see the parent report's determinism-probe numbers for why
  that's expected to be sample-dependent, not evidence of stability).
- `cargo clippy --release -- -D warnings` clean on both crates (with and
  without the `mincut-forget` feature).
- 30/30 repeated runs of the compiled test binary with
  `mincut_trials` reduced from `10`/`3` to `1` (see Consequences) — no
  flakes observed.

## Consequences

- `ruvector_mincut::RuVectorGraphAnalyzer::partition()` (and, by extension,
  `min_cut()`, `is_well_connected()`, `CommunityDetector`, and
  `GraphPartitioner`, all of which share the same `partition()`/
  `materialize_partition()` path in `integration/mod.rs`) is now provably
  deterministic and complement-correct on the tested topology, without any
  API change.
- `ruvector_agent_memory::graph_forget::MincutGatedForgetting`'s
  `mincut_trials` field defaults to `1` instead of `3`; its two unit tests
  no longer need the `10`-trial majority-vote workaround. This is a
  measured ~10x latency reduction for any caller that previously had to
  retry to work around the bug (31 lib tests: 1.05s vs. ~10.5s).
  `mincut_trials` remains a public field (defense-in-depth; not removed).
- No change to the 2026-09-05 acceptance verdict: mincut-gated forgetting
  remains rejected for production use as designed, for the reason ADR-345
  already gave independently (global min-cut does not isolate the specific
  memories a human calls "the bridges" in this benchmark's clustered
  synthetic data), now confirmed without a live non-determinism confound.
- `WitnessHandle::materialize_partition()` itself is **not** fixed at its
  definition — only its one current production call site works around its
  bug. See Open Questions.

## Alternatives Considered

See the parent nightly report's "Rejected Alternatives" section
(more `mincut_trials`; a `WitnessHandle` constructor-level fix; returning
all tied optima) for the full reasoning. Summary: a call-site-local fix was
chosen over a constructor-level fix because it closes 100% of the currently
reachable blast radius (confirmed by grep: one non-test caller) at a
fraction of the risk of changing `WitnessHandle::new()`'s signature across
every module and test that constructs one.

## Implementation Plan

Already implemented and merged as part of this same change:

1. `instance/bounded.rs`: two `sort_unstable()` calls, doc comments
   explaining why.
2. `integration/mod.rs`: `partition()`'s match arm rewritten to compute the
   complement against `self.graph.vertices()`.
3. `ruvector-agent-memory/src/graph_forget.rs`: doc comments updated to
   record root cause + fix (history preserved, not deleted); `mincut_trials`
   defaults `3 → 1`; unit tests' `mincut_trials = 10` workaround removed.

No feature flag needed: both fixes are behavior corrections to existing,
already-shipped code paths, not new capability.

## API Shape

No public signature changed.
`RuVectorGraphAnalyzer::partition() -> Option<(Vec<VertexId>, Vec<VertexId>)>`
is unchanged; only its returned *values* are now correct.
`MincutGatedForgetting::mincut_trials: usize` is unchanged in type and
semantics; only its constructors' default value changed
(`soft()`/`hard()`, `3 → 1`).

## Feature Flags

None added or changed. `ruvector-agent-memory`'s existing `mincut-forget`
optional feature (gating the `ruvector-mincut` path dependency) is
unaffected.

## Benchmark Evidence

See Evidence above and the parent nightly report's full raw output
(`docs/research/nightly/2026-09-21-mincut-determinism-fix/README.md`,
"Benchmark Results (raw)").

## Security

No new attack surface. Determinism is a strict improvement for any
downstream witness-verification or replay consumer; no new dependencies,
no new unsafe code, no new parsing of untrusted input.

## Governance

No proof-gating, MCP authority, or autonomous-mutation surface touched.

## Failure Modes

See the parent nightly report's "Failure Modes (the core finding)"
section for the full mechanism of both bugs, including why fixing only
bug 1 (tie-breaking) made the observed failure *more* consistent (100%
wrong) rather than better — direct evidence the two bugs are independent
and both needed fixing.

## Migration

None required. Existing callers of `RuVectorGraphAnalyzer::partition()`
and `MincutGatedForgetting` get corrected behavior and (for the latter) a
faster default with no code changes on their part. Any caller that had
manually raised `mincut_trials` above `1` to work around the bug may lower
it back down; not required for correctness, only for latency.

## Rollback

Revert the two `ruvector-mincut` commits and the `graph_forget.rs` doc/
default changes; `mincut_trials` defaults return to `3`/test-local `10`.
No data migration, no schema change, no persisted state affected.

## Rejection Criteria

This fix should be reverted or reworked if: (a) a future concurrent-update
or WASM-target-specific test finds either `sort_unstable()` call
insufficient (e.g., a race in `self.vertices` population itself, not just
its later iteration); or (b) the deferred `WitnessHandle` constructor-level
fix (Open Questions) finds the call-site-local fix here was masking rather
than fully addressing the complement bug for some case this ADR's evidence
did not cover.

## Open Questions

1. Should `WitnessHandle::new()` take a `total_vertices` parameter so
   `materialize_partition()` is correct at its definition, not just at
   today's one call site? (Deferred; see Alternatives Considered.)
2. Does `ruvector_mincut::algorithm::DynamicMinCut`, used directly in place
   of `MinCutWrapper`/`BoundedInstance`, both preserve this fix's
   correctness and reduce the still-unaddressed per-call latency (tens to
   hundreds of milliseconds, confirmed unaffected by this ADR's fixes)?
   (Parent report's Next Research item 1.)
3. Do `fragment/`, `expander/`, and `jtree/` share the same
   "`HashMap`/`HashSet` iteration feeding a first-match/tie-break decision"
   bug class found in `bounded.rs`? Not audited by this ADR.
