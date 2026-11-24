# ADR-361: Eliminate Redundant λ-Instance Recomputation in `MinCutWrapper::process_instances`

## Status

Accepted. Non-breaking performance fix, merged into `ruvector-mincut`
directly (no feature flag — the prior behavior was a performance bug, not
a documented tradeoff). Two new default trait methods added to
`ProperCutInstance`; no existing signature changed.

## Context

ADR-345 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`)
rejected an agent-memory integration (`MincutGatedForgetting`) built on
`ruvector_mincut::RuVectorGraphAnalyzer::partition()`, on two independent
grounds. One was latency: a fixed-degree ring k-NN graph (`k=8`) at
`n=19` measured **69,269.9ms** for a single `partition()` call, against
76.8ms–11,415.0ms at `n=50..400` — a "small, perfectly regular ring
graph, plausibly an adversarial/degenerate case," left unexplained. The
2026-09-11 nightly (ADR-346) fixed a separate, independent determinism
bug in the same code path, confirmed the latency problem was distinct
and unfixed, and named the next step explicitly: "Root-cause the
`partition()` latency scaling problem from ADR-345 ... likely by
evaluating `connectivity::polylog::PolylogConnectivity` ... as a
replacement backend" — the "remaining blocker for `MincutGatedForgetting`."

This run investigated that lead and found the premise wrong before
writing any code. `PolylogConnectivity` (`crates/ruvector-mincut/src/connectivity/polylog.rs`)
is a pure dynamic-connectivity oracle — `insert_edge`/`delete_edge`/`connected(u,v)`/`component_count()`,
no min-cut, no boundary/witness, no "which side" API. It cannot be a
"replacement backend" for `BoundedInstance::query()` in any direct sense;
it solves a different problem, and is already unrelated to the actual hot
path (`MinCutWrapper`'s own `conn_ds: DynamicConnectivity` field, a
different, older struct, handles the cheap "is the graph disconnected at
all" short-circuit, not the expensive part).

The real, verified-by-reading root cause is in
`MinCutWrapper::process_instances` (`crates/ruvector-mincut/src/wrapper/mod.rs`):

- The paper algorithm maintains `MAX_INSTANCES = 100` geometric λ-ranges
  (`λ_min[i] = floor(1.2^i)`, `λ_max[i] = floor(1.2^(i+1))`), and sweeps
  them in increasing order from a cached `last_min_cut`-derived
  `start_idx`, instantiating and querying each `BoundedInstance` in turn
  until one reports `ValueInRange`.
- `start_idx` is `0` whenever `last_min_cut` is `None` — always true on a
  cold wrapper, which every in-tree caller constructs fresh
  (`RuVectorGraphAnalyzer::new()`, called by `CommunityDetector`,
  `GraphPartitioner`, and every `MincutGatedForgetting` compaction call),
  so the warm-start optimization never actually triggers in practice.
- For `n < 20`, `BoundedInstance::query()` calls `brute_force_min_cut()`:
  an exhaustive `2^n`-subset enumeration that computes the
  **unconstrained global** minimum cut — it does not use
  `[lambda_min, lambda_max]` to prune at all, only to classify the
  already-fully-computed answer afterward (`if value <= self.lambda_max`).

Because each swept instance `i` is a **freshly constructed**
`BoundedInstance` (`MinCutWrapper`'s `instances: Vec<Option<Box<dyn
ProperCutInstance>>>` starts every index at `None`), the identical
bound-independent `2^n` enumeration is redundantly recomputed **once per
λ-instance visited**, not once total. This exactly explains both
measurements without invoking non-determinism or hardware variance: a
graph whose true min cut falls in instance 0's range (e.g. the
`mincut_determinism_probe.rs` bridge graph, min cut 1) pays the brute
force once; a graph whose true min cut requires sweeping many ranges
(e.g. the `k=8` regular ring, where every vertex has degree 8) pays it
once per range visited before the right one is reached.

## Hypothesis

```text
Given MinCutWrapper::process_instances() on a cold wrapper (start_idx=0),
over a graph with n < 20 vertices whose true minimum cut value is not in
instance 0's range [1, 1],

when BoundedInstance exposes the true, bound-independent value it already
computes via brute_force_min_cut() even when reporting AboveRange
(ProperCutInstance::exact_value_hint), and process_instances uses it to
jump directly to the λ-instance that actually contains that value —
seeding that instance (ProperCutInstance::seed_known_value) instead of
letting it recompute the identical answer itself,

then end-to-end query() latency for such graphs drops by roughly the
factor of λ-instances previously swept, with no change to the returned
cut value or witness,

subject to: every existing ruvector-mincut test remaining green: the
n<20 (brute-force) and n>=20 (LocalKCut) code paths untouched for graphs
where no skip is possible (true value already in instance 0's range, or
n>=20); and determinism (ADR-346) preserved.
```

Full investigation, methodology, and raw benchmark output are in
`docs/research/nightly/2026-11-24-mincut-redundant-sweep/README.md`.

## Decision

1. `ProperCutInstance` (`crates/ruvector-mincut/src/instance/traits.rs`)
   gains two new methods, both with no-op/`None` defaults so no existing
   implementor (`StubInstance`, any external one) needs to change:
   - `exact_value_hint(&self) -> Option<(u64, WitnessHandle)>` — the
     true, bound-independent cut value and witness, when the
     implementation's computation doesn't depend on its own bounds.
   - `seed_known_value(&mut self, value: u64, witness: &WitnessHandle)` —
     accept an externally-computed `(value, witness)` pair to skip
     recomputing it internally.
2. `BoundedInstance` (`crates/ruvector-mincut/src/instance/bounded.rs`)
   implements both, gated on `self.vertices.len() < 20` (the same
   threshold `query()` already uses to choose the brute-force path):
   `exact_value_hint` returns the cached `best_witness` (already
   populated by `brute_force_min_cut()` regardless of range, per the
   existing code); `seed_known_value` writes directly into
   `best_witness` and the boundary cache, which `query()`'s existing
   top-of-function cached-witness check (`value >= lambda_min && value
   <= lambda_max`) already honors unmodified.
3. `MinCutWrapper::process_instances`
   (`crates/ruvector-mincut/src/wrapper/mod.rs`) changes its `for i in
   start_idx..MAX_INSTANCES` sweep into a `while i < MAX_INSTANCES` loop.
   On `AboveRange`, it now calls `instance.exact_value_hint()`; if
   `Some((value, witness))`, it computes `target =
   binary_search_lambda_max(&self.lambda_max, value)` (a new associated
   function factored out of the existing `find_instance_for_value`, as a
   plain-slice function rather than a `&self` method — necessary so it
   can be called while a `&mut self.instances[i]` borrow is still live)
   and, if `target > i`, jumps directly there, carrying `(value,
   witness)` to be passed to the target instance's `seed_known_value`
   once it is instantiated. If no hint is available (the `n >= 20`
   `LocalKCut` path, where the result genuinely depends on the range),
   the sweep proceeds exactly as before, one instance at a time.
4. New regression tests: four white-box tests on `BoundedInstance`
   (`exact_value_hint`/`seed_known_value` behavior, including a
   byte-identical-witness equivalence proof between an independently
   computed answer and a seeded one) and three `MinCutWrapper`-level
   tests, including one (`test_skip_ahead_elides_intermediate_instances_k6`)
   that asserts `num_instances() == 2` after one query on K6 — a direct,
   implementation-level signature that the skip actually happened, not
   just that the returned value is correct.
5. New example `crates/ruvector-mincut/examples/process_instances_sweep_variants.rs`:
   a three-variant comparison (`baseline` / `candidate_b` exponential
   doubling+binary search / `candidate_a` the shipped fix), all three
   fed the same canonical, de-duplicated edge list, with `assert_eq!`
   gates proving all three agree on the returned value.
6. New criterion benchmark group `bounded_skip_ahead_small_graph` in
   `crates/ruvector-mincut/benches/bounded_bench.rs`, as an ongoing
   regression guard (fast — milliseconds — unlike the pathological case
   this fixes, which is deliberately not run under criterion's default
   sampling; see that file's doc comment).

## Evidence

All measurements: release build, this session's container (`rustc
1.97.0`, `x86_64`), `ruvector-agent-memory`'s existing
`examples/mincut_scaling_probe.rs` and `mincut_determinism_probe.rs`
(unmodified — reused, not rewritten, per this process's convention), and
the new `process_instances_sweep_variants.rs`. Baseline measured by
`git stash`-ing this change and rebuilding; brute-force call counts
confirmed by temporary `eprintln!` instrumentation in
`brute_force_min_cut()` (added, measured, then removed before this diff
— not part of the shipped change).

| Metric | Baseline (this hardware) | Fixed (this hardware) | Result |
|---|---|---|---|
| `mincut_scaling_probe`, n=19 (k=8 ring), `partition()` | 81,766.3ms / 83,409.6ms (two runs) | 4,922.8ms / 5,104.8ms (two runs) | **~16-17x faster** |
| `mincut_scaling_probe`, n=50 | 75.5ms / 78.3ms | 74.6ms | unchanged (noise-level) — n>=20 path, no hint available |
| `mincut_scaling_probe`, n=100 | 415.0ms / 416.8ms | 400.8ms | unchanged (noise-level) |
| `mincut_scaling_probe`, n=200 | 1562.2ms / 1647.6ms | 1497.6ms | unchanged (noise-level) |
| `mincut_scaling_probe`, n=400 | 5242.9ms / 5385.7ms | 4899.5ms | unchanged (noise-level) |
| `brute_force_min_cut()` invocations, n=19 ring (instrumented count) | 16 | 1 | matches the ~16x latency ratio exactly |
| `mincut_determinism_probe` (bridge graph, min cut 1, instance 0 already in range — no skip possible) | ~841-1130ms/call (prior nightlies) | 1072.1ms/call, 50/50 trials correct, 0% degenerate | unchanged — correctly **not** affected, since no skip applies here |
| `process_instances_sweep_variants`, n=19 k=8 ring: baseline / candidate_b (exp. doubling) / candidate_a (shipped) | 16 / 10 / 2 instance visits; 95,736ms / 61,060ms / 6,415ms | — | all three return identical value (16); strict ordering baseline > candidate_b > candidate_a on both visits and latency |
| `process_instances_sweep_variants`, n=10 k=6 ring (secondary case) | 13 / 8 / 2 visits; 65.4ms / 40.4ms / 5.4ms | — | all three return identical value (9) |
| `wrapper::tests::test_skip_ahead_elides_intermediate_instances_k6` (K6, min cut 5) | — | `num_instances() == 2` after one query (vs. 9 pre-fix) | PASS |
| Full `ruvector-mincut` test suite (`--lib --tests`, default features) | — | 521 lib + 18+8+16+8+20 integration = all green | PASS (0 failed, 3 pre-existing ignored) |
| Full `ruvector-mincut` test suite, `--all-features` | — | 718 passed, 1 failed | 1 failure (`wasm::canonical::tests::*`) confirmed **pre-existing** — fails identically on unmodified code (verified via `git stash`), unrelated module, not touched by this change |
| `ruvector-agent-memory` test suite, `mincut-forget` feature | — | 51+1+10+8+13 = all green | PASS |
| Clippy, `ruvector-mincut --all-features` | — | 0 warnings in any touched file (1 pre-existing warning in dependency `ruvector-graph`, unrelated) | PASS |

Raw benchmark/test commands are in the nightly README's [Benchmark
Methodology](../research/nightly/2026-11-24-mincut-redundant-sweep/README.md#benchmark-methodology)
section.

## Consequences

**Positive:**
- Removes ADR-345's stated blocker for `MincutGatedForgetting` on
  exactly the measured dimension (latency on small, high-min-cut
  graphs), though — see Open Questions — `MincutGatedForgetting` itself
  remains rejected pending the separate `n>=20` scaling problem and the
  determinism-fix-introduced environment variance, neither addressed
  here.
- All consumers of `RuVectorGraphAnalyzer::partition()`,
  `CommunityDetector`, and `GraphPartitioner` get the speedup for free on
  next rebuild — no migration.
- Falsifies, with evidence, a specific technical direction (swapping in
  `PolylogConnectivity`) that two prior nightlies had queued as the next
  step, so no future run rediscovers it blindly.
- The `ProperCutInstance` trait extension is a generically useful hook:
  any future bound-independent instance implementation (not just
  brute-force) can opt into the same skip-ahead for free via the default
  `None`/no-op fallback costing nothing to non-participating
  implementors.

**Negative / accepted tradeoffs:**
- Only the `n < 20` brute-force path is fixed. The `n >= 20`
  `search_for_cuts`/`LocalKCut` path's own scaling problem (ADR-345's
  "~77ms (n=50) to ~11.4s (n=400)," worse-than-quadratic) is untouched —
  `exact_value_hint` correctly returns `None` there, by design, since a
  local, budget-bounded search's result is genuinely range-dependent,
  not a free bound-independent fact to exploit the same way. This is
  deliberate scope discipline (Pass 3's "is the hypothesis falsifiable"
  and "simpler alternative" questions), not an oversight — see Open
  Questions.
- `process_instances` is slightly more complex (a `while` loop with a
  carried `pending_seed` instead of a plain `for`), though the added
  logic is a single, well-tested conditional branch (~20 of the
  function's ~115 lines).

## Alternatives

- **Wire `PolylogConnectivity` into `BoundedInstance` as a replacement
  connectivity backend** (the direction both prior nightlies queued).
  Rejected: structural mismatch. `PolylogConnectivity`'s only surface is
  `insert_edge`/`delete_edge`/`connected`/`component_count` — no min-cut,
  no witness, no boundary. It cannot answer "what is the cut value and
  which side is which," which is what `BoundedInstance::query()` must
  return. Even narrowly swapped in for `MinCutWrapper`'s cheap
  `conn_ds` fast-path check, it would not touch the measured bottleneck
  (confirmed by investigation before writing any code, per Pass 3 of
  this process).
- **Exponential-doubling probe + binary search over λ-instance indices**
  (`candidate_b` in `process_instances_sweep_variants.rs`), without
  exploiting that the brute-force value is exact. Real, measured,
  correct (returns the same value), and strictly better than the
  original linear sweep (O(log target) vs. O(target) recomputations:
  10 vs. 16 visits on the n=19 case). Rejected as the shipped fix
  because it is strictly dominated by `candidate_a` (2 visits, 1 real
  recomputation) on every measured dimension — correctness, visit count,
  and wall-clock — with no offsetting advantage. Kept in the example as
  the honest "is there a simpler alternative that doesn't require
  understanding the specific redundancy" comparison point.
- **Memoize `brute_force_min_cut()`'s result in a cache shared across
  `BoundedInstance` objects, keyed by graph content hash.** Would also
  work, but requires a shared, invalidated-on-mutation cache structure
  at the wrapper level — strictly more moving parts than exposing the
  value the instance already computed and seeding the one instance that
  needs it. Rejected as disproportionate complexity for the same result.

## Implementation Plan

Already implemented in this PR:

1. `crates/ruvector-mincut/src/instance/traits.rs` — add
   `exact_value_hint`/`seed_known_value` default methods to
   `ProperCutInstance`.
2. `crates/ruvector-mincut/src/instance/bounded.rs` — implement both for
   `BoundedInstance`; add four regression tests.
3. `crates/ruvector-mincut/src/wrapper/mod.rs` — skip-ahead logic in
   `process_instances`; factor `binary_search_lambda_max` out of
   `find_instance_for_value`; add three regression tests.
4. `crates/ruvector-mincut/examples/process_instances_sweep_variants.rs`
   — new three-variant comparison harness.
5. `crates/ruvector-mincut/benches/bounded_bench.rs` — new
   `benchmark_skip_ahead_small_graph` criterion group.

## API Shape

Two new trait methods on `ProperCutInstance`, both with defaults
(`None` / no-op), so this is additive, not breaking:

```rust
fn exact_value_hint(&self) -> Option<(u64, WitnessHandle)> { None }
fn seed_known_value(&mut self, _value: u64, _witness: &WitnessHandle) {}
```

No existing public signature (`MinCutWrapper::query`,
`RuVectorGraphAnalyzer::partition`, `ProperCutInstance::query`, etc.)
changed.

## Feature Flags

None. This is a correctness-preserving performance fix to the default
(always-on) code path, not a new capability — no flag is appropriate,
consistent with ADR-346's precedent for this same module.

## Benchmark Evidence

See [Evidence](#evidence) above and the nightly README's full
methodology, raw output, and the three-variant comparison's complete
printed output.

## Security

No security-relevant surface changed. No new unsafe code, no new
dependency, no change to witness/signature semantics. The fix is purely
about *which* instance computes an answer and *when*, not what the
answer means.

## Governance

Standard PR review. No schema, wire-format, or persisted-data changes.

## Failure Modes

- If a future caller relies on `MinCutWrapper` re-visiting every
  intermediate λ-instance as an (undocumented, accidental) side effect —
  e.g., to pre-warm `self.instances[k]` for `k` between `start_idx` and
  the true target — that side effect no longer happens. No such reliance
  was found in this workspace (`CommunityDetector`, `GraphPartitioner`,
  `MincutGatedForgetting` all only consume the final `query()`/`partition()`
  result), but it is a behavior change worth naming explicitly.
- The skip-ahead is a strict no-op whenever `exact_value_hint()` returns
  `None` (always true for `n >= 20`) or `target <= i` (the common case:
  true value already in the first visited range) — verified by
  `test_no_skip_when_first_instance_already_in_range` and
  `test_large_graph_unaffected_by_skip_ahead`.

## Migration

None required. No signature changes; recompiling dependent crates
against this version is sufficient.

## Rollback

Revert the three source-file changes (`traits.rs`, `bounded.rs`,
`wrapper/mod.rs`); the new tests, example, and benchmark can be deleted
or left in place (the tests will fail against pre-fix code, which is the
correct, expected outcome of a regression test for a performance bug
that has been reintroduced). Reverting reintroduces the ~16x latency
regression on small, high-min-cut graphs — not recommended.

## Rejection Criteria

This decision should be revisited if:

- A future `ProperCutInstance` implementor's `exact_value_hint` turns
  out to be unsound (returns a value that does not actually hold
  regardless of bounds) — this would silently corrupt `process_instances`'
  skip-ahead for that implementor. Mitigated today by the trait's
  documented contract and `BoundedInstance`'s narrow, verified-correct
  gate (`vertices.len() < 20`, matching `query()`'s own branch exactly).
- Profiling shows the `n >= 20` `LocalKCut` path's separate scaling
  problem (still open — see Open Questions) dominates real workloads
  enough that this fix's contribution becomes immaterial in practice.

## Open Questions

1. Does the `n >= 20` `search_for_cuts`/`LocalKCut` path have an
   analogous redundancy, or a structurally different bottleneck?
   Unresolved — explicitly out of scope this run (see Alternatives); the
   fork's investigation found the redundant-sweep mechanism applies
   cleanly only where a single query computes an unconstrained,
   range-independent answer, which brute force does and `LocalKCut`'s
   budgeted local search does not. Flagged as next research.
2. Does this fix, combined with ADR-346's determinism fix, change
   `MincutGatedForgetting`'s overall acceptance verdict (ADR-345,
   REJECTED)? Not re-run this session — deliberately: Step 32 of this
   process ("Do not change the hypothesis after seeing benchmark
   results") and the scope discipline of finishing *this* specific
   thread rather than re-opening a different, larger rejected
   experiment. Flagged as next research; the `n>=20` scaling problem
   (Open Question 1) would need addressing first regardless.
3. Would a shared, content-hash-keyed memoization cache (the rejected
   alternative) generalize better to a future instance implementation
   that is *expensive but still bound-independent* in a way brute force
   isn't (e.g., an exact algorithm with superlinear but sub-exponential
   cost)? Unresolved; `exact_value_hint`'s current contract (return the
   value directly) is sufficient for every implementor in this
   workspace today.
