# ADR-353: In-Repo Rust Pareto Frontier for the Darwin Evolutionary Harness

## Status

Proposed

## Context

[ADR-335](./ADR-335-repo-native-optimization-manifest.md) Decision §6
records, as an explicit open item (work package "WP31"):

> **Pareto frontier — decide explicitly.** `darwin.ts` passes
> `selection: "pareto"` to `evolve()` from the **external** npm package
> `@metaharness/darwin@^0.9.1`. This repo consumes a frontier
> implementation it does not own. ... requires WP31 to either (a)
> implement an in-repo frontier that this program controls, or (b)
> explicitly accept the external dependency with a recorded rationale.
> Silently depending on it while claiming frontier semantics is not an
> option.

No Rust crate in this workspace implements Pareto dominance or
non-dominated sorting — `grep -ri pareto crates/` returns zero matches; the
concept exists only in `crates/ruvector-sota-bench/harness/src/darwin.ts`
(TypeScript, consuming the external package) and in
`schemas/optimization-manifest-v1.json`'s `promotion.pareto` schema fields
(`objectives: [{name, direction}]`, `require_non_dominated: const true`),
which already declare the shape a frontier computation needs to accept
without specifying who computes it.

`crates/sona/src/darwin_guard.rs` (ADR-271) already hosts this repository's
reward-hacking screening for evolutionary/Darwin-style search — a
deterministic-monitor guard that screens *one* candidate's fitness for
degeneracy. It does not rank candidates against each other, which is the
gap this ADR addresses.

## Hypothesis

```text
Given a set of N candidate configurations from a Darwin evolutionary
generation, each scored on M objectives (the shape
schemas/optimization-manifest-v1.json's promotion.pareto.objectives already
declares per repository),

when the Pareto frontier is computed via three independent in-repo Rust
algorithms — brute-force all-pairs (the formal definition), NSGA-II's fast
non-dominated sort, and a 2D-specialized O(N log N) divide-and-conquer
skyline algorithm (Kung, Luccio & Preparata, 1975) —

then all three must return bit-identical frontier membership on every
generated test instance, and at least one in-repo algorithm must measurably
outperform naive brute force at Darwin-lineage-relevant N,

subject to: zero external dependency in the new code path, deterministic
output for fixed input ordering, and the full pre-existing ruvector-sona
test suite remaining green.
```

## Decision

1. **Add `crates/sona/src/darwin_pareto.rs`** implementing `Direction`,
   `dominates`, `brute_force_frontier`, `nsga2_frontier`,
   `kung_frontier_2d`, `select_frontier` (dispatcher), `crowding_distance`,
   and `is_dominated_by_frontier` (the literal `require_non_dominated`
   gate). Declared via `pub mod darwin_pareto;` in `crates/sona/src/lib.rs`,
   following `darwin_guard`'s existing convention (module declared, not
   re-exported at crate root).
2. **`select_frontier` dispatches by objective count**: the specialized
   `O(N log N)` `kung_frontier_2d` for exactly two objectives, and
   `nsga2_frontier` for any other count — chosen because the benchmark
   evidence below shows `nsga2_frontier` is *not* a practical win over
   brute force for `M = 2`, not because of an a priori complexity argument.
3. **Do not wire this module into `darwin.ts`'s actual `selection: "pareto"`
   call site in this change.** `@metaharness/darwin@^0.9.1` remains the live
   dependency for now. This ADR documents and ships the in-repo
   *alternative*; the FFI/WASM integration decision that would actually
   retire the external dependency is scoped to a follow-up (see
   [Implementation Plan](#implementation-plan)).
4. **No equivalence claim against `@metaharness/darwin`'s own algorithm is
   made.** Its internals are not characterized by ADR-335 or this ADR.
   `darwin_pareto` is verified against the formal mathematical definition
   of Pareto dominance (via `brute_force_frontier` as ground truth)
   instead, which is a stronger, dependency-independent standard.

## Evidence

- **Correctness**: `cargo test -p ruvector-sona --lib` — 115 passed, 0
  failed (11 new tests for `darwin_pareto`, including
  `random_2d_instances_agree_with_brute_force` — 200 seeds — and
  `random_3d_and_4d_instances_agree_with_brute_force` — 200 seeds across
  M=3 and M=4). Zero disagreements between `nsga2_frontier`/
  `kung_frontier_2d` and `brute_force_frontier` across 407 total generated
  and hand-authored cases.
- **Benchmark** (`cargo bench -p ruvector-sona --bench darwin_pareto_bench
  -- --quick`; `rustc 1.97.0`, Linux x86_64, release profile):

  | N | baseline (brute force) | candidate_A (NSGA-II) | candidate_B (Kung 2D) |
  |---|---|---|---|
  | 100 | 2.03 µs | 10.81 µs (5.3x slower) | 2.65 µs (1.3x slower) |
  | 500 | 10.10 µs | 921.72 µs (91.3x slower) | 26.86 µs (2.66x slower) |
  | 2,000 | 90.94 µs | 16.32 ms (179.5x slower) | 85.19 µs (6% **faster**) |
  | 5,000 | 472.28 µs | 124.67 ms (264.0x slower) | 299.39 µs (58% **faster**) |
  | 20,000 | not measured (cost) | not measured (cost) | 1.79 ms |
  | 100,000 | not measured (cost) | not measured (cost) | 15.42 ms |

  Full methodology, raw output, and the measurement-cost rationale for the
  unmeasured cells are in the companion nightly research document.
- **Finding, stated plainly**: the original design assumption —
  "`nsga2_frontier` is the right choice once a lineage accumulates many
  generations' candidates," based on its textbook complexity class — is
  **falsified** by this benchmark. `nsga2_frontier` does unconditional
  `O(N^2)` work with no early exit; `brute_force_frontier`'s per-point scan
  exits as soon as one dominator is found, which captures nearly all the
  available speedup on the sparse-frontier data this experiment's random
  2D instances produce. Only `kung_frontier_2d`'s genuinely different
  `O(N log N)` complexity class produces a measured win, and only above a
  crossover between N=2,000 and N=5,000. The module's doc comments were
  rewritten during this run to state the measured result instead of the
  falsified assumption.

## Consequences

### Positive

- This repository gains a dependency-free, test-verified Pareto frontier
  primitive it fully controls, closing the correctness half of ADR-335
  WP31.
- The `select_frontier` dispatcher is evidence-based rather than textbook-based:
  it never routes `M=2` traffic through the algorithm this run measured to
  be slower than the thing it was meant to improve on.
- A documented, falsified sub-hypothesis (NSGA-II's assumed scaling
  advantage) is retained as evidence rather than silently corrected,
  per this program's evidence-retention norms.

### Negative

- The external dependency (`@metaharness/darwin@^0.9.1`) is **not** removed
  by this change — `darwin.ts` is untouched. ADR-335 WP31 remains only
  half-closed (in-repo alternative exists; it is not yet the live path)
  until the follow-up FFI integration lands.
- `nsga2_frontier` remains the only `M > 2` option despite being measurably
  the weakest algorithm in this module for `M = 2` — acceptable because it
  is only reached for `M > 2`, where no faster in-repo alternative exists
  yet (see [Rejection Criteria](#rejection-criteria) for when this should be
  revisited).
- Benchmark coverage stops at N=5,000 for two of the three algorithms on
  measurement-cost grounds; the N=20,000/100,000 comparison rows for those
  two are extrapolated, not measured.

## Alternatives

- **Implement general-M Kung et al. (1975) divide-and-conquer** instead of
  specializing to M=2. Rejected for this change: materially higher
  implementation-correctness risk for a single-night change; `nsga2_frontier`
  covers `M > 2` correctly, if not fast.
- **Reverse-engineer and claim equivalence with `@metaharness/darwin`'s own
  algorithm.** Rejected: unverifiable without inspecting a package this ADR
  does not characterize, and constraint-level guidance (no treating external
  results as comparable without demonstrably equivalent methodology) argues
  against it. Verifying against the formal definition of Pareto dominance is
  the chosen, stronger standard.
- **Wire this module into `darwin.ts` immediately via FFI/WASM.** Rejected
  as out of scope for this change: that is a TypeScript-side design decision
  (how `normalizePolicy`'s closed lever set and the "intersect, never
  substitute" manifest rule in ADR-335 interact with a WASM-backed call)
  that deserves its own scoped review rather than being bundled into a
  Rust-only correctness/benchmark change.
- **Keep the original (pre-benchmark) doc-comment claim that NSGA-II wins at
  scale.** Rejected once the benchmark disagreed with it — shipping a
  plausible-sounding but measured-false performance claim in a doc comment
  would be worse than shipping none.

## Implementation Plan

1. (This change) `crates/sona/src/darwin_pareto.rs` + tests + benchmark +
   example, as described in [Decision](#decision).
2. (Follow-up, separately scoped) Design the `darwin.ts` ↔ `darwin_pareto`
   FFI/WASM boundary and switch `selection: "pareto"`'s implementation over,
   retiring `@metaharness/darwin@^0.9.1` for that call site. This is the
   change that actually closes ADR-335 WP31's "(a) implement an in-repo
   frontier" branch in full.
3. (Follow-up) Smoke-test `ruvector-sona`'s `wasm32` build with this module
   included.
4. (Follow-up, conditional) A general-M fast frontier algorithm, only if a
   real `M > 2` use case needing better-than-`nsga2_frontier` performance
   materializes (see [Rejection Criteria](#rejection-criteria)).

## API Shape

```rust
pub enum Direction { Maximize, Minimize }

pub fn dominates(objectives: &[Direction], a: &[f64], b: &[f64]) -> bool;
pub fn brute_force_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize>;
pub fn nsga2_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize>;
pub fn kung_frontier_2d(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize>; // panics if objectives.len() != 2
pub fn select_frontier(objectives: &[Direction], points: &[&[f64]]) -> Vec<usize>; // dispatcher
pub fn crowding_distance(objectives: &[Direction], points: &[&[f64]], front: &[usize]) -> Vec<f64>;
pub fn is_dominated_by_frontier(objectives: &[Direction], candidate: &[f64], frontier: &[&[f64]]) -> bool;
```

`objectives`, `Direction`, and the `require_non_dominated` semantics of
`is_dominated_by_frontier` mirror `schemas/optimization-manifest-v1.json`'s
`promotion.pareto` fields directly, so a future manifest-driven caller needs
no translation layer beyond deserializing `"maximize"`/`"minimize"` strings
into `Direction`.

## Feature Flags

None added. `darwin_pareto` is unconditionally compiled as part of
`ruvector-sona`'s existing `src/lib.rs` module list, with no new optional
dependency (criterion/rand were already present as dev-dependencies for the
pre-existing `sona_bench`).

## Benchmark Evidence

See [Evidence](#evidence) above and the companion nightly research document
(`docs/research/nightly/2026-10-16-darwin-pareto-frontier/README.md`) for
full methodology, raw Criterion output, and scaling analysis.

## Security

No new attack surface: pure computation over caller-supplied `f64` slices,
no I/O, no unsafe code, no new dependencies. `kung_frontier_2d`'s panic on
`objectives.len() != 2` is a deliberate fail-fast against programmer error,
not a reachable DoS surface — the module has no external caller yet.

## Governance

Standard PR review. No schema, wire-format, or persisted-data change.
`schemas/optimization-manifest-v1.json` was read for shape reference, not
modified.

## Failure Modes

- `nsga2_frontier` is measurably the slowest option for `M = 2` — mitigated
  by `select_frontier` never routing `M = 2` traffic through it.
- `kung_frontier_2d` panics (not silently mis-answers) for `objectives.len()
  != 2` — tested.
- Float equality for tie detection has no epsilon tolerance — acceptable for
  deterministic benchmark-sourced inputs, flagged as an open question for
  any future use with noisy/averaged metrics.

## Migration

None required — purely additive module, zero existing callers, no changed
public API elsewhere in the workspace.

## Rollback

Revert the four changed/added files in `crates/sona/` (`darwin_pareto.rs`,
`lib.rs`'s one-line module declaration, `darwin_pareto_bench.rs`,
`Cargo.toml`'s `[[bench]]` entry) and the new example. No downstream code
depends on this module (none exists yet), so rollback has zero blast
radius.

## Rejection Criteria

Revisit `nsga2_frontier`'s continued use as the `M > 2` fallback if:

- A real `M > 2` use case needs materially better throughput than this
  benchmark measured (124.67 ms at N=5,000, M=2 — M>2 would be slower per
  objective added), at which point implementing a general-M fast algorithm
  (deferred in [Alternatives](#alternatives)) should be reconsidered.
- A future benchmark at larger N finds `nsga2_frontier` eventually
  overtakes brute force at some crossover not observed in this run's
  N≤5,000 range — would not invalidate this ADR's `M=2` dispatch decision
  (which never reaches `nsga2_frontier` for `M=2` regardless), but should be
  recorded as a correction to this document's stated finding.

## Open Questions

1. Does `@metaharness/darwin@^0.9.1`'s actual Pareto algorithm match the
   formal definition this module implements, or does it do something
   subtly different (e.g. epsilon-dominance, a different tie-breaking
   rule)? Not investigated — the external package's internals remain
   unexamined by this ADR.
2. Is `kung_frontier_2d`'s above-ideal scaling constant at N≥20,000 (see
   companion research doc) a sorting/allocation artifact or something more
   structural? Flagged as a profiling follow-up, not resolved here.
3. What is the right epsilon-tolerance policy for tie detection if this
   module is ever fed noisy (not deterministically benchmark-sourced)
   objective values?
