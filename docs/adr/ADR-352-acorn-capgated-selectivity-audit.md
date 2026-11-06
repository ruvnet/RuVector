# ADR-352: ACORN-style degree/γ-augmentation and hierarchical seeding for `ruvector-capgated` — audited and rejected at production budget

**Status**: Rejected (evidence retained)
**Date**: 2026-10-08
**Author**: Nightly Research Agent
**Branch**: `claude/focused-darwin-fj1krt`
**Crate**: `crates/ruvector-capgated`
**Related**: ADR-268 (Capability-Gated ANN Search, defines `CapGraphIndex` and its "replace with HNSW for production" TODO), ADR-227 (Proof-Gated Writes), the 2026-04-26 nightly ACORN implementation (`crates/ruvector-acorn`)

> **Numbering note.** `scripts/adr-index.mjs` reported `ADR-352` as next-available against `main` at this branch's base commit. This repository's nightly process runs many parallel branches off the same base; other open PRs from the same window may independently claim the same number (a known, tolerated condition — see ADR-351's own numbering note for precedent). If `352` is taken at merge time, renumber and keep this note.

---

## Context

ADR-268 introduced `ruvector-capgated`'s `CapGraphIndex`: a flat, single-layer k-NN graph that enforces per-vector capability checks during traversal, expanding a visited node's neighbours regardless of that node's own authorisation (to avoid beam starvation under filtering — the same insight ACORN, SIGMOD 2024, formalised for generic metadata predicates). The crate's source documents, verbatim, that this construction is "fine for PoC, replace with HNSW for production," and its own benchmark (`src/bin/benchmark.rs`) only measures 12.5%–37.5% authorised selectivity. ACORN's paper reports standard filtered-HNSW recall collapsing specifically below ~2% selectivity — a regime this crate had never measured, despite its own README naming "thousands of agents on one index" (implying single-digit-percent or lower per-agent selectivity) as its motivating use case.

The 2026-04-26 nightly run (`docs/research/nightly/2026-04-26-acorn-filtered-hnsw/`) already implemented and validated ACORN's two levers — γ-augmented degree and predicate-agnostic traversal — for *generic* boolean predicates in a separate crate, `ruvector-acorn`. It did not test the capability-bitset (`CapMask`) model `ruvector-capgated` uses, and did not test whether those levers transfer to a crate whose predicate-agnostic traversal piece ACORN motivates is already implemented.

## Hypothesis

```text
Given a capability-gated corpus (n=4000, d=64) where each vector requires one of
64 capability bits and a querier holds one bit (auth ≈ 1.5625%),

H1: CapGraphIndex built with degree=48 (γ=4x) instead of degree=12 increases
    recall@10 by >= 10 percentage points vs. the unmodified baseline, holding
    traversal and ef budget fixed.

H2: a new sparse-top-layer hierarchy for entry-point seeding (instead of fixed,
    evenly-spaced-by-index entry points) increases recall@10 by >= 5 percentage
    points vs. baseline, holding degree and ef budget fixed.

Subject to: build time <= 3x baseline, QPS >= 30% (H1) / 70% (H2) of baseline,
no regression > 2pp at the crate's existing high-access scenario, and all
pre-existing crate tests remaining green.
```

## Decision

**Reject both H1 and H2** at the crate's existing production-facing default budget (`ef_multiplier=30`). Do **not** change `CapGraphIndex`'s default degree and do **not** add a hierarchical variant to any default construction path.

**Retain**, as tested, documented, opt-in code (not wired into any default):

- `HierarchicalCapGraphIndex` (new module, `crates/ruvector-capgated/src/hierarchical.rs`), implementing the crate's existing `CapGatedIndex` trait.
- A new benchmark/research artifact, `crates/ruvector-capgated/examples/selectivity_acorn_bench.rs`, kept in-tree so a future run does not re-propose either lever in isolation without first re-deriving the `ef`-budget interaction this run found.

No existing public API changed. `src/bin/benchmark.rs` and its acceptance thresholds are untouched.

## Evidence

Full raw output, methodology, and the supplementary root-cause diagnostic are in `docs/research/nightly/2026-10-08-acorn-capgated-selectivity/README.md`. Summary:

| Variant | Low-access (1.56%) recall@10 | QPS | Build(ms) |
|---|---:|---:|---:|
| baseline: `CapGraphIndex(degree=12)` | 0.585 | 5,619 | 890 |
| candidate_A: `CapGraphIndex(degree=48, γ=4)` | 0.548 | 2,539 | 845 |
| candidate_B: `HierarchicalCapGraphIndex(degree=12, 1/16 top)` | 0.587 | 5,632 | 832 |

Both H1 (`Δ=-0.0373`, needed `≥+0.10`) and H2 (`Δ=+0.0013`, needed `≥+0.05`) fail at this budget. A supplementary `ef`-budget sweep (run after the frozen verdict; does not affect it) shows candidate_A overtakes baseline once `ef_multiplier≥~100` (visiting ≥25% of the corpus) and eventually reaches recall=1.000, while baseline plateaus at 0.979 even when visiting the entire graph — a pre-existing directed-graph connectivity gap in `CapGraphIndex`, not introduced by this change.

28/28 crate tests pass (22 pre-existing, unchanged; 6 new, for `HierarchicalCapGraphIndex`). `cargo clippy -p ruvector-capgated --release -- -D warnings` is clean (see PR test plan).

## Consequences

- No behaviour change for any existing caller of `ruvector-capgated`.
- Two new, real, independently falsifiable findings are now documented and available to any future work on this crate: (1) degree/seeding tuning must be budget-aware, not budget-agnostic — a fix applied without raising `ef` can make things worse; (2) the existing flat k-NN graph has a measurable, non-zero recall ceiling at full traversal, due to directed-edge asymmetry, affecting every caller equally regardless of authorisation.
- The natural next lever (a budget-aware or wall-clock stopping rule, see "Open questions") is now the named highest-priority follow-up for this crate, ahead of re-attempting degree or layering changes in isolation.

## Alternatives considered

- **Modify `ruvector-acorn` instead of `ruvector-capgated`.** Rejected: `ruvector-acorn` already has this result for generic predicates (2026-04-26); the open question was specifically whether it transfers to the `CapMask` bitset model, which requires testing `ruvector-capgated` itself.
- **"Strict isolation" traversal (don't expand unauthorised nodes' neighbours).** `cap_graph.rs` already documents this as a security/recall tradeoff distinct from degree or layering; testing it here would have conflated two independent variables. Left for a future, separately-scoped run if security requirements change.
- **Raising `ef_multiplier` as the fix, tested now instead of named as future work.** Considered but rejected for this run: the frozen hypothesis was about degree and seeding specifically; changing the budget after seeing H1/H2 fail would be adjusting the acceptance criterion post-hoc, which this harness's rules forbid. It is named explicitly as the next experiment instead.

## Implementation plan (if a future run revisits this)

1. Implement a budget-aware stopping rule (e.g., stop when the k-th-best candidate in the result heap hasn't improved in the last `w` pops) or a wall-clock budget, as `CapGatedIndex`'s third configurable axis alongside degree and entry-point strategy.
2. Re-run this exact harness (`selectivity_acorn_bench.rs`) with the new stopping rule substituted for the fixed `ef_multiplier`, at the same frozen thresholds, before concluding anything new.
3. Only then reconsider whether γ-augmentation or hierarchical seeding should move from opt-in to default.

## API shape

No API changes in this ADR. A future budget-aware variant would most naturally add a `StoppingRule` enum (`FixedNodeCount(usize)` | `WallClock(Duration)` | `Adaptive{window: usize}`) to `CapGraphIndex`/`HierarchicalCapGraphIndex`'s builder methods, replacing the single `ef_multiplier: usize` field — not designed or implemented here.

## Feature flags

None introduced. Both new types are plain public modules, usable but not default-constructed anywhere.

## Benchmark evidence

See **Evidence** above and the full report. Command: `cargo run --release -p ruvector-capgated --example selectivity_acorn_bench`. Deterministic (fixed LCG seed); reproduced identically across two independent runs in this session.

## Security

No new attack surface: no I/O, no `unsafe`, no new dependencies, capability-check logic copied verbatim from the existing, already-reviewed `CapGraphIndex::search`. The connectivity-gap finding is a recall limitation affecting all callers equally (authorised nodes can go unreturned to everyone, not leaked to anyone) — not an authorization bypass.

## Governance

Purely additive change. No existing acceptance thresholds, default construction paths, or public API were modified.

## Failure modes

See the research report's "Failure modes considered (attack pass)" section: benchmark representativeness (synthetic uniform-random capability assignment vs. possible real-world geometric clustering by tenant), hardware-dependent absolute timings (relative orderings are large enough — 2–3x — to be robust), and no delete/concurrent-update path tested (none exists in this crate today).

## Migration

None required; nothing changes for existing callers.

## Rollback

N/A — no default behaviour changed. If `HierarchicalCapGraphIndex` or the γ-augmented configuration were ever later promoted to a default and needed rollback, reverting to `CapGraphIndex::new(dims, 12, entry_points)` (the current default) fully restores prior behaviour.

## Rejection criteria

Already applied in this ADR: both H1 and H2 were rejected because their measured recall deltas (`-0.0373` and `+0.0013`) fell short of their frozen thresholds (`+0.10` and `+0.05`) at the production-facing budget. See **What would have overturned this** in the research report for the specific counterfactuals that were checked and did not hold.

## Open questions

1. Does a budget-aware or wall-clock stopping rule close the low-`ef` gap found here without needing the 3–10x larger fixed node-count budget this run measured as necessary? (Named next experiment.)
2. Does the baseline's directed-graph connectivity ceiling (0.979 at full traversal) grow or shrink as corpus size scales beyond n=4,000? Untested.
3. Does real (non-uniform-random) capability-to-geometry correlation in production workloads make this entire selectivity concern moot, or does it only reduce its severity? Untested; named as a limitation.
