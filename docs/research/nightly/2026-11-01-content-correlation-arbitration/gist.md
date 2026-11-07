# Closing a documented blind spot in multi-agent memory arbitration

## Problem

Agent-memory systems that retrieve multiple memories for a decision often
want to know: how many *independent* sources actually support this claim?
RuVector's `ruvector-agent-memory` crate ships a correlation-aware
arbitration module (`arbitration.rs`, informed by the CAMA paper,
arXiv:2608.19701) that answers this by clustering retrieved memories into
evidence lineages based on **declared causal ancestry** — if ten memories
all trace back to one source observation, they count as one lineage, not
ten.

That module's own documentation is unusually candid about what it
*doesn't* catch: a memory that copies another memory's content, but
declares no parent, looks exactly like a fresh, independent source. Twenty
agents re-asserting one unsourced rumor arbitrate to twenty "independent"
lineages. The module's docs name the fix: pair it with content-level
near-duplicate detection. Nobody had built that pairing.

## Hypothesis

> Given a retrieved memory set containing an astroturf cluster of N
> parentless memories whose content is near-identical, with no declared
> causal parent, when content-similarity near-duplicate detection is
> layered on top of declared-ancestry arbitration, then the astroturf
> cluster's lineage count drops by at least 70% relative to
> declared-ancestry-only arbitration, while a disjoint set of genuinely
> independent memories sees 0% false merges — subject to the added pass
> costing under 50ms at 500 memories and the existing downgrade-only score
> invariant holding throughout.

## Technical design

A new, purely additive module, `content_correlation.rs`, that:

1. Calls the existing `arbitrate()` as an opaque first pass (zero
   modifications to the carefully invariant-checked arbitration code).
2. Takes a caller-supplied embedding per memory (deliberately not sourced
   from the memory type itself, which is content-addressed over an opaque
   byte payload — bridging content to a vector is the integrator's job).
3. Computes pairwise cosine similarity across the declared lineages'
   members and offers two independent strategies:
   - **HardMerge**: union near-duplicate lineages via union-find.
   - **SoftDiscount**: keep lineages separate, multiply a flagged
     lineage's weight by `1 - strength`.
4. Re-derives the sufficiency outcome from the transformed verdict.

Both strategies are provably downgrade-only: a merged lineage's weight is
the max over a strictly larger member set (never more than either
constituent), and a discount factor is bounded in `[0,1]`. The
implementation re-checks this at runtime and errors loudly rather than
clamping if it's ever violated — matching the existing module's posture.

## Actual implementation

- `crates/ruvector-agent-memory/src/content_correlation.rs` (~340 lines):
  `ContentCorrelationConfig`, `CorrelationMode`, `ContentCorrelationError`,
  `arbitrate_with_content_correlation`.
- `crates/ruvector-agent-memory/tests/content_correlation.rs`: 7
  integration tests, including one that reuses the *exact* astroturf
  fixture already locked into the arbitration module's own test suite, for
  a direct apples-to-apples comparison.
- `crates/ruvector-agent-memory/examples/content_correlation_bench.rs`: the
  benchmark below.
- No changes to any existing file's logic; no new dependencies.

## Actual benchmark evidence

Run: `cargo run --release -p ruvector-agent-memory --example
content_correlation_bench` (rustc 1.97.0, 4-core x86_64, Linux 6.18.44).

```
--- Scenario 1: astroturf cluster (N=20, no declared parent) ---
baseline (declared-only):  lineages=20   naive=18.0000 arbitrated=16.2000
candidate A (HardMerge):   lineages=1    naive=18.0000 arbitrated=0.8100  (lineage reduction: 95.0%)
candidate B (SoftDiscount):lineages=20   naive=18.0000 arbitrated=8.1000
ACCEPTANCE (astroturf lineage reduction >= 70%): PASS

--- Scenario 2: independent evidence (N=20, orthogonal content) ---
baseline lineages=20 -> candidate A lineages=20 (false merges: 0, rate: 0.0%)
ACCEPTANCE (false-merge rate == 0%): PASS

--- Scenario 3: mixed (astroturf cluster + independent witnesses, same call) ---
baseline lineages=20 -> candidate A lineages=11 (expect ~11: 1 merged astroturf lineage + 10 independent)
ACCEPTANCE (mixed scenario collapses to exactly 11 lineages): PASS

--- Scenario 4: latency scaling (HardMerge, worst case L≈N, 20 reps/size) ---
N           mean(ms)     p50(ms)     p95(ms)
20             0.054       0.053       0.062
50             0.205       0.200       0.231
100            0.669       0.671       0.793
200            2.490       2.306       3.325
500           15.029      14.111      20.715
ACCEPTANCE (p95 latency at N=500 < 50ms): PASS (measured: 20.715ms)

=== Overall acceptance: ACCEPT ===
```

`cargo test -p ruvector-agent-memory --lib --tests`: 84/84 passed, zero
regressions to the crate's 77 pre-existing tests. `cargo clippy
--all-targets`: zero warnings attributable to this change.

## Limitations

This is deliberately narrow and says so in its own docs: it catches
near-verbatim content re-assertion in embedding space, not paraphrase or
translation; it was validated against one similarity threshold (0.95) and
two synthetic fixture shapes (tight-noise-ball astroturf, orthogonal
independent), not a sweep over real topically-related-but-independent
content; and it costs `O(N²)` in the worst case (validated to N=500, not
beyond). None of this is hidden in the write-up — the point of a nightly
research run is honest evidence, including about where the evidence stops.

## Production relevance

The module is already exported as public API (`lib.rs`), opt-in (no
default-on behavior change — `arbitrate()` itself is untouched), and
zero-regression. An integrator adopts it by wiring `embeddings` from
whatever vector index already holds the memory's embedding and picking
`HardMerge` for a hard corroboration-count signal or `SoftDiscount` for a
softer one.

## RuVector ecosystem implications

This is the kind of contribution that's easy to miss in a mature research
codebase: not a new algorithm, but closing a gap an existing, well-tested
module already named explicitly in its own documentation. It bridges two
subsystems — vector/embedding similarity and causal-fusion provenance —
that this crate already has in one place but had not yet connected for
this purpose.

## Future direction

An ANN/LSH-prefiltered near-duplicate step to push past the quadratic
cost ceiling (likely composing with `ruvector-mincut`'s existing k-NN
graph construction); a threshold-sensitivity sweep on real embeddings; and,
longer-term, a learned/evolved threshold via this repository's existing
Darwin machinery once a labeled correlated/independent dataset exists.

## References

- ADR-330 (this repository): CAMA-pattern correlation-aware memory
  arbitration.
- CAMA, "Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent
  Memory Arbitration" (arXiv:2608.19701).
- ADR-320 (this repository): cross-source causal fusion layer.
- Full write-up: `docs/research/nightly/2026-11-01-content-correlation-arbitration/README.md`
- ADR: `docs/adr/ADR-350-content-correlation-augmented-arbitration.md`
