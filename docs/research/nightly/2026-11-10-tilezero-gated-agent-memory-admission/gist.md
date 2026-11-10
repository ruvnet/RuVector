# Wiring a content-inspecting admission gate into an agent memory ledger

## Problem

`ruvector-agent-memory`'s `TransactionalLedger` moves a candidate statement
from `Pending` to `Accepted` through a `ProofGate` trait. Both existing
implementations admit unconditionally with respect to content:
`AlwaysAdmitGate` has no check at all, and `WriteGateAdapter` only chains
receipts tamper-evidently — it never looks at the bytes being accepted.
Nothing in the admission path can refuse a statement because of what it
says.

Separately, a different crate in the same workspace,
`cognitum-gate-tilezero`, implements a fully tested three-filter admission
decision (`ThreeFilterDecision`: a structural min-cut check, a shift/drift
check, and an evidence-accumulation check, evaluated over a `ReducedGraph`
summary) for a 256-tile WASM fabric's central arbiter. That decision logic
had never been connected to any real content store.

## Hypothesis

Wiring `ThreeFilterDecision` behind a new `ProofGate` implementation,
built from a sliding window of recently-admitted content compared against
each new candidate by similarity, should refuse more synthetically
"incoherent" candidates than the existing gates (which refuse none by
construction) while still admitting "coherent" candidates at a comparable
rate — with the latency cost measured and reported honestly, not hidden
behind a favorable threshold choice.

## Design

`TileZeroGate` (`crates/ruvector-agent-memory/src/tilezero_gate.rs`):

```rust
pub trait ProofGate {
    fn admit(&mut self, entry_id: u64, content: &[u8]) -> Result<AcceptanceReceipt, LedgerError>;
}
```

`admit()` derives a fixed-64-dimension pseudo-embedding from the raw
content bytes using the feature-hashing trick (every overlapping 3-byte
n-gram hashes to a dimension and a sign; no vocabulary or model needed),
compares it (mean cosine similarity) against a sliding window of the last
64 admitted entries' embeddings, and maps that single scalar onto
`ReducedGraph`'s three fields via its own public setters before calling
`ThreeFilterDecision::evaluate`. A `Permit` admits and joins the window; a
`Deny` or `Defer` both become `Err` (the trait has no third outcome).

Two design choices are worth stating plainly rather than glossing over,
because they are the main limits on what this experiment actually shows:

1. **The embedding is byte-level, not semantic.** `ProofGate::admit`'s
   signature carries raw bytes, no embedding channel — changing it was out
   of scope. Two paraphrases of the same idea in different words will not
   score as similar under this scheme. The synthetic test workload below
   is built to be separable *by this specific transform*, so the result
   demonstrates the wiring and decision logic work on a workload
   engineered for them — not that a real semantic poisoning attempt would
   be caught at the same rate.
2. **All three filters collapse to one signal.** A real 256-tile
   deployment has independent per-tile structural, drift, and evidence
   signals. A single ledger admission has none of that, so this
   integration derives all three `ReducedGraph` fields from the same mean
   cosine similarity via three different scalings. The three
   `GateThresholds` fields are, here, three cut points on one curve, not
   three independent checks.

`cognitum-gate-tilezero`'s optional `mincut` feature (pinned to a stale
crates.io `ruvector-mincut = "0.1.30"` against the workspace's 2.3.1) was
deliberately left disabled — reading `supergraph.rs` confirms `ReducedGraph`
is a plain scalar struct with public setters, independent of any mincut
crate.

## Implementation

New, feature-gated (`tilezero-gate`, off by default) files in
`ruvector-agent-memory`: `src/tilezero_gate.rs` (the gate),
`tests/tilezero_gate_tests.rs` (8 tests: classification on deterministic
fixtures, a witness-before-mutation invariant test with `TileZeroGate`
installed, and feature-flag compilation sanity), and
`examples/tilezero_gate_bench.rs` (the benchmark and a bounded parameter
sweep). `cognitum-gate-tilezero` itself was read, not modified.

## Real benchmark evidence

500 trials each of "coherent" (templated sentences, randomized fill-ins —
heavy byte-trigram overlap by construction) and "incoherent" (random
printable-ASCII strings, no template structure) content, seed 42, release
build, measuring `accept()`'s own wall-clock latency (the call that
actually consults the gate):

```text
always_admit / coherent      permit= 500 refused=   0  mean=   0.548us  throughput=  710231.3 ops/s
always_admit / incoherent    permit= 500 refused=   0  mean=   0.508us  throughput=  781778.9 ops/s
tilezero_default / coherent  permit= 496 refused=   4  mean=   4.124us  throughput=  205303.4 ops/s
tilezero_default / incoherent permit=   0 refused= 500  mean=   4.859us  throughput=  177321.4 ops/s
```

With `cognitum-gate-tilezero`'s unmodified default `GateThresholds`:
incoherent Deny+Defer rate 100.0% (vs. `AlwaysAdmitGate`'s 0% by
construction — 100 percentage points, against a pre-registered 50pp bar),
coherent Permit rate 99.2% (against an 85% bar), at ~3.96us added latency
per admission. Full raw command output is committed alongside this
document (`benchmark_raw_output.txt`), so the numbers above are
independently checkable against the actual run.

## A bounded parameter sweep found a real trap, not just a tuning win

A 13-configuration ("Darwin-lite": default + 3 generations × 4 mutated
`GateThresholds` candidates) sweep, scored by a pre-registered fitness
formula (separation, latency, throughput, memory, correctness,
reproducibility), surfaced something not anticipated when the mapping
constants were chosen: `ReducedGraph::new()`'s cold-start default sets
`aggregate_evidence = 100.0`, and the evidence filter's Permit condition
is `e_value >= tau_permit`. The hand-picked default (`tau_permit = 100.0`)
sits exactly on that boundary — any mutated config with `tau_permit >
100.0` breaks the cold-start bypass, so the gate's sliding window never
fills from warm-up, and the gate then permanently Defers *everything*,
cheaply (an empty window skips the whole similarity scan, which is
*also* why these broken configs showed up to 3x lower latency than the
working default — exactly the kind of degenerate case a latency-weighted
score could reward if correctness weren't checked first). Eleven of
twelve mutated candidates fell into this trap by chance of the fixed
mutation seed; the correctness hard-gate was extended, after observing
this, to require the gate's own warm-up to actually succeed, before
computing the promotion decision. One surviving candidate
(`tau_permit≈92.0`, avoiding the trap) modestly beat the default on
fitness (0.6151 vs 0.5988: 100.0% vs 99.2% coherent permit, at the cost of
a 0.4% incoherent leak vs. 0%) and is recommended, not shipped as the
compiled-in default.

## Verification

All pre-existing `ruvector-agent-memory` tests remain green with the
feature on (92 tests total) and the feature correctly compiles out to
zero additional tests and identical pre-existing test counts when off (48
tests). A dedicated test
(`tilezero_gate_failing_sink_no_witness_no_mutation`) forces the witness
sink to fail after `TileZeroGate` has already Permitted a candidate, and
confirms the ledger applies no state mutation, appends no history record,
and persists no witness record — the pre-existing "no witness, no
mutation" invariant holds unchanged with the new gate installed. The
test does document a known, pre-existing asymmetry (also present for the
existing `WriteGateAdapter`): the gate itself is consulted, and its own
internal state (the sliding window) advances, before the witness sink is
asked anything — a documented ordering, not a new bug. `cargo fmt`,
`cargo build --release`, and `cargo clippy --all-targets` are clean for
every command run in this pass, with zero warnings attributable to any
file this work touched.

## Limitations

The embedding is byte-level, not semantic — see Design. The three filters
collapse to one signal in this integration. The workload is synthetic
and non-adversarial (a hashing-scheme-aware attacker was not modeled).
No concurrent-writer or fault-injection scenario beyond the one
pre-registered sink-failure test was measured. The 12-candidate parameter
sweep is a small sample of a 6-dimensional continuous space — it found one
real trap and one modest improvement, not a proof of global optimality.

## Production relevance

The gate is additive and feature-flagged off by default: zero behavior
change for any existing deployment, zero existing test regression, and a
documented, bounded way to opt in where even an imperfect content-level
filter is better than none. It is not a drop-in replacement for a real
semantic-embedding-based review step, and should be described to any
downstream user accordingly.

## Ecosystem implications

This is the first real consumer of `cognitum-gate-tilezero`'s decision
logic outside its native 256-tile fabric, and the first content-inspecting
`ProofGate` in `ruvector-agent-memory`. It composes with, and does not
replace, the existing `WriteGateAdapter`/tamper-evidence lineage and the
witness-signing lineage from an earlier nightly run — different gates
address different properties (content review vs. receipt-chain tamper
evidence vs. provenance signing), and a real deployment would likely want
more than one layered together.

## Future direction

Replacing the byte-trigram embedding with a real semantic one, building a
genuinely multi-signal (not single-scalar-collapsed) version using
synthetic multi-tile reports, and a systematic (not 12-sample) fuzz of
`GateThresholds` space for further degenerate configurations are the most
promising next steps — see the companion nightly research document's
"Next Research" section for the full list.

## References

- `docs/adr/ADR-357-tilezero-gated-agent-memory-admission.md`
- `docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/README.md`
  (the full research report this gist summarizes)
- `crates/ruvector-agent-memory/src/ledger.rs`, `src/tilezero_gate.rs`
- `crates/cognitum-gate-tilezero/src/decision.rs`, `src/supergraph.rs`
- Weinberger, K. et al. (2009), "Feature Hashing for Large Scale
  Multitask Learning" (the general technique the embedding applies)
