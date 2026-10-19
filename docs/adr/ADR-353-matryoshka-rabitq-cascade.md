# ADR-353: Matryoshka-RaBitQ Cascade — RaBitQ Coarse Stage for Matryoshka Funnels

## Status

Proposed. New, additive, opt-in crate (`ruvector-matryoshka-rabitq`); no
existing crate's public API or default behaviour changes. Acceptance is
scope-bound (see Hypothesis and Rejection criteria below) — not a claim of
universal superiority over the existing `TwoStageIndex`.

## Context

`ruvector-matryoshka` (ADR-264, 2026-06-21 nightly) implemented a
coarse-to-fine funnel over Matryoshka-structured embeddings: a cheap HNSW
search at a truncated prefix dimension generates candidates, an exact
rerank at full dimension picks the final top-k. Its coarse stage is an HNSW
graph, which this repository has twice found to carry real maintenance
cost independent of raw speed: non-deterministic partition results tied to
`DashMap` iteration order (ADR-346) and delete-repair complexity
(2026-06-18 `hnsw-delete-repair` nightly).

`ruvector-rabitq` (2026-04-23 nightly) implemented RaBitQ (Gao & Long,
SIGMOD 2024, arXiv:2405.12497): a random-rotation 1-bit quantization scheme
with a provable error bound, but found that a 1-bit scan alone is too lossy
to serve as a *final* ranking (12-40% recall@10 without rerank).

No prior work in this repository, nor any found in a SOTA search (RaBitQ
itself; AdANNS, arXiv:2305.19435; Matryoshka Hash Representations, alphaXiv
2609.07276; Matryoshka Quantization, arXiv:2502.06786 — ruled out as a
different problem, model-weight quantization), combines the two: using a
RaBitQ-style binary code specifically as a Matryoshka funnel's coarse
*candidate-generation* stage, where its recall weakness as a final answer
should not matter because an exact full-dim rerank always follows it.

## Hypothesis

```text
Given a Matryoshka-structured dataset of N vectors at full dimension D=128
(coarse_dim=32), generated deterministically from a fixed seed,

when the funnel's coarse candidate-generation stage is implemented as a
RaBitQ-style 1-bit random-rotation XNOR-popcount brute-force scan at
coarse_dim instead of an HNSW graph at coarse_dim (both followed by the same
exact full-dim rerank over the same candidate-pool size),

then end-to-end recall@10 should be within 0.02 of the HNSW-coarse funnel's
recall@10, coarse-stage memory should be at most 50% of the HNSW-coarse
stage's memory, and mean query latency should stay within 2x of the
HNSW-coarse funnel's mean latency,

subject to: both the full-dim baseline and the HNSW-coarse funnel first
clearing independent sanity recall floors (>= 0.80 and >= 0.70
respectively), so a degraded baseline cannot manufacture a false pass.
```

## Decision

1. Add a new, standalone, opt-in crate `crates/ruvector-matryoshka-rabitq`,
   registered in the root `Cargo.toml` workspace `members` list immediately
   after `crates/ruvector-matryoshka`.
2. The crate depends on `ruvector-matryoshka` (path dependency) for its
   `Searcher` trait, `MatryoshkaConfig`, dataset generator, and the
   `FullDimIndex`/`TwoStageIndex` baseline/candidate-A implementations —
   reused unmodified, not duplicated.
3. Add `RaBitQCascadeIndex`, a new `Searcher` implementation using a
   self-contained, minimal reimplementation of RaBitQ's rotation + 1-bit
   encode + symmetric angular estimator (`src/rabitq_code.rs`), scoped to
   exactly what a coarse-ranking signal needs — not a dependency on the
   larger `ruvector-rabitq` crate's asymmetric estimator, Hadamard rotation,
   or persistence layer, none of which this use case needs.
4. No change to `ruvector-matryoshka`, `ruvector-rabitq`, or any other
   existing crate. No feature flags needed since this is a new, independent
   crate rather than a modification.

## Evidence

Full methodology, raw console output for every run, and the memory/latency
closed-form derivations are in
`docs/research/nightly/2026-10-19_matryoshka-rabitq-cascade/README.md`.
Summary, from real `cargo run --release` executions (Linux x86_64, 4 vCPU,
`rustc 1.97.0`):

At the declared configuration (N=3,000, D=128, coarse_dim=32, k=10, 200
queries, seed=42, reproduced at seed=7):

| Variant | Recall@10 | Mean latency (us) | Coarse-stage memory (KB) |
|---|---|---|---|
| FullDimHNSW (baseline) | 1.000 | 174.3 | 1,875.0 |
| TwoStage (candidate A) | 0.903 | 82.0 | 750.0 |
| RaBitQCascade (candidate B) | 0.994 | 92.6 | 27.4 |

All three pre-declared acceptance margins passed (recall within 0.02,
memory at 3.7% of the 50% bound, latency at 113% of the 82.0us baseline,
within the 2x bound) — **ACCEPT** at this configuration, reproduced at a
second seed.

A separate scale probe at N=20,000 showed the hypothesis's latency margin
fails (candidate B at 1,120us vs. candidate A's 150.6us, ~7.4x) — but the
run is recorded as **INCONCLUSIVE**, not REJECT, because the pre-existing
HNSW baseline and candidate A's own recall floors failed first at that N
under this crate's inherited default HNSW parameters, a limitation of the
sibling crate's defaults, not of this ADR's change. See the research
README's Failure modes section for the full, undiluted account, including
the fact that the exact latency crossover N between 3,000 and 20,000 was
not bisected.

7 unit tests pass; `cargo clippy --release -p ruvector-matryoshka-rabitq --
-D warnings` is clean.

## Consequences

**Positive:**
- A measurably smaller (27x at the tested scale), construction-cheaper
  (142ms vs. 392-912ms build at N=3,000) coarse-stage option exists for
  memory-constrained or frequently-rebuilt agent-memory deployments.
- No graph to repair on delete (architectural argument; not benchmarked this
  round — see Open questions).
- Purely additive; zero risk to existing crates' behaviour.

**Negative / risk:**
- `O(N)` brute-force coarse scan does not scale indefinitely; above some
  unbisected N it loses to graph traversal on latency. Deploying this
  without first confirming the target N sits below that crossover is a real
  production risk.
- Two implementations of RaBitQ's rotation/encoding now exist in this
  workspace (`ruvector-rabitq`'s full version and this crate's minimal
  subset) with no shared code — accepted as the simpler, lower-coupling
  choice for this scope, but is latent duplication a future maintainer
  should be aware of if the minimal version ever needs the asymmetric
  estimator too.

## Alternatives considered

See the nightly research README's full candidate table (12 candidates,
novelty-gated). Most directly relevant alternative: use `ruvector-rabitq`
as a path dependency instead of reimplementing the rotation/encode subset.
Rejected for this scope because `ruvector-rabitq`'s `BinaryCode`/
`RandomRotation` carry a persistence layer (`serde`), an asymmetric
estimator, and a Hadamard-rotation variant this cascade does not need,
and pulling the whole crate as a dependency for ~15% of its surface was
judged higher coupling cost than a ~160-line, independently-tested,
self-contained reimplementation of just the needed math.

## Implementation plan

Already implemented in full as part of this ADR (not a plan for future
work): crate structure, `Searcher` impl, benchmark binary, unit tests,
`Cargo.toml` workspace registration. Follow-up work (bisecting the
crossover N, fixing the sibling crate's large-N HNSW defaults, a hybrid
scan/graph `Searcher`, a verified `wasm32` build) is tracked in the research
README's "Next research" section, not in this ADR's scope.

## API shape

```rust
pub struct RaBitQCascadeIndex { /* ... */ }

impl ruvector_matryoshka::Searcher for RaBitQCascadeIndex {
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self;
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize>;
    fn name(&self) -> &'static str;
}

impl RaBitQCascadeIndex {
    pub fn build_with_seed(config: &MatryoshkaConfig, vectors: &[Vec<f32>], seed: u64) -> Self;
    pub fn coarse_stage_bytes(&self) -> usize;
}
```

No changes to the existing `Searcher` trait or `MatryoshkaConfig` struct.

## Feature flags

None. The crate is opt-in by virtue of being a separate crate a consumer
must explicitly depend on; no existing crate's `Cargo.toml` or default
feature set changes.

## Benchmark evidence

See Evidence section above and the full research README for raw console
output of all three runs (two seeds at N=3,000, one scale probe at
N=20,000) plus the bounded candidate-pool sweep.

## Security

No new attack surface: no network I/O, no untrusted deserialization, no
filesystem access beyond what Cargo itself performs. The random rotation is
a decorrelation tool, not an access-control or encryption mechanism, and
must not be mistaken for one (unlike `ruvector-capgated`'s bitset capability
tokens, which this crate does not interact with).

## Governance

Purely additive; no data retention, deletion, or privacy implications. All
benchmark data is synthetic and seed-generated.

## Failure modes

See the research README's dedicated section: (1) the measured `O(N)` vs.
sub-linear-graph latency crossover, un-bisected, between N=3,000 and
N=20,000; (2) the sibling crate's own pre-existing HNSW recall collapse at
N=20,000 under its inherited default parameters, out of this ADR's scope to
fix; (3) delete/update behaviour under concurrent mutation, architecturally
argued but not benchmarked.

## Migration

None required — new crate, no consumer is forced to adopt it. A future
migration of `ruvector-agent-memory` or similar to offer this as a
configurable backend would be a separate, explicit decision and ADR.

## Rollback

Trivial: remove the crate from the workspace `members` list and delete
`crates/ruvector-matryoshka-rabitq/`. No other crate depends on it.

## Rejection criteria

This ADR's "Accepted" status (once promoted past Proposed) should be
revisited if: (a) the crossover N, once bisected, turns out to sit below
any realistic production N for the crates this was motivated by
(`ruvector-agent-memory`, `ruvector-edge-*`), making the memory win
practically unusable at relevant scale; or (b) a hybrid or alternative
design (per Next research) strictly dominates this one on all three
measured axes.

## Open questions

1. What is the exact latency crossover N? (Not bisected this run.)
2. Does the architectural delete-simplicity argument hold under an actual
   concurrent-mutation benchmark?
3. Does this port cleanly to `wasm32-unknown-unknown`? (Plausible given its
   dependency footprint; not verified.)
4. Should `ruvector-matryoshka-rabitq` eventually depend on `ruvector-rabitq`
   directly once/if that crate's `BinaryCode` is refactored to expose a
   lighter-weight symmetric-only mode, removing the current duplication?
