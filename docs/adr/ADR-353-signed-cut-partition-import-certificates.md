# ADR-353: Signed Cut-Partition Import Certificates

## Status

Proposed. New, opt-in, standalone crate `ruvector-partition-import`. No
existing crate's default behavior changes.

## Context

Several existing RuVector capabilities compute a deterministic
minimum-cut partition over an agent-memory graph for structural decisions
— namespace merge, graph-condense, mincut-gated forgetting
(`ADR-345`/`2026-09-05-mincut-gated-forgetting`), deterministic partition
selection (`2026-09-11-mincut-partition-determinism`). Separately,
`ruvector-agent-memory::witness_signing` (`ADR-347`) and
`ruvector-retrieval-receipt::signing` establish an Ed25519
signed-statement convention for provenance within a single agent's trust
domain.

Neither lineage addresses what happens when a *partition itself* needs to
cross from one agent's trust domain to another's — the scenario this ADR
calls "agent A has already computed a partition and wants agent B to use
it without B blindly trusting it or B paying the full recomputation cost
again." This is a concrete, recurring shape in federated/edge agent-memory
sync, RVF cross-node import, and ruFlo cross-agent workflows (all cited as
analytical/future integration points below, none implemented here).

## Hypothesis

```text
Given a weighted graph G and a correct global-minimum-cut partition P of
G with cut value C (computed once by a trusted baseline algorithm),

when agent A signs a certificate over
digest(canonical_encoding(G), canonical_encoding(P), C) with Ed25519 and
sends (G, P, certificate) to agent B,

then B's import latency (verify + accept) should be substantially lower
than B's baseline import latency (recompute the global mincut from
scratch on G),

subject to: (a) zero false negatives on tamper-in-transit detection;
(b) the accepted cut value must exactly equal the baseline's recomputed
cut value when the certificate is honest; (c) build green, tests green.
```

A second, equally important finding this ADR documents: a signature alone
cannot distinguish an honest partition from a validly-signed but
genuinely wrong one. This is not a flaw in the design to be fixed; it is
a structural property of what a signature proves (authenticity of the
message) versus what it does not prove (correctness of the message's
claim). The decision below is made with this limitation explicit, not
despite it.

## Decision

Add a new, standalone crate, `ruvector-partition-import`, implementing:

1. A deterministic synthetic weighted-graph generator and canonical
   (hash-stable) encodings of graphs and partitions
   (`src/graph.rs`).
2. A self-contained sparse Stoer-Wagner global minimum-cut reference
   algorithm (`src/baseline.rs`) as both the "recompute from scratch"
   comparison point and the correctness ground truth. **Not** a path
   dependency on `ruvector-mincut` — see "Alternatives" below.
3. Ed25519 cut certificates (`src/certificate.rs`) binding
   `(digest(G), digest(P), cut_value)`, following the
   `MSG_TAG || ... || digest` domain-separated convention already used by
   `ruvector-agent-memory::witness_signing` and
   `ruvector-retrieval-receipt::signing`, for workspace consistency
   (no dependency edge to either crate).
4. Two import strategies (`src/candidates.rs`):
   - `candidate_a` ("trust-the-signature"): accept the partition
     unconditionally once the signature verifies. Fast; structurally
     cannot detect a signed-but-wrong partition.
   - `candidate_b` ("verify + spot-check"): additionally sample a
     fraction `k` of vertices and check a necessary (not sufficient)
     local-optimality condition for global minimum-cut optimality at
     each sampled vertex, `O(degree(v))` per vertex. Rejects and falls
     back to recomputation if any sampled vertex fails.
5. A benchmark binary (`src/bin/benchmark.rs`) measuring real latency,
   payload size, tamper-detection rate, and signed-but-suboptimal
   detection rate across `N in {500, 2000, 8000, 20000}`.

This is promoted as an available, opt-in capability for any RuVector
component that needs to move a partition across a trust boundary — not
wired into any existing crate's default call path.

## Evidence

From one real `cargo run --release -p ruvector-partition-import --bin
benchmark` execution (`Linux 6.18.44-fc-v140`, 4 vCPUs, `rustc 1.97.0`,
seed `20261022`, signer seed `0x42`); see
`docs/research/nightly/2026-10-22-verified-partition-import/README.md`
for the full raw output and methodology:

| N | baseline mean | candidate_a mean | candidate_a speedup | candidate_b(k=0.5) mean |
|---|---|---|---|---|
| 500 | 54.451 ms | 125.4 us | 434x | 188.1 us |
| 2000 | 1003.9 ms | 473.2 us | 2121x | 720.0 us |
| 8000 | 21,923.5 ms | 1302.4 us | 16,833x | 3174.1 us |
| 20000 | 156,929.3 ms | 4800.8 us | 32,688x | 12,663.2 us |

**Acceptance gate (1)** (candidate_a mean <= 25% of baseline mean, at
N=20000): measured ratio is 0.0031% — PASS by roughly three orders of
magnitude beyond the required bar. **ACCEPT** on all four acceptance
gates (see the research report's Acceptance Result table for the full
breakdown).

Tamper-in-transit detection: 100% for both candidates at every tested `N`.
Correctness: the accepted cut value from both candidates exactly matched
baseline's recomputed cut value in every honest-signer trial at every
`N`. Signed-but-suboptimal detection for `candidate_b` (2% of vertices
perturbed): see the full curve in the research report — `candidate_a`
wrongly accepted the suboptimal partition in 100% of adversarial trials
at every `N`, exactly as the structural argument above predicts.

## Consequences

**Positive:**

- A measured, real, large speedup (two to four orders of magnitude at the
  tested sizes — see evidence table) for the honest-signer case over
  always recomputing.
- `candidate_a`'s structural limitation is now documented with a measured
  number (100% wrong-acceptance in the adversarial test) rather than
  asserted abstractly, making it harder for a future integrator to pick
  `candidate_a` by default without understanding the tradeoff.
- `candidate_b` gives a tunable (via sampling fraction `k`) partial
  defense with a real measured detection curve, letting a deployment
  choose a cost/detection tradeoff deliberately.
- The certificate's fixed-size signed message (a digest triple, not the
  raw graph/partition) keeps sign/verify cost `O(1)` in graph size,
  consistent with the workspace's existing signing conventions.

**Negative / costs:**

- A second, independent global-mincut implementation now exists in the
  workspace (alongside `ruvector-mincut`'s own algorithms), which is a
  real maintenance surface, explicitly accepted as the cost of isolating
  this run's hypothesis from a larger integration effort (see
  Alternatives).
- `candidate_b`'s spot-check gives no formal guarantee; a deployment that
  treats "passed the spot-check" as "proven optimal" is misusing it. This
  must be stated clearly in any downstream documentation that references
  this capability.
- No WASM, `no_std`, or non-container-hardware measurement exists yet.

## Alternatives

- **Integrate with `ruvector-mincut`'s existing `certificate`/`canonical`
  modules directly**, rather than writing a self-contained reference
  algorithm. Rejected for this pass: reading both modules (523 + 1244
  lines) showed a dynamic-min-cut-maintenance and
  canonical-tie-breaking-focused API, a materially larger integration
  than the available 30-45 minute exploration budget could responsibly
  absorb while still producing real, honest benchmark evidence this same
  pass. Flagged as a legitimate follow-up (see Open Questions) once a
  larger time budget is available — this is a statement about this run's
  constraints, not a judgment on `ruvector-mincut`'s suitability.
- **Dense `O(V^3)` adjacency-matrix Stoer-Wagner.** Infeasible at
  `N=20000` (3.2 GB `f64` matrix); rejected for sparse
  `HashMap`-adjacency with a lazy-deletion binary heap instead.
- **Sign the raw graph/partition bytes** instead of a digest triple.
  Rejected: would make signing cost scale with graph size, defeating the
  near-`O(E)`-import latency goal.
- **No spot-check at all (ship only `candidate_a`).** Rejected: this
  ADR's whole point is to make the signed-but-wrong tradeoff visible and
  give at least a partial, measured mitigation (`candidate_b`), not to
  ship a capability whose only failure mode is silent.

## Implementation Plan

1. Land `ruvector-partition-import` as a standalone crate (this ADR).
2. No existing crate's call sites change. Adoption is opt-in per
   integrator.
3. Future: wire into a concrete cross-agent transfer path (ruFlo task,
   RVF export/import, or a federated agent-memory sync job) once one of
   those has a concrete need — not speculatively built here.
4. Future: attempt the `ruvector-mincut` integration with a larger time
   budget, informed by this run's correctness-test scaffolding.

## API Shape

```rust
// graph.rs
pub fn generate_graph(n: usize, seed: u64, avg_degree: usize) -> WeightedGraph;
pub struct WeightedGraph { pub n: usize, pub edges: Vec<(u32, u32, f64)> }
pub struct Partition { pub side: Vec<bool> }

// baseline.rs
pub fn global_min_cut(graph: &WeightedGraph) -> MinCutResult;
pub struct MinCutResult { pub cut_value: f64, pub partition: Partition }

// certificate.rs
pub struct Signer { /* Ed25519 keypair */ }
impl Signer {
    pub fn sign(&self, graph: &WeightedGraph, partition: &Partition, cut_value: f64) -> CutCertificate;
}
pub fn verify_against(vk: &VerifyingKey, graph: &WeightedGraph, partition: &Partition, cert: &CutCertificate) -> bool;

// candidates.rs
pub fn candidate_a(graph: &WeightedGraph, partition: &Partition, cert: &CutCertificate, vk: &VerifyingKey) -> ImportDecision;
pub fn candidate_b(graph: &WeightedGraph, partition: &Partition, cert: &CutCertificate, vk: &VerifyingKey, sample_fraction: f64, sample_seed: u64) -> ImportDecision;
pub struct ImportDecision { pub outcome: ImportOutcome, pub cut_value: f64, pub partition: Partition, pub recomputed: bool }
```

## Feature Flags

None. The crate has no Cargo feature gates; all functionality is always
compiled in.

## Benchmark Evidence

See "Evidence" above and the full raw output in the nightly research
report's Benchmark Results section. Reproduce with:

```bash
cargo test -p ruvector-partition-import
cargo run --release -p ruvector-partition-import --bin benchmark
```

## Security

- Standard Ed25519 (`ed25519-dalek`) + SHA-256 (`sha2`); no novel
  cryptographic construction.
- `verify_against` recomputes both digests from the actually-received
  graph/partition before checking the signature — using
  `verify_signature` alone, without digest recomputation, would be
  insufficient (an attacker could replay an old, honestly-signed
  certificate against different data).
- `candidate_a` must be documented, anywhere it is offered as a choice,
  as providing authenticity only, never correctness — this ADR's own
  measured evidence (100% wrong-acceptance under adversarial test) is the
  citation for that warning.
- Key management (generation, rotation, storage, revocation) is out of
  scope, matching `ruvector-retrieval-receipt::signing`'s and
  `ruvector-agent-memory::witness_signing`'s existing disclaimers.

## Governance

None beyond standard code review for a new, opt-in crate. No existing
invariant changes.

## Failure Modes

See the research report's Failure Modes section (singleton-side spot-check
exclusion, sample-seed predictability, `candidate_a`'s unconditional
trust, Stoer-Wagner's own scaling cost). Summarized: `candidate_a` is
appropriate only when A is fully trusted for correctness, not merely
authenticated; `candidate_b` narrows but does not close that gap.

## Migration

None — new, standalone, opt-in crate with no existing call sites to
migrate.

## Rollback

Deleting the crate and its workspace-member entry fully reverts this
change; nothing else in the workspace depends on it.

## Rejection Criteria

This capability should be considered for removal or non-promotion if:

- A future, more rigorous integration with `ruvector-mincut` shows this
  self-contained Stoer-Wagner reference is materially slower or less
  correct than reusing the existing crate, making the standalone
  implementation pure maintenance burden.
- No real integrator (ruFlo, RVF, federated-memory sync) adopts this
  capability within a reasonable follow-up window, suggesting the
  motivating scenario was speculative rather than load-bearing.
- A stronger verification scheme (e.g. an actual cut-optimality proof,
  not a sampled necessary condition) becomes available at comparable
  cost, obsoleting `candidate_b`'s spot-check approach.

## Open Questions

1. Should `candidate_b`'s sampling fraction be a fixed per-deployment
   policy, or should it be dynamically chosen based on the declared trust
   level of the signer?
2. Is there a cheaper *sufficient* (not just necessary) condition for
   partition optimality that could be spot-checked, strengthening
   `candidate_b` without paying full recomputation cost?
3. Should `sample_seed` be required to come from the verifier's own
   randomness (generated after receiving the certificate) rather than
   accepted as a caller-supplied parameter, to close the
   sample-seed-predictability failure mode by API design rather than by
   caller discipline?
4. Is a real `ruvector-mincut` integration worth a dedicated follow-up
   pass with a larger time budget, and if so, which of its canonical
   partition-selection semantics would this crate's correctness
   comparisons need to account for?

## References

- `docs/research/nightly/2026-10-22-verified-partition-import/README.md` —
  full research report, methodology, and raw benchmark output.
- `docs/research/nightly/2026-10-22-verified-partition-import/gist.md` —
  standalone technical article version.
- `docs/adr/ADR-347-witness-signer-tarl-ledger.md` — the signing
  convention this ADR's certificate mirrors.
- `docs/research/nightly/2026-09-11-mincut-partition-determinism/README.md`,
  `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md` —
  the mincut-over-agent-memory lineage this ADR's motivating scenario
  connects to.
- `crates/ruvector-mincut/src/certificate/mod.rs`,
  `crates/ruvector-mincut/src/canonical/mod.rs` — read in full before
  deciding against integration (see Alternatives).
