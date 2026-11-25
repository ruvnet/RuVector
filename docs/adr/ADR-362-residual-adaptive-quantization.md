# ADR-362: Sheaf-Residual-Gated Quantization Bit-Width for Agent Memory

## Status

Rejected (for production use as designed). Experimental crate
`ruvector-residual-quant` retained as evidence and reference
implementation, not promoted, not wired into `ruvector-agent-memory`'s
compaction path. The crate builds, its 15 unit tests pass, and the
benchmark binary is a reusable harness for any future attempt at
quantization bit-allocation signals (see Open Questions).

## Context

`ruvector-agent-memory` stores every memory at full `f32` precision. Two
prior nightly experiments tried to make per-vector quantization
bit-width *adaptive* to a structural graph signal, so that "important"
memories keep precision and "redundant" memories compress harder:

- **ADR-305** (2026-08-15): gated on mutual-kNN mutuality. The adaptive
  candidate recovered only 14% of the uniform-4-bit recall gap while
  spending 75% of the baseline's memory — rejected for a budget mismatch
  (not memory-matched against baseline).
- **ADR-353** (2026-10-24): gated on `ruvector-mincut`'s dynamic-mincut
  conductance. Recall@10 collapsed 0.657 -> 0.301 — rejected because the
  mincut gate only fires on a true between-community boundary, and a
  unimodal cluster has none, so `cut_cost` degenerated to ~0 everywhere
  (signal collapse).

Both used graph/neighborhood-density proxies, not the sheaf-Laplacian
"contradiction energy" formula of ADR-CE-001/ADR-CE-011
(`E(S) = sum_e w_e |r_e|^2`, `r_e = rho_u(x_u) - rho_v(x_v)`), which lives
in `crates/prime-radiant` (workspace-excluded, per-edge API only, no
per-node aggregation). ADR-353's own Open Questions section explicitly
flagged "a plain local-density score, decoupled from min-cut, untested"
as the natural next probe. This ADR runs both: the untested sheaf-residual
signal, and, as an honest control at the *same* matched budget, the
density probe ADR-353 never got to.

## Hypothesis

```text
Given a synthetic corpus of 1500 48-dim vectors in 6 Gaussian clusters,
with 8% of vectors planted as "boundary" memories (interpolated 50/50
between two randomly paired cluster centers, standing in for a
contradictory/surprising agent memory) and a k=10 cosine kNN consistency
graph,

when per-vector quantization bit-width is allocated by ranking vectors on
per-node sheaf-residual energy (shared-restriction-map simplification of
ADR-CE-001: r_e = A_e(x_u - x_v), A_e a random Gaussian projection per
edge, aggregated as the mean incident-edge energy) and giving the top 25%
8 bits / the bottom 75% 2 bits ("residual_gated"),

versus the identical two-bucket allocation driven by local-density /
outlier score instead ("density_gated"),

versus the identical two-bucket allocation with the hi-bit 25% chosen
uniformly at random ("random", the baseline) —

then residual_gated's recall@10 restricted to boundary-vector queries
should exceed random's by >= 0.05 absolute,

subject to: residual_gated's overall recall@10 not falling more than 0.02
below random's overall recall@10, all three policies spending an
identical total byte budget (verified every seed), and the witness hash
chain over every compression decision verifying intact.
```

Pre-registered before any run; not modified after seeing results.

## Decision

**Do not adopt** sheaf-residual-gated (or density-gated) adaptive bit
allocation for `ruvector-agent-memory` as designed. The boundary-subset
gate passed but the overall-recall regression gate failed, and a
diagnostic measurement made in advance of running the main experiment
(residual-energy AUC against the planted-boundary label) showed the
signal is statistically indistinguishable from chance at the one task it
would need to win to be interesting as a *distinct* mechanism from the
already-known density heuristic.

`ruvector-residual-quant` is kept in the workspace, building and tested,
as:

1. A clean, reusable matched-budget benchmark harness (dataset generator
   with ground-truth labels, kNN graph, scalar quantizer, AUC/Pearson
   diagnostics, hash-chained witness log) for the next attempt at this
   question.
2. Retained negative evidence: this is the third independent proxy signal
   (mutual-kNN, mincut, sheaf-residual) in this family to fail, each for
   a distinct, now-documented reason.

## Evidence

Benchmark: `cargo run --release -p ruvector-residual-quant --bin benchmark`,
5 seeds (1-5), n=1500, dim=48, 6 clusters, boundary_frac=0.08, k=10,
edge_dim=12, hi_bits=8, lo_bits=2, hi_fraction=0.25 (matched average
3.5 bits/dim for every policy), 250 queries/seed. Full raw stdout in
`docs/research/nightly/2026-11-25-residual-adaptive-quantization/raw-runs.txt`.

```text
Residual-energy AUC (separating planted boundary vectors): mean=0.467 std=0.072
Local-density AUC (same task):                              mean=0.529 std=0.089
Pearson corr(residual_score, density_score):                mean=0.865 std=0.013

Policy          Overall recall@10      Boundary recall@10    Bytes
random          0.240 ± 0.012          0.793 ± 0.049          43500
residual_gated  0.206 ± 0.008          0.887 ± 0.022          43500
density_gated   0.198 ± 0.010          0.906 ± 0.024          43500

Gate: residual_boundary - random_boundary >= 0.05  -> 0.094  PASS
Gate: residual_overall >= random_overall - 0.02    -> 0.206 >= 0.220  FAIL
Gate: matched byte budget every seed               PASS
Gate: witness chain verifies every policy/seed     PASS

ACCEPTANCE RESULT: REJECT
```

Interpretation, in the order the numbers actually support it:

1. **The signal does not predict its own target.** AUC 0.467 means
   ranking vectors by sheaf-residual energy is no better than a coin flip
   at telling planted-boundary memories apart from ordinary ones. This
   falsifies the sub-claim the whole design rests on.
2. **It is a noisy density proxy, exactly as flagged in advance.** The
   crate's own doc comment predicted that a shared-per-edge random
   projection would, by Johnson-Lindenstrauss concentration, make
   aggregate residual energy correlate with plain neighbor distance. The
   measured Pearson correlation (0.865) confirms it. Under this
   simplification, "sheaf-residual" is not a mechanistically distinct
   signal from "local density" — it is the same thing with extra
   machinery and extra variance (wider std on both AUC and recall).
3. **Both candidates still raised boundary recall — by the same
   mechanism, not by finding contradictions.** Boundary vectors sit
   between two clusters, so their true nearest neighbors are themselves
   atypical, low-local-density, cluster-edge vectors. Ranking by
   residual/density energy promotes exactly that kind of vector into the
   high-bit bucket, which happens to include many of a boundary query's
   real neighbors — not because the signal detected "this one is a
   boundary memory" (it didn't; AUC is chance), but because it detected
   "this one is geometrically atypical," which correlates with but is
   not identical to the planted label.
4. **That same mechanism is why overall recall fell.** The 75% majority
   of vectors are typical, dense, interior-cluster memories whose true
   neighbors are each other. Both gated policies systematically move the
   high-bit budget away from that majority toward the geometrically
   sparse minority, so the majority's own mutual recall — which
   dominates the *overall* average — degrades. `random`'s identical
   two-bucket structure without this systematic bias does not pay that
   cost, which is why it has the best overall recall of the three despite
   spending the same budget.
5. **Density-gated, not residual-gated, is the better of the two
   adaptive policies** on every measured axis (boundary recall 0.906 vs.
   0.887; overall recall 0.198 vs. 0.206 — both equally failing the
   overall-regression gate, but density's deficit is the simpler-to-reason
   -about standard outlier heuristic, with no new math and no
   projection-dimension hyperparameter). There is no case, on this
   evidence, for the sheaf-residual construction over the plain density
   control it was raced against.

## Consequences

- No change to `ruvector-agent-memory`'s storage model: memories remain
  full-precision `f32` by default.
- The ecosystem now has three independent, *matched-budget-tested*
  REJECTs for "structural/neighborhood signal gates quantization
  bit-width," each with a distinct, documented root cause. A fourth
  attempt in this family should either (a) use a genuinely different
  class of signal — one that is not reducible to neighbor distance by
  construction (see Open Questions), or (b) abandon the two-bucket
  hi/lo scheme itself, since §Evidence item 4 suggests the overall-recall
  cost may be inherent to any such scheme that reallocates from a large
  low-variance majority to a small high-variance minority, regardless of
  which signal selects the minority.
- `ruvector-residual-quant`'s witness log and matched-budget harness are
  directly reusable scaffolding for that next attempt, which is why the
  crate is kept rather than deleted.

## Alternatives Considered

- **Uniform quantization at the matched average bit-width (3.5
  bits/dim).** Not run as a fourth policy in this experiment (the
  pre-registered design compared three *selection* policies at one fixed
  two-bucket shape), but `random`'s result is a reasonable proxy: a true
  uniform-3.5-bit baseline would not show the majority-vs-minority
  variance trade-off at all, and is likely to beat all three two-bucket
  policies on overall recall while losing on boundary recall. Worth
  running explicitly in a follow-up (see Open Questions).
- **Learned/calibrated restriction maps** (ADR-CE-002's stated direction,
  rather than this ADR's random-per-edge simplification) could produce a
  residual signal that is not reducible to distance. Not attempted here:
  it requires a training loop and labeled or self-supervised
  consistency data, which is a materially larger undertaking than a
  one-night experiment, and should not be attempted until a shared
  projection is first ruled in or out (now ruled out).
- **Per-endpoint restriction maps** (`rho_u != rho_v`, the fully general
  ADR-CE-001 form) instead of the shared-per-edge simplification used
  here. Considered and rejected for this run on the grounds that
  per-node-independent random maps would inject node-identity noise
  unrelated to genuine disagreement (see crate doc comment); a principled
  per-endpoint map still requires structure (e.g. shared type/role
  embeddings), which again reduces to the "needs calibration" alternative
  above.

## Implementation Plan

Not applicable — rejected prior to any production integration step.

## API Shape

`ruvector-residual-quant` (research-only, not re-exported by
`ruvector-agent-memory`):

```rust
pub fn gen_dataset(cfg: &DatasetConfig) -> Vec<Entry>;
pub fn build_knn_graph(vectors: &[Vector], k: usize) -> Vec<Edge>;
pub fn node_residual_energies(n: usize, edges: &[Edge], vectors: &[Vector],
    seed: u64, edge_dim: usize, dim: usize) -> Vec<f32>;
pub fn node_density_outlier_scores(n: usize, edges: &[Edge], vectors: &[Vector]) -> Vec<f32>;
pub fn quantize_vector(v: &[f32], bits: u8) -> QuantizedVector;
pub fn allocate_bits(scores: &[f32], hi_bits: u8, lo_bits: u8,
    hi_fraction: f32, policy: Policy, seed: u64) -> Vec<u8>;
pub struct WitnessLog { /* append(), verify() */ }
```

If a future attempt is promoted, the integration point would be a new
`BitAllocationPolicy` hook in `ruvector-agent-memory::compaction`,
feeding its decisions through the existing `WitnessSink` /
`SignedWitnessSink` (ADR-347) rather than this crate's standalone
hash-chained log.

## Feature Flags

None added. The crate is a standalone workspace member with no feature
gates and no default-on effect on any other crate.

## Benchmark Evidence

See §Evidence above and
`docs/research/nightly/2026-11-25-residual-adaptive-quantization/raw-runs.txt`
for the complete 5-seed stdout capture. Reproduce with:

```sh
cargo test -p ruvector-residual-quant
cargo run --release -p ruvector-residual-quant --bin benchmark
```

## Security

No production attack surface — the crate is not wired into any live
system. Flagged for any future promotion attempt: an adaptive
bit-allocation policy driven by a function of the memory's own content
is, in principle, steerable by whoever controls that content (e.g. an
agent or user shaping text so its embedding scores as "low importance"
to dodge aggressive compression, or shaping another memory's embedding
to force a target into the low-bit bucket). This experiment's synthetic
corpus has no adversarial-input test for that; it would be required
before any production proposal in this family.

## Governance

An allocation policy that silently decides which memories keep precision
and which degrade is an implicit retention/forgetting policy and should
carry the same auditability bar as `ruvector-agent-memory`'s existing
eviction witnessing (ADR-134, ADR-345's `witnessed_compaction`). This
crate's unsigned hash-chained `WitnessLog` demonstrates the shape of that
audit trail but is not a substitute for the signed `SignedWitnessSink`
(ADR-347) a production version would need.

## Failure Modes

- Signal collapse to a known-simpler proxy (observed here, r=0.865 vs.
  density).
- Chance-level AUC against the intended target (observed here, 0.467).
- Overall-recall regression from majority/minority budget reallocation
  (observed here, and plausibly structural to the two-bucket scheme
  itself — see Consequences).
- Not tested in this run: behavior under deletes, concurrent updates, or
  corpus scale beyond n=1500 (O(n^2) brute-force kNN graph construction
  would need an approximate index, e.g. HNSW, to scale).

## Migration

None — no production code path changed.

## Rollback

None required — nothing beyond the new, isolated, non-default crate was
touched.

## Rejection Criteria

Met in full: the pre-registered overall-recall-regression gate failed
(0.206 < 0.220), and the supporting diagnostic (residual-energy AUC
against the planted-boundary label) showed the core signal is at chance
level. Per the pre-registration, neither the hypothesis nor the
thresholds were altered after seeing these numbers.

## Open Questions

1. Is the overall-recall regression inherent to *any* two-bucket
   hi/lo-bit reallocation at this hi_fraction/bit spread, independent of
   selection signal? A fourth policy — uniform allocation at the same
   matched 3.5 bits/dim average, with no selection at all — would
   isolate this directly and is the single highest-value next
   experiment.
2. Would per-endpoint (not shared-per-edge) restriction maps, calibrated
   rather than random, break the measured residual/density correlation
   and give sheaf-residual energy predictive power the random-projection
   simplification lacks? This requires a training or calibration step
   this nightly run deliberately did not attempt.
3. Does a smaller hi_fraction (e.g. 5-10%, closer to the true boundary
   rate of 8%) change the majority-vs-minority trade-off enough to pass
   the overall-recall gate while still buying a boundary-recall
   improvement signal, for either density_gated or a future proxy?
4. Real embeddings (e.g. sentence-transformer output on an actual agent
   transcript) instead of synthetic Gaussian clusters — would the
   boundary-vector construction's "midpoint between two centers" stand-in
   for contradiction hold up, or does real text produce a qualitatively
   different outlier/boundary geometry?
