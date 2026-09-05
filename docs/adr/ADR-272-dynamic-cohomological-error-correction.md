# ADR-272: Dynamic Cohomological Error Correction — `ruvector-cohomology`

- **Status**: Implemented (Phases 0–5 of issue #669)
- **Date**: 2026-07-12
- **Issue**: [#669 — RFC: Dynamic Cohomological Error Correction](https://github.com/ruvnet/RuVector/issues/669)
- **Research spec**: `docs/research/dynamic-cohomological-error-correction.md` (branch `research/dynamic-cohomological-error-correction`)
- **Relates to**: `prime-radiant` (cellular sheaves), `ruvector-mincut` (dynamic cuts)

---

## Context

RuVector needs a mathematically correct **affine syndrome and repair engine** over
dynamic cellular sheaves: given heterogeneous local observations connected by
transformation rules, decide whether a globally coherent explanation exists,
localize the irreducible contradiction, emit a deterministic signed witness, and
compute the least costly repair or quarantine boundary.

`prime-radiant` supplies sheaf structures and `ruvector-mincut` supplies dynamic
cuts, but the existing cohomology path had gaps (issue #669): closure-based
restriction maps (unhashable, untransposable), a dense-only Laplacian assembled
from identity differences rather than both endpoint restrictions, dominant-power
iteration used to infer nullity, no affine observations `b`, no canonical
witnesses, no sparse repair, and no exact dynamic synchronization.

## Mathematical correction

The previous ADR-level statement "nontrivial `H¹` means the sheaf admits no
global section" is **wrong for linear sheaves** — the zero section is always a
global section. The corrected semantics (now also reflected in
`prime-radiant/src/cohomology/obstruction.rs`):

- `H⁰(G;F) = ker δ` — the space of **global sections**.
- `H¹(G;F) = C¹ / im δ` — **edge data unexplainable by any vertex assignment**.

Production consistency is an affine problem. For observations `b ∈ C¹` and
confidence weights `W`:

```
x* = argmin_x ‖W^{1/2}(δx − b)‖²          (weighted least squares)
s  = b − δ(δᵀWδ)†δᵀW b = b − δx*          (canonical syndrome)
```

`‖s‖_W = 0` ⇔ a globally coherent explanation exists; `‖s‖_W > 0` certifies an
irreducible contradiction whose weighted support localizes the responsible
constraints. The syndrome is gauge-invariant; only the representative `x*` needs
a gauge (minimum-norm, via CG from a zero start).

## Decision

Ship a new crate `crates/ruvector-cohomology` (pure Rust; deps: `serde`,
`sha2`, `thiserror`; optional `mincut` feature → `ruvector-mincut`):

1. **Explicit operators** (`operator.rs`): `LinearRestriction` trait +
   serializable `RestrictionMap` families (identity, scale, projection,
   orthogonal, bounded dense) with `apply`/`apply_transpose`, sound norm
   bounds, canonical SHA-256 hashes, and boundary validation (dimension,
   norm, orthogonality, finiteness) before allocation.
2. **Block-sparse coboundary** (`block_sparse.rs`): deterministic sorted
   indexing and orientation (tail = lower id), matrix-free `δ`, `δᵀ`,
   `δᵀWδ`, dense assembly only for the reference oracle.
3. **Affine syndrome** (`affine.rs`, `syndrome.rs`): observations + weights +
   gauge; production path = matrix-free CG on the normal equations with a
   residual certificate; reference path = column-pivoted rank-revealing
   Householder QR (exact); ranked support ordered on **quantized** energies.
4. **Harmonic canonicalization** (`harmonic.rs`): harmonic basis via full QR
   null space, canonicalized deterministically with QR against a hashed probe
   (degenerate subspaces included); LOBPCG (never dominant power iteration)
   for nullity/spectral-gap estimates.
5. **Group-sparse repair** (`repair.rs`, `solvers/admm.rs`): group-lasso ADMM
   over per-edge cost `c_e = α·business + β·security + γ·latency + η·reversal`,
   protected edges hard-zero unless explicitly authorized, optional debias
   refit on the selected support so the repair cancels its contradiction
   exactly. Diagnosis is separate from authorization to act.
6. **Dynamic maintenance** (`dynamic.rs`, `partition.rs`): bounded cell
   partition, all eight edit classes, dirty-cell tracking, cheap
   `estimate()`, exact `flush()` (equivalent to batch assembly; verified to
   1e-8 in tests, hash-identical witnesses), periodic partition rebuild to
   bound fragmentation drift.
7. **Orthogonal transport** (`transport.rs`): frame updates transport state
   exactly (`x_new = Q_new Q_oldᵀ x_old`, `ρ ← ρ Rᵀ`) leaving the syndrome
   invariant; cycle-consistency defect and a coherent-negative-control guard
   gate restriction-map updates.
8. **Mincut coupling** (`mincut_bridge.rs`): syndrome → per-edge tension
   (`τ_e = α‖s_e‖ + β·leverage + γ·uncertainty + ζ·severity`), quarantine
   boundary via deterministic internal Dinic max-flow between contaminated
   seeds and the healthy region (optional global cut via `ruvector-mincut`),
   isolate-first vs repair-first policies, signed intervention receipts with
   post-intervention verification.
9. **Canonical witnesses** (`witness.rs`): epochs, sorted identifiers,
   operator/observation/weight hashes, gauge + solver config, quantized
   energies/coordinates/support, repair summary, SHA-256 seal. Solver float
   noise never enters the hash — only quantized values do.

## Consequences

- Contradiction detection, localization, repair, and containment are one
  deterministic operator usable by Ruflo agent memory, policy consistency,
  embedding migration, telemetry reconciliation, and Byzantine isolation
  (Phase 6 integrations build on this crate).
- `prime-radiant` keeps its existing types; this crate adapts rather than
  replaces them. Replacement of the legacy spectrum path can proceed once
  parity gates pass (per the RFC's migration rule).
- Known scope notes: per-edge weights are scalar (block-diagonal `W` per edge
  stalk is a straightforward extension); statistical leverage is currently
  the energy-fraction proxy; the 100k-vertex performance gate needs the
  reference target hardware — the bench suite (`benches/`) covers the edit
  latency, flush, localization, repair, and containment dimensions.

## Verified properties (test suite)

- Exact dense-oracle agreement ≤ 1e-10; closed-form cycle syndrome match.
- Zero false contradiction certificates on coherent constructions, including
  under random orthogonal frame drift (energy invariant to 1e-9).
- `flush()` ≡ batch (energy and witness hash) after mixed edit streams.
- 100 repeated runs and permuted insertion orders → identical witness hash.
- Group-lasso repair objective within 10% of exhaustive optimum on tractable
  fixtures; protected edges immutable without authorization.
- Byzantine cluster quarantined along its cheap bridge boundary; combined
  containment loop verifies post-repair energy under threshold.
