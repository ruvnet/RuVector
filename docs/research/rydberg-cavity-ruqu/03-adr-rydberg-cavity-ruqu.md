# ADR: Rydberg-Cavity Superradiant Clock Phase Module for ruQu

## Status

Proposed

## Context

A theory and simulation result demonstrates that frustrated triangular Rydberg arrays coupled to a quantized optical cavity produce a novel superradiant clock (SRC) phase. This phase features coexisting spatial clock order and macroscopic photon occupation, driven by long-range light-matter coupling that lifts frustration-induced degeneracy.

This result opens a use case for ruQu as a phase discovery and control engine for cavity-Rydberg matter. The system is not a direct hardware execution target — it is a quantum many-body simulation target that benefits from variational search, symmetry reduction, and structural memory over the phase space.

## Decision

Implement the Rydberg-cavity SRC system as a ruQu module with four integration points:

### 1. Hamiltonian Simulation

Model the effective Rydberg plus cavity Hamiltonian in ruQu for phase diagram exploration, spectral analysis, and order parameter tracking.

**Rationale:** This is the cleanest fit. The Hamiltonian is well-defined, the observables are standard, and the phase classification logic is explicit.

**Scope:**
- Triangular lattice geometry with periodic and open boundary conditions
- Jaynes-Cummings cavity coupling with tunable parameters
- Adaptive photon truncation with convergence monitoring
- Symmetry-reduced exact diagonalization for N ≤ 24

### 2. Variational Phase Discovery

Use ruQu to search for the SRC regime by sweeping detuning, coupling strength, filling fraction, and frustration geometry, then measure observables including photon number, structure factor, and clock order parameters.

**Rationale:** The paper explicitly ties the phase to competition between threefold and sixfold clock terms. A systematic sweep with observable extraction can map this competition quantitatively.

**Scope:**
- Parameter grid scanning with adaptive boundary refinement
- Bayesian optimization for sample-efficient phase boundary detection
- Multiple solver backends (exact, variational, mean-field)
- Phase classification from observable thresholds

### 3. Hybrid Classical-Quantum Workflow

Since the published result came from quantum Monte Carlo-style simulation rather than near-term gate hardware, ruQu serves as a hybrid engine for surrogate modeling, parameter search, reduced Hilbert space methods, and digital-analog emulation planning.

**Rationale:** The system's Hilbert space grows exponentially with atom count and photon cutoff. Practical exploration requires hybrid methods that combine exact and approximate techniques.

**Scope:**
- Mean-field initialization with quantum fluctuation correction
- Reduced basis methods for parameter interpolation
- Tensor network methods for quasi-1D geometries
- Surrogate models trained on computed phase points

### 4. RuVector Integration

Use RuVector to index phase regions, symmetry sectors, metastable transitions, and witness logs of parameter sweeps. This provides structural memory over the phase space instead of raw amplitudes.

**Rationale:** Phase exploration is cumulative. Without memory, each sweep starts from scratch. RuVector's HNSW indexing and coherence engine enable intelligent re-use of computed results and phase-boundary-aware sampling.

**Scope:**
- HNSW-indexed phase point storage (16-dimensional embeddings)
- Mincut-based phase boundary detection via coherence engine
- Witness logs for reproducibility and provenance
- Cache-aware computation that skips explored regions

## Consequences

### Positive

- ruQu gains a concrete, well-defined many-body physics use case
- RuVector's phase memory capabilities are exercised on a real scientific problem
- The module architecture (Hamiltonian builder → symmetry reduction → solver → observables → phase memory) is reusable for other cavity QED and frustrated magnet systems
- Acceptance test is clear: recover the SRC phase consistent with published results

### Negative

- Hilbert space scaling limits exact results to small systems (N ≤ 24)
- No experimental data exists for direct validation (theory only)
- Multiple solver backends increase implementation and maintenance surface
- Photon truncation introduces systematic errors that require convergence analysis

### Risks

- The SRC phase may be a finite-size artifact not visible at N=12 (mitigated by finite-size scaling across N=12, 18, 24)
- Cavity truncation at low N_max may miss the superradiant transition (mitigated by adaptive truncation with convergence checks)
- Phase classification thresholds may need tuning per system size (mitigated by finite-size-aware threshold scaling)

## Module Structure

```
ruqu/
  rydberg_cavity/
    mod.rs              — module root, public API
    hamiltonian.rs      — Hamiltonian builder (H_Rydberg + H_cavity + H_coupling)
    lattice.rs          — lattice geometry definitions
    symmetry.rs         — symmetry reduction (translational, point group, clock)
    solver.rs           — solver dispatch (exact, variational, mean-field)
    observables.rs      — order parameter extraction and phase classification
    scanner.rs          — parameter sweep engine with adaptive refinement
    ruvector.rs         — RuVector phase memory interface
    types.rs            — shared types (PhaseLabel, ParamPoint, PhaseResult)
    tests/
      mod.rs
      hamiltonian_tests.rs
      symmetry_tests.rs
      phase_classification_tests.rs
      acceptance_test.rs  — SRC recovery acceptance test
```

## Acceptance Criteria

1. Given a triangular Rydberg lattice (N ≥ 12) with cavity coupling, the module recovers a parameter region where |ψ_clock| > 0.1 and ⟨n_ph⟩/N > 0.01 coexist
2. The transition boundary between SRC and adjacent phases is detected and stored in RuVector
3. The transition along the Z₂ symmetry line shows first-order character (discontinuous order parameter or bimodal distribution)
4. Phase diagram is consistent with published results at comparable system sizes
5. All computed points carry witness hashes for reproducibility
6. RuVector queries correctly retrieve phase points, boundaries, and metastable states

## Related ADRs

- ADR-001: RuVector core architecture (HNSW indexing)
- ADR-014: Coherence engine (phase boundary detection)
- ADR-CE-001: Sheaf Laplacian coherence (transition characterization)
- ADR-CE-004: Signed event log (witness logs)
- ADR-048: Sublinear graph attention (GNN-guided sampling)
- ADR-051: Physics-informed graph layers (Hamiltonian structure)

---

*This ADR defines the architectural decision to implement the Rydberg-cavity superradiant clock phase as a ruQu simulation module with RuVector phase memory integration.*
