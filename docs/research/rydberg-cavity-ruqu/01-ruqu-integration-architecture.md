# ruQu Integration Architecture: Rydberg-Cavity Phase Discovery Engine

## Framing

ruQu as a **phase discovery and control layer for cavity-Rydberg matter**.

This is not a direct plug-in for ruQu the way you would drop in a gate set or a standard circuit primitive. The system is a quantum many-body model and simulation target inside ruQu. The paper's result is a cavity QED style many-body system: frustrated triangular Rydberg arrays plus a quantized cavity field. Their simulations show a superradiant clock phase where spatial clock order coexists with macroscopic photon occupation, attributed to long-range light-matter coupling lifting frustration-driven degeneracy.

The win is not "run the paper." The win is **"turn ruQu into a search engine for exotic light-matter phases."**

---

## 1. Minimal Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ruQu Phase Engine                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUT                                                  │
│  ├── Rydberg lattice geometry (triangular, honeycomb,   │
│  │   kagome, custom)                                    │
│  ├── Cavity mode parameters (frequency, decay rate,     │
│  │   mode volume)                                       │
│  ├── Interaction strengths (C₆, C₃ coefficients)       │
│  ├── Detuning schedule Δ(t)                             │
│  └── Drive schedule Ω(t)                                │
│                                                         │
│  CORE                                                   │
│  ├── Effective Hamiltonian builder                      │
│  ├── Symmetry reduction engine                          │
│  ├── Variational / Monte Carlo-assisted search          │
│  ├── Observable extraction pipeline                     │
│  └── RuVector phase memory interface                    │
│                                                         │
│  OUTPUT                                                 │
│  ├── Phase label (normal, clock, superradiant, SRC)     │
│  ├── Order parameters (ψ_clock, Φ_SR, structure factor)│
│  ├── Critical boundaries (phase diagram slices)         │
│  └── Candidate control schedules                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Module Decomposition

### 2.1 Hamiltonian Builder (`ruqu::rydberg_cavity::hamiltonian`)

Constructs the effective Hamiltonian for the Rydberg-cavity system.

**Components:**

```
H = H_Rydberg + H_cavity + H_coupling + H_drive

H_Rydberg = Σ_<i,j> V_ij n_i n_j          (Rydberg blockade interaction)
H_cavity  = ω_c a†a                        (cavity mode)
H_coupling = (g/√N) Σ_i (a†σ_i⁻ + a σ_i⁺) (Jaynes-Cummings coupling)
H_drive   = Ω Σ_i (σ_i⁺ + σ_i⁻)          (coherent drive)
           - Δ Σ_i n_i                      (detuning)
```

**Input parameters:**
- `LatticeGeometry` — defines site positions, neighbor lists, boundary conditions
- `InteractionProfile` — V_ij coefficients (van der Waals C₆/r⁶ or dipolar C₃/r³)
- `CavityParams` — frequency ω_c, coupling g, decay κ, photon cutoff N_max
- `DriveParams` — Rabi frequency Ω, detuning Δ, time-dependent schedules

**Key design choice:** The photon Hilbert space must be truncated. Use adaptive truncation: start with N_max = 5-10 and increase until observables converge. This is the primary bottleneck for system size.

### 2.2 Symmetry Reduction (`ruqu::rydberg_cavity::symmetry`)

Exploits lattice and internal symmetries to reduce the effective Hilbert space.

**Symmetry sectors:**
- **Translational** — momentum-space block diagonalization using lattice periodicity
- **Point group** — C₃v or C₆v for triangular lattice, reduces blocks by factor 6-12
- **Particle number** — total excitation conservation (atoms + photons) when drive is off
- **Z₃/Z₆ clock symmetry** — decompose into clock sectors for targeted search

**Implementation:** Represent symmetry operations as sparse permutation matrices. Project the Hamiltonian into each sector before diagonalization. This is critical for reaching system sizes where the SRC phase is visible.

### 2.3 Solver Backend (`ruqu::rydberg_cavity::solver`)

Multiple solver strategies, selected based on system size and target accuracy.

| Method | System Size | Photon Modes | Accuracy | Use Case |
|--------|-------------|--------------|----------|----------|
| Exact diagonalization | ≤ 18 sites | ≤ 10 photons | Exact | Benchmarking, small systems |
| DMRG / tensor network | ≤ 100 sites | ≤ 20 photons | Controlled | 1D/quasi-1D geometries |
| Variational Monte Carlo | ≤ 200 sites | ≤ 50 photons | Approximate | 2D phase search |
| Mean-field + fluctuations | Unlimited | Unlimited | Qualitative | Phase diagram overview |
| Truncated Wigner | ≤ 500 sites | Unlimited | Semi-classical | Dynamics, time evolution |

**Hybrid strategy:** Use mean-field to identify candidate regions, then refine with variational Monte Carlo, then validate with exact diagonalization on small clusters.

### 2.4 Observable Pipeline (`ruqu::rydberg_cavity::observables`)

Extracts physical observables from the quantum state.

**Primary observables:**

| Observable | Formula | Phase Indicator |
|-----------|---------|-----------------|
| Clock order parameter | ψ_clock = (1/N) Σ exp(i·2π·s_i/p)·exp(i·Q·r_i) | Nonzero → clock order |
| Photon number | ⟨n_ph⟩ = ⟨a†a⟩ | Macroscopic → superradiance |
| Superradiant order | Φ_SR = ⟨a⟩/√N | Nonzero → SR transition |
| Structure factor | S(Q) = (1/N)|Σ n_i exp(i·Q·r_i)|² | Peak at Q → density wave |
| Entanglement entropy | S_vN = -Tr(ρ_A ln ρ_A) | Scaling → phase character |
| Photon-atom correlation | g⁽²⁾ = ⟨n_ph n_atom⟩ / ⟨n_ph⟩⟨n_atom⟩ | Bunching → correlation |

**Phase classification logic:**

```
if |ψ_clock| > threshold AND ⟨n_ph⟩/N > threshold:
    phase = SRC  (superradiant clock)
elif |ψ_clock| > threshold AND ⟨n_ph⟩/N ≈ 0:
    phase = CLOCK (clock order only)
elif |ψ_clock| ≈ 0 AND ⟨n_ph⟩/N > threshold:
    phase = SR   (superradiant only)
else:
    phase = NORMAL (disordered)
```

### 2.5 Phase Scanner (`ruqu::rydberg_cavity::scanner`)

Systematic parameter sweep engine.

**Sweep dimensions:**
- Detuning Δ/Ω ratio
- Coupling strength g/Ω ratio
- Filling fraction (sublattice occupation)
- Frustration geometry (lattice type, next-nearest-neighbor strength)
- Cavity decay rate κ/g ratio

**Sweep strategies:**
1. **Grid scan** — coarse grid over full parameter space
2. **Adaptive refinement** — subdivide cells near detected transitions
3. **Bayesian optimization** — sample-efficient search for phase boundaries
4. **RuVector-guided** — use cached phase neighborhoods to skip explored regions

### 2.6 RuVector Interface (`ruqu::rydberg_cavity::ruvector`)

Connects phase exploration to RuVector's structural memory.

**Stored objects:**
- Phase labels indexed by parameter coordinates
- Order parameter values at each explored point
- Symmetry sector tags
- Metastable state records
- Transition boundary coordinates
- Witness logs (hash of parameter + observable snapshot)

**Query patterns:**
- "Find all parameter points in the SRC phase with photon number > 10"
- "Retrieve the nearest explored point to (Δ=2.5, g=0.3, κ=0.1)"
- "List all phase boundaries crossed during sweep #47"

**HNSW integration:** Phase-space points are embedded as vectors (parameter coordinates + order parameter values). HNSW indexing enables fast nearest-neighbor search to avoid redundant exploration.

---

## 3. Workflow: Variational Phase Discovery

```
Step 1: Define lattice and cavity
    → LatticeGeometry(triangular, N=12, PBC)
    → CavityParams(ω_c=1.0, g=0.3, κ=0.05, N_max=8)

Step 2: Build Hamiltonian in symmetry-reduced sectors
    → H_k for each momentum sector k
    → H_k,s for each clock symmetry sector s

Step 3: Coarse phase scan
    → Sweep Δ/Ω ∈ [0, 5], g/Ω ∈ [0, 1]
    → Measure ψ_clock, ⟨n_ph⟩, S(Q) at each point
    → Classify phases, store in RuVector

Step 4: Adaptive refinement near transitions
    → Identify cells with phase boundary crossings
    → Subdivide and rescan at higher resolution
    → Update RuVector phase map

Step 5: Validate SRC identification
    → Check coexistence: |ψ_clock| > 0 AND ⟨n_ph⟩/N > 0
    → Verify first-order character at Z₂ line
    → Compare with paper's published phase diagram

Step 6: Extract control schedules
    → Identify parameter paths that traverse from normal → SRC
    → Optimize for adiabatic preparation feasibility
    → Output candidate experimental protocols
```

---

## 4. Hilbert Space Management

### 4.1 The Problem

The Hilbert space grows as:

```
dim(H) = 2^N_atoms × N_photon_max
```

For N=24 atoms with N_max=10 photons: dim ≈ 1.7 × 10⁸. This is already borderline for exact methods.

### 4.2 Fix Path

**Truncated cavity occupation.** Keep only photon numbers 0 through N_max where N_max is adaptively chosen. Start small, increase until observables converge to within tolerance.

**Symmetry sector pruning.** Project into specific momentum and clock symmetry sectors. Reduces effective dimension by factor 12-36 for the triangular lattice.

**Tensor network methods.** For quasi-1D geometries (cylinders, strips), use DMRG with cavity mode as an auxiliary site. The entanglement structure of the SRC phase determines the required bond dimension.

**RuVector caching.** Cache phase-space neighborhoods so that parameter sweeps do not repeatedly explore equivalent regions. HNSW indexing over the parameter-observable space enables O(log N) lookup of previously computed points.

**Reduced basis methods.** For parameter sweeps, compute the full solution at a few anchor points, then interpolate using a reduced basis constructed from those solutions.

---

## 5. Acceptance Test

**Given:**
- A triangular Rydberg lattice (N ≥ 12 sites, periodic boundary conditions)
- Quantized cavity coupling with tunable strength g

**ruQu must:**
1. Recover a region in the (Δ, g) parameter space where clock order (|ψ_clock| > 0) and nonzero photon density (⟨n_ph⟩/N > 0) coexist
2. Map the transition boundary as coupling varies
3. Identify the first-order character of the transition along the Z₂ symmetry line
4. Produce results consistent with the paper's SRC phase description
5. Store the full phase map in RuVector with witness logs

**Quantitative targets:**
- Clock order parameter magnitude |ψ_clock| > 0.1 in the SRC region
- Photon density ⟨n_ph⟩/N > 0.01 in the SRC region
- Phase boundary location within 10% of published values (where comparable system sizes are available)
- Full parameter sweep completed in under 4 hours on standard hardware for N=12

---

## 6. API Surface

### 6.1 Core Types

```rust
struct RydbergCavitySystem {
    lattice: LatticeGeometry,
    cavity: CavityParams,
    interactions: InteractionProfile,
    drive: DriveParams,
}

struct PhaseResult {
    label: PhaseLabel,           // Normal, Clock, SR, SRC
    order_params: OrderParams,   // ψ_clock, Φ_SR, S(Q)
    energy: f64,
    photon_number: f64,
    entanglement: f64,
    witness_hash: Hash,
}

enum PhaseLabel {
    Normal,
    Clock,
    Superradiant,
    SuperradiantClock,  // The target phase
    Unknown,
}

struct PhaseDiagram {
    grid: Vec<(ParamPoint, PhaseResult)>,
    boundaries: Vec<PhaseBoundary>,
    metadata: SweepMetadata,
}
```

### 6.2 Core Functions

```rust
// Build and solve
fn build_hamiltonian(system: &RydbergCavitySystem) -> SparseMatrix;
fn reduce_symmetry(h: &SparseMatrix, sector: &SymmetrySector) -> SparseMatrix;
fn solve(h: &SparseMatrix, method: SolverMethod) -> QuantumState;

// Measure
fn measure_clock_order(state: &QuantumState, lattice: &LatticeGeometry) -> Complex;
fn measure_photon_number(state: &QuantumState) -> f64;
fn measure_structure_factor(state: &QuantumState, q: &[f64]) -> f64;
fn classify_phase(observables: &OrderParams) -> PhaseLabel;

// Scan
fn phase_scan(system: &RydbergCavitySystem, grid: &ParamGrid) -> PhaseDiagram;
fn adaptive_refine(diagram: &PhaseDiagram, tolerance: f64) -> PhaseDiagram;

// RuVector integration
fn store_phase_point(point: &ParamPoint, result: &PhaseResult) -> WitnessId;
fn query_nearest_phase(point: &ParamPoint, k: usize) -> Vec<PhaseResult>;
fn load_phase_diagram(sweep_id: &SweepId) -> PhaseDiagram;
```

---

## 7. RVF Cognitive Container Layer

The architecture gains a deployment and verification layer through RVF phase capsules.

### 7.1 Phase Capsule Packaging

The entire module — solver WASM, phase diagram, HNSW index, receipt chain — packages into a single RVF container:

```
┌─────────────────────────────────────────────────────────┐
│               RVF Phase Capsule                          │
├─────────────────────────────────────────────────────────┤
│  Layer 5: Phase Intelligence (this module)              │
│  Layer 4: Cognitive Container (RVF + WIT + handles)     │
│  Layer 3: Structural Memory (RuVector + HNSW)           │
│  Layer 2: Deterministic Runtime (WASM canonical ABI)    │
│  Layer 1: Verification (Ed25519 receipts + witnesses)   │
└─────────────────────────────────────────────────────────┘
```

### 7.2 WIT Interface

The module exports a WIT world (`ruqu:rydberg-cavity@0.1.0`) with typed resource handles:

- `graph-fragment` — the Hamiltonian as a capability-gated sparse matrix
- `receipt` — composable computation proof
- `budget-token` — deterministic compute governance

### 7.3 Deployment Model

| Target | Runtime | Budget Policy | Verification |
|--------|---------|---------------|--------------|
| Server sweep | Wasmtime | 5s/point | Full re-execution |
| Browser explorer | Browser WASM | 200ms/point | Signature only |
| Edge sensor | Embedded Wasmtime | 10ms/point | Signature + HNSW lookup |
| Agent swarm | Orchestration layer | Fair-share | Hub coherence check |
| Cognitum tile | WASM microkernel | Budget-governed | Split (tile sig + hub structure) |

### 7.4 Immutable Kernel + COW State

The solver bytecode and configuration are immutable (provenance anchor). The phase diagram, embeddings, and index evolve via COW snapshots (ADR-026). Each snapshot produces a receipt linking back to the immutable kernel.

See [04-rvf-wit-cognitive-runtime.md](04-rvf-wit-cognitive-runtime.md) and [05-portable-phase-capsules.md](05-portable-phase-capsules.md) for full design.

---

## 8. Honest Assessment

### Strengths
- Strong fit for ruQu as a simulation and optimization engine
- Natural integration with RuVector for structural memory over phase space
- Variational and hybrid methods are well-suited to the problem
- The SRC phase has clear, measurable order parameters
- RVF capsule packaging enables deterministic reproducibility and federated collaboration
- WIT typed handles provide capability-based security for phase computation

### Weak Points
- Hilbert space blows up fast with atoms + photons
- Exact results limited to ~18-24 sites
- No experimental data to validate against (theory only)
- Clock order detection requires careful finite-size analysis
- WASM deterministic float handling requires careful validation for edge cases

### Mitigation
- Truncated cavity, symmetry pruning, tensor methods
- RuVector caching to avoid redundant exploration
- Benchmark against published QMC results at available system sizes
- Use multiple system sizes for finite-size scaling
- RVF receipt chains provide cryptographic proof of computational correctness
- Budget tokens prevent runaway solvers on resource-constrained targets

---

*This document defines the ruQu integration architecture for the Rydberg-cavity superradiant clock phase.*
*Next: [02-ruvector-phase-memory.md](02-ruvector-phase-memory.md) — RuVector phase memory integration details.*
*See also: [04-rvf-wit-cognitive-runtime.md](04-rvf-wit-cognitive-runtime.md) — RVF WIT cognitive runtime integration.*
