# RVF WIT Cognitive Runtime: Portable Phase Intelligence Containers

## From File Format to Cognitive Runtime Contract

The RVF WIT world plus typed handle architecture moves RVF from a file format into a portable cognitive runtime contract. For the Rydberg-cavity ruQu module, this means phase discovery results — Hamiltonians, phase diagrams, order parameters, control schedules — become self-verifying, portable, budget-governed computation artifacts that can run identically on edge devices, servers, browsers, or agent swarms.

This document bridges the Rydberg-cavity phase discovery engine (docs 00-03) with the RVF cognitive container architecture (ADR-030, ADR-032, ADR-039, ADR-063).

---

## 1. What the WIT World Enables

### 1.1 Runtime-Agnostic Phase Intelligence

Instead of shipping a solver that depends on a specific language runtime, you ship a component with a WIT interface. The ruQu Rydberg-cavity module compiles to a WASM component that any WIT-compatible host can execute.

```
WIT world: ruqu-rydberg-cavity

  import types {
    record lattice-geometry { ... }
    record cavity-params { ... }
    record param-point { ... }
    record phase-result { ... }
    enum phase-label { normal, clock, superradiant, superradiant-clock, unknown }
  }

  import ruvector {
    resource graph-fragment { ... }
    resource receipt { ... }
    resource budget-token { ... }
  }

  export hamiltonian {
    build: func(lattice: lattice-geometry, cavity: cavity-params)
           -> result<graph-fragment, error>
  }

  export solver {
    solve: func(graph: borrow<graph-fragment>, method: solver-method,
                budget: budget-token)
           -> result<quantum-state, error>
  }

  export observables {
    measure-clock-order: func(state: borrow<quantum-state>,
                              lattice: lattice-geometry)
                         -> result<complex, error>
    measure-photon-number: func(state: borrow<quantum-state>)
                           -> result<f64, error>
    classify-phase: func(params: order-params)
                    -> result<phase-label, error>
  }

  export scanner {
    phase-scan: func(system: rydberg-cavity-system, grid: param-grid,
                     budget: budget-token)
                -> result<phase-diagram, error>
  }
```

**Portability matrix:**

| Environment | Host Runtime | Use Case |
|-------------|-------------|----------|
| Edge device | Wasmtime embedded | Field quantum sensor calibration |
| Server | Wasmtime / Cloud Run | Large-scale phase diagram computation |
| Browser | WASM runtime | Interactive phase explorer |
| Agent swarm | Orchestration layer | Distributed parameter sweep |
| Cognitum tile | WASM microkernel | Real-time phase monitoring |

A single RVF container carrying the ruQu Rydberg-cavity solver runs on all of these without recompilation.

### 1.2 Deterministic Cross-Host Phase Verification

Typed handles and canonical ABI eliminate the nondeterminism that plagues scientific computing:

| Problem | Cause | WIT Fix |
|---------|-------|---------|
| Floating-point divergence | Different FPU rounding modes | Canonical ABI fixes rounding |
| Endian differences | CPU architectures | WASM is little-endian spec |
| String encoding variance | Runtime differences | UTF-8 canonical encoding |
| Memory layout variance | Compiler struct padding | Canonical ABI record layout |
| Random seed handling | Host RNG differences | Deterministic xorshift64 in guest |

By pushing verification inside WASM with Ed25519 signatures:

```
same RVF container
  → same Hamiltonian
  → same phase diagram
  → same receipt
  → same witness
```

This is essential for reproducible quantum simulation. Two independent labs running the same RVF phase discovery container on different hardware must get identical results. The receipt proves it.

### 1.3 Capability-Based Phase Execution

The WIT resources create a capability model for phase computation:

```wit
resource graph-fragment {
  // The Hamiltonian as a sparse matrix in RuVector graph format
  // Only code with the handle can read or mutate
}

resource receipt {
  // Proof of computation: which Hamiltonian, which solver, which result
  // Cryptographically bound to the graph fragment
}

resource budget-token {
  // Compute budget: wall-clock time, memory, or solver iterations
  // Enforced by the runtime, not the guest
}
```

Execution flow for a phase point computation:

```
import_hamiltonian(lattice, cavity) → graph_fragment handle
validate_hamiltonian(handle)        → receipt (structure verified)
mint_budget(5_000_ms)               → budget_token
solve(handle, method, token)        → quantum_state + receipt
measure_observables(state)          → order_params + receipt
classify_phase(order_params)        → phase_label + receipt
compose_receipts(...)               → final_receipt (full provenance)
```

Benefits:
- No raw memory access to intermediate quantum states
- Bounded compute prevents runaway solvers
- Explicit permissions for each operation
- Deterministic metering regardless of host hardware

This maps directly to the proof-gated mutation protocol (ADR-047). The `graph_fragment` handle is a `ProofGate<SparseMatrix>` — you cannot access the Hamiltonian without presenting a valid receipt.

### 1.4 Budgeted Phase Computation

The `budget_token` concept enables deterministic compute governance for phase sweeps:

```
budget_token = {
  wall_clock: 5_000 ms,
  memory: 256 MB,
  solver_iterations: 10_000,
  hnsw_queries: 1_000
}
```

If the solver exceeds any budget dimension:

```
solve(graph, method, token) → Err(BudgetExhausted { dimension: "wall_clock", used: 5001, limit: 5000 })
```

Application to phase scanning:

| System | Budget Policy | Benefit |
|--------|--------------|---------|
| Edge quantum sensor | 50 ms per phase point | Real-time classification |
| Server sweep | 5 s per phase point | Thorough exploration |
| Browser interactive | 200 ms per point | Responsive UI |
| Agent swarm | Fair-share per agent | No starvation |
| Cognitum tile | 10 ms per point | Thermal control |

For the Rydberg-cavity system specifically: exact diagonalization at N=12 with N_max=8 photons takes ~100 ms per point. Mean-field takes ~1 ms. The budget token lets the runtime automatically select the solver method based on available compute, falling back to cheaper methods when the budget is tight.

### 1.5 Portable Phase Proof Chains

Receipts become portable witnesses for phase discovery:

```
graph_fragment (Hamiltonian)
  + receipt (solver execution proof)
  + manifest (RVF container signature)
  = verifiable phase computation
```

An RVF phase container can prove:
- Which lattice geometry was used
- Which Hamiltonian was constructed
- Which solver method ran
- Which version of the manifest signed it
- Which phase was classified
- Which order parameters were measured
- Which budget was consumed

That is **proof-carrying quantum simulation**.

---

## 2. Phase Capsule Architecture

### 2.1 The Phase Capsule

A phase capsule is an RVF container carrying a complete, self-verifying phase computation:

```
┌──────────────────────────────────────────────────────┐
│                RVF Phase Capsule                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  MANIFEST (0x05)                                     │
│    version, segment directory, root hash             │
│                                                      │
│  META_SEG (0x07)                                     │
│    lattice geometry, cavity params, solver config    │
│                                                      │
│  WASM_SEG (0x40)                                     │
│    ruQu Rydberg-cavity solver (compiled WASM)        │
│                                                      │
│  VECTOR_SEG (0x01)                                   │
│    phase-space embeddings (16-dim, HNSW-indexed)     │
│                                                      │
│  HNSW_SEG (0x02)                                     │
│    navigable small-world graph over phase points     │
│                                                      │
│  GRAPH_SEG (0x08)                                    │
│    Hamiltonian sparse matrix (COO format)            │
│    phase boundary graph (adjacency list)             │
│                                                      │
│  DATA_SEG (0x09)                                     │
│    phase diagram grid (param → phase label)          │
│    order parameter tables                            │
│    critical boundary coordinates                     │
│                                                      │
│  WITNESS_SEG (0x0A)                                  │
│    SHAKE-256 hash chain:                             │
│      build → solve → measure → classify → sign       │
│                                                      │
│  RECEIPT_SEG (0x0B)                                  │
│    per-point computation receipts                    │
│    composed sweep receipt                            │
│    budget consumption log                            │
│                                                      │
│  CRYPTO_SEG (0x0C)                                   │
│    Ed25519 signature over all segments               │
│    ML-DSA-65 post-quantum signature (optional)       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2.2 Immutable Kernel, COW State

The phase capsule follows a two-layer model:

**Immutable layer** (the kernel):
- WASM solver bytecode
- Manifest and signatures
- Lattice geometry definition
- Solver configuration

This layer is the verifiable provenance anchor. It never changes.

**COW layer** (the state):
- Phase diagram grid (grows as more points are computed)
- Phase-space embeddings (appended)
- HNSW index (updated)
- Receipt chain (extended)
- Order parameter tables (accumulated)

Each state snapshot produces a new receipt linking back to the immutable kernel. The COW mechanism from ADR-026 (RVCOW) provides cluster-granularity copy-on-write with SIMD-aligned slabs, so state updates are O(delta) not O(total).

**Lifecycle:**

```
1. Create capsule with immutable kernel
     manifest + WASM solver + lattice config
     → signed, sealed

2. Compute phase points (COW state evolves)
     for each (Δ, g) in grid:
       solve(kernel, params) → phase_result
       append phase_result to COW state
       extend receipt chain
       update HNSW index

3. Freeze snapshot
     freeze COW state → immutable generation
     compose all receipts → sweep receipt
     sign sweep receipt

4. Ship or query
     the capsule now contains:
       immutable kernel (provenance)
       frozen state (results)
       receipt chain (proof)
     any host can verify and query
```

### 2.3 Split Verification Model

For deployment across heterogeneous hardware:

**Tile-level verification** (fast, local):
- Ed25519 signature check on manifest (~0.1 ms)
- Receipt hash chain verification (~0.01 ms per receipt)
- Budget token validation
- Result: "this artifact is signed and unmodified"

**Hub-level verification** (structural, slower):
- Phase diagram coherence via mincut analysis
- Order parameter consistency across neighboring points
- Finite-size scaling validation
- Sheaf Laplacian eigenvalue analysis for transition characterization
- Result: "this artifact's phase classification is structurally sound"

```
                    ┌─────────────────┐
                    │   Hub Verifier   │
                    │  (structural)    │
                    │                  │
                    │  mincut check    │
                    │  coherence scan  │
                    │  sheaf analysis  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
        │  Tile A    │ │  Tile B    │ │  Tile C    │
        │ signature  │ │ signature  │ │ signature  │
        │ receipt    │ │ receipt    │ │ receipt    │
        │ budget     │ │ budget     │ │ budget     │
        │            │ │            │ │            │
        │ solve(Δ₁)  │ │ solve(Δ₂)  │ │ solve(Δ₃)  │
        └────────────┘ └────────────┘ └────────────┘
```

Each tile runs a portion of the parameter sweep within its budget. The hub collects results, verifies structural coherence, and composes the final phase diagram.

---

## 3. Concrete WIT Interface

### 3.1 Types World

```wit
package ruqu:rydberg-cavity@0.1.0;

world phase-engine {

  // --- Imported types ---

  record lattice-geometry {
    topology: lattice-type,
    num-sites: u32,
    boundary: boundary-condition,
    positions: list<tuple<f64, f64>>,
    neighbor-list: list<list<u32>>,
  }

  enum lattice-type { triangular, honeycomb, kagome, square, custom }
  enum boundary-condition { periodic, open, cylindrical }

  record cavity-params {
    frequency: f64,
    coupling-g: f64,
    decay-kappa: f64,
    photon-cutoff: u32,
  }

  record interaction-profile {
    c6-coefficient: f64,
    c3-coefficient: f64,
    blockade-radius: f64,
  }

  record drive-params {
    rabi-frequency: f64,
    detuning: f64,
    schedule: option<list<tuple<f64, f64>>>,
  }

  record rydberg-cavity-system {
    lattice: lattice-geometry,
    cavity: cavity-params,
    interactions: interaction-profile,
    drive: drive-params,
  }

  record order-params {
    clock-order-magnitude: f64,
    clock-order-phase: f64,
    photon-number: f64,
    superradiant-order: f64,
    structure-factor-peak: f64,
    energy: f64,
    gap: f64,
    entanglement-entropy: f64,
  }

  enum phase-label { normal, clock, superradiant, superradiant-clock, unknown }

  record phase-result {
    label: phase-label,
    order-params: order-params,
    witness-hash: list<u8>,
    budget-consumed: budget-report,
  }

  record budget-report {
    wall-clock-ms: u64,
    memory-bytes: u64,
    solver-iterations: u64,
  }

  // --- Resources (typed handles) ---

  resource graph-fragment {
    constructor(system: rydberg-cavity-system);
    num-sites: func() -> u32;
    hilbert-dim: func() -> u64;
    symmetry-sectors: func() -> list<string>;
  }

  resource receipt {
    verify: func() -> result<bool, string>;
    hash: func() -> list<u8>;
    compose: func(other: borrow<receipt>) -> receipt;
  }

  resource budget-token {
    constructor(wall-clock-ms: u64, memory-bytes: u64, iterations: u64);
    remaining: func() -> budget-report;
    is-exhausted: func() -> bool;
  }

  // --- Exported functions ---

  export build-hamiltonian: func(system: rydberg-cavity-system)
    -> result<graph-fragment, string>;

  export reduce-symmetry: func(graph: borrow<graph-fragment>, sector: string)
    -> result<graph-fragment, string>;

  export solve: func(graph: borrow<graph-fragment>, method: string,
                     budget: budget-token)
    -> result<tuple<list<f64>, receipt>, string>;

  export measure-observables: func(state: list<f64>,
                                   lattice: lattice-geometry)
    -> result<order-params, string>;

  export classify-phase: func(params: order-params)
    -> result<phase-label, string>;

  export phase-scan: func(system: rydberg-cavity-system,
                          detuning-range: tuple<f64, f64>,
                          coupling-range: tuple<f64, f64>,
                          grid-resolution: u32,
                          budget: budget-token)
    -> result<list<phase-result>, string>;
}
```

### 3.2 RuVector Integration World

```wit
package ruqu:phase-memory@0.1.0;

world phase-memory {

  use ruqu:rydberg-cavity/types.{param-point, phase-result, phase-label};

  resource phase-index {
    constructor(namespace: string);
    store: func(point: param-point, result: phase-result) -> result<u64, string>;
    query-nearest: func(point: param-point, k: u32) -> result<list<phase-result>, string>;
    query-by-phase: func(label: phase-label) -> result<list<phase-result>, string>;
    count: func() -> u64;
  }

  resource boundary-tracker {
    constructor(index: borrow<phase-index>);
    detect-boundaries: func(tolerance: f64) -> result<list<boundary-segment>, string>;
    classify-transition: func(segment: borrow<boundary-segment>)
      -> result<transition-order, string>;
  }

  record boundary-segment {
    phase-a: phase-label,
    phase-b: phase-label,
    points: list<param-point>,
    sharpness: f64,
  }

  enum transition-order { first, second, crossover, unknown }
}
```

---

## 4. SOTA Integration: Current Research Context

### 4.1 Rydberg-Cavity QED State of the Art

The superradiant clock phase result sits within a rapidly developing field. Key SOTA elements:

**Frustrated quantum magnets in cavities.** Recent theoretical work extends beyond the triangular lattice to kagome and pyrochlore geometries, where frustration-induced degeneracy is even more extreme. The cavity coupling mechanism described in the SRC paper generalizes: any frustrated lattice plus a quantized cavity can exhibit novel photon-dressed phases.

**Quantum Monte Carlo for cavity systems.** The computational methods used in the SRC paper (effective field theory plus stochastic sampling) represent the current best approach for 2D cavity-coupled systems. Tensor network methods (DMRG, PEPS) are advancing but struggle with the cavity mode's infinite-range coupling. Variational approaches (neural quantum states, VQE-inspired ansatze) are emerging as alternatives for larger systems.

**Experimental progress.** Rydberg atom arrays in optical cavities are being realized in several labs. Tweezer arrays provide site-resolved control of individual atoms. Cavity integration adds the global coupling mode. The gap between theory and experiment is narrowing — the SRC phase could be experimentally accessible within 2-5 years.

**Digital-analog quantum simulation.** Hybrid approaches that combine digital gate operations with analog Hamiltonian evolution are being explored for simulating exactly these kinds of systems on near-term quantum hardware. The ruQu hybrid workflow maps to this paradigm.

### 4.2 RVF and WASM Component Model State of the Art

**WASI Preview 2 and the Component Model.** The WebAssembly Component Model (released 2024, stabilizing through 2025-2026) provides the WIT interface definition language, canonical ABI for cross-component communication, and typed resource handles. This is the foundation for the phase capsule architecture.

**Deterministic WASM execution.** Recent work on deterministic floating-point in WASM (wasm-float proposal, canonical NaN handling) addresses the reproducibility requirement for scientific computing. The canonical ABI's treatment of floats ensures cross-host consistency.

**Capability-based security in WASM.** The resource handle model in WIT implements an object-capability security pattern. Combined with WASI's capability-based filesystem and network access, this provides the security model needed for proof-gated phase computation.

**Post-quantum signatures.** ML-DSA (formerly CRYSTALS-Dilithium) is standardized in FIPS 204 (2024). RVF's optional ML-DSA-65 signatures provide quantum-resistant attestation for phase capsules — relevant since the capsules themselves model quantum systems.

### 4.3 Verifiable Scientific Computing

**Reproducibility crisis in computational physics.** A growing body of work documents failures in reproducing published simulation results. The receipt-based provenance chain in the phase capsule directly addresses this: every phase point carries cryptographic proof of its computation.

**Proof-carrying computation.** The concept of attaching machine-checkable proofs to computational results originated in programming language theory (proof-carrying code, Necula 1997). Extending this to scientific simulation is an active research direction. The phase capsule's receipt chain is an instance of this pattern applied to quantum many-body physics.

**Federated scientific computing.** Distributed parameter sweeps across institutions require result integrity. The phase capsule's split verification model (tile signature + hub coherence) enables federated phase diagram construction where no single party needs to trust the others.

---

## 5. Strategic Stack Integration

The Rydberg-cavity ruQu module, when packaged as RVF phase capsules, creates a five-layer stack:

```
┌─────────────────────────────────────────────┐
│  Layer 5: Phase Intelligence                 │
│  ruQu solver + phase classifier              │
│  Outputs: phase labels, order parameters,    │
│           control schedules                  │
├─────────────────────────────────────────────┤
│  Layer 4: Cognitive Container (RVF)          │
│  Immutable kernel + COW state                │
│  WIT interface + typed handles               │
│  Budget tokens + receipt chain               │
├─────────────────────────────────────────────┤
│  Layer 3: Structural Memory (RuVector)       │
│  HNSW-indexed phase-space embeddings         │
│  Coherence engine + mincut detection         │
│  Witness logs + provenance tracking          │
├─────────────────────────────────────────────┤
│  Layer 2: Deterministic Runtime (WASM)       │
│  Canonical ABI + deterministic floats        │
│  Capability-based execution                  │
│  Cross-platform identical results            │
├─────────────────────────────────────────────┤
│  Layer 1: Verification (Receipts)            │
│  Ed25519 / ML-DSA-65 signatures              │
│  SHAKE-256 witness chains                    │
│  Proof-gated mutation protocol               │
└─────────────────────────────────────────────┘
```

The result is a verifiable phase discovery operating system. Phase intelligence is not just computed — it is attested, portable, budget-governed, and structurally verified.

---

## 6. Connection to Existing ADRs

| ADR | Connection |
|-----|------------|
| ADR-026 (RVCOW) | COW state layer for evolving phase diagrams |
| ADR-029 (RVF canonical format) | Container format for phase capsules |
| ADR-030 (RVF cognitive container) | Self-booting phase solver |
| ADR-032 (RVF WASM integration) | WASM segment for solver bytecode |
| ADR-039 (RVF solver WASM) | Handle-based API pattern for phase solver |
| ADR-047 (Proof-gated mutation) | Receipt-gated access to Hamiltonian and state |
| ADR-063 (WASM executable nodes) | Deterministic cross-host execution model |
| ADR-CE-004 (Signed event log) | Witness chain for phase computations |
| ADR-CE-012 (Gate refusal witness) | Budget exhaustion logging |

---

*This document defines the RVF WIT cognitive runtime architecture for the Rydberg-cavity ruQu phase discovery module.*
*Previous: [03-adr-rydberg-cavity-ruqu.md](03-adr-rydberg-cavity-ruqu.md)*
*Next: [05-portable-phase-capsules.md](05-portable-phase-capsules.md)*
