# Portable Phase Capsules: Verifiable Quantum Simulation Artifacts

## The Core Idea

A phase capsule is a single RVF file that contains everything needed to reproduce, verify, and extend a quantum phase discovery result. It carries the solver, the data, the proofs, and the structural memory — all in one deterministic, portable, self-verifying artifact.

This is a different deployment primitive from anything that currently exists in computational physics.

---

## 1. What Exists Today vs. What This Changes

### Today: Non-Reproducible Simulation

```
model code (specific language version)
  + numerical libraries (specific build flags)
  + hardware (specific FPU behavior)
  + environment (OS, compiler, linker)
  + parameters (buried in scripts or notebooks)
  = results that may not reproduce
```

The reproducibility crisis in computational physics is well-documented. Two groups running the "same" quantum Monte Carlo code on different hardware routinely get different phase boundaries. The differences are small but scientifically significant — especially near phase transitions where order parameters are sensitive to numerical precision.

### With Phase Capsules: Deterministic Attestable Simulation

```
RVF phase capsule (self-contained)
  + any WIT-compatible WASM runtime
  = identical results everywhere
  + cryptographic proof of computation
```

The capsule carries its own solver bytecode, parameters, and verification logic. The WASM canonical ABI eliminates hardware-dependent floating-point divergence. The receipt chain provides cryptographic proof that specific inputs produced specific outputs.

---

## 2. Capsule Lifecycle

### 2.1 Creation

A phase capsule is created by a research workflow:

```
Phase 1: Define
  researcher specifies:
    lattice geometry (triangular, N=12, PBC)
    cavity parameters (ω_c=1.0, g=0.3, κ=0.05)
    sweep ranges (Δ ∈ [0,5], g ∈ [0,1])
    solver configuration (exact diag, symmetry-reduced)
    budget policy (5s per point, 256MB memory)

Phase 2: Build
  compile ruQu solver → WASM component
  embed lattice geometry → META_SEG
  embed solver config → META_SEG
  sign manifest → CRYPTO_SEG
  → immutable kernel created

Phase 3: Compute
  for each parameter point in grid:
    load kernel
    mint budget token
    build Hamiltonian → graph_fragment
    solve → quantum_state + receipt
    measure → order_params + receipt
    classify → phase_label + receipt
    compose receipts → point_receipt
    append to VECTOR_SEG (embedding)
    append to DATA_SEG (results)
    extend WITNESS_SEG (hash chain)
    update HNSW_SEG (index)

Phase 4: Freeze
  snapshot COW state → immutable generation
  compose all point_receipts → sweep_receipt
  sign final state → CRYPTO_SEG update
  → capsule sealed
```

### 2.2 Distribution

Phase capsules are distributed as single files:

```
rydberg_triangular_n12_src_v1.rvf  (~2-10 MB typical)

Contents:
  WASM solver:     ~200 KB (compiled, optimized)
  Phase diagram:   ~500 KB (10,000 grid points × 50 bytes each)
  HNSW index:      ~200 KB (navigable small-world graph)
  Embeddings:      ~600 KB (10,000 × 16-dim × f32)
  Receipts:        ~400 KB (10,000 point receipts + sweep receipt)
  Witness chain:   ~100 KB (SHAKE-256 hash chain)
  Metadata:        ~50 KB  (lattice, cavity, solver config)
  Signatures:      ~1 KB   (Ed25519 + optional ML-DSA-65)
```

### 2.3 Verification

Any recipient can verify the capsule:

**Fast verification** (< 1 ms):
1. Check Ed25519 signature over manifest
2. Verify manifest hash matches segment hashes
3. Confirm WASM bytecode hash matches signed manifest

**Full verification** (seconds to minutes):
1. Re-execute a random sample of phase points using the embedded solver
2. Compare results against stored values (must match exactly due to determinism)
3. Verify receipt chain integrity (each receipt links to predecessor)
4. Run structural coherence checks (neighboring phase points should have consistent labels)

**Deep verification** (minutes to hours):
1. Re-execute the entire parameter sweep
2. Independently construct the HNSW index and compare
3. Run mincut analysis on the phase boundary
4. Perform finite-size scaling analysis

### 2.4 Extension

Phase capsules support COW extension:

```
base capsule: rydberg_triangular_n12_src_v1.rvf
  ↓ COW fork
extension: rydberg_triangular_n12_src_v1_refined.rvf
  adds: 5,000 additional points near phase boundary
  adds: higher-resolution boundary interpolation
  inherits: original kernel, original 10,000 points
  delta: ~300 KB additional data

  ↓ COW fork
extension: rydberg_triangular_n18_src_v1.rvf
  adds: N=18 system size computation
  adds: finite-size scaling analysis
  inherits: kernel (modified for N=18)
  delta: ~2 MB (larger Hilbert space → more data per point)
```

Each extension carries a receipt chain linking back to the base capsule. The provenance is fully traceable.

---

## 3. Federated Phase Discovery

### 3.1 The Problem

A single research group cannot exhaustively explore the phase space of even a moderately sized Rydberg-cavity system. The parameter space is at minimum 5-dimensional (Δ, g, κ, filling, lattice size), and each point requires significant computation. Collaboration is necessary, but collaboration introduces trust questions.

### 3.2 Federated Capsule Protocol

Phase capsules enable trustless federated phase discovery:

```
Step 1: Coordinator publishes a kernel capsule
  Contains: solver WASM, lattice config, sweep protocol
  Signed by: coordinator's key
  Does NOT contain: any computed results

Step 2: Participants claim parameter regions
  Lab A: Δ ∈ [0, 2.5], g ∈ [0, 0.5]
  Lab B: Δ ∈ [2.5, 5], g ∈ [0, 0.5]
  Lab C: Δ ∈ [0, 2.5], g ∈ [0.5, 1]
  Lab D: Δ ∈ [2.5, 5], g ∈ [0.5, 1]

Step 3: Each participant computes their region
  Uses the same kernel (deterministic)
  Produces a regional capsule with receipts
  Signs with their own key

Step 4: Coordinator collects and verifies
  Spot-checks: re-computes random points from each region
  Results must match exactly (deterministic WASM)
  If any discrepancy: reject that participant's contribution
  Structural checks: phase boundaries should be continuous across regions

Step 5: Coordinator composes
  Merges regional capsules into a unified phase diagram
  Produces a composite receipt referencing all contributions
  Signs the composite capsule

Result: a phase diagram that no single group computed,
  but that anyone can verify point-by-point
```

### 3.3 Trust Model

| Verification Level | Who Trusts Whom | Mechanism |
|---|---|---|
| Signature | Everyone trusts the signer's key | Ed25519 / ML-DSA-65 |
| Determinism | Everyone trusts WASM canonical ABI | Re-execution |
| Structure | Everyone trusts the coherence engine | Mincut analysis |
| Completeness | Everyone trusts the sweep protocol | Coverage verification |

No participant needs to trust any other participant's hardware, software, or honesty. The capsule's determinism guarantee means any discrepancy is detectable.

---

## 4. Deployment Scenarios

### 4.1 Quantum Sensor Calibration (Edge)

A Rydberg quantum sensor in the field needs to classify the phase of its atomic array in real time. It carries a pre-computed phase capsule:

```
Cognitum tile
  loads: rydberg_sensor_calibration.rvf
  verifies: Ed25519 signature (0.1 ms)
  queries: HNSW index with current (Δ, g) → nearest phase point
  returns: phase_label + confidence (< 1 ms total)

If current parameters are far from any cached point:
  mints: budget_token(10_ms)
  runs: mean-field solver for approximate classification
  stores: result in COW extension
  flags: "needs hub verification"
```

### 4.2 Interactive Phase Explorer (Browser)

A researcher explores the phase diagram in a web browser:

```
Browser loads: rydberg_triangular_n12_explorer.rvf
  WASM solver runs in browser
  HNSW index enables real-time nearest-neighbor lookup
  User clicks on phase diagram → shows order parameters
  User drags parameter sliders → live re-computation
  Budget: 200 ms per interaction

If user wants higher resolution:
  browser computes additional points
  extends capsule via COW
  optionally: uploads extension to shared repository
```

### 4.3 Distributed Swarm Sweep (Server)

An agent swarm distributes a large-scale parameter sweep:

```
Orchestrator:
  publishes kernel capsule
  partitions parameter space into 100 regions
  spawns 100 agent tasks

Each agent:
  receives kernel capsule + region assignment
  mints budget_token(1_hour)
  computes all points in region
  produces regional capsule with receipts
  returns capsule to orchestrator

Orchestrator:
  verifies all regional capsules
  composes into unified phase diagram
  runs structural coherence checks
  publishes final capsule
```

---

## 5. Connection to RuVector Coherence

The phase capsule's receipt chain connects to RuVector's coherence engine at three levels:

### 5.1 Point-Level Coherence

Each phase point's order parameters define a coherence field value. The coherence engine monitors whether neighboring points in parameter space have consistent phase labels. Inconsistencies trigger adaptive refinement.

### 5.2 Boundary-Level Coherence

Phase boundaries should be smooth curves in parameter space. The mincut algorithm detects the boundary and measures its sharpness. A boundary that fragments into disconnected segments suggests insufficient resolution or a numerical artifact.

### 5.3 Capsule-Level Coherence

When multiple capsules cover overlapping parameter regions, their results must be consistent in the overlap zone. The coherence engine compares order parameters from different capsules at the same parameter points. Disagreement signals either a nondeterminism bug (should not happen with WASM) or a corrupted capsule.

---

## 6. Why This Matters

The deeper idea: when phase intelligence is packaged as a verifiable, portable, budget-governed artifact, it stops being a one-off computation and becomes a reusable, composable, trustable building block.

A phase capsule is not a paper supplement. It is not a dataset. It is not a container image. It is an executable proof that a specific physical system exhibits a specific phase structure under specific conditions. Any host can run it, verify it, extend it, or challenge it.

That shifts quantum simulation from "publish results, hope others can reproduce" to "publish artifacts, anyone can verify in milliseconds."

For the Rydberg-cavity superradiant clock phase specifically: the SRC phase is a theory prediction that has not been experimentally confirmed. A phase capsule carrying the SRC discovery would let any group independently verify the theoretical prediction, extend it to different lattice sizes or geometries, and eventually compare against experimental data — all with cryptographic proof that the computation was performed correctly.

---

*This document describes portable phase capsule architecture for verifiable quantum simulation.*
*Previous: [04-rvf-wit-cognitive-runtime.md](04-rvf-wit-cognitive-runtime.md)*
*Series: [00](00-superradiant-clock-phase.md) → [01](01-ruqu-integration-architecture.md) → [02](02-ruvector-phase-memory.md) → [03](03-adr-rydberg-cavity-ruqu.md) → [04](04-rvf-wit-cognitive-runtime.md) → 05*
