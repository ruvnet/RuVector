# RuVector Phase Memory Integration for Rydberg-Cavity Systems

## Purpose

RuVector provides **structural memory over quantum phase space** — the ability to index, cache, query, and reason about regions of a many-body phase diagram without recomputing from scratch. For the Rydberg-cavity superradiant clock system, this transforms phase exploration from brute-force scanning into an intelligent, cumulative search.

---

## 1. What Gets Stored

### 1.1 Phase Point Records

Each explored point in parameter space produces a record:

```
PhasePoint {
    // Parameter coordinates
    params: {
        detuning: f64,          // Δ/Ω
        coupling: f64,          // g/Ω
        cavity_decay: f64,      // κ/g
        filling: f64,           // sublattice occupation fraction
        lattice_size: usize,    // N atoms
        photon_cutoff: usize,   // N_max
    },

    // Measured observables
    observables: {
        clock_order: Complex,   // ψ_clock
        photon_number: f64,     // ⟨n_ph⟩
        sr_order: f64,          // Φ_SR
        structure_factor: Vec<f64>,  // S(Q) at key wavevectors
        energy: f64,            // ground state energy
        gap: f64,               // excitation gap
        entanglement: f64,      // von Neumann entropy
    },

    // Classification
    phase: PhaseLabel,          // Normal, Clock, SR, SRC
    symmetry_sector: String,    // e.g., "k=K, Z3=ω"
    confidence: f64,            // classification confidence

    // Provenance
    solver_method: String,
    convergence: f64,
    timestamp: DateTime,
    sweep_id: SweepId,
    witness_hash: Hash,         // reproducibility witness
}
```

### 1.2 Phase Boundaries

Detected transitions between phases:

```
PhaseBoundary {
    phase_a: PhaseLabel,
    phase_b: PhaseLabel,
    transition_order: TransitionOrder,  // First, Second, Crossover
    boundary_points: Vec<ParamPoint>,   // interpolated boundary curve
    critical_exponents: Option<CriticalExponents>,
    z2_line: bool,                      // is this the Z₂ symmetry line?
}
```

### 1.3 Metastable States

States that are local minima but not global ground states:

```
MetastableRecord {
    params: ParamPoint,
    energy_above_ground: f64,
    lifetime_estimate: Option<f64>,
    decay_path: Option<Vec<ParamPoint>>,
    phase_label: PhaseLabel,
}
```

---

## 2. Embedding Strategy

### 2.1 Phase-Space Vectors

Each phase point is embedded as a vector for HNSW indexing:

```
embedding = [
    // Normalized parameter coordinates (dimensions 0-4)
    Δ/Ω_max,
    g/Ω_max,
    κ/g_max,
    filling,
    1/N,  // inverse system size for finite-size awareness

    // Normalized observables (dimensions 5-11)
    |ψ_clock|,
    arg(ψ_clock) / 2π,
    ⟨n_ph⟩ / N,
    Φ_SR,
    S(Q_clock) / N,
    gap / Ω,
    S_vN / ln(2),

    // Phase one-hot (dimensions 12-15)
    is_normal,
    is_clock,
    is_sr,
    is_src,
]
```

Total embedding dimension: 16.

### 2.2 Distance Metric

Use a weighted Euclidean distance that emphasizes parameter-space proximity for exploration queries and observable-space proximity for phase classification queries:

```
d_explore(a, b) = √(w_param · ||params_a - params_b||² + w_obs · ||obs_a - obs_b||²)
    where w_param = 0.8, w_obs = 0.2

d_classify(a, b) = √(w_param · ||params_a - params_b||² + w_obs · ||obs_a - obs_b||²)
    where w_param = 0.2, w_obs = 0.8
```

### 2.3 HNSW Configuration

```
HNSW params:
    M = 16              (connections per layer)
    ef_construction = 200 (build-time search width)
    ef_search = 50       (query-time search width)
    max_elements = 10^6  (sufficient for dense phase scans)
```

---

## 3. Query Patterns

### 3.1 Exploration Queries

**"Have I explored near this parameter point?"**

```rust
fn is_explored(point: &ParamPoint, radius: f64) -> bool {
    let neighbors = hnsw.search(embed_params(point), k=1);
    neighbors[0].distance < radius
}
```

**"What phase is expected at this point?"**

```rust
fn predict_phase(point: &ParamPoint) -> (PhaseLabel, f64) {
    let neighbors = hnsw.search(embed_params(point), k=5);
    // Majority vote weighted by inverse distance
    majority_vote(&neighbors)
}
```

**"Where should I sample next?"**

```rust
fn suggest_next_point(diagram: &PhaseDiagram) -> ParamPoint {
    // Find the point with maximum uncertainty
    // = farthest from any explored point AND near a predicted boundary
    let boundary_points = diagram.boundaries.iter().flat_map(|b| &b.boundary_points);
    let unexplored = boundary_points
        .filter(|p| !is_explored(p, tolerance))
        .max_by(|p| uncertainty(p));
    unexplored
}
```

### 3.2 Classification Queries

**"Find all SRC phase points with photon number above threshold"**

```rust
fn find_src_points(min_photon: f64) -> Vec<PhasePoint> {
    ruvector::query()
        .where_phase(PhaseLabel::SuperradiantClock)
        .where_observable("photon_number", Op::Gt, min_photon)
        .execute()
}
```

**"Retrieve the phase boundary between Clock and SRC"**

```rust
fn get_boundary(a: PhaseLabel, b: PhaseLabel) -> PhaseBoundary {
    ruvector::boundaries()
        .between(a, b)
        .with_interpolation(CubicSpline)
        .execute()
}
```

### 3.3 Comparison Queries

**"How does the phase diagram change between N=12 and N=24?"**

```rust
fn finite_size_comparison(sizes: &[usize]) -> FiniteSizeReport {
    let diagrams: Vec<_> = sizes.iter()
        .map(|n| ruvector::load_diagram().where_size(*n).execute())
        .collect();
    finite_size_analysis(&diagrams)
}
```

---

## 4. Coherence Engine Integration

### 4.1 Mapping to Coherence Signals

The RuVector coherence engine (ADR-014, ADR-015) operates on sheaf Laplacian fields over graphs. The Rydberg-cavity phase space maps naturally:

| Coherence Concept | Rydberg-Cavity Analog |
|---|---|
| Node | Parameter-space point |
| Edge weight | Observable similarity between nearby points |
| Coherence field value | Order parameter magnitude |
| Sheaf section | Phase label assignment |
| Sheaf Laplacian eigenvalue | Transition sharpness |
| Mincut | Phase boundary |

### 4.2 Phase Boundary as Mincut

The phase boundary between SRC and adjacent phases corresponds to a minimum cut in the coherence graph:

1. **Build graph:** Nodes are explored parameter points. Edge weights are exp(-d²/σ²) where d is observable-space distance.
2. **Detect cut:** The minimum cut separates nodes in different phases. The cut value indicates transition sharpness (small cut = first-order, large cut = crossover).
3. **Track evolution:** As system size N increases, the cut value scaling distinguishes genuine phase transitions from finite-size artifacts.

### 4.3 Witness Logs

Every phase point computation produces a witness:

```
Witness {
    input_hash: Hash(params + solver_config),
    output_hash: Hash(observables + phase_label),
    computation_time: Duration,
    convergence_metric: f64,
    solver_version: String,
}
```

Witnesses enable:
- Reproducibility verification
- Regression detection when solver code changes
- Provenance tracking for published results

---

## 5. Caching Strategy

### 5.1 What to Cache

| Level | Content | Eviction Policy |
|-------|---------|-----------------|
| L1 (hot) | Recent sweep results, active boundary refinement | LRU, 1000 points |
| L2 (warm) | Converged phase diagram for each system size | Keep all |
| L3 (cold) | Raw wavefunctions and density matrices | Evict after observable extraction |
| L4 (archive) | Witness logs and provenance records | Permanent |

### 5.2 Cache-Aware Sweeping

Before computing a new phase point, check the cache:

```
fn compute_or_fetch(point: &ParamPoint) -> PhaseResult {
    // Check L1 cache
    if let Some(result) = l1_cache.get(point, tolerance=1e-3) {
        return result;
    }

    // Check HNSW for nearby computed points
    let neighbors = hnsw.search(embed(point), k=3);
    if neighbors[0].distance < interpolation_threshold {
        // Interpolate from neighbors if they are close enough
        // and in the same phase (no boundary crossing)
        if same_phase(&neighbors) {
            return interpolate(&neighbors, point);
        }
    }

    // Compute from scratch
    let result = solve(point);
    l1_cache.insert(point, result.clone());
    hnsw.insert(embed_full(&result));
    result
}
```

---

## 6. Practical Workflow: Phase Map Construction

```
1. Initialize RuVector namespace "rydberg_cavity_src"

2. Coarse scan (100×100 grid over Δ, g)
   → Store all points in RuVector
   → Classify phases
   → Identify candidate boundaries

3. Boundary refinement
   → Query RuVector for boundary-adjacent points
   → Adaptively subdivide cells crossing boundaries
   → Rescan at 4× resolution near boundaries
   → Update boundary interpolation

4. Finite-size scaling
   → Repeat for N = 12, 18, 24
   → Store each as separate sweep in RuVector
   → Run finite-size comparison queries
   → Identify which transitions survive thermodynamic extrapolation

5. SRC validation
   → Query: find_src_points(min_photon=0.01)
   → Verify coexistence of clock order and photon density
   → Compare boundary location with published results
   → Generate witness-verified phase diagram

6. Export
   → Phase diagram as parameterized boundary curves
   → Candidate control schedules for experimental realization
   → Full witness log for reproducibility
```

---

## 7. Connection to Existing RuVector Capabilities

| RuVector Feature | Application Here |
|---|---|
| HNSW indexing (ADR-001) | Fast nearest-neighbor search in phase space |
| Coherence engine (ADR-014) | Phase boundary detection via mincut |
| Sheaf Laplacian (ADR-CE-001) | Transition characterization |
| Witness logs (ADR-CE-004) | Computation provenance |
| Mincut detection | Phase transition identification |
| GNN routing (gnn-v2) | Adaptive sampling guidance |

---

*This document details the RuVector phase memory integration for the Rydberg-cavity ruQu module.*
*Previous: [01-ruqu-integration-architecture.md](01-ruqu-integration-architecture.md)*
*Next: [03-adr-rydberg-cavity-ruqu.md](03-adr-rydberg-cavity-ruqu.md)*
