# RuVector Consciousness API

## Overview

The `ruvector-consciousness` crate implements IIT 4.0 (Albantakis et al. 2023) -- the current state-of-the-art framework for computing integrated information and consciousness metrics. Written in Rust with SIMD acceleration, zero-alloc hot paths, and rayon parallelism.

Available via:

- **Rust API** -- direct crate dependency (`ruvector-consciousness`)
- **REST API** -- `pi.ruv.io/v1/consciousness/*`
- **MCP Tools** -- `brain_consciousness_compute`, `brain_consciousness_status`

Key departure from IIT 3.0: replaces KL divergence with the Earth Mover's Distance (intrinsic difference), making the measure topology-aware and intrinsic to the system rather than observer-relative.

## Algorithms

### IIT 4.0 Mechanism-Level phi (`iit4_phi`)

Computes the integrated information phi for a single mechanism (subset of system elements).

**Intrinsic difference** replaces KL divergence from IIT 3.0. Uses Wasserstein-1 (EMD) on the discrete state space, which reduces to the cumulative L1 difference for 1D distributions:

```
d(p, q) = sum_i |cumsum(p - q)_i|
```

**Cause/effect repertoires:**
- Cause repertoire: P(past_purview | mechanism = s) -- how the mechanism in state s constrains the distribution over past purview states.
- Effect repertoire: P(future_purview | mechanism = s) -- how the mechanism constrains future purview states.

**Mechanism partition search (MIP):**
- Enumerates all bipartitions of the mechanism.
- For each partition, computes the product of partitioned repertoires.
- phi = min(phi_cause, phi_effect), where each is the intrinsic difference between the intact and best-partitioned repertoire.
- Single-element mechanisms use selectivity (distance from uniform) directly.

**Complexity:** O(2^(2n)) for n binary elements -- all mechanisms times all purviews times all partitions.

### Cause-Effect Structure (`ces`)

The CES is the central object in IIT 4.0. It is the full set of distinctions and relations specified by a system in a state -- the "shape" of experience.

**Distinction enumeration:**
- Enumerates all 2^n - 1 non-empty subsets of elements as candidate mechanisms.
- Computes phi for each via `mechanism_phi`.
- Retains mechanisms with phi > threshold (default 1e-6).
- Sorted by phi descending.

**Relation computation:**
- Pairwise relations between distinctions with overlapping purviews.
- Relation phi = sqrt(phi_i * phi_j) * overlap_fraction.
- Only relations with phi > 1e-10 are retained.

**System-level Phi (big phi):**
- Measures irreducibility of the entire CES under system bipartition.
- For each bipartition, mechanisms spanning the cut have their phi zeroed.
- Phi = min over all bipartitions of the intrinsic difference between intact and partitioned distinction vectors.

**Parallelism:** With the `parallel` feature, mechanism enumeration uses rayon for n >= 5 elements (3-6x speedup).

**Limits:** Max 12 elements (4096 states). Returns `ConsciousnessError::SystemTooLarge` above that.

### Integrated Information Decomposition (`phi_id`)

Implements Mediano et al. (2021) PhiID. Decomposes the mutual information I(past; future) into four information atoms:

| Atom | Meaning |
|------|---------|
| Redundancy | Information shared across all sources (MMI measure, Barrett 2015) |
| Unique_A | Information only source A carries |
| Unique_B | Information only source B carries |
| Synergy | Information available only from the whole system jointly |

Constraint: I_total = redundancy + unique_A + unique_B + synergy.

Also computes **transfer entropy** TE(A -> B) = I(A_past; B_future | B_past), measuring directional information flow.

**Redundancy measure:** Uses MMI (Minimum Mutual Information): I_min = min(I(A; future), I(B; future)).

### Partial Information Decomposition (`pid`)

Implements Williams & Beer (2010) framework for multi-source decomposition.

**I_min specific information:**

```
I_min(S1, S2, ...; T) = sum_t p(t) * min_i I_spec(S_i; t)
```

where I_spec(S; t) = D_KL(P(S|T=t) || P(S)) is the specific information source S provides about target outcome t.

Supports arbitrary numbers of sources (not limited to bipartite). The decomposition satisfies:

```
I_total = redundancy + sum(unique_i) + synergy
```

### Streaming phi (`streaming`)

Online estimation for time-series data (EEG, fMRI, BCI).

**Empirical TPM:** Built incrementally from observed state transitions. Normalized lazily at query time.

**Exponential forgetting:** Configurable factor lambda in (0, 1]. All transition counts are multiplied by lambda before each new observation, allowing adaptation to non-stationary dynamics.

**EWMA smoothing:** Exponentially weighted moving average of phi estimates with configurable alpha. Reduces noise in the phi trajectory.

**CUSUM change-point detection:** Cumulative sum control chart on phi deviations from the running mean. Fires when cumulative positive or negative deviation exceeds the threshold (default 3.0). Resets after detection.

**Variance tracking:** Online Welford's algorithm for running variance of phi estimates.

### PAC Bounds (`bounds`)

Provable confidence intervals for approximate phi.

**Spectral-Cheeger (deterministic):**
- Lower bound: Fiedler value lambda_2 / (2 * d_max) from the MI Laplacian.
- Upper bound: sqrt(2 * lambda_2) via Cheeger inequality.
- Confidence: 1.0 (deterministic, not probabilistic).

**Hoeffding concentration:**
- For k stochastic samples with observed range B:
  epsilon = B * sqrt(ln(2/delta) / (2k))
- Interval: [phi_hat - epsilon, phi_hat + epsilon] with probability >= 1 - delta.

**Empirical Bernstein (tighter for low variance):**
- Uses sample variance V instead of worst-case range:
  epsilon = sqrt(2V * ln(3/delta) / k) + 3B * ln(3/delta) / (3(k-1))
- Strictly tighter than Hoeffding when variance is small relative to range.

## REST API (pi.ruv.io)

### POST /v1/consciousness/compute

Compute consciousness metrics for a transition system.

**Request:**

```json
{
  "tpm": [0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0],
  "n": 4,
  "state": 0,
  "algorithm": "auto",
  "phi_threshold": 1e-6,
  "partition_mask": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tpm` | `f64[]` | yes | Flattened n x n row-major transition probability matrix |
| `n` | `usize` | yes | Number of states (must be power of 2, >= 2) |
| `state` | `usize` | yes | Current state index in [0, n) |
| `algorithm` | `string` | no | One of: `iit4_phi`, `ces`, `phi_id`, `pid`, `bounds`, `auto` (default: `auto`) |
| `phi_threshold` | `f64` | no | Minimum phi for CES distinctions (default: 1e-6) |
| `partition_mask` | `u64` | no | Bitmask for PhiID/PID source partition (default: split in half) |

**Auto-selection:** `auto` routes to `ces` for n <= 4 elements, `iit4_phi` otherwise.

**Response (common envelope):**

```json
{
  "algorithm": "ces",
  "phi": 0.234,
  "num_elements": 2,
  "num_states": 4,
  "elapsed_us": 142,
  "details": { ... }
}
```

**Algorithm-specific `details`:**

`iit4_phi`:
```json
{
  "phi_cause": 0.312,
  "phi_effect": 0.234,
  "mechanism_elements": 2
}
```

`ces`:
```json
{
  "big_phi": 0.118,
  "sum_phi": 0.546,
  "num_distinctions": 3,
  "num_relations": 2,
  "sum_relation_phi": 0.546,
  "distinctions": [
    { "mechanism": "11", "phi": 0.234, "phi_cause": 0.312, "phi_effect": 0.234 },
    { "mechanism": "1", "phi": 0.156, "phi_cause": 0.156, "phi_effect": 0.201 }
  ]
}
```

`phi_id`:
```json
{
  "total_mi": 0.451,
  "redundancy": 0.102,
  "unique": [0.123, 0.089],
  "synergy": 0.137,
  "transfer_entropy": 0.045
}
```

`pid`:
```json
{
  "redundancy": 0.102,
  "unique": [0.123, 0.089],
  "synergy": 0.137,
  "total_mi": 0.451,
  "num_sources": 2
}
```

`bounds`:
```json
{
  "lower_bound": 0.089,
  "upper_bound": 0.412,
  "confidence": 1.0,
  "samples": 0,
  "method": "spectral-cheeger"
}
```

**Error responses** return HTTP 400 with `{"error": "..."}`.

### GET /v1/consciousness/status

Returns subsystem capabilities. No authentication required.

```json
{
  "available": true,
  "version": "4.0",
  "framework": "IIT 4.0 (Albantakis et al. 2023)",
  "algorithms": [
    { "name": "iit4_phi", "description": "IIT 4.0 mechanism-level phi with intrinsic information (EMD)" },
    { "name": "ces", "description": "Full Cause-Effect Structure: distinctions, relations, big Phi" },
    { "name": "phi_id", "description": "Integrated Information Decomposition: redundancy, synergy, unique" },
    { "name": "pid", "description": "Partial Information Decomposition (Williams-Beer I_min)" },
    { "name": "streaming", "description": "Online streaming phi with EWMA, CUSUM change-point detection" },
    { "name": "bounds", "description": "PAC-style bounds: spectral-Cheeger, Hoeffding, empirical Bernstein" },
    { "name": "auto", "description": "Auto-select algorithm based on system size and budget" }
  ],
  "max_elements": 12,
  "max_states_exact": 4096,
  "features": [
    "intrinsic_difference_emd",
    "cause_effect_repertoires",
    "mechanism_partition_search",
    "relation_computation",
    "streaming_change_point",
    "confidence_intervals"
  ]
}
```

## MCP Tools

### brain_consciousness_compute

Proxies to `POST /v1/consciousness/compute`. Accepts the same parameters:

```json
{
  "tpm": [0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25, 0.0, 0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0],
  "n": 4,
  "state": 0,
  "algorithm": "ces"
}
```

Required fields: `tpm`, `n`, `state`. Optional: `algorithm`, `phi_threshold`, `partition_mask`.

### brain_consciousness_status

Proxies to `GET /v1/consciousness/status`. No parameters. Returns algorithm list and capabilities.

## Rust API

### Quick Start

```rust
use ruvector_consciousness::types::{TransitionMatrix, ComputeBudget, Mechanism};
use ruvector_consciousness::iit4::mechanism_phi;

// 4-state system (2 binary elements): AND gate
let tpm = TransitionMatrix::new(4, vec![
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.0, 0.0,  0.0,  1.0,
]);

// Mechanism = both elements (bitmask 0b11), 2 elements total
let mech = Mechanism::new(0b11, 2);
let dist = mechanism_phi(&tpm, &mech, 0);
println!("phi = {:.6}", dist.phi);
println!("phi_cause = {:.6}, phi_effect = {:.6}", dist.phi_cause, dist.phi_effect);
```

### CES Example

```rust
use ruvector_consciousness::types::{TransitionMatrix, ComputeBudget};
use ruvector_consciousness::ces::{compute_ces, ces_complexity};

let tpm = TransitionMatrix::new(4, vec![
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.0, 0.0,  0.0,  1.0,
]);

let budget = ComputeBudget::exact();
let ces = compute_ces(&tpm, 0, 1e-6, &budget).unwrap();

let (num_distinctions, num_relations, sum_phi) = ces_complexity(&ces);
println!("Big Phi = {:.6}", ces.big_phi);
println!("Distinctions: {}, Relations: {}, Sum phi: {:.6}",
    num_distinctions, num_relations, sum_phi);

for d in &ces.distinctions {
    println!("  mechanism {:0b}: phi={:.6} (cause={:.6}, effect={:.6})",
        d.mechanism.elements, d.phi, d.phi_cause, d.phi_effect);
}
```

### PhiID Example

```rust
use ruvector_consciousness::types::TransitionMatrix;
use ruvector_consciousness::phi_id::compute_phi_id;

let tpm = TransitionMatrix::new(4, vec![
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.0, 0.0,  0.0,  1.0,
]);

// Partition mask 0b0011: elements {0,1} vs {2,3}
let result = compute_phi_id(&tpm, 0b0011).unwrap();
println!("Total MI: {:.6}", result.total_mi);
println!("Redundancy: {:.6}", result.redundancy);
println!("Unique: {:?}", result.unique);
println!("Synergy: {:.6}", result.synergy);
println!("Transfer entropy: {:.6}", result.transfer_entropy);
```

### PID Example

```rust
use ruvector_consciousness::types::TransitionMatrix;
use ruvector_consciousness::pid::compute_pid;

let tpm = TransitionMatrix::new(4, vec![
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.0, 0.0,  0.0,  1.0,
]);

let sources = vec![vec![0], vec![1]];
let target = vec![0, 1];
let result = compute_pid(&tpm, &sources, &target).unwrap();

// Verify decomposition: redundancy + unique + synergy = total MI
let sum = result.redundancy + result.unique.iter().sum::<f64>() + result.synergy;
assert!((sum - result.total_mi).abs() < 1e-6);
```

### Streaming Example

```rust
use ruvector_consciousness::types::ComputeBudget;
use ruvector_consciousness::streaming::StreamingPhiEstimator;
use ruvector_consciousness::phi::SpectralPhiEngine;

let mut estimator = StreamingPhiEstimator::new(4)
    .with_forgetting_factor(0.99)   // slow forgetting
    .with_ewma_alpha(0.1)           // smooth phi trajectory
    .with_cusum_threshold(3.0);     // change-point sensitivity

let engine = SpectralPhiEngine::default();
let budget = ComputeBudget::fast();

// Feed observations from a time series
let observations = [0, 1, 3, 2, 0, 1, 3, 3, 0, 1, 2, 3];
for &state in &observations {
    if let Some(result) = estimator.observe(state, &engine, &budget) {
        println!("t={}: phi={:.4}, ewma={:.4}, var={:.6}, change={}",
            result.time_steps, result.phi, result.phi_ewma,
            result.phi_variance, result.change_detected);
    }
}
```

### Bounds Example

```rust
use ruvector_consciousness::types::{TransitionMatrix, ComputeBudget};
use ruvector_consciousness::bounds::{
    spectral_bounds, hoeffding_bound, empirical_bernstein_bound,
    compute_phi_with_bounds,
};
use ruvector_consciousness::phi::SpectralPhiEngine;

let tpm = TransitionMatrix::new(4, vec![
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.5, 0.25, 0.25, 0.0,
    0.0, 0.0,  0.0,  1.0,
]);

// Deterministic spectral bounds
let bound = spectral_bounds(&tpm).unwrap();
println!("Spectral: [{:.4}, {:.4}] (confidence {})",
    bound.lower, bound.upper, bound.confidence);

// Hoeffding bound from stochastic sampling
let hb = hoeffding_bound(0.5, 1000, 1.0, 0.05);
println!("Hoeffding 95%: [{:.4}, {:.4}]", hb.lower, hb.upper);

// Empirical Bernstein from phi samples
let samples = vec![0.3, 0.35, 0.32, 0.31, 0.33, 0.34, 0.30, 0.36];
let eb = empirical_bernstein_bound(&samples, 0.05);
println!("Bernstein 95%: [{:.4}, {:.4}]", eb.lower, eb.upper);

// Combined: run engine + attach bounds
let engine = SpectralPhiEngine::default();
let budget = ComputeBudget::fast();
let (result, bound) = compute_phi_with_bounds(&engine, &tpm, Some(0), &budget, 0.05).unwrap();
println!("Phi = {:.4}, bound = [{:.4}, {:.4}]", result.phi, bound.lower, bound.upper);
```

## Performance

| Optimization | Impact |
|-------------|--------|
| Single-pass `cause_repertoire` | O(n) vs O(n * purview_size) -- accumulates into purview buckets in one global-state sweep |
| Mirror-partition skip | 2x for bipartitions -- `BipartitionIter` enumerates [1, 2^n - 2), masks and complements are equivalent |
| Rayon parallel CES | 3-6x for n >= 5 elements -- mechanism enumeration parallelized across cores |
| Inline EMD + selectivity | Avoids allocation for distance computation -- cumulative sum in a single loop |
| Stack buffers | Small arrays (purview size <= 64) avoid heap allocation |
| Lazy TPM normalization | Streaming module normalizes counts only at query time, not on every observation |
| Zero-alloc arena | Bump allocator for temporary buffers in hot loops |
| SIMD-accelerated KL/entropy | AVX2 vectorized divergence and entropy for large distributions |

## Error Handling

All functions return `Result<T, ConsciousnessError>`. Error variants:

| Variant | Cause |
|---------|-------|
| `PhiNonConvergence` | Approximate algorithm did not converge within budget |
| `NumericalInstability` | NaN/Inf in matrix operations at a specific partition |
| `BudgetExhausted` | Time or partition limit exceeded |
| `InvalidInput` | Validation failure (dimension mismatch, non-finite values, invalid TPM rows) |
| `SystemTooLarge` | n > 12 elements for exact CES |

## Brain Categories

When storing consciousness results in pi.ruv.io shared memory:

- `consciousness` -- IIT 4.0 metrics, phi, CES, big Phi, distinctions, relations
- `information_decomposition` -- PhiID, PID, redundancy/synergy analysis

## References

- Albantakis, L., et al. (2023). "Integrated Information Theory (IIT) 4.0: Formulating the Properties of Phenomenal Existence in Physical Terms." *PLoS Computational Biology*.
- Mediano, P.A.M., et al. (2021). "Towards an Extended Taxonomy of Information Dynamics via Integrated Information Decomposition." *Physical Review E*.
- Williams, P.L. & Beer, R.D. (2010). "Nonnegative Decomposition of Multivariate Information." *arXiv:1004.2515*.
- Tononi, G. (2004). "An Information Integration Theory of Consciousness." *BMC Neuroscience*.
- Barrett, A.B. (2015). "Exploration of Synergistic and Redundant Information Sharing in Static and Dynamical Gaussian Systems." *Physical Review E*.
