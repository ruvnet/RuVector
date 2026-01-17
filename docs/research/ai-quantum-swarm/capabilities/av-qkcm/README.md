# AV-QKCM: Anytime-Valid Quantum Kernel Coherence Monitor

> Quantum kernel-based monitoring with anytime-valid statistical guarantees

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | Tier 1 (Immediate) |
| **Score** | 90/100 |
| **Integration** | ruQu (e-value framework), cognitum-gate |
| **Proposed Crate** | `ruvector-quantum-monitor` |

## Problem Statement

Current coherence monitoring:
1. Uses fixed-window statistics that can miss gradual drift
2. Lacks rigorous statistical guarantees for streaming data
3. Classical kernels may miss quantum-specific correlations

## Solution

Anytime-valid monitoring using:
1. Quantum kernel MMD (Maximum Mean Discrepancy) for distribution comparison
2. E-value based sequential testing (integrates with ruQu)
3. Confidence sequences that are valid at any stopping time

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AV-QKCM Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Syndrome    ┌──────────────┐    ┌──────────────┐               │
│  Stream ────►│ Feature      │───►│ Quantum      │               │
│              │ Extraction   │    │ Kernel       │               │
│              └──────────────┘    └──────┬───────┘               │
│                                         │                        │
│  Reference   ┌──────────────┐          │                        │
│  Distribution│ Baseline     │──────────┤                        │
│              │ Kernel       │          │                        │
│              └──────────────┘          ▼                        │
│                                  ┌──────────────┐               │
│                                  │ Kernel MMD   │               │
│                                  │ Statistic    │               │
│                                  └──────┬───────┘               │
│                                         │                        │
│                                         ▼                        │
│                                  ┌──────────────┐    ┌────────┐ │
│                                  │ E-Value      │───►│ Drift  │ │
│                                  │ Accumulator  │    │ Alert  │ │
│                                  └──────────────┘    └────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Quantum Kernel for Syndrome Distributions

Use variational quantum circuits to compute kernel similarity:

```rust
/// Quantum kernel: k(x,y) = |⟨ψ(x)|ψ(y)⟩|²
pub struct QuantumKernel {
    /// Parameterized quantum circuit
    feature_map: FeatureMapCircuit,

    /// Number of qubits
    n_qubits: usize,

    /// Backend (simulator or hardware)
    backend: QuantumBackend,
}

impl QuantumKernel {
    /// Compute kernel matrix for batch of syndromes
    pub fn kernel_matrix(&self, x: &[SyndromeFeatures], y: &[SyndromeFeatures]) -> KernelMatrix {
        let mut K = Array2::zeros((x.len(), y.len()));

        for (i, xi) in x.iter().enumerate() {
            for (j, yj) in y.iter().enumerate() {
                // Encode features into circuit
                let psi_x = self.feature_map.encode(xi);
                let psi_y = self.feature_map.encode(yj);

                // Compute fidelity via swap test or direct overlap
                K[(i, j)] = self.backend.fidelity(&psi_x, &psi_y);
            }
        }

        KernelMatrix(K)
    }

    /// Streaming kernel for online monitoring
    pub fn incremental_kernel(&self, new_point: &SyndromeFeatures) -> IncrementalKernelUpdate {
        // Only compute new row/column, not full matrix
        // O(n) instead of O(n²)
    }
}
```

### 2. Anytime-Valid E-Value Testing

Integrates directly with ruQu's e-value framework:

```rust
/// E-value based sequential test for distribution shift
pub struct SequentialMMDTest {
    /// Baseline kernel mean embedding
    baseline_embedding: KernelMeanEmbedding,

    /// Running e-value (from ruQu)
    e_value: ruqu::EValueAccumulator,

    /// Confidence sequence
    confidence_seq: ConfidenceSequence,
}

impl SequentialMMDTest {
    /// Update with new observation
    pub fn update(&mut self, syndrome: &SyndromeFeatures) -> MonitorResult {
        // 1. Compute kernel distance to baseline
        let mmd_stat = self.compute_mmd(syndrome);

        // 2. Convert to likelihood ratio for e-value
        let likelihood_ratio = self.mmd_to_lr(mmd_stat);

        // 3. Update e-value (multiplicative)
        self.e_value.accumulate(likelihood_ratio);

        // 4. Update confidence sequence
        self.confidence_seq.update(mmd_stat);

        // 5. Check for drift
        MonitorResult {
            e_value: self.e_value.current(),
            confidence_interval: self.confidence_seq.interval(),
            drift_detected: self.e_value.current() > self.threshold,
            drift_magnitude: self.confidence_seq.lower_bound(),
        }
    }

    /// Key property: valid at ANY stopping time
    pub fn is_anytime_valid(&self) -> bool {
        // E-values are always valid: E[e-value] ≤ 1 under null
        true
    }
}
```

### 3. Confidence Sequences

Unlike confidence intervals, valid for continuous monitoring:

```rust
/// Confidence sequence: Pr(θ ∈ CI_t for all t) ≥ 1-α
pub struct ConfidenceSequence {
    /// Running mean
    mean: f64,
    /// Running variance
    variance: f64,
    /// Sample count
    n: usize,
    /// Confidence level
    alpha: f64,
}

impl ConfidenceSequence {
    /// Width shrinks as O(√(log(n)/n)) - slower than CLT but always valid
    pub fn interval(&self) -> (f64, f64) {
        // Using mixture martingale approach
        let width = self.hedged_ci_width();
        (self.mean - width, self.mean + width)
    }

    fn hedged_ci_width(&self) -> f64 {
        // From "Time-uniform, nonparametric, nonasymptotic confidence sequences"
        // Howard et al. (2021)
        let rho = 1.0 / (self.n as f64 + 1.0);
        let log_term = (2.0 / self.alpha).ln() + (1.0 + self.n as f64).ln().ln();

        (2.0 * self.variance * log_term / self.n as f64).sqrt()
            + rho * log_term / (3.0 * self.n as f64)
    }
}
```

## Integration with RuVector

### ruQu Integration

```rust
// In ruqu crate
impl QuantumFabric {
    pub fn with_quantum_monitor(mut self, monitor: QuantumCoherenceMonitor) -> Self {
        self.monitor = Some(monitor);
        self
    }

    pub fn process_cycle_monitored(&mut self, syndrome: &SyndromeRound) -> MonitoredDecision {
        // 1. Standard coherence assessment
        let coherence = self.assess_coherence(syndrome);

        // 2. Quantum kernel monitoring
        let monitor_result = self.monitor.as_mut()
            .map(|m| m.update(&syndrome.features()));

        // 3. Fuse decisions
        MonitoredDecision {
            gate: coherence.decision,
            drift_alert: monitor_result.map(|r| r.drift_detected).unwrap_or(false),
            e_value: monitor_result.map(|r| r.e_value),
            confidence_interval: monitor_result.map(|r| r.confidence_interval),
        }
    }
}

// E-value accumulator shared with existing ruQu infrastructure
use ruqu::evidence::EValueAccumulator;
```

### cognitum-gate Integration

```rust
// Extend TileZero with quantum monitoring
impl TileZero {
    pub fn with_distribution_monitor(mut self, monitor: QuantumCoherenceMonitor) -> Self {
        self.distribution_monitor = Some(monitor);
        self
    }

    /// Enhanced decision with drift awareness
    pub async fn decide_with_monitoring(&self, action: &ActionContext) -> EnhancedToken {
        let base_token = self.decide(action).await;

        // Check for distribution drift
        if let Some(ref monitor) = self.distribution_monitor {
            let drift_status = monitor.drift_status();

            if drift_status.drift_detected {
                // Elevate Permit to Defer if drift detected
                return EnhancedToken {
                    decision: match base_token.decision {
                        GateDecision::Permit => GateDecision::Defer,
                        other => other,
                    },
                    drift_warning: true,
                    drift_magnitude: drift_status.magnitude,
                    ..base_token
                };
            }
        }

        EnhancedToken::from(base_token)
    }
}
```

## Research Tasks

- [ ] Literature review: Anytime-valid inference, quantum kernels, MMD
- [ ] Design feature map circuit for syndrome encoding
- [ ] Implement confidence sequences in Rust
- [ ] Benchmark quantum vs classical kernels on drift detection
- [ ] Integration tests with ruQu e-value framework
- [ ] Calibrate thresholds for false positive/negative rates
- [ ] Real-time performance optimization

## References

1. [Time-uniform confidence sequences](https://arxiv.org/abs/1906.09712) - Howard et al.
2. [Quantum Kernels for Real-World Predictions](https://arxiv.org/abs/2111.03474)
3. [E-values: Calibration, combination, and applications](https://arxiv.org/abs/1912.06116)
4. [arXiv:2511.09491 - Window-based drift detection](https://arxiv.org/abs/2511.09491)
