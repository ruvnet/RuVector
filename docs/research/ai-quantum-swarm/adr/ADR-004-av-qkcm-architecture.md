# ADR-004: Anytime-Valid Quantum Kernel Coherence Monitor (AV-QKCM) Architecture

## Status
Accepted

## Date
2026-01-17

## Context

The AI-Quantum Swarm system requires real-time monitoring of quantum coherence for trust decisions
in distributed tile-based architectures. Traditional statistical tests suffer from two critical
limitations:

1. **Fixed sample sizes**: Classical hypothesis tests require pre-specified sample sizes,
   making them unsuitable for streaming data where decisions may need to be made at any time.

2. **Multiple testing issues**: Repeated testing on accumulating data leads to inflated Type I
   error rates (alpha spending problem).

We need a monitoring system that:
- Provides valid statistical inference at any stopping time
- Detects distribution drift in quantum syndrome patterns
- Integrates with ruQu's evidence framework and cognitum-gate-tilezero
- Operates efficiently in streaming settings with O(1) memory per observation

## Decision

We implement the **Anytime-Valid Quantum Kernel Coherence Monitor (AV-QKCM)** as a new crate
`ruvector-quantum-monitor` with the following architecture:

### Core Components

```
ruvector-quantum-monitor/
  src/
    kernel.rs      # Quantum feature maps and kernel computation
    evalue.rs      # E-value accumulation and sequential testing
    confidence.rs  # Confidence sequences (time-uniform intervals)
    monitor.rs     # Main QuantumCoherenceMonitor interface
    error.rs       # Error types
    lib.rs         # Public API and re-exports
```

### Mathematical Foundation

#### 1. Quantum Kernel

We use a simulated quantum kernel based on parameterized quantum circuits:

```
k(x, y) = |<phi(x)|phi(y)>|^2
```

where `|phi(x)>` is the quantum state produced by encoding classical data `x` through
angle encoding and variational rotations:

```
|phi(x)> = U_var(theta) * U_enc(x) |0>^n
```

This provides an expressive kernel that captures complex nonlinear relationships.

#### 2. Maximum Mean Discrepancy (MMD)

For two distributions P (baseline) and Q (streaming), the squared MMD is:

```
MMD^2(P, Q) = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
```

Under H_0: P = Q, MMD^2 = 0. Under H_1: P != Q, MMD^2 > 0.

#### 3. E-Value Sequential Testing

E-values provide anytime-valid inference. An e-value E_t satisfies:

```
E_0[E_t] <= 1    (for all stopping times tau)
```

We construct e-values using the betting martingale approach (Shekhar & Ramdas, 2023):

```
E_t = prod_{i=1}^{t} (1 + lambda_i * h_i)
```

where h_i is a centered statistic based on MMD and lambda_i is an adaptive betting fraction.

By Ville's inequality:

```
P_0(exists t: E_t >= 1/alpha) <= alpha
```

This allows valid p-values at any stopping time: p_t = min(1, 1/E_t).

#### 4. Confidence Sequences

Following Howard et al. (2021), we construct time-uniform confidence intervals:

```
C_t = [mu_hat_t - width_t, mu_hat_t + width_t]
```

where:

```
width_t = sqrt(2 * sigma^2 * (t + rho) / t * log((t + rho) / (rho * alpha^2)))
```

These achieve the optimal O(sqrt(log(n)/n)) asymptotic width while maintaining
time-uniform coverage: P(forall t: mu in C_t) >= 1 - alpha.

### Integration Architecture

```
                    +-------------------+
                    |  Streaming Data   |
                    +--------+----------+
                             |
                             v
+----------------+   +-------+--------+   +------------------+
| QuantumKernel  |<->| MMD Estimator  |<->| StreamingAccum   |
| (feature maps) |   | (U-statistic)  |   | (incremental)    |
+----------------+   +-------+--------+   +------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
      +-------+-------+            +--------+---------+
      | EValueTest    |            | ConfidenceSeq    |
      | (betting      |            | (time-uniform    |
      |  martingale)  |            |  intervals)      |
      +-------+-------+            +--------+---------+
              |                             |
              +--------------+--------------+
                             |
                             v
              +--------------+--------------+
              | QuantumCoherenceMonitor     |
              | - set_baseline()            |
              | - observe()                 |
              | - status()                  |
              +--------------+--------------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
    +-----+------+    +------+------+    +------+-----+
    | ruQu       |    | tilezero    |    | Alerts     |
    | Evidence   |    | Trust Gate  |    | & Actions  |
    +------------+    +-------------+    +------------+
```

### Key Design Decisions

1. **Streaming-First**: The monitor maintains O(1) memory per observation through:
   - Incremental kernel updates (StreamingKernelAccumulator)
   - Running sufficient statistics for MMD
   - No storage of historical samples required

2. **Anytime Validity**: All statistical outputs are valid at any stopping time:
   - E-values satisfy the martingale property
   - P-values are always upper bounds on Type I error
   - Confidence intervals have guaranteed coverage

3. **Adaptive Betting**: The betting fraction lambda adapts based on observed variance
   using a variant of the Kelly criterion, balancing detection power and stability.

4. **Thread Safety**: SharedMonitor provides a thread-safe wrapper using parking_lot::RwLock
   for concurrent access in multi-agent systems.

### State Machine

```
          set_baseline()
               |
               v
+-------------+-------------+
|       UNINITIALIZED       |
+-------------+-------------+
               |
               | (baseline set)
               v
+-------------+-------------+
|        MONITORING         |<----+
+-------------+-------------+     |
               |                  |
               | (drift detected) | (cooldown expires)
               v                  |
+-------------+-------------+     |
|      DRIFT_DETECTED       |-----+
+-------------+-------------+
               |
               | (samples continue)
               v
+-------------+-------------+
|         COOLDOWN          |------> MONITORING
+------------+--------------+
```

### Configuration

```rust
MonitorConfig {
    kernel: QuantumKernelConfig {
        n_qubits: 4,          // Quantum circuit size
        n_layers: 2,          // Variational circuit depth
        sigma: 1.0,           // Bandwidth parameter
        use_entanglement: true,
    },
    evalue: EValueConfig {
        alpha: 0.05,          // Significance level
        bet_fraction: 0.5,    // Initial betting fraction
        adaptive_betting: true,
    },
    confidence: ConfidenceSequenceConfig {
        confidence_level: 0.95,
        rho: 1.0,             // Intrinsic time offset
        empirical_variance: true,
    },
    min_baseline_samples: 20,
    alert_cooldown: 100,      // Samples between alerts
}
```

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| set_baseline(n) | O(n^2) | Pre-compute baseline kernel matrix |
| observe(1) | O(n) | n = baseline size, incremental update |
| observe_batch(m) | O(m*n) | Amortized per-sample cost |
| Memory | O(n^2 + d*2^q) | Baseline kernel + feature dim |

where:
- n = baseline sample size
- m = batch size
- d = data dimension
- q = number of qubits

## Consequences

### Positive

1. **Statistical Rigor**: Anytime-valid guarantees prevent p-hacking and allow
   flexible stopping rules while maintaining error control.

2. **Streaming Efficiency**: O(1) per-observation memory enables continuous
   monitoring of high-throughput data streams.

3. **Integration Ready**: Direct integration with ruQu evidence framework and
   cognitum-gate-tilezero for tile-level trust decisions.

4. **Expressive Kernels**: Quantum-inspired feature maps capture complex
   distribution differences that linear methods would miss.

### Negative

1. **Computational Cost**: Quantum kernel computation has O(2^q) scaling with
   qubit count, limiting to q <= 10 in practice.

2. **Baseline Requirement**: Requires sufficient baseline samples to establish
   reference distribution accurately.

3. **Complexity**: The mathematical framework (e-values, martingales, confidence
   sequences) adds conceptual complexity compared to simple threshold-based monitoring.

### Neutral

1. **Classical Simulation**: Uses classical simulation of quantum circuits rather
   than actual quantum hardware, providing reproducibility at the cost of not
   leveraging potential quantum advantages.

## References

1. Ramdas, A., et al. (2023). "Game-Theoretic Statistics and Safe Anytime-Valid Inference"
   Statistical Science.

2. Howard, S.R., et al. (2021). "Time-uniform, nonparametric, nonasymptotic confidence
   sequences" Annals of Statistics.

3. Shekhar, S., & Ramdas, A. (2023). "Nonparametric Two-Sample Testing by Betting"
   NeurIPS.

4. Gretton, A., et al. (2012). "A Kernel Two-Sample Test" JMLR.

5. Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces"
   Physical Review Letters.

## Implementation Status

- [x] Core kernel module with quantum feature maps
- [x] E-value accumulator and sequential test
- [x] Confidence sequences with Howard et al. bounds
- [x] Main QuantumCoherenceMonitor interface
- [x] Thread-safe SharedMonitor wrapper
- [ ] Integration with ruQu EvidenceAccumulator (feature-gated)
- [ ] Integration with cognitum-gate-tilezero (feature-gated)
- [ ] Benchmarks comparing detection latency vs. threshold methods
- [ ] WASM bindings for browser-based monitoring
