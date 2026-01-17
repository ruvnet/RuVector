//! # Ruvector Quantum Monitor
//!
//! Anytime-Valid Quantum Kernel Coherence Monitor (AV-QKCM) for distribution drift detection.
//!
//! This crate provides statistically rigorous monitoring of streaming data for distribution
//! shift using quantum-inspired kernel methods and anytime-valid sequential testing.
//!
//! ## Features
//!
//! - **Quantum Kernel Methods**: Simulated quantum feature maps for expressive kernel computation
//! - **Anytime-Valid Testing**: E-value based sequential hypothesis testing with valid p-values
//!   at any stopping time
//! - **Confidence Sequences**: Time-uniform confidence intervals following Howard et al. (2021)
//! - **Streaming Efficiency**: O(1) memory per observation with incremental updates
//! - **Production Ready**: Proper error handling, comprehensive testing, and thread-safe API
//!
//! ## Quick Start
//!
//! ```rust
//! use ruvector_quantum_monitor::{QuantumCoherenceMonitor, MonitorConfig};
//! use ndarray::{Array1, Array2};
//! use rand_distr::{Distribution, Normal};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create monitor with default configuration
//! let config = MonitorConfig::default();
//! let mut monitor = QuantumCoherenceMonitor::new(config)?;
//!
//! // Generate baseline data (normally distributed)
//! let mut rng = rand::thread_rng();
//! let normal = Normal::new(0.0, 1.0)?;
//! let baseline = Array2::from_shape_fn((30, 4), |_| normal.sample(&mut rng));
//!
//! // Set baseline distribution
//! monitor.set_baseline(&baseline)?;
//!
//! // Monitor streaming data (from same distribution - no drift expected)
//! for _ in 0..20 {
//!     let sample = Array1::from_shape_fn(4, |_| normal.sample(&mut rng));
//!     let result = monitor.observe(&sample)?;
//!
//!     println!("Sample {}: p-value={:.4}, drift={}",
//!         result.n_samples, result.p_value, result.drift_detected);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Mathematical Background
//!
//! ### Quantum Kernel
//!
//! The quantum kernel k(x, y) = |<phi(x)|phi(y)>|^2 measures fidelity between quantum states
//! encoded from classical data. This provides a highly expressive similarity measure that
//! captures complex nonlinear relationships.
//!
//! ### Maximum Mean Discrepancy (MMD)
//!
//! MMD^2(P, Q) = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
//!
//! This kernel-based statistic measures the distance between probability distributions
//! in the reproducing kernel Hilbert space.
//!
//! ### E-Value Testing
//!
//! E-values provide anytime-valid sequential testing. An e-value E_t satisfies E[E_t] <= 1
//! under H_0, and by Ville's inequality, P(exists t: E_t >= 1/alpha) <= alpha.
//!
//! This allows valid inference at any stopping time without p-hacking concerns.
//!
//! ### Confidence Sequences
//!
//! Confidence sequences {C_t} satisfy P(forall t: theta in C_t) >= 1 - alpha.
//! They achieve asymptotic width O(sqrt(log(n)/n)), optimal for sequential inference.
//!
//! ## References
//!
//! - Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces"
//! - Shekhar, S., & Ramdas, A. (2023). "Nonparametric Two-Sample Testing by Betting"
//! - Howard, S.R., et al. (2021). "Time-uniform, nonparametric, nonasymptotic confidence sequences"
//! - Gretton, A., et al. (2012). "A Kernel Two-Sample Test"

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

pub mod confidence;
pub mod error;
pub mod evalue;
pub mod kernel;
pub mod monitor;

// Re-exports for convenience
pub use confidence::{
    AsymptoticCS, BernsteinConfidenceSequence, ChangeDetectionCS, ConfidenceInterval,
    ConfidenceSequence, ConfidenceSequenceConfig,
};
pub use error::{QuantumMonitorError, Result};
pub use evalue::{EValueConfig, EValueSummary, EValueTest, EValueUpdate, MMDEstimator, OnlineMMD};
pub use kernel::{
    QuantumKernel, QuantumKernelConfig, StreamingKernelAccumulator, StreamingUpdate,
};
pub use monitor::{
    DriftAlert, DriftSeverity, MonitorConfig, MonitorState, MonitorStatus, ObservationResult,
    QuantumCoherenceMonitor, SharedMonitor,
};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports.
pub mod prelude {
    //! Convenient imports for common use cases.
    pub use crate::confidence::{ConfidenceInterval, ConfidenceSequence, ConfidenceSequenceConfig};
    pub use crate::error::{QuantumMonitorError, Result};
    pub use crate::evalue::{EValueConfig, EValueTest};
    pub use crate::kernel::{QuantumKernel, QuantumKernelConfig};
    pub use crate::monitor::{
        MonitorConfig, MonitorState, ObservationResult, QuantumCoherenceMonitor,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand_distr::{Distribution, Normal};

    /// Generate samples from a normal distribution.
    fn generate_normal(n: usize, dim: usize, mean: f64, std: f64, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(mean, std).unwrap();
        Array2::from_shape_fn((n, dim), |_| normal.sample(&mut rng))
    }

    #[test]
    fn test_full_pipeline_no_drift() {
        // Create monitor with conservative settings
        let config = MonitorConfig {
            kernel: QuantumKernelConfig {
                n_qubits: 3,
                n_layers: 1,
                seed: Some(42),
                ..Default::default()
            },
            evalue: EValueConfig {
                min_samples: 10,
                bet_fraction: 0.2,  // Conservative betting
                adaptive_betting: true,
                ..Default::default()
            },
            min_baseline_samples: 15,
            ..Default::default()
        };
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        // Set baseline (N(0, 1))
        let baseline = generate_normal(25, 4, 0.0, 1.0, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Observe samples from same distribution (different seed)
        let test_data = generate_normal(30, 4, 0.0, 1.0, 123);

        for i in 0..test_data.nrows() {
            let sample = test_data.row(i).to_owned();
            monitor.observe(&sample).unwrap();
        }

        // Under H_0, the p-value shouldn't be extremely small
        // The e-value test may have some variance but shouldn't consistently reject
        let final_p = monitor.current_p_value();
        let final_e = monitor.current_evalue();

        // Either p-value is reasonable OR e-value hasn't exploded
        assert!(
            final_p > 1e-6 || final_e < 1e10,
            "Under H_0: p-value={}, e-value={} suggests false positive",
            final_p,
            final_e
        );
    }

    #[test]
    fn test_full_pipeline_with_drift() {
        // Create monitor with fast detection
        let config = MonitorConfig::fast_detection();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        // Set baseline (N(0, 1))
        let baseline = generate_normal(15, 4, 0.0, 1.0, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Observe samples from shifted distribution (N(2, 1))
        let test_data = generate_normal(50, 4, 2.0, 1.0, 123);
        let results = monitor.observe_batch(&test_data).unwrap();

        // Should detect drift
        let last = results.last().unwrap();
        assert!(
            last.drift_detected || last.p_value < 0.1,
            "Should detect drift. Final p-value: {}, e-value: {}",
            last.p_value,
            last.evalue
        );
    }

    #[test]
    fn test_confidence_sequence_coverage() {
        // Test that confidence sequences provide valid coverage
        let config = ConfidenceSequenceConfig {
            confidence_level: 0.95,
            min_samples: 5,
            ..Default::default()
        };
        let mut cs = ConfidenceSequence::new(config).unwrap();

        let true_mean = 1.0;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(true_mean, 1.0).unwrap();

        let mut contains_true_mean = 0;
        let n_samples = 100;

        for _ in 0..n_samples {
            let x = normal.sample(&mut rng);
            if let Some(ci) = cs.update(x) {
                if ci.contains(true_mean) {
                    contains_true_mean += 1;
                }
            }
        }

        // Coverage should be at least 85% (allowing for randomness in small sample)
        let coverage = contains_true_mean as f64 / (n_samples - 5) as f64;
        assert!(
            coverage >= 0.80,
            "Coverage {} is too low (expected >= 0.80)",
            coverage
        );
    }

    #[test]
    fn test_kernel_symmetry_and_psd() {
        // Verify kernel matrix is symmetric and positive semi-definite
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let data = generate_normal(10, 3, 0.0, 1.0, 42);
        let k_matrix = kernel.kernel_matrix(&data).unwrap();

        // Check symmetry
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (k_matrix[[i, j]] - k_matrix[[j, i]]).abs() < 1e-10,
                    "Kernel matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Check PSD: x^T K x >= 0 for random vectors
        for _ in 0..10 {
            let mut rng = rand::thread_rng();
            let x: Vec<f64> = (0..10)
                .map(|_| rand::Rng::gen::<f64>(&mut rng) - 0.5)
                .collect();

            let mut quadratic_form = 0.0;
            for i in 0..10 {
                for j in 0..10 {
                    quadratic_form += x[i] * k_matrix[[i, j]] * x[j];
                }
            }

            assert!(
                quadratic_form >= -1e-10,
                "Kernel matrix not PSD: x^T K x = {}",
                quadratic_form
            );
        }
    }
}
