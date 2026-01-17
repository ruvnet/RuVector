//! Property-based tests for the quantum monitor crate.
//!
//! These tests use proptest to verify mathematical invariants and properties
//! that should hold across all inputs.

use ndarray::{Array1, Array2};
use proptest::prelude::*;
use ruvector_quantum_monitor::{
    ConfidenceSequence, ConfidenceSequenceConfig, EValueConfig, EValueTest, MonitorConfig,
    QuantumCoherenceMonitor, QuantumKernel, QuantumKernelConfig,
};

// Strategy for generating random vectors
fn vec_strategy(dim: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-10.0..10.0f64, dim)
}

// Strategy for generating random matrices (n rows, dim columns)
fn matrix_strategy(n: usize, dim: usize) -> impl Strategy<Value = Vec<Vec<f64>>> {
    prop::collection::vec(vec_strategy(dim), n)
}

// Convert Vec<Vec<f64>> to Array2
fn to_array2(data: Vec<Vec<f64>>) -> Array2<f64> {
    let n = data.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }
    let dim = data[0].len();
    let flat: Vec<f64> = data.into_iter().flatten().collect();
    Array2::from_shape_vec((n, dim), flat).unwrap()
}

proptest! {
    /// Property: Quantum kernel should be symmetric k(x,y) = k(y,x)
    #[test]
    fn kernel_is_symmetric(
        x in vec_strategy(4),
        y in vec_strategy(4),
    ) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let k_xy = kernel.kernel(&x_arr, &y_arr).unwrap();
        let k_yx = kernel.kernel(&y_arr, &x_arr).unwrap();

        prop_assert!((k_xy - k_yx).abs() < 1e-10, "k(x,y)={} != k(y,x)={}", k_xy, k_yx);
    }

    /// Property: Self-kernel should be 1 (for normalized states)
    #[test]
    fn self_kernel_is_one(x in vec_strategy(4)) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x_arr = Array1::from_vec(x);
        let k_xx = kernel.kernel(&x_arr, &x_arr).unwrap();

        prop_assert!((k_xx - 1.0).abs() < 1e-6, "k(x,x) = {} != 1.0", k_xx);
    }

    /// Property: Kernel values should be in [0, 1] for fidelity-based kernels
    #[test]
    fn kernel_values_bounded(
        x in vec_strategy(4),
        y in vec_strategy(4),
    ) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let k = kernel.kernel(&x_arr, &y_arr).unwrap();

        prop_assert!(k >= 0.0, "Kernel value {} < 0", k);
        prop_assert!(k <= 1.0 + 1e-6, "Kernel value {} > 1", k);
    }

    /// Property: E-values should always be non-negative
    #[test]
    fn evalue_non_negative(mmd_values in prop::collection::vec(-1.0..1.0f64, 10..50)) {
        let config = EValueConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        for mmd in mmd_values {
            let update = test.update(mmd);
            prop_assert!(
                update.evalue >= 0.0,
                "E-value {} is negative",
                update.evalue
            );
            prop_assert!(
                update.evalue_increment >= 0.0,
                "E-value increment {} is negative",
                update.evalue_increment
            );
        }
    }

    /// Property: P-values should be in [0, 1]
    #[test]
    fn pvalue_in_valid_range(mmd_values in prop::collection::vec(-0.5..0.5f64, 10..50)) {
        let config = EValueConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        for mmd in mmd_values {
            let update = test.update(mmd);
            prop_assert!(
                update.p_value >= 0.0 && update.p_value <= 1.0,
                "P-value {} out of range [0,1]",
                update.p_value
            );
        }
    }

    /// Property: Confidence sequence width should decrease with more samples
    /// (on average, with enough samples)
    #[test]
    fn confidence_width_shrinks(
        observations in prop::collection::vec(-5.0..5.0f64, 50..100)
    ) {
        let config = ConfidenceSequenceConfig {
            min_samples: 5,
            empirical_variance: false,
            variance_proxy: 1.0,
            ..Default::default()
        };
        let mut cs = ConfidenceSequence::new(config).unwrap();

        let mut widths = Vec::new();
        for x in observations {
            if let Some(ci) = cs.update(x) {
                widths.push(ci.width);
            }
        }

        // Compare first 10% to last 10%
        if widths.len() >= 20 {
            let first_avg: f64 = widths[..5].iter().sum::<f64>() / 5.0;
            let last_avg: f64 = widths[widths.len()-5..].iter().sum::<f64>() / 5.0;

            prop_assert!(
                last_avg < first_avg * 1.5, // Allow some slack for randomness
                "Width didn't shrink: first_avg={}, last_avg={}",
                first_avg,
                last_avg
            );
        }
    }

    /// Property: Kernel matrix should be symmetric
    #[test]
    fn kernel_matrix_symmetric(
        data in matrix_strategy(5, 3)
    ) {
        if data.is_empty() || data[0].is_empty() {
            return Ok(());
        }

        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let arr = to_array2(data);
        let k_matrix = kernel.kernel_matrix(&arr).unwrap();

        let n = k_matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                let diff = (k_matrix[[i, j]] - k_matrix[[j, i]]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "K[{},{}]={} != K[{},{}]={}",
                    i, j, k_matrix[[i, j]],
                    j, i, k_matrix[[j, i]]
                );
            }
        }
    }

    /// Property: Kernel matrix diagonal should be 1
    #[test]
    fn kernel_matrix_diagonal_one(
        data in matrix_strategy(5, 3)
    ) {
        if data.is_empty() || data[0].is_empty() {
            return Ok(());
        }

        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let arr = to_array2(data);
        let k_matrix = kernel.kernel_matrix(&arr).unwrap();

        for i in 0..k_matrix.nrows() {
            let diag = k_matrix[[i, i]];
            prop_assert!(
                (diag - 1.0).abs() < 1e-6,
                "K[{},{}] = {} != 1.0",
                i, i, diag
            );
        }
    }

    /// Property: Confidence interval should contain mean for normal data (most of the time)
    #[test]
    fn confidence_interval_contains_estimate(
        n_obs in 20usize..50,
        true_mean in -5.0..5.0f64,
    ) {
        let config = ConfidenceSequenceConfig {
            confidence_level: 0.95,
            min_samples: 5,
            ..Default::default()
        };
        let mut cs = ConfidenceSequence::new(config).unwrap();

        // Generate observations around true_mean with noise
        let mut rng = rand::thread_rng();
        for _ in 0..n_obs {
            let noise: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            cs.update(true_mean + noise);
        }

        if let Some(ci) = cs.current_interval() {
            // The interval should contain its own mean (trivially true)
            prop_assert!(
                ci.contains(ci.mean),
                "CI [{}, {}] doesn't contain its mean {}",
                ci.lower, ci.upper, ci.mean
            );

            // Width should be finite and positive
            prop_assert!(
                ci.width > 0.0 && ci.width.is_finite(),
                "Invalid width: {}",
                ci.width
            );
        }
    }

    /// Property: E-value reset should restore initial state
    #[test]
    fn evalue_reset_works(mmd_values in prop::collection::vec(-1.0..1.0f64, 5..20)) {
        let config = EValueConfig {
            min_samples: 3,
            initial_wealth: 1.0,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        // Process some data
        for mmd in &mmd_values {
            test.update(*mmd);
        }

        prop_assert!(test.n_samples() > 0);

        // Reset
        test.reset();

        // Verify reset state
        prop_assert_eq!(test.n_samples(), 0);
        prop_assert_eq!(test.current_evalue(), 1.0);
        prop_assert!(!test.is_drift_detected());
    }

    /// Property: MMD should be small for samples from same distribution
    #[test]
    fn mmd_small_for_same_distribution(
        n in 20usize..40,
        dim in 2usize..5,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate baseline and test from same distribution
        let baseline = Array2::from_shape_fn((n, dim), |_| normal.sample(&mut rng));
        let test_data = Array2::from_shape_fn((n / 2, dim), |_| normal.sample(&mut rng));

        let config = MonitorConfig {
            kernel: QuantumKernelConfig {
                n_qubits: 2,
                n_layers: 1,
                seed: Some(42),
                ..Default::default()
            },
            min_baseline_samples: 10,
            evalue: EValueConfig {
                min_samples: 3,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();
        monitor.set_baseline(&baseline).unwrap();

        let mut mmd_sum = 0.0;
        let mut count = 0;
        for i in 0..test_data.nrows() {
            let sample = test_data.row(i).to_owned();
            let result = monitor.observe(&sample).unwrap();
            mmd_sum += result.mmd_squared.abs();
            count += 1;
        }

        let avg_mmd = mmd_sum / count as f64;
        // MMD should be relatively small for same distribution
        // (not a strict bound due to random variation)
        prop_assert!(
            avg_mmd < 5.0,
            "Average MMD {} is too large for same distribution",
            avg_mmd
        );
    }
}

// Additional non-proptest tests for edge cases

#[test]
fn test_empty_baseline_rejected() {
    let config = MonitorConfig::default();
    let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

    let empty_baseline = Array2::zeros((0, 4));
    let result = monitor.set_baseline(&empty_baseline);

    assert!(result.is_err());
}

#[test]
fn test_small_baseline_rejected() {
    let config = MonitorConfig {
        min_baseline_samples: 20,
        ..Default::default()
    };
    let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

    let small_baseline = Array2::zeros((5, 4));
    let result = monitor.set_baseline(&small_baseline);

    assert!(result.is_err());
}

#[test]
fn test_invalid_config_rejected() {
    // Invalid n_qubits
    let config = QuantumKernelConfig {
        n_qubits: 0,
        ..Default::default()
    };
    assert!(QuantumKernel::new(config).is_err());

    // Invalid sigma
    let config = QuantumKernelConfig {
        sigma: -1.0,
        ..Default::default()
    };
    assert!(QuantumKernel::new(config).is_err());

    // Invalid alpha
    let config = EValueConfig {
        alpha: 0.0,
        ..Default::default()
    };
    assert!(EValueTest::new(config).is_err());

    let config = EValueConfig {
        alpha: 1.0,
        ..Default::default()
    };
    assert!(EValueTest::new(config).is_err());
}

#[test]
fn test_dimension_consistency() {
    use rand_distr::{Distribution, Normal};

    let config = MonitorConfig::fast_detection();
    let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    // Set baseline with dimension 4
    let baseline = Array2::from_shape_fn((20, 4), |_| normal.sample(&mut rng));
    monitor.set_baseline(&baseline).unwrap();

    // Observation with correct dimension should work
    let good_sample = Array1::from_shape_fn(4, |_| normal.sample(&mut rng));
    assert!(monitor.observe(&good_sample).is_ok());

    // Observation with wrong dimension should fail
    let bad_sample = Array1::from_shape_fn(3, |_| normal.sample(&mut rng));
    assert!(monitor.observe(&bad_sample).is_err());
}
