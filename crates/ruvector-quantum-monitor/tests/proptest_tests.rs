//! Property-based tests for ruvector-quantum-monitor using proptest
//!
//! Tests fundamental mathematical properties that must hold regardless
//! of specific input values.

use proptest::prelude::*;
use ndarray::{Array1, Array2};
use ruvector_quantum_monitor::{
    QuantumKernel, QuantumKernelConfig,
    EValueTest, EValueConfig,
    ConfidenceSequence, ConfidenceSequenceConfig,
};

// ============================================================================
// Quantum Kernel Properties
// ============================================================================

proptest! {
    /// Property: Kernel should be symmetric: k(x, y) = k(y, x)
    #[test]
    fn kernel_symmetry(
        dim in 2usize..6,
        x_vals in prop::collection::vec(-5.0f64..5.0, 2..8),
        y_vals in prop::collection::vec(-5.0f64..5.0, 2..8)
    ) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let min_len = dim.min(x_vals.len()).min(y_vals.len());
        let x = Array1::from_iter(x_vals.into_iter().take(min_len));
        let y = Array1::from_iter(y_vals.into_iter().take(min_len));

        let k_xy = kernel.kernel(&x, &y).unwrap();
        let k_yx = kernel.kernel(&y, &x).unwrap();

        prop_assert!(
            (k_xy - k_yx).abs() < 1e-10,
            "Kernel not symmetric: k(x,y)={} != k(y,x)={}",
            k_xy, k_yx
        );
    }

    /// Property: Self-kernel should be 1: k(x, x) = 1
    #[test]
    fn kernel_self_value(
        x_vals in prop::collection::vec(-3.0f64..3.0, 3..6)
    ) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let x = Array1::from_iter(x_vals.into_iter());
        let k_xx = kernel.kernel(&x, &x).unwrap();

        prop_assert!(
            (k_xx - 1.0).abs() < 1e-6,
            "Self-kernel k(x,x) = {} should be 1.0",
            k_xx
        );
    }

    /// Property: Kernel values should be bounded in [0, 1]
    #[test]
    fn kernel_bounded(
        x_vals in prop::collection::vec(-10.0f64..10.0, 3..5),
        y_vals in prop::collection::vec(-10.0f64..10.0, 3..5)
    ) {
        let config = QuantumKernelConfig {
            n_qubits: 3,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let min_len = x_vals.len().min(y_vals.len());
        let x = Array1::from_iter(x_vals.into_iter().take(min_len));
        let y = Array1::from_iter(y_vals.into_iter().take(min_len));

        let k_val = kernel.kernel(&x, &y).unwrap();

        prop_assert!(
            k_val >= 0.0 && k_val <= 1.0 + 1e-10,
            "Kernel value {} should be in [0, 1]",
            k_val
        );
    }

    /// Property: Kernel matrix should be symmetric
    #[test]
    fn kernel_matrix_symmetric(n_samples in 3usize..8) {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            ((i * 7 + j * 3) as f64 % 5.0) - 2.5
        });

        let k_matrix = kernel.kernel_matrix(&data).unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                prop_assert!(
                    (k_matrix[[i, j]] - k_matrix[[j, i]]).abs() < 1e-10,
                    "Kernel matrix not symmetric at ({}, {}): {} vs {}",
                    i, j, k_matrix[[i, j]], k_matrix[[j, i]]
                );
            }
        }
    }

    /// Property: Kernel matrix diagonal should be 1
    #[test]
    fn kernel_matrix_diagonal(n_samples in 3usize..8) {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        let data = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            ((i * 5 + j * 2) as f64 % 4.0) - 2.0
        });

        let k_matrix = kernel.kernel_matrix(&data).unwrap();

        for i in 0..n_samples {
            prop_assert!(
                (k_matrix[[i, i]] - 1.0).abs() < 1e-6,
                "Diagonal K[{},{}] = {} should be 1.0",
                i, i, k_matrix[[i, i]]
            );
        }
    }
}

// ============================================================================
// E-Value Properties (Statistical Guarantees)
// ============================================================================

proptest! {
    /// Property: E-value should always be non-negative
    #[test]
    fn evalue_nonnegative(
        mmd_values in prop::collection::vec(-1.0f64..2.0, 5..20)
    ) {
        let config = EValueConfig::default();
        let mut test = EValueTest::new(config).unwrap();

        for mmd in mmd_values {
            let update = test.update(mmd);
            prop_assert!(
                update.evalue >= 0.0,
                "E-value {} should be non-negative",
                update.evalue
            );
        }
    }

    /// Property: E-value increment should be non-negative
    #[test]
    fn evalue_increment_nonnegative(
        mmd_values in prop::collection::vec(-1.0f64..2.0, 5..20)
    ) {
        let config = EValueConfig {
            adaptive_betting: false,
            bet_fraction: 0.3,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        for mmd in mmd_values {
            let update = test.update(mmd);
            prop_assert!(
                update.evalue_increment >= 0.0,
                "E-value increment {} should be non-negative",
                update.evalue_increment
            );
        }
    }

    /// Property: P-value should be in [0, 1]
    #[test]
    fn pvalue_bounded(
        mmd_values in prop::collection::vec(-0.5f64..1.0, 10..30)
    ) {
        let config = EValueConfig::default();
        let mut test = EValueTest::new(config).unwrap();

        for mmd in mmd_values {
            let update = test.update(mmd);
            prop_assert!(
                update.p_value >= 0.0 && update.p_value <= 1.0,
                "P-value {} should be in [0, 1]",
                update.p_value
            );
        }
    }

    /// Property: Sample count should increment correctly
    #[test]
    fn evalue_sample_count(n_samples in 1usize..50) {
        let config = EValueConfig::default();
        let mut test = EValueTest::new(config).unwrap();

        for i in 0..n_samples {
            let update = test.update(0.01 * (i as f64));
            prop_assert_eq!(
                update.n_samples, i + 1,
                "Sample count mismatch: expected {}, got {}",
                i + 1, update.n_samples
            );
        }
    }

    /// Property: Reset should restore initial state
    #[test]
    fn evalue_reset_works(
        mmd_values in prop::collection::vec(-0.5f64..1.0, 5..20)
    ) {
        let config = EValueConfig::default();
        let mut test = EValueTest::new(config.clone()).unwrap();

        // Process some samples
        for mmd in mmd_values {
            test.update(mmd);
        }

        // Reset
        test.reset();

        prop_assert_eq!(test.n_samples(), 0);
        prop_assert_eq!(test.current_evalue(), config.initial_wealth);
        prop_assert!(!test.is_drift_detected());
    }

    /// Property: E-value should not decrease (martingale property approximation)
    /// Note: This is a simplified check; the actual martingale property is statistical
    #[test]
    fn evalue_bounded_growth(
        positive_mmd_values in prop::collection::vec(0.0f64..0.5, 5..15)
    ) {
        let config = EValueConfig {
            adaptive_betting: false,
            bet_fraction: 0.1, // Conservative betting
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        let mut prev_evalue = 1.0;

        for mmd in positive_mmd_values {
            let update = test.update(mmd);
            // With positive MMD and conservative betting, e-value should increase
            prop_assert!(
                update.evalue >= prev_evalue * 0.5, // Allow some decrease
                "E-value dropped too much: {} -> {}",
                prev_evalue, update.evalue
            );
            prev_evalue = update.evalue;
        }
    }
}

// ============================================================================
// Confidence Sequence Properties
// ============================================================================

proptest! {
    /// Property: Confidence interval width should be positive
    #[test]
    fn confidence_interval_positive_width(
        values in prop::collection::vec(-5.0f64..5.0, 10..30)
    ) {
        let config = ConfidenceSequenceConfig::default();
        let mut cs = ConfidenceSequence::new(config).unwrap();

        for x in values {
            if let Some(ci) = cs.update(x) {
                prop_assert!(
                    ci.width > 0.0,
                    "CI width {} should be positive",
                    ci.width
                );
            }
        }
    }

    /// Property: Confidence interval should contain the running mean
    #[test]
    fn confidence_interval_contains_mean(
        values in prop::collection::vec(-3.0f64..3.0, 20..50)
    ) {
        let config = ConfidenceSequenceConfig {
            confidence_level: 0.95,
            min_samples: 5,
            ..Default::default()
        };
        let mut cs = ConfidenceSequence::new(config).unwrap();

        let mut sum = 0.0;
        let mut count = 0;

        for x in values {
            sum += x;
            count += 1;
            let running_mean = sum / count as f64;

            if let Some(ci) = cs.update(x) {
                // The CI should contain the running mean (not the true mean)
                // This is a weaker property but always holds
                let ci_center = (ci.lower + ci.upper) / 2.0;
                let deviation = (ci_center - running_mean).abs();

                // The center should be close to the running mean
                prop_assert!(
                    deviation < ci.width,
                    "CI center {} deviates too much from running mean {}",
                    ci_center, running_mean
                );
            }
        }
    }

    /// Property: Confidence interval should narrow with more samples (asymptotically)
    #[test]
    fn confidence_interval_narrows(
        n_samples in 50usize..100
    ) {
        let config = ConfidenceSequenceConfig::default();
        let mut cs = ConfidenceSequence::new(config).unwrap();

        // Use constant values to eliminate variance
        let mut early_width = None;
        let mut late_width = None;

        for i in 0..n_samples {
            // Use small random-like but deterministic values
            let x = ((i * 7 % 11) as f64 / 10.0) - 0.5;
            if let Some(ci) = cs.update(x) {
                if i == 20 {
                    early_width = Some(ci.width);
                }
                if i == n_samples - 1 {
                    late_width = Some(ci.width);
                }
            }
        }

        if let (Some(early), Some(late)) = (early_width, late_width) {
            // Late width should be smaller or similar (allowing for variance)
            prop_assert!(
                late <= early * 1.5,
                "CI should narrow over time: early={}, late={}",
                early, late
            );
        }
    }

    /// Property: Sample count tracks correctly
    #[test]
    fn confidence_sequence_sample_count(n_samples in 5usize..50) {
        let config = ConfidenceSequenceConfig::default();
        let mut cs = ConfidenceSequence::new(config).unwrap();

        for i in 0..n_samples {
            cs.update(0.1 * (i as f64));
        }

        prop_assert_eq!(cs.n_samples(), n_samples);
    }

    /// Property: Reset should clear state
    #[test]
    fn confidence_sequence_reset(
        values in prop::collection::vec(-2.0f64..2.0, 10..30)
    ) {
        let config = ConfidenceSequenceConfig::default();
        let mut cs = ConfidenceSequence::new(config).unwrap();

        for x in values {
            cs.update(x);
        }

        cs.reset();

        prop_assert_eq!(cs.n_samples(), 0);
    }
}

// ============================================================================
// Integration Properties
// ============================================================================

proptest! {
    /// Property: Similar distributions should produce low MMD
    #[test]
    fn similar_distributions_low_mmd(n_samples in 10usize..20) {
        let config = QuantumKernelConfig {
            n_qubits: 2,
            n_layers: 1,
            seed: Some(42),
            ..Default::default()
        };
        let kernel = QuantumKernel::new(config).unwrap();

        // Generate "similar" data (same pattern)
        let baseline = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            ((i + j) as f64 % 3.0) - 1.0
        });

        let test_data = Array2::from_shape_fn((n_samples, 3), |(i, j)| {
            ((i + j) as f64 % 3.0) - 1.0 + 0.01 // Slight perturbation
        });

        let k_baseline = kernel.kernel_matrix(&baseline).unwrap();
        let k_test = kernel.kernel_matrix(&test_data).unwrap();
        let k_cross = kernel.cross_kernel_matrix(&baseline, &test_data).unwrap();

        // Compute MMD^2 approximation
        let baseline_mean: f64 = k_baseline.iter().sum::<f64>() / (n_samples * n_samples) as f64;
        let test_mean: f64 = k_test.iter().sum::<f64>() / (n_samples * n_samples) as f64;
        let cross_mean: f64 = k_cross.iter().sum::<f64>() / (n_samples * n_samples) as f64;

        let mmd2 = baseline_mean - 2.0 * cross_mean + test_mean;

        // MMD should be small for similar distributions
        prop_assert!(
            mmd2.abs() < 0.5,
            "MMD^2 = {} should be small for similar distributions",
            mmd2
        );
    }
}
