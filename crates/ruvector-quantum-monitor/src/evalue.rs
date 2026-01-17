//! E-Value Sequential Testing for Anytime-Valid Inference.
//!
//! This module implements e-value based sequential hypothesis testing for
//! distribution shift detection using Maximum Mean Discrepancy (MMD).
//!
//! # Mathematical Background
//!
//! ## E-Values
//!
//! An e-value E_t is a non-negative random variable with E[E_t] <= 1 under H_0.
//! The key property is that the product E_1 * E_2 * ... * E_t is also an e-value
//! (Ville's inequality), allowing anytime-valid sequential testing.
//!
//! ## Sequential MMD Test
//!
//! For testing H_0: P = Q vs H_1: P != Q using kernel MMD:
//!
//! ```text
//! MMD^2(P,Q) = E_P[k(X,X')] - 2*E_{P,Q}[k(X,Y)] + E_Q[k(Y,Y')]
//! ```
//!
//! The test statistic is converted to an e-value using the likelihood ratio
//! approach of Shekhar & Ramdas (2023).
//!
//! # References
//!
//! - Shekhar, S., & Ramdas, A. (2023). "Nonparametric Two-Sample Testing by
//!   Betting" (NeurIPS 2023)
//! - Vovk, V. & Wang, R. (2021). "E-values: Calibration, combination and applications"
//! - Gretton, A., et al. (2012). "A Kernel Two-Sample Test" (JMLR)

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::{QuantumMonitorError, Result};

/// Configuration for the E-value sequential test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EValueConfig {
    /// Significance level alpha for the test (e.g., 0.05).
    pub alpha: f64,

    /// Initial wealth for the betting strategy (typically 1.0).
    pub initial_wealth: f64,

    /// Fraction of wealth to bet on each round (0 < lambda < 1).
    /// Smaller values are more conservative but slower to detect drift.
    pub bet_fraction: f64,

    /// Minimum number of samples before declaring drift.
    pub min_samples: usize,

    /// Maximum e-value before numerical overflow protection.
    pub max_evalue: f64,

    /// Whether to use adaptive betting based on observed variance.
    pub adaptive_betting: bool,
}

impl Default for EValueConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            initial_wealth: 1.0,
            bet_fraction: 0.5,
            min_samples: 10,
            max_evalue: 1e100,
            adaptive_betting: true,
        }
    }
}

impl EValueConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "alpha",
                "must be in (0, 1)",
            ));
        }
        if self.initial_wealth <= 0.0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "initial_wealth",
                "must be positive",
            ));
        }
        if self.bet_fraction <= 0.0 || self.bet_fraction >= 1.0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "bet_fraction",
                "must be in (0, 1)",
            ));
        }
        if self.min_samples == 0 {
            return Err(QuantumMonitorError::invalid_parameter(
                "min_samples",
                "must be at least 1",
            ));
        }
        Ok(())
    }

    /// Get the rejection threshold (1/alpha by Ville's inequality).
    pub fn rejection_threshold(&self) -> f64 {
        1.0 / self.alpha
    }
}

/// Sequential E-value test for distribution drift detection.
///
/// This struct maintains the running e-value and provides anytime-valid
/// p-values for the hypothesis test H_0: P = Q.
#[derive(Debug, Clone)]
pub struct EValueTest {
    config: EValueConfig,
    /// Current accumulated e-value (product of all e-values).
    current_evalue: f64,
    /// Number of samples processed.
    n_samples: usize,
    /// Running sum of squared MMD estimates (for variance estimation).
    mmd_squared_sum: f64,
    /// Running sum of MMD estimates (for mean estimation).
    mmd_sum: f64,
    /// History of e-values for analysis.
    evalue_history: Vec<f64>,
    /// History of MMD values.
    mmd_history: Vec<f64>,
    /// Whether drift has been detected.
    drift_detected: bool,
    /// Sample index when drift was detected.
    drift_detection_time: Option<usize>,
}

impl EValueTest {
    /// Create a new E-value test with the given configuration.
    pub fn new(config: EValueConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            current_evalue: 1.0, // Start at initial wealth
            n_samples: 0,
            mmd_squared_sum: 0.0,
            mmd_sum: 0.0,
            evalue_history: Vec::new(),
            mmd_history: Vec::new(),
            drift_detected: false,
            drift_detection_time: None,
        })
    }

    /// Update the e-value with a new MMD observation.
    ///
    /// # Mathematical Details
    ///
    /// We use the CUSUM-like betting martingale:
    ///
    /// ```text
    /// E_t = E_{t-1} * (1 + lambda * sign(MMD^2) * |MMD^2| / sigma_t)
    /// ```
    ///
    /// where sigma_t is the estimated standard deviation of MMD^2 under H_0.
    ///
    /// # Arguments
    ///
    /// * `mmd_squared` - The observed squared MMD statistic.
    ///
    /// # Returns
    ///
    /// `EValueUpdate` containing the new e-value and test decision.
    pub fn update(&mut self, mmd_squared: f64) -> EValueUpdate {
        self.n_samples += 1;

        // Update running statistics
        self.mmd_sum += mmd_squared;
        self.mmd_squared_sum += mmd_squared * mmd_squared;
        self.mmd_history.push(mmd_squared);

        // Estimate variance of MMD under H_0
        let variance = self.estimate_variance();

        // Compute the likelihood ratio / e-value increment
        let evalue_increment = self.compute_evalue_increment(mmd_squared, variance);

        // Update accumulated e-value (multiplicative)
        self.current_evalue *= evalue_increment;

        // Clip to prevent overflow
        self.current_evalue = self.current_evalue.min(self.config.max_evalue);

        self.evalue_history.push(self.current_evalue);

        // Check for drift detection
        let threshold = self.config.rejection_threshold();
        if !self.drift_detected
            && self.n_samples >= self.config.min_samples
            && self.current_evalue >= threshold
        {
            self.drift_detected = true;
            self.drift_detection_time = Some(self.n_samples);
        }

        EValueUpdate {
            evalue: self.current_evalue,
            evalue_increment,
            mmd_squared,
            p_value: self.anytime_p_value(),
            drift_detected: self.drift_detected,
            detection_time: self.drift_detection_time,
            n_samples: self.n_samples,
        }
    }

    /// Compute the e-value increment for a single observation.
    ///
    /// Uses the betting strategy from Shekhar & Ramdas (2023).
    /// Under H_0, MMD^2 should be close to 0. Under H_1, it should be positive.
    /// We use a one-sided test betting on positive MMD values.
    fn compute_evalue_increment(&self, mmd_squared: f64, variance: f64) -> f64 {
        let lambda = if self.config.adaptive_betting {
            self.adaptive_bet_fraction(variance)
        } else {
            self.config.bet_fraction
        };

        // Standard deviation with floor to prevent division by zero
        let std_dev = variance.sqrt().max(1e-10);

        // For one-sided test against H_0: MMD^2 = 0
        // We normalize by the standard deviation for scale-invariance
        // Clamp normalized value to prevent extreme e-values
        let normalized_mmd = (mmd_squared / std_dev).clamp(-5.0, 5.0);

        // Betting martingale with capped increments
        // E_t = 1 + lambda * (X_t - threshold) where threshold ~ 0 under H_0
        // We bet that MMD will be positive under H_1
        let increment = 1.0 + lambda * normalized_mmd;

        // Clamp to [0.1, 10] to prevent extreme swings
        increment.clamp(0.1, 10.0)
    }

    /// Compute adaptive bet fraction based on observed variance.
    ///
    /// Kelly criterion suggests optimal betting proportional to edge/odds.
    fn adaptive_bet_fraction(&self, variance: f64) -> f64 {
        if self.n_samples < 5 || variance < 1e-10 {
            return self.config.bet_fraction;
        }

        let mean_mmd = self.mmd_sum / self.n_samples as f64;
        let std_dev = variance.sqrt();

        // Kelly fraction: bet = edge / variance
        // Edge is estimated from mean MMD under potential H_1
        let edge = mean_mmd.abs();
        let kelly = edge / std_dev;

        // Cap at configured maximum and use fractional Kelly for safety
        (kelly * 0.5).min(self.config.bet_fraction).max(0.01)
    }

    /// Estimate variance of MMD^2 under H_0.
    ///
    /// Under H_0 (no drift), MMD^2 ~ N(0, sigma^2/n) asymptotically.
    fn estimate_variance(&self) -> f64 {
        if self.n_samples < 2 {
            return 1.0; // Prior variance estimate
        }

        let n = self.n_samples as f64;
        let mean = self.mmd_sum / n;
        let variance = (self.mmd_squared_sum / n) - (mean * mean);

        // Floor variance to prevent numerical issues
        variance.max(1e-10)
    }

    /// Get anytime-valid p-value.
    ///
    /// p = 1/E_t by Ville's inequality, valid at any stopping time.
    pub fn anytime_p_value(&self) -> f64 {
        (1.0 / self.current_evalue).min(1.0)
    }

    /// Get the current e-value.
    pub fn current_evalue(&self) -> f64 {
        self.current_evalue
    }

    /// Check if drift has been detected.
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Get the detection time (sample index).
    pub fn detection_time(&self) -> Option<usize> {
        self.drift_detection_time
    }

    /// Get number of samples processed.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the e-value history.
    pub fn evalue_history(&self) -> &[f64] {
        &self.evalue_history
    }

    /// Get the MMD history.
    pub fn mmd_history(&self) -> &[f64] {
        &self.mmd_history
    }

    /// Reset the test state.
    pub fn reset(&mut self) {
        self.current_evalue = self.config.initial_wealth;
        self.n_samples = 0;
        self.mmd_squared_sum = 0.0;
        self.mmd_sum = 0.0;
        self.evalue_history.clear();
        self.mmd_history.clear();
        self.drift_detected = false;
        self.drift_detection_time = None;
    }

    /// Get test statistics summary.
    pub fn summary(&self) -> EValueSummary {
        let mean_mmd = if self.n_samples > 0 {
            self.mmd_sum / self.n_samples as f64
        } else {
            0.0
        };

        let variance = self.estimate_variance();

        EValueSummary {
            n_samples: self.n_samples,
            current_evalue: self.current_evalue,
            p_value: self.anytime_p_value(),
            mean_mmd,
            mmd_std_dev: variance.sqrt(),
            drift_detected: self.drift_detected,
            detection_time: self.drift_detection_time,
        }
    }
}

/// Result of an e-value update.
#[derive(Debug, Clone)]
pub struct EValueUpdate {
    /// Current accumulated e-value.
    pub evalue: f64,
    /// E-value increment from this sample.
    pub evalue_increment: f64,
    /// Observed MMD^2 value.
    pub mmd_squared: f64,
    /// Anytime-valid p-value (1/E).
    pub p_value: f64,
    /// Whether drift has been detected.
    pub drift_detected: bool,
    /// Sample index when drift was detected (if any).
    pub detection_time: Option<usize>,
    /// Total number of samples processed.
    pub n_samples: usize,
}

/// Summary statistics for the e-value test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EValueSummary {
    /// Number of samples processed.
    pub n_samples: usize,
    /// Current accumulated e-value.
    pub current_evalue: f64,
    /// Anytime-valid p-value.
    pub p_value: f64,
    /// Mean of observed MMD^2 values.
    pub mean_mmd: f64,
    /// Standard deviation of MMD^2 values.
    pub mmd_std_dev: f64,
    /// Whether drift has been detected.
    pub drift_detected: bool,
    /// Detection time (sample index).
    pub detection_time: Option<usize>,
}

/// MMD (Maximum Mean Discrepancy) estimator.
///
/// Provides unbiased estimators for MMD^2 using the quantum kernel.
#[derive(Debug, Clone)]
#[allow(dead_code)] // baseline_kernel stored for potential future use in variance estimation
pub struct MMDEstimator {
    /// Cached baseline kernel matrix (for future variance estimation).
    baseline_kernel: Array2<f64>,
    /// Mean of baseline self-kernel.
    baseline_mean: f64,
    /// Number of baseline samples.
    n_baseline: usize,
}

impl MMDEstimator {
    /// Create a new MMD estimator from baseline kernel matrix.
    ///
    /// # Arguments
    ///
    /// * `baseline_kernel` - Kernel matrix K[i,j] = k(x_i, x_j) for baseline samples.
    pub fn new(baseline_kernel: Array2<f64>) -> Result<Self> {
        let n = baseline_kernel.nrows();
        if n != baseline_kernel.ncols() {
            return Err(QuantumMonitorError::dimension_mismatch(n, baseline_kernel.ncols()));
        }
        if n < 2 {
            return Err(QuantumMonitorError::insufficient_samples(2, n));
        }

        // Compute unbiased mean of baseline kernel (excluding diagonal)
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum += baseline_kernel[[i, j]];
                    count += 1;
                }
            }
        }
        let baseline_mean = sum / count as f64;

        Ok(Self {
            baseline_kernel,
            baseline_mean,
            n_baseline: n,
        })
    }

    /// Compute unbiased MMD^2 estimate between baseline and test samples.
    ///
    /// Uses the U-statistic estimator:
    /// ```text
    /// MMD^2_u = (1/m(m-1)) sum_{i!=j} k(x_i, x_j)
    ///         - (2/mn) sum_{i,j} k(x_i, y_j)
    ///         + (1/n(n-1)) sum_{i!=j} k(y_i, y_j)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `cross_kernel` - Kernel matrix between baseline and test samples.
    /// * `test_kernel` - Kernel matrix for test samples.
    pub fn mmd_squared_unbiased(
        &self,
        cross_kernel: &Array2<f64>,
        test_kernel: &Array2<f64>,
    ) -> Result<f64> {
        let m = self.n_baseline;
        let n = test_kernel.nrows();

        if cross_kernel.nrows() != m || cross_kernel.ncols() != n {
            return Err(QuantumMonitorError::dimension_mismatch(
                m * n,
                cross_kernel.nrows() * cross_kernel.ncols(),
            ));
        }

        // Mean of cross-kernel
        let cross_sum: f64 = cross_kernel.iter().sum();
        let cross_mean = cross_sum / (m * n) as f64;

        // Mean of test self-kernel (excluding diagonal)
        let mut test_sum = 0.0;
        let mut test_count = 0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    test_sum += test_kernel[[i, j]];
                    test_count += 1;
                }
            }
        }
        let test_mean = if test_count > 0 {
            test_sum / test_count as f64
        } else {
            1.0 // Single sample case
        };

        // MMD^2 = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
        let mmd_squared = self.baseline_mean - 2.0 * cross_mean + test_mean;

        Ok(mmd_squared)
    }

    /// Compute MMD^2 with variance estimate for hypothesis testing.
    ///
    /// Returns both the MMD^2 estimate and its estimated variance
    /// under the null hypothesis.
    pub fn mmd_squared_with_variance(
        &self,
        cross_kernel: &Array2<f64>,
        test_kernel: &Array2<f64>,
    ) -> Result<(f64, f64)> {
        let mmd2 = self.mmd_squared_unbiased(cross_kernel, test_kernel)?;

        // Variance estimation using the approach from Gretton et al. (2012)
        // Under H_0, Var(MMD^2_u) approx 2/m^2 * (expected kernel variance)
        let m = self.n_baseline as f64;
        let n = test_kernel.nrows() as f64;

        // Approximate variance of kernel values
        let kernel_var = self.estimate_kernel_variance(test_kernel);

        // Variance of U-statistic estimator
        let var_estimate = 4.0 * kernel_var / m.min(n);

        Ok((mmd2, var_estimate.max(1e-10)))
    }

    /// Estimate variance of kernel values.
    fn estimate_kernel_variance(&self, kernel: &Array2<f64>) -> f64 {
        let n = kernel.nrows();
        if n < 2 {
            return 1.0;
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = kernel[[i, j]];
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 1.0;
        }

        let mean = sum / count as f64;
        let variance = (sum_sq / count as f64) - mean * mean;

        variance.max(1e-10)
    }

    /// Get the number of baseline samples.
    pub fn n_baseline(&self) -> usize {
        self.n_baseline
    }

    /// Get the mean baseline kernel value.
    pub fn baseline_mean(&self) -> f64 {
        self.baseline_mean
    }
}

/// Online MMD estimator for streaming data.
///
/// Computes running MMD estimates without storing all historical data.
#[derive(Debug, Clone)]
pub struct OnlineMMD {
    /// Baseline mean kernel value.
    baseline_mean: f64,
    /// Sum of cross-kernel values.
    cross_sum: f64,
    /// Sum of test self-kernel values.
    test_sum: f64,
    /// Count of cross-kernel pairs.
    cross_count: usize,
    /// Count of test self-kernel pairs.
    test_count: usize,
}

impl OnlineMMD {
    /// Create a new online MMD estimator.
    pub fn new(baseline_mean: f64) -> Self {
        Self {
            baseline_mean,
            cross_sum: 0.0,
            test_sum: 0.0,
            cross_count: 0,
            test_count: 0,
        }
    }

    /// Update with new cross-kernel values (baseline vs new sample).
    pub fn update_cross(&mut self, cross_values: &[f64]) {
        self.cross_sum += cross_values.iter().sum::<f64>();
        self.cross_count += cross_values.len();
    }

    /// Update with new test self-kernel values.
    pub fn update_test(&mut self, test_values: &[f64]) {
        self.test_sum += test_values.iter().sum::<f64>();
        self.test_count += test_values.len();
    }

    /// Get current MMD^2 estimate.
    pub fn mmd_squared(&self) -> f64 {
        if self.cross_count == 0 {
            return 0.0;
        }

        let cross_mean = self.cross_sum / self.cross_count as f64;
        let test_mean = if self.test_count > 0 {
            self.test_sum / self.test_count as f64
        } else {
            1.0 // Single sample
        };

        self.baseline_mean - 2.0 * cross_mean + test_mean
    }

    /// Reset the online estimator.
    pub fn reset(&mut self) {
        self.cross_sum = 0.0;
        self.test_sum = 0.0;
        self.cross_count = 0;
        self.test_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_evalue_config_validation() {
        let mut config = EValueConfig::default();
        assert!(config.validate().is_ok());

        config.alpha = 0.0;
        assert!(config.validate().is_err());

        config.alpha = 1.0;
        assert!(config.validate().is_err());

        config.alpha = 0.05;
        config.bet_fraction = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_evalue_test_creation() {
        let config = EValueConfig::default();
        let test = EValueTest::new(config).unwrap();

        assert_eq!(test.current_evalue(), 1.0);
        assert_eq!(test.n_samples(), 0);
        assert!(!test.is_drift_detected());
    }

    #[test]
    fn test_evalue_update_no_drift() {
        let config = EValueConfig {
            alpha: 0.05,
            min_samples: 5,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        // Simulate null hypothesis (MMD near 0)
        for _ in 0..20 {
            let mmd = 0.001 * (rand::random::<f64>() - 0.5);
            let update = test.update(mmd);
            // E-value should stay reasonable under H_0
            assert!(update.evalue > 0.0);
        }

        // Under H_0, drift should not be detected with high probability
        let summary = test.summary();
        assert_eq!(summary.n_samples, 20);
        // P-value should not be extremely small under H_0
        assert!(summary.p_value > 0.001);
    }

    #[test]
    fn test_evalue_update_with_drift() {
        let config = EValueConfig {
            alpha: 0.05,
            min_samples: 5,
            bet_fraction: 0.5,
            ..Default::default()
        };
        let mut test = EValueTest::new(config).unwrap();

        // Simulate drift (consistently positive MMD)
        for _ in 0..50 {
            let mmd = 0.5 + 0.1 * rand::random::<f64>(); // Large positive MMD
            test.update(mmd);
        }

        // Should detect drift with large MMD
        let summary = test.summary();
        assert!(
            summary.drift_detected || summary.p_value < 0.1,
            "Should detect or suspect drift with large MMD values. P-value: {}",
            summary.p_value
        );
    }

    #[test]
    fn test_anytime_validity() {
        let config = EValueConfig {
            adaptive_betting: false,
            bet_fraction: 0.3,  // Conservative betting
            min_samples: 5,
            ..Default::default()
        };
        let mut test = EValueTest::new(config.clone()).unwrap();

        // Under H_0, MMD^2 should be centered around 0 (or a small positive value due to bias)
        // We simulate with noise centered at 0
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            // Simulate MMD^2 under H_0: small values with random sign
            let mmd: f64 = 0.01 * (rand::Rng::gen::<f64>(&mut rng) - 0.5);
            test.update(mmd);
        }

        // Under H_0 with proper calibration, e-value should not explode
        // The p-value = 1/E should not be extremely small consistently
        let final_evalue = test.current_evalue();
        let final_pvalue = test.anytime_p_value();

        // The e-value should stay bounded (not extremely large) under H_0
        assert!(
            final_evalue < 1000.0 || final_pvalue > 0.001,
            "E-value {} too large under H_0 (p-value {})",
            final_evalue,
            final_pvalue
        );
    }

    #[test]
    fn test_evalue_reset() {
        let config = EValueConfig::default();
        let mut test = EValueTest::new(config).unwrap();

        test.update(0.1);
        test.update(0.2);
        assert_eq!(test.n_samples(), 2);

        test.reset();
        assert_eq!(test.n_samples(), 0);
        assert_eq!(test.current_evalue(), 1.0);
        assert!(!test.is_drift_detected());
    }

    #[test]
    fn test_mmd_estimator() {
        // Create a simple kernel matrix
        let baseline_kernel = Array2::from_shape_fn((10, 10), |(i, j)| {
            if i == j {
                1.0
            } else {
                0.8 // High correlation in baseline
            }
        });

        let estimator = MMDEstimator::new(baseline_kernel).unwrap();
        assert_eq!(estimator.n_baseline(), 10);
        assert!((estimator.baseline_mean() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_mmd_squared_computation() {
        let baseline_kernel = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i == j { 1.0 } else { 0.9 }
        });

        let estimator = MMDEstimator::new(baseline_kernel).unwrap();

        // Test kernel similar to baseline -> low MMD
        let test_kernel = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i == j { 1.0 } else { 0.88 }
        });

        let cross_kernel = Array2::from_elem((5, 3), 0.89);

        let (mmd2, var) = estimator
            .mmd_squared_with_variance(&cross_kernel, &test_kernel)
            .unwrap();

        // MMD should be small for similar distributions
        assert!(mmd2.abs() < 0.5, "MMD^2 = {} should be small", mmd2);
        assert!(var > 0.0, "Variance should be positive");
    }

    #[test]
    fn test_online_mmd() {
        let mut online = OnlineMMD::new(0.9);

        // Add some cross-kernel values
        online.update_cross(&[0.85, 0.87, 0.86]);
        online.update_test(&[0.88, 0.89]);

        let mmd2 = online.mmd_squared();
        assert!(mmd2.is_finite());

        online.reset();
        assert_eq!(online.mmd_squared(), 0.0);
    }
}
