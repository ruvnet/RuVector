//! Main Quantum Kernel Coherence Monitor.
//!
//! This module provides the primary interface for monitoring distribution drift
//! using quantum kernel methods with anytime-valid statistical guarantees.
//!
//! # Architecture
//!
//! The monitor combines three key components:
//!
//! 1. **Quantum Kernel** - Encodes data into quantum feature space and computes
//!    kernel-based similarity measures (MMD).
//!
//! 2. **E-Value Testing** - Provides sequential hypothesis testing with
//!    anytime-valid p-values using the betting martingale approach.
//!
//! 3. **Confidence Sequences** - Tracks running confidence intervals for the
//!    MMD statistic with time-uniform coverage guarantees.
//!
//! # Usage
//!
//! ```ignore
//! use ruvector_quantum_monitor::{QuantumCoherenceMonitor, MonitorConfig};
//! use ndarray::Array2;
//!
//! // Create monitor
//! let config = MonitorConfig::default();
//! let mut monitor = QuantumCoherenceMonitor::new(config)?;
//!
//! // Set baseline distribution
//! let baseline = Array2::from_shape_fn((100, 4), |_| rand::random::<f64>());
//! monitor.set_baseline(&baseline)?;
//!
//! // Monitor streaming data
//! for sample in streaming_data {
//!     let result = monitor.observe(&sample)?;
//!     if result.drift_detected {
//!         println!("Drift detected at sample {}", result.n_samples);
//!     }
//! }
//! ```

use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::confidence::{ChangeDetectionCS, ConfidenceInterval, ConfidenceSequence, ConfidenceSequenceConfig};
use crate::error::{QuantumMonitorError, Result};
use crate::evalue::{EValueConfig, EValueSummary, EValueTest, EValueUpdate};
use crate::kernel::{QuantumKernel, QuantumKernelConfig, StreamingKernelAccumulator};

/// Configuration for the quantum coherence monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Configuration for the quantum kernel.
    pub kernel: QuantumKernelConfig,
    /// Configuration for E-value testing.
    pub evalue: EValueConfig,
    /// Configuration for confidence sequences.
    pub confidence: ConfidenceSequenceConfig,
    /// Minimum baseline samples required.
    pub min_baseline_samples: usize,
    /// Window size for rolling MMD estimation (0 for cumulative).
    pub rolling_window: usize,
    /// Whether to emit alerts on drift detection.
    pub alert_on_drift: bool,
    /// Cooldown period after drift detection before re-alerting.
    pub alert_cooldown: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            kernel: QuantumKernelConfig::default(),
            evalue: EValueConfig::default(),
            confidence: ConfidenceSequenceConfig::default(),
            min_baseline_samples: 20,
            rolling_window: 0, // Cumulative by default
            alert_on_drift: true,
            alert_cooldown: 100,
        }
    }
}

impl MonitorConfig {
    /// Create a config optimized for fast detection (lower sample requirements).
    pub fn fast_detection() -> Self {
        Self {
            kernel: QuantumKernelConfig {
                n_qubits: 3,
                n_layers: 1,
                ..Default::default()
            },
            evalue: EValueConfig {
                min_samples: 5,
                bet_fraction: 0.6,
                ..Default::default()
            },
            min_baseline_samples: 10,
            ..Default::default()
        }
    }

    /// Create a config optimized for high precision (more conservative).
    pub fn high_precision() -> Self {
        Self {
            kernel: QuantumKernelConfig {
                n_qubits: 5,
                n_layers: 3,
                ..Default::default()
            },
            evalue: EValueConfig {
                alpha: 0.01,
                min_samples: 20,
                bet_fraction: 0.3,
                ..Default::default()
            },
            confidence: ConfidenceSequenceConfig {
                confidence_level: 0.99,
                ..Default::default()
            },
            min_baseline_samples: 50,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        self.kernel.validate()?;
        self.evalue.validate()?;
        self.confidence.validate()?;

        if self.min_baseline_samples < 2 {
            return Err(QuantumMonitorError::invalid_parameter(
                "min_baseline_samples",
                "must be at least 2",
            ));
        }

        Ok(())
    }
}

/// State of the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitorState {
    /// Monitor is not initialized with baseline data.
    Uninitialized,
    /// Monitor is collecting baseline samples.
    CollectingBaseline,
    /// Monitor is actively monitoring for drift.
    Monitoring,
    /// Drift has been detected.
    DriftDetected,
    /// Monitor is in cooldown after drift detection.
    Cooldown,
}

/// Result of observing a new sample.
#[derive(Debug, Clone)]
pub struct ObservationResult {
    /// Current e-value.
    pub evalue: f64,
    /// Anytime-valid p-value.
    pub p_value: f64,
    /// Estimated MMD^2 value.
    pub mmd_squared: f64,
    /// Current confidence interval for MMD.
    pub confidence_interval: Option<ConfidenceInterval>,
    /// Whether drift has been detected.
    pub drift_detected: bool,
    /// Sample index when drift was detected (if any).
    pub detection_time: Option<usize>,
    /// Total number of streaming samples observed.
    pub n_samples: usize,
    /// Current monitor state.
    pub state: MonitorState,
}

/// Alert generated when drift is detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftAlert {
    /// Time (sample index) when drift was detected.
    pub detection_time: usize,
    /// Final p-value at detection.
    pub p_value: f64,
    /// Final e-value at detection.
    pub evalue: f64,
    /// Estimated MMD^2 at detection.
    pub mmd_squared: f64,
    /// Confidence interval at detection.
    pub confidence_interval: Option<ConfidenceInterval>,
    /// Severity level (based on effect size).
    pub severity: DriftSeverity,
}

/// Severity of detected drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    /// Minor drift (small MMD).
    Minor,
    /// Moderate drift.
    Moderate,
    /// Severe drift (large MMD).
    Severe,
}

impl DriftSeverity {
    /// Determine severity from MMD^2 value.
    pub fn from_mmd_squared(mmd2: f64) -> Self {
        if mmd2 < 0.1 {
            Self::Minor
        } else if mmd2 < 0.5 {
            Self::Moderate
        } else {
            Self::Severe
        }
    }
}

/// Quantum Kernel Coherence Monitor for distribution drift detection.
///
/// This is the main struct for monitoring streaming data for distribution shift
/// using quantum kernel methods with anytime-valid statistical guarantees.
pub struct QuantumCoherenceMonitor {
    config: MonitorConfig,
    state: MonitorState,
    /// Quantum kernel for computing similarities.
    kernel: QuantumKernel,
    /// Streaming kernel accumulator for efficient updates.
    kernel_accumulator: Option<StreamingKernelAccumulator>,
    /// E-value sequential test.
    evalue_test: EValueTest,
    /// Confidence sequence for MMD.
    confidence_seq: ConfidenceSequence,
    /// Baseline data (stored for potential re-analysis).
    baseline: Option<Array2<f64>>,
    /// Count of samples since last alert.
    samples_since_alert: usize,
    /// History of alerts.
    alert_history: Vec<DriftAlert>,
    /// Dimension of input data.
    data_dim: Option<usize>,
}

impl QuantumCoherenceMonitor {
    /// Create a new quantum coherence monitor.
    pub fn new(config: MonitorConfig) -> Result<Self> {
        config.validate()?;

        let kernel = QuantumKernel::new(config.kernel.clone())?;
        let evalue_test = EValueTest::new(config.evalue.clone())?;
        let confidence_seq = ConfidenceSequence::new(config.confidence.clone())?;

        Ok(Self {
            config,
            state: MonitorState::Uninitialized,
            kernel,
            kernel_accumulator: None,
            evalue_test,
            confidence_seq,
            baseline: None,
            samples_since_alert: 0,
            alert_history: Vec::new(),
            data_dim: None,
        })
    }

    /// Set the baseline distribution for comparison.
    ///
    /// The baseline represents the "expected" distribution that streaming
    /// samples will be compared against for drift detection.
    pub fn set_baseline(&mut self, baseline: &Array2<f64>) -> Result<()> {
        if baseline.nrows() < self.config.min_baseline_samples {
            return Err(QuantumMonitorError::insufficient_samples(
                self.config.min_baseline_samples,
                baseline.nrows(),
            ));
        }

        info!(
            "Setting baseline with {} samples of dimension {}",
            baseline.nrows(),
            baseline.ncols()
        );

        self.data_dim = Some(baseline.ncols());
        self.baseline = Some(baseline.clone());

        // Initialize streaming accumulator with baseline
        let mut accumulator = StreamingKernelAccumulator::new(self.kernel.clone());
        accumulator.set_baseline(baseline)?;

        self.kernel_accumulator = Some(accumulator);
        self.state = MonitorState::Monitoring;

        // Reset test statistics
        self.evalue_test.reset();
        self.confidence_seq.reset();
        self.samples_since_alert = 0;

        Ok(())
    }

    /// Observe a new sample and update the monitor.
    ///
    /// This is the main method for streaming monitoring. Call this for each
    /// new data point to check for distribution drift.
    pub fn observe(&mut self, sample: &Array1<f64>) -> Result<ObservationResult> {
        // Check state
        if self.state == MonitorState::Uninitialized {
            return Err(QuantumMonitorError::NotInitialized(
                "Baseline not set. Call set_baseline() first.".to_string(),
            ));
        }

        // Check dimensions
        if let Some(dim) = self.data_dim {
            if sample.len() != dim {
                return Err(QuantumMonitorError::dimension_mismatch(dim, sample.len()));
            }
        }

        // Update streaming kernel accumulator
        let accumulator = self.kernel_accumulator.as_mut().unwrap();
        let _update = accumulator.add_sample(sample)?;

        // Get current MMD^2 estimate
        let mmd_squared = accumulator.mmd_squared();

        // Update E-value test
        let evalue_update = self.evalue_test.update(mmd_squared);

        // Update confidence sequence
        let ci = self.confidence_seq.update(mmd_squared);

        // Update state based on drift detection
        self.samples_since_alert += 1;

        let drift_detected = evalue_update.drift_detected;

        if drift_detected {
            if self.state != MonitorState::DriftDetected {
                self.state = MonitorState::DriftDetected;

                if self.config.alert_on_drift {
                    let alert = DriftAlert {
                        detection_time: evalue_update.n_samples,
                        p_value: evalue_update.p_value,
                        evalue: evalue_update.evalue,
                        mmd_squared,
                        confidence_interval: ci.clone(),
                        severity: DriftSeverity::from_mmd_squared(mmd_squared),
                    };

                    warn!(
                        "DRIFT DETECTED at sample {} (p={:.6}, MMD^2={:.6})",
                        alert.detection_time, alert.p_value, alert.mmd_squared
                    );

                    self.alert_history.push(alert);
                    self.samples_since_alert = 0;
                }
            } else if self.samples_since_alert >= self.config.alert_cooldown {
                // In cooldown - suppress repeated alerts
                self.state = MonitorState::Cooldown;
            }
        } else if self.state == MonitorState::Cooldown
            && self.samples_since_alert >= self.config.alert_cooldown
        {
            // Exit cooldown
            self.state = MonitorState::Monitoring;
        }

        Ok(ObservationResult {
            evalue: evalue_update.evalue,
            p_value: evalue_update.p_value,
            mmd_squared,
            confidence_interval: ci,
            drift_detected,
            detection_time: evalue_update.detection_time,
            n_samples: evalue_update.n_samples,
            state: self.state,
        })
    }

    /// Observe multiple samples at once (batch update).
    pub fn observe_batch(&mut self, samples: &Array2<f64>) -> Result<Vec<ObservationResult>> {
        let mut results = Vec::with_capacity(samples.nrows());

        for i in 0..samples.nrows() {
            let sample = samples.row(i).to_owned();
            results.push(self.observe(&sample)?);
        }

        Ok(results)
    }

    /// Get the current monitor state.
    pub fn state(&self) -> MonitorState {
        self.state
    }

    /// Check if drift has been detected.
    pub fn is_drift_detected(&self) -> bool {
        self.state == MonitorState::DriftDetected
    }

    /// Get the current E-value.
    pub fn current_evalue(&self) -> f64 {
        self.evalue_test.current_evalue()
    }

    /// Get the current anytime-valid p-value.
    pub fn current_p_value(&self) -> f64 {
        self.evalue_test.anytime_p_value()
    }

    /// Get the E-value test summary.
    pub fn evalue_summary(&self) -> EValueSummary {
        self.evalue_test.summary()
    }

    /// Get the current confidence interval for MMD.
    pub fn confidence_interval(&self) -> Option<ConfidenceInterval> {
        self.confidence_seq.current_interval()
    }

    /// Get the number of streaming samples observed.
    pub fn n_samples(&self) -> usize {
        self.evalue_test.n_samples()
    }

    /// Get the alert history.
    pub fn alert_history(&self) -> &[DriftAlert] {
        &self.alert_history
    }

    /// Reset the monitor (keeping baseline).
    ///
    /// This resets all test statistics but keeps the baseline distribution.
    pub fn reset(&mut self) -> Result<()> {
        if let Some(baseline) = &self.baseline.clone() {
            self.set_baseline(baseline)?;
        }
        self.alert_history.clear();
        Ok(())
    }

    /// Reset completely (clear baseline too).
    pub fn reset_full(&mut self) {
        self.state = MonitorState::Uninitialized;
        self.kernel_accumulator = None;
        self.evalue_test.reset();
        self.confidence_seq.reset();
        self.baseline = None;
        self.samples_since_alert = 0;
        self.alert_history.clear();
        self.data_dim = None;
    }

    /// Get the configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Get comprehensive status.
    pub fn status(&self) -> MonitorStatus {
        MonitorStatus {
            state: self.state,
            n_baseline_samples: self.baseline.as_ref().map(|b| b.nrows()).unwrap_or(0),
            n_streaming_samples: self.n_samples(),
            current_evalue: self.current_evalue(),
            current_p_value: self.current_p_value(),
            drift_detected: self.is_drift_detected(),
            n_alerts: self.alert_history.len(),
        }
    }
}

/// Comprehensive status of the monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorStatus {
    /// Current state.
    pub state: MonitorState,
    /// Number of baseline samples.
    pub n_baseline_samples: usize,
    /// Number of streaming samples observed.
    pub n_streaming_samples: usize,
    /// Current e-value.
    pub current_evalue: f64,
    /// Current p-value.
    pub current_p_value: f64,
    /// Whether drift has been detected.
    pub drift_detected: bool,
    /// Number of alerts generated.
    pub n_alerts: usize,
}

/// Thread-safe wrapper for the monitor.
pub struct SharedMonitor(Arc<RwLock<QuantumCoherenceMonitor>>);

impl SharedMonitor {
    /// Create a new shared monitor.
    pub fn new(config: MonitorConfig) -> Result<Self> {
        Ok(Self(Arc::new(RwLock::new(QuantumCoherenceMonitor::new(config)?))))
    }

    /// Set the baseline distribution.
    pub fn set_baseline(&self, baseline: &Array2<f64>) -> Result<()> {
        self.0.write().set_baseline(baseline)
    }

    /// Observe a new sample.
    pub fn observe(&self, sample: &Array1<f64>) -> Result<ObservationResult> {
        self.0.write().observe(sample)
    }

    /// Get current status.
    pub fn status(&self) -> MonitorStatus {
        self.0.read().status()
    }

    /// Check if drift is detected.
    pub fn is_drift_detected(&self) -> bool {
        self.0.read().is_drift_detected()
    }

    /// Clone the Arc for sharing.
    pub fn clone_arc(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl Clone for SharedMonitor {
    fn clone(&self) -> Self {
        self.clone_arc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand_distr::{Distribution, Normal};

    fn generate_baseline(n: usize, dim: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        Array2::from_shape_fn((n, dim), |_| normal.sample(&mut rng))
    }

    fn generate_shifted_samples(n: usize, dim: usize, shift: f64, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(shift, 1.0).unwrap();

        Array2::from_shape_fn((n, dim), |_| normal.sample(&mut rng))
    }

    #[test]
    fn test_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = QuantumCoherenceMonitor::new(config).unwrap();

        assert_eq!(monitor.state(), MonitorState::Uninitialized);
        assert_eq!(monitor.n_samples(), 0);
    }

    #[test]
    fn test_monitor_requires_baseline() {
        let config = MonitorConfig::default();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let sample = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let result = monitor.observe(&sample);

        assert!(result.is_err());
    }

    #[test]
    fn test_baseline_setup() {
        let config = MonitorConfig::default();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let baseline = generate_baseline(30, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        assert_eq!(monitor.state(), MonitorState::Monitoring);
    }

    #[test]
    fn test_insufficient_baseline() {
        let config = MonitorConfig {
            min_baseline_samples: 50,
            ..Default::default()
        };
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let baseline = generate_baseline(20, 4, 42);
        let result = monitor.set_baseline(&baseline);

        assert!(result.is_err());
    }

    #[test]
    fn test_no_drift_detection() {
        let config = MonitorConfig {
            kernel: QuantumKernelConfig {
                n_qubits: 3,
                n_layers: 1,
                seed: Some(42),  // Fixed seed for reproducibility
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

        // Set baseline
        let baseline = generate_baseline(25, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Observe samples from same distribution (different seed)
        let samples = generate_baseline(30, 4, 123);
        let results = monitor.observe_batch(&samples).unwrap();

        // Check the final p-value - should not be extremely small under H_0
        let final_pvalue = monitor.current_p_value();

        // With kernel MMD, some variance is expected
        // The key is that p-value shouldn't be extremely small consistently
        assert!(
            final_pvalue > 1e-6 || results.iter().all(|r| !r.drift_detected),
            "P-value {} is too small under null hypothesis, drift_detected in {} samples",
            final_pvalue,
            results.iter().filter(|r| r.drift_detected).count()
        );
    }

    #[test]
    fn test_drift_detection() {
        let config = MonitorConfig::fast_detection();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        // Set baseline (mean = 0)
        let baseline = generate_baseline(20, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Observe samples from shifted distribution (mean = 3)
        let shifted = generate_shifted_samples(50, 4, 3.0, 123);
        let results = monitor.observe_batch(&shifted).unwrap();

        // Should eventually detect drift
        let final_result = results.last().unwrap();
        assert!(
            final_result.drift_detected || final_result.p_value < 0.1,
            "Should detect or suspect drift. P-value: {}, e-value: {}",
            final_result.p_value,
            final_result.evalue
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = MonitorConfig::default();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let baseline = generate_baseline(30, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Try to observe sample with wrong dimension
        let wrong_dim = Array1::from_vec(vec![0.1, 0.2]); // 2-dim instead of 4
        let result = monitor.observe(&wrong_dim);

        assert!(result.is_err());
    }

    #[test]
    fn test_monitor_reset() {
        let config = MonitorConfig::fast_detection();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let baseline = generate_baseline(20, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        // Observe some samples
        let samples = generate_baseline(10, 4, 123);
        monitor.observe_batch(&samples).unwrap();

        assert!(monitor.n_samples() > 0);

        // Reset
        monitor.reset().unwrap();
        assert_eq!(monitor.n_samples(), 0);
        assert_eq!(monitor.state(), MonitorState::Monitoring);
    }

    #[test]
    fn test_full_reset() {
        let config = MonitorConfig::fast_detection();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let baseline = generate_baseline(20, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        monitor.reset_full();

        assert_eq!(monitor.state(), MonitorState::Uninitialized);
    }

    #[test]
    fn test_status() {
        let config = MonitorConfig::fast_detection();
        let mut monitor = QuantumCoherenceMonitor::new(config).unwrap();

        let status = monitor.status();
        assert_eq!(status.state, MonitorState::Uninitialized);
        assert_eq!(status.n_baseline_samples, 0);

        let baseline = generate_baseline(20, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        let status = monitor.status();
        assert_eq!(status.state, MonitorState::Monitoring);
        assert_eq!(status.n_baseline_samples, 20);
    }

    #[test]
    fn test_shared_monitor() {
        let config = MonitorConfig::fast_detection();
        let monitor = SharedMonitor::new(config).unwrap();

        let baseline = generate_baseline(20, 4, 42);
        monitor.set_baseline(&baseline).unwrap();

        let sample = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let result = monitor.observe(&sample).unwrap();

        assert!(result.mmd_squared.is_finite());
        assert!(result.evalue > 0.0);
    }

    #[test]
    fn test_drift_severity() {
        assert_eq!(DriftSeverity::from_mmd_squared(0.05), DriftSeverity::Minor);
        assert_eq!(DriftSeverity::from_mmd_squared(0.3), DriftSeverity::Moderate);
        assert_eq!(DriftSeverity::from_mmd_squared(1.0), DriftSeverity::Severe);
    }

    #[test]
    fn test_config_presets() {
        let fast = MonitorConfig::fast_detection();
        let precise = MonitorConfig::high_precision();

        assert!(fast.min_baseline_samples < precise.min_baseline_samples);
        assert!(fast.evalue.alpha > precise.evalue.alpha);
    }
}
