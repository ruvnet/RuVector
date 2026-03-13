//! Calibration Engine for QAT (ADR-090 Phase 2)
//!
//! This module implements the calibration pipeline that initializes
//! quantization scales from representative data before QAT training.
//!
//! ## Calibration Process
//!
//! 1. **Collect Statistics**: Run forward passes on calibration data
//! 2. **Compute Ranges**: Per-layer min/max or percentile statistics
//! 3. **Initialize Scales**: Set optimal alpha values for Pi-quantization
//! 4. **Serialize Artifacts**: Save calibration state for reproducibility
//!
//! ## System Invariants
//!
//! - **INV-2**: Scale positivity - calibrated alpha > 0
//! - **INV-5**: Calibration artifacts are serializable and reproducible
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::qat::{CalibrationEngine, CalibrationConfig};
//!
//! let config = CalibrationConfig::default();
//! let mut engine = CalibrationEngine::new(config);
//!
//! // Feed calibration samples
//! for batch in calibration_data {
//!     engine.observe(&batch)?;
//! }
//!
//! // Compute final scales
//! let result = engine.finalize()?;
//! result.save("calibration.json")?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::time::Instant;

use crate::error::{Result, RuvLLMError};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of calibration samples
    pub num_samples: usize,
    /// Percentile for range estimation (e.g., 99.9 to exclude outliers)
    pub percentile: f32,
    /// Method for scale initialization
    pub method: CalibrationMethod,
    /// Whether to use per-channel calibration
    pub per_channel: bool,
    /// Minimum scale value (to ensure INV-2)
    pub min_scale: f32,
    /// Include tool-use samples in calibration
    pub include_tool_use: bool,
    /// Include reasoning samples in calibration
    pub include_reasoning: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            percentile: 99.9,
            method: CalibrationMethod::MinMax,
            per_channel: true,
            min_scale: 1e-8,
            include_tool_use: true,
            include_reasoning: true,
        }
    }
}

/// Calibration method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Use min/max of observed values
    MinMax,
    /// Use percentile-based range
    Percentile,
    /// Mean squared error minimization
    Mse,
    /// Entropy-based (for activation quantization)
    Entropy,
}

// ============================================================================
// Calibration Sample
// ============================================================================

/// A single calibration sample
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    /// Sample ID
    pub id: String,
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask
    pub attention_mask: Option<Vec<u8>>,
    /// Domain (reasoning, tool_use, general)
    pub domain: CalibrationDomain,
}

/// Domain type for mixed-domain calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationDomain {
    /// General text completion
    General,
    /// Reasoning tasks (GSM8K, etc.)
    Reasoning,
    /// Tool use / function calling
    ToolUse,
    /// Code generation
    Code,
}

// ============================================================================
// Statistics Collector
// ============================================================================

/// Statistics collected during calibration
#[derive(Debug, Clone, Default)]
struct LayerStats {
    /// Minimum observed value
    min: f32,
    /// Maximum observed value
    max: f32,
    /// Sum for mean calculation
    sum: f64,
    /// Sum of squares for variance
    sum_sq: f64,
    /// Number of observations
    count: usize,
    /// Sorted values for percentile (if using percentile method)
    values: Vec<f32>,
}

impl LayerStats {
    fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            values: Vec::new(),
        }
    }

    fn observe(&mut self, value: f32) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value as f64;
        self.sum_sq += (value as f64).powi(2);
        self.count += 1;
    }

    fn observe_for_percentile(&mut self, value: f32) {
        self.observe(value);
        self.values.push(value);
    }

    fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }

    fn std(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            let mean = self.sum / self.count as f64;
            let variance = (self.sum_sq / self.count as f64) - mean.powi(2);
            variance.sqrt() as f32
        }
    }

    fn percentile(&mut self, p: f32) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (self.values.len() - 1) as f32) as usize;
        self.values[idx.min(self.values.len() - 1)]
    }

    fn range(&self) -> f32 {
        self.max - self.min
    }
}

// ============================================================================
// Calibration Result
// ============================================================================

/// Result of calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Per-layer scales (layer_name -> scale)
    pub scales: HashMap<String, f32>,
    /// Per-layer zero points (layer_name -> zero)
    pub zeros: HashMap<String, f32>,
    /// Per-channel scales if per_channel=true
    pub channel_scales: Option<HashMap<String, Vec<f32>>>,
    /// Calibration configuration used
    pub config: CalibrationConfig,
    /// Statistics summary
    pub stats: CalibrationStats,
}

impl CalibrationResult {
    /// Save to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| RuvLLMError::Model(format!("Serialization failed: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| RuvLLMError::Model(format!("Write failed: {}", e)))?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| RuvLLMError::Model(format!("Read failed: {}", e)))?;
        let result = serde_json::from_str(&json)
            .map_err(|e| RuvLLMError::Model(format!("Deserialization failed: {}", e)))?;
        Ok(result)
    }

    /// Get scale for a layer
    pub fn get_scale(&self, layer: &str) -> Option<f32> {
        self.scales.get(layer).copied()
    }

    /// Get zero point for a layer
    pub fn get_zero(&self, layer: &str) -> Option<f32> {
        self.zeros.get(layer).copied()
    }
}

/// Calibration statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationStats {
    /// Number of samples processed
    pub samples_processed: usize,
    /// Number of layers calibrated
    pub layers_calibrated: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Domain breakdown
    pub domain_counts: HashMap<String, usize>,
}

// ============================================================================
// Calibration Engine
// ============================================================================

/// Main calibration engine
///
/// Collects activation statistics and computes optimal quantization scales.
pub struct CalibrationEngine {
    /// Configuration
    config: CalibrationConfig,
    /// Per-layer statistics
    layer_stats: HashMap<String, LayerStats>,
    /// Samples observed
    samples_observed: usize,
    /// Start time
    start_time: Option<Instant>,
    /// Domain counts
    domain_counts: HashMap<CalibrationDomain, usize>,
}

impl CalibrationEngine {
    /// Create a new calibration engine
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            samples_observed: 0,
            start_time: None,
            domain_counts: HashMap::new(),
        }
    }

    /// Start calibration
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Observe layer activations
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `activations` - Activation values
    pub fn observe_layer(&mut self, layer_name: &str, activations: &[f32]) {
        let stats = self
            .layer_stats
            .entry(layer_name.to_string())
            .or_insert_with(LayerStats::new);

        match self.config.method {
            CalibrationMethod::Percentile => {
                for &val in activations {
                    stats.observe_for_percentile(val);
                }
            }
            _ => {
                for &val in activations {
                    stats.observe(val);
                }
            }
        }
    }

    /// Observe a calibration sample
    ///
    /// This is called for each sample to track domain distribution.
    pub fn observe_sample(&mut self, sample: &CalibrationSample) {
        self.samples_observed += 1;
        *self.domain_counts.entry(sample.domain).or_insert(0) += 1;
    }

    /// Check if calibration is complete
    pub fn is_complete(&self) -> bool {
        self.samples_observed >= self.config.num_samples
    }

    /// Finalize calibration and compute scales
    pub fn finalize(&mut self) -> Result<CalibrationResult> {
        if self.layer_stats.is_empty() {
            return Err(RuvLLMError::Model(
                "No calibration data collected".to_string(),
            ));
        }

        let mut scales = HashMap::new();
        let mut zeros = HashMap::new();

        for (layer_name, stats) in &mut self.layer_stats {
            let (scale, zero) = match self.config.method {
                CalibrationMethod::MinMax => {
                    let range = stats.range();
                    let scale = (range / 7.0).max(self.config.min_scale); // 3-bit = 8 levels
                    let zero = stats.min;
                    (scale, zero)
                }
                CalibrationMethod::Percentile => {
                    let low = stats.percentile(100.0 - self.config.percentile);
                    let high = stats.percentile(self.config.percentile);
                    let range = high - low;
                    let scale = (range / 7.0).max(self.config.min_scale);
                    let zero = low;
                    (scale, zero)
                }
                CalibrationMethod::Mse => {
                    // Use 3-sigma rule for MSE-optimal range
                    let mean = stats.mean();
                    let std = stats.std();
                    let range = 6.0 * std; // 3-sigma on each side
                    let scale = (range / 7.0).max(self.config.min_scale);
                    let zero = mean - 3.0 * std;
                    (scale, zero)
                }
                CalibrationMethod::Entropy => {
                    // Simplified entropy-based (use histogram)
                    let range = stats.range();
                    let scale = (range / 7.0).max(self.config.min_scale);
                    let zero = stats.min;
                    (scale, zero)
                }
            };

            // Ensure INV-2: scale > 0
            debug_assert!(scale > 0.0, "INV-2: Scale must be positive");

            scales.insert(layer_name.clone(), scale);
            zeros.insert(layer_name.clone(), zero);
        }

        let duration_ms = self
            .start_time
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let domain_counts: HashMap<String, usize> = self
            .domain_counts
            .iter()
            .map(|(k, v)| (format!("{:?}", k), *v))
            .collect();

        Ok(CalibrationResult {
            scales,
            zeros,
            channel_scales: None, // TODO: Implement per-channel
            config: self.config.clone(),
            stats: CalibrationStats {
                samples_processed: self.samples_observed,
                layers_calibrated: self.layer_stats.len(),
                duration_ms,
                domain_counts,
            },
        })
    }

    /// Get current progress
    pub fn progress(&self) -> (usize, usize) {
        (self.samples_observed, self.config.num_samples)
    }

    /// Reset the engine for reuse
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.samples_observed = 0;
        self.start_time = None;
        self.domain_counts.clear();
    }
}

// ============================================================================
// Pi-Quantization Scale Initialization
// ============================================================================

/// Initialize Pi-quantization scale from calibration data
///
/// For Pi-quantization, the scale alpha is computed such that:
/// step_size = alpha * pi / k
///
/// # Arguments
///
/// * `range` - Value range from calibration
/// * `bits` - Number of quantization bits
/// * `k` - Pi divisor (typically 3-5)
///
/// # Returns
///
/// Optimal alpha value (guaranteed > 0 per INV-2)
pub fn init_pi_scale(range: f32, bits: u8, k: u8) -> f32 {
    let num_levels = (1 << bits) - 1; // e.g., 7 for 3-bit
    let step_size = range / num_levels as f32;

    // alpha = step_size * k / pi
    let alpha = (step_size * k as f32) / PI;

    // Ensure INV-2: alpha > 0
    alpha.max(1e-8)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_config() {
        let config = CalibrationConfig::default();
        assert_eq!(config.num_samples, 128);
        assert!(config.per_channel);
    }

    #[test]
    fn test_layer_stats() {
        let mut stats = LayerStats::new();
        for i in 0..100 {
            stats.observe(i as f32);
        }

        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 99.0);
        assert_eq!(stats.count, 100);
        assert!((stats.mean() - 49.5).abs() < 0.01);
    }

    #[test]
    fn test_calibration_engine() {
        let config = CalibrationConfig {
            num_samples: 10,
            ..Default::default()
        };
        let mut engine = CalibrationEngine::new(config);
        engine.start();

        // Observe some data
        let activations: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        engine.observe_layer("layer.0.attn.q_proj", &activations);
        engine.observe_layer("layer.0.attn.v_proj", &activations);

        let sample = CalibrationSample {
            id: "test".to_string(),
            input_ids: vec![1, 2, 3],
            attention_mask: None,
            domain: CalibrationDomain::General,
        };

        for _ in 0..10 {
            engine.observe_sample(&sample);
        }

        assert!(engine.is_complete());

        let result = engine.finalize().unwrap();
        assert_eq!(result.stats.samples_processed, 10);
        assert!(result.scales.contains_key("layer.0.attn.q_proj"));
    }

    #[test]
    fn test_init_pi_scale() {
        let range = 2.0;
        let alpha = init_pi_scale(range, 3, 4);

        // alpha should be positive (INV-2)
        assert!(alpha > 0.0);

        // Verify step size
        let step = alpha * PI / 4.0;
        assert!((step - range / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_calibration_result_serialization() {
        let mut scales = HashMap::new();
        scales.insert("layer.0".to_string(), 0.1);

        let result = CalibrationResult {
            scales,
            zeros: HashMap::new(),
            channel_scales: None,
            config: CalibrationConfig::default(),
            stats: CalibrationStats::default(),
        };

        // Test JSON serialization
        let json = serde_json::to_string(&result).unwrap();
        let restored: CalibrationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.scales.get("layer.0"), restored.scales.get("layer.0"));
    }
}
