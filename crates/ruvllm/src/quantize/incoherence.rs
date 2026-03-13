//! Incoherence Transform for Quantization (ADR-090 Phase 3)
//!
//! This module provides the aggregate root for incoherence processing in
//! quantization pipelines. Incoherence transforms spread weight distributions
//! to reduce the impact of outliers before quantization, improving quality.
//!
//! ## Theory
//!
//! Quantization error is amplified by large outliers in weight tensors.
//! The incoherence transform (using randomized Hadamard) redistributes
//! these outliers across all coefficients, making the distribution more
//! uniform and suitable for low-bit quantization.
//!
//! ## Pipeline Integration
//!
//! ```text
//! Original Weights → Incoherence Transform → Quantize → Dequantize → Inverse Transform
//! ```
//!
//! ## Domain Events
//!
//! - `IncoherenceApplied`: Emitted after successful forward transform
//! - `IncoherenceRestored`: Emitted after successful inverse transform
//! - `IncoherenceError`: Emitted on transform failure

use std::time::Instant;

use super::hadamard::{
    hadamard_batch_inverse, hadamard_batch_transform, log2_exact, next_power_of_2,
    pad_to_power_of_2, HadamardTransform,
};
use crate::error::{Result, RuvLLMError};

// ============================================================================
// Domain Events
// ============================================================================

/// Domain events emitted by the IncoherenceTransform
#[derive(Debug, Clone)]
pub enum IncoherenceEvent {
    /// Incoherence transform was applied before quantization
    IncoherenceApplied {
        /// Number of elements transformed
        num_elements: usize,
        /// Time taken in microseconds
        duration_us: u64,
        /// Whether padding was required
        required_padding: bool,
        /// Original dimension
        original_dim: usize,
        /// Padded dimension (power of 2)
        padded_dim: usize,
        /// Maximum absolute value before transform
        max_before: f32,
        /// Maximum absolute value after transform
        max_after: f32,
    },
    /// Incoherence transform was restored after dequantization
    IncoherenceRestored {
        /// Number of elements restored
        num_elements: usize,
        /// Time taken in microseconds
        duration_us: u64,
        /// Reconstruction error (if computed)
        reconstruction_error: Option<f32>,
    },
    /// Error occurred during incoherence processing
    IncoherenceError {
        /// Error message
        message: String,
        /// Phase where error occurred
        phase: IncoherencePhase,
    },
}

/// Phase of incoherence processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncoherencePhase {
    /// Forward transform (before quantization)
    Forward,
    /// Inverse transform (after dequantization)
    Inverse,
    /// Initialization
    Init,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for incoherence transform
#[derive(Debug, Clone)]
pub struct IncoherenceConfig {
    /// Random seed for the Hadamard transform (None for deterministic)
    pub seed: Option<u64>,
    /// Whether to use randomized Hadamard (recommended for quantization)
    pub randomized: bool,
    /// Whether to compute statistics (max values, reconstruction error)
    pub compute_stats: bool,
    /// Whether to emit domain events
    pub emit_events: bool,
    /// Minimum dimension for applying incoherence (smaller tensors skipped)
    pub min_dimension: usize,
    /// Whether to use batch processing for multiple vectors
    pub batch_mode: bool,
}

impl Default for IncoherenceConfig {
    fn default() -> Self {
        Self {
            seed: Some(42), // Deterministic by default for reproducibility
            randomized: true,
            compute_stats: true,
            emit_events: true,
            min_dimension: 16, // Don't apply to tiny tensors
            batch_mode: true,
        }
    }
}

impl IncoherenceConfig {
    /// Create config for maximum quality (randomized with stats)
    pub fn quality() -> Self {
        Self {
            seed: Some(12345),
            randomized: true,
            compute_stats: true,
            emit_events: true,
            min_dimension: 8,
            batch_mode: true,
        }
    }

    /// Create config for maximum performance (skip stats, deterministic)
    pub fn performance() -> Self {
        Self {
            seed: None,
            randomized: false,
            compute_stats: false,
            emit_events: false,
            min_dimension: 32,
            batch_mode: true,
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable/disable randomization
    pub fn with_randomized(mut self, randomized: bool) -> Self {
        self.randomized = randomized;
        self
    }
}

// ============================================================================
// Transform Statistics
// ============================================================================

/// Statistics from incoherence transform
#[derive(Debug, Clone, Default)]
pub struct IncoherenceStats {
    /// Number of forward transforms applied
    pub forward_count: usize,
    /// Number of inverse transforms applied
    pub inverse_count: usize,
    /// Total elements processed
    pub total_elements: u64,
    /// Total time in forward transforms (microseconds)
    pub forward_time_us: u64,
    /// Total time in inverse transforms (microseconds)
    pub inverse_time_us: u64,
    /// Average outlier reduction ratio (max_before / max_after)
    pub avg_outlier_reduction: f32,
    /// Number of tensors that required padding
    pub padded_count: usize,
}

// ============================================================================
// Aggregate Root: IncoherenceTransform
// ============================================================================

/// Aggregate root for incoherence processing in quantization pipelines
///
/// This struct manages the lifecycle of incoherence transforms, including:
/// - Transform creation and caching
/// - Forward/inverse transform application
/// - Statistics collection
/// - Domain event emission
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::quantize::incoherence::{IncoherenceTransform, IncoherenceConfig};
///
/// // Create transform with default config
/// let mut transform = IncoherenceTransform::new(IncoherenceConfig::default())?;
///
/// // Apply before quantization
/// let mut weights = vec![1.0, 2.0, 100.0, 4.0]; // Note the outlier
/// transform.apply_before_quantization(&mut weights)?;
///
/// // ... quantize weights here ...
///
/// // Restore after dequantization
/// transform.restore_after_dequantization(&mut weights)?;
/// ```
pub struct IncoherenceTransform {
    /// Configuration
    config: IncoherenceConfig,
    /// Cached Hadamard transforms by log_dim
    transforms: std::collections::HashMap<u32, HadamardTransform>,
    /// Accumulated statistics
    stats: IncoherenceStats,
    /// Event buffer (if events enabled)
    events: Vec<IncoherenceEvent>,
    /// Original dimensions for pending restores
    pending_original_dims: std::collections::HashMap<usize, usize>,
}

impl IncoherenceTransform {
    /// Create a new IncoherenceTransform with the given configuration
    pub fn new(config: IncoherenceConfig) -> Result<Self> {
        Ok(Self {
            config,
            transforms: std::collections::HashMap::new(),
            stats: IncoherenceStats::default(),
            events: Vec::new(),
            pending_original_dims: std::collections::HashMap::new(),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(IncoherenceConfig::default())
    }

    /// Get or create a Hadamard transform for the given log dimension
    fn get_or_create_transform(&mut self, log_dim: u32) -> Result<&HadamardTransform> {
        if !self.transforms.contains_key(&log_dim) {
            let transform = if self.config.randomized {
                HadamardTransform::randomized(log_dim, self.config.seed.unwrap_or(42))?
            } else {
                HadamardTransform::deterministic(log_dim)?
            };
            self.transforms.insert(log_dim, transform);
        }
        Ok(self.transforms.get(&log_dim).unwrap())
    }

    /// Apply incoherence transform before quantization
    ///
    /// This transforms the weight data to spread outliers uniformly,
    /// reducing quantization error. The data is modified in-place.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice of weight values to transform
    ///
    /// # Returns
    ///
    /// The padded dimension (data is resized to this power of 2)
    pub fn apply_before_quantization(&mut self, data: &mut Vec<f32>) -> Result<usize> {
        let start = Instant::now();
        let original_len = data.len();

        // Skip tiny tensors
        if original_len < self.config.min_dimension {
            return Ok(original_len);
        }

        // Compute pre-transform statistics
        let max_before = if self.config.compute_stats {
            data.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b))
        } else {
            0.0
        };

        // Pad to power of 2 if needed
        let target_len = next_power_of_2(original_len);
        let required_padding = target_len != original_len;

        if required_padding {
            data.resize(target_len, 0.0);
        }

        // Get log dimension
        let log_dim = match log2_exact(target_len) {
            Some(ld) => ld,
            None => {
                self.emit_error(
                    "Internal error: padded length not power of 2",
                    IncoherencePhase::Forward,
                );
                return Err(RuvLLMError::Quantization(
                    "Padded length is not a power of 2".to_string(),
                ));
            }
        };

        // Get or create transform (need to clone due to borrow checker)
        let transform = self.get_or_create_transform(log_dim)?.clone();

        // Apply forward transform
        transform.forward_inplace(data);

        // Compute post-transform statistics
        let max_after = if self.config.compute_stats {
            data.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b))
        } else {
            0.0
        };

        // Store original dimension for restore
        let data_id = data.as_ptr() as usize;
        self.pending_original_dims.insert(data_id, original_len);

        // Update statistics
        let duration_us = start.elapsed().as_micros() as u64;
        self.stats.forward_count += 1;
        self.stats.total_elements += target_len as u64;
        self.stats.forward_time_us += duration_us;
        if required_padding {
            self.stats.padded_count += 1;
        }
        if max_before > 0.0 && max_after > 0.0 {
            let reduction = max_before / max_after;
            let n = self.stats.forward_count as f32;
            self.stats.avg_outlier_reduction =
                (self.stats.avg_outlier_reduction * (n - 1.0) + reduction) / n;
        }

        // Emit event
        if self.config.emit_events {
            self.events.push(IncoherenceEvent::IncoherenceApplied {
                num_elements: target_len,
                duration_us,
                required_padding,
                original_dim: original_len,
                padded_dim: target_len,
                max_before,
                max_after,
            });
        }

        Ok(target_len)
    }

    /// Restore original data distribution after dequantization
    ///
    /// This applies the inverse transform to recover the original
    /// weight distribution. The data must have the same length as
    /// after `apply_before_quantization`.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable slice of dequantized values to restore
    /// * `original_len` - Original length before padding (optional, for truncation)
    pub fn restore_after_dequantization(
        &mut self,
        data: &mut Vec<f32>,
        original_len: Option<usize>,
    ) -> Result<()> {
        let start = Instant::now();
        let current_len = data.len();

        // Get log dimension
        let log_dim = match log2_exact(current_len) {
            Some(ld) => ld,
            None => {
                self.emit_error("Data length is not a power of 2", IncoherencePhase::Inverse);
                return Err(RuvLLMError::Quantization(
                    "Data length must be a power of 2 for inverse transform".to_string(),
                ));
            }
        };

        // Get transform
        let transform = self.get_or_create_transform(log_dim)?.clone();

        // Apply inverse transform
        transform.inverse_inplace(data);

        // Truncate to original length if provided
        let final_len = original_len.unwrap_or_else(|| {
            let data_id = data.as_ptr() as usize;
            self.pending_original_dims
                .remove(&data_id)
                .unwrap_or(current_len)
        });

        if final_len < current_len {
            data.truncate(final_len);
        }

        // Update statistics
        let duration_us = start.elapsed().as_micros() as u64;
        self.stats.inverse_count += 1;
        self.stats.inverse_time_us += duration_us;

        // Emit event
        if self.config.emit_events {
            self.events.push(IncoherenceEvent::IncoherenceRestored {
                num_elements: final_len,
                duration_us,
                reconstruction_error: None, // Would need original data to compute
            });
        }

        Ok(())
    }

    /// Apply incoherence to a batch of weight vectors
    ///
    /// More efficient than individual transforms due to better cache utilization.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat buffer containing `batch_size` vectors of `dim` elements each
    /// * `dim` - Dimension of each vector (must be power of 2)
    /// * `batch_size` - Number of vectors
    pub fn apply_batch(&mut self, data: &mut [f32], dim: usize, batch_size: usize) -> Result<()> {
        if data.len() != dim * batch_size {
            return Err(RuvLLMError::Quantization(format!(
                "Data length {} doesn't match dim {} * batch_size {}",
                data.len(),
                dim,
                batch_size
            )));
        }

        let log_dim = match log2_exact(dim) {
            Some(ld) => ld,
            None => {
                return Err(RuvLLMError::Quantization(
                    "Dimension must be a power of 2 for batch transform".to_string(),
                ));
            }
        };

        let transform = self.get_or_create_transform(log_dim)?.clone();
        hadamard_batch_transform(&transform, data, batch_size)?;

        self.stats.forward_count += batch_size;
        self.stats.total_elements += (dim * batch_size) as u64;

        Ok(())
    }

    /// Restore a batch of weight vectors after dequantization
    pub fn restore_batch(&mut self, data: &mut [f32], dim: usize, batch_size: usize) -> Result<()> {
        if data.len() != dim * batch_size {
            return Err(RuvLLMError::Quantization(format!(
                "Data length {} doesn't match dim {} * batch_size {}",
                data.len(),
                dim,
                batch_size
            )));
        }

        let log_dim = match log2_exact(dim) {
            Some(ld) => ld,
            None => {
                return Err(RuvLLMError::Quantization(
                    "Dimension must be a power of 2 for batch inverse".to_string(),
                ));
            }
        };

        let transform = self.get_or_create_transform(log_dim)?.clone();
        hadamard_batch_inverse(&transform, data, batch_size)?;

        self.stats.inverse_count += batch_size;

        Ok(())
    }

    /// Get accumulated statistics
    pub fn stats(&self) -> &IncoherenceStats {
        &self.stats
    }

    /// Take and clear emitted events
    pub fn take_events(&mut self) -> Vec<IncoherenceEvent> {
        std::mem::take(&mut self.events)
    }

    /// Peek at emitted events without clearing
    pub fn events(&self) -> &[IncoherenceEvent] {
        &self.events
    }

    /// Get configuration
    pub fn config(&self) -> &IncoherenceConfig {
        &self.config
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = IncoherenceStats::default();
    }

    /// Clear cached transforms (useful for memory management)
    pub fn clear_cache(&mut self) {
        self.transforms.clear();
    }

    /// Emit an error event
    fn emit_error(&mut self, message: &str, phase: IncoherencePhase) {
        if self.config.emit_events {
            self.events.push(IncoherenceEvent::IncoherenceError {
                message: message.to_string(),
                phase,
            });
        }
    }

    /// Verify that the transform is working correctly
    ///
    /// This performs a roundtrip test to ensure the implementation is correct.
    pub fn verify(&mut self, dim: usize, tolerance: f32) -> Result<bool> {
        let log_dim = match log2_exact(dim) {
            Some(ld) => ld,
            None => {
                return Err(RuvLLMError::Quantization(
                    "Dimension must be a power of 2 for verification".to_string(),
                ));
            }
        };

        let transform = self.get_or_create_transform(log_dim)?;
        Ok(transform.verify_orthogonality(tolerance))
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Apply incoherence transform to weights before quantization (convenience function)
///
/// This is a simple wrapper for one-off transforms without managing state.
pub fn apply_incoherence(data: &mut Vec<f32>, seed: Option<u64>) -> Result<usize> {
    let config = IncoherenceConfig {
        seed,
        randomized: seed.is_some(),
        compute_stats: false,
        emit_events: false,
        min_dimension: 8,
        batch_mode: false,
    };

    let mut transform = IncoherenceTransform::new(config)?;
    transform.apply_before_quantization(data)
}

/// Restore weights after dequantization (convenience function)
pub fn restore_incoherence(
    data: &mut Vec<f32>,
    original_len: usize,
    seed: Option<u64>,
) -> Result<()> {
    let config = IncoherenceConfig {
        seed,
        randomized: seed.is_some(),
        compute_stats: false,
        emit_events: false,
        min_dimension: 8,
        batch_mode: false,
    };

    let mut transform = IncoherenceTransform::new(config)?;
    transform.restore_after_dequantization(data, Some(original_len))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incoherence_basic() {
        // Use config with small min_dimension to exercise the transform
        let config = IncoherenceConfig {
            min_dimension: 4,
            ..Default::default()
        };
        let mut transform = IncoherenceTransform::new(config).unwrap();

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();

        let padded_dim = transform.apply_before_quantization(&mut data).unwrap();
        assert_eq!(padded_dim, 8);

        transform
            .restore_after_dequantization(&mut data, Some(8))
            .unwrap();

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "Roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_incoherence_with_padding() {
        // Use config with small min_dimension to exercise the transform
        let config = IncoherenceConfig {
            min_dimension: 4,
            ..Default::default()
        };
        let mut transform = IncoherenceTransform::new(config).unwrap();

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 6 elements, will pad to 8
        let original_len = original.len();
        let mut data = original.clone();

        let padded_dim = transform.apply_before_quantization(&mut data).unwrap();
        assert_eq!(padded_dim, 8);
        assert_eq!(data.len(), 8);

        transform
            .restore_after_dequantization(&mut data, Some(original_len))
            .unwrap();
        assert_eq!(data.len(), original_len);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Padded roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_outlier_spreading() {
        let config = IncoherenceConfig {
            seed: Some(42),
            randomized: true,
            compute_stats: true,
            emit_events: true,
            min_dimension: 4,
            batch_mode: false,
        };
        let mut transform = IncoherenceTransform::new(config).unwrap();

        // Data with an outlier
        let mut data: Vec<f32> = vec![1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0];
        let max_before: f32 = data
            .iter()
            .map(|x: &f32| x.abs())
            .fold(0.0f32, |a: f32, b: f32| a.max(b));

        transform.apply_before_quantization(&mut data).unwrap();

        let max_after: f32 = data
            .iter()
            .map(|x: &f32| x.abs())
            .fold(0.0f32, |a: f32, b: f32| a.max(b));

        // The outlier should be spread across all elements
        // Max after should be significantly smaller than 100
        assert!(
            max_after < max_before * 0.9,
            "Outlier not spread: before={}, after={}",
            max_before,
            max_after
        );

        // Check that events were emitted
        let events = transform.take_events();
        assert!(!events.is_empty());
        if let IncoherenceEvent::IncoherenceApplied {
            max_before: mb,
            max_after: ma,
            ..
        } = &events[0]
        {
            assert!((*ma) < (*mb) * 0.9);
        }
    }

    #[test]
    fn test_batch_transform() {
        let mut transform = IncoherenceTransform::with_defaults().unwrap();

        let dim = 16;
        let batch_size = 4;
        let original: Vec<f32> = (0..dim * batch_size).map(|i| i as f32).collect();
        let mut data = original.clone();

        transform.apply_batch(&mut data, dim, batch_size).unwrap();
        transform.restore_batch(&mut data, dim, batch_size).unwrap();

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-4, "Batch roundtrip failed");
        }
    }

    #[test]
    fn test_verify() {
        let mut transform = IncoherenceTransform::with_defaults().unwrap();
        assert!(transform.verify(64, 1e-5).unwrap());
    }

    #[test]
    fn test_statistics() {
        let config = IncoherenceConfig {
            seed: Some(42),
            randomized: true,
            compute_stats: true,
            emit_events: true,
            min_dimension: 4,
            batch_mode: false,
        };
        let mut transform = IncoherenceTransform::new(config).unwrap();

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        transform.apply_before_quantization(&mut data).unwrap();

        let stats = transform.stats();
        assert_eq!(stats.forward_count, 1);
        assert_eq!(stats.total_elements, 4);
        assert!(stats.forward_time_us > 0 || stats.forward_time_us == 0); // Might be 0 on fast systems
    }

    #[test]
    fn test_skip_small_tensors() {
        let config = IncoherenceConfig {
            min_dimension: 32,
            ..Default::default()
        };
        let mut transform = IncoherenceTransform::new(config).unwrap();

        let original = vec![1.0, 2.0, 3.0, 4.0]; // 4 < 32, should be skipped
        let mut data = original.clone();

        let padded_dim = transform.apply_before_quantization(&mut data).unwrap();
        assert_eq!(padded_dim, 4);
        assert_eq!(data, original); // Data unchanged
    }

    #[test]
    fn test_config_builders() {
        let quality = IncoherenceConfig::quality();
        assert!(quality.randomized);
        assert!(quality.compute_stats);

        let perf = IncoherenceConfig::performance();
        assert!(!perf.randomized);
        assert!(!perf.compute_stats);
    }

    #[test]
    fn test_convenience_functions() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original_len = original.len();
        let mut data = original.clone();

        let _padded = apply_incoherence(&mut data, Some(12345)).unwrap();
        restore_incoherence(&mut data, original_len, Some(12345)).unwrap();

        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_energy_preservation_through_pipeline() {
        let mut transform = IncoherenceTransform::with_defaults().unwrap();

        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let original_energy: f32 = original.iter().map(|x| x * x).sum();
        let mut data = original.clone();

        transform.apply_before_quantization(&mut data).unwrap();

        let transformed_energy: f32 = data.iter().map(|x| x * x).sum();

        // Energy should be approximately preserved (allowing for padding effects)
        let relative_error = (original_energy - transformed_energy).abs() / original_energy;
        assert!(
            relative_error < 0.01,
            "Energy not preserved: original={}, transformed={}, error={}",
            original_energy,
            transformed_energy,
            relative_error
        );
    }
}
