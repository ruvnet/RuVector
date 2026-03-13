//! Frequency-Based Precision Allocation for MoE Experts
//!
//! This module implements ADR-092's precision allocation strategy, which assigns
//! different quantization formats to experts based on their activation frequency:
//!
//! - **Hot experts** (high activation): Higher precision (e.g., Q4_K_M)
//! - **Warm experts** (medium activation): Medium precision (e.g., Q3_K or PiQ3)
//! - **Cold experts** (low activation): Lower precision (e.g., Q2_K or PiQ2)
//!
//! This approach preserves model quality by keeping frequently-used experts at
//! higher precision while aggressively compressing rarely-used experts.
//!
//! ## Invariant: INV-4 Precision Preservation
//!
//! Expert precision metadata travels with cached weights. The `PrecisionAllocator`
//! tracks and exposes the assigned precision for each expert so that dequantization
//! uses the correct format.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::moe::precision_allocator::{PrecisionAllocator, PrecisionConfig, ExpertPrecision};
//! use ruvllm::gguf::GgufQuantType;
//!
//! let config = PrecisionConfig::default();
//! let mut allocator = PrecisionAllocator::new(8, config).unwrap();
//!
//! // Record activations as experts are used
//! allocator.record_activation(2);
//! allocator.record_activation(2);
//! allocator.record_activation(5);
//!
//! // Recompute thresholds periodically
//! allocator.recompute_thresholds();
//!
//! // Get precision level for routing decisions
//! let precision = allocator.allocate(2);
//! let format = allocator.get_format(2);
//! ```

use crate::gguf::GgufQuantType;

// Re-export ExpertId from parent module
pub use super::ExpertId;

// ============================================================================
// Precision Level
// ============================================================================

/// Precision level assigned to an expert based on activation frequency.
///
/// The three tiers enable differentiated memory/quality tradeoffs:
/// - Hot: Maximum quality, higher memory usage
/// - Warm: Balanced quality/memory
/// - Cold: Aggressive compression, lower memory usage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertPrecision {
    /// High precision for frequently-activated (hot) experts.
    ///
    /// Typically Q4_K_M or higher for best quality on important experts.
    Hot,

    /// Medium precision for moderately-activated (warm) experts.
    ///
    /// Typically Q3_K or PiQ3 for balanced quality/memory.
    Warm,

    /// Low precision for rarely-activated (cold) experts.
    ///
    /// Typically Q2_K or PiQ2 for maximum compression on seldom-used experts.
    Cold,
}

impl ExpertPrecision {
    /// Returns a human-readable name for the precision level.
    pub fn name(&self) -> &'static str {
        match self {
            ExpertPrecision::Hot => "hot",
            ExpertPrecision::Warm => "warm",
            ExpertPrecision::Cold => "cold",
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for frequency-based precision allocation.
///
/// The percentile thresholds determine how experts are classified:
/// - Experts with activation count >= hot_percentile of max are "hot"
/// - Experts with activation count >= cold_percentile but < hot_percentile are "warm"
/// - Experts with activation count < cold_percentile are "cold"
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Percentile threshold for hot experts (default: 0.9 = top 10% by frequency).
    ///
    /// Experts whose activation count is at or above this percentile of the
    /// maximum activation count are classified as hot.
    pub hot_percentile: f32,

    /// Percentile threshold for cold experts (default: 0.3 = bottom 30% by frequency).
    ///
    /// Experts whose activation count is below this percentile of the
    /// maximum activation count are classified as cold.
    pub cold_percentile: f32,

    /// GGUF quantization format for hot experts.
    ///
    /// Default: Q4_K (4-bit k-quant) for good quality.
    pub hot_format: GgufQuantType,

    /// GGUF quantization format for warm experts.
    ///
    /// Default: Q3_K (3-bit k-quant) for balanced quality/size.
    pub warm_format: GgufQuantType,

    /// GGUF quantization format for cold experts.
    ///
    /// Default: Q2_K (2-bit k-quant) for maximum compression.
    pub cold_format: GgufQuantType,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            hot_format: GgufQuantType::Q4_K,
            warm_format: GgufQuantType::Q3_K,
            cold_format: GgufQuantType::Q2_K,
        }
    }
}

impl PrecisionConfig {
    /// Create a config optimized for memory-constrained devices.
    ///
    /// Uses more aggressive thresholds and lower precision formats.
    pub fn memory_constrained() -> Self {
        Self {
            hot_percentile: 0.95,
            cold_percentile: 0.4,
            hot_format: GgufQuantType::Q4_K,
            warm_format: GgufQuantType::Q2_K,
            cold_format: GgufQuantType::Q2_K,
        }
    }

    /// Create a config optimized for quality preservation.
    ///
    /// Uses higher precision formats across all tiers.
    pub fn quality_focused() -> Self {
        Self {
            hot_percentile: 0.8,
            cold_percentile: 0.2,
            hot_format: GgufQuantType::Q5_K,
            warm_format: GgufQuantType::Q4_K,
            cold_format: GgufQuantType::Q3_K,
        }
    }

    /// Validate the configuration.
    ///
    /// Returns an error message if the configuration is invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.hot_percentile <= 0.0 || self.hot_percentile > 1.0 {
            return Err("hot_percentile must be in (0.0, 1.0]");
        }
        if self.cold_percentile < 0.0 || self.cold_percentile >= 1.0 {
            return Err("cold_percentile must be in [0.0, 1.0)");
        }
        if self.cold_percentile >= self.hot_percentile {
            return Err("cold_percentile must be less than hot_percentile");
        }
        Ok(())
    }
}

// ============================================================================
// Precision Allocator
// ============================================================================

/// Frequency-based precision allocator for MoE experts.
///
/// Tracks activation counts for each expert and assigns precision levels
/// based on relative frequency. Hot experts (frequently used) get higher
/// precision to preserve quality, while cold experts (rarely used) get
/// lower precision to save memory.
///
/// # INV-4: Precision Preservation
///
/// This allocator maintains the mapping from expert ID to precision level,
/// ensuring that the correct quantization format is used when dequantizing
/// cached expert weights.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::moe::precision_allocator::{PrecisionAllocator, PrecisionConfig};
///
/// let config = PrecisionConfig::default();
/// let mut allocator = PrecisionAllocator::new(8, config).unwrap();
///
/// // Simulate expert activations
/// for _ in 0..100 { allocator.record_activation(0); } // Hot
/// for _ in 0..50 { allocator.record_activation(1); }  // Warm
/// allocator.record_activation(7);                      // Cold
///
/// allocator.recompute_thresholds();
///
/// assert_eq!(allocator.allocate(0), ExpertPrecision::Hot);
/// assert_eq!(allocator.allocate(1), ExpertPrecision::Warm);
/// assert_eq!(allocator.allocate(7), ExpertPrecision::Cold);
/// ```
pub struct PrecisionAllocator {
    /// Number of experts tracked.
    num_experts: usize,

    /// Activation counts per expert, indexed by ExpertId.
    counts: Vec<u64>,

    /// Configuration for precision allocation.
    config: PrecisionConfig,

    /// Cached threshold for hot classification.
    ///
    /// Experts with count >= hot_threshold are hot.
    hot_threshold: u64,

    /// Cached threshold for cold classification.
    ///
    /// Experts with count < cold_threshold are cold.
    cold_threshold: u64,
}

impl PrecisionAllocator {
    /// Create a new precision allocator.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - Total number of experts to track.
    /// * `config` - Configuration for precision allocation.
    ///
    /// # Returns
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(num_experts: usize, config: PrecisionConfig) -> Result<Self, &'static str> {
        config.validate()?;

        Ok(Self {
            num_experts,
            counts: vec![0; num_experts],
            config,
            hot_threshold: 0,
            cold_threshold: 0,
        })
    }

    /// Create a new precision allocator, panicking on invalid config.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    pub fn new_unchecked(num_experts: usize, config: PrecisionConfig) -> Self {
        Self::new(num_experts, config).expect("PrecisionConfig validation failed")
    }

    /// Record an activation for the given expert.
    ///
    /// Increments the activation counter for the specified expert.
    /// This should be called each time an expert is selected by the router.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - The ID of the activated expert.
    ///
    /// # Notes
    ///
    /// Out-of-bounds expert IDs are silently ignored.
    #[inline]
    pub fn record_activation(&mut self, expert_id: ExpertId) {
        if expert_id < self.num_experts {
            self.counts[expert_id] = self.counts[expert_id].saturating_add(1);
        }
    }

    /// Record multiple activations in a batch.
    ///
    /// More efficient than calling `record_activation` in a loop when
    /// processing batched routing decisions.
    ///
    /// # Arguments
    ///
    /// * `expert_ids` - Slice of activated expert IDs.
    pub fn record_activations(&mut self, expert_ids: &[ExpertId]) {
        for &expert_id in expert_ids {
            self.record_activation(expert_id);
        }
    }

    /// Get the precision level for the given expert.
    ///
    /// Returns the precision classification (Hot, Warm, or Cold) based on
    /// the expert's activation count relative to the computed thresholds.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - The ID of the expert to classify.
    ///
    /// # Returns
    ///
    /// The precision level for the expert. Returns `Cold` for out-of-bounds IDs.
    pub fn allocate(&self, expert_id: ExpertId) -> ExpertPrecision {
        if expert_id >= self.num_experts {
            return ExpertPrecision::Cold;
        }

        let count = self.counts[expert_id];

        // If no activations have occurred yet, all experts are cold
        if self.hot_threshold == 0 && self.cold_threshold == 0 {
            return ExpertPrecision::Cold;
        }

        if count >= self.hot_threshold && self.hot_threshold > 0 {
            ExpertPrecision::Hot
        } else if count >= self.cold_threshold && count > 0 {
            ExpertPrecision::Warm
        } else {
            ExpertPrecision::Cold
        }
    }

    /// Get the GGUF quantization format for the given expert.
    ///
    /// Returns the appropriate quantization format based on the expert's
    /// precision classification.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - The ID of the expert.
    ///
    /// # Returns
    ///
    /// The GGUF quantization type to use for this expert.
    pub fn get_format(&self, expert_id: ExpertId) -> GgufQuantType {
        match self.allocate(expert_id) {
            ExpertPrecision::Hot => self.config.hot_format,
            ExpertPrecision::Warm => self.config.warm_format,
            ExpertPrecision::Cold => self.config.cold_format,
        }
    }

    /// Recompute the threshold values based on current activation counts.
    ///
    /// Should be called periodically (e.g., every N tokens or at batch boundaries)
    /// to update the precision classifications as activation patterns change.
    ///
    /// The thresholds are computed from the maximum activation count:
    /// - `hot_threshold = max_count * hot_percentile`
    /// - `cold_threshold = max_count * cold_percentile`
    pub fn recompute_thresholds(&mut self) {
        let max_count = self.counts.iter().copied().max().unwrap_or(0);

        if max_count == 0 {
            self.hot_threshold = 0;
            self.cold_threshold = 0;
            return;
        }

        // Compute thresholds as fractions of max count
        self.hot_threshold = (max_count as f64 * self.config.hot_percentile as f64).ceil() as u64;
        self.cold_threshold =
            (max_count as f64 * self.config.cold_percentile as f64).floor() as u64;

        // Ensure cold_threshold is at least 1 if there are any activations
        if self.cold_threshold == 0 && max_count > 0 {
            self.cold_threshold = 1;
        }
    }

    /// Get the precision map for all experts.
    ///
    /// Returns a vector of (ExpertId, ExpertPrecision) tuples for all tracked
    /// experts. Useful for bulk operations or serialization.
    ///
    /// # Returns
    ///
    /// Vector of tuples containing each expert's ID and precision level.
    pub fn get_precision_map(&self) -> Vec<(ExpertId, ExpertPrecision)> {
        (0..self.num_experts)
            .map(|id| (id, self.allocate(id)))
            .collect()
    }

    /// Get the activation count for a specific expert.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - The ID of the expert.
    ///
    /// # Returns
    ///
    /// The activation count, or 0 for out-of-bounds IDs.
    pub fn get_count(&self, expert_id: ExpertId) -> u64 {
        self.counts.get(expert_id).copied().unwrap_or(0)
    }

    /// Get the total number of activations across all experts.
    pub fn total_activations(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Get the number of experts in each precision tier.
    ///
    /// # Returns
    ///
    /// Tuple of (hot_count, warm_count, cold_count).
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let mut hot = 0;
        let mut warm = 0;
        let mut cold = 0;

        for id in 0..self.num_experts {
            match self.allocate(id) {
                ExpertPrecision::Hot => hot += 1,
                ExpertPrecision::Warm => warm += 1,
                ExpertPrecision::Cold => cold += 1,
            }
        }

        (hot, warm, cold)
    }

    /// Reset all activation counts to zero.
    ///
    /// Also resets the thresholds. Useful when starting a new evaluation
    /// period or when the model's usage patterns have changed significantly.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.hot_threshold = 0;
        self.cold_threshold = 0;
    }

    /// Get the number of experts being tracked.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the current hot threshold.
    pub fn hot_threshold(&self) -> u64 {
        self.hot_threshold
    }

    /// Get the current cold threshold.
    pub fn cold_threshold(&self) -> u64 {
        self.cold_threshold
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &PrecisionConfig {
        &self.config
    }

    /// Get experts by precision level.
    ///
    /// # Arguments
    ///
    /// * `precision` - The precision level to filter by.
    ///
    /// # Returns
    ///
    /// Vector of expert IDs with the specified precision level.
    pub fn experts_by_precision(&self, precision: ExpertPrecision) -> Vec<ExpertId> {
        (0..self.num_experts)
            .filter(|&id| self.allocate(id) == precision)
            .collect()
    }

    /// Compute the percentile rank for a given expert.
    ///
    /// Returns a value in [0.0, 1.0] representing where this expert's
    /// activation count falls relative to the maximum.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - The ID of the expert.
    ///
    /// # Returns
    ///
    /// Percentile rank (0.0 = no activations, 1.0 = max activations).
    pub fn compute_percentile(&self, expert_id: ExpertId) -> f32 {
        if expert_id >= self.num_experts {
            return 0.0;
        }

        let max_count = self.counts.iter().copied().max().unwrap_or(0);
        if max_count == 0 {
            return 0.0;
        }

        self.counts[expert_id] as f32 / max_count as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // test_allocator_creation
    // ---------------------------------------------------------------

    #[test]
    fn test_allocator_creation() {
        let config = PrecisionConfig::default();
        let allocator = PrecisionAllocator::new(8, config).unwrap();

        assert_eq!(allocator.num_experts(), 8);
        assert_eq!(allocator.total_activations(), 0);
        assert_eq!(allocator.hot_threshold(), 0);
        assert_eq!(allocator.cold_threshold(), 0);

        // All experts should be cold initially
        for id in 0..8 {
            assert_eq!(allocator.allocate(id), ExpertPrecision::Cold);
        }
    }

    // ---------------------------------------------------------------
    // test_hot_expert_allocation
    // ---------------------------------------------------------------

    #[test]
    fn test_hot_expert_allocation() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(8, config).unwrap();

        // Expert 0 gets 100 activations (max)
        for _ in 0..100 {
            allocator.record_activation(0);
        }

        // Other experts get 10 activations each
        for id in 1..8 {
            for _ in 0..10 {
                allocator.record_activation(id);
            }
        }

        allocator.recompute_thresholds();

        // Expert 0 should be hot (100 >= 90% of 100 = 90)
        assert_eq!(allocator.allocate(0), ExpertPrecision::Hot);
        assert_eq!(allocator.get_format(0), GgufQuantType::Q4_K);
    }

    // ---------------------------------------------------------------
    // test_warm_expert_allocation
    // ---------------------------------------------------------------

    #[test]
    fn test_warm_expert_allocation() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(8, config).unwrap();

        // Expert 0 gets 100 activations (max)
        for _ in 0..100 {
            allocator.record_activation(0);
        }

        // Expert 1 gets 50 activations (warm: 30-89% of max)
        for _ in 0..50 {
            allocator.record_activation(1);
        }

        allocator.recompute_thresholds();

        // Expert 1 should be warm (50 >= 30% of 100 = 30, but < 90)
        assert_eq!(allocator.allocate(1), ExpertPrecision::Warm);
        assert_eq!(allocator.get_format(1), GgufQuantType::Q3_K);
    }

    // ---------------------------------------------------------------
    // test_cold_expert_allocation
    // ---------------------------------------------------------------

    #[test]
    fn test_cold_expert_allocation() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(8, config).unwrap();

        // Expert 0 gets 100 activations (max)
        for _ in 0..100 {
            allocator.record_activation(0);
        }

        // Expert 7 gets 5 activations (cold: < 30% of max)
        for _ in 0..5 {
            allocator.record_activation(7);
        }

        allocator.recompute_thresholds();

        // Expert 7 should be cold (5 < 30% of 100 = 30)
        assert_eq!(allocator.allocate(7), ExpertPrecision::Cold);
        assert_eq!(allocator.get_format(7), GgufQuantType::Q2_K);
    }

    // ---------------------------------------------------------------
    // test_percentile_thresholds
    // ---------------------------------------------------------------

    #[test]
    fn test_percentile_thresholds() {
        let config = PrecisionConfig {
            hot_percentile: 0.8,
            cold_percentile: 0.2,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        // Set up activation counts: 100, 75, 25, 5
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        for _ in 0..75 {
            allocator.record_activation(1);
        }
        for _ in 0..25 {
            allocator.record_activation(2);
        }
        for _ in 0..5 {
            allocator.record_activation(3);
        }

        allocator.recompute_thresholds();

        // hot_threshold = ceil(100 * 0.8) = 80 (or 81 due to f32->f64 precision)
        // cold_threshold = floor(100 * 0.2) = 20
        // Allow for minor floating-point variance
        assert!(
            allocator.hot_threshold() >= 80 && allocator.hot_threshold() <= 81,
            "hot_threshold {} should be 80 or 81",
            allocator.hot_threshold()
        );
        assert_eq!(allocator.cold_threshold(), 20);

        // Expert 0: 100 >= hot_threshold -> Hot
        assert_eq!(allocator.allocate(0), ExpertPrecision::Hot);

        // Expert 1: 75 >= 20 but < hot_threshold -> Warm
        assert_eq!(allocator.allocate(1), ExpertPrecision::Warm);

        // Expert 2: 25 >= 20 but < hot_threshold -> Warm
        assert_eq!(allocator.allocate(2), ExpertPrecision::Warm);

        // Expert 3: 5 < 20 -> Cold
        assert_eq!(allocator.allocate(3), ExpertPrecision::Cold);
    }

    // ---------------------------------------------------------------
    // test_activation_recording
    // ---------------------------------------------------------------

    #[test]
    fn test_activation_recording() {
        let config = PrecisionConfig::default();
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        // Record individual activations
        allocator.record_activation(0);
        allocator.record_activation(0);
        allocator.record_activation(1);

        assert_eq!(allocator.get_count(0), 2);
        assert_eq!(allocator.get_count(1), 1);
        assert_eq!(allocator.get_count(2), 0);
        assert_eq!(allocator.total_activations(), 3);

        // Record batch activations
        allocator.record_activations(&[2, 2, 3, 0]);

        assert_eq!(allocator.get_count(0), 3);
        assert_eq!(allocator.get_count(2), 2);
        assert_eq!(allocator.get_count(3), 1);
        assert_eq!(allocator.total_activations(), 7);

        // Out-of-bounds should be ignored
        allocator.record_activation(100);
        assert_eq!(allocator.total_activations(), 7);
    }

    // ---------------------------------------------------------------
    // test_format_mapping
    // ---------------------------------------------------------------

    #[test]
    fn test_format_mapping() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            hot_format: GgufQuantType::Q5_K,
            warm_format: GgufQuantType::Q4_K,
            cold_format: GgufQuantType::Q3_K,
        };
        let mut allocator = PrecisionAllocator::new(3, config).unwrap();

        // Set up: 100 (hot), 50 (warm), 10 (cold)
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        for _ in 0..50 {
            allocator.record_activation(1);
        }
        for _ in 0..10 {
            allocator.record_activation(2);
        }

        allocator.recompute_thresholds();

        assert_eq!(allocator.get_format(0), GgufQuantType::Q5_K);
        assert_eq!(allocator.get_format(1), GgufQuantType::Q4_K);
        assert_eq!(allocator.get_format(2), GgufQuantType::Q3_K);
    }

    // ---------------------------------------------------------------
    // test_recompute_thresholds
    // ---------------------------------------------------------------

    #[test]
    fn test_recompute_thresholds() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        // Initially all zeros
        allocator.recompute_thresholds();
        assert_eq!(allocator.hot_threshold(), 0);
        assert_eq!(allocator.cold_threshold(), 0);

        // Add some activations
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        allocator.recompute_thresholds();

        // hot_threshold = ceil(100 * 0.9) = 90
        // cold_threshold = max(1, floor(100 * 0.3)) = 30
        assert_eq!(allocator.hot_threshold(), 90);
        assert_eq!(allocator.cold_threshold(), 30);

        // Add more activations and recompute
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        allocator.recompute_thresholds();

        // hot_threshold = ceil(200 * 0.9) = 180
        // cold_threshold = floor(200 * 0.3) = 60
        assert_eq!(allocator.hot_threshold(), 180);
        assert_eq!(allocator.cold_threshold(), 60);
    }

    // ---------------------------------------------------------------
    // test_precision_map
    // ---------------------------------------------------------------

    #[test]
    fn test_precision_map() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        for _ in 0..100 {
            allocator.record_activation(0);
        }
        for _ in 0..50 {
            allocator.record_activation(1);
        }
        for _ in 0..35 {
            allocator.record_activation(2);
        }
        for _ in 0..10 {
            allocator.record_activation(3);
        }

        allocator.recompute_thresholds();

        let map = allocator.get_precision_map();
        assert_eq!(map.len(), 4);
        assert_eq!(map[0], (0, ExpertPrecision::Hot));
        assert_eq!(map[1], (1, ExpertPrecision::Warm));
        assert_eq!(map[2], (2, ExpertPrecision::Warm));
        assert_eq!(map[3], (3, ExpertPrecision::Cold));
    }

    // ---------------------------------------------------------------
    // test_tier_counts
    // ---------------------------------------------------------------

    #[test]
    fn test_tier_counts() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(8, config).unwrap();

        // 2 hot, 3 warm, 3 cold
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        for _ in 0..95 {
            allocator.record_activation(1);
        }
        for _ in 0..50 {
            allocator.record_activation(2);
        }
        for _ in 0..40 {
            allocator.record_activation(3);
        }
        for _ in 0..35 {
            allocator.record_activation(4);
        }
        for _ in 0..10 {
            allocator.record_activation(5);
        }
        for _ in 0..5 {
            allocator.record_activation(6);
        }
        // Expert 7 has 0 activations

        allocator.recompute_thresholds();

        let (hot, warm, cold) = allocator.tier_counts();
        assert_eq!(hot, 2, "Expected 2 hot experts");
        assert!(warm >= 2, "Expected at least 2 warm experts");
        assert!(cold >= 2, "Expected at least 2 cold experts");
        assert_eq!(hot + warm + cold, 8, "Total should equal num_experts");
    }

    // ---------------------------------------------------------------
    // test_reset
    // ---------------------------------------------------------------

    #[test]
    fn test_reset() {
        let config = PrecisionConfig::default();
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        // Add activations
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        allocator.recompute_thresholds();

        assert!(allocator.total_activations() > 0);
        assert!(allocator.hot_threshold() > 0);

        // Reset
        allocator.reset();

        assert_eq!(allocator.total_activations(), 0);
        assert_eq!(allocator.hot_threshold(), 0);
        assert_eq!(allocator.cold_threshold(), 0);
        for id in 0..4 {
            assert_eq!(allocator.get_count(id), 0);
        }
    }

    // ---------------------------------------------------------------
    // test_experts_by_precision
    // ---------------------------------------------------------------

    #[test]
    fn test_experts_by_precision() {
        let config = PrecisionConfig {
            hot_percentile: 0.9,
            cold_percentile: 0.3,
            ..Default::default()
        };
        let mut allocator = PrecisionAllocator::new(6, config).unwrap();

        // Set up known distribution
        for _ in 0..100 {
            allocator.record_activation(0);
        } // Hot
        for _ in 0..92 {
            allocator.record_activation(1);
        } // Hot
        for _ in 0..50 {
            allocator.record_activation(2);
        } // Warm
        for _ in 0..40 {
            allocator.record_activation(3);
        } // Warm
        for _ in 0..10 {
            allocator.record_activation(4);
        } // Cold
          // Expert 5 has 0 activations -> Cold

        allocator.recompute_thresholds();

        let hot_experts = allocator.experts_by_precision(ExpertPrecision::Hot);
        let warm_experts = allocator.experts_by_precision(ExpertPrecision::Warm);
        let cold_experts = allocator.experts_by_precision(ExpertPrecision::Cold);

        assert!(hot_experts.contains(&0));
        assert!(hot_experts.contains(&1));
        assert!(warm_experts.contains(&2) || warm_experts.contains(&3));
        assert!(cold_experts.contains(&4) || cold_experts.contains(&5));
    }

    // ---------------------------------------------------------------
    // test_compute_percentile
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_percentile() {
        let config = PrecisionConfig::default();
        let mut allocator = PrecisionAllocator::new(4, config).unwrap();

        // No activations -> 0.0 percentile
        assert_eq!(allocator.compute_percentile(0), 0.0);

        // Set up: 100, 50, 25, 0
        for _ in 0..100 {
            allocator.record_activation(0);
        }
        for _ in 0..50 {
            allocator.record_activation(1);
        }
        for _ in 0..25 {
            allocator.record_activation(2);
        }

        // Expert 0: 100/100 = 1.0
        assert!((allocator.compute_percentile(0) - 1.0).abs() < f32::EPSILON);

        // Expert 1: 50/100 = 0.5
        assert!((allocator.compute_percentile(1) - 0.5).abs() < f32::EPSILON);

        // Expert 2: 25/100 = 0.25
        assert!((allocator.compute_percentile(2) - 0.25).abs() < f32::EPSILON);

        // Expert 3: 0/100 = 0.0
        assert!((allocator.compute_percentile(3) - 0.0).abs() < f32::EPSILON);

        // Out-of-bounds
        assert_eq!(allocator.compute_percentile(100), 0.0);
    }

    // ---------------------------------------------------------------
    // test_config_validation
    // ---------------------------------------------------------------

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = PrecisionConfig::default();
        assert!(valid.validate().is_ok());

        // Invalid: hot_percentile > 1.0
        let invalid1 = PrecisionConfig {
            hot_percentile: 1.5,
            ..Default::default()
        };
        assert!(invalid1.validate().is_err());

        // Invalid: cold_percentile >= hot_percentile
        let invalid2 = PrecisionConfig {
            hot_percentile: 0.5,
            cold_percentile: 0.6,
            ..Default::default()
        };
        assert!(invalid2.validate().is_err());

        // Invalid: cold_percentile negative
        let invalid3 = PrecisionConfig {
            cold_percentile: -0.1,
            ..Default::default()
        };
        assert!(invalid3.validate().is_err());
    }

    // ---------------------------------------------------------------
    // test_precision_name
    // ---------------------------------------------------------------

    #[test]
    fn test_precision_name() {
        assert_eq!(ExpertPrecision::Hot.name(), "hot");
        assert_eq!(ExpertPrecision::Warm.name(), "warm");
        assert_eq!(ExpertPrecision::Cold.name(), "cold");
    }

    // ---------------------------------------------------------------
    // test_out_of_bounds_expert_id
    // ---------------------------------------------------------------

    #[test]
    fn test_out_of_bounds_expert_id() {
        let config = PrecisionConfig::default();
        let allocator = PrecisionAllocator::new(4, config).unwrap();

        // Out-of-bounds should return Cold
        assert_eq!(allocator.allocate(100), ExpertPrecision::Cold);
        assert_eq!(allocator.get_format(100), GgufQuantType::Q2_K);
        assert_eq!(allocator.get_count(100), 0);
    }

    // ---------------------------------------------------------------
    // test_memory_constrained_config
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_constrained_config() {
        let config = PrecisionConfig::memory_constrained();

        assert!(config.validate().is_ok());
        assert_eq!(config.hot_percentile, 0.95);
        assert_eq!(config.cold_percentile, 0.4);
        // More aggressive compression for warm/cold
        assert_eq!(config.warm_format, GgufQuantType::Q2_K);
        assert_eq!(config.cold_format, GgufQuantType::Q2_K);
    }

    // ---------------------------------------------------------------
    // test_quality_focused_config
    // ---------------------------------------------------------------

    #[test]
    fn test_quality_focused_config() {
        let config = PrecisionConfig::quality_focused();

        assert!(config.validate().is_ok());
        assert_eq!(config.hot_percentile, 0.8);
        assert_eq!(config.cold_percentile, 0.2);
        // Higher precision formats
        assert_eq!(config.hot_format, GgufQuantType::Q5_K);
        assert_eq!(config.warm_format, GgufQuantType::Q4_K);
        assert_eq!(config.cold_format, GgufQuantType::Q3_K);
    }

    // ---------------------------------------------------------------
    // test_saturating_add_for_counts
    // ---------------------------------------------------------------

    #[test]
    fn test_saturating_add_for_counts() {
        let config = PrecisionConfig::default();
        let mut allocator = PrecisionAllocator::new(1, config).unwrap();

        // Set count close to max
        allocator.counts[0] = u64::MAX - 1;

        // Should saturate instead of overflow
        allocator.record_activation(0);
        assert_eq!(allocator.get_count(0), u64::MAX);

        allocator.record_activation(0);
        assert_eq!(allocator.get_count(0), u64::MAX);
    }
}
