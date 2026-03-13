//! Expert Affinity Tracking (ADR-092)
//!
//! This module implements EMA-based expert affinity tracking for memory-aware
//! MoE routing. The affinity scores track which experts are frequently activated,
//! enabling:
//!
//! - **Predictive Prefetching**: Load experts with high affinity before they're needed
//! - **Affinity-Aware Eviction**: Evict low-affinity experts first
//! - **Precision Allocation**: Assign higher precision to frequently-used experts
//!
//! ## Key Invariant (INV-2 from ADR-092)
//!
//! **Affinity Monotonicity**: EMA-based affinity scores decrease monotonically
//! without new activations. This ensures predictable eviction behavior.
//!
//! ## Algorithm
//!
//! On each update:
//! 1. All scores are decayed: `score = score * decay`
//! 2. Activated experts receive a boost: `score = min(score + boost, 1.0)`
//!
//! The decay factor (typically 0.95-0.99) controls how quickly old activations
//! are "forgotten". Higher values provide longer memory.

use super::ExpertId;

/// SIMD-optimized decay for f32 scores.
///
/// Applies `scores[i] *= decay` for all elements using platform-specific
/// SIMD intrinsics when available, with scalar fallback.
#[inline]
fn decay_scores_simd(scores: &mut [f32], decay: f32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        decay_scores_neon(scores, decay);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        decay_scores_avx2(scores, decay);
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        decay_scores_scalar(scores, decay);
    }
}

/// Scalar fallback for decay
#[inline]
fn decay_scores_scalar(scores: &mut [f32], decay: f32) {
    for score in scores.iter_mut() {
        *score *= decay;
    }
}

/// NEON-optimized decay for ARM64
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn decay_scores_neon(scores: &mut [f32], decay: f32) {
    use std::arch::aarch64::*;

    let len = scores.len();
    let chunks = len / 4;
    let remainder = len % 4;

    unsafe {
        let decay_vec = vdupq_n_f32(decay);
        let ptr = scores.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let vals = vld1q_f32(ptr.add(offset));
            let result = vmulq_f32(vals, decay_vec);
            vst1q_f32(ptr.add(offset), result);
        }

        // Handle remainder with scalar
        for i in (chunks * 4)..len {
            *scores.get_unchecked_mut(i) *= decay;
        }
    }
}

/// AVX2-optimized decay for x86_64
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn decay_scores_avx2(scores: &mut [f32], decay: f32) {
    use std::arch::x86_64::*;

    let len = scores.len();
    let chunks = len / 8;

    unsafe {
        let decay_vec = _mm256_set1_ps(decay);
        let ptr = scores.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let vals = _mm256_loadu_ps(ptr.add(offset));
            let result = _mm256_mul_ps(vals, decay_vec);
            _mm256_storeu_ps(ptr.add(offset), result);
        }

        // Handle remainder with scalar
        for i in (chunks * 8)..len {
            *scores.get_unchecked_mut(i) *= decay;
        }
    }
}

/// Configuration for expert affinity tracking.
///
/// # Example
///
/// ```rust
/// use ruvllm::moe::AffinityConfig;
///
/// let config = AffinityConfig::with_num_experts(8)
///     .with_decay(0.95)
///     .with_activation_boost(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Number of experts in the model.
    pub num_experts: usize,

    /// EMA decay factor applied to all scores on each update.
    ///
    /// Range: `0.0 < decay < 1.0`
    /// - Higher values (e.g., 0.99) = longer memory, slower forgetting
    /// - Lower values (e.g., 0.95) = shorter memory, faster adaptation
    ///
    /// Default: 0.99
    pub decay: f32,

    /// Boost value added to activated experts.
    ///
    /// The score after boosting is clamped to `[0.0, max_score]`.
    ///
    /// Default: 1.0
    pub activation_boost: f32,

    /// Maximum affinity score (clamping bound).
    ///
    /// Scores are clamped to `[0.0, max_score]` after boosting.
    ///
    /// Default: 1.0
    pub max_score: f32,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            decay: 0.99,
            activation_boost: 1.0,
            max_score: 1.0,
        }
    }
}

impl AffinityConfig {
    /// Create config for a specific number of experts with default decay and boost.
    pub fn with_num_experts(num_experts: usize) -> Self {
        Self {
            num_experts,
            ..Default::default()
        }
    }

    /// Builder: set the decay factor.
    ///
    /// Values are clamped to `[0.0, 1.0]`.
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay.clamp(0.0, 1.0);
        self
    }

    /// Builder: set the activation boost.
    ///
    /// Negative values are clamped to 0.
    pub fn with_activation_boost(mut self, boost: f32) -> Self {
        self.activation_boost = boost.max(0.0);
        self
    }

    /// Builder: set the maximum score.
    ///
    /// Values are clamped to be at least 0.0.
    pub fn with_max_score(mut self, max_score: f32) -> Self {
        self.max_score = max_score.max(0.0);
        self
    }
}

/// EMA-based expert affinity tracker (ADR-092).
///
/// Tracks which experts are frequently activated using Exponential Moving Average
/// scores. This enables memory-aware routing decisions:
///
/// - Experts with high affinity should be kept in cache
/// - Experts with low affinity can be evicted or use lower precision
/// - High-affinity experts are good prefetch candidates
///
/// # Invariant INV-2: Affinity Monotonicity
///
/// Without new activations, all affinity scores decrease monotonically
/// according to the decay factor. This property is critical for predictable
/// eviction behavior.
///
/// # Example
///
/// ```rust
/// use ruvllm::moe::{ExpertAffinity, AffinityConfig};
///
/// let config = AffinityConfig::with_num_experts(8).with_decay(0.95);
/// let mut affinity = ExpertAffinity::new(config);
///
/// // Experts 2 and 5 were selected this round
/// affinity.update(&[2, 5]);
///
/// // Get current affinity scores
/// assert!(affinity.get_score(2) > affinity.get_score(0));
///
/// // Get top experts for prefetching
/// let top3 = affinity.top_k_by_affinity(3);
/// ```
#[derive(Debug, Clone)]
pub struct ExpertAffinity {
    /// EMA scores per expert, range `[0.0, 1.0]`.
    scores: Vec<f32>,
    /// Configuration parameters.
    config: AffinityConfig,
    /// Total activation count per expert (for precision allocation).
    total_activations: Vec<u64>,
}

impl ExpertAffinity {
    /// Create a new affinity tracker with the given configuration
    pub fn new(config: AffinityConfig) -> Self {
        Self {
            scores: vec![0.0; config.num_experts],
            total_activations: vec![0; config.num_experts],
            config,
        }
    }

    /// Update affinity for activated experts
    ///
    /// This method:
    /// 1. Applies decay to ALL expert scores (INV-2: monotonic decay)
    /// 2. Boosts scores for activated experts
    ///
    /// # Arguments
    ///
    /// * `activated` - Expert IDs that were activated this step
    pub fn update(&mut self, activated: &[ExpertId]) {
        // Step 1: Decay all scores (INV-2: monotonic without activation)
        // Use SIMD-optimized decay when available
        decay_scores_simd(&mut self.scores, self.config.decay);

        // Step 2: Boost activated experts
        for &id in activated {
            if id < self.scores.len() {
                self.scores[id] =
                    (self.scores[id] + self.config.activation_boost).min(self.config.max_score);
                self.total_activations[id] += 1;
            }
        }
    }

    /// Get the affinity score for a specific expert
    pub fn score(&self, expert_id: ExpertId) -> f32 {
        self.scores.get(expert_id).copied().unwrap_or(0.0)
    }

    /// Alias for [`score`](Self::score) for API consistency.
    #[inline]
    pub fn get_score(&self, expert_id: ExpertId) -> f32 {
        self.score(expert_id)
    }

    /// Get all affinity scores.
    ///
    /// The returned slice is indexed by expert ID.
    #[inline]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Alias for [`scores`](Self::scores) for API consistency.
    #[inline]
    pub fn get_scores(&self) -> &[f32] {
        self.scores()
    }

    /// Get total activation count for an expert.
    ///
    /// This count is never reset by `update()` and is useful for
    /// long-term precision allocation decisions.
    ///
    /// # Returns
    ///
    /// Total number of times this expert has been activated, or `0` if
    /// the expert ID is out of range.
    #[inline]
    pub fn activation_count(&self, expert_id: ExpertId) -> u64 {
        self.total_activations.get(expert_id).copied().unwrap_or(0)
    }

    /// Get all activation counts.
    ///
    /// The returned slice is indexed by expert ID.
    #[inline]
    pub fn get_activation_counts(&self) -> &[u64] {
        &self.total_activations
    }

    /// Get experts sorted by affinity score (highest first)
    ///
    /// Useful for prefetching decisions. NaN values are treated as lowest priority.
    pub fn top_k_by_affinity(&self, k: usize) -> Vec<ExpertId> {
        let mut indexed: Vec<(ExpertId, f32)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(id, &s)| (id, if s.is_finite() { s } else { f32::NEG_INFINITY }))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0)) // Deterministic tie-breaking by ID
        });
        indexed.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Get experts sorted by total activation count (highest first)
    ///
    /// Useful for precision allocation decisions.
    pub fn top_k_by_frequency(&self, k: usize) -> Vec<ExpertId> {
        let mut indexed: Vec<(ExpertId, u64)> =
            self.total_activations.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Get the least-affinity expert from a set of candidates
    ///
    /// Useful for eviction decisions. NaN values are treated as lowest (evict first).
    pub fn least_affinity(&self, candidates: &[ExpertId]) -> Option<ExpertId> {
        candidates.iter().copied().min_by(|&a, &b| {
            let score_a = self.score(a);
            let score_b = self.score(b);
            // NaN handling: treat NaN as NEG_INFINITY for eviction priority
            let sa = if score_a.is_finite() {
                score_a
            } else {
                f32::NEG_INFINITY
            };
            let sb = if score_b.is_finite() {
                score_b
            } else {
                f32::NEG_INFINITY
            };
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b)) // Deterministic tie-breaking
        })
    }

    /// Compute percentile rank of an expert's activation frequency
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means highest frequency.
    pub fn frequency_percentile(&self, expert_id: ExpertId) -> f32 {
        let count = self.activation_count(expert_id);
        let lower = self
            .total_activations
            .iter()
            .filter(|&&c| c < count)
            .count();
        let equal = self
            .total_activations
            .iter()
            .filter(|&&c| c == count)
            .count();
        let n = self.total_activations.len();
        if n == 0 {
            return 0.5;
        }
        (lower as f32 + 0.5 * equal as f32) / n as f32
    }

    /// Reset all affinity scores to zero
    pub fn reset(&mut self) {
        self.scores.fill(0.0);
        self.total_activations.fill(0);
    }

    /// Get the number of experts tracked
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }

    /// Get the configuration
    pub fn config(&self) -> &AffinityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================================
    // Tests following ADR-092 specification (10+ required tests)
    // =====================================================================

    /// Test 1: Affinity creation initializes all scores to zero
    #[test]
    fn test_affinity_creation() {
        let config = AffinityConfig::with_num_experts(8);
        let affinity = ExpertAffinity::new(config);

        assert_eq!(affinity.num_experts(), 8);
        assert_eq!(affinity.scores().len(), 8);
        assert_eq!(affinity.get_scores().len(), 8);
        assert!(affinity.scores().iter().all(|&s| s == 0.0));
        assert!(affinity.get_activation_counts().iter().all(|&c| c == 0));
    }

    /// Test 2: Update decays ALL scores, even those not activated
    #[test]
    fn test_update_decays_all() {
        let config = AffinityConfig::with_num_experts(4)
            .with_decay(0.5)
            .with_activation_boost(1.0);
        let mut affinity = ExpertAffinity::new(config);

        // Activate all experts to set initial scores
        affinity.update(&[0, 1, 2, 3]);

        // All should be at 1.0 (boost clamped to max_score)
        for &score in affinity.scores() {
            assert!((score - 1.0).abs() < 1e-6);
        }

        // Update with ONLY expert 0 -> all others should decay
        affinity.update(&[0]);

        // Expert 0: 1.0 * 0.5 + 1.0 = 1.5 -> clamped to 1.0
        assert!((affinity.score(0) - 1.0).abs() < 1e-6);

        // Experts 1, 2, 3: 1.0 * 0.5 = 0.5 (no boost)
        for id in 1..4 {
            assert!(
                (affinity.score(id) - 0.5).abs() < 1e-6,
                "Expert {} should decay to 0.5, got {}",
                id,
                affinity.score(id)
            );
        }
    }

    /// Test 3: Update boosts activated experts
    #[test]
    fn test_update_boosts_activated() {
        let config = AffinityConfig::with_num_experts(4)
            .with_decay(0.9)
            .with_activation_boost(0.5);
        let mut affinity = ExpertAffinity::new(config);

        // Activate experts 1 and 3
        affinity.update(&[1, 3]);

        // Experts 1 and 3 should have boost
        // Score = 0.0 * 0.9 + 0.5 = 0.5
        assert!((affinity.score(1) - 0.5).abs() < 1e-6);
        assert!((affinity.score(3) - 0.5).abs() < 1e-6);

        // Others should be 0 (0.0 * 0.9 = 0.0, no boost)
        assert_eq!(affinity.score(0), 0.0);
        assert_eq!(affinity.score(2), 0.0);
    }

    /// Test 4: INV-2 Property - Monotonic decay without activations
    #[test]
    fn test_monotonic_decay() {
        let config = AffinityConfig::with_num_experts(8).with_decay(0.95);
        let mut affinity = ExpertAffinity::new(config);

        // Activate some experts
        affinity.update(&[1, 3, 5, 7]);

        // Record initial scores
        let scores_t0 = affinity.scores().to_vec();

        // Multiple updates with NO activations
        for iteration in 0..10 {
            let scores_before = affinity.scores().to_vec();
            affinity.update(&[]); // Empty update
            let scores_after = affinity.scores().to_vec();

            // INV-2: All scores must decrease monotonically
            for (i, (&before, &after)) in scores_before.iter().zip(scores_after.iter()).enumerate()
            {
                assert!(
                    after <= before,
                    "INV-2 violated at iteration {}: score[{}] increased from {} to {}",
                    iteration,
                    i,
                    before,
                    after
                );
            }
        }

        // All activated scores should have decayed significantly from t0
        for (i, (&t0, &current)) in scores_t0.iter().zip(affinity.scores().iter()).enumerate() {
            if t0 > 0.0 {
                assert!(
                    current < t0,
                    "Score[{}] did not decay: {} -> {}",
                    i,
                    t0,
                    current
                );
            }
        }
    }

    /// Test 5: Top-K by affinity returns correct experts in order
    #[test]
    fn test_top_k_by_affinity() {
        // Use decay=1.0 (no decay) to test pure ordering by activation count
        let config = AffinityConfig::with_num_experts(6)
            .with_decay(1.0)
            .with_activation_boost(0.1);
        let mut affinity = ExpertAffinity::new(config);

        // Create distinct affinity levels by activating experts different times
        // Expert 3: activated 5 times -> score = 0.5
        for _ in 0..5 {
            affinity.update(&[3]);
        }

        // Expert 1: activated 3 times -> score = 0.3
        for _ in 0..3 {
            affinity.update(&[1]);
        }

        // Expert 5: activated 1 time -> score = 0.1
        affinity.update(&[5]);

        // Verify scores are as expected (no decay)
        assert!(
            (affinity.score(3) - 0.5).abs() < 1e-6,
            "Expert 3 score: {}",
            affinity.score(3)
        );
        assert!(
            (affinity.score(1) - 0.3).abs() < 1e-6,
            "Expert 1 score: {}",
            affinity.score(1)
        );
        assert!(
            (affinity.score(5) - 0.1).abs() < 1e-6,
            "Expert 5 score: {}",
            affinity.score(5)
        );

        // Top-2 should be [3, 1] (highest scores)
        let top2 = affinity.top_k_by_affinity(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], 3, "Expert 3 should be top");
        assert_eq!(top2[1], 1, "Expert 1 should be second");

        // Top-4 should include 3, 1, 5, and one of the zeros
        let top4 = affinity.top_k_by_affinity(4);
        assert_eq!(top4.len(), 4);
        assert_eq!(top4[0], 3);
        assert_eq!(top4[1], 1);
        assert_eq!(top4[2], 5);

        // Top-10 (more than available) should return all 6
        let top10 = affinity.top_k_by_affinity(10);
        assert_eq!(top10.len(), 6);
    }

    /// Test 6: Score is clamped to max_score (default 1.0)
    #[test]
    fn test_score_clamped_to_one() {
        let config = AffinityConfig::with_num_experts(4)
            .with_decay(0.99)
            .with_activation_boost(1.0);
        let mut affinity = ExpertAffinity::new(config);

        // Activate expert 0 many times
        for _ in 0..100 {
            affinity.update(&[0]);
        }

        // Score should be clamped at max_score (1.0)
        assert!(
            (affinity.score(0) - 1.0).abs() < 1e-6,
            "Score should be clamped to 1.0, got {}",
            affinity.score(0)
        );

        // Should never exceed 1.0
        assert!(
            affinity.score(0) <= 1.0,
            "Score {} exceeds max_score",
            affinity.score(0)
        );
    }

    /// Test 7: Activation counting tracks total activations
    #[test]
    fn test_activation_counting() {
        let config = AffinityConfig::with_num_experts(4);
        let mut affinity = ExpertAffinity::new(config);

        // Activate experts with different frequencies
        affinity.update(&[0, 1]); // +1 each
        affinity.update(&[0, 2]); // +1 to 0, 2
        affinity.update(&[0]); // +1 to 0

        assert_eq!(affinity.activation_count(0), 3);
        assert_eq!(affinity.activation_count(1), 1);
        assert_eq!(affinity.activation_count(2), 1);
        assert_eq!(affinity.activation_count(3), 0);

        // Out of range should return 0
        assert_eq!(affinity.activation_count(100), 0);
    }

    /// Test 8: Reset clears all scores and activation counts
    #[test]
    fn test_reset() {
        let config = AffinityConfig::with_num_experts(4);
        let mut affinity = ExpertAffinity::new(config);

        // Build up some state
        for _ in 0..10 {
            affinity.update(&[0, 1, 2, 3]);
        }

        // Verify state is non-zero
        assert!(affinity.score(0) > 0.0);
        assert!(affinity.activation_count(0) > 0);

        // Reset
        affinity.reset();

        // All scores should be 0
        for &score in affinity.scores() {
            assert_eq!(score, 0.0);
        }

        // All activation counts should be 0
        for &count in affinity.get_activation_counts() {
            assert_eq!(count, 0);
        }
    }

    /// Test 9: Empty update only decays, no boosts
    #[test]
    fn test_empty_update() {
        let config = AffinityConfig::with_num_experts(4).with_decay(0.9);
        let mut affinity = ExpertAffinity::new(config);

        // Set initial state
        affinity.update(&[0, 1, 2, 3]);

        let counts_before = affinity.get_activation_counts().to_vec();

        // Empty update
        affinity.update(&[]);

        // Scores should decay (we verified in monotonic decay test)
        // Activation counts should NOT change
        assert_eq!(affinity.get_activation_counts(), &counts_before);
    }

    /// Test 10: Multiple updates sequence produces correct state
    #[test]
    fn test_multiple_updates_sequence() {
        let config = AffinityConfig::with_num_experts(8)
            .with_decay(0.8)
            .with_activation_boost(0.5);
        let mut affinity = ExpertAffinity::new(config);

        // Simulate a realistic workload:
        // Expert 0 and 1 are "hot" (activated frequently)
        // Expert 7 is activated once then never

        // Round 1: Activate 0, 1, 7
        affinity.update(&[0, 1, 7]);
        assert!((affinity.score(0) - 0.5).abs() < 1e-6);
        assert!((affinity.score(7) - 0.5).abs() < 1e-6);

        // Round 2: Activate 0, 1 only
        affinity.update(&[0, 1]);
        // Expert 0: 0.5 * 0.8 + 0.5 = 0.9
        assert!((affinity.score(0) - 0.9).abs() < 1e-6);
        // Expert 7: 0.5 * 0.8 = 0.4 (no boost)
        assert!((affinity.score(7) - 0.4).abs() < 1e-6);

        // Round 3: Activate 0, 1 again
        affinity.update(&[0, 1]);
        // Expert 0: 0.9 * 0.8 + 0.5 = 1.22 -> clamped to 1.0
        assert!((affinity.score(0) - 1.0).abs() < 1e-6);
        // Expert 7: 0.4 * 0.8 = 0.32
        assert!((affinity.score(7) - 0.32).abs() < 1e-6);

        // After 3 rounds:
        // - Expert 0, 1 should be top (activated every round)
        // - Expert 7 should be decaying
        let top2 = affinity.top_k_by_affinity(2);
        assert!(top2.contains(&0));
        assert!(top2.contains(&1));

        // Activation counts
        assert_eq!(affinity.activation_count(0), 3);
        assert_eq!(affinity.activation_count(1), 3);
        assert_eq!(affinity.activation_count(7), 1);
    }

    // =====================================================================
    // Additional tests beyond the 10 required
    // =====================================================================

    /// Test: Out-of-bounds expert IDs are silently ignored
    #[test]
    fn test_out_of_bounds_experts_ignored() {
        let config = AffinityConfig::with_num_experts(4);
        let mut affinity = ExpertAffinity::new(config);

        // Include invalid expert IDs
        affinity.update(&[0, 1, 100, 200, 3]);

        // Valid experts should be updated
        assert!(affinity.score(0) > 0.0);
        assert!(affinity.score(1) > 0.0);
        assert!(affinity.score(3) > 0.0);

        // Expert 2 was not activated
        assert_eq!(affinity.score(2), 0.0);

        // Activation counts for valid experts
        assert_eq!(affinity.activation_count(0), 1);
        assert_eq!(affinity.activation_count(100), 0);
    }

    /// Test: Config builder methods work correctly
    #[test]
    fn test_config_builders() {
        let config = AffinityConfig::with_num_experts(16)
            .with_decay(0.95)
            .with_activation_boost(0.75)
            .with_max_score(2.0);

        assert_eq!(config.num_experts, 16);
        assert!((config.decay - 0.95).abs() < 1e-6);
        assert!((config.activation_boost - 0.75).abs() < 1e-6);
        assert!((config.max_score - 2.0).abs() < 1e-6);
    }

    /// Test: Decay is clamped to [0.0, 1.0]
    #[test]
    fn test_decay_clamp() {
        let config = AffinityConfig::with_num_experts(4).with_decay(1.5);
        assert!(
            (config.decay - 1.0).abs() < 1e-6,
            "Decay should be clamped to 1.0"
        );

        let config2 = AffinityConfig::with_num_experts(4).with_decay(-0.5);
        assert!(
            (config2.decay - 0.0).abs() < 1e-6,
            "Decay should be clamped to 0.0"
        );
    }

    /// Test: Frequency percentile calculation
    #[test]
    fn test_frequency_percentile() {
        let config = AffinityConfig::with_num_experts(4);
        let mut affinity = ExpertAffinity::new(config);

        // Expert 0: 1 activation
        // Expert 1: 5 activations
        // Expert 2: 3 activations
        // Expert 3: 0 activations
        affinity.update(&[0]);
        for _ in 0..5 {
            affinity.update(&[1]);
        }
        for _ in 0..3 {
            affinity.update(&[2]);
        }

        // Expert 1 should have highest percentile
        let pct_1 = affinity.frequency_percentile(1);
        let pct_3 = affinity.frequency_percentile(3);

        assert!(
            pct_1 > pct_3,
            "Expert 1 should have higher percentile than 3"
        );
        assert!(pct_1 > 0.5, "Expert 1 should be above median");
    }

    /// Test: Least affinity from candidates
    #[test]
    fn test_least_affinity() {
        // Use decay=1.0 (no decay) and small boost to get distinct scores
        let config = AffinityConfig::with_num_experts(4)
            .with_decay(1.0)
            .with_activation_boost(0.1);
        let mut affinity = ExpertAffinity::new(config);

        // Different activation levels -> different scores
        // Expert 0: 5 activations -> score = 0.5
        for _ in 0..5 {
            affinity.update(&[0]);
        }
        // Expert 1: 2 activations -> score = 0.2
        for _ in 0..2 {
            affinity.update(&[1]);
        }
        // Expert 2: 1 activation -> score = 0.1
        affinity.update(&[2]);

        // Verify scores
        assert!((affinity.score(0) - 0.5).abs() < 1e-6);
        assert!((affinity.score(1) - 0.2).abs() < 1e-6);
        assert!((affinity.score(2) - 0.1).abs() < 1e-6);

        let candidates = vec![0, 1, 2];
        let least = affinity.least_affinity(&candidates);

        // Expert 2 has lowest affinity (0.1)
        assert_eq!(least, Some(2));

        // Empty candidates
        let empty: Vec<ExpertId> = vec![];
        assert_eq!(affinity.least_affinity(&empty), None);
    }

    /// Test: Top-K by frequency
    #[test]
    fn test_top_k_by_frequency() {
        let config = AffinityConfig::with_num_experts(4);
        let mut affinity = ExpertAffinity::new(config);

        affinity.update(&[0]);
        affinity.update(&[1]);
        affinity.update(&[1]);
        affinity.update(&[2]);
        affinity.update(&[2]);
        affinity.update(&[2]);

        let top_2 = affinity.top_k_by_frequency(2);
        assert_eq!(top_2.len(), 2);
        // Expert 2 activated 3 times, highest
        assert_eq!(top_2[0], 2);
        // Expert 1 activated 2 times, second
        assert_eq!(top_2[1], 1);
    }

    /// Test: Default config values
    #[test]
    fn test_default_config() {
        let config = AffinityConfig::default();
        assert_eq!(config.num_experts, 8);
        assert!((config.decay - 0.99).abs() < 1e-6);
        assert!((config.activation_boost - 1.0).abs() < 1e-6);
        assert!((config.max_score - 1.0).abs() < 1e-6);
    }

    // =====================================================================
    // P1 Optimization Tests: SIMD decay
    // =====================================================================

    /// Test: SIMD decay with non-aligned sizes (tests remainder handling)
    #[test]
    fn test_simd_decay_non_aligned() {
        // Test sizes that don't align to SIMD widths (4 for NEON, 8 for AVX2)
        for size in [1, 3, 5, 7, 9, 15, 17, 33, 65] {
            let config = AffinityConfig::with_num_experts(size).with_decay(0.5);
            let mut affinity = ExpertAffinity::new(config);

            // Activate all experts
            let all_experts: Vec<usize> = (0..size).collect();
            affinity.update(&all_experts);

            // All should be at 1.0
            for &score in affinity.scores() {
                assert!((score - 1.0).abs() < 1e-6);
            }

            // Decay once
            affinity.update(&[]);

            // All should be at 0.5
            for (i, &score) in affinity.scores().iter().enumerate() {
                assert!(
                    (score - 0.5).abs() < 1e-6,
                    "Expert {} score should be 0.5, got {}",
                    i,
                    score
                );
            }
        }
    }

    /// Test: SIMD decay with large expert count
    #[test]
    fn test_simd_decay_large() {
        let config = AffinityConfig::with_num_experts(256).with_decay(0.9);
        let mut affinity = ExpertAffinity::new(config);

        // Activate first 128 experts
        let activated: Vec<usize> = (0..128).collect();
        affinity.update(&activated);

        // Decay multiple times
        for _ in 0..10 {
            affinity.update(&[]);
        }

        // All activated scores should have decayed significantly
        let expected = 0.9f32.powi(10);
        for i in 0..128 {
            let score = affinity.score(i);
            assert!(
                (score - expected).abs() < 1e-5,
                "Expert {} score should be ~{}, got {}",
                i,
                expected,
                score
            );
        }

        // Non-activated should still be 0
        for i in 128..256 {
            assert_eq!(affinity.score(i), 0.0);
        }
    }

    /// Test: SIMD decay correctness (scalar vs SIMD equivalence)
    #[test]
    fn test_simd_decay_correctness() {
        let config = AffinityConfig::with_num_experts(64)
            .with_decay(0.87)
            .with_activation_boost(0.33);
        let mut affinity = ExpertAffinity::new(config);

        // Activate various experts with a pattern
        affinity.update(&[0, 7, 15, 23, 31, 39, 47, 55, 63]);

        // Record scores
        let scores_before: Vec<f32> = affinity.scores().to_vec();

        // Decay
        affinity.update(&[]);

        // Verify each score decayed correctly
        for (i, (&before, &after)) in scores_before
            .iter()
            .zip(affinity.scores().iter())
            .enumerate()
        {
            let expected = before * 0.87;
            assert!(
                (after - expected).abs() < 1e-6,
                "Expert {} decay incorrect: {} * 0.87 = {}, got {}",
                i,
                before,
                expected,
                after
            );
        }
    }
}
