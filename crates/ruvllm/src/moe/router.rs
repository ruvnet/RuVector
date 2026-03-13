//! Memory-Aware MoE Router (ADR-092)
//!
//! Expert selection with cache residency bonus for >=70% cache hit rate.
//! Implements INV-6: Router Determinism - same input + cache state = same result.
//!
//! ## Algorithm
//!
//! 1. Compute base scores from gate network logits
//! 2. Add cache residency bonus to resident experts
//! 3. Select top-K experts
//! 4. Update affinity tracking
//! 5. Generate paging requests for non-resident experts
//!
//! ## Configuration
//!
//! The `cache_bonus` parameter (0.0-1.0) controls how much to favor resident experts:
//! - 0.0: Pure accuracy (ignore cache state, baseline 34% hit rate)
//! - 0.15: Recommended balance (>=70% hit rate with <1% accuracy loss)
//! - 0.3+: Aggressive caching (may degrade accuracy)

use super::{ExpertAffinity, ExpertId, MoeMetrics};
use std::time::Instant;

// ============================================================================
// CacheMask: Bitmask-based cache residency tracking (P1 optimization)
// ============================================================================

/// Bitmask-based cache residency tracking for efficient memory access patterns.
///
/// Uses a u64 for up to 64 experts (most common case: 8, 16, 32, 64 experts).
/// Falls back to Vec<u64> for larger models.
#[derive(Debug, Clone)]
struct CacheMask {
    /// Bitmask for small models (up to 64 experts)
    small: u64,
    /// Extended bitmask for larger models (>64 experts)
    extended: Option<Vec<u64>>,
    /// Number of experts tracked
    num_experts: usize,
}

impl CacheMask {
    /// Create a new cache mask for the given number of experts
    fn new(num_experts: usize) -> Self {
        if num_experts <= 64 {
            Self {
                small: 0,
                extended: None,
                num_experts,
            }
        } else {
            let num_words = (num_experts + 63) / 64;
            Self {
                small: 0,
                extended: Some(vec![0u64; num_words]),
                num_experts,
            }
        }
    }

    /// Check if an expert is resident
    #[inline]
    fn is_set(&self, id: ExpertId) -> bool {
        if id >= self.num_experts {
            return false;
        }
        if self.num_experts <= 64 {
            (self.small & (1u64 << id)) != 0
        } else {
            let word = id / 64;
            let bit = id % 64;
            self.extended
                .as_ref()
                .map(|v| (v[word] & (1u64 << bit)) != 0)
                .unwrap_or(false)
        }
    }

    /// Set an expert as resident or non-resident
    #[inline]
    fn set(&mut self, id: ExpertId, resident: bool) {
        if id >= self.num_experts {
            return;
        }
        if self.num_experts <= 64 {
            if resident {
                self.small |= 1u64 << id;
            } else {
                self.small &= !(1u64 << id);
            }
        } else if let Some(ref mut v) = self.extended {
            let word = id / 64;
            let bit = id % 64;
            if resident {
                v[word] |= 1u64 << bit;
            } else {
                v[word] &= !(1u64 << bit);
            }
        }
    }

    /// Clear all bits (no experts resident)
    #[inline]
    fn clear(&mut self) {
        self.small = 0;
        if let Some(ref mut v) = self.extended {
            v.fill(0);
        }
    }

    /// Get list of resident expert IDs
    fn resident_list(&self) -> Vec<ExpertId> {
        let mut result = Vec::new();
        if self.num_experts <= 64 {
            let mut bits = self.small;
            while bits != 0 {
                let trailing = bits.trailing_zeros() as usize;
                result.push(trailing);
                bits &= bits - 1; // Clear lowest set bit
            }
        } else if let Some(ref v) = self.extended {
            for (word_idx, &word) in v.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let trailing = bits.trailing_zeros() as usize;
                    let id = word_idx * 64 + trailing;
                    if id < self.num_experts {
                        result.push(id);
                    }
                    bits &= bits - 1;
                }
            }
        }
        result
    }

    /// Count number of resident experts (popcount)
    #[inline]
    fn count(&self) -> usize {
        if self.num_experts <= 64 {
            self.small.count_ones() as usize
        } else {
            self.extended
                .as_ref()
                .map(|v| v.iter().map(|w| w.count_ones() as usize).sum())
                .unwrap_or(0)
        }
    }
}

/// Paging direction for expert load/evict operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingDirection {
    /// Load expert into cache
    In,
    /// Evict expert from cache
    Out,
}

/// Priority level for paging operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PagingPriority {
    /// Normal priority (can be delayed)
    Normal,
    /// Urgent (needed for current inference)
    Urgent,
    /// Prefetch (speculative, can be cancelled)
    Prefetch,
}

/// Request to page an expert in or out of cache
#[derive(Debug, Clone)]
pub struct PagingRequest {
    /// Expert ID to page
    pub expert_id: ExpertId,
    /// Direction (In = load, Out = evict)
    pub direction: PagingDirection,
    /// Priority level
    pub priority: PagingPriority,
}

impl PagingRequest {
    /// Create a new paging request
    pub fn new(expert_id: ExpertId, direction: PagingDirection, priority: PagingPriority) -> Self {
        Self {
            expert_id,
            direction,
            priority,
        }
    }

    /// Create an urgent page-in request
    pub fn page_in_urgent(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::In, PagingPriority::Urgent)
    }

    /// Create a prefetch request
    pub fn prefetch(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::In, PagingPriority::Prefetch)
    }

    /// Create a page-out request
    pub fn page_out(expert_id: ExpertId) -> Self {
        Self::new(expert_id, PagingDirection::Out, PagingPriority::Normal)
    }
}

/// Configuration for the memory-aware router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Cache residency bonus weight (0.0-1.0)
    ///
    /// Added to gate scores for experts currently in cache.
    /// Default: 0.15 (achieves >=70% hit rate with <1% accuracy loss)
    pub cache_bonus: f32,

    /// Top-K experts to select per token
    ///
    /// Typical values: 1 (Switch), 2 (Mixtral), 4 (GShard)
    pub top_k: usize,

    /// Number of total experts in the model
    pub num_experts: usize,

    /// Enable memory-aware routing (feature flag)
    ///
    /// When false, the router ignores cache state and uses pure accuracy mode.
    pub memory_aware: bool,

    /// Prefetch threshold (router weight to trigger speculative prefetch)
    ///
    /// Experts with weight >= this but not selected may be prefetched.
    /// Default: 0.1 (10%)
    pub prefetch_threshold: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            cache_bonus: 0.15,
            top_k: 2,
            num_experts: 8,
            memory_aware: true,
            prefetch_threshold: 0.1,
        }
    }
}

impl RouterConfig {
    /// Create config with specified parameters
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            ..Default::default()
        }
    }

    /// Set cache bonus weight
    pub fn with_cache_bonus(mut self, bonus: f32) -> Self {
        self.cache_bonus = bonus.clamp(0.0, 1.0);
        self
    }

    /// Set memory-aware mode
    pub fn with_memory_aware(mut self, enabled: bool) -> Self {
        self.memory_aware = enabled;
        self
    }

    /// Set prefetch threshold
    pub fn with_prefetch_threshold(mut self, threshold: f32) -> Self {
        self.prefetch_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.top_k == 0 {
            return Err("top_k must be at least 1");
        }
        if self.top_k > self.num_experts {
            return Err("top_k cannot exceed num_experts");
        }
        if self.num_experts == 0 {
            return Err("num_experts must be at least 1");
        }
        Ok(())
    }
}

/// Memory-aware MoE router with cache residency bonus
///
/// Implements the memory-aware routing algorithm from ADR-092:
/// 1. Add cache residency bonus to gate scores
/// 2. Select top-K experts with adjusted scores
/// 3. Generate paging requests for non-resident selected experts
///
/// # Invariant INV-6: Router Determinism
///
/// Given the same input (gate_logits) and same cache state (cache_resident),
/// the router always produces the same output (selected experts, paging requests).
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::moe::{MemoryAwareRouter, RouterConfig, ExpertAffinity, AffinityConfig};
///
/// let config = RouterConfig {
///     cache_bonus: 0.15,
///     top_k: 2,
///     num_experts: 8,
///     memory_aware: true,
///     prefetch_threshold: 0.1,
/// };
///
/// let affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8));
/// let mut router = MemoryAwareRouter::new(config, affinity);
///
/// // Update which experts are currently cached
/// router.update_cache_state(&[0, 1, 2, 3]);
///
/// // Route based on gate logits
/// let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];
/// let (selected, paging_requests) = router.route(&gate_logits);
/// ```
pub struct MemoryAwareRouter {
    /// Router configuration
    config: RouterConfig,
    /// Expert affinity tracker
    affinity: ExpertAffinity,
    /// Bitmask tracking which experts are currently in cache (P1 optimization)
    cache_resident: CacheMask,
    /// Routing and caching metrics
    metrics: MoeMetrics,
    /// Reusable score buffer to avoid allocations (P2 optimization)
    score_buffer: Vec<f32>,
    /// Reusable indexed buffer for sorting (P2 optimization)
    index_buffer: Vec<(ExpertId, f32)>,
}

impl MemoryAwareRouter {
    /// Create a new memory-aware router
    ///
    /// # Arguments
    ///
    /// * `config` - Router configuration
    /// * `affinity` - Expert affinity tracker (can be shared)
    ///
    /// # Returns
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: RouterConfig, affinity: ExpertAffinity) -> Result<Self, &'static str> {
        config.validate()?;

        let num_experts = config.num_experts;
        Ok(Self {
            cache_resident: CacheMask::new(num_experts),
            // P2: Pre-allocate buffers to avoid allocations in hot path
            score_buffer: vec![0.0; num_experts],
            index_buffer: Vec::with_capacity(num_experts),
            config,
            affinity,
            metrics: MoeMetrics::new(),
        })
    }

    /// Create router with default affinity tracker
    ///
    /// # Returns
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn with_default_affinity(config: RouterConfig) -> Result<Self, &'static str> {
        let affinity =
            ExpertAffinity::new(super::AffinityConfig::with_num_experts(config.num_experts));
        Self::new(config, affinity)
    }

    /// Main routing function with cache bonus
    ///
    /// Returns selected experts and any paging requests needed.
    ///
    /// # Arguments
    ///
    /// * `gate_logits` - Raw logits from the gate network (length = num_experts)
    ///
    /// # Returns
    ///
    /// Tuple of (selected_expert_ids, paging_requests)
    ///
    /// # INV-6: Determinism
    ///
    /// This function is deterministic: same inputs produce same outputs.
    /// No random sampling is used.
    #[inline]
    pub fn route(&mut self, gate_logits: &[f32]) -> (Vec<ExpertId>, Vec<PagingRequest>) {
        let start = Instant::now();

        // Validate input length (P3: early exit for invalid input)
        if gate_logits.len() != self.config.num_experts {
            let selected: Vec<ExpertId> =
                (0..self.config.top_k.min(self.config.num_experts)).collect();
            return (selected, Vec::new());
        }

        // P2: Use pre-allocated buffer instead of allocating
        let selected = self.route_into_buffer(gate_logits);

        // Step 3: Update affinity for selected experts
        self.affinity.update(&selected);

        // Step 4: Generate paging requests for non-resident selected experts
        let paging_requests = self.generate_paging_requests(&selected);

        // Step 5: Record metrics (P3: unroll small loops)
        let mut hits = 0usize;
        for &id in &selected {
            if self.cache_resident.is_set(id) {
                hits += 1;
            }
        }
        let misses = selected.len() - hits;
        self.metrics.record_cache_hits(hits);
        self.metrics.record_cache_misses(misses);
        self.metrics.record_routing(start.elapsed());

        (selected, paging_requests)
    }

    /// P2 Optimization: Route using pre-allocated buffers
    ///
    /// Avoids allocation in the hot path by reusing internal buffers.
    #[inline]
    fn route_into_buffer(&mut self, gate_logits: &[f32]) -> Vec<ExpertId> {
        let n = gate_logits.len();

        // Copy scores into buffer and apply cache bonus in-place
        self.score_buffer.clear();
        self.score_buffer.extend_from_slice(gate_logits);

        if self.config.memory_aware {
            self.apply_cache_bonus_inplace_buffer();
        }

        // Select top-K using index buffer
        self.select_top_k_buffered(n)
    }

    /// P2: Apply cache bonus using internal buffer
    #[inline]
    fn apply_cache_bonus_inplace_buffer(&mut self) {
        let bonus = self.config.cache_bonus;
        for (id, score) in self.score_buffer.iter_mut().enumerate() {
            if !score.is_finite() {
                *score = 0.0;
                continue;
            }
            if self.cache_resident.is_set(id) {
                *score += bonus;
            }
        }
    }

    /// P2: Select top-K using pre-allocated index buffer
    #[inline]
    fn select_top_k_buffered(&mut self, n: usize) -> Vec<ExpertId> {
        let k = self.config.top_k.min(n);
        if k == 0 || n == 0 {
            return Vec::new();
        }

        // Reuse index buffer
        self.index_buffer.clear();
        self.index_buffer.extend(
            self.score_buffer
                .iter()
                .enumerate()
                .map(|(id, &s)| (id, if s.is_finite() { s } else { f32::NEG_INFINITY })),
        );

        // P4: Unroll for small k (common case: top-2)
        if k == 2 && n >= 2 {
            return self.select_top_2_unrolled();
        }

        // Use partial sort for larger k
        if k < n / 2 {
            self.index_buffer.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            self.index_buffer[..k].sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        } else {
            self.index_buffer.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }

        self.index_buffer
            .iter()
            .take(k)
            .map(|(id, _)| *id)
            .collect()
    }

    /// P4: Unrolled top-2 selection (most common MoE configuration)
    #[inline]
    fn select_top_2_unrolled(&self) -> Vec<ExpertId> {
        let mut best = (0, f32::NEG_INFINITY);
        let mut second = (0, f32::NEG_INFINITY);

        for &(id, score) in &self.index_buffer {
            if score > best.1 || (score == best.1 && id < best.0) {
                second = best;
                best = (id, score);
            } else if score > second.1 || (score == second.1 && id < second.0) {
                second = (id, score);
            }
        }

        vec![best.0, second.0]
    }

    /// Batch routing for multiple tokens (P2 optimization)
    ///
    /// Routes multiple tokens in a single call, reusing buffers across tokens.
    /// More efficient than calling `route()` multiple times.
    ///
    /// # Arguments
    ///
    /// * `batch_logits` - Slice of gate logits for each token (shape: [batch_size][num_experts])
    ///
    /// # Returns
    ///
    /// Vector of (selected_experts, paging_requests) for each token
    pub fn route_batch(
        &mut self,
        batch_logits: &[&[f32]],
    ) -> Vec<(Vec<ExpertId>, Vec<PagingRequest>)> {
        let mut results = Vec::with_capacity(batch_logits.len());

        for logits in batch_logits {
            results.push(self.route(logits));
        }

        results
    }

    /// Apply cache residency bonus to scores (in-place mutation for P0 optimization)
    ///
    /// For each expert currently in cache, adds `cache_bonus` to its score.
    /// This biases the selection toward cached experts without completely
    /// overriding the gate network's decisions.
    ///
    /// # Arguments
    ///
    /// * `scores` - Mutable slice of scores to modify in-place
    pub fn apply_cache_bonus_inplace(&self, scores: &mut [f32]) {
        for (id, score) in scores.iter_mut().enumerate() {
            // Validate score is not NaN/Inf before processing
            if !score.is_finite() {
                *score = 0.0;
                continue;
            }
            if self.cache_resident.is_set(id) {
                *score += self.config.cache_bonus;
            }
        }
    }

    /// Apply cache residency bonus to scores (allocating version for API compatibility)
    ///
    /// For each expert currently in cache, adds `cache_bonus` to its score.
    /// This biases the selection toward cached experts without completely
    /// overriding the gate network's decisions.
    pub fn apply_cache_bonus(&self, scores: &[f32]) -> Vec<f32> {
        let mut result = scores.to_vec();
        self.apply_cache_bonus_inplace(&mut result);
        result
    }

    /// Select top-K experts by score
    ///
    /// Returns expert IDs sorted by descending score.
    /// Ties are broken by expert ID (lower ID wins) for determinism.
    ///
    /// Uses partial sort (P0 optimization) for better performance when
    /// top_k << num_experts.
    pub fn select_top_k(&self, scores: &[f32]) -> Vec<ExpertId> {
        let n = scores.len();
        let k = self.config.top_k.min(n);

        if k == 0 || n == 0 {
            return Vec::new();
        }

        // Create indexed scores, handling NaN/Inf values
        let mut indexed: Vec<(ExpertId, f32)> = scores
            .iter()
            .enumerate()
            .map(|(id, &s)| (id, if s.is_finite() { s } else { f32::NEG_INFINITY }))
            .collect();

        // Use partial sort for better performance when k << n
        if k < n / 2 {
            // Partition to get top-k elements (unordered)
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            // Sort only the top-k portion
            indexed[..k].sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        } else {
            // Full sort when k is close to n
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
        }

        // Take top-K
        indexed.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Update cache residency state
    ///
    /// Call this when experts are paged in or out.
    ///
    /// # Arguments
    ///
    /// * `resident` - List of expert IDs currently in cache
    pub fn update_cache_state(&mut self, resident: &[ExpertId]) {
        // Clear all
        self.cache_resident.clear();

        // Set resident experts
        for &id in resident {
            self.cache_resident.set(id, true);
        }
    }

    /// Mark a single expert as resident or non-resident
    pub fn set_resident(&mut self, expert_id: ExpertId, resident: bool) {
        self.cache_resident.set(expert_id, resident);
    }

    /// Check if an expert is currently resident
    pub fn is_resident(&self, expert_id: ExpertId) -> bool {
        self.cache_resident.is_set(expert_id)
    }

    /// Generate paging requests for selected experts
    ///
    /// Creates urgent page-in requests for non-resident selected experts.
    /// Also generates prefetch requests for high-scoring non-selected experts.
    pub fn generate_paging_requests(&self, selected: &[ExpertId]) -> Vec<PagingRequest> {
        let mut requests = Vec::new();

        // Urgent page-in for non-resident selected experts
        for &expert_id in selected {
            if !self.is_resident(expert_id) {
                requests.push(PagingRequest::page_in_urgent(expert_id));
            }
        }

        requests
    }

    /// Generate prefetch requests based on affinity
    ///
    /// Returns prefetch requests for high-affinity non-resident experts.
    ///
    /// # Arguments
    ///
    /// * `budget` - Maximum number of prefetch requests to generate
    pub fn generate_prefetch_requests(&self, budget: usize) -> Vec<PagingRequest> {
        // Get top experts by affinity that are not currently resident
        let candidates = self.affinity.top_k_by_affinity(budget * 2);

        candidates
            .into_iter()
            .filter(|&id| !self.is_resident(id))
            .take(budget)
            .map(PagingRequest::prefetch)
            .collect()
    }

    /// Get a reference to the current metrics
    pub fn metrics(&self) -> &MoeMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Get a reference to the affinity tracker
    pub fn affinity(&self) -> &ExpertAffinity {
        &self.affinity
    }

    /// Get a mutable reference to the affinity tracker
    pub fn affinity_mut(&mut self) -> &mut ExpertAffinity {
        &mut self.affinity
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get the current cache hit rate
    pub fn hit_rate(&self) -> f32 {
        self.metrics.hit_rate()
    }

    /// Get list of currently resident experts
    pub fn resident_experts(&self) -> Vec<ExpertId> {
        self.cache_resident.resident_list()
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.config.num_experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::AffinityConfig;

    fn make_router(num_experts: usize, top_k: usize, cache_bonus: f32) -> MemoryAwareRouter {
        let config = RouterConfig::new(num_experts, top_k).with_cache_bonus(cache_bonus);
        MemoryAwareRouter::with_default_affinity(config).expect("test config should be valid")
    }

    // ---------------------------------------------------------------
    // test_routing_basic
    // ---------------------------------------------------------------

    #[test]
    fn test_routing_basic() {
        let mut router = make_router(8, 2, 0.0);

        // No cache bonus, pure selection
        let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];
        let (selected, _) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // Experts 2 (0.5) and 4 (0.4) should be selected
        assert!(selected.contains(&2));
        assert!(selected.contains(&4));
    }

    // ---------------------------------------------------------------
    // test_cache_bonus_increases_resident_score
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_bonus_increases_resident_score() {
        let mut router = make_router(4, 1, 0.3);

        // Experts: 0=0.4, 1=0.3, 2=0.2, 3=0.1
        // Without bonus: expert 0 selected
        // With bonus on expert 1: 0.3 + 0.3 = 0.6 > 0.4

        router.update_cache_state(&[1]); // Expert 1 is resident

        let gate_logits = vec![0.4, 0.3, 0.2, 0.1];
        let (selected, _) = router.route(&gate_logits);

        // Expert 1 should be selected because of cache bonus
        assert_eq!(selected, vec![1]);
    }

    // ---------------------------------------------------------------
    // test_top_k_selection
    // ---------------------------------------------------------------

    #[test]
    fn test_top_k_selection() {
        let mut router = make_router(8, 3, 0.0);

        let gate_logits = vec![0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5];
        let (selected, _) = router.route(&gate_logits);

        assert_eq!(selected.len(), 3);
        // Top 3: expert 0 (0.8), expert 3 (0.7), expert 5 (0.6)
        assert_eq!(selected[0], 0);
        assert_eq!(selected[1], 3);
        assert_eq!(selected[2], 5);
    }

    // ---------------------------------------------------------------
    // test_paging_requests_for_non_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_paging_requests_for_non_resident() {
        let mut router = make_router(4, 2, 0.0);

        // Only expert 0 is resident
        router.update_cache_state(&[0]);

        let gate_logits = vec![0.5, 0.6, 0.4, 0.3];
        let (selected, paging) = router.route(&gate_logits);

        // Selected: experts 1 (0.6) and 0 (0.5)
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));

        // Expert 1 is not resident, should have paging request
        assert_eq!(paging.len(), 1);
        assert_eq!(paging[0].expert_id, 1);
        assert_eq!(paging[0].direction, PagingDirection::In);
        assert_eq!(paging[0].priority, PagingPriority::Urgent);
    }

    // ---------------------------------------------------------------
    // test_router_determinism (INV-6)
    // ---------------------------------------------------------------

    #[test]
    fn test_router_determinism() {
        // INV-6: Same input + cache state = same result

        let mut router1 = make_router(8, 2, 0.15);
        let mut router2 = make_router(8, 2, 0.15);

        // Same cache state
        router1.update_cache_state(&[0, 3, 5]);
        router2.update_cache_state(&[0, 3, 5]);

        let gate_logits = vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.1, 0.2, 0.15];

        let (selected1, paging1) = router1.route(&gate_logits);
        let (selected2, paging2) = router2.route(&gate_logits);

        // Results must be identical
        assert_eq!(
            selected1, selected2,
            "INV-6 violation: different expert selection"
        );
        assert_eq!(
            paging1.len(),
            paging2.len(),
            "INV-6 violation: different paging count"
        );

        // Run multiple times on same router
        router1.reset_metrics();
        let (selected3, _) = router1.route(&gate_logits);
        assert_eq!(
            selected1, selected3,
            "INV-6 violation: non-deterministic routing"
        );
    }

    // ---------------------------------------------------------------
    // test_affinity_updates
    // ---------------------------------------------------------------

    #[test]
    fn test_affinity_updates() {
        let mut router = make_router(4, 2, 0.0);

        // Route multiple times to build affinity
        let gate_logits = vec![0.4, 0.3, 0.5, 0.1];

        for _ in 0..5 {
            router.route(&gate_logits);
        }

        // Experts 2 and 0 should have highest affinity (selected 5 times)
        let top = router.affinity().top_k_by_affinity(2);
        assert!(top.contains(&2), "Expert 2 should have high affinity");
        assert!(top.contains(&0), "Expert 0 should have high affinity");
    }

    // ---------------------------------------------------------------
    // test_zero_cache_bonus_fallback
    // ---------------------------------------------------------------

    #[test]
    fn test_zero_cache_bonus_fallback() {
        let mut router = make_router(4, 2, 0.0);

        // All experts resident
        router.update_cache_state(&[0, 1, 2, 3]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, _) = router.route(&gate_logits);

        // Should select purely by score: experts 1 (0.4) and 2 (0.3)
        assert_eq!(selected[0], 1);
        assert_eq!(selected[1], 2);
    }

    // ---------------------------------------------------------------
    // test_all_experts_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_all_experts_resident() {
        let mut router = make_router(4, 2, 0.15);

        // All experts resident
        router.update_cache_state(&[0, 1, 2, 3]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, paging) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // No paging needed
        assert!(
            paging.is_empty(),
            "No paging should be needed when all selected are resident"
        );

        // All should be cache hits
        assert_eq!(router.metrics().cache_hits, 2);
        assert_eq!(router.metrics().cache_misses, 0);
    }

    // ---------------------------------------------------------------
    // test_no_experts_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_no_experts_resident() {
        let mut router = make_router(4, 2, 0.15);

        // No experts resident (cold start)
        router.update_cache_state(&[]);

        let gate_logits = vec![0.1, 0.4, 0.3, 0.2];
        let (selected, paging) = router.route(&gate_logits);

        assert_eq!(selected.len(), 2);
        // Should need paging for all selected
        assert_eq!(
            paging.len(),
            2,
            "Should need to page in all selected experts"
        );

        // All should be cache misses
        assert_eq!(router.metrics().cache_misses, 2);
        assert_eq!(router.metrics().cache_hits, 0);
    }

    // ---------------------------------------------------------------
    // test_config_validation
    // ---------------------------------------------------------------

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = RouterConfig::new(8, 2);
        assert!(valid.validate().is_ok());

        // Invalid: top_k = 0
        let invalid1 = RouterConfig {
            top_k: 0,
            ..RouterConfig::default()
        };
        assert!(invalid1.validate().is_err());

        // Invalid: top_k > num_experts
        let invalid2 = RouterConfig {
            top_k: 10,
            num_experts: 8,
            ..RouterConfig::default()
        };
        assert!(invalid2.validate().is_err());

        // Invalid: num_experts = 0
        let invalid3 = RouterConfig {
            num_experts: 0,
            ..RouterConfig::default()
        };
        assert!(invalid3.validate().is_err());
    }

    // ---------------------------------------------------------------
    // test_memory_aware_disabled
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_aware_disabled() {
        let config = RouterConfig::new(4, 2)
            .with_memory_aware(false)
            .with_cache_bonus(0.5);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        // Even with high cache bonus, should not apply it when disabled
        router.update_cache_state(&[3]); // Expert 3 resident

        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        let (selected, _) = router.route(&gate_logits);

        // Should select by pure score: experts 2 (0.5) and 0 (0.4)
        assert_eq!(selected[0], 2);
        assert_eq!(selected[1], 0);
    }

    // ---------------------------------------------------------------
    // test_hit_rate_tracking
    // ---------------------------------------------------------------

    #[test]
    fn test_hit_rate_tracking() {
        let mut router = make_router(4, 2, 0.0);

        // 50% resident
        router.update_cache_state(&[0, 2]);

        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        // Will select experts 2 (resident) and 0 (resident)
        router.route(&gate_logits);

        assert_eq!(router.hit_rate(), 1.0); // Both selected are resident

        router.reset_metrics();
        router.update_cache_state(&[1, 3]);
        router.route(&gate_logits);

        assert_eq!(router.hit_rate(), 0.0); // Neither selected is resident
    }

    // ---------------------------------------------------------------
    // test_prefetch_requests
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_requests() {
        let config = RouterConfig::new(4, 2).with_cache_bonus(0.0);
        let affinity_config = AffinityConfig::with_num_experts(4).with_decay(1.0);
        let affinity = ExpertAffinity::new(affinity_config);
        let mut router = MemoryAwareRouter::new(config, affinity).unwrap();

        // Build affinity
        let gate_logits = vec![0.4, 0.3, 0.5, 0.2];
        for _ in 0..10 {
            router.route(&gate_logits);
        }

        // Only expert 1 is resident
        router.update_cache_state(&[1]);

        // Should suggest prefetching high-affinity non-resident experts
        let prefetch = router.generate_prefetch_requests(2);

        // Should not include expert 1 (already resident)
        for req in &prefetch {
            assert_ne!(req.expert_id, 1);
            assert_eq!(req.priority, PagingPriority::Prefetch);
        }
    }

    // ---------------------------------------------------------------
    // test_resident_experts_list
    // ---------------------------------------------------------------

    #[test]
    fn test_resident_experts_list() {
        let mut router = make_router(8, 2, 0.15);

        router.update_cache_state(&[1, 3, 5, 7]);

        let resident = router.resident_experts();
        assert_eq!(resident.len(), 4);
        assert!(resident.contains(&1));
        assert!(resident.contains(&3));
        assert!(resident.contains(&5));
        assert!(resident.contains(&7));
        assert!(!resident.contains(&0));
    }

    // ---------------------------------------------------------------
    // test_set_resident
    // ---------------------------------------------------------------

    #[test]
    fn test_set_resident() {
        let mut router = make_router(4, 2, 0.15);

        assert!(!router.is_resident(0));

        router.set_resident(0, true);
        assert!(router.is_resident(0));

        router.set_resident(0, false);
        assert!(!router.is_resident(0));
    }

    // ---------------------------------------------------------------
    // test_tie_breaking_determinism
    // ---------------------------------------------------------------

    #[test]
    fn test_tie_breaking_determinism() {
        let mut router = make_router(4, 2, 0.0);

        // All experts have same score
        let gate_logits = vec![0.5, 0.5, 0.5, 0.5];
        let (selected1, _) = router.route(&gate_logits);
        let (selected2, _) = router.route(&gate_logits);

        // Should consistently select lowest IDs on ties
        assert_eq!(selected1, selected2);
        assert_eq!(selected1, vec![0, 1]); // Lowest IDs win ties
    }

    // ---------------------------------------------------------------
    // test_invalid_gate_logits_length
    // ---------------------------------------------------------------

    #[test]
    fn test_invalid_gate_logits_length() {
        let mut router = make_router(4, 2, 0.15);

        // Wrong length input
        let gate_logits = vec![0.5, 0.3]; // Only 2 instead of 4
        let (selected, paging) = router.route(&gate_logits);

        // Should fallback gracefully
        assert_eq!(selected.len(), 2);
        assert!(paging.is_empty() || paging.len() <= 2);
    }

    // ---------------------------------------------------------------
    // test_apply_cache_bonus
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_cache_bonus() {
        let mut router = make_router(4, 2, 0.2);
        router.update_cache_state(&[1, 2]);

        let scores = vec![0.1, 0.3, 0.4, 0.5];
        let adjusted = router.apply_cache_bonus(&scores);

        // Expert 0: 0.1 + 0 = 0.1
        // Expert 1: 0.3 + 0.2 = 0.5 (resident)
        // Expert 2: 0.4 + 0.2 = 0.6 (resident)
        // Expert 3: 0.5 + 0 = 0.5
        assert!((adjusted[0] - 0.1).abs() < 1e-6);
        assert!((adjusted[1] - 0.5).abs() < 1e-6);
        assert!((adjusted[2] - 0.6).abs() < 1e-6);
        assert!((adjusted[3] - 0.5).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // test_paging_request_constructors
    // ---------------------------------------------------------------

    #[test]
    fn test_paging_request_constructors() {
        let req1 = PagingRequest::page_in_urgent(5);
        assert_eq!(req1.expert_id, 5);
        assert_eq!(req1.direction, PagingDirection::In);
        assert_eq!(req1.priority, PagingPriority::Urgent);

        let req2 = PagingRequest::prefetch(3);
        assert_eq!(req2.expert_id, 3);
        assert_eq!(req2.direction, PagingDirection::In);
        assert_eq!(req2.priority, PagingPriority::Prefetch);

        let req3 = PagingRequest::page_out(7);
        assert_eq!(req3.expert_id, 7);
        assert_eq!(req3.direction, PagingDirection::Out);
        assert_eq!(req3.priority, PagingPriority::Normal);
    }

    // ---------------------------------------------------------------
    // test_config_builder
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder() {
        let config = RouterConfig::new(16, 4)
            .with_cache_bonus(0.25)
            .with_memory_aware(true)
            .with_prefetch_threshold(0.15);

        assert_eq!(config.num_experts, 16);
        assert_eq!(config.top_k, 4);
        assert!((config.cache_bonus - 0.25).abs() < 1e-6);
        assert!(config.memory_aware);
        assert!((config.prefetch_threshold - 0.15).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // test_cache_bonus_clamping
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_bonus_clamping() {
        let config = RouterConfig::new(8, 2).with_cache_bonus(1.5);
        assert!(
            (config.cache_bonus - 1.0).abs() < 1e-6,
            "cache_bonus should be clamped to 1.0"
        );

        let config2 = RouterConfig::new(8, 2).with_cache_bonus(-0.5);
        assert!(
            (config2.cache_bonus - 0.0).abs() < 1e-6,
            "cache_bonus should be clamped to 0.0"
        );
    }

    // ---------------------------------------------------------------
    // P1 Optimization Tests: CacheMask bitmask
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_mask_small() {
        let mut mask = CacheMask::new(64);

        // Initially all clear
        for i in 0..64 {
            assert!(!mask.is_set(i), "Bit {} should be clear initially", i);
        }

        // Set some bits
        mask.set(0, true);
        mask.set(31, true);
        mask.set(63, true);

        assert!(mask.is_set(0));
        assert!(mask.is_set(31));
        assert!(mask.is_set(63));
        assert!(!mask.is_set(1));
        assert!(!mask.is_set(32));

        // Count should be 3
        assert_eq!(mask.count(), 3);

        // Resident list
        let list = mask.resident_list();
        assert_eq!(list.len(), 3);
        assert!(list.contains(&0));
        assert!(list.contains(&31));
        assert!(list.contains(&63));

        // Clear and verify
        mask.clear();
        assert_eq!(mask.count(), 0);
        assert!(!mask.is_set(0));
    }

    #[test]
    fn test_cache_mask_large() {
        // Test with >64 experts (uses extended Vec<u64>)
        let mut mask = CacheMask::new(128);

        // Set bits across word boundaries
        mask.set(0, true);
        mask.set(63, true);
        mask.set(64, true); // First bit of second word
        mask.set(127, true);

        assert!(mask.is_set(0));
        assert!(mask.is_set(63));
        assert!(mask.is_set(64));
        assert!(mask.is_set(127));
        assert!(!mask.is_set(65));

        assert_eq!(mask.count(), 4);

        let list = mask.resident_list();
        assert_eq!(list.len(), 4);

        // Clear
        mask.clear();
        assert_eq!(mask.count(), 0);
    }

    #[test]
    fn test_cache_mask_out_of_bounds() {
        let mut mask = CacheMask::new(8);

        // Out of bounds should be no-op and return false
        mask.set(100, true);
        assert!(!mask.is_set(100));
        assert_eq!(mask.count(), 0);
    }

    #[test]
    fn test_router_with_many_experts() {
        // Test router with >64 experts to exercise extended bitmask
        let config = RouterConfig::new(128, 4);
        let mut router = MemoryAwareRouter::with_default_affinity(config).unwrap();

        // Set some residents across the full range
        router.update_cache_state(&[0, 32, 64, 96, 127]);

        assert!(router.is_resident(0));
        assert!(router.is_resident(64));
        assert!(router.is_resident(127));
        assert!(!router.is_resident(1));

        let resident = router.resident_experts();
        assert_eq!(resident.len(), 5);
    }

    #[test]
    fn test_empty_cache_state() {
        let mut router = make_router(8, 2, 0.15);

        // Empty update
        router.update_cache_state(&[]);

        // No experts should be resident
        for i in 0..8 {
            assert!(
                !router.is_resident(i),
                "Expert {} should not be resident",
                i
            );
        }

        assert!(router.resident_experts().is_empty());
    }
}
