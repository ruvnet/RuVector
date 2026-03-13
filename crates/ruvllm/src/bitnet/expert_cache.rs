//! Expert Hot-Set Cache and MoE Batch Scheduler
//!
//! This module implements memory bandwidth optimizations for MoE inference:
//!
//! - **ExpertCache**: Tracks which experts are "hot" (recently/frequently accessed)
//!   and manages eviction to keep working-set size bounded. With top-K=2 active
//!   experts per token but 8 total experts per layer, naive traversal thrashes
//!   L2/L3 cache. The hot-set cache keeps the 4 most relevant experts warm.
//!
//! - **MoeBatchScheduler**: Reorders expert execution across a token batch so that
//!   all tokens routed to the same expert are processed contiguously. This converts
//!   random expert access into sequential scans, maximizing cache-line reuse.
//!
//! - **Prefetcher trait**: Abstraction for platform-specific memory prefetch
//!   intrinsics (x86 `_mm_prefetch`, aarch64 `__pld`). Currently ships with a
//!   no-op implementation; architecture-specific backends can be added without
//!   changing call sites.
//!
//! ## Memory Layout Context
//!
//! Each expert's ternary weights occupy roughly `ceil(rows * cols / 4)` packed
//! bytes plus `ceil(rows * cols / block_size) * 4` scale bytes. For a 30B MoE
//! model with `intermediate_size=11008` and `hidden_size=4096`:
//!
//! ```text
//! gate_proj: 11008 * 4096 * 2 bits / 8 = ~11.3 MB packed
//! up_proj:   11008 * 4096 * 2 bits / 8 = ~11.3 MB packed
//! down_proj: 4096 * 11008 * 2 bits / 8 = ~11.3 MB packed
//! Total per expert: ~33.9 MB packed + scales
//! ```
//!
//! With 8 experts that is ~271 MB per layer. Keeping only 4 hot halves the
//! cache pressure while covering the top-2 active plus 2 likely next picks.

use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Eviction policy for the expert hot-set cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used: evict the expert with the oldest access timestamp.
    Lru,
    /// Least Frequently Used: evict the expert with the lowest total access count.
    Lfu,
    /// Adaptive: use LFU when frequency distribution is skewed (top expert has
    /// 3x the accesses of the least-used), otherwise fall back to LRU. This
    /// handles both steady-state routing (where certain experts dominate) and
    /// transient shifts (where recency matters more).
    Adaptive,
}

/// Configuration for the expert hot-set cache.
#[derive(Debug, Clone)]
pub struct ExpertCacheConfig {
    /// Maximum number of experts kept in the hot set.
    ///
    /// Default is 4: with top-K=2 active per token, keeping 4 warm provides
    /// temporal locality for the next 1-2 tokens without over-provisioning.
    pub max_hot_experts: usize,

    /// Router weight threshold for speculative prefetch.
    ///
    /// If an expert's softmax weight exceeds this threshold but the expert is
    /// not in the current top-K selection, it is a prefetch candidate. This
    /// catches experts that are "almost selected" and likely to be needed soon.
    ///
    /// Default is 0.1 (10% softmax probability).
    pub prefetch_threshold: f32,

    /// Eviction policy when the hot set is full and a new expert must be admitted.
    pub eviction_policy: EvictionPolicy,
}

impl Default for ExpertCacheConfig {
    fn default() -> Self {
        Self {
            max_hot_experts: 4,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Runtime statistics for the expert cache.
///
/// Tracks hits, misses, evictions, and prefetch effectiveness to enable
/// tuning of `max_hot_experts` and `prefetch_threshold` parameters.
#[derive(Debug, Clone, Default)]
pub struct ExpertCacheStats {
    /// Number of accesses where the expert was already in the hot set.
    pub hits: usize,
    /// Number of accesses where the expert was not in the hot set.
    pub misses: usize,
    /// Number of experts evicted from the hot set.
    pub evictions: usize,
    /// Number of accesses that hit an expert that was speculatively prefetched.
    pub prefetch_hits: usize,
}

impl ExpertCacheStats {
    /// Compute the cache hit rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if no accesses have been recorded.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f32 / total as f32
    }
}

// ============================================================================
// ExpertCache
// ============================================================================

/// Hot-set cache for MoE expert weights.
///
/// Maintains a bounded set of "hot" expert IDs whose weight tensors should be
/// kept in CPU cache (L2/L3). The cache does not own the weight data itself;
/// it tracks which expert IDs are hot so that the inference loop can skip
/// unnecessary memory traffic for cold experts.
///
/// # Usage
///
/// ```rust,ignore
/// use ruvllm::bitnet::expert_cache::{ExpertCache, ExpertCacheConfig};
///
/// let config = ExpertCacheConfig::default();
/// let mut cache = ExpertCache::new(8, config);
///
/// // Record that experts 2 and 5 were selected by the router
/// let hit_2 = cache.access(2); // false (cold miss on first access)
/// let hit_5 = cache.access(5); // false
///
/// // Next token: expert 2 selected again
/// let hit_2 = cache.access(2); // true (hot hit)
/// ```
pub struct ExpertCache {
    /// Total number of experts in the model (per layer).
    num_experts: usize,
    /// (expert_id, last_access_timestamp) for each expert currently in the hot set.
    hot_set: Vec<(usize, u64)>,
    /// Per-expert total access count, indexed by expert_id. Used for LFU eviction.
    frequency: Vec<usize>,
    /// Set of expert IDs that were admitted via speculative prefetch (not yet
    /// accessed by the router). Used to track prefetch hit effectiveness.
    prefetched: Vec<bool>,
    /// Cache configuration.
    config: ExpertCacheConfig,
    /// Runtime statistics.
    stats: ExpertCacheStats,
    /// Monotonically increasing counter used as a logical timestamp for LRU ordering.
    access_counter: u64,
}

impl ExpertCache {
    /// Create a new expert cache.
    ///
    /// # Arguments
    ///
    /// * `num_experts` - Total number of experts per layer in the model.
    /// * `config` - Cache configuration (hot-set size, thresholds, policy).
    pub fn new(num_experts: usize, config: ExpertCacheConfig) -> Self {
        Self {
            num_experts,
            hot_set: Vec::with_capacity(config.max_hot_experts),
            frequency: vec![0; num_experts],
            prefetched: vec![false; num_experts],
            config,
            stats: ExpertCacheStats::default(),
            access_counter: 0,
        }
    }

    /// Record an access to the given expert.
    ///
    /// If the expert is already in the hot set this is a cache hit: its
    /// timestamp is refreshed and its frequency count is incremented.
    ///
    /// If the expert is cold (not in the hot set) this is a cache miss: the
    /// expert is admitted (potentially evicting another), and the miss is
    /// recorded in stats.
    ///
    /// # Returns
    ///
    /// `true` if the expert was already hot (cache hit), `false` otherwise.
    pub fn access(&mut self, expert_id: usize) -> bool {
        self.access_counter += 1;
        let timestamp = self.access_counter;

        // Always bump frequency
        if expert_id < self.num_experts {
            self.frequency[expert_id] += 1;
        }

        // Check if expert is already in the hot set
        if let Some(pos) = self.hot_set.iter().position(|&(id, _)| id == expert_id) {
            // Hit: refresh timestamp
            self.hot_set[pos].1 = timestamp;
            self.stats.hits += 1;

            // Track prefetch effectiveness
            if expert_id < self.prefetched.len() && self.prefetched[expert_id] {
                self.stats.prefetch_hits += 1;
                self.prefetched[expert_id] = false;
            }

            return true;
        }

        // Miss: admit the expert
        self.stats.misses += 1;
        self.admit(expert_id);
        false
    }

    /// Check whether a not-yet-selected expert should be speculatively prefetched.
    ///
    /// Returns `true` if:
    /// 1. The expert is not already in the hot set, AND
    /// 2. Its router weight exceeds the configured `prefetch_threshold`.
    ///
    /// The caller is responsible for actually performing the prefetch (e.g.,
    /// issuing prefetch instructions or touching the memory).
    pub fn should_prefetch(&self, expert_id: usize, router_weight: f32) -> bool {
        if router_weight <= self.config.prefetch_threshold {
            return false;
        }
        !self.is_hot(expert_id)
    }

    /// Suggest which expert to evict from the hot set.
    ///
    /// Returns `None` if the hot set is not full. Otherwise returns the
    /// expert_id that should be evicted according to the configured policy.
    pub fn suggest_eviction(&self) -> Option<usize> {
        if self.hot_set.len() < self.config.max_hot_experts {
            return None;
        }

        match self.config.eviction_policy {
            EvictionPolicy::Lru => self.suggest_lru_eviction(),
            EvictionPolicy::Lfu => self.suggest_lfu_eviction(),
            EvictionPolicy::Adaptive => self.suggest_adaptive_eviction(),
        }
    }

    /// Evict a specific expert from the hot set.
    ///
    /// No-op if the expert is not currently hot.
    pub fn evict(&mut self, expert_id: usize) {
        if let Some(pos) = self.hot_set.iter().position(|&(id, _)| id == expert_id) {
            self.hot_set.swap_remove(pos);
            self.stats.evictions += 1;
        }
    }

    /// Admit an expert into the hot set.
    ///
    /// If the hot set is already at capacity, evicts one expert first according
    /// to the configured eviction policy. If the expert is already hot, this
    /// is a no-op.
    pub fn admit(&mut self, expert_id: usize) {
        // Already hot: nothing to do
        if self.is_hot(expert_id) {
            return;
        }

        // Evict if at capacity
        if self.hot_set.len() >= self.config.max_hot_experts {
            if let Some(victim) = self.suggest_eviction() {
                self.evict(victim);
            }
        }

        let timestamp = self.access_counter;
        self.hot_set.push((expert_id, timestamp));
    }

    /// Admit an expert via speculative prefetch.
    ///
    /// Like `admit`, but marks the expert as prefetched so that a subsequent
    /// `access` hit can be attributed to the prefetch in stats.
    pub fn prefetch_admit(&mut self, expert_id: usize) {
        if expert_id < self.prefetched.len() {
            self.prefetched[expert_id] = true;
        }
        self.admit(expert_id);
    }

    /// Check whether the given expert is currently in the hot set.
    pub fn is_hot(&self, expert_id: usize) -> bool {
        self.hot_set.iter().any(|&(id, _)| id == expert_id)
    }

    /// Return a reference to the current cache statistics.
    pub fn stats(&self) -> &ExpertCacheStats {
        &self.stats
    }

    /// Reset all statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = ExpertCacheStats::default();
    }

    /// Return the current number of experts in the hot set.
    pub fn hot_count(&self) -> usize {
        self.hot_set.len()
    }

    /// Return the configured maximum hot-set size.
    pub fn max_hot(&self) -> usize {
        self.config.max_hot_experts
    }

    /// Get list of currently hot experts.
    ///
    /// Returns the expert IDs currently in the hot set, in no particular order.
    /// Useful for prefetch decisions and cache diagnostics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvllm::bitnet::expert_cache::{ExpertCache, ExpertCacheConfig};
    ///
    /// let mut cache = ExpertCache::new(8, ExpertCacheConfig::default());
    /// cache.access(2);
    /// cache.access(5);
    ///
    /// let hot = cache.hot_experts();
    /// assert!(hot.contains(&2));
    /// assert!(hot.contains(&5));
    /// ```
    pub fn hot_experts(&self) -> Vec<usize> {
        self.hot_set.iter().map(|&(id, _)| id).collect()
    }

    /// Suggest eviction with affinity awareness.
    ///
    /// Combines the base eviction score (from LRU/LFU/Adaptive policy) with
    /// affinity scores to make better eviction decisions. Experts with high
    /// affinity are less likely to be evicted even if they have low frequency
    /// or old access times.
    ///
    /// # Algorithm
    ///
    /// For each hot expert, compute a combined score:
    /// ```text
    /// eviction_score = (1 - affinity_weight) * base_score + affinity_weight * (1 - affinity)
    /// ```
    ///
    /// Where:
    /// - `base_score` is normalized LRU/LFU score (0=least likely to evict, 1=most likely)
    /// - `affinity` is the expert's affinity score from `ExpertAffinity`
    /// - `affinity_weight` controls the influence of affinity (0.0-1.0)
    ///
    /// The expert with the **highest** eviction_score is suggested for eviction.
    ///
    /// # Arguments
    ///
    /// * `affinity` - The expert affinity tracker (from `moe::ExpertAffinity`)
    /// * `affinity_weight` - How much affinity influences eviction (0.0-1.0)
    ///   - 0.0 = pure base policy (LRU/LFU/Adaptive)
    ///   - 1.0 = pure affinity-based (evict lowest affinity)
    ///   - 0.3-0.5 = recommended balance
    ///
    /// # Returns
    ///
    /// `Some(expert_id)` if the hot set is full and an expert should be evicted.
    /// `None` if the hot set is not full.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvllm::bitnet::expert_cache::{ExpertCache, ExpertCacheConfig};
    /// use ruvllm::moe::{ExpertAffinity, AffinityConfig};
    ///
    /// let mut cache = ExpertCache::new(8, ExpertCacheConfig::default());
    /// let mut affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8));
    ///
    /// // Fill the hot set
    /// for i in 0..4 { cache.access(i); }
    ///
    /// // Update affinity - expert 0 has high affinity
    /// for _ in 0..10 { affinity.update(&[0]); }
    ///
    /// // Should NOT suggest expert 0 despite being LRU
    /// let victim = cache.suggest_eviction_with_affinity(&affinity, 0.5);
    /// assert_ne!(victim, Some(0));
    /// ```
    pub fn suggest_eviction_with_affinity(
        &self,
        affinity: &crate::moe::ExpertAffinity,
        affinity_weight: f32,
    ) -> Option<usize> {
        if self.hot_set.len() < self.config.max_hot_experts {
            return None;
        }

        // Clamp weight to valid range
        let weight = affinity_weight.clamp(0.0, 1.0);

        // If weight is 0, just use base policy
        if weight < 1e-6 {
            return self.suggest_eviction();
        }

        // Compute base scores based on policy
        let base_scores = self.compute_base_eviction_scores();

        if base_scores.is_empty() {
            return None;
        }

        // Find expert with highest combined eviction score
        let mut best_victim: Option<usize> = None;
        let mut best_score: f32 = f32::MIN;

        for &(id, _) in &self.hot_set {
            let base_score = base_scores.get(&id).copied().unwrap_or(0.5);
            let expert_affinity = affinity.score(id);

            // Combined score: higher = more likely to evict
            // (1 - affinity) means low affinity -> high eviction likelihood
            let combined = (1.0 - weight) * base_score + weight * (1.0 - expert_affinity);

            if combined > best_score {
                best_score = combined;
                best_victim = Some(id);
            }
        }

        best_victim
    }

    /// Prefetch experts based on affinity predictions.
    ///
    /// Selects the top experts by affinity score that are not already in the
    /// hot set, up to the given budget, and admits them via prefetch.
    ///
    /// # Arguments
    ///
    /// * `affinity` - The expert affinity tracker
    /// * `budget` - Maximum number of experts to prefetch
    ///
    /// # Returns
    ///
    /// Vector of expert IDs that were actually prefetched (may be fewer than
    /// `budget` if the hot set is nearly full or all high-affinity experts
    /// are already hot).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvllm::bitnet::expert_cache::{ExpertCache, ExpertCacheConfig};
    /// use ruvllm::moe::{ExpertAffinity, AffinityConfig};
    ///
    /// let config = ExpertCacheConfig { max_hot_experts: 4, ..Default::default() };
    /// let mut cache = ExpertCache::new(8, config);
    /// let mut affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8));
    ///
    /// // Build up affinity for experts 3 and 5
    /// for _ in 0..5 { affinity.update(&[3, 5]); }
    ///
    /// // Prefetch top 2 by affinity
    /// let prefetched = cache.prefetch_by_affinity(&affinity, 2);
    ///
    /// assert!(prefetched.contains(&3) || prefetched.contains(&5));
    /// assert!(cache.is_hot(3) || cache.is_hot(5));
    /// ```
    pub fn prefetch_by_affinity(
        &mut self,
        affinity: &crate::moe::ExpertAffinity,
        budget: usize,
    ) -> Vec<usize> {
        if budget == 0 {
            return Vec::new();
        }

        // Get top experts by affinity
        let top_experts = affinity.top_k_by_affinity(self.num_experts);

        let mut prefetched = Vec::with_capacity(budget);

        for expert_id in top_experts {
            if prefetched.len() >= budget {
                break;
            }

            // Skip if already hot
            if self.is_hot(expert_id) {
                continue;
            }

            // Skip if hot set is full and we can't make room
            if self.hot_set.len() >= self.config.max_hot_experts {
                // Try to evict using affinity-aware policy
                if let Some(victim) = self.suggest_eviction_with_affinity(affinity, 0.5) {
                    self.evict(victim);
                } else {
                    break; // Can't make room
                }
            }

            // Admit via prefetch
            self.prefetch_admit(expert_id);
            prefetched.push(expert_id);
        }

        prefetched
    }

    // --- Private helpers ---

    /// Compute normalized base eviction scores for all hot experts.
    ///
    /// Returns a map of expert_id -> score where:
    /// - 0.0 = least likely to evict
    /// - 1.0 = most likely to evict
    fn compute_base_eviction_scores(&self) -> HashMap<usize, f32> {
        let mut scores = HashMap::new();

        if self.hot_set.is_empty() {
            return scores;
        }

        match self.config.eviction_policy {
            EvictionPolicy::Lru => {
                // LRU: older timestamp = higher eviction score
                let timestamps: Vec<u64> = self.hot_set.iter().map(|&(_, ts)| ts).collect();
                let min_ts = timestamps.iter().copied().min().unwrap_or(0);
                let max_ts = timestamps.iter().copied().max().unwrap_or(1);
                let range = (max_ts - min_ts) as f32;

                for &(id, ts) in &self.hot_set {
                    let score = if range > 0.0 {
                        1.0 - ((ts - min_ts) as f32 / range)
                    } else {
                        0.5
                    };
                    scores.insert(id, score);
                }
            }
            EvictionPolicy::Lfu => {
                // LFU: lower frequency = higher eviction score
                let freqs: Vec<usize> = self
                    .hot_set
                    .iter()
                    .map(|&(id, _)| self.frequency.get(id).copied().unwrap_or(0))
                    .collect();
                let min_freq = freqs.iter().copied().min().unwrap_or(0);
                let max_freq = freqs.iter().copied().max().unwrap_or(1);
                let range = (max_freq - min_freq) as f32;

                for &(id, _) in &self.hot_set {
                    let freq = self.frequency.get(id).copied().unwrap_or(0);
                    let score = if range > 0.0 {
                        1.0 - ((freq - min_freq) as f32 / range)
                    } else {
                        0.5
                    };
                    scores.insert(id, score);
                }
            }
            EvictionPolicy::Adaptive => {
                // Adaptive: check skewness and use appropriate policy
                let freqs: Vec<usize> = self
                    .hot_set
                    .iter()
                    .map(|&(id, _)| self.frequency.get(id).copied().unwrap_or(0))
                    .collect();
                let max_freq = freqs.iter().copied().max().unwrap_or(0);
                let min_freq = freqs.iter().copied().min().unwrap_or(0);

                if min_freq > 0 && max_freq >= 3 * min_freq {
                    // Skewed: use LFU scores
                    let range = (max_freq - min_freq) as f32;
                    for &(id, _) in &self.hot_set {
                        let freq = self.frequency.get(id).copied().unwrap_or(0);
                        let score = if range > 0.0 {
                            1.0 - ((freq - min_freq) as f32 / range)
                        } else {
                            0.5
                        };
                        scores.insert(id, score);
                    }
                } else {
                    // Not skewed: use LRU scores
                    let timestamps: Vec<u64> = self.hot_set.iter().map(|&(_, ts)| ts).collect();
                    let min_ts = timestamps.iter().copied().min().unwrap_or(0);
                    let max_ts = timestamps.iter().copied().max().unwrap_or(1);
                    let range = (max_ts - min_ts) as f32;

                    for &(id, ts) in &self.hot_set {
                        let score = if range > 0.0 {
                            1.0 - ((ts - min_ts) as f32 / range)
                        } else {
                            0.5
                        };
                        scores.insert(id, score);
                    }
                }
            }
        }

        scores
    }

    /// LRU eviction: pick the expert with the smallest (oldest) timestamp.
    fn suggest_lru_eviction(&self) -> Option<usize> {
        self.hot_set
            .iter()
            .min_by_key(|&&(_, ts)| ts)
            .map(|&(id, _)| id)
    }

    /// LFU eviction: pick the hot expert with the lowest total access frequency.
    fn suggest_lfu_eviction(&self) -> Option<usize> {
        self.hot_set
            .iter()
            .min_by_key(|&&(id, _)| self.frequency.get(id).copied().unwrap_or(0))
            .map(|&(id, _)| id)
    }

    /// Adaptive eviction: use LFU when frequency distribution is skewed,
    /// otherwise fall back to LRU.
    fn suggest_adaptive_eviction(&self) -> Option<usize> {
        if self.hot_set.is_empty() {
            return None;
        }

        let freqs: Vec<usize> = self
            .hot_set
            .iter()
            .map(|&(id, _)| self.frequency.get(id).copied().unwrap_or(0))
            .collect();

        let max_freq = freqs.iter().copied().max().unwrap_or(0);
        let min_freq = freqs.iter().copied().min().unwrap_or(0);

        // If the most-accessed expert has >= 3x the accesses of the least-accessed,
        // the distribution is skewed enough that frequency is a better signal.
        if min_freq > 0 && max_freq >= 3 * min_freq {
            self.suggest_lfu_eviction()
        } else {
            self.suggest_lru_eviction()
        }
    }
}

// ============================================================================
// MoE Batch Scheduler
// ============================================================================

/// A batch of tokens routed to the same expert, produced by `MoeBatchScheduler`.
#[derive(Debug, Clone)]
pub struct ExpertBatch {
    /// The expert ID that all tokens in this batch are routed to.
    pub expert_id: usize,
    /// Indices into the original token batch identifying which tokens are included.
    pub token_indices: Vec<usize>,
    /// Per-token router weights for this expert (same order as `token_indices`).
    pub weights: Vec<f32>,
}

/// Reorders expert execution across a token batch to maximize cache reuse.
///
/// Without batching, each token processes its top-K experts independently:
/// ```text
/// Token 0: Expert 2, Expert 5
/// Token 1: Expert 5, Expert 3
/// Token 2: Expert 2, Expert 7
/// ```
///
/// This causes expert weights to be loaded, evicted, and reloaded. The batch
/// scheduler groups tokens by expert:
/// ```text
/// Expert 2: Token 0 (w=0.6), Token 2 (w=0.7)
/// Expert 3: Token 1 (w=0.3)
/// Expert 5: Token 0 (w=0.4), Token 1 (w=0.7)
/// Expert 7: Token 2 (w=0.3)
/// ```
///
/// Now each expert's weights are loaded once and applied to all relevant tokens
/// before moving on.
pub struct MoeBatchScheduler;

impl MoeBatchScheduler {
    /// Schedule a batch of routing decisions into expert-grouped batches.
    ///
    /// # Arguments
    ///
    /// * `routing_decisions` - For each token in the batch, a tuple of
    ///   `(token_index, Vec<(expert_id, router_weight)>)` describing which
    ///   experts were selected and their normalized weights.
    ///
    /// # Returns
    ///
    /// A vector of `ExpertBatch` structs, one per unique expert referenced in
    /// the routing decisions, sorted by expert_id for deterministic ordering.
    pub fn schedule(routing_decisions: &[(usize, Vec<(usize, f32)>)]) -> Vec<ExpertBatch> {
        // Collect all (expert_id -> Vec<(token_idx, weight)>)
        let mut expert_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for &(token_idx, ref experts) in routing_decisions {
            for &(expert_id, weight) in experts {
                expert_map
                    .entry(expert_id)
                    .or_default()
                    .push((token_idx, weight));
            }
        }

        // Build sorted batches
        let mut batches: Vec<ExpertBatch> = expert_map
            .into_iter()
            .map(|(expert_id, entries)| {
                let (token_indices, weights): (Vec<usize>, Vec<f32>) = entries.into_iter().unzip();
                ExpertBatch {
                    expert_id,
                    token_indices,
                    weights,
                }
            })
            .collect();

        // Sort by expert_id for deterministic execution order
        batches.sort_by_key(|b| b.expert_id);
        batches
    }
}

// ============================================================================
// Prefetcher Trait
// ============================================================================

/// Abstraction for platform-specific memory prefetch instructions.
///
/// Implementations can issue hardware prefetch hints (e.g., x86 `_mm_prefetch`
/// with `_MM_HINT_T0`, aarch64 `__pld`) to pull expert weight data into cache
/// ahead of the GEMV kernel touching it.
///
/// The trait is object-safe to allow runtime dispatch between platform backends.
pub trait Prefetcher: Send + Sync {
    /// Issue a prefetch hint for a region of memory.
    ///
    /// # Arguments
    ///
    /// * `data` - The backing byte slice (e.g., `TernaryTensor::packed_data`).
    /// * `offset` - Byte offset into `data` where the prefetch region starts.
    /// * `len` - Number of bytes to prefetch. Implementations may round up to
    ///   cache-line granularity.
    ///
    /// # Safety
    ///
    /// This is a hint only. Implementations must not cause faults if `offset + len`
    /// exceeds `data.len()`.
    fn prefetch(&self, data: &[u8], offset: usize, len: usize);
}

/// No-op prefetcher used when platform-specific intrinsics are not available.
///
/// All calls are silent no-ops. This is the default prefetcher for portable builds.
pub struct NullPrefetcher;

impl Prefetcher for NullPrefetcher {
    #[inline(always)]
    fn prefetch(&self, _data: &[u8], _offset: usize, _len: usize) {
        // Intentionally empty. On x86_64, this would be:
        //   unsafe { std::arch::x86_64::_mm_prefetch(ptr, _MM_HINT_T0); }
        // On aarch64:
        //   unsafe { std::arch::aarch64::__pld(ptr); }
    }
}

// ============================================================================
// Memory Layout Helpers
// ============================================================================

/// Cache line size in bytes (standard for x86_64 and most aarch64 cores).
const CACHE_LINE_BYTES: usize = 64;

/// Round a pointer-sized address up to the nearest 64-byte cache-line boundary.
///
/// This is useful for ensuring that expert weight buffers start on cache-line
/// boundaries to avoid false sharing and partial-line fetches.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::bitnet::expert_cache::align_to_cache_line;
///
/// assert_eq!(align_to_cache_line(0), 0);
/// assert_eq!(align_to_cache_line(1), 64);
/// assert_eq!(align_to_cache_line(64), 64);
/// assert_eq!(align_to_cache_line(65), 128);
/// ```
#[inline]
pub fn align_to_cache_line(ptr: usize) -> usize {
    (ptr + CACHE_LINE_BYTES - 1) & !(CACHE_LINE_BYTES - 1)
}

/// Compute the memory footprint of a single expert's packed ternary data.
///
/// An expert projection (e.g., gate_proj) with shape `(rows, cols)` and the
/// given `block_size` occupies:
/// - Packed data: `ceil(rows * cols / 4)` bytes (2 bits per weight, 4 per byte)
/// - Scales: `ceil(rows * cols / block_size) * 4` bytes (one FP32 per block)
///
/// The returned value is the sum, **not** cache-line aligned.
///
/// # Arguments
///
/// * `rows` - Number of output features (e.g., intermediate_size).
/// * `cols` - Number of input features (e.g., hidden_size).
/// * `block_size` - Elements per quantization block (typically 256).
#[inline]
pub fn expert_memory_footprint(rows: usize, cols: usize, block_size: usize) -> usize {
    let total_elements = rows * cols;
    let packed_bytes = (total_elements + 3) / 4;
    let num_blocks = (total_elements + block_size - 1) / block_size;
    let scale_bytes = num_blocks * 4; // FP32 = 4 bytes
    packed_bytes + scale_bytes
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helper: create a default cache with given expert count and max hot
    // ---------------------------------------------------------------

    fn make_cache(num_experts: usize, max_hot: usize, policy: EvictionPolicy) -> ExpertCache {
        let config = ExpertCacheConfig {
            max_hot_experts: max_hot,
            prefetch_threshold: 0.1,
            eviction_policy: policy,
        };
        ExpertCache::new(num_experts, config)
    }

    // ---------------------------------------------------------------
    // 1. LRU eviction order is correct
    // ---------------------------------------------------------------

    #[test]
    fn test_lru_eviction_order() {
        let mut cache = make_cache(8, 3, EvictionPolicy::Lru);

        // Fill the hot set: 0, 1, 2
        cache.access(0);
        cache.access(1);
        cache.access(2);

        // All three should be hot
        assert!(cache.is_hot(0));
        assert!(cache.is_hot(1));
        assert!(cache.is_hot(2));

        // Access expert 0 again to refresh its timestamp
        cache.access(0);

        // Now admit expert 3 -> should evict expert 1 (oldest unrefresfreshed)
        cache.access(3);

        assert!(
            cache.is_hot(0),
            "Expert 0 was refreshed, should still be hot"
        );
        assert!(!cache.is_hot(1), "Expert 1 should have been evicted (LRU)");
        assert!(
            cache.is_hot(2),
            "Expert 2 was accessed after 1, should survive"
        );
        assert!(cache.is_hot(3), "Expert 3 was just admitted");
    }

    // ---------------------------------------------------------------
    // 2. LFU eviction order is correct
    // ---------------------------------------------------------------

    #[test]
    fn test_lfu_eviction_order() {
        let mut cache = make_cache(8, 3, EvictionPolicy::Lfu);

        // Expert 0: accessed 3 times
        cache.access(0);
        cache.access(0);
        cache.access(0);

        // Expert 1: accessed 1 time
        cache.access(1);

        // Expert 2: accessed 2 times
        cache.access(2);
        cache.access(2);

        // Hot set: {0, 1, 2}, frequencies: 0->3, 1->1, 2->2
        assert!(cache.is_hot(0));
        assert!(cache.is_hot(1));
        assert!(cache.is_hot(2));

        // Admit expert 3 -> should evict expert 1 (frequency=1, lowest)
        cache.access(3);

        assert!(cache.is_hot(0), "Expert 0 (freq=3) should survive");
        assert!(
            !cache.is_hot(1),
            "Expert 1 (freq=1) should be evicted by LFU"
        );
        assert!(cache.is_hot(2), "Expert 2 (freq=2) should survive");
        assert!(cache.is_hot(3), "Expert 3 was just admitted");
    }

    // ---------------------------------------------------------------
    // 3. Hot set respects max_hot_experts limit
    // ---------------------------------------------------------------

    #[test]
    fn test_hot_set_respects_limit() {
        let mut cache = make_cache(16, 4, EvictionPolicy::Lru);

        // Access more experts than max_hot
        for i in 0..10 {
            cache.access(i);
        }

        // Should never exceed 4 hot experts
        assert!(
            cache.hot_count() <= 4,
            "Hot count {} exceeds max of 4",
            cache.hot_count()
        );
        assert_eq!(cache.hot_count(), 4);
    }

    // ---------------------------------------------------------------
    // 4. Access returns hit=true for hot experts
    // ---------------------------------------------------------------

    #[test]
    fn test_access_returns_hit_for_hot() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        // First access is always a miss
        assert!(!cache.access(3));

        // Second access should be a hit
        assert!(cache.access(3));
        assert!(cache.access(3));
    }

    // ---------------------------------------------------------------
    // 5. Access returns hit=false for cold experts
    // ---------------------------------------------------------------

    #[test]
    fn test_access_returns_miss_for_cold() {
        let mut cache = make_cache(8, 2, EvictionPolicy::Lru);

        // Fill: 0, 1
        cache.access(0);
        cache.access(1);

        // Access 2 -> evicts 0, returns false (miss)
        assert!(!cache.access(2));
        // Access 3 -> evicts 1, returns false (miss)
        assert!(!cache.access(3));

        // Now 0 and 1 are cold, accessing them is a miss
        assert!(!cache.access(0));
    }

    // ---------------------------------------------------------------
    // 6. Hit rate calculation is correct
    // ---------------------------------------------------------------

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        // No accesses -> 0.0
        assert_eq!(cache.stats().hit_rate(), 0.0);

        // 1 miss (first access to expert 0)
        cache.access(0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hit_rate(), 0.0);

        // 1 hit (second access to expert 0)
        cache.access(0);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate() - 0.5).abs() < 1e-6);

        // 2 more hits
        cache.access(0);
        cache.access(0);
        // Total: 3 hits, 1 miss => 3/4 = 0.75
        assert!((cache.stats().hit_rate() - 0.75).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // 7. Prefetch threshold works
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_threshold() {
        let config = ExpertCacheConfig {
            max_hot_experts: 4,
            prefetch_threshold: 0.15,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(8, config);

        // Expert 0 is not hot -> should prefetch if weight > 0.15
        assert!(cache.should_prefetch(0, 0.2));
        assert!(cache.should_prefetch(0, 0.16));
        assert!(!cache.should_prefetch(0, 0.15)); // at threshold, not above
        assert!(!cache.should_prefetch(0, 0.1));
        assert!(!cache.should_prefetch(0, 0.0));

        // Make expert 0 hot -> should NOT prefetch (already hot)
        cache.access(0);
        assert!(!cache.should_prefetch(0, 0.5));
    }

    // ---------------------------------------------------------------
    // 8. Batch scheduler groups tokens by expert
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_scheduler_groups_by_expert() {
        let routing = vec![
            (0, vec![(2, 0.6), (5, 0.4)]),
            (1, vec![(5, 0.7), (3, 0.3)]),
            (2, vec![(2, 0.7), (7, 0.3)]),
        ];

        let batches = MoeBatchScheduler::schedule(&routing);

        // Should have 4 unique experts: 2, 3, 5, 7
        assert_eq!(batches.len(), 4);

        // Batches should be sorted by expert_id
        let expert_ids: Vec<usize> = batches.iter().map(|b| b.expert_id).collect();
        assert_eq!(expert_ids, vec![2, 3, 5, 7]);

        // Expert 2: tokens 0 and 2
        let batch_2 = &batches[0];
        assert_eq!(batch_2.expert_id, 2);
        assert_eq!(batch_2.token_indices, vec![0, 2]);
        assert_eq!(batch_2.weights, vec![0.6, 0.7]);

        // Expert 3: token 1 only
        let batch_3 = &batches[1];
        assert_eq!(batch_3.expert_id, 3);
        assert_eq!(batch_3.token_indices, vec![1]);
        assert_eq!(batch_3.weights, vec![0.3]);

        // Expert 5: tokens 0 and 1
        let batch_5 = &batches[2];
        assert_eq!(batch_5.expert_id, 5);
        assert_eq!(batch_5.token_indices, vec![0, 1]);
        assert_eq!(batch_5.weights, vec![0.4, 0.7]);

        // Expert 7: token 2 only
        let batch_7 = &batches[3];
        assert_eq!(batch_7.expert_id, 7);
        assert_eq!(batch_7.token_indices, vec![2]);
        assert_eq!(batch_7.weights, vec![0.3]);
    }

    // ---------------------------------------------------------------
    // 9. Batch scheduler handles single-token case
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_scheduler_single_token() {
        let routing = vec![(0, vec![(4, 0.65), (1, 0.35)])];

        let batches = MoeBatchScheduler::schedule(&routing);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].expert_id, 1);
        assert_eq!(batches[0].token_indices, vec![0]);
        assert_eq!(batches[0].weights, vec![0.35]);

        assert_eq!(batches[1].expert_id, 4);
        assert_eq!(batches[1].token_indices, vec![0]);
        assert_eq!(batches[1].weights, vec![0.65]);
    }

    // ---------------------------------------------------------------
    // 10. Cache stats accumulate correctly
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_stats_accumulate() {
        let mut cache = make_cache(8, 2, EvictionPolicy::Lru);

        // Misses: 0, 1
        cache.access(0); // miss
        cache.access(1); // miss
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().evictions, 0);

        // Hit: 0
        cache.access(0); // hit
        assert_eq!(cache.stats().hits, 1);

        // Miss + eviction: 2 evicts 1 (LRU)
        cache.access(2); // miss, evicts 1
        assert_eq!(cache.stats().misses, 3);
        assert_eq!(cache.stats().evictions, 1);

        // Hit: 0 (still hot)
        cache.access(0); // hit
        assert_eq!(cache.stats().hits, 2);

        // Reset
        cache.reset_stats();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().prefetch_hits, 0);
    }

    // ---------------------------------------------------------------
    // 11. Eviction happens when hot set is full
    // ---------------------------------------------------------------

    #[test]
    fn test_eviction_when_full() {
        let mut cache = make_cache(8, 3, EvictionPolicy::Lru);

        cache.access(0);
        cache.access(1);
        cache.access(2);
        assert_eq!(cache.hot_count(), 3);
        assert_eq!(cache.stats().evictions, 0);

        // Admitting a 4th expert must trigger an eviction
        cache.access(3);
        assert_eq!(cache.hot_count(), 3);
        assert_eq!(cache.stats().evictions, 1);
        assert!(!cache.is_hot(0), "Expert 0 (oldest) should be evicted");
        assert!(cache.is_hot(3));
    }

    // ---------------------------------------------------------------
    // 12. Memory footprint calculation is correct
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_footprint_calculation() {
        // 256 x 256 tensor, block_size = 256
        // total = 65536 elements
        // packed = ceil(65536/4) = 16384 bytes
        // blocks = ceil(65536/256) = 256
        // scales = 256 * 4 = 1024 bytes
        // total = 16384 + 1024 = 17408
        let footprint = expert_memory_footprint(256, 256, 256);
        assert_eq!(footprint, 17408);

        // 1 x 4 tensor, block_size = 256
        // total = 4 elements
        // packed = ceil(4/4) = 1 byte
        // blocks = ceil(4/256) = 1
        // scales = 1 * 4 = 4 bytes
        // total = 5
        let footprint_small = expert_memory_footprint(1, 4, 256);
        assert_eq!(footprint_small, 5);

        // 11008 x 4096 tensor (realistic gate_proj), block_size = 256
        let rows = 11008usize;
        let cols = 4096usize;
        let total = rows * cols; // 45088768
        let packed = (total + 3) / 4; // 11272192
        let blocks = (total + 255) / 256; // 176128
        let scales_bytes = blocks * 4; // 704512
        let expected = packed + scales_bytes; // 11976704
        assert_eq!(expert_memory_footprint(rows, cols, 256), expected);
    }

    // ---------------------------------------------------------------
    // 13. align_to_cache_line works correctly
    // ---------------------------------------------------------------

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(63), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);
        assert_eq!(align_to_cache_line(128), 128);
        assert_eq!(align_to_cache_line(129), 192);
    }

    // ---------------------------------------------------------------
    // 14. NullPrefetcher does not panic
    // ---------------------------------------------------------------

    #[test]
    fn test_null_prefetcher_noop() {
        let prefetcher = NullPrefetcher;
        let data = vec![0u8; 1024];

        // Should not panic even with out-of-range offset
        prefetcher.prefetch(&data, 0, 64);
        prefetcher.prefetch(&data, 512, 256);
        prefetcher.prefetch(&data, 2000, 100); // offset > data.len(), still no-op
        prefetcher.prefetch(&[], 0, 0);
    }

    // ---------------------------------------------------------------
    // 15. Adaptive eviction switches between LRU and LFU
    // ---------------------------------------------------------------

    #[test]
    fn test_adaptive_eviction_policy() {
        let mut cache = make_cache(8, 3, EvictionPolicy::Adaptive);

        // Create skewed frequency distribution:
        // Expert 0: 9 accesses, Expert 1: 3 accesses, Expert 2: 1 access
        for _ in 0..9 {
            cache.access(0);
        }
        for _ in 0..3 {
            cache.access(1);
        }
        cache.access(2);

        // Frequencies: 0->9, 1->3, 2->1
        // max_freq(9) >= 3 * min_freq(1) -> skewed -> use LFU
        // LFU evicts expert 2 (frequency=1)
        cache.access(3);

        assert!(
            cache.is_hot(0),
            "Expert 0 (freq=9) should survive adaptive LFU"
        );
        assert!(
            cache.is_hot(1),
            "Expert 1 (freq=3) should survive adaptive LFU"
        );
        assert!(
            !cache.is_hot(2),
            "Expert 2 (freq=1) should be evicted by adaptive LFU"
        );
        assert!(cache.is_hot(3), "Expert 3 was just admitted");
    }

    // ---------------------------------------------------------------
    // 16. Prefetch admit tracks prefetch hits
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_admit_tracks_hits() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        // Prefetch-admit expert 5
        cache.prefetch_admit(5);
        assert!(cache.is_hot(5));
        assert_eq!(cache.stats().prefetch_hits, 0);

        // Access expert 5 -> should count as a prefetch hit
        let hit = cache.access(5);
        assert!(hit, "Expert 5 is in hot set via prefetch");
        assert_eq!(cache.stats().prefetch_hits, 1);

        // Second access should not count as prefetch hit again
        cache.access(5);
        assert_eq!(cache.stats().prefetch_hits, 1);
    }

    // ---------------------------------------------------------------
    // 17. Batch scheduler handles empty input
    // ---------------------------------------------------------------

    #[test]
    fn test_batch_scheduler_empty() {
        let routing: Vec<(usize, Vec<(usize, f32)>)> = vec![];
        let batches = MoeBatchScheduler::schedule(&routing);
        assert!(batches.is_empty());
    }

    // ---------------------------------------------------------------
    // 18. ExpertCacheConfig default values
    // ---------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let config = ExpertCacheConfig::default();
        assert_eq!(config.max_hot_experts, 4);
        assert!((config.prefetch_threshold - 0.1).abs() < 1e-6);
        assert_eq!(config.eviction_policy, EvictionPolicy::Lru);
    }

    // ---------------------------------------------------------------
    // 19. suggest_eviction returns None when not full
    // ---------------------------------------------------------------

    #[test]
    fn test_suggest_eviction_none_when_not_full() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        assert!(cache.suggest_eviction().is_none());

        cache.access(0);
        assert!(cache.suggest_eviction().is_none());

        cache.access(1);
        cache.access(2);
        assert!(cache.suggest_eviction().is_none());

        // Fill to capacity
        cache.access(3);
        assert!(cache.suggest_eviction().is_some());
    }

    // ---------------------------------------------------------------
    // 20. Admit is idempotent for already-hot experts
    // ---------------------------------------------------------------

    #[test]
    fn test_admit_idempotent() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        cache.admit(0);
        cache.admit(1);
        assert_eq!(cache.hot_count(), 2);

        // Re-admitting should not duplicate
        cache.admit(0);
        cache.admit(1);
        assert_eq!(cache.hot_count(), 2);
    }

    // ---------------------------------------------------------------
    // 21. hot_experts returns current hot set
    // ---------------------------------------------------------------

    #[test]
    fn test_hot_experts_list() {
        let mut cache = make_cache(8, 4, EvictionPolicy::Lru);

        cache.access(2);
        cache.access(5);
        cache.access(7);

        let hot = cache.hot_experts();

        assert_eq!(hot.len(), 3);
        assert!(hot.contains(&2));
        assert!(hot.contains(&5));
        assert!(hot.contains(&7));
        assert!(!hot.contains(&0));
    }

    // ---------------------------------------------------------------
    // 22. Eviction with affinity prefers low affinity experts
    // ---------------------------------------------------------------

    #[test]
    fn test_eviction_with_affinity_prefers_low_affinity() {
        use crate::moe::{AffinityConfig, ExpertAffinity};

        let mut cache = make_cache(8, 3, EvictionPolicy::Lru);
        // Use small activation_boost to create meaningful differences
        let mut affinity = ExpertAffinity::new(
            AffinityConfig::with_num_experts(8)
                .with_decay(1.0)
                .with_activation_boost(0.1),
        );

        // Fill cache with experts 0, 1, 2
        cache.access(0);
        cache.access(1);
        cache.access(2);

        // Expert 0 has high affinity (many activations): 10 * 0.1 = 1.0 (clamped)
        for _ in 0..10 {
            affinity.update(&[0]);
        }

        // Expert 2 has low affinity (few activations): 1 * 0.1 = 0.1
        affinity.update(&[2]);

        // Expert 1 has medium affinity: 5 * 0.1 = 0.5
        for _ in 0..5 {
            affinity.update(&[1]);
        }

        // With pure affinity_weight=1.0, should suggest evicting expert 2 (lowest affinity)
        let victim = cache.suggest_eviction_with_affinity(&affinity, 1.0);

        // Expert 2 should be evicted (lowest affinity=0.1)
        assert_eq!(victim, Some(2), "Should evict lowest affinity expert");
    }

    // ---------------------------------------------------------------
    // 23. Prefetch by affinity respects budget
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_by_affinity_respects_budget() {
        use crate::moe::{AffinityConfig, ExpertAffinity};

        let config = ExpertCacheConfig {
            max_hot_experts: 6,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(8, config);
        let mut affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8).with_decay(1.0));

        // Build affinity for experts 3, 5, 7
        for _ in 0..5 {
            affinity.update(&[3, 5, 7]);
        }

        // Prefetch with budget of 2
        let prefetched = cache.prefetch_by_affinity(&affinity, 2);

        // Should prefetch at most 2 experts
        assert!(prefetched.len() <= 2, "Should respect budget");
        assert!(
            prefetched.len() >= 1,
            "Should prefetch at least 1 high-affinity expert"
        );

        // All prefetched should now be hot
        for &id in &prefetched {
            assert!(cache.is_hot(id), "Prefetched expert should be hot");
        }
    }

    // ---------------------------------------------------------------
    // 24. Prefetch skips already hot experts
    // ---------------------------------------------------------------

    #[test]
    fn test_prefetch_skips_already_hot() {
        use crate::moe::{AffinityConfig, ExpertAffinity};

        let config = ExpertCacheConfig {
            max_hot_experts: 4,
            prefetch_threshold: 0.1,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = ExpertCache::new(8, config);
        let mut affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8).with_decay(1.0));

        // Make expert 3 hot via access
        cache.access(3);

        // Build highest affinity for expert 3
        for _ in 0..10 {
            affinity.update(&[3]);
        }

        // Build lower affinity for expert 5
        for _ in 0..5 {
            affinity.update(&[5]);
        }

        // Prefetch with budget of 2
        let prefetched = cache.prefetch_by_affinity(&affinity, 2);

        // Expert 3 should NOT be in prefetched (already hot)
        assert!(
            !prefetched.contains(&3),
            "Should not prefetch already-hot expert"
        );

        // Expert 5 should be prefetched
        assert!(
            prefetched.contains(&5),
            "Should prefetch next highest affinity expert"
        );
    }

    // ---------------------------------------------------------------
    // 25. Affinity weighted eviction blends scores correctly
    // ---------------------------------------------------------------

    #[test]
    fn test_affinity_weighted_eviction() {
        use crate::moe::{AffinityConfig, ExpertAffinity};

        let mut cache = make_cache(8, 3, EvictionPolicy::Lru);
        let mut affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8).with_decay(1.0));

        // Fill cache: 0, 1, 2 in that order (LRU order: 0 is oldest)
        cache.access(0);
        cache.access(1);
        cache.access(2);

        // Give expert 0 very high affinity
        for _ in 0..20 {
            affinity.update(&[0]);
        }

        // Expert 1 and 2 have zero affinity

        // With weight=0.0 (pure LRU), should evict expert 0 (oldest)
        let victim_lru = cache.suggest_eviction_with_affinity(&affinity, 0.0);
        assert_eq!(victim_lru, Some(0), "Weight 0 should use pure LRU");

        // With weight=1.0 (pure affinity), should evict expert 1 or 2 (lowest affinity)
        let victim_affinity = cache.suggest_eviction_with_affinity(&affinity, 1.0);
        assert!(
            victim_affinity == Some(1) || victim_affinity == Some(2),
            "Weight 1.0 should evict lowest affinity"
        );

        // With weight=0.5 (balanced), expert 0's high affinity should protect it
        let victim_balanced = cache.suggest_eviction_with_affinity(&affinity, 0.5);
        assert_ne!(
            victim_balanced,
            Some(0),
            "Balanced weight should protect high-affinity expert"
        );
    }

    // ---------------------------------------------------------------
    // 26. Zero affinity weight falls back to base policy
    // ---------------------------------------------------------------

    #[test]
    fn test_zero_affinity_weight_fallback() {
        use crate::moe::{AffinityConfig, ExpertAffinity};

        let mut cache = make_cache(8, 3, EvictionPolicy::Lfu);
        let affinity = ExpertAffinity::new(AffinityConfig::with_num_experts(8));

        // Expert 0: accessed 1 time (lowest freq)
        cache.access(0);

        // Expert 1: accessed 3 times
        cache.access(1);
        cache.access(1);
        cache.access(1);

        // Expert 2: accessed 2 times
        cache.access(2);
        cache.access(2);

        // With weight=0, should behave exactly like base policy (LFU)
        let victim_base = cache.suggest_eviction();
        let victim_zero_weight = cache.suggest_eviction_with_affinity(&affinity, 0.0);

        assert_eq!(
            victim_base, victim_zero_weight,
            "Zero weight should match base policy"
        );
        assert_eq!(victim_base, Some(0), "LFU should evict lowest frequency");
    }
}
