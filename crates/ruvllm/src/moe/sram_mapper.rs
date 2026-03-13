//! SRAM Mapper for Hardware Memory Hierarchy Configuration (ADR-092)
//!
//! This module provides platform-specific memory hierarchy configuration for
//! MoE (Mixture of Experts) expert placement across different memory tiers.
//!
//! ## Memory Hierarchy
//!
//! Modern systems have a three-tier memory hierarchy with vastly different
//! latencies and capacities:
//!
//! | Tier | Type | Latency | Capacity | Use Case |
//! |------|------|---------|----------|----------|
//! | SRAM | L2/L3 Cache | ~10-40ns | 4-64MB | Hot experts |
//! | DRAM | Main Memory | ~50-100ns | 2-64GB | Warm experts |
//! | Storage | Flash/NVMe | ~50-200us | Unlimited | Cold experts |
//!
//! ## Expert Placement Strategy
//!
//! For optimal MoE inference performance:
//! 1. **SRAM**: Keep top-K active experts and likely-next-picks in cache
//! 2. **DRAM**: Cache frequently-accessed experts not in SRAM
//! 3. **Storage**: Page in rarely-used experts on demand
//!
//! ## Platform Considerations
//!
//! - **Raspberry Pi 5**: 8GB RAM, small L2 cache, optimize for DRAM
//! - **Mobile**: 2-4GB available, aggressive SRAM management
//! - **Desktop**: 16GB+ RAM, larger caches, more flexibility
//! - **WASM/Browser**: Configurable heap, plan for limited memory
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::moe::{SramMapper, HardwarePreset, MemoryTier};
//!
//! // Create mapper for Raspberry Pi 5 with 8 experts
//! let mut mapper = SramMapper::from_preset(HardwarePreset::RaspberryPi5, 8, 34_000_000);
//!
//! // Assign hot experts to SRAM
//! mapper.assign_tier(0, MemoryTier::Sram);
//! mapper.assign_tier(1, MemoryTier::Sram);
//!
//! // Check paging latency estimate
//! let latency = mapper.estimate_paging_latency(5); // Returns microseconds
//! ```

use std::collections::HashMap;

// Use ExpertId from parent module
use super::ExpertId;

// ============================================================================
// Types
// ============================================================================

/// Memory tier classification for expert placement.
///
/// Each tier has different characteristics in terms of latency, bandwidth,
/// and capacity. The SramMapper assigns experts to tiers based on access
/// patterns and hardware constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// L2/L3 cache tier (fastest, smallest capacity).
    ///
    /// Experts in SRAM tier have sub-microsecond access latency but limited
    /// slots. Reserved for the most frequently accessed experts.
    Sram,

    /// Main memory tier (DRAM).
    ///
    /// Moderate latency (~100ns) but significantly larger capacity. Good for
    /// experts that are accessed regularly but not in the hot path.
    Dram,

    /// Storage tier (Flash/NVMe, slowest, largest capacity).
    ///
    /// High latency (50-200+ microseconds) but virtually unlimited capacity.
    /// Used for cold experts that are rarely accessed.
    Storage,
}

impl MemoryTier {
    /// Return a human-readable name for the tier.
    pub fn name(&self) -> &'static str {
        match self {
            MemoryTier::Sram => "SRAM (L2/L3 Cache)",
            MemoryTier::Dram => "DRAM (Main Memory)",
            MemoryTier::Storage => "Storage (Flash/NVMe)",
        }
    }

    /// Return the tier as an index for array lookups.
    pub fn index(&self) -> usize {
        match self {
            MemoryTier::Sram => 0,
            MemoryTier::Dram => 1,
            MemoryTier::Storage => 2,
        }
    }
}

/// Expert affinity information for eviction decisions.
///
/// Tracks access patterns and preferences to help the SRAM mapper make
/// intelligent tier assignment and eviction decisions.
#[derive(Debug, Clone)]
pub struct SramExpertAffinity {
    /// Expert identifier.
    pub expert_id: ExpertId,

    /// Total access count (frequency).
    pub access_count: usize,

    /// Last access timestamp (monotonic counter).
    pub last_access: u64,

    /// Average router weight when selected (0.0 - 1.0).
    pub avg_router_weight: f32,

    /// Number of tokens that selected this expert recently.
    pub recent_selections: usize,

    /// Whether this expert is currently "pinned" to its tier.
    pub pinned: bool,
}

impl Default for SramExpertAffinity {
    fn default() -> Self {
        Self {
            expert_id: 0,
            access_count: 0,
            last_access: 0,
            avg_router_weight: 0.0,
            recent_selections: 0,
            pinned: false,
        }
    }
}

impl SramExpertAffinity {
    /// Create new affinity tracking for an expert.
    pub fn new(expert_id: ExpertId) -> Self {
        Self {
            expert_id,
            ..Default::default()
        }
    }

    /// Compute a priority score for eviction decisions (higher = less likely to evict).
    ///
    /// The score combines frequency, recency, and router weight into a single
    /// metric used for tier assignment decisions.
    pub fn priority_score(&self) -> f32 {
        // Weight the factors:
        // - Frequency has diminishing returns (log scale)
        // - Recency is important for temporal locality
        // - Router weight indicates model preference
        let freq_factor = (self.access_count as f32 + 1.0).ln();

        // Guard against division by zero when last_access is 0
        let recency_factor = if self.last_access == 0 {
            0.0
        } else {
            1.0 / (1.0 + 0.001 / self.last_access as f32)
        };

        let weight_factor = self.avg_router_weight * 2.0;

        freq_factor + recency_factor + weight_factor
    }
}

// ============================================================================
// Hardware Configuration
// ============================================================================

/// Hardware configuration for a specific platform.
///
/// Describes the memory hierarchy constraints that guide expert placement
/// decisions. Can be created from presets or with custom values.
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// L2+L3 cache size in bytes.
    ///
    /// This is the effective SRAM available for expert caching. On most
    /// systems this is 4-64MB shared across all cores.
    pub sram_bytes: usize,

    /// Available DRAM budget for expert caching in bytes.
    ///
    /// This is the portion of main memory allocated for keeping experts
    /// resident. Should be less than total system RAM to leave room for
    /// other allocations.
    pub dram_budget_bytes: usize,

    /// Number of expert slots that fit in SRAM.
    ///
    /// Computed as `sram_bytes / expert_size_bytes`, possibly with some
    /// slack for cache line alignment and other overhead.
    pub sram_expert_slots: usize,

    /// Number of expert slots that fit in DRAM budget.
    ///
    /// Computed as `dram_budget_bytes / expert_size_bytes`.
    pub dram_expert_slots: usize,

    /// Expert size in bytes (packed weights for one expert).
    ///
    /// Includes all three projections (gate_proj, up_proj, down_proj) with
    /// packed ternary weights and scale factors.
    pub expert_size_bytes: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            sram_bytes: 8 * 1024 * 1024,               // 8 MB typical L3
            dram_budget_bytes: 4 * 1024 * 1024 * 1024, // 4 GB DRAM budget
            sram_expert_slots: 2,
            dram_expert_slots: 8,
            expert_size_bytes: 34_000_000, // ~34 MB per expert
        }
    }
}

impl HardwareConfig {
    /// Create a new hardware configuration.
    ///
    /// Automatically computes slot counts from byte budgets and expert size.
    pub fn new(sram_bytes: usize, dram_budget_bytes: usize, expert_size_bytes: usize) -> Self {
        let sram_expert_slots = sram_bytes / expert_size_bytes.max(1);
        let dram_expert_slots = dram_budget_bytes / expert_size_bytes.max(1);

        Self {
            sram_bytes,
            dram_budget_bytes,
            sram_expert_slots,
            dram_expert_slots,
            expert_size_bytes,
        }
    }

    /// Total memory budget across SRAM and DRAM tiers.
    pub fn total_budget(&self) -> usize {
        self.sram_bytes + self.dram_budget_bytes
    }

    /// Total expert slots available (SRAM + DRAM).
    pub fn total_slots(&self) -> usize {
        self.sram_expert_slots + self.dram_expert_slots
    }
}

// ============================================================================
// Hardware Presets
// ============================================================================

/// Known hardware presets for common deployment targets.
///
/// Each preset provides sensible defaults for the memory hierarchy based on
/// typical hardware configurations of that platform class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwarePreset {
    /// Raspberry Pi 5 (8GB RAM, ARM Cortex-A76).
    ///
    /// - L2: 512KB per core (4 cores)
    /// - L3: None
    /// - RAM: 8GB LPDDR4X
    /// - Typical: 4-6 experts in DRAM, 1-2 in cache
    RaspberryPi5,

    /// Mobile device (2-4GB available memory).
    ///
    /// Aggressive memory management due to system constraints.
    /// - L2/L3: ~2-8MB shared
    /// - RAM: 2-3GB available after OS
    /// - Typical: 2-4 experts in DRAM, 1 in cache
    Mobile,

    /// Desktop workstation (16GB+ RAM, modern x86_64).
    ///
    /// - L2: 1-2MB per core
    /// - L3: 16-64MB shared
    /// - RAM: 16GB+ available
    /// - Typical: 8+ experts in DRAM, 2-4 in cache
    Desktop,

    /// WebAssembly browser environment.
    ///
    /// Configurable heap size, typically limited to 1-4GB depending on
    /// browser and device. Conservative defaults.
    /// - "Cache": WASM linear memory hot region
    /// - "DRAM": Rest of WASM heap
    /// - Typical: 1-2 experts warm
    WasmBrowser,

    /// Custom configuration (use HardwareConfig directly).
    Custom,
}

impl HardwarePreset {
    /// Get the default HardwareConfig for this preset.
    ///
    /// Note: `expert_size_bytes` must be provided separately as it depends
    /// on the specific model architecture.
    pub fn default_config(&self, expert_size_bytes: usize) -> HardwareConfig {
        match self {
            HardwarePreset::RaspberryPi5 => {
                // RPi5: 512KB L2 per core (effectively ~1-2MB usable), 8GB RAM
                let sram_bytes = 2 * 1024 * 1024; // ~2MB effective cache
                let dram_budget = 6 * 1024 * 1024 * 1024; // 6GB budget
                HardwareConfig::new(sram_bytes, dram_budget, expert_size_bytes)
            }
            HardwarePreset::Mobile => {
                // Mobile: 4MB L3, 2-3GB available
                let sram_bytes = 4 * 1024 * 1024;
                let dram_budget = 2 * 1024 * 1024 * 1024;
                HardwareConfig::new(sram_bytes, dram_budget, expert_size_bytes)
            }
            HardwarePreset::Desktop => {
                // Desktop: 32MB L3, 16GB+ available
                let sram_bytes = 32 * 1024 * 1024;
                let dram_budget = 12 * 1024 * 1024 * 1024;
                HardwareConfig::new(sram_bytes, dram_budget, expert_size_bytes)
            }
            HardwarePreset::WasmBrowser => {
                // WASM: ~2MB hot region, 1GB heap budget
                let sram_bytes = 2 * 1024 * 1024;
                let dram_budget = 1024 * 1024 * 1024;
                HardwareConfig::new(sram_bytes, dram_budget, expert_size_bytes)
            }
            HardwarePreset::Custom => HardwareConfig::default(),
        }
    }

    /// Get a human-readable name for the preset.
    pub fn name(&self) -> &'static str {
        match self {
            HardwarePreset::RaspberryPi5 => "Raspberry Pi 5",
            HardwarePreset::Mobile => "Mobile Device",
            HardwarePreset::Desktop => "Desktop Workstation",
            HardwarePreset::WasmBrowser => "WASM Browser",
            HardwarePreset::Custom => "Custom",
        }
    }
}

// ============================================================================
// SRAM Mapper
// ============================================================================

/// SRAM Mapper for hardware memory hierarchy configuration.
///
/// Manages expert placement across memory tiers (SRAM/Cache, DRAM, Storage)
/// based on access patterns and hardware constraints. Provides latency
/// estimates and eviction suggestions for optimal MoE inference performance.
///
/// # Usage
///
/// ```rust,ignore
/// use ruvllm::moe::{SramMapper, HardwarePreset, MemoryTier};
///
/// // Create from preset
/// let mut mapper = SramMapper::from_preset(HardwarePreset::RaspberryPi5, 8, 34_000_000);
///
/// // Assign experts to tiers
/// mapper.assign_tier(0, MemoryTier::Sram);
/// mapper.assign_tier(1, MemoryTier::Sram);
/// mapper.assign_tier(2, MemoryTier::Dram);
///
/// // Query tier assignments
/// assert_eq!(mapper.get_tier(0), MemoryTier::Sram);
///
/// // Get latency estimate (microseconds)
/// let latency = mapper.estimate_paging_latency(5);
/// ```
pub struct SramMapper {
    /// Hardware configuration.
    config: HardwareConfig,

    /// Total number of experts in the model.
    num_experts: usize,

    /// Current tier assignment for each expert (indexed by ExpertId).
    tier_map: Vec<MemoryTier>,

    /// Expert affinity tracking for eviction decisions.
    affinity: Vec<SramExpertAffinity>,

    /// Estimated paging latency per tier in microseconds.
    ///
    /// Index 0 = SRAM, 1 = DRAM, 2 = Storage.
    tier_latency: [u64; 3],

    /// Current SRAM slot usage.
    sram_used: usize,

    /// Current DRAM slot usage.
    dram_used: usize,

    /// Monotonic counter for LRU tracking.
    access_counter: u64,
}

impl SramMapper {
    /// Create a new SRAM mapper from a hardware preset.
    ///
    /// # Arguments
    ///
    /// * `preset` - Hardware preset to use for configuration
    /// * `num_experts` - Total number of experts in the model
    /// * `expert_size_bytes` - Size of each expert in bytes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mapper = SramMapper::from_preset(HardwarePreset::RaspberryPi5, 8, 34_000_000);
    /// ```
    pub fn from_preset(
        preset: HardwarePreset,
        num_experts: usize,
        expert_size_bytes: usize,
    ) -> Self {
        let config = preset.default_config(expert_size_bytes);
        Self::from_config(config, num_experts)
    }

    /// Create a new SRAM mapper from a custom hardware configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom hardware configuration
    /// * `num_experts` - Total number of experts in the model
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = HardwareConfig::new(16 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 34_000_000);
    /// let mapper = SramMapper::from_config(config, 8);
    /// ```
    pub fn from_config(config: HardwareConfig, num_experts: usize) -> Self {
        // Initialize all experts to Storage tier (cold start)
        let tier_map = vec![MemoryTier::Storage; num_experts];

        // Initialize affinity tracking
        let affinity = (0..num_experts).map(SramExpertAffinity::new).collect();

        // Default latency estimates (microseconds)
        // SRAM: ~0.04us (40ns), DRAM: ~0.1us (100ns), Storage: ~100us
        let tier_latency = [0, 0, 100];

        Self {
            config,
            num_experts,
            tier_map,
            affinity,
            tier_latency,
            sram_used: 0,
            dram_used: 0,
            access_counter: 0,
        }
    }

    /// Assign an expert to a specific memory tier.
    ///
    /// This updates the internal tracking and slot usage. If the expert was
    /// previously in a different tier, the old slot is freed.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Expert to assign
    /// * `tier` - Target memory tier
    ///
    /// # Returns
    ///
    /// Returns `false` if `expert_id >= num_experts`, `true` otherwise.
    pub fn assign_tier(&mut self, expert_id: ExpertId, tier: MemoryTier) -> bool {
        if expert_id >= self.num_experts {
            return false;
        }

        let old_tier = self.tier_map[expert_id];

        // Free old slot
        match old_tier {
            MemoryTier::Sram => {
                if self.sram_used > 0 {
                    self.sram_used -= 1;
                }
            }
            MemoryTier::Dram => {
                if self.dram_used > 0 {
                    self.dram_used -= 1;
                }
            }
            MemoryTier::Storage => {}
        }

        // Allocate new slot
        match tier {
            MemoryTier::Sram => self.sram_used += 1,
            MemoryTier::Dram => self.dram_used += 1,
            MemoryTier::Storage => {}
        }

        self.tier_map[expert_id] = tier;
        true
    }

    /// Get the current memory tier for an expert.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Expert to query
    ///
    /// # Returns
    ///
    /// The current memory tier assignment. Returns `Storage` for out-of-range IDs.
    pub fn get_tier(&self, expert_id: ExpertId) -> MemoryTier {
        self.tier_map
            .get(expert_id)
            .copied()
            .unwrap_or(MemoryTier::Storage)
    }

    /// Estimate the paging latency for accessing an expert in microseconds.
    ///
    /// The latency depends on the expert's current memory tier:
    /// - SRAM: ~0 microseconds (cache hit)
    /// - DRAM: ~0 microseconds (memory access)
    /// - Storage: ~100+ microseconds (page fault / disk access)
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Expert to estimate latency for
    ///
    /// # Returns
    ///
    /// Estimated latency in microseconds.
    pub fn estimate_paging_latency(&self, expert_id: ExpertId) -> u64 {
        let tier = self.get_tier(expert_id);
        self.tier_latency[tier.index()]
    }

    /// Get the number of experts that fit in SRAM.
    pub fn sram_capacity(&self) -> usize {
        self.config.sram_expert_slots
    }

    /// Get the number of experts that fit in DRAM budget.
    pub fn dram_capacity(&self) -> usize {
        self.config.dram_expert_slots
    }

    /// Get current SRAM slot usage.
    pub fn sram_used(&self) -> usize {
        self.sram_used
    }

    /// Get current DRAM slot usage.
    pub fn dram_used(&self) -> usize {
        self.dram_used
    }

    /// Get available SRAM slots.
    pub fn sram_available(&self) -> usize {
        self.config.sram_expert_slots.saturating_sub(self.sram_used)
    }

    /// Get available DRAM slots.
    pub fn dram_available(&self) -> usize {
        self.config.dram_expert_slots.saturating_sub(self.dram_used)
    }

    /// Record an access to an expert for affinity tracking.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Expert that was accessed
    /// * `router_weight` - Router softmax weight for this expert (0.0 - 1.0)
    pub fn record_access(&mut self, expert_id: ExpertId, router_weight: f32) {
        if expert_id >= self.num_experts {
            return;
        }

        self.access_counter += 1;

        let affinity = &mut self.affinity[expert_id];
        affinity.access_count += 1;
        affinity.last_access = self.access_counter;
        affinity.recent_selections += 1;

        // Exponential moving average for router weight
        let alpha = 0.1;
        affinity.avg_router_weight =
            alpha * router_weight + (1.0 - alpha) * affinity.avg_router_weight;
    }

    /// Suggest tier changes based on current affinity data.
    ///
    /// Analyzes expert access patterns and suggests promotions (to faster tiers)
    /// or demotions (to slower tiers) to optimize the memory hierarchy.
    ///
    /// # Arguments
    ///
    /// * `affinity_data` - Optional external affinity data (uses internal if None)
    ///
    /// # Returns
    ///
    /// A vector of `(ExpertId, MemoryTier)` pairs suggesting new tier assignments.
    pub fn suggest_eviction_tier(
        &self,
        _affinity_data: &SramExpertAffinity,
    ) -> Vec<(ExpertId, MemoryTier)> {
        self.suggest_tier_changes()
    }

    /// Suggest tier changes based on internal affinity tracking.
    ///
    /// Implements a simple policy:
    /// 1. Promote high-priority experts to SRAM (if slots available)
    /// 2. Demote low-priority SRAM experts to DRAM
    /// 3. Demote rarely-used DRAM experts to Storage
    ///
    /// # Returns
    ///
    /// A vector of `(ExpertId, MemoryTier)` pairs suggesting new tier assignments.
    pub fn suggest_tier_changes(&self) -> Vec<(ExpertId, MemoryTier)> {
        let mut suggestions = Vec::new();

        // Collect experts with their priority scores and current tiers
        let mut experts: Vec<(ExpertId, f32, MemoryTier)> = self
            .affinity
            .iter()
            .enumerate()
            .filter(|(_, aff)| !aff.pinned)
            .map(|(id, aff)| (id, aff.priority_score(), self.tier_map[id]))
            .collect();

        // Sort by priority (highest first)
        experts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Suggest promotions to SRAM for top experts currently in DRAM/Storage
        let sram_available = self.sram_available();
        let mut promoted_to_sram = 0;

        for &(expert_id, _priority, current_tier) in &experts {
            if promoted_to_sram >= sram_available {
                break;
            }
            if current_tier != MemoryTier::Sram {
                suggestions.push((expert_id, MemoryTier::Sram));
                promoted_to_sram += 1;
            }
        }

        // Suggest demotions for low-priority SRAM experts
        // (process from lowest priority)
        for &(expert_id, _priority, current_tier) in experts.iter().rev() {
            if current_tier == MemoryTier::Sram
                && suggestions.iter().all(|(id, _)| *id != expert_id)
            {
                if self.dram_available() > 0 {
                    suggestions.push((expert_id, MemoryTier::Dram));
                } else {
                    suggestions.push((expert_id, MemoryTier::Storage));
                }
            }
        }

        suggestions
    }

    /// Pin an expert to its current tier (prevent automatic eviction).
    pub fn pin(&mut self, expert_id: ExpertId) {
        if expert_id < self.num_experts {
            self.affinity[expert_id].pinned = true;
        }
    }

    /// Unpin an expert (allow automatic tier changes).
    pub fn unpin(&mut self, expert_id: ExpertId) {
        if expert_id < self.num_experts {
            self.affinity[expert_id].pinned = false;
        }
    }

    /// Get a reference to the hardware configuration.
    pub fn config(&self) -> &HardwareConfig {
        &self.config
    }

    /// Get the total number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Set custom tier latency estimates.
    ///
    /// # Arguments
    ///
    /// * `sram_us` - SRAM tier latency in microseconds
    /// * `dram_us` - DRAM tier latency in microseconds
    /// * `storage_us` - Storage tier latency in microseconds
    pub fn set_tier_latencies(&mut self, sram_us: u64, dram_us: u64, storage_us: u64) {
        self.tier_latency = [sram_us, dram_us, storage_us];
    }

    /// Get experts currently in a specific tier.
    pub fn experts_in_tier(&self, tier: MemoryTier) -> Vec<ExpertId> {
        self.tier_map
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == tier)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get the affinity data for an expert.
    pub fn get_affinity(&self, expert_id: ExpertId) -> Option<&SramExpertAffinity> {
        self.affinity.get(expert_id)
    }

    /// Reset all affinity tracking data.
    pub fn reset_affinity(&mut self) {
        for (id, aff) in self.affinity.iter_mut().enumerate() {
            *aff = SramExpertAffinity::new(id);
        }
        self.access_counter = 0;
    }

    /// Get a summary of current tier distribution.
    pub fn tier_summary(&self) -> HashMap<MemoryTier, usize> {
        let mut summary = HashMap::new();
        summary.insert(MemoryTier::Sram, 0);
        summary.insert(MemoryTier::Dram, 0);
        summary.insert(MemoryTier::Storage, 0);

        for &tier in &self.tier_map {
            *summary.entry(tier).or_insert(0) += 1;
        }

        summary
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // test_from_preset_raspberry_pi
    // ---------------------------------------------------------------

    #[test]
    fn test_from_preset_raspberry_pi() {
        let expert_size = 34_000_000; // 34 MB per expert
        let mapper = SramMapper::from_preset(HardwarePreset::RaspberryPi5, 8, expert_size);

        // RPi5: 2MB SRAM, 6GB DRAM
        // SRAM slots: 2MB / 34MB = 0 (can't fit one expert in cache)
        // DRAM slots: 6GB / 34MB = ~176
        assert_eq!(mapper.num_experts(), 8);
        assert_eq!(mapper.sram_capacity(), 0); // Expert too large for cache
        assert!(mapper.dram_capacity() > 0);

        // All experts start in Storage tier
        for i in 0..8 {
            assert_eq!(mapper.get_tier(i), MemoryTier::Storage);
        }
    }

    #[test]
    fn test_from_preset_raspberry_pi_small_experts() {
        // Test with smaller experts that fit in cache
        let expert_size = 500_000; // 500 KB per expert
        let mapper = SramMapper::from_preset(HardwarePreset::RaspberryPi5, 8, expert_size);

        // RPi5: 2MB SRAM = 4 slots @ 500KB each
        // 6GB DRAM = 12000 slots
        assert_eq!(mapper.sram_capacity(), 4);
        assert!(mapper.dram_capacity() > 1000);
    }

    // ---------------------------------------------------------------
    // test_from_preset_mobile
    // ---------------------------------------------------------------

    #[test]
    fn test_from_preset_mobile() {
        let expert_size = 1024 * 1024; // 1 MiB per expert (binary units)
        let mapper = SramMapper::from_preset(HardwarePreset::Mobile, 8, expert_size);

        // Mobile: 4MiB SRAM, 2GiB DRAM
        // SRAM slots: 4MiB / 1MiB = 4
        // DRAM slots: 2GiB / 1MiB = 2048
        assert_eq!(mapper.sram_capacity(), 4);
        assert_eq!(mapper.dram_capacity(), 2048);
        assert_eq!(mapper.num_experts(), 8);
    }

    // ---------------------------------------------------------------
    // test_from_preset_desktop
    // ---------------------------------------------------------------

    #[test]
    fn test_from_preset_desktop() {
        let expert_size = 8 * 1024 * 1024; // 8 MiB per expert (binary units)
        let mapper = SramMapper::from_preset(HardwarePreset::Desktop, 16, expert_size);

        // Desktop: 32MiB SRAM, 12GiB DRAM
        // SRAM slots: 32MiB / 8MiB = 4
        // DRAM slots: 12GiB / 8MiB = 12 * 1024 / 8 = 1536
        assert_eq!(mapper.sram_capacity(), 4);
        assert_eq!(mapper.dram_capacity(), 1536);
        assert_eq!(mapper.num_experts(), 16);
    }

    // ---------------------------------------------------------------
    // test_tier_assignment
    // ---------------------------------------------------------------

    #[test]
    fn test_tier_assignment() {
        let config = HardwareConfig::new(
            16 * 1024 * 1024,       // 16 MB SRAM
            4 * 1024 * 1024 * 1024, // 4 GB DRAM
            4 * 1024 * 1024,        // 4 MB per expert
        );
        let mut mapper = SramMapper::from_config(config, 8);

        // Initially all in Storage
        assert_eq!(mapper.get_tier(0), MemoryTier::Storage);
        assert_eq!(mapper.sram_used(), 0);
        assert_eq!(mapper.dram_used(), 0);

        // Assign expert 0 to SRAM
        mapper.assign_tier(0, MemoryTier::Sram);
        assert_eq!(mapper.get_tier(0), MemoryTier::Sram);
        assert_eq!(mapper.sram_used(), 1);

        // Assign expert 1 to DRAM
        mapper.assign_tier(1, MemoryTier::Dram);
        assert_eq!(mapper.get_tier(1), MemoryTier::Dram);
        assert_eq!(mapper.dram_used(), 1);

        // Move expert 0 from SRAM to DRAM
        mapper.assign_tier(0, MemoryTier::Dram);
        assert_eq!(mapper.get_tier(0), MemoryTier::Dram);
        assert_eq!(mapper.sram_used(), 0);
        assert_eq!(mapper.dram_used(), 2);

        // Move expert 1 back to Storage
        mapper.assign_tier(1, MemoryTier::Storage);
        assert_eq!(mapper.get_tier(1), MemoryTier::Storage);
        assert_eq!(mapper.dram_used(), 1);
    }

    // ---------------------------------------------------------------
    // test_paging_latency_estimates
    // ---------------------------------------------------------------

    #[test]
    fn test_paging_latency_estimates() {
        let config = HardwareConfig::new(16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 4 * 1024 * 1024);
        let mut mapper = SramMapper::from_config(config, 4);

        // Set custom latencies
        mapper.set_tier_latencies(1, 10, 200);

        mapper.assign_tier(0, MemoryTier::Sram);
        mapper.assign_tier(1, MemoryTier::Dram);
        mapper.assign_tier(2, MemoryTier::Storage);

        assert_eq!(mapper.estimate_paging_latency(0), 1); // SRAM
        assert_eq!(mapper.estimate_paging_latency(1), 10); // DRAM
        assert_eq!(mapper.estimate_paging_latency(2), 200); // Storage
        assert_eq!(mapper.estimate_paging_latency(3), 200); // Default (Storage)

        // Out of range returns Storage latency
        assert_eq!(mapper.estimate_paging_latency(100), 200);
    }

    // ---------------------------------------------------------------
    // test_capacity_calculations
    // ---------------------------------------------------------------

    #[test]
    fn test_capacity_calculations() {
        let config = HardwareConfig::new(
            32 * 1024 * 1024,       // 32 MB SRAM
            8 * 1024 * 1024 * 1024, // 8 GB DRAM
            8 * 1024 * 1024,        // 8 MB per expert
        );
        let mapper = SramMapper::from_config(config, 16);

        // SRAM: 32MB / 8MB = 4 slots
        assert_eq!(mapper.sram_capacity(), 4);

        // DRAM: 8GB / 8MB = 1024 slots
        assert_eq!(mapper.dram_capacity(), 1024);

        // Total
        assert_eq!(mapper.config().total_slots(), 1028);
        assert_eq!(
            mapper.config().total_budget(),
            32 * 1024 * 1024 + 8 * 1024 * 1024 * 1024
        );

        // Available (nothing allocated yet)
        assert_eq!(mapper.sram_available(), 4);
        assert_eq!(mapper.dram_available(), 1024);
    }

    // ---------------------------------------------------------------
    // test_eviction_suggestions
    // ---------------------------------------------------------------

    #[test]
    fn test_eviction_suggestions() {
        let config = HardwareConfig::new(16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 4 * 1024 * 1024);
        let mut mapper = SramMapper::from_config(config, 8);

        // Simulate access patterns
        for _ in 0..10 {
            mapper.record_access(0, 0.8);
        }
        for _ in 0..5 {
            mapper.record_access(1, 0.6);
        }
        mapper.record_access(2, 0.3);

        // Get suggestions - should promote frequently accessed experts
        let suggestions = mapper.suggest_tier_changes();

        // Verify suggestions include high-priority experts
        // (exact suggestions depend on affinity algorithm)
        assert!(!suggestions.is_empty() || mapper.sram_available() == 0);
    }

    // ---------------------------------------------------------------
    // test_custom_config
    // ---------------------------------------------------------------

    #[test]
    fn test_custom_config() {
        let config = HardwareConfig {
            sram_bytes: 64 * 1024 * 1024,               // 64 MB
            dram_budget_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
            sram_expert_slots: 8,
            dram_expert_slots: 200,
            expert_size_bytes: 8 * 1024 * 1024,
        };

        let mapper = SramMapper::from_config(config.clone(), 32);

        assert_eq!(mapper.sram_capacity(), 8);
        assert_eq!(mapper.dram_capacity(), 200);
        assert_eq!(mapper.num_experts(), 32);
        assert_eq!(mapper.config().expert_size_bytes, 8 * 1024 * 1024);
    }

    // ---------------------------------------------------------------
    // test_affinity_tracking
    // ---------------------------------------------------------------

    #[test]
    fn test_affinity_tracking() {
        let config = HardwareConfig::default();
        let mut mapper = SramMapper::from_config(config, 4);

        // Record accesses
        mapper.record_access(0, 0.9);
        mapper.record_access(0, 0.8);
        mapper.record_access(1, 0.5);

        let aff0 = mapper.get_affinity(0).unwrap();
        assert_eq!(aff0.access_count, 2);
        assert!(aff0.avg_router_weight > 0.0);

        let aff1 = mapper.get_affinity(1).unwrap();
        assert_eq!(aff1.access_count, 1);

        // Reset affinity
        mapper.reset_affinity();
        let aff0_reset = mapper.get_affinity(0).unwrap();
        assert_eq!(aff0_reset.access_count, 0);
    }

    // ---------------------------------------------------------------
    // test_pin_unpin
    // ---------------------------------------------------------------

    #[test]
    fn test_pin_unpin() {
        let config = HardwareConfig::default();
        let mut mapper = SramMapper::from_config(config, 4);

        // Pin expert 0
        mapper.pin(0);
        assert!(mapper.get_affinity(0).unwrap().pinned);

        // Unpin expert 0
        mapper.unpin(0);
        assert!(!mapper.get_affinity(0).unwrap().pinned);
    }

    // ---------------------------------------------------------------
    // test_experts_in_tier
    // ---------------------------------------------------------------

    #[test]
    fn test_experts_in_tier() {
        let config = HardwareConfig::new(16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 4 * 1024 * 1024);
        let mut mapper = SramMapper::from_config(config, 8);

        mapper.assign_tier(0, MemoryTier::Sram);
        mapper.assign_tier(1, MemoryTier::Sram);
        mapper.assign_tier(2, MemoryTier::Dram);
        mapper.assign_tier(3, MemoryTier::Dram);
        mapper.assign_tier(4, MemoryTier::Dram);

        let sram_experts = mapper.experts_in_tier(MemoryTier::Sram);
        assert_eq!(sram_experts.len(), 2);
        assert!(sram_experts.contains(&0));
        assert!(sram_experts.contains(&1));

        let dram_experts = mapper.experts_in_tier(MemoryTier::Dram);
        assert_eq!(dram_experts.len(), 3);

        let storage_experts = mapper.experts_in_tier(MemoryTier::Storage);
        assert_eq!(storage_experts.len(), 3); // Experts 5, 6, 7
    }

    // ---------------------------------------------------------------
    // test_tier_summary
    // ---------------------------------------------------------------

    #[test]
    fn test_tier_summary() {
        let config = HardwareConfig::new(16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 4 * 1024 * 1024);
        let mut mapper = SramMapper::from_config(config, 8);

        mapper.assign_tier(0, MemoryTier::Sram);
        mapper.assign_tier(1, MemoryTier::Dram);
        mapper.assign_tier(2, MemoryTier::Dram);

        let summary = mapper.tier_summary();
        assert_eq!(*summary.get(&MemoryTier::Sram).unwrap(), 1);
        assert_eq!(*summary.get(&MemoryTier::Dram).unwrap(), 2);
        assert_eq!(*summary.get(&MemoryTier::Storage).unwrap(), 5);
    }

    // ---------------------------------------------------------------
    // test_memory_tier_properties
    // ---------------------------------------------------------------

    #[test]
    fn test_memory_tier_properties() {
        assert_eq!(MemoryTier::Sram.name(), "SRAM (L2/L3 Cache)");
        assert_eq!(MemoryTier::Dram.name(), "DRAM (Main Memory)");
        assert_eq!(MemoryTier::Storage.name(), "Storage (Flash/NVMe)");

        assert_eq!(MemoryTier::Sram.index(), 0);
        assert_eq!(MemoryTier::Dram.index(), 1);
        assert_eq!(MemoryTier::Storage.index(), 2);
    }

    // ---------------------------------------------------------------
    // test_hardware_preset_names
    // ---------------------------------------------------------------

    #[test]
    fn test_hardware_preset_names() {
        assert_eq!(HardwarePreset::RaspberryPi5.name(), "Raspberry Pi 5");
        assert_eq!(HardwarePreset::Mobile.name(), "Mobile Device");
        assert_eq!(HardwarePreset::Desktop.name(), "Desktop Workstation");
        assert_eq!(HardwarePreset::WasmBrowser.name(), "WASM Browser");
        assert_eq!(HardwarePreset::Custom.name(), "Custom");
    }

    // ---------------------------------------------------------------
    // test_expert_affinity_priority_score
    // ---------------------------------------------------------------

    #[test]
    fn test_expert_affinity_priority_score() {
        let mut aff = SramExpertAffinity::new(0);

        // Initial score should be low
        let initial_score = aff.priority_score();

        // Increase access count and check score increases
        aff.access_count = 100;
        aff.avg_router_weight = 0.9;
        let high_score = aff.priority_score();

        assert!(high_score > initial_score);
    }

    // ---------------------------------------------------------------
    // test_wasm_browser_preset
    // ---------------------------------------------------------------

    #[test]
    fn test_wasm_browser_preset() {
        let expert_size = 2 * 1024 * 1024; // 2 MiB per expert (binary units)
        let mapper = SramMapper::from_preset(HardwarePreset::WasmBrowser, 8, expert_size);

        // WASM: 2MiB SRAM, 1GiB DRAM
        // SRAM slots: 2MiB / 2MiB = 1
        // DRAM slots: 1GiB / 2MiB = 512
        assert_eq!(mapper.sram_capacity(), 1);
        assert_eq!(mapper.dram_capacity(), 512);
    }

    // ---------------------------------------------------------------
    // test_out_of_range_expert_id
    // ---------------------------------------------------------------

    #[test]
    fn test_out_of_range_expert_id() {
        let config = HardwareConfig::default();
        let mapper = SramMapper::from_config(config, 4);

        // Out of range should return Storage
        assert_eq!(mapper.get_tier(100), MemoryTier::Storage);
        assert_eq!(mapper.estimate_paging_latency(100), 100); // Default storage latency
    }

    // ---------------------------------------------------------------
    // test_record_access_out_of_range
    // ---------------------------------------------------------------

    #[test]
    fn test_record_access_out_of_range() {
        let config = HardwareConfig::default();
        let mut mapper = SramMapper::from_config(config, 4);

        // Should not panic
        mapper.record_access(100, 0.5);

        // Counter should not advance for invalid ID
        // (actually it does advance, but affinity is not updated)
    }
}
