//! # Brain Operation Rewards
//!
//! Defines the rUv reward schedule for brain-related operations.
//! Designed for accessibility (free tier for reads) and sustainability
//! (rewards decrease as the network matures via a halving schedule).
//!
//! ## Design Principles
//!
//! 1. **Free to read, earn to write**: No barriers to consuming knowledge;
//!    contributions earn rUv.
//! 2. **Halving schedule**: Rewards halve every 100K brain operations,
//!    similar to Bitcoin halvings but at the application layer.
//! 3. **Reputation floor**: Even newcomers earn (at 0.5x), just less
//!    than established nodes.
//! 4. **Sustainable cap**: Max rUv minted per epoch is bounded by the
//!    protocol budget.
//! 5. **Transparency**: All pricing and rewards visible via helper methods.
//!
//! ## Reward Table
//!
//! ```text
//! ┌──────────────┬──────────┬───────────────────────────────────┐
//! │  Operation   │ Cost     │ Base Reward                       │
//! ├──────────────┼──────────┼───────────────────────────────────┤
//! │  Search      │ FREE     │ 0 rUv                             │
//! │  Status      │ FREE     │ 0 rUv                             │
//! │  List        │ FREE     │ 0 rUv                             │
//! │  Share       │ 0 rUv    │ 2 rUv (if quality > 0.5)          │
//! │  Vote        │ 0 rUv    │ 0.1 rUv per vote                  │
//! │  Embedding   │ 0 rUv    │ 1 rUv per embedding               │
//! │  LoRA Train  │ 0 rUv    │ 5 rUv per accepted gradient       │
//! │  WASM Compute│ 0 rUv    │ compute-time based                │
//! └──────────────┴──────────┴───────────────────────────────────┘
//! ```

use serde::{Serialize, Deserialize};
use super::reputation::ReputationTier;

/// Number of brain operations between each halving
pub const HALVING_INTERVAL: u64 = 100_000;

/// Maximum halvings before reward floor kicks in
pub const MAX_HALVINGS: u32 = 10;

/// Default epoch budget in rUv (caps total minting per epoch)
pub const DEFAULT_EPOCH_BUDGET: u64 = 1_000_000;

/// Floor reward multiplier -- rewards never go below this fraction
/// of the base amount, even after many halvings
pub const FLOOR_MULTIPLIER: f64 = 0.01;

/// Types of brain operations with their associated parameters.
///
/// Free-to-read, earn-to-write model:
/// - Search, Status, List: FREE (no cost, no reward)
/// - Share: costs 0 rUv, earns 2 rUv if quality > 0.5
/// - Vote: costs 0 rUv, earns fractional rUv per vote
/// - Embedding: costs 0 rUv, earns 1 rUv per embedding
/// - LoraTraining: costs 0 rUv, earns 5 rUv per accepted gradient
/// - WasmCompute: costs 0 rUv, earns based on compute time
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BrainOpType {
    /// Search the brain -- always free, no reward
    Search,
    /// Check brain status -- always free, no reward
    Status,
    /// List brain contents -- always free, no reward
    List,
    /// Share knowledge with the brain (quality 0.0-1.0)
    Share { quality: f64 },
    /// Vote on brain content
    Vote,
    /// Generate an embedding
    Embedding,
    /// LoRA training contribution
    LoraTraining,
    /// WASM compute contribution
    WasmCompute { seconds: u64 },
}

/// Result of a brain reward calculation.
///
/// `cost` is how much the user pays (0 for free-tier ops).
/// `reward` is how much the user earns.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BrainRewardResult {
    /// Cost to the user in rUv (0 for free operations)
    pub cost: u64,
    /// Reward earned in rUv
    pub reward: u64,
    /// Whether this operation is in the free tier
    pub is_free_tier: bool,
    /// The halving multiplier applied (1.0 at genesis, 0.5 after first halving, etc.)
    pub halving_multiplier: f64,
    /// The reputation multiplier applied
    pub reputation_multiplier: f64,
}

/// Brain reward engine that tracks epochs and budgets.
///
/// Implements a halving schedule where rewards decrease as the network
/// matures, combined with per-epoch budgets to cap total minting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainRewards {
    /// Current epoch number (rewards decrease over epochs)
    epoch: u64,
    /// Total brain operations across all epochs
    total_ops: u64,
    /// Operations in the current epoch
    epoch_ops: u64,
    /// Budget remaining in the current epoch (rUv)
    epoch_budget: u64,
    /// Maximum budget per epoch
    max_epoch_budget: u64,
    /// Total rUv minted through brain rewards (lifetime)
    total_minted: u64,
}

impl BrainRewards {
    /// Create a new brain rewards engine with default budget.
    pub fn new() -> Self {
        Self {
            epoch: 0,
            total_ops: 0,
            epoch_ops: 0,
            epoch_budget: DEFAULT_EPOCH_BUDGET,
            max_epoch_budget: DEFAULT_EPOCH_BUDGET,
            total_minted: 0,
        }
    }

    /// Create a brain rewards engine with a custom epoch budget.
    pub fn with_budget(max_epoch_budget: u64) -> Self {
        Self {
            epoch: 0,
            total_ops: 0,
            epoch_ops: 0,
            epoch_budget: max_epoch_budget,
            max_epoch_budget,
            total_minted: 0,
        }
    }

    /// Get the current epoch number.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get total brain operations processed (lifetime).
    pub fn total_ops(&self) -> u64 {
        self.total_ops
    }

    /// Get operations processed in the current epoch.
    pub fn epoch_ops(&self) -> u64 {
        self.epoch_ops
    }

    /// Get the remaining budget for this epoch.
    pub fn epoch_budget(&self) -> u64 {
        self.epoch_budget
    }

    /// Get total rUv minted through brain rewards (lifetime).
    pub fn total_minted(&self) -> u64 {
        self.total_minted
    }

    /// Calculate the halving multiplier based on total network operations.
    ///
    /// Rewards halve every `HALVING_INTERVAL` operations. After
    /// `MAX_HALVINGS`, the multiplier is floored at `FLOOR_MULTIPLIER`.
    pub fn halving_multiplier(&self) -> f64 {
        let halvings = (self.total_ops / HALVING_INTERVAL) as u32;
        if halvings >= MAX_HALVINGS {
            return FLOOR_MULTIPLIER;
        }
        let mult = 0.5_f64.powi(halvings as i32);
        mult.max(FLOOR_MULTIPLIER)
    }

    /// Calculate the reputation multiplier for a given tier.
    ///
    /// This extends the existing tier multipliers to include the
    /// Newcomer tier at 0.5x.
    pub fn reputation_multiplier(tier: &ReputationTier) -> f64 {
        match tier {
            ReputationTier::Newcomer => 0.5,
            ReputationTier::Bronze => 1.0,
            ReputationTier::Silver => 1.1,
            ReputationTier::Gold => 1.25,
            ReputationTier::Platinum => 1.5,
        }
    }

    /// Calculate reward for a brain operation.
    ///
    /// Returns a `BrainRewardResult` with cost and reward.
    /// Free-tier operations (Search, Status, List) always cost 0 and earn 0.
    /// Earn-tier operations cost 0 but earn rUv scaled by halving and reputation.
    pub fn calculate(&self, op: &BrainOpType, reputation_tier: &ReputationTier) -> BrainRewardResult {
        let halving = self.halving_multiplier();
        let rep_mult = Self::reputation_multiplier(reputation_tier);

        match op {
            // Free tier: no cost, no reward
            BrainOpType::Search | BrainOpType::Status | BrainOpType::List => {
                BrainRewardResult {
                    cost: 0,
                    reward: 0,
                    is_free_tier: true,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }

            // Share: earn 2 rUv base if quality > 0.5, otherwise 0
            BrainOpType::Share { quality } => {
                let base_reward = if *quality > 0.5 { 2.0 } else { 0.0 };
                let reward = (base_reward * halving * rep_mult) as u64;
                BrainRewardResult {
                    cost: 0,
                    reward: reward.min(self.epoch_budget),
                    is_free_tier: false,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }

            // Vote: earn 0.1 rUv base (rounds to at least 1 if multiplied high enough)
            BrainOpType::Vote => {
                // Use fixed-point: 0.1 rUv base
                // At genesis with Platinum: 0.1 * 1.0 * 1.5 = 0.15 -> rounds to 0
                // We use a minimum of 1 rUv for non-zero results to avoid zero rewards
                let raw = 0.1 * halving * rep_mult;
                let reward = if raw >= 0.05 { (raw as u64).max(1) } else { 0 };
                BrainRewardResult {
                    cost: 0,
                    reward: reward.min(self.epoch_budget),
                    is_free_tier: false,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }

            // Embedding: earn 1 rUv base
            BrainOpType::Embedding => {
                let reward = (1.0 * halving * rep_mult) as u64;
                BrainRewardResult {
                    cost: 0,
                    reward: reward.min(self.epoch_budget),
                    is_free_tier: false,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }

            // LoRA training: earn 5 rUv base
            BrainOpType::LoraTraining => {
                let reward = (5.0 * halving * rep_mult) as u64;
                BrainRewardResult {
                    cost: 0,
                    reward: reward.min(self.epoch_budget),
                    is_free_tier: false,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }

            // WASM compute: earn 1 rUv per second base
            BrainOpType::WasmCompute { seconds } => {
                let base = *seconds as f64;
                let reward = (base * halving * rep_mult) as u64;
                BrainRewardResult {
                    cost: 0,
                    reward: reward.min(self.epoch_budget),
                    is_free_tier: false,
                    halving_multiplier: halving,
                    reputation_multiplier: rep_mult,
                }
            }
        }
    }

    /// Record a brain operation and return the reward result.
    ///
    /// This both calculates the reward and updates internal state
    /// (operation counters, budget, minted totals).
    pub fn record_operation(&mut self, op: &BrainOpType, reputation_tier: &ReputationTier) -> BrainRewardResult {
        let result = self.calculate(op, reputation_tier);

        // Update counters
        self.total_ops += 1;
        self.epoch_ops += 1;

        // Deduct from epoch budget
        if result.reward > 0 && self.epoch_budget >= result.reward {
            self.epoch_budget -= result.reward;
            self.total_minted += result.reward;
        }

        result
    }

    /// Advance to the next epoch, resetting the epoch budget.
    ///
    /// The budget resets to the max epoch budget, and the epoch
    /// operation counter resets. Total ops and total minted persist.
    pub fn advance_epoch(&mut self) {
        self.epoch += 1;
        self.epoch_ops = 0;
        self.epoch_budget = self.max_epoch_budget;
    }

    /// Get a human-readable summary of the reward schedule as JSON.
    ///
    /// Useful for dashboards and transparency displays.
    pub fn get_schedule_summary(&self) -> String {
        let halving = self.halving_multiplier();
        let halvings_completed = (self.total_ops / HALVING_INTERVAL).min(MAX_HALVINGS as u64);
        let next_halving_at = (self.total_ops / HALVING_INTERVAL + 1) * HALVING_INTERVAL;
        let ops_until_next = next_halving_at - self.total_ops;

        let share_reward = format!("{:.1} rUv (quality > 0.5)", 2.0 * halving);
        let vote_reward = format!("{:.2} rUv", 0.1 * halving);
        let embedding_reward = format!("{:.1} rUv", 1.0 * halving);
        let lora_reward = format!("{:.1} rUv", 5.0 * halving);
        let wasm_reward = format!("{:.1} rUv/second", 1.0 * halving);

        let summary = serde_json::json!({
            "epoch": self.epoch,
            "total_operations": self.total_ops,
            "epoch_operations": self.epoch_ops,
            "epoch_budget_remaining": self.epoch_budget,
            "max_epoch_budget": self.max_epoch_budget,
            "total_minted": self.total_minted,
            "current_halving_multiplier": halving,
            "halvings_completed": halvings_completed,
            "next_halving_at": next_halving_at,
            "operations_until_next_halving": ops_until_next,
            "free_tier_operations": ["Search", "Status", "List"],
            "earn_tier_operations": {
                "Share": share_reward,
                "Vote": vote_reward,
                "Embedding": embedding_reward,
                "LoraTraining": lora_reward,
                "WasmCompute": wasm_reward
            },
            "reputation_multipliers": {
                "Newcomer": 0.5,
                "Bronze": 1.0,
                "Silver": 1.1,
                "Gold": 1.25,
                "Platinum": 1.5
            }
        });
        serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string())
    }
}

impl Default for BrainRewards {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_tier_operations() {
        let rewards = BrainRewards::new();

        // Search is free
        let result = rewards.calculate(&BrainOpType::Search, &ReputationTier::Newcomer);
        assert_eq!(result.cost, 0);
        assert_eq!(result.reward, 0);
        assert!(result.is_free_tier);

        // Status is free
        let result = rewards.calculate(&BrainOpType::Status, &ReputationTier::Bronze);
        assert_eq!(result.cost, 0);
        assert_eq!(result.reward, 0);
        assert!(result.is_free_tier);

        // List is free
        let result = rewards.calculate(&BrainOpType::List, &ReputationTier::Platinum);
        assert_eq!(result.cost, 0);
        assert_eq!(result.reward, 0);
        assert!(result.is_free_tier);
    }

    #[test]
    fn test_share_reward() {
        let rewards = BrainRewards::new();

        // High quality share earns reward
        let result = rewards.calculate(
            &BrainOpType::Share { quality: 0.8 },
            &ReputationTier::Bronze,
        );
        assert_eq!(result.cost, 0);
        assert_eq!(result.reward, 2); // 2 * 1.0 halving * 1.0 rep = 2
        assert!(!result.is_free_tier);

        // Low quality share earns nothing
        let result = rewards.calculate(
            &BrainOpType::Share { quality: 0.3 },
            &ReputationTier::Bronze,
        );
        assert_eq!(result.cost, 0);
        assert_eq!(result.reward, 0);
    }

    #[test]
    fn test_vote_reward() {
        let rewards = BrainRewards::new();

        // Vote at genesis with Bronze (1.0x rep)
        let result = rewards.calculate(&BrainOpType::Vote, &ReputationTier::Bronze);
        assert_eq!(result.cost, 0);
        // 0.1 * 1.0 * 1.0 = 0.1 -> rounds up to minimum of 1
        assert_eq!(result.reward, 1);
    }

    #[test]
    fn test_embedding_reward() {
        let rewards = BrainRewards::new();

        let result = rewards.calculate(&BrainOpType::Embedding, &ReputationTier::Gold);
        assert_eq!(result.cost, 0);
        // 1.0 * 1.0 halving * 1.25 rep = 1.25 -> 1
        assert_eq!(result.reward, 1);
    }

    #[test]
    fn test_lora_training_reward() {
        let rewards = BrainRewards::new();

        let result = rewards.calculate(&BrainOpType::LoraTraining, &ReputationTier::Platinum);
        assert_eq!(result.cost, 0);
        // 5.0 * 1.0 halving * 1.5 rep = 7.5 -> 7
        assert_eq!(result.reward, 7);
    }

    #[test]
    fn test_wasm_compute_reward() {
        let rewards = BrainRewards::new();

        let result = rewards.calculate(
            &BrainOpType::WasmCompute { seconds: 10 },
            &ReputationTier::Silver,
        );
        assert_eq!(result.cost, 0);
        // 10 * 1.0 halving * 1.1 rep = 11.0 -> 11
        assert_eq!(result.reward, 11);
    }

    #[test]
    fn test_newcomer_gets_half_rewards() {
        let rewards = BrainRewards::new();

        let newcomer = rewards.calculate(&BrainOpType::LoraTraining, &ReputationTier::Newcomer);
        let bronze = rewards.calculate(&BrainOpType::LoraTraining, &ReputationTier::Bronze);

        // Newcomer: 5 * 1.0 * 0.5 = 2.5 -> 2
        assert_eq!(newcomer.reward, 2);
        // Bronze: 5 * 1.0 * 1.0 = 5
        assert_eq!(bronze.reward, 5);
        // Newcomer earns less but still earns
        assert!(newcomer.reward > 0);
        assert!(newcomer.reward < bronze.reward);
    }

    #[test]
    fn test_halving_schedule() {
        let mut rewards = BrainRewards::new();

        // At genesis, multiplier is 1.0
        assert!((rewards.halving_multiplier() - 1.0).abs() < 0.001);

        // Simulate 100K operations
        rewards.total_ops = HALVING_INTERVAL;
        assert!((rewards.halving_multiplier() - 0.5).abs() < 0.001);

        // After 200K operations
        rewards.total_ops = 2 * HALVING_INTERVAL;
        assert!((rewards.halving_multiplier() - 0.25).abs() < 0.001);

        // After max halvings, hits floor
        rewards.total_ops = (MAX_HALVINGS as u64 + 5) * HALVING_INTERVAL;
        assert!((rewards.halving_multiplier() - FLOOR_MULTIPLIER).abs() < 0.001);
    }

    #[test]
    fn test_epoch_budget_cap() {
        let mut rewards = BrainRewards::with_budget(10);

        // Record a LoRA training that would earn 5
        let result = rewards.record_operation(&BrainOpType::LoraTraining, &ReputationTier::Bronze);
        assert_eq!(result.reward, 5);
        assert_eq!(rewards.epoch_budget(), 5);

        // Record another -- still fits in budget
        let result = rewards.record_operation(&BrainOpType::LoraTraining, &ReputationTier::Bronze);
        assert_eq!(result.reward, 5);
        assert_eq!(rewards.epoch_budget(), 0);

        // Budget is exhausted, rewards are capped to 0
        let result = rewards.calculate(&BrainOpType::LoraTraining, &ReputationTier::Bronze);
        // Reward is min(5, 0) = 0 because budget is 0
        assert_eq!(result.reward, 0);
    }

    #[test]
    fn test_advance_epoch_resets_budget() {
        let mut rewards = BrainRewards::with_budget(100);

        // Spend some budget
        rewards.record_operation(&BrainOpType::LoraTraining, &ReputationTier::Bronze);
        assert!(rewards.epoch_budget() < 100);
        assert_eq!(rewards.epoch_ops(), 1);

        // Advance epoch
        rewards.advance_epoch();
        assert_eq!(rewards.epoch_budget(), 100);
        assert_eq!(rewards.epoch_ops(), 0);
        assert_eq!(rewards.epoch(), 1);
        // Total ops persist
        assert_eq!(rewards.total_ops(), 1);
    }

    #[test]
    fn test_record_operation_updates_counters() {
        let mut rewards = BrainRewards::new();

        rewards.record_operation(&BrainOpType::Search, &ReputationTier::Bronze);
        assert_eq!(rewards.total_ops(), 1);
        assert_eq!(rewards.epoch_ops(), 1);
        // Search earns 0, so total_minted stays 0
        assert_eq!(rewards.total_minted(), 0);

        rewards.record_operation(&BrainOpType::Embedding, &ReputationTier::Bronze);
        assert_eq!(rewards.total_ops(), 2);
        assert_eq!(rewards.total_minted(), 1); // 1 rUv for embedding
    }

    #[test]
    fn test_schedule_summary_is_valid_json() {
        let rewards = BrainRewards::new();
        let summary = rewards.get_schedule_summary();
        let parsed: serde_json::Value = serde_json::from_str(&summary).unwrap();
        assert!(parsed.is_object());
        assert!(parsed["epoch"].is_number());
        assert!(parsed["free_tier_operations"].is_array());
    }

    #[test]
    fn test_default_impl() {
        let rewards = BrainRewards::default();
        assert_eq!(rewards.epoch(), 0);
        assert_eq!(rewards.total_ops(), 0);
        assert_eq!(rewards.epoch_budget(), DEFAULT_EPOCH_BUDGET);
    }
}
