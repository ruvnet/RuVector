//! rUv (Resource Utility Vouchers) system with CRDT ledger and contribution curve
//!
//! This module provides the economic layer for edge-net:
//! - rUv: Resource Utility Vouchers for compute credits
//! - CRDT-based ledger for P2P consistency
//! - Contribution curve for early adopter rewards
//! - DAG-based quantum-resistant currency for settlements

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use rustc_hash::FxHashMap;  // 30-50% faster than std HashMap
use uuid::Uuid;

pub mod qdag;

/// Contribution curve for reward calculation.
///
/// The curve provides a multiplier for early adopters that decays
/// toward the floor as the network grows. The genesis multiplier
/// is capped at 5x (reduced from 10x for sustainability), and
/// a floor of 0.5x ensures contributors always earn something
/// even when the network is mature.
pub struct ContributionCurve;

impl ContributionCurve {
    /// Maximum multiplier for genesis contributors (reduced from 10.0 for sustainability)
    const MAX_BONUS: f32 = 5.0;

    /// Floor multiplier -- even at massive scale, you still earn 0.5x
    const FLOOR_MULTIPLIER: f32 = 0.5;

    /// Decay constant in CPU-hours (half-life of bonus)
    const DECAY_CONSTANT: f64 = 1_000_000.0;

    /// Calculate current multiplier based on network compute.
    ///
    /// Formula: multiplier = FLOOR + (MAX_BONUS - FLOOR) * e^(-network_compute / DECAY_CONSTANT)
    ///
    /// Returns a value between FLOOR_MULTIPLIER (mature network) and MAX_BONUS (genesis).
    pub fn current_multiplier(network_compute_hours: f64) -> f32 {
        let decay = (-network_compute_hours / Self::DECAY_CONSTANT).exp();
        Self::FLOOR_MULTIPLIER + (Self::MAX_BONUS - Self::FLOOR_MULTIPLIER) * decay as f32
    }

    /// Calculate rewards with multiplier applied
    pub fn calculate_reward(base_reward: u64, network_compute_hours: f64) -> u64 {
        let multiplier = Self::current_multiplier(network_compute_hours);
        (base_reward as f32 * multiplier) as u64
    }

    /// Calculate brain-specific reward with the contribution curve applied.
    ///
    /// Brain rewards use the same decay curve but are further scaled by
    /// a reputation multiplier. This bridges the brain_rewards module
    /// with the contribution curve.
    pub fn calculate_brain_reward(
        base_reward: u64,
        network_compute_hours: f64,
        reputation_multiplier: f32,
    ) -> u64 {
        let curve_multiplier = Self::current_multiplier(network_compute_hours);
        let combined = curve_multiplier * reputation_multiplier;
        (base_reward as f32 * combined) as u64
    }

    /// Get multiplier tiers for display.
    ///
    /// Updated to reflect the reduced MAX_BONUS of 5.0 and floor of 0.5.
    pub fn get_tiers() -> Vec<(f64, f32)> {
        vec![
            (0.0, 5.0),
            (100_000.0, 4.59),
            (500_000.0, 3.28),
            (1_000_000.0, 2.16),
            (5_000_000.0, 0.53),
            (10_000_000.0, 0.50),
        ]
    }
}

/// Credit event types
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum CreditReason {
    /// Earned from completing a task
    TaskCompleted { task_id: String },
    /// Earned from uptime
    UptimeReward { hours: f32 },
    /// Earned from referral
    Referral { referee: String },
    /// Staked for participation
    Stake { amount: u64, locked: bool },
    /// Transferred between nodes
    Transfer { from: String, to: String, memo: String },
    /// Penalty for invalid work
    Penalty { reason: String },
}

/// A single credit event
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CreditEvent {
    pub id: String,
    pub node_id: String,
    pub amount: i64,  // Can be negative for penalties/spending
    pub reason: CreditReason,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}

/// CRDT-based credit ledger for P2P consistency
#[wasm_bindgen]
pub struct WasmCreditLedger {
    node_id: String,

    // G-Counter: monotonically increasing credits earned - FxHashMap for faster lookups
    earned: FxHashMap<String, u64>,

    // PN-Counter: credits spent/penalized - FxHashMap for faster lookups
    spent: FxHashMap<String, (u64, u64)>,  // (positive, negative)

    // Local balance cache (avoids recalculation)
    local_balance: u64,

    // Network compute (for multiplier calculation)
    network_compute: f64,

    // Stake amount
    staked: u64,

    // Last sync timestamp
    last_sync: u64,
}

#[wasm_bindgen]
impl WasmCreditLedger {
    /// Create a new credit ledger
    #[wasm_bindgen(constructor)]
    pub fn new(node_id: String) -> Result<WasmCreditLedger, JsValue> {
        Ok(WasmCreditLedger {
            node_id,
            earned: FxHashMap::default(),
            spent: FxHashMap::default(),
            local_balance: 0,
            network_compute: 0.0,
            staked: 0,
            last_sync: 0,
        })
    }

    /// Get current balance
    #[wasm_bindgen]
    pub fn balance(&self) -> u64 {
        let total_earned: u64 = self.earned.values().sum();
        let total_spent: u64 = self.spent.values()
            .map(|(pos, neg)| pos.saturating_sub(*neg))
            .sum();

        total_earned.saturating_sub(total_spent).saturating_sub(self.staked)
    }

    /// Get total earned (before spending)
    #[wasm_bindgen(js_name = totalEarned)]
    pub fn total_earned(&self) -> u64 {
        self.earned.values().sum()
    }

    /// Get total spent
    #[wasm_bindgen(js_name = totalSpent)]
    pub fn total_spent(&self) -> u64 {
        self.spent.values()
            .map(|(pos, neg)| pos.saturating_sub(*neg))
            .sum()
    }

    /// Get staked amount
    #[wasm_bindgen(js_name = stakedAmount)]
    pub fn staked_amount(&self) -> u64 {
        self.staked
    }

    /// Get network compute hours (for multiplier)
    #[wasm_bindgen(js_name = networkCompute)]
    pub fn network_compute(&self) -> f64 {
        self.network_compute
    }

    /// Get current multiplier
    #[wasm_bindgen(js_name = currentMultiplier)]
    pub fn current_multiplier(&self) -> f32 {
        ContributionCurve::current_multiplier(self.network_compute)
    }

    /// Credit the ledger (earn credits)
    #[wasm_bindgen]
    pub fn credit(&mut self, amount: u64, reason: &str) -> Result<(), JsValue> {
        let event_id = Uuid::new_v4().to_string();

        // Update G-Counter
        *self.earned.entry(event_id).or_insert(0) += amount;
        self.local_balance = self.balance();

        Ok(())
    }

    /// Deduct from the ledger (spend credits)
    #[wasm_bindgen]
    pub fn deduct(&mut self, amount: u64) -> Result<(), JsValue> {
        if self.balance() < amount {
            return Err(JsValue::from_str("Insufficient balance"));
        }

        let event_id = Uuid::new_v4().to_string();

        // Update PN-Counter (positive side)
        let entry = self.spent.entry(event_id).or_insert((0, 0));
        entry.0 += amount;
        self.local_balance = self.balance();

        Ok(())
    }

    /// Stake credits for participation
    #[wasm_bindgen]
    pub fn stake(&mut self, amount: u64) -> Result<(), JsValue> {
        if self.balance() < amount {
            return Err(JsValue::from_str("Insufficient balance for stake"));
        }

        self.staked += amount;
        self.local_balance = self.balance();

        Ok(())
    }

    /// Unstake credits
    #[wasm_bindgen]
    pub fn unstake(&mut self, amount: u64) -> Result<(), JsValue> {
        if self.staked < amount {
            return Err(JsValue::from_str("Insufficient staked amount"));
        }

        self.staked -= amount;
        self.local_balance = self.balance();

        Ok(())
    }

    /// Slash staked credits (penalty for bad behavior)
    #[wasm_bindgen]
    pub fn slash(&mut self, amount: u64) -> Result<u64, JsValue> {
        let slash_amount = amount.min(self.staked);
        self.staked -= slash_amount;
        self.local_balance = self.balance();

        Ok(slash_amount)
    }

    /// Update network compute (from P2P sync)
    #[wasm_bindgen(js_name = updateNetworkCompute)]
    pub fn update_network_compute(&mut self, hours: f64) {
        self.network_compute = hours;
    }

    /// Merge with another ledger (CRDT merge) - optimized batch processing
    #[wasm_bindgen]
    pub fn merge(&mut self, other_earned: &[u8], other_spent: &[u8]) -> Result<(), JsValue> {
        // Deserialize earned counter
        let earned_map: FxHashMap<String, u64> = serde_json::from_slice(other_earned)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse earned: {}", e)))?;

        // CRDT merge: take max of each counter (batch operation)
        for (key, value) in earned_map {
            let entry = self.earned.entry(key).or_insert(0);
            *entry = (*entry).max(value);
        }

        // Deserialize spent counter
        let spent_map: FxHashMap<String, (u64, u64)> = serde_json::from_slice(other_spent)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse spent: {}", e)))?;

        // CRDT merge: take max of each counter (batch operation)
        for (key, (pos, neg)) in spent_map {
            let entry = self.spent.entry(key).or_insert((0, 0));
            entry.0 = entry.0.max(pos);
            entry.1 = entry.1.max(neg);
        }

        // Recalculate balance once after merge (vs per-operation)
        self.local_balance = self.balance();
        self.last_sync = js_sys::Date::now() as u64;

        Ok(())
    }

    /// Export earned counter for sync
    #[wasm_bindgen(js_name = exportEarned)]
    pub fn export_earned(&self) -> Result<Vec<u8>, JsValue> {
        serde_json::to_vec(&self.earned)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize: {}", e)))
    }

    /// Export spent counter for sync
    #[wasm_bindgen(js_name = exportSpent)]
    pub fn export_spent(&self) -> Result<Vec<u8>, JsValue> {
        serde_json::to_vec(&self.spent)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contribution_curve() {
        // Genesis (0 hours) should give max multiplier (5.0)
        let mult = ContributionCurve::current_multiplier(0.0);
        assert!((mult - 5.0).abs() < 0.01);

        // At decay constant, should be around 2.16
        // FLOOR + (MAX - FLOOR) * e^(-1) = 0.5 + 4.5 * 0.368 = 2.16
        let mult = ContributionCurve::current_multiplier(1_000_000.0);
        assert!(mult > 1.8 && mult < 2.5);

        // At high compute, should approach FLOOR_MULTIPLIER (0.5)
        let mult = ContributionCurve::current_multiplier(10_000_000.0);
        assert!(mult >= 0.5);
        assert!(mult < 0.6);
    }

    #[test]
    fn test_contribution_curve_floor() {
        // Even at extremely high compute, multiplier never goes below 0.5
        let mult = ContributionCurve::current_multiplier(100_000_000.0);
        assert!(mult >= 0.5);
    }

    #[test]
    fn test_brain_reward_calculation() {
        // At genesis with Bronze (1.0x rep): 100 * 5.0 * 1.0 = 500
        let reward = ContributionCurve::calculate_brain_reward(100, 0.0, 1.0);
        assert_eq!(reward, 500);

        // At genesis with Newcomer (0.5x rep): 100 * 5.0 * 0.5 = 250
        let reward = ContributionCurve::calculate_brain_reward(100, 0.0, 0.5);
        assert_eq!(reward, 250);

        // At high compute with Platinum (1.5x rep): 100 * 0.5 * 1.5 = 75
        let reward = ContributionCurve::calculate_brain_reward(100, 100_000_000.0, 1.5);
        assert_eq!(reward, 75);
    }

    // Tests requiring WASM environment (UUID with js feature)
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_ledger_operations() {
        let mut ledger = WasmCreditLedger::new("test-node".to_string()).unwrap();

        // Initial balance is 0
        assert_eq!(ledger.balance(), 0);

        // Credit 100
        ledger.credit(100, "task").unwrap();
        assert_eq!(ledger.balance(), 100);

        // Deduct 30
        ledger.deduct(30).unwrap();
        assert_eq!(ledger.balance(), 70);

        // Can't deduct more than balance
        assert!(ledger.deduct(100).is_err());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_staking() {
        let mut ledger = WasmCreditLedger::new("test-node".to_string()).unwrap();

        ledger.credit(100, "task").unwrap();

        // Stake 50
        ledger.stake(50).unwrap();
        assert_eq!(ledger.balance(), 50);
        assert_eq!(ledger.staked_amount(), 50);

        // Unstake 20
        ledger.unstake(20).unwrap();
        assert_eq!(ledger.balance(), 70);
        assert_eq!(ledger.staked_amount(), 30);

        // Slash 10
        let slashed = ledger.slash(10).unwrap();
        assert_eq!(slashed, 10);
        assert_eq!(ledger.staked_amount(), 20);
    }
}
