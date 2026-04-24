//! CRDT-based state merging for parallel subagents (ADR-103 B7).
//!
//! This module provides deterministic conflict resolution when multiple
//! subagents execute in parallel and produce overlapping state updates.
//!
//! # Architecture
//!
//! - **VectorClock**: Tracks causal ordering of events across nodes
//! - **LwwRegister**: Last-Write-Wins register with vector clock timestamps
//! - **CrdtState**: A map of LWW registers representing subagent state
//! - **merge_subagent_results**: Deterministically merges parallel results
//!
//! # Example
//!
//! ```
//! use rvagent_subagents::crdt_merge::{CrdtState, merge_subagent_results};
//!
//! let mut parent = CrdtState::new(0);
//! parent.set("key1", b"parent_value".to_vec());
//!
//! let mut child1 = CrdtState::new(1);
//! child1.set("key1", b"child1_value".to_vec());
//! child1.set("key2", b"child1_only".to_vec());
//!
//! let mut child2 = CrdtState::new(2);
//! child2.set("key1", b"child2_value".to_vec());
//! child2.set("key3", b"child2_only".to_vec());
//!
//! merge_subagent_results(&mut parent, vec![child1, child2]).unwrap();
//!
//! // parent now has deterministic merge of all three states
//! assert!(parent.get("key1").is_some());
//! assert!(parent.get("key2").is_some());
//! assert!(parent.get("key3").is_some());
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during CRDT merge operations.
#[derive(Debug, Error)]
pub enum MergeError {
    /// Node IDs must be unique across all states being merged.
    #[error("Duplicate node ID {0} detected in merge operation")]
    DuplicateNodeId(u32),

    /// Merge operation requires at least one subagent result.
    #[error("Cannot merge empty result set")]
    EmptyResultSet,

    /// Internal consistency error (should never happen in correct usage).
    #[error("Internal merge error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// VectorClock
// ---------------------------------------------------------------------------

/// Vector clock for tracking causal ordering of events.
///
/// A vector clock is a map from node IDs to logical timestamps. It allows
/// us to determine whether two events are causally related (one happened
/// before the other) or concurrent (neither caused the other).
///
/// # Properties
///
/// - `vc1 < vc2` (happens-before) iff all entries in vc1 are <= vc2 and at least one is strictly <
/// - `vc1 || vc2` (concurrent) iff neither happens-before relationship holds
///
/// # Example
///
/// ```
/// use rvagent_subagents::crdt_merge::VectorClock;
///
/// let mut vc1 = VectorClock::new(1);
/// vc1.tick(1); // {1: 1}
///
/// let mut vc2 = VectorClock::new(2);
/// vc2.tick(2); // {2: 1}
///
/// assert!(!vc1.happens_before(&vc2)); // concurrent
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorClock {
    clocks: HashMap<u32, u64>,
}

impl VectorClock {
    /// Create a new vector clock with the given node ID initialized to 0.
    pub fn new(node_id: u32) -> Self {
        let mut clocks = HashMap::new();
        clocks.insert(node_id, 0);
        Self { clocks }
    }

    /// Increment the logical timestamp for the given node.
    pub fn tick(&mut self, node_id: u32) {
        *self.clocks.entry(node_id).or_insert(0) += 1;
    }

    /// Merge this clock with another, taking the maximum timestamp for each node.
    pub fn merge(&mut self, other: &VectorClock) {
        for (&node_id, &timestamp) in &other.clocks {
            let entry = self.clocks.entry(node_id).or_insert(0);
            *entry = (*entry).max(timestamp);
        }
    }

    /// Check if this clock happens-before the other clock.
    ///
    /// Returns true iff all entries in self are <= other and at least one is strictly <.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;

        // Check all entries in self
        for (&node_id, &self_ts) in &self.clocks {
            let other_ts = other.clocks.get(&node_id).copied().unwrap_or(0);
            if self_ts > other_ts {
                return false; // self has a greater timestamp, so not happens-before
            }
            if self_ts < other_ts {
                at_least_one_less = true;
            }
        }

        // Check if other has entries not in self (those count as "greater")
        for &node_id in other.clocks.keys() {
            if !self.clocks.contains_key(&node_id) {
                at_least_one_less = true;
            }
        }

        at_least_one_less
    }

    /// Get the timestamp for a specific node (returns 0 if not present).
    pub fn get(&self, node_id: u32) -> u64 {
        self.clocks.get(&node_id).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// LwwRegister
// ---------------------------------------------------------------------------

/// Last-Write-Wins register with vector clock timestamps.
///
/// An LWW register resolves conflicts by choosing the value with the
/// highest timestamp. If timestamps are equal, we use node_id as a
/// deterministic tie-breaker (higher node_id wins).
///
/// # Example
///
/// ```
/// use rvagent_subagents::crdt_merge::LwwRegister;
///
/// let reg1 = LwwRegister::new(b"value1".to_vec(), 10, 1);
/// let reg2 = LwwRegister::new(b"value2".to_vec(), 20, 2);
///
/// assert!(reg2.timestamp() > reg1.timestamp()); // reg2 wins
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LwwRegister<T> {
    value: T,
    timestamp: u64,
    node_id: u32,
}

impl<T> LwwRegister<T> {
    /// Create a new LWW register with the given value, timestamp, and node ID.
    pub fn new(value: T, timestamp: u64, node_id: u32) -> Self {
        Self {
            value,
            timestamp,
            node_id,
        }
    }

    /// Get a reference to the value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get the timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get the node ID.
    pub fn node_id(&self) -> u32 {
        self.node_id
    }

    /// Determine if this register should win over another in a merge.
    ///
    /// Returns true if self should be kept, false if other should win.
    pub fn should_win_over(&self, other: &Self) -> bool {
        if self.timestamp > other.timestamp {
            return true;
        }
        if self.timestamp < other.timestamp {
            return false;
        }
        // Tie-breaker: higher node_id wins (deterministic)
        self.node_id > other.node_id
    }
}

// ---------------------------------------------------------------------------
// CrdtState
// ---------------------------------------------------------------------------

/// CRDT-based state that can merge deterministically.
///
/// A CrdtState is a collection of LWW registers, each identified by a string key.
/// When two CrdtStates are merged, conflicts are resolved by choosing the
/// register with the highest timestamp (and node_id as tie-breaker).
///
/// # Example
///
/// ```
/// use rvagent_subagents::crdt_merge::CrdtState;
///
/// let mut state1 = CrdtState::new(1);
/// state1.set("key1", b"value1".to_vec());
///
/// let mut state2 = CrdtState::new(2);
/// state2.set("key1", b"value2".to_vec());
///
/// state1.merge(&state2);
/// // state1 now has the value with the highest timestamp
/// ```
#[derive(Clone, Debug)]
pub struct CrdtState {
    clock: VectorClock,
    node_id: u32,
    registers: HashMap<String, LwwRegister<Vec<u8>>>,
}

impl CrdtState {
    /// Create a new CRDT state with the given node ID.
    pub fn new(node_id: u32) -> Self {
        Self {
            clock: VectorClock::new(node_id),
            node_id,
            registers: HashMap::new(),
        }
    }

    /// Set a key-value pair, incrementing the local clock.
    pub fn set(&mut self, key: &str, value: Vec<u8>) {
        self.clock.tick(self.node_id);
        let timestamp = self.clock.get(self.node_id);
        let register = LwwRegister::new(value, timestamp, self.node_id);
        self.registers.insert(key.to_string(), register);
    }

    /// Get the value for a key (if present).
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.registers.get(key).map(|reg| reg.value().as_slice())
    }

    /// Get the vector clock for this state.
    pub fn clock(&self) -> &VectorClock {
        &self.clock
    }

    /// Get the node ID for this state.
    pub fn node_id(&self) -> u32 {
        self.node_id
    }

    /// Get all keys in this state.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.registers.keys()
    }

    /// Deterministic merge of another CRDT state into this one.
    ///
    /// For each key in `other`:
    /// - If we don't have the key, insert it
    /// - If we have the key, keep the register with the highest timestamp
    /// - If timestamps are equal, keep the register with the higher node_id
    ///
    /// The vector clocks are also merged (taking max timestamp per node).
    pub fn merge(&mut self, other: &CrdtState) {
        // Merge vector clocks
        self.clock.merge(&other.clock);

        // Merge registers
        for (key, other_reg) in &other.registers {
            match self.registers.get(key) {
                Some(existing_reg) => {
                    // Keep the register with the highest timestamp (or node_id as tie-breaker)
                    if !existing_reg.should_win_over(other_reg) {
                        self.registers.insert(key.clone(), other_reg.clone());
                    }
                }
                None => {
                    // Key doesn't exist in self, so insert it
                    self.registers.insert(key.clone(), other_reg.clone());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge utilities
// ---------------------------------------------------------------------------

/// Merge multiple subagent states into a parent state.
///
/// This is the primary API for merging parallel subagent results. It performs
/// a deterministic merge of all child states into the parent.
///
/// # Arguments
///
/// - `parent`: The parent agent's state (will be modified in-place)
/// - `results`: A vector of child states from parallel subagents
///
/// # Errors
///
/// - `MergeError::EmptyResultSet` if `results` is empty
/// - `MergeError::DuplicateNodeId` if any two states share the same node_id
///
/// # Example
///
/// ```
/// use rvagent_subagents::crdt_merge::{CrdtState, merge_subagent_results};
///
/// let mut parent = CrdtState::new(0);
/// let child1 = CrdtState::new(1);
/// let child2 = CrdtState::new(2);
///
/// merge_subagent_results(&mut parent, vec![child1, child2]).unwrap();
/// ```
pub fn merge_subagent_results(
    parent: &mut CrdtState,
    results: Vec<CrdtState>,
) -> Result<(), MergeError> {
    if results.is_empty() {
        return Err(MergeError::EmptyResultSet);
    }

    // Check for duplicate node IDs (including parent)
    let mut seen_ids = std::collections::HashSet::new();
    seen_ids.insert(parent.node_id);

    for state in &results {
        if !seen_ids.insert(state.node_id) {
            return Err(MergeError::DuplicateNodeId(state.node_id));
        }
    }

    // Merge each child state into the parent
    for state in results {
        parent.merge(&state);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_new() {
        let vc = VectorClock::new(1);
        assert_eq!(vc.get(1), 0);
        assert_eq!(vc.get(2), 0);
    }

    #[test]
    fn test_vector_clock_tick() {
        let mut vc = VectorClock::new(1);
        vc.tick(1);
        assert_eq!(vc.get(1), 1);
        vc.tick(1);
        assert_eq!(vc.get(1), 2);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut vc1 = VectorClock::new(1);
        vc1.tick(1);
        vc1.tick(1);

        let mut vc2 = VectorClock::new(2);
        vc2.tick(2);

        vc1.merge(&vc2);
        assert_eq!(vc1.get(1), 2);
        assert_eq!(vc1.get(2), 1);
    }

    #[test]
    fn test_vector_clock_happens_before() {
        let mut vc1 = VectorClock::new(1);
        vc1.tick(1);

        let mut vc2 = VectorClock::new(1);
        vc2.tick(1);
        vc2.tick(1);

        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut vc1 = VectorClock::new(1);
        vc1.tick(1);

        let mut vc2 = VectorClock::new(2);
        vc2.tick(2);

        // Neither happens-before the other (concurrent)
        assert!(!vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
    }

    #[test]
    fn test_lww_register_should_win_over() {
        let reg1 = LwwRegister::new(b"value1".to_vec(), 10, 1);
        let reg2 = LwwRegister::new(b"value2".to_vec(), 20, 2);

        assert!(!reg1.should_win_over(&reg2)); // reg2 has higher timestamp
        assert!(reg2.should_win_over(&reg1));
    }

    #[test]
    fn test_lww_register_tie_breaker() {
        let reg1 = LwwRegister::new(b"value1".to_vec(), 10, 1);
        let reg2 = LwwRegister::new(b"value2".to_vec(), 10, 2);

        assert!(!reg1.should_win_over(&reg2)); // same timestamp, reg2 has higher node_id
        assert!(reg2.should_win_over(&reg1));
    }

    #[test]
    fn test_crdt_state_set_get() {
        let mut state = CrdtState::new(1);
        state.set("key1", b"value1".to_vec());

        assert_eq!(state.get("key1"), Some(b"value1".as_slice()));
        assert_eq!(state.get("key2"), None);
    }

    #[test]
    fn test_crdt_state_merge_no_conflict() {
        let mut state1 = CrdtState::new(1);
        state1.set("key1", b"value1".to_vec());

        let mut state2 = CrdtState::new(2);
        state2.set("key2", b"value2".to_vec());

        state1.merge(&state2);

        assert_eq!(state1.get("key1"), Some(b"value1".as_slice()));
        assert_eq!(state1.get("key2"), Some(b"value2".as_slice()));
    }

    #[test]
    fn test_crdt_state_merge_with_conflict() {
        let mut state1 = CrdtState::new(1);
        state1.set("key1", b"value1".to_vec());

        // Simulate state2 writing after state1
        let mut state2 = CrdtState::new(2);
        state2.clock.merge(&state1.clock);
        state2.set("key1", b"value2".to_vec());

        state1.merge(&state2);

        // state2's value should win (higher timestamp)
        assert_eq!(state1.get("key1"), Some(b"value2".as_slice()));
    }

    #[test]
    fn test_crdt_state_merge_concurrent_writes() {
        // Simulate two concurrent writes to the same key
        let mut state1 = CrdtState::new(1);
        state1.set("key1", b"value1".to_vec());

        let mut state2 = CrdtState::new(2);
        state2.set("key1", b"value2".to_vec());

        state1.merge(&state2);

        // state2's value should win (same timestamp, higher node_id)
        assert_eq!(state1.get("key1"), Some(b"value2".as_slice()));
    }

    #[test]
    fn test_merge_subagent_results_empty() {
        let mut parent = CrdtState::new(0);
        let result = merge_subagent_results(&mut parent, vec![]);
        assert!(matches!(result, Err(MergeError::EmptyResultSet)));
    }

    #[test]
    fn test_merge_subagent_results_duplicate_node_id() {
        let mut parent = CrdtState::new(0);
        let child1 = CrdtState::new(1);
        let child2 = CrdtState::new(1); // Duplicate

        let result = merge_subagent_results(&mut parent, vec![child1, child2]);
        assert!(matches!(result, Err(MergeError::DuplicateNodeId(1))));
    }

    #[test]
    fn test_merge_subagent_results_success() {
        let mut parent = CrdtState::new(0);
        parent.set("parent_key", b"parent_value".to_vec());

        let mut child1 = CrdtState::new(1);
        child1.set("child1_key", b"child1_value".to_vec());
        child1.set("shared_key", b"child1_shared".to_vec());

        let mut child2 = CrdtState::new(2);
        child2.set("child2_key", b"child2_value".to_vec());
        child2.set("shared_key", b"child2_shared".to_vec());

        merge_subagent_results(&mut parent, vec![child1, child2]).unwrap();

        assert_eq!(parent.get("parent_key"), Some(b"parent_value".as_slice()));
        assert_eq!(parent.get("child1_key"), Some(b"child1_value".as_slice()));
        assert_eq!(parent.get("child2_key"), Some(b"child2_value".as_slice()));

        // For shared_key, child2 should win (higher node_id, concurrent writes)
        assert_eq!(parent.get("shared_key"), Some(b"child2_shared".as_slice()));
    }

    #[test]
    fn test_merge_three_way() {
        let mut parent = CrdtState::new(0);
        parent.set("key1", b"parent".to_vec());

        let mut child1 = CrdtState::new(1);
        child1.clock.merge(&parent.clock);
        child1.set("key1", b"child1".to_vec());

        let mut child2 = CrdtState::new(2);
        child2.clock.merge(&parent.clock);
        child2.set("key1", b"child2".to_vec());

        let mut child3 = CrdtState::new(3);
        child3.clock.merge(&parent.clock);
        child3.set("key1", b"child3".to_vec());

        merge_subagent_results(&mut parent, vec![child1, child2, child3]).unwrap();

        // child3 should win (highest node_id, all concurrent)
        assert_eq!(parent.get("key1"), Some(b"child3".as_slice()));
    }

    #[test]
    fn test_merge_causal_ordering() {
        let mut state0 = CrdtState::new(0);
        state0.set("key1", b"v0".to_vec());

        // state1 observes state0, then writes
        let mut state1 = CrdtState::new(1);
        state1.clock.merge(&state0.clock);
        state1.set("key1", b"v1".to_vec());

        // state2 observes state1, then writes
        let mut state2 = CrdtState::new(2);
        state2.clock.merge(&state1.clock);
        state2.set("key1", b"v2".to_vec());

        state0.merge(&state1);
        state0.merge(&state2);

        // state2's write happened after state1's, so it should win
        assert_eq!(state0.get("key1"), Some(b"v2".as_slice()));
    }

    #[test]
    fn test_merge_preserves_parent_keys() {
        let mut parent = CrdtState::new(0);
        parent.set("parent_only", b"parent_value".to_vec());

        let mut child = CrdtState::new(1);
        child.set("child_only", b"child_value".to_vec());

        merge_subagent_results(&mut parent, vec![child]).unwrap();

        assert_eq!(parent.get("parent_only"), Some(b"parent_value".as_slice()));
        assert_eq!(parent.get("child_only"), Some(b"child_value".as_slice()));
    }

    #[test]
    fn test_merge_many_keys() {
        let mut parent = CrdtState::new(0);
        for i in 0..100 {
            parent.set(&format!("key{}", i), format!("parent_{}", i).into_bytes());
        }

        let mut child1 = CrdtState::new(1);
        for i in 50..150 {
            child1.set(&format!("key{}", i), format!("child1_{}", i).into_bytes());
        }

        let mut child2 = CrdtState::new(2);
        for i in 100..200 {
            child2.set(&format!("key{}", i), format!("child2_{}", i).into_bytes());
        }

        merge_subagent_results(&mut parent, vec![child1, child2]).unwrap();

        // Parent's exclusive keys (0-49) should remain
        assert_eq!(parent.get("key0"), Some(b"parent_0".as_slice()));

        // Child1's exclusive keys (150-199 are child2's, so check 50-99 that overlap)
        // For concurrent writes, child2 wins due to higher node_id
        assert_eq!(parent.get("key150"), Some(b"child2_150".as_slice()));
    }
}
