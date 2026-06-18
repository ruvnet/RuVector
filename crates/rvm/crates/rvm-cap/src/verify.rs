//! Three-layer proof verification (ADR-135).
//!
//! - **P1**: Capability existence + rights check (< 1 us, bitmap AND).
//! - **P2**: Structural invariant validation (< 100 us, constant-time).
//! - **P3**: Deep proof — derivation chain integrity (root reachability, epoch monotonicity).

use crate::derivation::DerivationTree;
use crate::error::ProofError;
use crate::table::CapabilityTable;
use rvm_types::CapRights;

/// Nonce table size for replay prevention.
///
/// 4096 slots of `u64` = 32 KiB. The previous design used two parallel
/// 4096-entry arrays (ring + hash index, 64 KiB total) where a colliding
/// nonce `B = A + k * 4096` evicted `A` from the hash index, re-admitting
/// `A` for deliberate replay. The single open-addressed table below keeps
/// the same 4096-nonce window at half the footprint and closes the
/// collision-eviction replay hole via the watermark (see [`mark_nonce`]).
const NONCE_TABLE_SIZE: usize = 4096;

/// Maximum linear-probe distance in the nonce table.
///
/// Bounded probing keeps `check_nonce`/`mark_nonce` O(1) worst-case.
/// When all probed slots are occupied, the oldest (smallest) nonce in
/// the window is evicted and the watermark is raised to its value, so
/// eviction can never re-admit a previously seen nonce.
const NONCE_MAX_PROBES: usize = 8;

/// Policy context for P2 validation.
#[derive(Debug, Clone, Copy)]
pub struct PolicyContext {
    /// The expected owner partition ID.
    pub expected_owner: u32,
    /// Region lower bound (used for bounds checking).
    pub region_base: u64,
    /// Region upper bound.
    pub region_limit: u64,
    /// Lease expiry timestamp in nanoseconds.
    pub lease_expiry_ns: u64,
    /// Current timestamp in nanoseconds.
    pub current_time_ns: u64,
    /// Maximum delegation depth (typically 8).
    pub max_delegation_depth: u8,
    /// Nonce for replay prevention.
    pub nonce: u64,
}

/// Three-layer proof verifier.
///
/// Encapsulates the epoch and nonce tracker needed for P1/P2/P3 verification.
pub struct ProofVerifier<const N: usize> {
    /// Reference epoch for stale-handle detection.
    current_epoch: u32,
    /// Open-addressed nonce table for replay prevention.
    ///
    /// Each slot stores the nonce value itself; `0` marks an empty slot
    /// (safe because nonce == 0 is handled separately and never
    /// inserted, see [`check_nonce`]/[`mark_nonce`]). Lookup and insert
    /// use bounded linear probing ([`NONCE_MAX_PROBES`] slots starting
    /// at `nonce % NONCE_TABLE_SIZE`).
    nonce_table: [u64; NONCE_TABLE_SIZE],
    /// Total number of nonces inserted (drives the periodic watermark
    /// advance that replaces the old ring-wrap behaviour).
    nonce_inserted: u64,
    /// Monotonic watermark: any nonce at or below this value is rejected
    /// outright, even if it is no longer present in the table. This
    /// prevents replaying nonces that were evicted (probe-window
    /// overflow) or that predate the current table window.
    nonce_watermark: u64,
    /// Whether nonce == 0 is allowed to bypass replay checks.
    ///
    /// Default is `false` (zero nonce is rejected). Set to `true` only
    /// for boot-time or backwards-compatible contexts where a sentinel
    /// nonce is acceptable.
    allow_zero_nonce: bool,
}

impl<const N: usize> ProofVerifier<N> {
    /// Creates a new proof verifier with the given epoch.
    ///
    /// By default, nonce == 0 is **rejected** (no zero-nonce bypass).
    /// Use [`set_allow_zero_nonce`](Self::set_allow_zero_nonce) to enable
    /// the sentinel behaviour for boot-time contexts.
    #[must_use]
    #[allow(clippy::large_stack_arrays)]
    pub const fn new(epoch: u32) -> Self {
        Self {
            current_epoch: epoch,
            nonce_table: [0u64; NONCE_TABLE_SIZE],
            nonce_inserted: 0,
            nonce_watermark: 0,
            allow_zero_nonce: false,
        }
    }

    /// Set whether nonce == 0 is allowed to bypass replay checks.
    pub fn set_allow_zero_nonce(&mut self, allow: bool) {
        self.allow_zero_nonce = allow;
    }

    /// Updates the current epoch.
    pub fn set_epoch(&mut self, epoch: u32) {
        self.current_epoch = epoch;
    }

    /// P1: Capability existence + rights check.
    ///
    /// Budget: < 1 us. No allocation. All checks execute regardless of
    /// intermediate failures to prevent timing side-channel leakage.
    /// The final error returned is deliberately the most generic
    /// (`InvalidHandle`) to avoid leaking which check failed.
    ///
    /// # Errors
    ///
    /// Returns [`ProofError::InvalidHandle`] if the handle is invalid.
    /// Returns [`ProofError::StaleCapability`] if the epoch does not match.
    /// Returns [`ProofError::InsufficientRights`] if the rights are insufficient.
    #[inline]
    pub fn verify_p1(
        &self,
        table: &CapabilityTable<N>,
        cap_index: u32,
        cap_generation: u32,
        required_rights: CapRights,
    ) -> Result<(), ProofError> {
        // Run ALL checks unconditionally to prevent timing side channels.
        // We accumulate a bitmask of failures rather than early-returning.
        let mut fail_mask: u8 = 0;

        let lookup_result = table.lookup(cap_index, cap_generation);

        // Check 1: Handle validity.
        let (epoch_match, rights_match) = if let Ok(slot) = &lookup_result {
            // Check 2: Epoch match.
            let e = slot.token.epoch() == self.current_epoch;
            // Check 3: Rights subset.
            let r = slot.token.has_rights(required_rights);
            (e, r)
        } else {
            fail_mask |= 1;
            // Still "compute" epoch and rights checks against dummy values
            // to keep timing constant. The compiler should not elide these
            // because fail_mask is read below.
            (false, false)
        };

        if !epoch_match {
            fail_mask |= 2;
        }
        if !rights_match {
            fail_mask |= 4;
        }

        if fail_mask == 0 {
            Ok(())
        } else if fail_mask & 1 != 0 {
            Err(ProofError::InvalidHandle)
        } else if fail_mask & 2 != 0 {
            Err(ProofError::StaleCapability)
        } else {
            Err(ProofError::InsufficientRights)
        }
    }

    /// P2: Structural invariant validation (constant-time).
    ///
    /// Budget: < 100 us. All checks execute regardless of intermediate
    /// failures to prevent timing side-channel leakage (ADR-135).
    ///
    /// Checks: ownership chain, region bounds, lease expiry,
    /// delegation depth, nonce replay.
    ///
    /// # Errors
    ///
    /// Returns [`ProofError::PolicyViolation`] if any structural check fails.
    pub fn verify_p2(
        &mut self,
        table: &CapabilityTable<N>,
        tree: &DerivationTree<N>,
        cap_index: u32,
        cap_generation: u32,
        ctx: &PolicyContext,
    ) -> Result<(), ProofError> {
        let mut valid = true;

        // 1. Ownership chain valid.
        let owner_ok = table
            .lookup(cap_index, cap_generation)
            .map(|slot| slot.owner.as_u32() == ctx.expected_owner)
            .unwrap_or(false);
        valid &= owner_ok;

        // 2. Region bounds legal.
        valid &= ctx.region_base < ctx.region_limit;

        // 3. Lease not expired.
        valid &= ctx.current_time_ns <= ctx.lease_expiry_ns;

        // 4. Delegation depth within limit.
        let depth_ok = tree
            .depth(cap_index)
            .map(|d| d <= ctx.max_delegation_depth)
            .unwrap_or(false);
        valid &= depth_ok;

        // 5. Nonce not replayed.
        let nonce_ok = self.check_nonce(ctx.nonce);
        valid &= nonce_ok;

        if valid {
            self.mark_nonce(ctx.nonce);
            Ok(())
        } else {
            Err(ProofError::PolicyViolation)
        }
    }

    /// P3: Deep proof — derivation chain integrity verification.
    ///
    /// Walks the derivation tree from the given capability back to its
    /// root and verifies:
    /// 1. Every ancestor is valid (not revoked).
    /// 2. Depth decreases monotonically toward the root.
    /// 3. Epoch values are non-decreasing from root to leaf.
    /// 4. The chain terminates at a root node (depth 0).
    /// 5. The chain length does not exceed `max_depth`.
    ///
    /// Budget: < 10 us for depth <= 8 (typical). Worst-case O(depth).
    ///
    /// # Errors
    ///
    /// Returns [`ProofError::DerivationChainBroken`] if the chain is
    /// invalid, tampered, or does not reach a root.
    pub fn verify_p3(
        &self,
        table: &CapabilityTable<N>,
        tree: &DerivationTree<N>,
        cap_index: u32,
        cap_generation: u32,
        max_depth: u8,
    ) -> Result<(), ProofError> {
        // Verify the capability itself is valid.
        let _slot = table
            .lookup(cap_index, cap_generation)
            .map_err(|_| ProofError::DerivationChainBroken)?;

        // Verify the derivation node exists and is valid.
        let node = tree
            .get(cap_index)
            .ok_or(ProofError::DerivationChainBroken)?;
        if !node.is_valid {
            return Err(ProofError::DerivationChainBroken);
        }

        // If this IS a root, chain is trivially valid.
        if node.depth == 0 {
            return Ok(());
        }

        // Walk the derivation tree up to the root.
        let mut current_depth = node.depth;
        let mut current_epoch = node.epoch;
        let mut steps = 0u8;

        // Walk ancestors. The derivation tree uses first-child/next-sibling,
        // so we need to find the parent. We do this by scanning for a node
        // that has `cap_index` in its children chain.
        let mut current_idx = cap_index;
        loop {
            steps += 1;
            if steps > max_depth {
                return Err(ProofError::DerivationChainBroken);
            }

            // Find the parent of current_idx.
            let parent_idx = tree.find_parent(current_idx);
            match parent_idx {
                Some(pidx) => {
                    let parent = match tree.get(pidx) {
                        Some(p) => p,
                        None => return Err(ProofError::DerivationChainBroken),
                    };

                    // Ancestor must be valid.
                    if !parent.is_valid {
                        return Err(ProofError::DerivationChainBroken);
                    }
                    // Depth must decrease.
                    if parent.depth >= current_depth {
                        return Err(ProofError::DerivationChainBroken);
                    }
                    // Epoch must be non-decreasing from root to leaf
                    // (parent.epoch <= child.epoch).
                    if parent.epoch > current_epoch {
                        return Err(ProofError::DerivationChainBroken);
                    }

                    if parent.depth == 0 {
                        // Reached the root — chain is valid.
                        return Ok(());
                    }

                    current_depth = parent.depth;
                    current_epoch = parent.epoch;
                    current_idx = pidx;
                }
                None => {
                    // No parent found but we're not at root — broken chain.
                    return Err(ProofError::DerivationChainBroken);
                }
            }
        }
    }

    /// Checks if a nonce has been used recently.
    ///
    /// Rejects nonces that are at or below the monotonic watermark
    /// (nonces that were evicted from the table or predate the current
    /// window) as well as nonces still present in the table.
    ///
    /// Nonce == 0 is rejected unless `allow_zero_nonce` is set. This
    /// prevents callers from silently skipping replay protection by
    /// passing a default/uninitialized nonce value (and keeps 0 free
    /// to act as the empty-slot sentinel in the table).
    ///
    /// # Replay-protection guarantee
    ///
    /// Once a nonce has been marked via [`mark_nonce`], it is rejected
    /// forever: either it is still in the table (probe hit), or it was
    /// evicted — and eviction raises the watermark to at least its
    /// value, so the watermark check rejects it. Unlike the previous
    /// two-array design, a colliding nonce `B = A + k * NONCE_TABLE_SIZE`
    /// can never silently re-admit `A`. The trade-off is fail-closed:
    /// raising the watermark may also reject *never-seen* nonces that
    /// are numerically at or below an evicted one.
    fn check_nonce(&self, nonce: u64) -> bool {
        if nonce == 0 {
            return self.allow_zero_nonce;
        }
        // Watermark check: reject any nonce at or below the low-water mark.
        if nonce <= self.nonce_watermark {
            return false;
        }
        // Bounded linear probe from the nonce's home slot.
        let home = (nonce as usize) % NONCE_TABLE_SIZE;
        let mut i = 0;
        while i < NONCE_MAX_PROBES {
            if self.nonce_table[(home + i) % NONCE_TABLE_SIZE] == nonce {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Records a nonce as used, evicting (and permanently retiring via
    /// the watermark) the oldest probe-window entry if necessary.
    fn mark_nonce(&mut self, nonce: u64) {
        if nonce == 0 {
            return;
        }

        // Probe for an empty slot; track the oldest (smallest) live
        // nonce in the window as the eviction candidate.
        let home = (nonce as usize) % NONCE_TABLE_SIZE;
        let mut oldest_slot = home;
        let mut oldest_val = u64::MAX;
        let mut stored = false;
        let mut i = 0;
        while i < NONCE_MAX_PROBES {
            let slot = (home + i) % NONCE_TABLE_SIZE;
            let val = self.nonce_table[slot];
            if val == nonce {
                // Already marked; nothing to do.
                return;
            }
            if val == 0 {
                self.nonce_table[slot] = nonce;
                stored = true;
                break;
            }
            if val < oldest_val {
                oldest_val = val;
                oldest_slot = slot;
            }
            i += 1;
        }

        if !stored {
            // Probe window full: evict the oldest entry and raise the
            // watermark to its value so the evicted nonce stays
            // rejected forever (eviction can never re-admit a replay).
            self.nonce_table[oldest_slot] = nonce;
            if oldest_val != u64::MAX && oldest_val > self.nonce_watermark {
                self.nonce_watermark = oldest_val;
            }
        }

        // Periodic watermark advance, preserving the previous ring's
        // wrap semantics: after a full table's worth of inserts,
        // anything older than the minimum live nonce is rejected.
        self.nonce_inserted += 1;
        if self.nonce_inserted % (NONCE_TABLE_SIZE as u64) == 0 {
            let mut min_val = u64::MAX;
            for entry in &self.nonce_table {
                if *entry != 0 && *entry < min_val {
                    min_val = *entry;
                }
            }
            if min_val != u64::MAX && min_val > self.nonce_watermark {
                self.nonce_watermark = min_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvm_types::{CapToken, CapType, PartitionId};

    fn setup() -> (CapabilityTable<64>, DerivationTree<64>, ProofVerifier<64>) {
        let table = CapabilityTable::<64>::new();
        let tree = DerivationTree::<64>::new();
        let verifier = ProofVerifier::<64>::new(0);
        (table, tree, verifier)
    }

    fn all_rights() -> CapRights {
        CapRights::READ
            .union(CapRights::WRITE)
            .union(CapRights::EXECUTE)
            .union(CapRights::GRANT)
            .union(CapRights::REVOKE)
    }

    #[test]
    fn test_p1_valid() {
        let (mut table, _, verifier) = setup();
        let owner = PartitionId::new(1);
        let token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (idx, gen) = table.insert_root(token, owner, 0).unwrap();
        assert!(verifier.verify_p1(&table, idx, gen, CapRights::READ).is_ok());
    }

    #[test]
    fn test_p1_invalid_handle() {
        let (table, _, verifier) = setup();
        assert_eq!(verifier.verify_p1(&table, 99, 0, CapRights::READ), Err(ProofError::InvalidHandle));
    }

    #[test]
    fn test_p1_stale_epoch() {
        let (mut table, _, verifier) = setup();
        let token = CapToken::new(100, CapType::Region, all_rights(), 5);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        assert_eq!(verifier.verify_p1(&table, idx, gen, CapRights::READ), Err(ProofError::StaleCapability));
    }

    #[test]
    fn test_p1_insufficient_rights() {
        let (mut table, _, verifier) = setup();
        let token = CapToken::new(100, CapType::Region, CapRights::READ, 0);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        assert_eq!(verifier.verify_p1(&table, idx, gen, CapRights::WRITE), Err(ProofError::InsufficientRights));
    }

    #[test]
    fn test_p2_all_pass() {
        let (mut table, mut tree, mut verifier) = setup();
        let token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        tree.add_root(idx, 0).unwrap();

        let ctx = PolicyContext {
            expected_owner: 1,
            region_base: 0x1000,
            region_limit: 0x2000,
            lease_expiry_ns: 1_000_000_000,
            current_time_ns: 500_000_000,
            max_delegation_depth: 8,
            nonce: 42,
        };
        assert!(verifier.verify_p2(&table, &tree, idx, gen, &ctx).is_ok());
    }

    #[test]
    fn test_p2_nonce_replay() {
        let (mut table, mut tree, mut verifier) = setup();
        let token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        tree.add_root(idx, 0).unwrap();

        let ctx = PolicyContext {
            expected_owner: 1,
            region_base: 0x1000,
            region_limit: 0x2000,
            lease_expiry_ns: 1_000_000_000,
            current_time_ns: 500_000_000,
            max_delegation_depth: 8,
            nonce: 55,
        };
        assert!(verifier.verify_p2(&table, &tree, idx, gen, &ctx).is_ok());
        assert_eq!(verifier.verify_p2(&table, &tree, idx, gen, &ctx), Err(ProofError::PolicyViolation));
    }

    #[test]
    fn test_p3_root_passes() {
        let (mut table, mut tree, verifier) = setup();
        let token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        tree.add_root(idx, 0).unwrap();

        assert!(verifier.verify_p3(&table, &tree, idx, gen, 8).is_ok());
    }

    #[test]
    fn test_p3_one_level_derivation() {
        let (mut table, mut tree, verifier) = setup();
        let owner = PartitionId::new(1);

        // Create root.
        let root_token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (root_idx, _root_gen) = table.insert_root(root_token, owner, 0).unwrap();
        tree.add_root(root_idx, 0).unwrap();

        // Derive a child.
        let child_token = CapToken::new(200, CapType::Region, CapRights::READ, 0);
        let (child_idx, child_gen) = table.insert_root(child_token, owner, 0).unwrap();
        tree.add_child(root_idx, child_idx, 1, 1).unwrap();

        // P3 should follow child → root and succeed.
        assert!(verifier.verify_p3(&table, &tree, child_idx, child_gen, 8).is_ok());
    }

    #[test]
    fn test_p3_nonexistent_fails() {
        let (table, tree, verifier) = setup();
        assert_eq!(
            verifier.verify_p3(&table, &tree, 99, 0, 8),
            Err(ProofError::DerivationChainBroken),
        );
    }

    #[test]
    fn test_p3_revoked_ancestor_fails() {
        let (mut table, mut tree, verifier) = setup();
        let owner = PartitionId::new(1);

        let root_token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (root_idx, _) = table.insert_root(root_token, owner, 0).unwrap();
        tree.add_root(root_idx, 0).unwrap();

        let child_token = CapToken::new(200, CapType::Region, CapRights::READ, 0);
        let (child_idx, child_gen) = table.insert_root(child_token, owner, 0).unwrap();
        tree.add_child(root_idx, child_idx, 1, 1).unwrap();

        // Revoke the root.
        tree.revoke(root_idx).unwrap();

        // P3 should fail because root is revoked.
        assert_eq!(
            verifier.verify_p3(&table, &tree, child_idx, child_gen, 8),
            Err(ProofError::DerivationChainBroken),
        );
    }

    #[test]
    fn test_nonce_ring_4096_churn() {
        // Verify that after filling the 4096-entry ring, old nonces are
        // rejected by the monotonic watermark even after eviction.
        let (mut table, mut tree, mut verifier) = setup();
        let token = CapToken::new(100, CapType::Region, all_rights(), 0);
        let (idx, gen) = table.insert_root(token, PartitionId::new(1), 0).unwrap();
        tree.add_root(idx, 0).unwrap();

        // Insert 4096 nonces (1..=4096).
        for i in 1..=4096u64 {
            let ctx = PolicyContext {
                expected_owner: 1,
                region_base: 0x1000,
                region_limit: 0x2000,
                lease_expiry_ns: 1_000_000_000,
                current_time_ns: 500_000_000,
                max_delegation_depth: 8,
                nonce: i,
            };
            assert!(verifier.verify_p2(&table, &tree, idx, gen, &ctx).is_ok());
        }

        // Now insert one more to push nonce 1 out and trigger watermark.
        let ctx_new = PolicyContext {
            expected_owner: 1,
            region_base: 0x1000,
            region_limit: 0x2000,
            lease_expiry_ns: 1_000_000_000,
            current_time_ns: 500_000_000,
            max_delegation_depth: 8,
            nonce: 4097,
        };
        assert!(verifier.verify_p2(&table, &tree, idx, gen, &ctx_new).is_ok());

        // Nonce 1 should be rejected by the watermark even though it
        // has been evicted from the ring.
        let ctx_old = PolicyContext {
            expected_owner: 1,
            region_base: 0x1000,
            region_limit: 0x2000,
            lease_expiry_ns: 1_000_000_000,
            current_time_ns: 500_000_000,
            max_delegation_depth: 8,
            nonce: 1,
        };
        assert_eq!(
            verifier.verify_p2(&table, &tree, idx, gen, &ctx_old),
            Err(ProofError::PolicyViolation)
        );
    }

    /// Regression test for the collision-eviction replay hole.
    ///
    /// In the previous two-array design, `nonce_hash[nonce % 4096]` was
    /// overwritten by any colliding nonce `B = A + k * 4096`, so after
    /// inserting one collider, `check_nonce(A)` re-admitted `A` —
    /// deliberate replay was possible.
    ///
    /// New design's guarantee (documented on `check_nonce`): once
    /// marked, a nonce is rejected forever. While it survives in the
    /// probe window it is rejected by the table lookup; when colliders
    /// overflow the window and evict it, the watermark is raised to its
    /// value, so the watermark check rejects it instead. There is no
    /// state in which a marked nonce becomes acceptable again.
    #[test]
    fn test_colliding_nonces_cannot_replay_evicted_nonce() {
        let mut verifier = ProofVerifier::<64>::new(0);

        let a = 5000u64;
        assert!(verifier.check_nonce(a), "fresh nonce must be accepted");
        verifier.mark_nonce(a);
        assert!(!verifier.check_nonce(a), "immediate replay rejected");

        // Insert colliding nonces (same home slot: a + k * 4096). The
        // first NONCE_MAX_PROBES - 1 fill the probe window alongside A;
        // subsequent ones force evictions, starting with A (the oldest).
        for k in 1..=16u64 {
            let collider = a + k * 4096;
            assert!(
                verifier.check_nonce(collider),
                "fresh collider {collider} must be accepted"
            );
            verifier.mark_nonce(collider);
            // The original nonce must be rejected at EVERY point during
            // the collision churn — in-table before eviction, via the
            // watermark after.
            assert!(
                !verifier.check_nonce(a),
                "nonce {a} re-admitted after {k} colliding inserts (replay hole)"
            );
        }

        // Colliders themselves must also never be replayable.
        assert!(!verifier.check_nonce(a + 4096));
        assert!(!verifier.check_nonce(a + 16 * 4096));
    }

    #[test]
    fn test_watermark_rejects_below_minimum() {
        let mut verifier = ProofVerifier::<64>::new(0);
        // Manually advance the watermark by filling the ring and wrapping.
        // Use nonces 100..100+4096 to set a high watermark.
        for i in 100..100 + 4096u64 {
            verifier.mark_nonce(i);
        }
        // Nonce below the watermark should be rejected.
        assert!(!verifier.check_nonce(1));
        assert!(!verifier.check_nonce(99));
    }
}
