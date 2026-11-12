//! Independent, periodic signing of `index_state_root`, decoupled from any
//! specific query or receipt.
//!
//! ADR-304's unsigned receipts and ADR-340's signed receipt roots both
//! authenticate *what a specific query returned*. Neither answers a simpler
//! question an auditor holding no receipt at all may still want to ask: "was
//! the index ever attested to be in the state committed by root `R`, and how
//! far can I trust that without replaying the entire write history?" This
//! module answers that by having the index owner periodically sign its own
//! `index_state_root` (the write-chain head exposed by
//! `ruvector_proof_gate::WriteGate::chain_root`), independent of query
//! traffic — the third Open Question named in ADR-340.
//!
//! Signing on every write (`interval_writes = 1`) gives zero staleness but
//! costs one signature per write. Signing every `W` writes bounds staleness
//! to `W - 1` writes but amortizes the signing cost by roughly `W`. This
//! module makes that tradeoff explicit and measurable; it does not pick a
//! default for production.
//!
//! An anchor authenticates *that a state root was attested*, not that the
//! index behind it is honest, complete, or still reachable — the same
//! caveats as [`crate::signing`] apply. It also does not replace
//! `HashChainGate::verify_integrity`'s O(n) full re-derivation: an anchor is
//! an O(1) checkpoint an auditor can trust without holding the full write
//! history, not a substitute for full-history integrity when that history is
//! available.

use crate::signing::{
    verify_root, AnchorContext, AnchorError, AnchorPurpose, Issuer, SignedRoot, VerifiedRoot,
};
use ed25519_dalek::VerifyingKey;

/// How often to anchor: every `interval_writes` admitted writes.
/// `interval_writes == 1` anchors on every write (zero staleness, maximum
/// signing cost); larger values bound staleness to `interval_writes - 1`
/// writes while amortizing signing cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateAnchorPolicy {
    interval_writes: u64,
}

impl StateAnchorPolicy {
    /// Fails closed on a zero interval rather than panicking or silently
    /// treating it as "anchor every write" — a caller-supplied policy value
    /// is untrusted input to this crate's API surface.
    pub fn new(interval_writes: u64) -> Result<Self, AnchorError> {
        if interval_writes == 0 {
            return Err(AnchorError::InvalidInterval);
        }
        Ok(Self { interval_writes })
    }

    pub const fn interval_writes(&self) -> u64 {
        self.interval_writes
    }
}

/// One anchored checkpoint: the signed `index_state_root` and the write
/// count at which it was taken.
#[derive(Clone, Copy, Debug)]
pub struct StateAnchor {
    pub write_count: u64,
    pub signed_root: SignedRoot,
}

/// An append-only, in-process log of periodic state anchors. Not a
/// persistence layer — a real deployment would durably store each
/// [`StateAnchor`] as it is produced (e.g. via a `ruFlo` periodic-anchoring
/// workflow). This type models the anchoring policy and the auditor-facing
/// queries over the resulting checkpoints, generically over whatever
/// `index_state_root` a caller's `WriteGate` produces.
pub struct StateAnchorLog {
    policy: StateAnchorPolicy,
    anchors: Vec<StateAnchor>,
}

impl StateAnchorLog {
    pub fn new(policy: StateAnchorPolicy) -> Self {
        Self {
            policy,
            anchors: Vec::new(),
        }
    }

    pub fn policy(&self) -> StateAnchorPolicy {
        self.policy
    }

    pub fn anchors(&self) -> &[StateAnchor] {
        &self.anchors
    }

    /// Call after every write with the gate's current `chain_root()` and
    /// `len()`. Anchors (signs `index_state_root`) only when `write_count`
    /// lands on an interval boundary; returns the new anchor when one was
    /// taken, `None` otherwise. `write_count == 0` never anchors — there is
    /// no state yet to attest to.
    pub fn observe_write(
        &mut self,
        issuer: &Issuer,
        scope_hash: [u8; 32],
        index_state_root: [u8; 32],
        write_count: u64,
        issued_at_unix_ms: u64,
    ) -> Option<StateAnchor> {
        if write_count == 0 || write_count % self.policy.interval_writes != 0 {
            return None;
        }
        let context = AnchorContext::new(AnchorPurpose::StateAnchor, scope_hash);
        let signed_root = issuer.sign_root(context, index_state_root, issued_at_unix_ms);
        let anchor = StateAnchor {
            write_count,
            signed_root,
        };
        self.anchors.push(anchor);
        Some(anchor)
    }

    /// The most recent anchor at or before `write_count`, if any. Anchors
    /// are appended in nondecreasing `write_count` order by construction
    /// (every call site advances `write_count` monotonically), so a reverse
    /// scan finds it in O(anchors since the match), not O(total writes).
    pub fn latest_at_or_before(&self, write_count: u64) -> Option<&StateAnchor> {
        self.anchors
            .iter()
            .rev()
            .find(|a| a.write_count <= write_count)
    }

    /// Writes since the most recent anchor at or before `write_count`. Under
    /// a correctly operating log this never exceeds `interval_writes - 1`
    /// once the first anchor has landed; this is what a real deployment
    /// would monitor to detect a stalled anchoring job.
    pub fn staleness_at(&self, write_count: u64) -> u64 {
        match self.latest_at_or_before(write_count) {
            Some(a) => write_count - a.write_count,
            None => write_count,
        }
    }
}

/// O(1) audit: verify that `claimed_root` was validly anchored, without
/// access to any query receipt or the write history itself — just the
/// signer's public key, the expected deployment scope, and one
/// [`StateAnchor`]. Returns `None` on any mismatch (wrong key, wrong scope,
/// wrong purpose, tampered root, or tampered signature).
pub fn verify_state_anchor(
    vk: &VerifyingKey,
    scope_hash: [u8; 32],
    claimed_root: [u8; 32],
    anchor: &StateAnchor,
) -> Option<VerifiedRoot> {
    let context = AnchorContext::new(AnchorPurpose::StateAnchor, scope_hash);
    verify_root(vk, context, &anchor.signed_root).filter(|v| v.root() == claimed_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signing::AnchorPurpose;

    const SCOPE: [u8; 32] = [3u8; 32];
    const ISSUED_AT: u64 = 1_788_134_400_000;

    fn fake_root(seed: u8) -> [u8; 32] {
        let mut root = [0u8; 32];
        root[0] = seed;
        root
    }

    #[test]
    fn zero_interval_is_rejected_without_panicking() {
        assert_eq!(
            StateAnchorPolicy::new(0).unwrap_err(),
            AnchorError::InvalidInterval
        );
    }

    #[test]
    fn interval_one_anchors_every_write_with_zero_staleness() {
        let issuer = Issuer::generate();
        let policy = StateAnchorPolicy::new(1).unwrap();
        let mut log = StateAnchorLog::new(policy);
        for w in 1u64..=20 {
            let anchored = log.observe_write(&issuer, SCOPE, fake_root(w as u8), w, ISSUED_AT);
            assert!(anchored.is_some(), "interval=1 must anchor every write");
            assert_eq!(log.staleness_at(w), 0);
        }
        assert_eq!(log.anchors().len(), 20);
    }

    #[test]
    fn periodic_interval_bounds_staleness_to_interval_minus_one() {
        let issuer = Issuer::generate();
        let interval = 8u64;
        let policy = StateAnchorPolicy::new(interval).unwrap();
        let mut log = StateAnchorLog::new(policy);
        let n = 100u64;
        let mut max_staleness = 0u64;
        let mut anchors_taken = 0usize;
        for w in 1..=n {
            if log
                .observe_write(&issuer, SCOPE, fake_root((w % 251) as u8), w, ISSUED_AT)
                .is_some()
            {
                anchors_taken += 1;
            }
            max_staleness = max_staleness.max(log.staleness_at(w));
        }
        assert_eq!(anchors_taken, (n / interval) as usize);
        assert_eq!(max_staleness, interval - 1);
    }

    #[test]
    fn verify_state_anchor_accepts_honest_anchor() {
        let issuer = Issuer::generate();
        let mut log = StateAnchorLog::new(StateAnchorPolicy::new(4).unwrap());
        let root = fake_root(42);
        let anchor = log
            .observe_write(&issuer, SCOPE, root, 4, ISSUED_AT)
            .expect("write_count=4 lands on interval=4 boundary");
        let verified = verify_state_anchor(&issuer.verifying_key, SCOPE, root, &anchor)
            .expect("honest anchor must verify");
        assert_eq!(verified.root(), root);
    }

    #[test]
    fn verify_state_anchor_rejects_root_signature_and_scope_tamper() {
        let issuer = Issuer::generate();
        let mut log = StateAnchorLog::new(StateAnchorPolicy::new(1).unwrap());
        let root = fake_root(7);
        let anchor = log
            .observe_write(&issuer, SCOPE, root, 1, ISSUED_AT)
            .unwrap();

        // Claimed root does not match what was actually anchored.
        assert!(verify_state_anchor(&issuer.verifying_key, SCOPE, fake_root(8), &anchor).is_none());

        // Signature byte flipped.
        let mut tampered = anchor;
        tampered.signed_root.signature[0] ^= 0xFF;
        assert!(verify_state_anchor(&issuer.verifying_key, SCOPE, root, &tampered).is_none());

        // Wrong scope.
        assert!(verify_state_anchor(&issuer.verifying_key, [9u8; 32], root, &anchor).is_none());

        // Wrong key.
        let impostor = Issuer::generate();
        assert!(verify_state_anchor(&impostor.verifying_key, SCOPE, root, &anchor).is_none());
    }

    #[test]
    fn state_anchor_purpose_is_isolated_from_receipt_and_batch() {
        let issuer = Issuer::generate();
        let root = fake_root(1);

        // A receipt-purpose or batch-purpose signature over the same bytes
        // must not satisfy verify_state_anchor: purpose is bound into the
        // signed statement, preventing cross-purpose replay.
        let receipt_signed = issuer.sign_root(
            AnchorContext::new(AnchorPurpose::Receipt, SCOPE),
            root,
            ISSUED_AT,
        );
        let fake_anchor = StateAnchor {
            write_count: 1,
            signed_root: receipt_signed,
        };
        assert!(verify_state_anchor(&issuer.verifying_key, SCOPE, root, &fake_anchor).is_none());

        // And a genuine state-anchor signature must not satisfy a
        // Receipt/Batch verification context.
        let mut log = StateAnchorLog::new(StateAnchorPolicy::new(1).unwrap());
        let anchor = log
            .observe_write(&issuer, SCOPE, root, 1, ISSUED_AT)
            .unwrap();
        assert!(verify_root(
            &issuer.verifying_key,
            AnchorContext::new(AnchorPurpose::Receipt, SCOPE),
            &anchor.signed_root
        )
        .is_none());
    }

    #[test]
    fn latest_at_or_before_returns_none_before_first_anchor() {
        let issuer = Issuer::generate();
        let mut log = StateAnchorLog::new(StateAnchorPolicy::new(10).unwrap());
        for w in 1u64..10 {
            log.observe_write(&issuer, SCOPE, fake_root(w as u8), w, ISSUED_AT);
        }
        assert!(log.latest_at_or_before(9).is_none());
        assert_eq!(log.staleness_at(9), 9);
        assert!(log.anchors().is_empty());
    }
}
