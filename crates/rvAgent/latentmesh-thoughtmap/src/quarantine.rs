//! Anomaly quarantine for misbehaving peers (ADR-326 §security; ADR-311).
//!
//! The WP7 concern flagged on RuVector#882: **a misbehaving peer must not be
//! able to trigger spurious topology churn.** A peer that degrades the shared
//! thought map — repeatedly sending messages that fail the causal gate, failing
//! integrity checks, or weakening good edges / injecting bad steps — accrues
//! anomaly strikes. Once over threshold the peer is **quarantined**: its
//! messages are dampened (not incorporated) and, crucially, its proposed
//! topology mutations are ignored, so it cannot arbitrarily reshape the graph.
//!
//! This is deliberately *not* a hair-trigger: a single failure dampens influence
//! but does not quarantine. Only sustained anomalous behavior quarantines,
//! bounding the damage one bad actor can do without penalizing transient noise.

use std::collections::HashMap;

use crate::capability::PeerId;

/// The kind of anomalous behavior observed from a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anomaly {
    /// A message failed the ADR-310 causal-attribution gate.
    CausalRejection,
    /// A message failed the ADR-311 integrity (HMAC) check.
    IntegrityFailure,
    /// The peer's incorporated contribution was later confirmed to have
    /// weakened a known-good path or injected a bad step.
    MapDegradation,
}

impl Anomaly {
    /// How many strikes this class of anomaly contributes. Integrity failures
    /// are treated as the strongest signal of a malicious/tampered channel.
    fn weight(self) -> u32 {
        match self {
            Anomaly::CausalRejection => 1,
            Anomaly::MapDegradation => 2,
            Anomaly::IntegrityFailure => 3,
        }
    }
}

/// Per-peer reputation and quarantine state.
#[derive(Debug, Clone)]
pub struct PeerReputation {
    pub peer: PeerId,
    pub strikes: u32,
    pub quarantined: bool,
}

/// Tracks anomalies per peer and decides quarantine.
#[derive(Debug, Clone)]
pub struct Quarantine {
    reputations: HashMap<PeerId, PeerReputation>,
    /// Strike total at or above which a peer is quarantined.
    pub strike_threshold: u32,
}

impl Default for Quarantine {
    fn default() -> Self {
        Self {
            reputations: HashMap::new(),
            strike_threshold: 3,
        }
    }
}

impl Quarantine {
    pub fn new(strike_threshold: u32) -> Self {
        Self {
            reputations: HashMap::new(),
            strike_threshold,
        }
    }

    fn entry(&mut self, peer: PeerId) -> &mut PeerReputation {
        self.reputations.entry(peer).or_insert(PeerReputation {
            peer,
            strikes: 0,
            quarantined: false,
        })
    }

    /// Record an anomaly for a peer, returning whether it is now quarantined.
    pub fn record(&mut self, peer: PeerId, anomaly: Anomaly) -> bool {
        let threshold = self.strike_threshold;
        let rep = self.entry(peer);
        rep.strikes = rep.strikes.saturating_add(anomaly.weight());
        if rep.strikes >= threshold {
            rep.quarantined = true;
        }
        rep.quarantined
    }

    /// Whether a peer is currently quarantined. Quarantined peers' messages are
    /// dampened and their topology mutations ignored.
    pub fn is_quarantined(&self, peer: PeerId) -> bool {
        self.reputations
            .get(&peer)
            .map(|r| r.quarantined)
            .unwrap_or(false)
    }

    pub fn strikes(&self, peer: PeerId) -> u32 {
        self.reputations.get(&peer).map(|r| r.strikes).unwrap_or(0)
    }

    /// Reinstate a peer whose good behavior has been re-established (e.g. after a
    /// witness-verified appeal). Kept explicit so reinstatement is always an
    /// auditable action, never automatic decay.
    pub fn reinstate(&mut self, peer: PeerId) {
        if let Some(rep) = self.reputations.get_mut(&peer) {
            rep.strikes = 0;
            rep.quarantined = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_failure_does_not_quarantine() {
        let mut q = Quarantine::default();
        assert!(!q.record(PeerId(7), Anomaly::CausalRejection));
        assert!(!q.is_quarantined(PeerId(7)));
    }

    #[test]
    fn sustained_anomalies_quarantine() {
        let mut q = Quarantine::default();
        q.record(PeerId(7), Anomaly::CausalRejection);
        q.record(PeerId(7), Anomaly::CausalRejection);
        let now = q.record(PeerId(7), Anomaly::CausalRejection);
        assert!(now);
        assert!(q.is_quarantined(PeerId(7)));
    }

    #[test]
    fn integrity_failure_quarantines_fast() {
        let mut q = Quarantine::default();
        // Weight 3 ≥ threshold 3 in one shot — a tampered channel is high signal.
        assert!(q.record(PeerId(9), Anomaly::IntegrityFailure));
    }

    #[test]
    fn reinstate_clears_state() {
        let mut q = Quarantine::default();
        q.record(PeerId(7), Anomaly::IntegrityFailure);
        assert!(q.is_quarantined(PeerId(7)));
        q.reinstate(PeerId(7));
        assert!(!q.is_quarantined(PeerId(7)));
        assert_eq!(q.strikes(PeerId(7)), 0);
    }
}
