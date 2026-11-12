//! LatentMesh transport seam (ADR-326 Decision §4, over ADR-309's greenfield
//! crates).
//!
//! Peer-to-peer messages in this topology are carried as **compressed latent
//! state** (KV-cache-style), *not* natural-language token streams — the
//! first-mover angle ADR-326 captures: testing DeAR's topology under a
//! communication substrate the paper itself does not use.
//!
//! ## Seam status (honest scope)
//!
//! ADR-309's greenfield `latentmesh-*` transport crates do not yet expose a wire
//! API on `origin/main`. So this module **defines the seam** — the
//! [`LatentTransport`] trait and the [`LatentMessage`] envelope — and ships a
//! working **in-process stub** ([`InProcessTransport`]) so WP23 is end-to-end
//! testable now. Full LatentMesh wire carriage (compression codec, framing,
//! `ruvnet/LatentMesh` wire-format coordination) is deferred to WP5; a WP5
//! transport crate implements [`LatentTransport`] without changing any caller.

use serde::{Deserialize, Serialize};

use crate::capability::PeerId;
use crate::thoughtmap::StepId;

/// Compressed latent state carried between agents (KV-cache-style). This slice
/// models it as an `f32` vector; the WP5 codec replaces the representation
/// behind the same envelope.
pub type LatentState = Vec<f32>;

/// Causal attribution evidence attached to every message (consumed by the
/// ADR-310 gate in [`crate::causal`]). The message claims it follows causally
/// from `antecedent`; the gate verifies that claim under controlled replacement
/// before the receiving agent may incorporate it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalTag {
    /// The thought-map step this message claims to causally follow from.
    pub antecedent: StepId,
}

/// A peer-to-peer message: compressed latent payload + causal attribution +
/// integrity tag. Nothing here is trusted until the causal gate and the
/// quarantine integrity check pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentMessage {
    pub from: PeerId,
    pub to: PeerId,
    /// Compressed latent contribution this message proposes to add.
    pub latent: LatentState,
    /// Causal attribution evidence (ADR-310).
    pub causal: CausalTag,
    /// HMAC-style integrity tag (ADR-311). A mismatch means tampering and the
    /// message is quarantined regardless of the sender's capability score.
    pub integrity: u64,
    #[serde(default)]
    pub summary: String,
}

/// The transport seam. A WP5 LatentMesh crate implements this; callers only ever
/// see this trait.
pub trait LatentTransport {
    /// Enqueue a message for carriage to `msg.to`.
    fn send(&mut self, msg: LatentMessage) -> Result<(), TransportError>;
    /// Dequeue the next message addressed to `me`, if any.
    fn recv(&mut self, me: PeerId) -> Option<LatentMessage>;
}

/// Transport-level failure. Wire-format/codec errors join here under WP5.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TransportError {
    #[error("transport channel closed")]
    Closed,
}

/// In-process FIFO stub of [`LatentTransport`] for tests and single-node runs.
/// **Not** the wire transport — that is WP5.
#[derive(Debug, Default)]
pub struct InProcessTransport {
    queue: Vec<LatentMessage>,
}

impl InProcessTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl LatentTransport for InProcessTransport {
    fn send(&mut self, msg: LatentMessage) -> Result<(), TransportError> {
        self.queue.push(msg);
        Ok(())
    }

    fn recv(&mut self, me: PeerId) -> Option<LatentMessage> {
        if let Some(pos) = self.queue.iter().position(|m| m.to == me) {
            Some(self.queue.remove(pos))
        } else {
            None
        }
    }
}

/// Deterministic integrity tag over a message's load-bearing fields. Stands in
/// for the ADR-311 HMAC; WP5 swaps in a keyed MAC. Exposed so senders can stamp
/// a message and the quarantine layer can recompute and compare.
pub fn integrity_tag(from: PeerId, to: PeerId, latent: &[f32], antecedent: StepId) -> u64 {
    // FNV-1a over the byte representation of the fields. Deterministic and
    // dependency-free; the point under test is "recomputed tag must match", which
    // any MAC satisfies.
    fn mix(hash: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *hash ^= b as u64;
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    mix(&mut hash, &from.0.to_le_bytes());
    mix(&mut hash, &to.0.to_le_bytes());
    mix(&mut hash, &antecedent.0.to_le_bytes());
    for v in latent {
        mix(&mut hash, &v.to_bits().to_le_bytes());
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_process_transport_routes_by_recipient() {
        let mut t = InProcessTransport::new();
        let msg = LatentMessage {
            from: PeerId(1),
            to: PeerId(2),
            latent: vec![1.0, 2.0],
            causal: CausalTag {
                antecedent: StepId(0),
            },
            integrity: 0,
            summary: String::new(),
        };
        t.send(msg.clone()).unwrap();
        assert!(t.recv(PeerId(3)).is_none(), "not addressed to 3");
        assert_eq!(t.recv(PeerId(2)).unwrap().from, PeerId(1));
        assert_eq!(t.pending(), 0);
    }

    #[test]
    fn integrity_tag_is_deterministic_and_sensitive() {
        let a = integrity_tag(PeerId(1), PeerId(2), &[1.0, 2.0], StepId(0));
        let same = integrity_tag(PeerId(1), PeerId(2), &[1.0, 2.0], StepId(0));
        let tampered = integrity_tag(PeerId(1), PeerId(2), &[1.0, 2.5], StepId(0));
        assert_eq!(a, same);
        assert_ne!(a, tampered);
    }
}
