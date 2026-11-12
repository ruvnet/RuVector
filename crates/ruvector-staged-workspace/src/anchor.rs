//! ADR-312 anchoring seam: workspace-state transitions as witness/receipt
//! records (ADR-318 §Decision 4).
//!
//! ADR-312 resolves the program's witness layer as a *shared record schema
//! plus a cross-layer anchoring contract*: records that must verify across
//! layers use ruflo ADR-322C's canonical encoding, SHA-256 identity
//! derivation, and domain-separated Ed25519 signatures
//! (`Ed25519(domainPrefix || 0x00 || canonicalBytes)`). This module defines
//! that boundary for workspace-state transitions:
//!
//! - [`WorkspaceTransitionRecord`] — the record for one artifact-lineage
//!   transition, with [`canonical_bytes`](WorkspaceTransitionRecord::canonical_bytes)
//!   (fixed field order, length-prefixed strings, little-endian integers),
//!   a SHA-256 [`record_id`](WorkspaceTransitionRecord::record_id) mirroring
//!   322C's `receiptId = SHA-256(canonical unsigned payload)`, and a
//!   domain-separated [`signing_preimage`](WorkspaceTransitionRecord::signing_preimage).
//! - [`TransitionAnchor`] — the trait the workspace emits records through.
//!   Wiring the full ruflo ADR-322C receipt emission (RFC 8785 JCS +
//!   Ed25519, ruflo#3066 formal spec) is deliberately out of this slice; a
//!   real implementation lives behind this trait, the same way
//!   `ruvector-agent-memory`'s `WitnessSink` stubs its WP8 RVM anchoring.
//!
//! **Fail-closed contract**: if `anchor` returns `Err`, the workspace
//! transition is aborted — "no anchor, no transition", matching ADR-134's
//! "no witness, no mutation" invariant already enforced by the WP4 ledger.

use serde::{Deserialize, Serialize};

use crate::artifact::ContentHash;

/// Domain-separation prefix for workspace-transition signing preimages,
/// following ADR-322C's `domainPrefix || 0x00 || canonicalBytes` scheme
/// (distinct from ruflo's `ruflo/flywheel-receipt/v1` and
/// `ruflo/flywheel-ledger-head/v1` domains).
pub const TRANSITION_DOMAIN: &str = "ruv/staged-workspace-transition/v1";

/// Evidence grade, using the three-value vocabulary of the program's
/// canonical witness/receipt contract (ruflo ADR-322C, adopted via
/// ADR-312) — the same vocabulary `ruvector-agent-memory::EvidenceGrade`
/// uses for TARL ledger transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceGrade {
    /// Re-derived from primary data (e.g. the workspace recomputed the
    /// content hash from the artifact's actual bytes at commit time).
    Recomputed,
    /// Backed by a verified cryptographic signature (produced only by a
    /// real ADR-322C anchor implementation; never by [`NoopAnchor`]).
    SignatureVerified,
    /// Asserted without independent recomputation or signature.
    TrustedAssertion,
}

impl EvidenceGrade {
    /// Compact code bound into the canonical encoding
    /// (1 = recomputed, 2 = signature-verified, 3 = trusted-assertion).
    pub fn code(self) -> u8 {
        match self {
            EvidenceGrade::Recomputed => 1,
            EvidenceGrade::SignatureVerified => 2,
            EvidenceGrade::TrustedAssertion => 3,
        }
    }
}

/// One artifact-lineage transition (create or update), as anchored at the
/// ADR-312 boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTransitionRecord {
    /// Workspace-scoped monotonic transition sequence number.
    pub sequence: u64,
    /// Wall-clock nanoseconds at commit time.
    pub timestamp_ns: u64,
    /// Lineage identity of the artifact that changed.
    pub artifact_id: String,
    /// Content hash of the version being superseded (`None` on create).
    pub prior_hash: Option<ContentHash>,
    /// Revision id being superseded (`None` on create).
    pub prior_revision: Option<u64>,
    /// Content hash of the newly committed version.
    pub new_hash: ContentHash,
    /// Revision id of the newly committed version.
    pub new_revision: u64,
    /// Identity of the actor that committed the write.
    pub actor_id: String,
    /// Evidence grade for the `new_hash` claim. The workspace always sets
    /// [`EvidenceGrade::Recomputed`]: it derived the hash from the
    /// artifact's actual bytes, not from a caller's assertion.
    pub evidence_grade: EvidenceGrade,
}

impl WorkspaceTransitionRecord {
    /// Canonical byte encoding: fixed field order, little-endian integers,
    /// u64-length-prefixed strings, `0x00`/`0x01` option tags. This is the
    /// crate-local stand-in for ADR-322C's RFC 8785 JCS canonical JSON —
    /// full JCS shape conformance is the ruflo#3066 follow-up, behind
    /// [`TransitionAnchor`].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::with_capacity(160 + self.artifact_id.len() + self.actor_id.len());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        put_str(&mut out, &self.artifact_id);
        match (&self.prior_hash, self.prior_revision) {
            (Some(h), Some(r)) => {
                out.push(0x01);
                out.extend_from_slice(&h.0);
                out.extend_from_slice(&r.to_le_bytes());
            }
            _ => out.push(0x00),
        }
        out.extend_from_slice(&self.new_hash.0);
        out.extend_from_slice(&self.new_revision.to_le_bytes());
        put_str(&mut out, &self.actor_id);
        out.push(self.evidence_grade.code());
        out
    }

    /// Record identity: `SHA-256(canonical_bytes)`, mirroring ADR-322C's
    /// `receiptId = SHA-256(JCS(unsigned receipt payload))` derivation.
    pub fn record_id(&self) -> ContentHash {
        ContentHash::of(&self.canonical_bytes())
    }

    /// Signing preimage per ADR-322C's domain-separation scheme:
    /// `TRANSITION_DOMAIN || 0x00 || canonical_bytes`. A real anchor signs
    /// this with Ed25519; this slice defines the preimage but does not sign.
    pub fn signing_preimage(&self) -> Vec<u8> {
        let canonical = self.canonical_bytes();
        let mut out = Vec::with_capacity(TRANSITION_DOMAIN.len() + 1 + canonical.len());
        out.extend_from_slice(TRANSITION_DOMAIN.as_bytes());
        out.push(0x00);
        out.extend_from_slice(&canonical);
        out
    }
}

/// Receipt returned by an anchor for one accepted transition record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorReceipt {
    /// The anchored record's identity (`SHA-256` of its canonical bytes).
    pub record_id: ContentHash,
    /// How strongly the anchoring itself is evidenced:
    /// [`EvidenceGrade::TrustedAssertion`] for [`NoopAnchor`];
    /// [`EvidenceGrade::SignatureVerified`] for a real ADR-322C
    /// Ed25519-signing anchor.
    pub grade: EvidenceGrade,
}

/// Error returned by an anchor that refuses a record. Per the fail-closed
/// contract, the workspace aborts the transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorError(pub String);

impl core::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "anchor rejected transition record: {}", self.0)
    }
}

impl std::error::Error for AnchorError {}

/// The ADR-312 anchoring seam. A workspace-state transition becomes durable
/// only if its record is accepted here; rejection aborts the transition
/// ("no anchor, no transition").
pub trait TransitionAnchor {
    /// Anchor one transition record, returning a receipt on acceptance.
    fn anchor(&mut self, record: &WorkspaceTransitionRecord) -> Result<AnchorReceipt, AnchorError>;
}

/// Accepts every record without signing — grades the anchoring
/// [`EvidenceGrade::TrustedAssertion`]. The in-process default, mirroring
/// `ruvector-agent-memory::NoopWitnessSink`; production use requires a real
/// ADR-322C anchor behind [`TransitionAnchor`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopAnchor;

impl TransitionAnchor for NoopAnchor {
    fn anchor(&mut self, record: &WorkspaceTransitionRecord) -> Result<AnchorReceipt, AnchorError> {
        Ok(AnchorReceipt {
            record_id: record.record_id(),
            grade: EvidenceGrade::TrustedAssertion,
        })
    }
}

/// Rejects every record — used to prove the fail-closed "no anchor, no
/// transition" property in tests.
#[derive(Debug, Default, Clone)]
pub struct RejectingAnchor {
    /// Reason echoed in the [`AnchorError`].
    pub reason: String,
}

impl TransitionAnchor for RejectingAnchor {
    fn anchor(&mut self, _: &WorkspaceTransitionRecord) -> Result<AnchorReceipt, AnchorError> {
        Err(AnchorError(if self.reason.is_empty() {
            "rejecting anchor".to_string()
        } else {
            self.reason.clone()
        }))
    }
}
