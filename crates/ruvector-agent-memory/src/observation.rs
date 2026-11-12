//! `AtomicObservation` — the atomic memory unit of the cross-source causal
//! fusion layer (ADR-320, PIR WP18).
//!
//! Informed by **MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`)**, which
//! preserves source-level evidence in an event-layer atomic memory and organizes
//! related atomic events into a causal fusion graph. This is deliberately a
//! first-party implementation of that *pattern* and is **explicitly distinct
//! from the unrelated, pre-existing `memfuse/memfuse` open-source memory layer**
//! — per ADR-320's binding naming discipline, no artifact here is ever named
//! `memfuse`.
//!
//! An [`AtomicObservation`] is the atomic unit written into ADR-307's
//! continuous-latent-state tier by any agent or sensor. Per ADR-320 §1 it
//! carries, at minimum: `source`, `time`, `confidence`, `tenant`, `signature`,
//! and `causal_parents`. It is **content-addressed**: its [`ObservationId`] is
//! the SHA-256 of its canonical byte encoding (everything except the
//! signature), so provenance — the load-bearing property of this layer — is a
//! cryptographic fact, not a bookkeeping convention: the id *is* a commitment
//! to the exact source, time, confidence, tenant, parents, and evidence.
//!
//! ## Reused primitives (no new hash or signature scheme — ADR-320 gate)
//!
//! - Hashing: [`rvf_types::sha256::sha256`] (the repo's FIPS 180-4 SHA-256).
//! - Signing: [`rvf_types::ed25519_sign`] / [`rvf_types::ed25519_verify`]
//!   (the repo's RFC 8032 Ed25519, the same `SignatureVerified`-grade evidence
//!   the ADR-322C witness contract anticipates).

use rvf_types::sha256::sha256;
use rvf_types::{ed25519_sign, ed25519_verify, Ed25519Keypair};

/// Domain-separation tag bound into every observation's signed preimage, so a
/// signature over an `AtomicObservation` can never be replayed as a signature
/// over some other RVF/ledger structure.
const OBSERVATION_DOMAIN: &[u8] = b"ADR-320-atomic-observation-v1";

/// Multi-tenant isolation boundary carried by every observation. Enforced as a
/// hard boundary at fusion time (ADR-320 Security Gates: an observation from
/// one tenant never fuses into another tenant's causal graph).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tenant(pub String);

impl Tenant {
    pub fn new(id: impl Into<String>) -> Self {
        Tenant(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The heterogeneous source kinds that converge into one causal graph. ADR-320
/// models RuView RF events, network telemetry, agent observations, and user
/// activity flowing into the same fusion; `Other` keeps the schema extensible
/// without a layout break (reserve codes >= 16 for out-of-tree kinds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceKind {
    /// RuView radio-frequency event stream.
    RuViewRf,
    /// Network / infrastructure telemetry.
    NetworkTelemetry,
    /// An agent's own first-party observation.
    AgentObservation,
    /// End-user activity signal.
    UserActivity,
    /// Extension point for out-of-tree source kinds (code >= 16).
    Other(u16),
}

impl SourceKind {
    /// Stable numeric code bound into the canonical (hashed) encoding.
    pub fn code(self) -> u16 {
        match self {
            SourceKind::RuViewRf => 1,
            SourceKind::NetworkTelemetry => 2,
            SourceKind::AgentObservation => 3,
            SourceKind::UserActivity => 4,
            SourceKind::Other(x) => x,
        }
    }
}

/// The originating identity of an observation: its kind, a stable string id
/// (e.g. `"sensor-7"`, `"agent-alpha"`), and the Ed25519 public key the
/// observation is signed under. Carrying the key inline makes verification
/// self-contained for this slice; a production deployment would resolve the key
/// from a tenant-scoped key registry (see Deferred scope).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationSource {
    pub kind: SourceKind,
    pub id: String,
    pub public_key: [u8; 32],
}

/// Content address of an [`AtomicObservation`]: SHA-256 over the domain tag and
/// the observation's canonical encoding, excluding the signature. Two
/// observations share an id **iff** they commit to the identical source, time,
/// confidence, tenant, causal parents, and evidence payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(pub [u8; 32]);

impl ObservationId {
    /// Lowercase hex rendering — used as the provenance content string when an
    /// observation is recorded as a governed transition in the WP4 ledger.
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
        s
    }
}

/// Errors constructing or validating an [`AtomicObservation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationError {
    /// `confidence` was not a finite value in `[0.0, 1.0]`.
    ConfidenceOutOfRange,
}

/// An atomic, source-tagged observation — the event-layer memory unit of
/// ADR-320. Fields follow ADR-320 §1 exactly; `payload` is the original
/// evidence body that provenance resolves back to.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicObservation {
    /// Originating agent/sensor identity (ADR-320 `source`).
    pub source: ObservationSource,
    /// Observation timestamp in nanoseconds (ADR-320 `time`).
    pub time_ns: u64,
    /// The source's own confidence in the observation, in `[0.0, 1.0]`
    /// (ADR-320 `confidence`).
    pub confidence: f32,
    /// Multi-tenant isolation boundary (ADR-320 `tenant`).
    pub tenant: Tenant,
    /// The `AtomicObservation`(s) that causally preceded and informed this one
    /// (ADR-320 `causal_parents`). Canonicalized (sorted, de-duplicated).
    pub causal_parents: Vec<ObservationId>,
    /// Original evidence body this observation records; provenance resolves
    /// back to these bytes.
    pub payload: Vec<u8>,
    /// Ed25519 signature over this observation's content address (ADR-320
    /// `signature`; RFC 8032, reused from `rvf-types`).
    pub signature: [u8; 64],
}

impl AtomicObservation {
    /// Construct and **sign** an observation. The `source.public_key` is
    /// overwritten with `keypair`'s public key so the stored source identity
    /// and the signing key can never disagree.
    ///
    /// `causal_parents` is canonicalized (sorted + de-duplicated) before the
    /// content address is computed, so parent ordering never affects identity.
    ///
    /// Returns [`ObservationError::ConfidenceOutOfRange`] if `confidence` is
    /// not finite in `[0.0, 1.0]`.
    pub fn new_signed(
        mut source: ObservationSource,
        time_ns: u64,
        confidence: f32,
        tenant: Tenant,
        mut causal_parents: Vec<ObservationId>,
        payload: Vec<u8>,
        keypair: &Ed25519Keypair,
    ) -> Result<Self, ObservationError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ObservationError::ConfidenceOutOfRange);
        }
        source.public_key = keypair.public_key();
        causal_parents.sort_unstable();
        causal_parents.dedup();

        let mut obs = AtomicObservation {
            source,
            time_ns,
            confidence,
            tenant,
            causal_parents,
            payload,
            signature: [0u8; 64],
        };
        let id = obs.compute_id();
        obs.signature = ed25519_sign(&keypair.secret_key(), &id.0);
        Ok(obs)
    }

    /// The observation's content address (recomputed from current field
    /// values; never trusts a stored value).
    pub fn id(&self) -> ObservationId {
        self.compute_id()
    }

    /// Verify the signature binds this exact content under `source.public_key`.
    ///
    /// Because the signed message is the content address, any post-signing
    /// mutation to any field (source, time, confidence, tenant, parents, or
    /// payload) changes the id and makes verification fail — this is the
    /// per-observation authenticity gate of ADR-320.
    pub fn verify(&self) -> bool {
        let id = self.compute_id();
        ed25519_verify(&self.source.public_key, &id.0, &self.signature)
    }

    /// Canonical, deterministic byte encoding of everything the id commits to
    /// (the signature is excluded — it is a function of the id, not an input).
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(OBSERVATION_DOMAIN);
        b.extend_from_slice(&self.source.kind.code().to_le_bytes());
        write_len_prefixed(&mut b, self.source.id.as_bytes());
        b.extend_from_slice(&self.source.public_key);
        b.extend_from_slice(&self.time_ns.to_le_bytes());
        // f32 committed by its bit pattern for exact, portable determinism.
        b.extend_from_slice(&self.confidence.to_bits().to_le_bytes());
        write_len_prefixed(&mut b, self.tenant.as_str().as_bytes());
        b.extend_from_slice(&(self.causal_parents.len() as u32).to_le_bytes());
        for parent in &self.causal_parents {
            b.extend_from_slice(&parent.0);
        }
        write_len_prefixed(&mut b, &self.payload);
        b
    }

    fn compute_id(&self) -> ObservationId {
        ObservationId(sha256(&self.canonical_bytes()))
    }
}

/// Append a `u32` length prefix (LE) followed by the bytes.
fn write_len_prefixed(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn kp() -> Ed25519Keypair {
        Ed25519Keypair::generate(&mut OsRng)
    }

    fn source(kind: SourceKind, id: &str) -> ObservationSource {
        ObservationSource {
            kind,
            id: id.to_string(),
            public_key: [0u8; 32],
        }
    }

    #[test]
    fn construction_and_signature_round_trip() {
        let keypair = kp();
        let obs = AtomicObservation::new_signed(
            source(SourceKind::AgentObservation, "agent-alpha"),
            1_000,
            0.9,
            Tenant::new("acme"),
            vec![],
            b"observed evidence".to_vec(),
            &keypair,
        )
        .unwrap();

        assert!(obs.verify(), "freshly signed observation must verify");
        // public key was bound from the signing keypair.
        assert_eq!(obs.source.public_key, keypair.public_key());
    }

    #[test]
    fn id_is_stable_and_parent_order_independent() {
        let keypair = kp();
        let p1 = ObservationId([1u8; 32]);
        let p2 = ObservationId([2u8; 32]);
        let mk = |parents: Vec<ObservationId>| {
            AtomicObservation::new_signed(
                source(SourceKind::RuViewRf, "rf-1"),
                42,
                0.5,
                Tenant::new("t"),
                parents,
                b"e".to_vec(),
                &keypair,
            )
            .unwrap()
        };
        // Parent list order must not change identity (canonicalized).
        assert_eq!(mk(vec![p1, p2]).id(), mk(vec![p2, p1]).id());
        // Duplicate parents collapse.
        assert_eq!(mk(vec![p1, p1, p2]).id(), mk(vec![p1, p2]).id());
    }

    #[test]
    fn tampered_payload_fails_verification() {
        let keypair = kp();
        let mut obs = AtomicObservation::new_signed(
            source(SourceKind::NetworkTelemetry, "netprobe"),
            7,
            0.75,
            Tenant::new("acme"),
            vec![],
            b"authentic".to_vec(),
            &keypair,
        )
        .unwrap();
        assert!(obs.verify());

        obs.payload = b"forged".to_vec();
        assert!(
            !obs.verify(),
            "post-signing mutation must break the signature"
        );
    }

    #[test]
    fn confidence_out_of_range_rejected() {
        let keypair = kp();
        for bad in [1.5f32, -0.1, f32::NAN] {
            let r = AtomicObservation::new_signed(
                source(SourceKind::UserActivity, "u"),
                0,
                bad,
                Tenant::new("t"),
                vec![],
                b"x".to_vec(),
                &keypair,
            );
            assert_eq!(r.unwrap_err(), ObservationError::ConfidenceOutOfRange);
        }
    }
}
