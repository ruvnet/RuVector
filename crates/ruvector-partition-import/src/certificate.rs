//! Ed25519 cut certificates.
//!
//! A certificate binds a signature to the triple
//! `(digest(G), digest(P), cut_value)`, following the
//! `MSG_TAG || ... || digest` domain-separated signing convention used
//! elsewhere in this workspace
//! (`ruvector-agent-memory::witness_signing`, `ruvector-retrieval-receipt::signing`):
//! the signed message is a fixed-size byte string, never the raw graph or
//! partition data (which can be arbitrarily large), so signing/verification
//! cost does not grow with graph size.
//!
//! A valid signature proves the certificate's signer produced exactly this
//! `(digest(G), digest(P), cut_value)` triple. It does **not** prove `P` is
//! a correct minimum-cut partition of `G` — that is the adversarial finding
//! this crate exists to demonstrate (see [`crate::candidates`]).

use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::graph::{Partition, WeightedGraph};

/// Domain tag for the signed cut-certificate message. Distinct from the
/// graph/partition encoding tags in `graph.rs` so a graph or partition
/// digest can never be mistaken for, or replayed as, a signed certificate
/// message.
const MSG_TAG: &[u8] = b"ruvector-partition-import:cut-certificate:v1:cert\0";
/// `graph_digest` (32) + `partition_digest` (32) + `cut_value` bits (8).
const STATEMENT_LEN: usize = 32 + 32 + 8;
const MESSAGE_LEN: usize = MSG_TAG.len() + STATEMENT_LEN;

/// The statement covered by a certificate's signature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutStatement {
    pub graph_digest: [u8; 32],
    pub partition_digest: [u8; 32],
    pub cut_value: f64,
}

impl CutStatement {
    fn canonical_bytes(&self) -> [u8; MESSAGE_LEN] {
        let mut bytes = [0u8; MESSAGE_LEN];
        let mut offset = 0;
        bytes[offset..offset + MSG_TAG.len()].copy_from_slice(MSG_TAG);
        offset += MSG_TAG.len();
        bytes[offset..offset + 32].copy_from_slice(&self.graph_digest);
        offset += 32;
        bytes[offset..offset + 32].copy_from_slice(&self.partition_digest);
        offset += 32;
        bytes[offset..offset + 8].copy_from_slice(&self.cut_value.to_bits().to_be_bytes());
        bytes
    }
}

/// A signed cut statement: what agent A sends to agent B alongside `(G,
/// P)`.
#[derive(Clone, Copy, Debug)]
pub struct CutCertificate {
    pub statement: CutStatement,
    pub signature: [u8; 64],
}

impl CutCertificate {
    pub fn wire_size_bytes(&self) -> usize {
        32 + 32 + 8 + 64
    }
}

/// An Ed25519 keypair representing agent A's cut-signing key.
pub struct Signer {
    signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl Signer {
    /// Generate a fresh random signing key (`OsRng`).
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Construct a signer from a fixed 32-byte seed. Used by the benchmark
    /// and tests so results are reproducible across runs without relying
    /// on `OsRng`.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Sign a cut statement directly from already-computed digests.
    pub fn sign_cut(
        &self,
        graph_digest: [u8; 32],
        partition_digest: [u8; 32],
        cut_value: f64,
    ) -> CutCertificate {
        let statement = CutStatement {
            graph_digest,
            partition_digest,
            cut_value,
        };
        let signature = self
            .signing_key
            .sign(&statement.canonical_bytes())
            .to_bytes();
        CutCertificate {
            statement,
            signature,
        }
    }

    /// Sign a cut statement derived from a graph, partition, and cut
    /// value. Convenience wrapper over [`Signer::sign_cut`].
    pub fn sign(
        &self,
        graph: &WeightedGraph,
        partition: &Partition,
        cut_value: f64,
    ) -> CutCertificate {
        self.sign_cut(graph.digest(), partition.digest(), cut_value)
    }
}

/// Stable SHA-256 identifier for a verifying key, for logging/diagnostics.
pub fn key_id(vk: &VerifyingKey) -> [u8; 32] {
    Sha256::digest(vk.as_bytes()).into()
}

/// Verify a certificate's signature against the statement it carries, with
/// no reference to any particular graph/partition data. This checks only
/// "did the holder of `vk` sign exactly this `(graph_digest,
/// partition_digest, cut_value)` triple" — it says nothing about whether
/// that triple is honest or correct.
pub fn verify_signature(vk: &VerifyingKey, cert: &CutCertificate) -> bool {
    let Ok(signature) = Signature::from_slice(&cert.signature) else {
        return false;
    };
    vk.verify_strict(&cert.statement.canonical_bytes(), &signature)
        .is_ok()
}

/// Verify a certificate against the actual graph and partition data it is
/// claimed to describe: recompute both digests from `graph`/`partition`,
/// confirm they match what the certificate's statement claims, and verify
/// the signature over that statement. This is what detects tamper-in-transit:
/// if `graph` or `partition` was altered after signing, the recomputed
/// digest will not match the statement, and if an attacker also rewrites
/// the statement to match the tampered data, the signature (over the new,
/// different statement bytes) will no longer verify under `vk`.
pub fn verify_against(
    vk: &VerifyingKey,
    graph: &WeightedGraph,
    partition: &Partition,
    cert: &CutCertificate,
) -> bool {
    let graph_digest = graph.digest();
    let partition_digest = partition.digest();
    if cert.statement.graph_digest != graph_digest
        || cert.statement.partition_digest != partition_digest
    {
        return false;
    }
    verify_signature(vk, cert)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generate_graph;

    fn sample() -> (WeightedGraph, Partition) {
        let g = generate_graph(40, 11, 6);
        let side = (0..g.n).map(|v| v % 2 == 0).collect();
        (g, Partition::new(side))
    }

    #[test]
    fn honest_round_trip_verifies() {
        let signer = Signer::from_seed([3u8; 32]);
        let (g, p) = sample();
        let cut = p.cut_value(&g);
        let cert = signer.sign(&g, &p, cut);
        assert!(verify_against(&signer.verifying_key, &g, &p, &cert));
    }

    #[test]
    fn graph_tamper_after_signing_is_detected() {
        let signer = Signer::from_seed([3u8; 32]);
        let (mut g, p) = sample();
        let cut = p.cut_value(&g);
        let cert = signer.sign(&g, &p, cut);
        g.edges[0].2 += 0.5;
        assert!(!verify_against(&signer.verifying_key, &g, &p, &cert));
    }

    #[test]
    fn partition_tamper_after_signing_is_detected() {
        let signer = Signer::from_seed([3u8; 32]);
        let (g, mut p) = sample();
        let cut = p.cut_value(&g);
        let cert = signer.sign(&g, &p, cut);
        p.side[0] = !p.side[0];
        assert!(!verify_against(&signer.verifying_key, &g, &p, &cert));
    }

    #[test]
    fn single_byte_signature_tamper_is_detected() {
        let signer = Signer::from_seed([3u8; 32]);
        let (g, p) = sample();
        let cut = p.cut_value(&g);
        let mut cert = signer.sign(&g, &p, cut);
        cert.signature[0] ^= 0xff;
        assert!(!verify_against(&signer.verifying_key, &g, &p, &cert));
    }

    #[test]
    fn wrong_key_fails_verification() {
        let signer = Signer::from_seed([3u8; 32]);
        let impostor = Signer::from_seed([4u8; 32]);
        let (g, p) = sample();
        let cut = p.cut_value(&g);
        let cert = signer.sign(&g, &p, cut);
        assert!(!verify_against(&impostor.verifying_key, &g, &p, &cert));
    }

    #[test]
    fn cut_value_tamper_is_detected() {
        let signer = Signer::from_seed([3u8; 32]);
        let (g, p) = sample();
        let cut = p.cut_value(&g);
        let mut cert = signer.sign(&g, &p, cut);
        cert.statement.cut_value += 1.0;
        // Statement digests still match the data, but the signature no
        // longer covers this (tampered) cut_value.
        assert!(!verify_against(&signer.verifying_key, &g, &p, &cert));
    }
}
