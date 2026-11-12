//! Artifact identity: content hash + revision id (ADR-318 §Decision 1).

// Lint suggests `as_chunks`, stable only since Rust 1.88 — past the declared
// `rust-version = "1.77"`. `unknown_lints` keeps older toolchains quiet.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use core::fmt;
use serde::{Deserialize, Serialize};

/// SHA-256 content hash of an artifact's canonical byte representation.
///
/// Computed with the repo's existing RVF-layer SHA-256
/// ([`rvf_types::sha256`], FIPS 180-4, verified against NIST vectors) — the
/// same SHA-256 the program's canonical receipt/witness contract (ruflo
/// ADR-322C, adopted via ADR-312) specifies. ADR-318 explicitly reuses the
/// existing hashing discipline rather than introducing a new hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Hash `content` (the artifact's canonical bytes).
    pub fn of(content: &[u8]) -> Self {
        Self(rvf_types::sha256::sha256(content))
    }

    /// Parse a 64-char hex rendering back into a hash (case-insensitive).
    /// Returns `None` for any other length or non-hex characters.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let bytes = hex.as_bytes();
        if bytes.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let hi = (chunk[0] as char).to_digit(16)?;
            let lo = (chunk[1] as char).to_digit(16)?;
            out[i] = ((hi << 4) | lo) as u8;
        }
        Some(Self(out))
    }

    /// Lowercase hex rendering (64 chars).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use core::fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// A reference to one **exact version** of one artifact.
///
/// This is the binding unit of ADR-318: every downstream reference (parser
/// result, tool call, diff, approval, output — see [`crate::ViewKind`])
/// carries an `ArtifactRef`, i.e. the content hash *and* revision id of the
/// version it read, not merely the artifact's identity. Modeled on
/// StagedWorkspace's (arXiv:2608.18050) binding of parsed records and review
/// diffs to native-file content hashes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// Stable identity of the artifact's lineage (e.g. a workspace path or
    /// RVF record id). Identity alone is NOT a version reference.
    pub artifact_id: String,
    /// SHA-256 of the exact version's canonical bytes.
    pub content_hash: ContentHash,
    /// Monotonic revision counter scoped to this artifact's lineage
    /// (starts at 1 for the first committed version).
    pub revision_id: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_known_sha256_vector() {
        // NIST vector: SHA-256("abc")
        let h = ContentHash::of(b"abc");
        assert_eq!(
            h.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn display_is_hex() {
        let h = ContentHash::of(b"");
        assert_eq!(format!("{h}").len(), 64);
    }
}
