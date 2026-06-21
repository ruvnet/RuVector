//! Witness chain hashing for integrity (ADR-134, ADR-142).
//!
//! When the `crypto-sha256` feature is enabled (default), full SHA-256 is
//! computed and then XOR-folded to fit the u32/u64 fields required by the
//! 64-byte `WitnessRecord` layout. When disabled, the legacy FNV-1a
//! implementation is used instead.

/// Re-export the canonical FNV-1a from rvm-types (always available as fallback).
pub use rvm_types::fnv1a_64;

// ---------------------------------------------------------------------------
// SHA-256 path (ADR-142 Phase 1)
// ---------------------------------------------------------------------------

/// Compute the chain hash using SHA-256, XOR-folded to u64.
///
/// `chain = H(prev_hash || sequence || record_hash)` where `record_hash`
/// is the self-integrity hash of the record's content bytes (see
/// [`compute_record_hash`]). Binding the content hash into the chain
/// means rewriting any record's content breaks every subsequent link,
/// not just the sequence numbering.
///
/// This is stored in the next record's `prev_hash` field (truncated to u32
/// by the caller per ADR-134's 64-byte record constraint). We compute the
/// full 256-bit digest and XOR-fold into 8 bytes for maximum entropy
/// preservation within the field-size budget.
#[cfg(feature = "crypto-sha256")]
#[must_use]
pub fn compute_chain_hash(prev_hash: u64, sequence: u64, record_hash: u64) -> u64 {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(prev_hash.to_le_bytes());
    hasher.update(sequence.to_le_bytes());
    hasher.update(record_hash.to_le_bytes());
    let digest = hasher.finalize();

    // XOR-fold 32 bytes (4 x u64) into a single u64
    xor_fold_256_to_u64(digest.as_ref())
}

/// Compute the self-integrity hash of record data using SHA-256,
/// XOR-folded to u64.
///
/// Takes a byte slice (typically the first 44 bytes of the record)
/// and computes SHA-256 over it, then folds to u64.
#[cfg(feature = "crypto-sha256")]
#[must_use]
pub fn compute_record_hash(data: &[u8]) -> u64 {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();

    xor_fold_256_to_u64(digest.as_ref())
}

/// XOR-fold a 32-byte SHA-256 digest into a single u64.
///
/// Splits the 256-bit digest into four 64-bit words and XORs them
/// together, preserving maximum entropy in the truncated output.
///
/// Accepts `&[u8]` (must be exactly 32 bytes) to work with both
/// `[u8; 32]` and `GenericArray<u8, U32>` via `AsRef<[u8]>`.
#[cfg(feature = "crypto-sha256")]
#[must_use]
fn xor_fold_256_to_u64(digest: &[u8]) -> u64 {
    debug_assert_eq!(digest.len(), 32);
    let mut bytes = [0u8; 8];

    bytes.copy_from_slice(&digest[0..8]);
    let w0 = u64::from_le_bytes(bytes);
    bytes.copy_from_slice(&digest[8..16]);
    let w1 = u64::from_le_bytes(bytes);
    bytes.copy_from_slice(&digest[16..24]);
    let w2 = u64::from_le_bytes(bytes);
    bytes.copy_from_slice(&digest[24..32]);
    let w3 = u64::from_le_bytes(bytes);

    w0 ^ w1 ^ w2 ^ w3
}

// ---------------------------------------------------------------------------
// FNV-1a fallback path (legacy, used when crypto-sha256 is disabled)
// ---------------------------------------------------------------------------

/// Compute the chain hash: FNV-1a of (`prev_hash` ++ sequence ++ `record_hash`).
///
/// `record_hash` is the self-integrity hash of the record's content
/// bytes (see [`compute_record_hash`]), so the chain binds record
/// content, not just ordering.
///
/// This is stored in the next record's `prev_hash` field (truncated to u32).
#[cfg(not(feature = "crypto-sha256"))]
#[must_use]
pub fn compute_chain_hash(prev_hash: u64, sequence: u64, record_hash: u64) -> u64 {
    let mut buf = [0u8; 24];
    buf[..8].copy_from_slice(&prev_hash.to_le_bytes());
    buf[8..16].copy_from_slice(&sequence.to_le_bytes());
    buf[16..24].copy_from_slice(&record_hash.to_le_bytes());
    fnv1a_64(&buf)
}

/// Compute the self-integrity hash of record data.
///
/// Takes a byte slice (typically the first 44 bytes of the record)
/// and computes FNV-1a over it.
#[cfg(not(feature = "crypto-sha256"))]
#[must_use]
pub fn compute_record_hash(data: &[u8]) -> u64 {
    fnv1a_64(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_empty() {
        // FNV-1a is always available via re-export regardless of feature.
        let hash = fnv1a_64(&[]);
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"hello witness";
        let h1 = fnv1a_64(data);
        let h2 = fnv1a_64(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        let h1 = fnv1a_64(b"aaa");
        let h2 = fnv1a_64(b"bbb");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chain_hash_deterministic() {
        let h1 = compute_chain_hash(0, 1, 7);
        let h2 = compute_chain_hash(0, 1, 7);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_chain_hash_differs_by_sequence() {
        let h1 = compute_chain_hash(0, 1, 7);
        let h2 = compute_chain_hash(0, 2, 7);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chain_hash_differs_by_prev() {
        let h1 = compute_chain_hash(100, 1, 7);
        let h2 = compute_chain_hash(200, 1, 7);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chain_hash_differs_by_record_hash() {
        // The chain hash must bind record content: two records at the
        // same position with different content hashes must produce
        // different chain values.
        let h1 = compute_chain_hash(100, 1, 7);
        let h2 = compute_chain_hash(100, 1, 8);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_record_hash_non_empty() {
        let data = [1u8, 2, 3, 4, 5];
        let h = compute_record_hash(&data);
        assert_ne!(h, 0);
    }

    #[cfg(feature = "crypto-sha256")]
    #[test]
    fn test_xor_fold_preserves_entropy() {
        // Different inputs must produce different folded outputs.
        use sha2::{Sha256, Digest};
        let d1 = Sha256::digest(b"alpha");
        let d2 = Sha256::digest(b"bravo");
        let f1 = xor_fold_256_to_u64(d1.as_ref());
        let f2 = xor_fold_256_to_u64(d2.as_ref());
        assert_ne!(f1, f2);
    }

    #[cfg(feature = "crypto-sha256")]
    #[test]
    fn test_sha256_chain_hash_is_not_fnv() {
        // Verify that with crypto-sha256 enabled, the output differs
        // from what FNV-1a would produce (i.e., SHA-256 path is active).
        let sha_h = compute_chain_hash(0, 1, 7);
        let mut buf = [0u8; 24];
        buf[..8].copy_from_slice(&0u64.to_le_bytes());
        buf[8..16].copy_from_slice(&1u64.to_le_bytes());
        buf[16..24].copy_from_slice(&7u64.to_le_bytes());
        let fnv_h = fnv1a_64(&buf);
        assert_ne!(sha_h, fnv_h, "SHA-256 path should produce different output than FNV-1a");
    }
}
