//! `collection_uid` (ADR-351 §4.3): a 128-bit identifier assigned by the
//! `TenantLedger` catalog at create time and never reused.
//!
//! The uid is `sha256("v2|collection_uid|" + salt[32] + seq_be[8] +
//! nonce[16])[0..16]`. The ledger persists its random per-tenant `salt` and a
//! monotonically increasing `seq`; **every allocation also draws a fresh
//! 16-byte nonce** from the platform CSPRNG. The nonce makes each uid
//! independent of ledger history: if ledger state is ever rolled back (an
//! operator point-in-time restore, a re-seed from a stale snapshot), the same
//! `(salt, seq)` does not reproduce a deleted collection's uid, so the next
//! create cannot land on an orphaned `VectorShard` whose `meta` row would
//! otherwise match exactly and resurrect tombstoned vectors.
//!
//! The ledger caller must still reject (and retry) an allocation that
//! collides with any catalog row, live or tombstoned, and collection delete
//! must wipe the DO's storage before the catalog row is tombstoned.

use crate::error::TenancyError;
use core::fmt;
use sha2::{Digest, Sha256};

/// Bytes in a collection uid.
pub const COLLECTION_UID_BYTES: usize = 16;
/// Bytes of ledger salt.
pub const UID_SALT_BYTES: usize = 32;
/// Bytes of fresh per-allocation nonce.
pub const UID_NONCE_BYTES: usize = 16;

const UID_DOMAIN: &[u8] = b"v2|collection_uid|";

/// A 128-bit collection uid. Wire form: 32 lowercase hex chars.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectionUid([u8; COLLECTION_UID_BYTES]);

impl CollectionUid {
    /// Wrap raw bytes.
    pub fn from_bytes(bytes: [u8; COLLECTION_UID_BYTES]) -> Self {
        CollectionUid(bytes)
    }

    /// Derive the uid for `(salt, seq, nonce)`. Pure; allocation goes through
    /// [`UidAllocator::allocate`], which supplies a fresh random `nonce`.
    pub fn derive(salt: &[u8; UID_SALT_BYTES], seq: u64, nonce: &[u8; UID_NONCE_BYTES]) -> Self {
        let mut h = Sha256::new();
        h.update(UID_DOMAIN);
        h.update(salt);
        h.update(seq.to_be_bytes());
        h.update(nonce);
        let digest = h.finalize();
        let mut out = [0u8; COLLECTION_UID_BYTES];
        out.copy_from_slice(&digest[..COLLECTION_UID_BYTES]);
        CollectionUid(out)
    }

    /// Strictly parse the wire form: exactly 32 **lowercase** hex chars.
    pub fn parse(input: &str) -> Result<Self, TenancyError> {
        let well_formed = input.len() == COLLECTION_UID_BYTES * 2
            && input
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !well_formed {
            return Err(TenancyError::MalformedIdentifier("collection_uid"));
        }
        let mut out = [0u8; COLLECTION_UID_BYTES];
        hex::decode_to_slice(input, &mut out)
            .map_err(|_| TenancyError::MalformedIdentifier("collection_uid"))?;
        Ok(CollectionUid(out))
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8; COLLECTION_UID_BYTES] {
        &self.0
    }

    /// Lowercase hex wire form.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for CollectionUid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CollectionUid({})", self.to_hex())
    }
}

impl fmt::Display for CollectionUid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Port for the platform CSPRNG (`crypto.getRandomValues` in the Worker).
#[cfg_attr(test, mockall::automock)]
pub trait EntropySource {
    /// Fill `out` with cryptographically secure random bytes.
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError>;
}

/// Ledger-side uid allocator. The ledger persists [`UidAllocator::salt`] and
/// [`UidAllocator::next_seq`] and must write the advanced `next_seq` in the
/// same transaction as the catalog row it allocated for.
#[derive(Clone, PartialEq, Eq)]
pub struct UidAllocator {
    salt: [u8; UID_SALT_BYTES],
    next_seq: u64,
}

/// Redacting: the salt is never printed.
impl fmt::Debug for UidAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UidAllocator")
            .field("salt", &"<redacted>")
            .field("next_seq", &self.next_seq)
            .finish()
    }
}

impl UidAllocator {
    /// Restore from persisted state.
    pub fn new(salt: [u8; UID_SALT_BYTES], next_seq: u64) -> Self {
        UidAllocator { salt, next_seq }
    }

    /// A brand-new allocator with a fresh random salt and `seq = 0`.
    pub fn fresh(entropy: &dyn EntropySource) -> Result<Self, TenancyError> {
        let mut salt = [0u8; UID_SALT_BYTES];
        entropy.fill(&mut salt)?;
        Ok(UidAllocator::new(salt, 0))
    }

    /// Allocate the next uid, mixing a fresh [`UID_NONCE_BYTES`] nonce from
    /// `entropy`. Fails (never wraps) when `seq` is exhausted; on any failure
    /// `next_seq` is unchanged.
    pub fn allocate(&mut self, entropy: &dyn EntropySource) -> Result<CollectionUid, TenancyError> {
        let seq = self.next_seq;
        let next = seq
            .checked_add(1)
            .ok_or(TenancyError::UidSequenceExhausted)?;
        let mut nonce = [0u8; UID_NONCE_BYTES];
        entropy.fill(&mut nonce)?;
        self.next_seq = next;
        Ok(CollectionUid::derive(&self.salt, seq, &nonce))
    }

    /// The salt to persist.
    pub fn salt(&self) -> &[u8; UID_SALT_BYTES] {
        &self.salt
    }

    /// The next sequence number to persist.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_uses_entropy_port_and_starts_at_zero() {
        let mut mock = MockEntropySource::new();
        mock.expect_fill().times(2).returning(|out| {
            out.fill(7);
            Ok(())
        });
        let mut a = UidAllocator::fresh(&mock).unwrap();
        assert_eq!(a.salt(), &[7u8; 32]);
        assert_eq!(a.next_seq(), 0);
        assert_eq!(
            a.allocate(&mock).unwrap(),
            CollectionUid::derive(&[7u8; 32], 0, &[7u8; 16])
        );
        assert_eq!(a.next_seq(), 1);
    }

    #[test]
    fn fresh_propagates_entropy_failure() {
        let mut mock = MockEntropySource::new();
        mock.expect_fill()
            .returning(|_| Err(TenancyError::MalformedIdentifier("entropy")));
        assert!(UidAllocator::fresh(&mock).is_err());
    }

    #[test]
    fn allocate_entropy_failure_does_not_advance_seq() {
        let mut mock = MockEntropySource::new();
        mock.expect_fill()
            .returning(|_| Err(TenancyError::MalformedIdentifier("entropy")));
        let mut a = UidAllocator::new([0; 32], 5);
        assert!(a.allocate(&mock).is_err());
        assert_eq!(a.next_seq(), 5);
    }

    #[test]
    fn allocate_never_wraps() {
        let mut mock = MockEntropySource::new();
        mock.expect_fill().times(0);
        let mut a = UidAllocator::new([0; 32], u64::MAX);
        assert_eq!(a.allocate(&mock), Err(TenancyError::UidSequenceExhausted));
        assert_eq!(a.next_seq(), u64::MAX);
    }

    #[test]
    fn debug_redacts_salt() {
        let a = UidAllocator::new([0xAB; 32], 3);
        let s = format!("{a:?}");
        assert!(s.contains("<redacted>"));
        assert!(!s.contains("171"), "salt byte leaked: {s}");
        assert!(!s.to_lowercase().contains("ab, ab"), "salt leaked: {s}");
    }
}
