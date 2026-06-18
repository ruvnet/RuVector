//! Versioned chain verification: v2 rules, v1 backward verification,
//! and version-dispatched byte-stream verification (ADR-134 v2).
//!
//! Readers dispatch on the wire version byte at
//! [`rvm_types::WIRE_VERSION_OFFSET`] within each record: `0` means a
//! 64-byte v1 record (legacy, verify-only), `2` means a 96-byte v2
//! record. v1 records may only appear as a prefix (a log migrated to
//! v2 never writes v1 again); the first v2 record of a mixed log must
//! anchor the verified v1 head via [`v1_head_to_genesis`].

use crate::hash::{compute_chain_hash, compute_record_hash};
use crate::log::fold_u64_to_u32;
use crate::v2::compute_chain_mac_v2;
use crate::replay::ChainIntegrityError;
use rvm_types::{WitnessRecord, WitnessRecordV2, WIRE_VERSION_OFFSET};

/// Errors detected during v2 chain verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainIntegrityErrorV2 {
    /// The record's version byte is not `2`.
    VersionMismatch {
        /// Sequence number of the offending record.
        sequence: u64,
        /// The version byte found.
        found: u8,
    },
    /// The 128-bit chain link is broken at the given sequence
    /// (reordering, splicing, or a tampered predecessor).
    ChainBreak {
        /// Sequence number of the broken record.
        sequence: u64,
    },
    /// The record's recomputed chain MAC does not match the stored one
    /// (content tampering).
    RecordCorrupted {
        /// Sequence number of the corrupted record.
        sequence: u64,
    },
    /// The record slice is empty.
    EmptyLog,
}

impl core::fmt::Display for ChainIntegrityErrorV2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::VersionMismatch { sequence, found } => {
                write!(f, "version mismatch at seq {sequence}: found {found}")
            }
            Self::ChainBreak { sequence } => write!(f, "v2 chain break at seq {sequence}"),
            Self::RecordCorrupted { sequence } => {
                write!(f, "corrupted v2 record at seq {sequence}")
            }
            Self::EmptyLog => write!(f, "empty log"),
        }
    }
}

/// Verify a contiguous slice of v2 records chained from the zero genesis.
///
/// # Errors
///
/// See [`verify_chain_v2_from`].
pub fn verify_chain_v2(
    records: &[WitnessRecordV2],
    key: &[u8; 32],
) -> Result<usize, ChainIntegrityErrorV2> {
    verify_chain_v2_from(records, key, &[0u8; 16])
}

/// Verify a contiguous slice of v2 records chained from `genesis`.
///
/// For each record: checks `version == 2`, checks the full-width
/// 128-bit link (`prev_mac` must equal the running chain head), then
/// recomputes `trunc128(BLAKE3_keyed(key, content || prev_mac))` and
/// compares against the stored `chain_mac`. One keyed compression per
/// record -- the same cost the writer paid.
///
/// Tail truncation is not detectable from the slice alone; compare the
/// final record's `chain_mac` against an externally anchored head
/// ([`crate::WitnessLogV2::chain_head`]) or verify a sealed segment.
///
/// # Errors
///
/// [`ChainIntegrityErrorV2::EmptyLog`] for an empty slice;
/// [`ChainIntegrityErrorV2::VersionMismatch`] for a non-v2 record;
/// [`ChainIntegrityErrorV2::ChainBreak`] for a broken link;
/// [`ChainIntegrityErrorV2::RecordCorrupted`] for tampered content.
pub fn verify_chain_v2_from(
    records: &[WitnessRecordV2],
    key: &[u8; 32],
    genesis: &[u8; 16],
) -> Result<usize, ChainIntegrityErrorV2> {
    if records.is_empty() {
        return Err(ChainIntegrityErrorV2::EmptyLog);
    }
    let mut head: [u8; 16] = *genesis;
    for record in records {
        verify_record_v2(record, key, &mut head)?;
    }
    Ok(records.len())
}

/// Verify a single v2 record against the running chain head, advancing
/// the head on success.
fn verify_record_v2(
    record: &WitnessRecordV2,
    key: &[u8; 32],
    head: &mut [u8; 16],
) -> Result<(), ChainIntegrityErrorV2> {
    if record.version != WitnessRecordV2::VERSION {
        return Err(ChainIntegrityErrorV2::VersionMismatch {
            sequence: record.sequence,
            found: record.version,
        });
    }
    if record.prev_mac != *head {
        return Err(ChainIntegrityErrorV2::ChainBreak {
            sequence: record.sequence,
        });
    }
    let bytes = record.to_bytes();
    let expected = compute_chain_mac_v2(
        key,
        &bytes[..WitnessRecordV2::CONTENT_LEN],
        &record.prev_mac,
    );
    if record.chain_mac != expected {
        return Err(ChainIntegrityErrorV2::RecordCorrupted {
            sequence: record.sequence,
        });
    }
    *head = record.chain_mac;
    Ok(())
}

/// Verify a contiguous slice of v2 records that spans multiple chain-key
/// epochs (R4 ratchet).
///
/// `initial_key` is the key the log was created with (`key_0`);
/// `epoch_boundaries` lists, in ascending order, the sequence number at
/// which each ratchet fired (the log's `sequence` at each seal): records
/// with `sequence >= epoch_boundaries[k]` are verified under
/// `key_{k+1} = ratchet_chain_key(key_k)`. Under
/// [`crate::CoveragePolicy::Strict`] each boundary equals
/// `seal.first_sequence + seal.count` of the corresponding chained seal,
/// so the boundaries are recoverable from the seal chain alone; under
/// `BestEffort` with dropped leaves the boundary is the *next* seal's
/// `first_sequence`.
///
/// Capability asymmetry (the point of the ratchet): the holder of
/// `key_0` can verify the entire log, while the logger — which only
/// retains the latest epoch key — can no longer forge any record older
/// than its last seal.
///
/// # Errors
///
/// Same as [`verify_chain_v2_from`].
pub fn verify_chain_v2_ratcheted(
    records: &[WitnessRecordV2],
    initial_key: &[u8; 32],
    genesis: &[u8; 16],
    epoch_boundaries: &[u64],
) -> Result<usize, ChainIntegrityErrorV2> {
    if records.is_empty() {
        return Err(ChainIntegrityErrorV2::EmptyLog);
    }
    let mut key = *initial_key;
    let mut head: [u8; 16] = *genesis;
    let mut next_boundary = 0usize;
    for record in records {
        while next_boundary < epoch_boundaries.len()
            && record.sequence >= epoch_boundaries[next_boundary]
        {
            key = crate::v2::ratchet_chain_key(&key);
            next_boundary += 1;
        }
        verify_record_v2(record, &key, &mut head)?;
    }
    Ok(records.len())
}

/// Map a verified v1 chain head (the 64-bit running chain value) into a
/// 16-byte v2 genesis: `trunc128(BLAKE3("rvm-witness v1->v2 genesis" || head))`.
///
/// Migration: verify the v1 log, then create the v2 log with
/// [`crate::WitnessLogV2::with_key_and_genesis`] passing this value.
/// The first v2 record's `prev_mac` then cryptographically binds the
/// entire v1 history.
#[must_use]
pub fn v1_head_to_genesis(v1_head: u64) -> [u8; 16] {
    let mut buf = [0u8; 34];
    buf[..26].copy_from_slice(b"rvm-witness v1->v2 genesis");
    buf[26..34].copy_from_slice(&v1_head.to_le_bytes());
    let hash = blake3::hash(&buf);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

/// Outcome of a successful versioned byte-stream verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogVerifySummary {
    /// Number of v1 (64-byte) records verified.
    pub v1_count: usize,
    /// Number of v2 (96-byte) records verified.
    pub v2_count: usize,
    /// Final 128-bit chain head. For a pure-v1 log this is
    /// [`v1_head_to_genesis`] of the v1 head, so the value is always
    /// comparable against an anchored v2 head.
    pub head_mac: [u8; 16],
}

/// Errors from versioned byte-stream verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogVerifyError {
    /// The byte stream contains no records.
    Empty,
    /// The stream ends mid-record at the given byte offset.
    Truncated {
        /// Byte offset of the incomplete record.
        offset: usize,
    },
    /// An unknown version byte was found at the given record offset.
    UnknownVersion {
        /// Byte offset of the offending record.
        offset: usize,
        /// The version byte found.
        version: u8,
    },
    /// A v1 record appeared after a v2 record (v1 writing is removed;
    /// migrated logs never interleave back to v1).
    V1AfterV2 {
        /// Byte offset of the offending record.
        offset: usize,
    },
    /// A v1 record failed v1 chain verification.
    V1(ChainIntegrityError),
    /// A v2 record failed v2 chain verification.
    V2(ChainIntegrityErrorV2),
}

/// Verify a serialized witness log, dispatching on the per-record
/// version byte (offset 19 within each record).
///
/// Supports pure-v1 streams (legacy logs, verified under the v1 folded
/// rules), pure-v2 streams (zero genesis), and mixed v1-then-v2 streams
/// where the first v2 record must anchor the v1 head via
/// [`v1_head_to_genesis`].
///
/// # Errors
///
/// Returns [`LogVerifyError`] on structural problems (truncation,
/// unknown version, v1-after-v2) or on the first failed chain check.
#[allow(clippy::missing_panics_doc)] // slice->array conversions cannot fail
pub fn verify_log_bytes(
    bytes: &[u8],
    key: &[u8; 32],
) -> Result<LogVerifySummary, LogVerifyError> {
    let mut offset = 0usize;
    let mut v1_count = 0usize;
    let mut v2_count = 0usize;
    let mut v1_chain: u64 = 0;
    let mut seen_v2 = false;
    let mut v2_head = [0u8; 16];

    while offset < bytes.len() {
        if offset + WIRE_VERSION_OFFSET >= bytes.len() {
            return Err(LogVerifyError::Truncated { offset });
        }
        let version = bytes[offset + WIRE_VERSION_OFFSET];
        match version {
            0 => {
                if seen_v2 {
                    return Err(LogVerifyError::V1AfterV2 { offset });
                }
                if offset + 64 > bytes.len() {
                    return Err(LogVerifyError::Truncated { offset });
                }
                let mut raw = [0u8; 64];
                raw.copy_from_slice(&bytes[offset..offset + 64]);
                let record = WitnessRecord::from_bytes(&raw);
                verify_record_v1(&record, &mut v1_chain).map_err(LogVerifyError::V1)?;
                v1_count += 1;
                offset += 64;
            }
            2 => {
                if offset + WitnessRecordV2::SIZE > bytes.len() {
                    return Err(LogVerifyError::Truncated { offset });
                }
                let mut raw = [0u8; 96];
                raw.copy_from_slice(&bytes[offset..offset + WitnessRecordV2::SIZE]);
                let record = WitnessRecordV2::from_bytes(&raw);
                if !seen_v2 {
                    // Chain start: zero genesis for pure-v2 streams, or
                    // the anchored v1 head for migrated logs.
                    v2_head = if v1_count > 0 {
                        v1_head_to_genesis(v1_chain)
                    } else {
                        [0u8; 16]
                    };
                    seen_v2 = true;
                }
                verify_record_v2(&record, key, &mut v2_head).map_err(LogVerifyError::V2)?;
                v2_count += 1;
                offset += WitnessRecordV2::SIZE;
            }
            other => {
                return Err(LogVerifyError::UnknownVersion {
                    offset,
                    version: other,
                });
            }
        }
    }

    if v1_count + v2_count == 0 {
        return Err(LogVerifyError::Empty);
    }
    let head_mac = if seen_v2 {
        v2_head
    } else {
        v1_head_to_genesis(v1_chain)
    };
    Ok(LogVerifySummary {
        v1_count,
        v2_count,
        head_mac,
    })
}

/// Verify a single v1 record against the running 64-bit chain value,
/// advancing it on success (the legacy folded rules, unchanged).
fn verify_record_v1(
    record: &WitnessRecord,
    chain: &mut u64,
) -> Result<(), ChainIntegrityError> {
    let expected_prev = fold_u64_to_u32(*chain);
    if record.prev_hash != expected_prev {
        return Err(ChainIntegrityError::ChainBreak {
            sequence: record.sequence,
        });
    }
    let record_hash = compute_record_hash(&record.to_bytes()[..WitnessRecord::CONTENT_LEN]);
    if record.record_hash != fold_u64_to_u32(record_hash) {
        return Err(ChainIntegrityError::RecordCorrupted {
            sequence: record.sequence,
        });
    }
    *chain = compute_chain_hash(*chain, record.sequence, record_hash);
    Ok(())
}

/// Verify a v1 record slice and also return the final 64-bit chain head.
///
/// Identical rules to [`crate::replay::verify_chain`]; the head is what
/// [`v1_head_to_genesis`] consumes when migrating a log to v2.
///
/// # Errors
///
/// Same as [`crate::replay::verify_chain`].
pub fn verify_chain_v1_with_head(
    records: &[WitnessRecord],
) -> Result<(usize, u64), ChainIntegrityError> {
    if records.is_empty() {
        return Err(ChainIntegrityError::EmptyLog);
    }
    let mut chain: u64 = 0;
    for record in records {
        verify_record_v1(record, &mut chain)?;
    }
    Ok((records.len(), chain))
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;
    use super::*;
    use crate::log::WitnessLog;
    use crate::v2::{default_chain_key, WitnessLogV2};
    use rvm_types::ActionKind;

    fn build_v2_chain(count: usize) -> (Vec<WitnessRecordV2>, [u8; 32]) {
        let log = WitnessLogV2::<64>::new();
        for i in 0..count {
            let mut r = WitnessRecordV2::zeroed();
            r.action_kind = ActionKind::SchedulerEpoch as u8;
            r.actor_partition_id = (i as u32) % 3 + 1;
            r.target_object_id = (i as u64) * 10;
            r.timestamp_ns = (i as u64) * 1000 + 100;
            log.append(r);
        }
        let mut records = alloc::vec![WitnessRecordV2::zeroed(); count];
        let copied = log.snapshot(&mut records);
        records.truncate(copied);
        (records, log.chain_key())
    }

    fn build_v1_chain(count: usize) -> Vec<WitnessRecord> {
        let log = WitnessLog::<64>::new();
        for i in 0..count {
            let mut r = WitnessRecord::zeroed();
            r.action_kind = ActionKind::SchedulerEpoch as u8;
            r.actor_partition_id = 1;
            r.target_object_id = i as u64;
            r.timestamp_ns = (i as u64) * 100 + 7;
            log.append(r);
        }
        let mut records = alloc::vec![WitnessRecord::zeroed(); count];
        let copied = log.snapshot(&mut records);
        records.truncate(copied);
        records
    }

    fn serialize_mixed(v1: &[WitnessRecord], v2: &[WitnessRecordV2]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for r in v1 {
            bytes.extend_from_slice(&r.to_bytes());
        }
        for r in v2 {
            bytes.extend_from_slice(&r.to_bytes());
        }
        bytes
    }

    // ---- v2 chain verification -------------------------------------

    #[test]
    fn v2_valid_chain_verifies() {
        let (records, key) = build_v2_chain(6);
        assert_eq!(verify_chain_v2(&records, &key), Ok(6));
    }

    #[test]
    fn v2_empty_chain_rejected() {
        let key = default_chain_key();
        assert_eq!(
            verify_chain_v2(&[], &key),
            Err(ChainIntegrityErrorV2::EmptyLog)
        );
    }

    #[test]
    fn v2_content_tamper_detected() {
        let (mut records, key) = build_v2_chain(6);
        records[2].payload = [0xEE; 8];
        assert_eq!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 2 })
        );

        let (mut records, key) = build_v2_chain(6);
        records[4].action_kind = ActionKind::CapabilityGrant as u8;
        assert_eq!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 4 })
        );

        // Tampering the LAST record is detected too (no following link
        // needed): the keyed MAC itself fails without the chain key.
        let (mut records, key) = build_v2_chain(6);
        records[5].target_object_id = 0xDEAD;
        assert_eq!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 5 })
        );
    }

    #[test]
    fn v2_tampered_content_with_unkeyed_recompute_still_fails() {
        // An attacker WITHOUT the chain key cannot recompute a valid
        // chain_mac. Simulate the best unkeyed forgery: recompute with
        // a guessed (wrong) key.
        let (mut records, key) = build_v2_chain(4);
        records[1].payload = [0xAB; 8];
        let wrong_key = crate::v2::derive_chain_key(b"attacker-guess");
        let bytes = records[1].to_bytes();
        records[1].chain_mac = compute_chain_mac_v2(
            &wrong_key,
            &bytes[..WitnessRecordV2::CONTENT_LEN],
            &records[1].prev_mac,
        );
        assert_eq!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 1 })
        );
    }

    #[test]
    fn v2_reordering_detected() {
        let (mut records, key) = build_v2_chain(6);
        records.swap(2, 3);
        assert!(matches!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::ChainBreak { .. })
        ));
    }

    #[test]
    fn v2_front_truncation_detected() {
        let (records, key) = build_v2_chain(6);
        // Dropping the first record breaks the genesis link.
        assert!(matches!(
            verify_chain_v2(&records[1..], &key),
            Err(ChainIntegrityErrorV2::ChainBreak { .. })
        ));
    }

    #[test]
    fn v2_tail_truncation_detected_via_anchored_head() {
        // Tail truncation passes slice verification by construction,
        // but the anchored head no longer matches.
        let log = WitnessLogV2::<64>::new();
        for i in 0..6u64 {
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = i;
            log.append(r);
        }
        let anchored_head = log.chain_head();
        let mut records = alloc::vec![WitnessRecordV2::zeroed(); 6];
        log.snapshot(&mut records);
        let truncated = &records[..5];
        let key = log.chain_key();
        assert!(verify_chain_v2(truncated, &key).is_ok());
        assert_ne!(
            truncated.last().unwrap().chain_mac,
            anchored_head,
            "anchored head must expose tail truncation"
        );
        assert_eq!(records.last().unwrap().chain_mac, anchored_head);
    }

    #[test]
    fn v2_version_mismatch_detected() {
        let (mut records, key) = build_v2_chain(3);
        records[1].version = 1;
        assert_eq!(
            verify_chain_v2(&records, &key),
            Err(ChainIntegrityErrorV2::VersionMismatch {
                sequence: 1,
                found: 1
            })
        );
    }

    #[test]
    fn v2_wrong_key_fails() {
        let (records, _key) = build_v2_chain(3);
        let wrong = crate::v2::derive_chain_key(b"not-the-key");
        assert!(verify_chain_v2(&records, &wrong).is_err());
    }

    // ---- R4: ratcheted multi-epoch verification ---------------------

    use crate::seal::{
        verify_inclusion, verify_seal, verify_seal_chain, Blake3SealSigner,
    };
    use crate::v2::{ratchet_chain_key, CoveragePolicy};

    #[test]
    fn ratcheted_chain_verifies_with_rederived_keys() {
        let signer = Blake3SealSigner::new([9u8; 32]);
        let key0 = crate::v2::derive_chain_key(b"ratchet-verify");
        let log = WitnessLogV2::<64, 4>::with_policy(key0, CoveragePolicy::Strict);
        let mut boundaries = Vec::new();
        let mut seals = Vec::new();
        for i in 0..12u64 {
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = i;
            log.try_append(r).unwrap();
            if (i + 1) % 4 == 0 {
                let (sealed, _) = log.seal_segment(&signer).unwrap();
                // Strict: epoch boundary recoverable from the seal.
                boundaries.push(sealed.first_sequence + u64::from(sealed.count));
                seals.push(sealed);
            }
        }
        let mut records = alloc::vec![WitnessRecordV2::zeroed(); 12];
        assert_eq!(log.snapshot(&mut records), 12);

        // No single key verifies across epochs any more.
        assert!(verify_chain_v2(&records, &key0).is_err());
        assert!(verify_chain_v2(&records, &log.chain_key()).is_err());
        // The initial-key holder re-derives every epoch (determinism).
        assert_eq!(
            verify_chain_v2_ratcheted(&records, &key0, &[0u8; 16], &boundaries),
            Ok(12)
        );
        // Wrong boundaries pair records with the wrong epoch key.
        assert!(verify_chain_v2_ratcheted(&records, &key0, &[0u8; 16], &[3, 8, 12]).is_err());
        // The seals the boundaries came from are themselves a valid chain.
        assert_eq!(verify_seal_chain(&seals, &signer), Ok(3));
    }

    #[test]
    fn stale_key_cannot_forge_post_ratchet_record() {
        let signer = Blake3SealSigner::new([9u8; 32]);
        let log = WitnessLogV2::<16, 8>::new();
        let key0 = log.chain_key();
        let mut r = WitnessRecordV2::zeroed();
        r.target_object_id = 1;
        log.append(r);
        log.seal_segment(&signer).unwrap(); // ratchet: key0 retired
        let mut r = WitnessRecordV2::zeroed();
        r.target_object_id = 2;
        log.append(r);

        let mut records = alloc::vec![WitnessRecordV2::zeroed(); 2];
        log.snapshot(&mut records);
        let boundaries = [1u64];
        assert_eq!(
            verify_chain_v2_ratcheted(&records, &key0, &[0u8; 16], &boundaries),
            Ok(2)
        );

        // An attacker who only kept the pre-ratchet key tampers the
        // epoch-1 record and recomputes its MAC with the stale key.
        records[1].target_object_id = 0xE71;
        let bytes = records[1].to_bytes();
        records[1].chain_mac = compute_chain_mac_v2(
            &key0,
            &bytes[..WitnessRecordV2::CONTENT_LEN],
            &records[1].prev_mac,
        );
        assert_eq!(
            verify_chain_v2_ratcheted(&records, &key0, &[0u8; 16], &boundaries),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 1 })
        );
    }

    #[test]
    fn post_compromise_key_cannot_rewrite_sealed_history() {
        // THE forward-security property: the attacker holds the
        // *current* (post-ratchet) chain key and rewrites a record in
        // an already-sealed segment, recomputing every downstream MAC
        // with the compromised key — the best forgery available
        // without key_0.
        let signer = Blake3SealSigner::new([7u8; 32]);
        let key0 = crate::v2::derive_chain_key(b"compromise-window");
        let log = WitnessLogV2::<64, 4>::with_policy(key0, CoveragePolicy::Strict);
        for i in 0..4u64 {
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = i;
            log.try_append(r).unwrap();
        }
        let (seal0, acc0) = log.seal_segment(&signer).unwrap();
        for i in 4..6u64 {
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = i;
            log.try_append(r).unwrap();
        }
        let compromised = log.chain_key(); // attacker's loot: key_1
        assert_eq!(compromised, ratchet_chain_key(&key0));

        let mut records = alloc::vec![WitnessRecordV2::zeroed(); 6];
        log.snapshot(&mut records);
        records[2].payload = [0xEE; 8];
        let mut head = records[1].chain_mac;
        for r in &mut records[2..] {
            r.prev_mac = head;
            let bytes = r.to_bytes();
            r.chain_mac = compute_chain_mac_v2(
                &compromised,
                &bytes[..WitnessRecordV2::CONTENT_LEN],
                &r.prev_mac,
            );
            head = r.chain_mac;
        }

        // 1. The initial-key holder rejects: epoch-0 records carry
        //    MACs the attacker could not have produced.
        assert_eq!(
            verify_chain_v2_ratcheted(&records, &key0, &[0u8; 16], &[4]),
            Err(ChainIntegrityErrorV2::RecordCorrupted { sequence: 2 })
        );

        // 2. Even WITHOUT key_0, the sealed history catches the
        //    rewrite: old segments are protected by their SEALS, not
        //    the (compromised) chain key. The honest leaf still proves
        //    inclusion; the forged record's MAC does not.
        let proof = acc0.proof_for_sequence(2).unwrap();
        assert!(verify_seal(&seal0, &signer));
        assert!(verify_inclusion(&seal0.root, &acc0.leaf(2).unwrap(), &proof));
        assert!(!verify_inclusion(&seal0.root, &records[2].chain_mac, &proof));
    }

    #[test]
    fn ratcheted_verify_with_no_boundaries_matches_plain() {
        let (records, key) = build_v2_chain(5);
        assert_eq!(
            verify_chain_v2_ratcheted(&records, &key, &[0u8; 16], &[]),
            Ok(5)
        );
        assert_eq!(
            verify_chain_v2_ratcheted(&[], &key, &[0u8; 16], &[]),
            Err(ChainIntegrityErrorV2::EmptyLog)
        );
    }

    // ---- v1 backward verification ----------------------------------

    #[test]
    fn v1_logs_still_verify_with_head() {
        let records = build_v1_chain(5);
        let (count, head) = verify_chain_v1_with_head(&records).unwrap();
        assert_eq!(count, 5);
        assert_ne!(head, 0);
        // Must agree with the legacy verifier.
        assert_eq!(crate::replay::verify_chain(&records), Ok(5));
    }

    #[test]
    fn v1_round_trips_through_bytes() {
        let records = build_v1_chain(4);
        for r in &records {
            let rt = WitnessRecord::from_bytes(&r.to_bytes());
            assert_eq!(rt.to_bytes(), r.to_bytes());
        }
    }

    #[test]
    fn v2_round_trips_through_bytes() {
        let (records, _) = build_v2_chain(4);
        for r in &records {
            let rt = WitnessRecordV2::from_bytes(&r.to_bytes());
            assert_eq!(rt.to_bytes(), r.to_bytes());
        }
    }

    // ---- versioned byte-stream verification ------------------------

    #[test]
    fn pure_v1_stream_verifies() {
        let v1 = build_v1_chain(5);
        let bytes = serialize_mixed(&v1, &[]);
        let key = default_chain_key();
        let summary = verify_log_bytes(&bytes, &key).unwrap();
        assert_eq!(summary.v1_count, 5);
        assert_eq!(summary.v2_count, 0);
        let (_, head) = verify_chain_v1_with_head(&v1).unwrap();
        assert_eq!(summary.head_mac, v1_head_to_genesis(head));
    }

    #[test]
    fn pure_v2_stream_verifies() {
        let (v2, key) = build_v2_chain(4);
        let bytes = serialize_mixed(&[], &v2);
        let summary = verify_log_bytes(&bytes, &key).unwrap();
        assert_eq!(summary.v1_count, 0);
        assert_eq!(summary.v2_count, 4);
        assert_eq!(summary.head_mac, v2.last().unwrap().chain_mac);
    }

    #[test]
    fn mixed_v1_then_v2_stream_verifies_when_anchored() {
        let v1 = build_v1_chain(3);
        let (_, head) = verify_chain_v1_with_head(&v1).unwrap();

        // Migrate: new v2 log anchored to the v1 head.
        let key = default_chain_key();
        let log = WitnessLogV2::<16>::with_key_and_genesis(key, v1_head_to_genesis(head));
        for i in 0..4u64 {
            let mut r = WitnessRecordV2::zeroed();
            r.action_kind = ActionKind::RegionMap as u8;
            r.target_object_id = i;
            log.append(r);
        }
        let mut v2 = alloc::vec![WitnessRecordV2::zeroed(); 4];
        log.snapshot(&mut v2);

        let bytes = serialize_mixed(&v1, &v2);
        let summary = verify_log_bytes(&bytes, &key).unwrap();
        assert_eq!(summary.v1_count, 3);
        assert_eq!(summary.v2_count, 4);
        assert_eq!(summary.head_mac, log.chain_head());
    }

    #[test]
    fn mixed_stream_with_unanchored_v2_rejected() {
        let v1 = build_v1_chain(3);
        // v2 chain that starts from zero genesis (NOT anchored to v1).
        let (v2, key) = build_v2_chain(2);
        let bytes = serialize_mixed(&v1, &v2);
        assert!(matches!(
            verify_log_bytes(&bytes, &key),
            Err(LogVerifyError::V2(ChainIntegrityErrorV2::ChainBreak { .. }))
        ));
    }

    #[test]
    fn mixed_stream_v1_tamper_detected() {
        let v1 = build_v1_chain(3);
        let (_, head) = verify_chain_v1_with_head(&v1).unwrap();
        let key = default_chain_key();
        let log = WitnessLogV2::<16>::with_key_and_genesis(key, v1_head_to_genesis(head));
        let mut r = WitnessRecordV2::zeroed();
        r.target_object_id = 1;
        log.append(r);
        let mut v2 = alloc::vec![WitnessRecordV2::zeroed(); 1];
        log.snapshot(&mut v2);

        let mut bytes = serialize_mixed(&v1, &v2);
        // Tamper a v1 payload byte (offset 36 of record 1).
        bytes[64 + 36] ^= 0xFF;
        assert!(matches!(
            verify_log_bytes(&bytes, &key),
            Err(LogVerifyError::V1(ChainIntegrityError::RecordCorrupted { .. }))
        ));
    }

    #[test]
    fn v1_after_v2_rejected() {
        let v1 = build_v1_chain(1);
        let (v2, key) = build_v2_chain(1);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&v2[0].to_bytes());
        bytes.extend_from_slice(&v1[0].to_bytes());
        assert!(matches!(
            verify_log_bytes(&bytes, &key),
            Err(LogVerifyError::V1AfterV2 { offset: 96 })
        ));
    }

    #[test]
    fn truncated_stream_rejected() {
        let (v2, key) = build_v2_chain(2);
        let bytes = serialize_mixed(&[], &v2);
        // Cut mid-record.
        assert!(matches!(
            verify_log_bytes(&bytes[..96 + 40], &key),
            Err(LogVerifyError::Truncated { offset: 96 })
        ));
    }

    #[test]
    fn unknown_version_rejected() {
        let (v2, key) = build_v2_chain(1);
        let mut bytes = serialize_mixed(&[], &v2);
        bytes[WIRE_VERSION_OFFSET] = 7;
        assert!(matches!(
            verify_log_bytes(&bytes, &key),
            Err(LogVerifyError::UnknownVersion {
                offset: 0,
                version: 7
            })
        ));
    }

    #[test]
    fn empty_stream_rejected() {
        let key = default_chain_key();
        assert_eq!(verify_log_bytes(&[], &key), Err(LogVerifyError::Empty));
    }
}
