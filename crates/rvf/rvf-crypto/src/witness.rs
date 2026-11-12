//! WITNESS_SEG support for cryptographic audit trails.
//!
//! Each witness entry chains to the previous via hashes, forming a
//! tamper-evident log. The chain uses SHAKE-256 for hash binding.

use alloc::vec::Vec;
use rvf_types::{ErrorCode, RvfError};

use crate::hash::shake256_256;

/// A single entry in a witness chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessEntry {
    /// Hash of the previous entry (zero for the first entry).
    pub prev_hash: [u8; 32],
    /// Hash of the action being witnessed.
    pub action_hash: [u8; 32],
    /// Nanosecond UNIX timestamp.
    pub timestamp_ns: u64,
    /// Witness type: 0x01=PROVENANCE, 0x02=COMPUTATION, etc.
    pub witness_type: u8,
}

/// Size of one serialized witness entry: 32 + 32 + 8 + 1 = 73 bytes.
const ENTRY_SIZE: usize = 73;

/// Serialize a `WitnessEntry` into bytes.
fn encode_entry(entry: &WitnessEntry) -> [u8; ENTRY_SIZE] {
    let mut buf = [0u8; ENTRY_SIZE];
    buf[0..32].copy_from_slice(&entry.prev_hash);
    buf[32..64].copy_from_slice(&entry.action_hash);
    buf[64..72].copy_from_slice(&entry.timestamp_ns.to_le_bytes());
    buf[72] = entry.witness_type;
    buf
}

/// Deserialize a `WitnessEntry` from bytes.
///
/// # Errors
///
/// Returns `TruncatedSegment` if `data` is shorter than `ENTRY_SIZE` (73) bytes.
fn decode_entry(data: &[u8]) -> Result<WitnessEntry, RvfError> {
    if data.len() < ENTRY_SIZE {
        return Err(RvfError::Code(ErrorCode::TruncatedSegment));
    }
    let mut prev_hash = [0u8; 32];
    prev_hash.copy_from_slice(&data[0..32]);
    let mut action_hash = [0u8; 32];
    action_hash.copy_from_slice(&data[32..64]);
    let timestamp_ns = u64::from_le_bytes(data[64..72].try_into().unwrap());
    let witness_type = data[72];
    Ok(WitnessEntry {
        prev_hash,
        action_hash,
        timestamp_ns,
        witness_type,
    })
}

/// Size in bytes of one serialized chain entry (the unit WITNESS_SEG payloads
/// are made of).
pub const WITNESS_ENTRY_SIZE: usize = ENTRY_SIZE;

/// Encode one chain entry linked to `prev_hash`: the `action_hash` field is
/// SHAKE-256 of `action`. This is the record the runtime appends to the file.
pub fn encode_chain_entry(
    prev_hash: &[u8; 32],
    action: &[u8],
    timestamp_ns: u64,
    witness_type: u8,
) -> [u8; ENTRY_SIZE] {
    encode_entry(&WitnessEntry {
        prev_hash: *prev_hash,
        action_hash: shake256_256(action),
        timestamp_ns,
        witness_type,
    })
}

/// The hash the NEXT entry must carry as its `prev_hash`: SHAKE-256 of this
/// entry's serialized bytes.
pub fn chain_entry_hash(entry_bytes: &[u8]) -> [u8; 32] {
    shake256_256(entry_bytes)
}

/// Create a witness chain from entries, linking each to the previous via hashes.
///
/// The first entry's `prev_hash` is set to all zeros (genesis).
/// Subsequent entries have `prev_hash` = SHAKE-256(previous entry bytes).
///
/// Returns the serialized chain as a byte vector.
pub fn create_witness_chain(entries: &[WitnessEntry]) -> Vec<u8> {
    let mut chain = Vec::with_capacity(entries.len() * ENTRY_SIZE);
    let mut prev_hash = [0u8; 32];

    for entry in entries {
        let mut linked = entry.clone();
        linked.prev_hash = prev_hash;
        let encoded = encode_entry(&linked);
        prev_hash = shake256_256(&encoded);
        chain.extend_from_slice(&encoded);
    }

    chain
}

/// How a WITNESS_SEG payload is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessPayloadKind {
    /// One or more fixed 73-byte chain entries (current format).
    Chain,
    /// The variable-length record written by rvf-runtime 0.4.0 and earlier
    /// (`type u8 | timestamp u64 | action_len u32 | action | prev_hash[32]`).
    /// Its hash is not the SHAKE-256 chain, so it cannot be chain-verified.
    Legacy,
    /// Neither layout.
    Unrecognized,
}

/// A parsed legacy (pre-chain) witness record. Informational only: nothing in
/// it can be cryptographically verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyWitnessRecord {
    pub witness_type: u8,
    pub timestamp_ns: u64,
    pub action: Vec<u8>,
    pub prev_hash: [u8; 32],
}

/// Parse the legacy record layout; `None` unless the lengths agree exactly.
pub fn parse_legacy_witness_record(p: &[u8]) -> Option<LegacyWitnessRecord> {
    if p.len() < 1 + 8 + 4 + 32 {
        return None;
    }
    let action_len = u32::from_le_bytes(p[9..13].try_into().ok()?) as usize;
    if p.len() != 13usize.checked_add(action_len)?.checked_add(32)? {
        return None;
    }
    Some(LegacyWitnessRecord {
        witness_type: p[0],
        timestamp_ns: u64::from_le_bytes(p[1..9].try_into().ok()?),
        action: p[13..13 + action_len].to_vec(),
        prev_hash: p[13 + action_len..].try_into().ok()?,
    })
}

/// Shape-only classification of a WITNESS_SEG payload (nothing is verified).
///
/// A payload that parses EXACTLY as a legacy record is `Legacy`, including the
/// rare 73-byte legacy record (45 + a 28-byte action) that has the same length
/// as one chain entry; a real chain entry parses as legacy only if bytes 9..13
/// of its `prev_hash` happen to equal 28 (about 2^-32). `Legacy` is never
/// "valid", so misclassifying a tampered chain entry as legacy cannot make a
/// file verify. Verifiers must try chain verification first and fall back to
/// this only when it fails.
pub fn classify_witness_payload(p: &[u8]) -> WitnessPayloadKind {
    if parse_legacy_witness_record(p).is_some() {
        WitnessPayloadKind::Legacy
    } else if p.is_empty() || p.len().is_multiple_of(ENTRY_SIZE) {
        WitnessPayloadKind::Chain
    } else {
        WitnessPayloadKind::Unrecognized
    }
}

/// Verify a witness chain's integrity.
///
/// Checks that each entry's `prev_hash` matches the SHAKE-256 hash of the
/// preceding entry. Returns the decoded entries if valid.
pub fn verify_witness_chain(data: &[u8]) -> Result<Vec<WitnessEntry>, RvfError> {
    verify_witness_chain_from(data, &[0u8; 32]).map(|(entries, _)| entries)
}

/// Like [`verify_witness_chain`], but the first entry must link to `start_prev`
/// (so a chain split across several WITNESS_SEGs can be verified end to end)
/// and the hash the next entry must carry is returned with the entries.
pub fn verify_witness_chain_from(
    data: &[u8],
    start_prev: &[u8; 32],
) -> Result<(Vec<WitnessEntry>, [u8; 32]), RvfError> {
    if data.is_empty() {
        return Ok((Vec::new(), *start_prev));
    }
    if !data.len().is_multiple_of(ENTRY_SIZE) {
        return Err(RvfError::Code(ErrorCode::TruncatedSegment));
    }

    let count = data.len() / ENTRY_SIZE;
    let mut entries = Vec::with_capacity(count);
    let mut expected_prev = *start_prev;

    for i in 0..count {
        let offset = i * ENTRY_SIZE;
        let entry_bytes = &data[offset..offset + ENTRY_SIZE];
        let entry = decode_entry(entry_bytes)?;

        if entry.prev_hash != expected_prev {
            return Err(RvfError::Code(ErrorCode::InvalidChecksum));
        }

        expected_prev = shake256_256(entry_bytes);
        entries.push(entry);
    }

    Ok((entries, expected_prev))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n: usize) -> Vec<WitnessEntry> {
        (0..n)
            .map(|i| WitnessEntry {
                prev_hash: [0u8; 32], // will be overwritten by create_witness_chain
                action_hash: shake256_256(&[i as u8]),
                timestamp_ns: 1_000_000_000 + i as u64,
                witness_type: 0x01,
            })
            .collect()
    }

    #[test]
    fn empty_chain() {
        let chain = create_witness_chain(&[]);
        assert!(chain.is_empty());
        let result = verify_witness_chain(&chain).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn single_entry_chain() {
        let entries = make_entries(1);
        let chain = create_witness_chain(&entries);
        assert_eq!(chain.len(), ENTRY_SIZE);
        let verified = verify_witness_chain(&chain).unwrap();
        assert_eq!(verified.len(), 1);
        assert_eq!(verified[0].prev_hash, [0u8; 32]);
    }

    #[test]
    fn multi_entry_chain() {
        let entries = make_entries(5);
        let chain = create_witness_chain(&entries);
        assert_eq!(chain.len(), 5 * ENTRY_SIZE);
        let verified = verify_witness_chain(&chain).unwrap();
        assert_eq!(verified.len(), 5);
        for (i, entry) in verified.iter().enumerate() {
            assert_eq!(entry.action_hash, entries[i].action_hash);
            assert_eq!(entry.timestamp_ns, entries[i].timestamp_ns);
        }
    }

    #[test]
    fn tampered_chain_detected() {
        let entries = make_entries(3);
        let mut chain = create_witness_chain(&entries);
        // Tamper with the second entry's action_hash byte
        chain[ENTRY_SIZE + 32] ^= 0xFF;
        let result = verify_witness_chain(&chain);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_chain_detected() {
        let entries = make_entries(2);
        let chain = create_witness_chain(&entries);
        let result = verify_witness_chain(&chain[..ENTRY_SIZE + 10]);
        assert!(result.is_err());
    }

    #[test]
    fn encode_chain_entry_matches_create_witness_chain() {
        let a = encode_chain_entry(&[0u8; 32], b"one", 10, 0x02);
        let h = chain_entry_hash(&a);
        let b = encode_chain_entry(&h, b"two", 11, 0x02);
        let mut chain = a.to_vec();
        chain.extend_from_slice(&b);
        let (entries, next) = verify_witness_chain_from(&chain, &[0u8; 32]).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(next, chain_entry_hash(&b));
        assert_eq!(entries[0].action_hash, shake256_256(b"one"));
        // A chain that starts mid-way verifies only from the right anchor.
        assert!(verify_witness_chain_from(&b, &[0u8; 32]).is_err());
        assert!(verify_witness_chain_from(&b, &h).is_ok());
    }

    #[test]
    fn legacy_record_is_recognised_and_never_valid_as_chain() {
        // The 70-byte record shape rvf-runtime 0.4.0 wrote for an ingest.
        let action = b"ingest:count=1000,epoch=2";
        let mut p = vec![0x02u8];
        p.extend_from_slice(&123u64.to_le_bytes());
        p.extend_from_slice(&(action.len() as u32).to_le_bytes());
        p.extend_from_slice(action);
        p.extend_from_slice(&[0u8; 32]);
        assert_eq!(classify_witness_payload(&p), WitnessPayloadKind::Legacy);
        let rec = parse_legacy_witness_record(&p).unwrap();
        assert_eq!(rec.action, action);
        assert!(
            verify_witness_chain(&p).is_err(),
            "legacy must not verify as a chain"
        );
        assert_eq!(
            classify_witness_payload(&[1, 2, 3]),
            WitnessPayloadKind::Unrecognized
        );
    }

    #[test]
    fn chain_links_are_correct() {
        let entries = make_entries(3);
        let chain = create_witness_chain(&entries);
        let verified = verify_witness_chain(&chain).unwrap();
        // First entry has zero prev_hash
        assert_eq!(verified[0].prev_hash, [0u8; 32]);
        // Second entry's prev_hash should equal hash of first entry's bytes
        let first_bytes = &chain[0..ENTRY_SIZE];
        let expected = shake256_256(first_bytes);
        assert_eq!(verified[1].prev_hash, expected);
    }
}
