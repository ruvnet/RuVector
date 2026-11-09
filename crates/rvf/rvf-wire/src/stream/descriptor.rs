//! Descriptor decoding: fixed layout, length-checked before any allocation
//! proportional to declared counts, authenticated before the table is parsed.

use super::{EpochPin, StreamPolicy};
use alloc::vec::Vec;
use rvf_crypto::{descriptor_hash, signer_key_id, verify_descriptor};
use rvf_types::stream::*;
use rvf_types::ErrorCode;

extern crate alloc;

/// A descriptor that passed every ADR-353 section 2 invariant.
#[derive(Clone, Debug)]
pub struct VerifiedDescriptor {
    /// Validated fixed header.
    pub header: StreamHeader,
    /// Table entries in body order.
    pub entries: Vec<StreamEntry>,
    /// SHA-256 of the complete signed descriptor.
    pub descriptor_hash: [u8; 32],
    /// The trusted key that verified the signature.
    pub signer: [u8; 32],
}

/// Decode and fully verify a signed descriptor under `policy`.
///
/// Order of checks (the first failure is reported):
/// 1. size cap; fixed header; `seg_count` cap; exact total length;
/// 2. `sig_len`; trusted signer lookup; Ed25519 `verify_strict`;
/// 3. expiry (with clock); expected `policy_hash`; epoch pin;
/// 4. table invariants and size policy.
pub fn decode_descriptor(
    bytes: &[u8],
    policy: &StreamPolicy<'_>,
) -> Result<VerifiedDescriptor, ErrorCode> {
    if bytes.len() > policy.max_descriptor_bytes {
        return Err(ErrorCode::StreamLimitExceeded);
    }
    let h = StreamHeader::parse(bytes)?;
    if h.seg_count > policy.max_segments {
        return Err(ErrorCode::StreamLimitExceeded);
    }
    if stream_descriptor_len(h.seg_count) != Some(bytes.len()) {
        return Err(ErrorCode::StreamDescriptorTruncated);
    }
    let sig_len_at = STREAM_HEADER_SIZE + h.table_len as usize;
    if u16::from_le_bytes([bytes[sig_len_at], bytes[sig_len_at + 1]]) as usize != STREAM_SIG_LEN {
        return Err(ErrorCode::StreamDescriptorTruncated);
    }

    let mut signer = None;
    let mut known = false;
    for pk in policy.trusted_keys {
        if signer_key_id(pk) == h.signer_key_id {
            known = true;
            if verify_descriptor(bytes, pk) {
                signer = Some(*pk);
                break;
            }
        }
    }
    let signer = match signer {
        Some(s) => s,
        None if known => return Err(ErrorCode::StreamBadSignature),
        None => return Err(ErrorCode::UnknownSigner),
    };

    if h.expires_ns != 0 {
        if let Some(now) = policy.now_ns {
            if now >= h.expires_ns {
                return Err(ErrorCode::StreamExpired);
            }
        }
    }
    if let Some(p) = policy.expected_policy_hash {
        if p != h.policy_hash {
            return Err(ErrorCode::StreamPolicyMismatch);
        }
    }
    let dhash = descriptor_hash(bytes);
    let check = |pin: Option<EpochPin>| -> Result<(), ErrorCode> {
        if let Some(pin) = pin {
            if h.epoch < pin.epoch {
                return Err(ErrorCode::StreamStaleEpoch);
            }
            if h.epoch == pin.epoch && pin.descriptor_hash != dhash {
                return Err(ErrorCode::StreamEpochConflict);
            }
        }
        Ok(())
    };
    if let Some(lookup) = policy.epoch_pin {
        check(lookup(&h.signer_key_id, &h.file_id))?;
    }
    if let Some(store) = policy.pins {
        check(store.get(&h.signer_key_id, &h.file_id))?;
    }

    let entries = parse_table(&bytes[STREAM_HEADER_SIZE..sig_len_at], &h, policy)?;
    Ok(VerifiedDescriptor {
        header: h,
        entries,
        descriptor_hash: dhash,
        signer,
    })
}

fn parse_table(
    table: &[u8],
    h: &StreamHeader,
    policy: &StreamPolicy<'_>,
) -> Result<Vec<StreamEntry>, ErrorCode> {
    use ErrorCode::{StreamLimitExceeded as Limit, StreamTableInvalid as Invalid};
    if h.seg_count == 0 {
        return Err(Invalid);
    }
    if h.body_len > policy.max_body_len {
        return Err(Limit);
    }
    let mut entries = Vec::with_capacity(h.seg_count as usize);
    let mut expect_off = 0u64;
    let mut class_total = [0u64; StreamClass::COUNT];
    let mut union_req = [0u32; STREAM_MAX_STAGES as usize];
    for (i, raw) in table.chunks_exact(STREAM_ENTRY_SIZE).enumerate() {
        let mut a = [0u8; STREAM_ENTRY_SIZE];
        a.copy_from_slice(raw);
        let e = StreamEntry::from_bytes(&a);
        if e.body_offset != expect_off || e.stored_len == 0 {
            return Err(Invalid);
        }
        if e.flags != 0 {
            return Err(Invalid);
        }
        if e.seg_type == STREAM_SEG_TYPE_ROOT_PAGE {
            if i + 1 != h.seg_count as usize
                || e.stored_len != rvf_types::ROOT_MANIFEST_SIZE as u64
                || e.uncompressed_len != 0
            {
                return Err(Invalid);
            }
        } else if e.stored_len < 64 {
            return Err(Invalid);
        }
        if e.stage >= STREAM_MAX_STAGES || e.requires_mask & (1u32 << e.stage) != 0 {
            return Err(Invalid);
        }
        if e.stored_len > policy.max_segment_bytes
            || e.uncompressed_len > policy.max_uncompressed_len
        {
            return Err(Limit);
        }
        if h.chunk_size().is_none() && e.stored_len > u32::MAX as u64 {
            return Err(Limit);
        }
        if h.chunk_count(e.stored_len).is_none() {
            return Err(Limit);
        }
        let c = StreamClass::of(e.seg_type) as usize;
        class_total[c] = class_total[c].checked_add(e.stored_len).ok_or(Limit)?;
        if class_total[c] > policy.max_class_bytes[c] {
            return Err(Limit);
        }
        expect_off = e.body_offset.checked_add(e.stored_len).ok_or(Invalid)?;
        union_req[e.stage as usize] |= e.requires_mask;
        entries.push(e);
    }
    if expect_off != h.body_len {
        return Err(Invalid);
    }
    let mut ids: Vec<u64> = entries.iter().map(|e| e.seg_id).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(Invalid);
    }
    stages_acyclic(&union_req)?;
    Ok(entries)
}

/// Stage `s` may require stage `t` only if `t` does not (transitively)
/// require `s`; otherwise no delivery order could ever release a chunk.
fn stages_acyclic(req: &[u32; STREAM_MAX_STAGES as usize]) -> Result<(), ErrorCode> {
    let mut done = 0u32;
    for _ in 0..STREAM_MAX_STAGES {
        let mut progressed = false;
        for (s, r) in req.iter().enumerate() {
            if done & (1 << s) == 0 && r & !done == 0 {
                done |= 1 << s;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    if done == u32::MAX {
        Ok(())
    } else {
        Err(ErrorCode::StreamTableInvalid)
    }
}
