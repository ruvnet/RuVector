//! Segment serialization: the 64-byte header, the payload, and the optional
//! Ed25519 signature footer that follows it.
//!
//! This is the single place the segment layout is written. [`crate::testkit`]
//! calls it too, so fixtures cannot drift from authored output.

use super::ED25519_SIGNATURE_LENGTH;
use ed25519_dalek::SigningKey;
use rvf_types::{
    SegmentFlags, SegmentType, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE, SEGMENT_VERSION,
};

/// Serialize one complete segment: header, payload, optional signature footer,
/// zero padding to the next 64-byte boundary.
///
/// This is the single place the segment layout is written. [`crate::testkit`]
/// calls it too, so fixtures cannot drift from authored output.
pub(crate) fn write_segment_bytes(
    seg_type: SegmentType,
    payload: &[u8],
    segment_id: u64,
    checksum_algo: u8,
    key: Option<&SigningKey>,
) -> Vec<u8> {
    let footer_len = key
        .map(|_| {
            rvf_types::SignatureFooter::compute_footer_length(ED25519_SIGNATURE_LENGTH as u16)
                as usize
        })
        .unwrap_or(0);
    let unpadded = SEGMENT_HEADER_SIZE + payload.len() + footer_len;
    let pad = unpadded.div_ceil(SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT - unpadded;

    let flags = match key {
        Some(_) => SegmentFlags::empty().with(SegmentFlags::SIGNED),
        None => SegmentFlags::empty(),
    };
    let head = segment_header(
        seg_type as u8,
        payload,
        flags,
        segment_id,
        checksum_algo,
        pad as u32,
    );

    let mut out = Vec::with_capacity(unpadded + pad);
    out.extend_from_slice(&head);
    out.extend_from_slice(payload);
    if let Some(key) = key {
        // `read_segment_header` cannot fail on bytes we just serialized; the
        // signature covers the header as the verifier will re-read it.
        let header = rvf_wire::read_segment_header(&head)
            .expect("author serialized a well-formed segment header");
        let footer = rvf_crypto::sign_segment(&header, payload, key);
        out.extend_from_slice(&rvf_crypto::encode_signature_footer(&footer));
    }
    out.resize(unpadded + pad, 0);
    out
}

/// Serialize a 64-byte segment header.
fn segment_header(
    seg_type: u8,
    payload: &[u8],
    flags: SegmentFlags,
    segment_id: u64,
    checksum_algo: u8,
    alignment_pad: u32,
) -> [u8; SEGMENT_HEADER_SIZE] {
    let content_hash = rvf_wire::hash::compute_content_hash(checksum_algo, payload);

    let mut b = [0u8; SEGMENT_HEADER_SIZE];
    b[0x00..0x04].copy_from_slice(&rvf_types::SEGMENT_MAGIC.to_le_bytes());
    b[0x04] = SEGMENT_VERSION;
    b[0x05] = seg_type;
    b[0x06..0x08].copy_from_slice(&flags.bits().to_le_bytes());
    b[0x08..0x10].copy_from_slice(&segment_id.to_le_bytes());
    b[0x10..0x18].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    // timestamp_ns stays zero: authored containers must be byte-reproducible.
    b[0x20] = checksum_algo;
    b[0x28..0x38].copy_from_slice(&content_hash);
    b[0x3C..0x40].copy_from_slice(&alignment_pad.to_le_bytes());
    b
}
