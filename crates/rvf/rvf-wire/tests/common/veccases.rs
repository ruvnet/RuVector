//! The negative-case table of the RVSD vector generator.
use super::common::*;
use super::vecmodel::*;
use rvf_types::ErrorCode;
use rvf_wire::stream::*;

pub fn pos(fr: &[Vec<u8>], ty: u8, seg: u64, idx: Option<u32>) -> usize {
    fr.iter()
        .position(|f| {
            f[0] == ty
                && f[1..9] == seg.to_le_bytes()
                && idx.is_none_or(|i| f[9..13] == i.to_le_bytes())
        })
        .unwrap()
}

pub struct Case {
    pub name: &'static str,
    pub stream: &'static str,
    pub what: &'static str,
    pub edits: Vec<Edit>,
    pub pol: Pol,
    pub reason: ErrorCode,
    pub state: &'static str,
}

pub fn cases(m1f: &Stream, m1s: &Stream, m2f: &Stream, other: &Stream) -> Vec<Case> {
    use ErrorCode as E;
    let fr = &m2f.frames;
    let (c0, c1) = (pos(fr, 1, 3, Some(0)), pos(fr, 1, 3, Some(1)));
    let t3 = pos(fr, 2, 3, None);
    let s5 = pos(fr, 1, 5, Some(0));
    let bundle = &m2f.bundle;
    let sk = key(0x2A);
    let df = |d: Vec<u8>| Edit::Replace {
        frame: 0,
        bytes: descriptor_frame(&d),
    };
    let c = |name, what, edits, pol, reason, state| Case {
        name,
        stream: "m2-file",
        what,
        edits,
        pol,
        reason,
        state,
    };
    let p0 = Pol::std(0);
    let mut v = vec![
        c(
            "descriptor-flip-epoch-bit",
            "bit flip in a signed header field",
            vec![Edit::Xor {
                frame: 0,
                offset: 5 + 0x20,
                mask: 1,
            }],
            p0.clone(),
            E::StreamBadSignature,
            "Rejected:StreamBadSignature",
        ),
        c(
            "descriptor-flip-table-bit",
            "bit flip inside the segment table",
            vec![Edit::Xor {
                frame: 0,
                offset: 5 + 0x90 + 45,
                mask: 0x80,
            }],
            p0.clone(),
            E::StreamBadSignature,
            "Rejected:StreamBadSignature",
        ),
        c(
            "descriptor-flip-signature-bit",
            "bit flip inside the signature",
            vec![Edit::Xor {
                frame: 0,
                offset: 5 + bundle.descriptor.len() - 3,
                mask: 4,
            }],
            p0.clone(),
            E::StreamBadSignature,
            "Rejected:StreamBadSignature",
        ),
        c(
            "descriptor-flip-magic",
            "magic changed",
            vec![Edit::Xor {
                frame: 0,
                offset: 5,
                mask: 1,
            }],
            p0.clone(),
            E::StreamBadMagic,
            "Rejected:StreamBadMagic",
        ),
        c(
            "descriptor-truncated",
            "last descriptor byte missing (frame len fixed up)",
            vec![df(bundle.descriptor[..bundle.descriptor.len() - 1].to_vec())],
            p0.clone(),
            E::StreamDescriptorTruncated,
            "Rejected:StreamDescriptorTruncated",
        ),
        c(
            "epoch-lower-than-pin",
            "replay of an older epoch",
            vec![],
            Pol {
                pin: Some(EpochPin {
                    epoch: 8,
                    descriptor_hash: [0x55; 32],
                }),
                ..p0.clone()
            },
            E::StreamStaleEpoch,
            "Rejected:StreamStaleEpoch",
        ),
        c(
            "epoch-equal-different-content",
            "equal epoch, pinned hash differs",
            vec![],
            Pol {
                pin: Some(EpochPin {
                    epoch: 7,
                    descriptor_hash: other.bundle.descriptor_hash,
                }),
                ..p0.clone()
            },
            E::StreamEpochConflict,
            "Rejected:StreamEpochConflict",
        ),
        c(
            "unknown-signer",
            "signer not in the trust set",
            vec![],
            Pol {
                trusted: vec![pk(0x2B)],
                ..p0.clone()
            },
            E::UnknownSigner,
            "Rejected:UnknownSigner",
        ),
        c(
            "descriptor-expired",
            "expires_ns reached on a trusted clock",
            vec![df(resign(bundle, &sk, |_, h| h.expires_ns = 1000))],
            Pol {
                now: Some(1000),
                ..p0.clone()
            },
            E::StreamExpired,
            "Rejected:StreamExpired",
        ),
        c(
            "size-lie-seg-count",
            "signed seg_count 16777215 with a short descriptor",
            vec![df(resign(bundle, &sk, |_, h| {
                h.seg_count = 0xFF_FFFF;
                h.table_len = 0xFF_FFFF * 72;
            }))],
            p0.clone(),
            E::StreamLimitExceeded,
            "Rejected:StreamLimitExceeded",
        ),
        c(
            "size-lie-uncompressed-len",
            "signed uncompressed_len u64::MAX",
            vec![df(resign(bundle, &sk, |es, _| {
                es[0].uncompressed_len = u64::MAX
            }))],
            p0.clone(),
            E::StreamLimitExceeded,
            "Rejected:StreamLimitExceeded",
        ),
        c(
            "size-lie-stored-len",
            "signed stored_len u64::MAX/2 (over the segment cap)",
            vec![df(resign(bundle, &sk, |es, _| {
                es[2].stored_len = u64::MAX / 2
            }))],
            p0.clone(),
            E::StreamLimitExceeded,
            "Rejected:StreamLimitExceeded",
        ),
        c(
            "size-lie-body-len",
            "signed body_len one byte short",
            vec![df(resign(bundle, &sk, |_, h| h.body_len -= 1))],
            p0.clone(),
            E::StreamTableInvalid,
            "Rejected:StreamTableInvalid",
        ),
        c(
            "table-gap",
            "signed gap between two segments",
            vec![df(resign(bundle, &sk, |es, _| es[3].body_offset += 64))],
            p0.clone(),
            E::StreamTableInvalid,
            "Rejected:StreamTableInvalid",
        ),
        c(
            "table-overlap",
            "signed overlap between two segments",
            vec![df(resign(bundle, &sk, |es, _| es[3].body_offset -= 64))],
            p0.clone(),
            E::StreamTableInvalid,
            "Rejected:StreamTableInvalid",
        ),
        c(
            "table-unsorted",
            "signed entries out of body order",
            vec![df(resign(bundle, &sk, |es, _| es.swap(1, 2)))],
            p0.clone(),
            E::StreamTableInvalid,
            "Rejected:StreamTableInvalid",
        ),
        c(
            "table-duplicate-seg-id",
            "signed duplicate seg_id",
            vec![df(resign(bundle, &sk, |es, _| es[2].seg_id = es[1].seg_id))],
            p0.clone(),
            E::StreamTableInvalid,
            "Rejected:StreamTableInvalid",
        ),
        c(
            "chunk-bit-flip",
            "bit flip in a chunk payload",
            vec![Edit::Xor {
                frame: c0,
                offset: 17 + 100,
                mask: 0x10,
            }],
            p0.clone(),
            E::StreamChunkHashMismatch,
            "Poisoned:StreamChunkHashMismatch",
        ),
        c(
            "table-bit-flip",
            "bit flip in a chunk table leaf",
            vec![Edit::Xor {
                frame: t3,
                offset: 13 + 40,
                mask: 1,
            }],
            p0.clone(),
            E::StreamTableHashMismatch,
            "Poisoned:StreamTableHashMismatch",
        ),
        c(
            "wrong-file-id-splice",
            "chunk validly hashed under another file_id (same signing key), spliced in",
            vec![Edit::Replace {
                frame: s5,
                bytes: other.frames[pos(&other.frames, 1, 5, Some(0))].clone(),
            }],
            p0.clone(),
            E::StreamChunkHashMismatch,
            "Poisoned:StreamChunkHashMismatch",
        ),
        c(
            "frame-len-lie",
            "chunk frame declares one byte more than carried",
            vec![Edit::Xor {
                frame: c0,
                offset: 13,
                mask: 1,
            }],
            p0.clone(),
            E::StreamFrameSizeLie,
            "Poisoned:StreamFrameSizeLie",
        ),
        c(
            "frame-cut-tail",
            "last frame cut by one byte",
            vec![Edit::CutTail {
                frame: fr.len() - 1,
                bytes: 1,
            }],
            p0.clone(),
            E::StreamFrameSizeLie,
            "Poisoned:StreamFrameSizeLie",
        ),
        c(
            "truncate-frames",
            "stream ends after 10 frames",
            vec![Edit::Keep(10)],
            p0.clone(),
            E::StreamBadState,
            "Receiving",
        ),
        c(
            "duplicate-with-different-bytes",
            "authentic chunk 0 followed by a copy with one flipped bit",
            vec![
                Edit::Dup(c0),
                Edit::Xor {
                    frame: c0 + 1,
                    offset: 17 + 9,
                    mask: 4,
                },
            ],
            p0.clone(),
            E::StreamChunkHashMismatch,
            "Poisoned:StreamChunkHashMismatch",
        ),
        c(
            "duplicate-flood",
            "chunk 0 delivered four times with max_duplicate_frames 2",
            vec![Edit::Dup(c0), Edit::Dup(c0), Edit::Dup(c0)],
            Pol {
                max_dup: Some(2),
                ..p0.clone()
            },
            E::StreamDuplicateFlood,
            "Poisoned:StreamDuplicateFlood",
        ),
        c(
            "descriptor-replayed-different",
            "a different validly signed descriptor (epoch 8) sent after admission",
            vec![Edit::Insert {
                at: 1,
                bytes: descriptor_frame(&resign(bundle, &sk, |_, h| h.epoch = 8)),
            }],
            p0.clone(),
            E::StreamUnexpectedFrame,
            "Poisoned:StreamUnexpectedFrame",
        ),
        c(
            "out-of-order-chunk",
            "chunk 1 before chunk 0",
            vec![Edit::Swap(c0, c1)],
            p0.clone(),
            E::StreamUnexpectedFrame,
            "Poisoned:StreamUnexpectedFrame",
        ),
        c(
            "deleted-chunk",
            "chunk 0 missing",
            vec![Edit::Drop(c0)],
            p0.clone(),
            E::StreamUnexpectedFrame,
            "Poisoned:StreamUnexpectedFrame",
        ),
        c(
            "inserted-unknown-frame",
            "frame of unknown type inserted",
            vec![Edit::Insert {
                at: 2,
                bytes: vec![0x7F, 1, 2, 3],
            }],
            p0.clone(),
            E::StreamUnexpectedFrame,
            "Poisoned:StreamUnexpectedFrame",
        ),
        c(
            "chunk-before-table",
            "chunk 0 sent before its chunk table",
            vec![Edit::Swap(t3, c0)],
            p0.clone(),
            E::StreamUnexpectedFrame,
            "Poisoned:StreamUnexpectedFrame",
        ),
        c(
            "identity-mismatch",
            "signed rvf_identity differs from SHA-256 of the body",
            vec![df(resign(bundle, &sk, |_, h| h.rvf_identity[0] ^= 1))],
            p0.clone(),
            E::StreamIdentityMismatch,
            "Poisoned:StreamIdentityMismatch",
        ),
    ];
    // Requires gate: stage order, but the stage-3 segment is delivered first.
    v.push(Case {
        name: "requires-unmet",
        stream: "m1-stage",
        what: "stage-3 segment delivered before stages 0..2 are verified",
        edits: vec![Edit::Swap(1, 7)],
        pol: p0,
        reason: E::StreamRequirementsUnmet,
        state: "Poisoned:StreamRequirementsUnmet",
    });
    let _ = (m1f, m1s);
    v
}
