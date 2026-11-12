//! Producer side: derive a signed descriptor from an unchanged RVF body and
//! emit frames in file order or stage order.

use super::frame::{chunk_frame, descriptor_frame, table_frame};
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use ed25519_dalek::SigningKey;
use rvf_crypto::stream_sha256 as sha256;
use rvf_crypto::{chunk_list_hash, descriptor_hash, leaf_hash, sign_descriptor, signer_key_id};
use rvf_types::stream::*;
use rvf_types::{ErrorCode, ROOT_MANIFEST_MAGIC_BYTES, ROOT_MANIFEST_SIZE, SEGMENT_HEADER_SIZE};

extern crate alloc;

const SIGNED_FLAG: u16 = 0x0004;
/// `seg_id` given to the trailing root-page pseudo-entry.
pub const ROOT_PAGE_SEG_ID: u64 = u64::MAX;

/// Configures and signs a descriptor for a body.
pub struct StreamDescriptorBuilder {
    file_id: [u8; 16],
    key: SigningKey,
    epoch: u64,
    policy_hash: [u8; 32],
    chunk_log2: u8,
    expires_ns: u64,
    stages: Vec<u8>,
    requires_lower: bool,
    identity: bool,
    complete_before_exec: bool,
}

/// A signed descriptor plus what is needed to generate frames.
#[derive(Clone, Debug)]
pub struct StreamBundle {
    /// The signed descriptor bytes.
    pub descriptor: Vec<u8>,
    /// SHA-256 of `descriptor`.
    pub descriptor_hash: [u8; 32],
    /// Parsed header.
    pub header: StreamHeader,
    /// Table in body order.
    pub entries: Vec<StreamEntry>,
}

impl StreamDescriptorBuilder {
    /// Defaults: epoch 1, zero policy hash, 64 KiB chunks, stage 0, identity on.
    pub fn new(file_id: [u8; 16], key: SigningKey) -> Self {
        Self {
            file_id,
            key,
            epoch: 1,
            policy_hash: [0; 32],
            chunk_log2: 16,
            expires_ns: 0,
            stages: Vec::new(),
            requires_lower: false,
            identity: true,
            complete_before_exec: false,
        }
    }
    /// Set the epoch.
    pub fn epoch(mut self, e: u64) -> Self {
        self.epoch = e;
        self
    }
    /// Set the capability policy hash.
    pub fn policy_hash(mut self, p: [u8; 32]) -> Self {
        self.policy_hash = p;
        self
    }
    /// `0xFF` for the segment-granular profile, else `12..=30`.
    pub fn chunk_log2(mut self, c: u8) -> Self {
        self.chunk_log2 = c;
        self
    }
    /// Expiry in ns (`0` = none).
    pub fn expires_ns(mut self, t: u64) -> Self {
        self.expires_ns = t;
        self
    }
    /// Stage per segment in body order (missing entries default to 0; the
    /// root page pseudo-entry takes the last listed stage).
    pub fn stages(mut self, s: &[u8]) -> Self {
        self.stages = s.to_vec();
        self
    }
    /// Set `requires_mask` of a stage-`s` segment to all stages below `s`.
    pub fn requires_lower_stages(mut self, on: bool) -> Self {
        self.requires_lower = on;
        self
    }
    /// Include the SHA-256 identity of the body.
    pub fn identity(mut self, on: bool) -> Self {
        self.identity = on;
        self
    }
    /// Set the `require_complete_before_exec` flag.
    pub fn require_complete_before_exec(mut self, on: bool) -> Self {
        self.complete_before_exec = on;
        self
    }

    /// Walk `body` into segments (plus a trailing 4096-byte `RVM0` root page
    /// as a `0xFE` pseudo-entry), commit to every chunk, and sign.
    pub fn build(&self, body: &[u8]) -> Result<StreamBundle, ErrorCode> {
        let mut flags = 0u16;
        if self.chunk_log2 != STREAM_CHUNK_LOG2_SEGMENT {
            flags |= STREAM_FLAG_CHUNKED;
        }
        if self.identity {
            flags |= STREAM_FLAG_IDENTITY_PRESENT;
        }
        if self.complete_before_exec {
            flags |= STREAM_FLAG_REQUIRE_COMPLETE_BEFORE_EXEC;
        }
        let pk = self.key.verifying_key().to_bytes();
        let mut header = StreamHeader {
            flags,
            chunk_log2: self.chunk_log2,
            file_id: self.file_id,
            epoch: self.epoch,
            body_len: body.len() as u64,
            rvf_identity: if self.identity { sha256(body) } else { [0; 32] },
            policy_hash: self.policy_hash,
            seg_count: 0,
            table_len: 0,
            expires_ns: self.expires_ns,
            signer_key_id: signer_key_id(&pk),
        };
        let mut entries = Vec::new();
        let mut off = 0usize;
        while off < body.len() {
            let rest = &body[off..];
            let (seg_id, seg_type, len, unc) = if rest.len() == ROOT_MANIFEST_SIZE
                && rest[..4] == ROOT_MANIFEST_MAGIC_BYTES
            {
                (ROOT_PAGE_SEG_ID, STREAM_SEG_TYPE_ROOT_PAGE, rest.len(), 0)
            } else {
                let h = rvf_wire_header(rest)?;
                let mut len = (SEGMENT_HEADER_SIZE as u64)
                    .checked_add(h.payload_length)
                    .ok_or(ErrorCode::TruncatedSegment)? as usize;
                if h.flags & SIGNED_FLAG != 0 {
                    let sl = rest
                        .get(len + 2..len + 4)
                        .ok_or(ErrorCode::TruncatedSegment)?;
                    len += 8 + u16::from_le_bytes([sl[0], sl[1]]) as usize;
                }
                len += h.alignment_pad as usize;
                (h.segment_id, h.seg_type, len, h.uncompressed_len as u64)
            };
            if len > rest.len() || len == 0 {
                return Err(ErrorCode::TruncatedSegment);
            }
            let i = entries.len();
            let stage = self
                .stages
                .get(i)
                .or(self.stages.last())
                .copied()
                .unwrap_or(0)
                .min(STREAM_MAX_STAGES - 1);
            let e = StreamEntry {
                seg_id,
                seg_type,
                stage,
                flags: 0,
                requires_mask: if self.requires_lower {
                    (1u32 << stage) - 1
                } else {
                    0
                },
                body_offset: off as u64,
                stored_len: len as u64,
                uncompressed_len: unc,
                chunk_table_hash: [0; 32],
            };
            let hash = segment_commitment(&header, &e, &rest[..len]).1;
            entries.push(StreamEntry {
                chunk_table_hash: hash,
                ..e
            });
            off += len;
        }
        header.seg_count = entries.len() as u32;
        header.table_len = header.seg_count * STREAM_ENTRY_SIZE as u32;
        let mut d = Vec::with_capacity(stream_descriptor_len(header.seg_count).unwrap_or(0));
        d.extend_from_slice(&header.to_bytes());
        for e in &entries {
            d.extend_from_slice(&e.to_bytes());
        }
        d.extend_from_slice(&(STREAM_SIG_LEN as u16).to_le_bytes());
        let sig = sign_descriptor(&d, &self.key);
        d.extend_from_slice(&sig);
        Ok(StreamBundle {
            descriptor_hash: descriptor_hash(&d),
            descriptor: d,
            header,
            entries,
        })
    }
}

fn rvf_wire_header(rest: &[u8]) -> Result<rvf_types::SegmentHeader, ErrorCode> {
    crate::reader::read_segment_header(rest).map_err(|e| match e {
        rvf_types::RvfError::Code(c) => c,
        _ => ErrorCode::InvalidMagic,
    })
}

/// Leaves and the entry commitment of one stored segment.
fn segment_commitment(
    h: &StreamHeader,
    e: &StreamEntry,
    stored: &[u8],
) -> (Vec<[u8; 32]>, [u8; 32]) {
    let n = h.chunk_count(e.stored_len).unwrap_or(1);
    let mut leaves = Vec::with_capacity(n as usize);
    for i in 0..n {
        let (a, b) = chunk_range(h, e.stored_len, i);
        leaves.push(leaf_hash(&h.file_id, e.seg_id, i, &stored[a..b]));
    }
    let commit = if n == 1 {
        leaves[0]
    } else {
        chunk_list_hash(&h.file_id, e.seg_id, &leaves)
    };
    (leaves, commit)
}

fn chunk_range(h: &StreamHeader, stored_len: u64, i: u32) -> (usize, usize) {
    let start = i as u64 * h.chunk_size().unwrap_or(0);
    let len = h.chunk_len(stored_len, i).unwrap_or(stored_len);
    (start as usize, (start + len) as usize)
}

impl StreamBundle {
    /// Frames for one segment: its table frame (if chunked) then its chunks.
    pub fn segment_frames(&self, body: &[u8], idx: usize) -> Vec<Vec<u8>> {
        let e = &self.entries[idx];
        let stored = &body[e.body_offset as usize..(e.body_offset + e.stored_len) as usize];
        let (leaves, _) = segment_commitment(&self.header, e, stored);
        let mut out = Vec::new();
        if leaves.len() > 1 {
            out.push(table_frame(e.seg_id, &leaves));
        }
        for i in 0..leaves.len() as u32 {
            let (a, b) = chunk_range(&self.header, e.stored_len, i);
            out.push(chunk_frame(e.seg_id, i, &stored[a..b]));
        }
        out
    }

    /// Descriptor frame then all segments in body order.
    pub fn frames_file_order<'a>(&'a self, body: &'a [u8]) -> FrameIter<'a> {
        self.frames_in_order(body, (0..self.entries.len()).collect())
    }

    /// Descriptor frame then segments sorted by `(stage, body order)`.
    pub fn frames_stage_order<'a>(&'a self, body: &'a [u8]) -> FrameIter<'a> {
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by_key(|&i| (self.entries[i].stage, i));
        self.frames_in_order(body, order)
    }

    /// Descriptor frame then segments in the given order.
    pub fn frames_in_order<'a>(&'a self, body: &'a [u8], order: Vec<usize>) -> FrameIter<'a> {
        let mut pending = VecDeque::new();
        pending.push_back(descriptor_frame(&self.descriptor));
        FrameIter {
            bundle: self,
            body,
            order: order.into(),
            pending,
        }
    }
}

/// Lazy frame iterator; builds one segment's frames at a time.
pub struct FrameIter<'a> {
    bundle: &'a StreamBundle,
    body: &'a [u8],
    order: VecDeque<usize>,
    pending: VecDeque<Vec<u8>>,
}

impl Iterator for FrameIter<'_> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        while self.pending.is_empty() {
            let i = self.order.pop_front()?;
            self.pending
                .extend(self.bundle.segment_frames(self.body, i));
        }
        self.pending.pop_front()
    }
}
