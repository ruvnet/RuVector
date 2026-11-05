//! The receiver state machine (ADR-353 sections 5 and 6).
//!
//! `push` takes one complete frame. Nothing is buffered here: the released
//! [`VerifiedChunk`] borrows the frame the caller passed, which has been hashed
//! in place. Memory held between calls is the descriptor, one small state
//! record per segment, and the leaf list of each segment whose chunk table was
//! accepted but which is not yet fully verified. All are bounded by the policy
//! (descriptor size cap, per-segment size cap), never by a declared frame size.

use super::frame::{FRAME_CHUNK, FRAME_CHUNK_TABLE, FRAME_DESCRIPTOR};
use super::{decode_descriptor, EpochPin, PinConflict, PinStore, StreamPolicy, VerifiedDescriptor};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use rvf_crypto::StreamSha256 as Sha256;
use rvf_crypto::{chunk_list_hash_flat, leaf_hash};
use rvf_types::sha256::ct_eq;
use rvf_types::stream::*;
use rvf_types::{ErrorCode, ROOT_MANIFEST_MAGIC_BYTES, SEGMENT_MAGIC};

extern crate alloc;

/// Receiver state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamState {
    /// Waiting for the descriptor frame.
    AwaitDescriptor,
    /// Descriptor verified; chunks are being accepted.
    Receiving,
    /// Input ended early; `resume` continues at chunk granularity.
    Suspended,
    /// Every segment verified, the byte count matches and the body identity
    /// is confirmed (or the descriptor carries none). Terminal and safe.
    Complete,
    /// Every segment verified, but chunks arrived out of file order so
    /// `rvf_identity` is NOT confirmed. `rvf_identity` must not be used as an
    /// identity until [`StreamVerifier::finish_identity`] moves this to
    /// `Complete`. The chunk commitments themselves are verified.
    CompleteUnverified,
    /// The descriptor was refused (terminal).
    Rejected(ErrorCode),
    /// Terminal failure: revoke, terminate, drop layers.
    Poisoned(ErrorCode),
}

/// Outcome of the whole-body identity check at completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityStatus {
    /// SHA-256 over the verified bytes in file order equals `rvf_identity`.
    Verified,
    /// Chunks did not arrive in file order; see [`StreamVerifier::finish_identity`].
    Unverifiable,
    /// The descriptor carries no identity (`identity_present` clear).
    NotPresent,
}

/// A chunk whose leaf hash matched and whose required stages are verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedChunk<'a> {
    /// Segment id.
    pub seg_id: u64,
    /// Chunk index within the segment.
    pub chunk_index: u32,
    /// Offset of the first byte in the body.
    pub body_offset: u64,
    /// The verified bytes.
    pub bytes: &'a [u8],
}

/// Result of accepting one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event<'a> {
    /// Descriptor accepted.
    DescriptorVerified {
        /// SHA-256 of the signed descriptor.
        descriptor_hash: [u8; 32],
        /// Number of segments.
        seg_count: u32,
        /// Total body bytes.
        body_len: u64,
    },
    /// An authentic, already verified chunk was received again: ignored.
    Duplicate {
        /// Segment id.
        seg_id: u64,
        /// Chunk index.
        chunk_index: u32,
    },
    /// A chunk for an already finished multi-chunk segment whose table has been
    /// freed: its length matched but its bytes cannot be checked any more. It is
    /// counted against the duplicate budget, never released and never to be
    /// logged or treated as authentic.
    Unverified {
        /// Segment id.
        seg_id: u64,
        /// Chunk index.
        chunk_index: u32,
    },
    /// The descriptor frame was received again, byte-identical: ignored.
    DuplicateDescriptor,
    /// A chunk table matched its commitment.
    TableVerified {
        /// Segment id.
        seg_id: u64,
    },
    /// A chunk was released.
    Chunk {
        /// The released bytes.
        chunk: VerifiedChunk<'a>,
        /// This chunk completed its segment.
        segment_verified: bool,
        /// This chunk completed its stage.
        stage_verified: Option<u8>,
        /// Set when this chunk completed the stream.
        complete: Option<IdentityStatus>,
    },
}

#[derive(Clone)]
struct SegState {
    n: u32,
    next: u32,
    leaves: Option<Vec<u8>>,
    done: bool,
}

#[derive(Clone)]
struct IdentityTracker {
    hasher: Sha256,
    pos: u64,
    feasible: bool,
}

/// Verifies a stream of frames against a signed descriptor.
#[derive(Clone)]
pub struct StreamVerifier<'p> {
    policy: StreamPolicy<'p>,
    state: StreamState,
    desc: Option<VerifiedDescriptor>,
    segs: Vec<SegState>,
    index: BTreeMap<u64, usize>,
    stage_remaining: [u32; STREAM_MAX_STAGES as usize],
    left: usize,
    faults: u32,
    verified_bytes: u64,
    table_bytes: u64,
    dup_cap: u64,
    dups: u64,
    admitted_pin: Option<EpochPin>,
    identity: Option<IdentityTracker>,
    identity_status: Option<IdentityStatus>,
    max_chunk_len: u64,
    max_table_payload: u64,
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

impl<'p> StreamVerifier<'p> {
    /// A verifier waiting for the descriptor frame.
    pub fn new(policy: StreamPolicy<'p>) -> Self {
        Self {
            policy,
            state: StreamState::AwaitDescriptor,
            desc: None,
            segs: Vec::new(),
            index: BTreeMap::new(),
            stage_remaining: [0; STREAM_MAX_STAGES as usize],
            left: 0,
            faults: 0,
            verified_bytes: 0,
            table_bytes: 0,
            dup_cap: 0,
            dups: 0,
            admitted_pin: None,
            identity: None,
            identity_status: None,
            max_chunk_len: 0,
            max_table_payload: 0,
        }
    }

    /// A verifier for a descriptor delivered out of band (sidecar or package
    /// contract). Returns `Rejected` details as `Err`.
    pub fn from_descriptor(bytes: &[u8], policy: StreamPolicy<'p>) -> Result<Self, ErrorCode> {
        let mut v = Self::new(policy);
        v.install(bytes)?;
        Ok(v)
    }

    /// Current state.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Counted faults so far.
    pub fn faults(&self) -> u32 {
        self.faults
    }

    /// Bytes released so far.
    pub fn verified_bytes(&self) -> u64 {
        self.verified_bytes
    }

    /// The verified descriptor, once accepted.
    pub fn descriptor(&self) -> Option<&VerifiedDescriptor> {
        self.desc.as_ref()
    }

    /// Identity status recorded at completion.
    pub fn identity_status(&self) -> Option<IdentityStatus> {
        self.identity_status
    }

    /// Next chunk index expected for a segment (`None` if unknown or done).
    pub fn next_chunk(&self, seg_id: u64) -> Option<u32> {
        let s = &self.segs[*self.index.get(&seg_id)?];
        (!s.done).then_some(s.next)
    }

    /// Largest payload a frame of this type may declare (0 = nothing legal).
    pub fn frame_payload_limit(&self, frame_type: u8) -> Option<u64> {
        match frame_type {
            FRAME_DESCRIPTOR => Some(self.policy.max_descriptor_bytes as u64),
            FRAME_CHUNK => Some(self.max_chunk_len),
            FRAME_CHUNK_TABLE => Some(self.max_table_payload),
            _ => None,
        }
    }

    /// Terminal failure raised by a consumer while handling the chunk that
    /// completed the stream (for example an ingest buffer budget or a layer that
    /// does not decode). Unlike [`Self::poison`], this also overrides `Complete`
    /// and `CompleteUnverified`: the consumer refused the final segment, so the
    /// stream must not be reported complete.
    pub fn poison_consumer(&mut self, code: ErrorCode) -> ErrorCode {
        if matches!(
            self.state,
            StreamState::Complete | StreamState::CompleteUnverified
        ) {
            self.state = StreamState::Poisoned(code);
            return code;
        }
        self.poison(code)
    }

    /// Terminal failure; used by the frame reader for size lies.
    pub fn poison(&mut self, code: ErrorCode) -> ErrorCode {
        if matches!(
            self.state,
            StreamState::AwaitDescriptor | StreamState::Receiving | StreamState::Poisoned(_)
        ) {
            self.state = StreamState::Poisoned(code);
        }
        code
    }

    /// Input ended before `Complete`: move to `Suspended`.
    pub fn end_of_input(&mut self) -> StreamState {
        if self.state == StreamState::Receiving {
            self.state = StreamState::Suspended;
        }
        self.state
    }

    /// Continue after `Suspended`; progress is kept at chunk granularity.
    pub fn resume(&mut self) -> bool {
        if self.state == StreamState::Suspended {
            self.state = StreamState::Receiving;
            true
        } else {
            false
        }
    }

    fn fault(&mut self, code: ErrorCode) -> ErrorCode {
        self.faults += 1;
        if self.faults > self.policy.fault_budget {
            self.state = StreamState::Poisoned(code);
        }
        code
    }

    fn install(&mut self, bytes: &[u8]) -> Result<Event<'static>, ErrorCode> {
        let d = match decode_descriptor(bytes, &self.policy) {
            Ok(d) => d,
            Err(e) => {
                self.state = StreamState::Rejected(e);
                return Err(e);
            }
        };
        let h = d.header;
        for (i, e) in d.entries.iter().enumerate() {
            // `n` is derived from the signed stored_len; the descriptor
            // decoder already proved it fits a u32.
            let n = h.chunk_count(e.stored_len).unwrap_or(1);
            self.index.insert(e.seg_id, i);
            self.segs.push(SegState {
                n,
                next: 0,
                leaves: None,
                done: false,
            });
            self.stage_remaining[e.stage as usize] += 1;
            let first = h.chunk_len(e.stored_len, 0).unwrap_or(0);
            self.max_chunk_len = self.max_chunk_len.max(first);
            if n > 1 {
                self.max_table_payload = self.max_table_payload.max(32 * n as u64);
            }
        }
        let total_chunks: u64 = self.segs.iter().map(|s| s.n as u64).sum();
        self.dup_cap = self
            .policy
            .max_duplicate_frames
            .unwrap_or(4 * total_chunks + 64);
        self.admitted_pin = self
            .policy
            .pins
            .and_then(|st| st.get(&h.signer_key_id, &h.file_id));
        self.left = d.entries.len();
        if h.flags & STREAM_FLAG_IDENTITY_PRESENT != 0 {
            self.identity = Some(IdentityTracker {
                hasher: Sha256::new(),
                pos: 0,
                feasible: true,
            });
        }
        let ev = Event::DescriptorVerified {
            descriptor_hash: d.descriptor_hash,
            seg_count: h.seg_count,
            body_len: h.body_len,
        };
        self.desc = Some(d);
        self.state = StreamState::Receiving;
        Ok(ev)
    }

    /// Accept one complete frame.
    ///
    /// `Err` means the frame was refused: nothing was released, and the
    /// expected next chunk did not advance. Whether the stream survived is in
    /// [`state`](Self::state).
    pub fn push<'a>(&mut self, frame: &'a [u8]) -> Result<Event<'a>, ErrorCode> {
        match self.state {
            StreamState::Poisoned(_) => return Err(ErrorCode::StreamPoisoned),
            StreamState::AwaitDescriptor => {
                if frame.len() < 5 || frame[0] != FRAME_DESCRIPTOR {
                    return Err(self.poison(ErrorCode::StreamUnexpectedFrame));
                }
                if rd32(frame, 1) as usize != frame.len() - 5 {
                    return Err(self.poison(ErrorCode::StreamFrameSizeLie));
                }
                return self.install(&frame[5..]);
            }
            StreamState::Receiving => {}
            _ => return Err(ErrorCode::StreamBadState),
        }
        match frame.first() {
            Some(&FRAME_CHUNK) => self.on_chunk(frame),
            Some(&FRAME_CHUNK_TABLE) => self.on_table(frame),
            Some(&FRAME_DESCRIPTOR) => self.on_descriptor_again(frame),
            _ => Err(self.fault(ErrorCode::StreamUnexpectedFrame)),
        }
    }

    fn count_duplicate(&mut self) -> Result<(), ErrorCode> {
        self.dups += 1;
        if self.dups > self.dup_cap {
            return Err(self.poison(ErrorCode::StreamDuplicateFlood));
        }
        Ok(())
    }

    /// A descriptor frame after admission: ignored if byte-identical, else a fault.
    fn on_descriptor_again<'a>(&mut self, f: &'a [u8]) -> Result<Event<'a>, ErrorCode> {
        if f.len() < 5 || rd32(f, 1) as usize != f.len() - 5 {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let same = f.len() - 5 <= self.policy.max_descriptor_bytes
            && rvf_crypto::descriptor_hash(&f[5..])
                == self.desc.as_ref().expect("installed").descriptor_hash;
        if !same {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        }
        self.count_duplicate()?;
        Ok(Event::DuplicateDescriptor)
    }

    fn on_table<'a>(&mut self, f: &'a [u8]) -> Result<Event<'a>, ErrorCode> {
        if f.len() < 13 {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let (seg_id, n) = (rd64(f, 1), rd32(f, 9));
        if (f.len() - 13) as u64 != 32 * n as u64 {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let Some(&i) = self.index.get(&seg_id) else {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        };
        let s = &self.segs[i];
        if s.n <= 1 || s.leaves.is_some() || s.done {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        }
        if n != s.n {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let want = 32 * n as u64;
        if self.table_bytes + want > self.policy.max_table_bytes {
            return Err(self.fault(ErrorCode::StreamLimitExceeded));
        }
        let d = self.desc.as_ref().expect("descriptor installed");
        let ok = chunk_list_hash_flat(&d.header.file_id, seg_id, &f[13..])
            .is_some_and(|h| ct_eq(&h, &d.entries[i].chunk_table_hash));
        if !ok {
            return Err(self.fault(ErrorCode::StreamTableHashMismatch));
        }
        self.segs[i].leaves = Some(f[13..].to_vec());
        self.table_bytes += want;
        Ok(Event::TableVerified { seg_id })
    }

    fn on_chunk<'a>(&mut self, f: &'a [u8]) -> Result<Event<'a>, ErrorCode> {
        if f.len() < 17 {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let (seg_id, idx, len) = (rd64(f, 1), rd32(f, 9), rd32(f, 13));
        let bytes = &f[17..];
        if len as usize != bytes.len() {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let Some(&i) = self.index.get(&seg_id) else {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        };
        let (h, e) = {
            let d = self.desc.as_ref().expect("descriptor installed");
            (d.header, d.entries[i])
        };
        let s = &self.segs[i];
        if idx < s.next {
            return self.on_duplicate_chunk(i, idx, f);
        }
        if s.done || idx != s.next {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        }
        if h.chunk_len(e.stored_len, idx) != Some(len as u64) {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        if s.n > 1 && s.leaves.is_none() {
            return Err(self.fault(ErrorCode::StreamUnexpectedFrame));
        }
        let mut need = e.requires_mask;
        while need != 0 {
            let st = need.trailing_zeros() as usize;
            need &= need - 1;
            if self.stage_remaining[st] != 0 {
                return Err(self.fault(ErrorCode::StreamRequirementsUnmet));
            }
        }
        let leaf = leaf_hash(&h.file_id, seg_id, idx, bytes);
        let want: &[u8] = match &s.leaves {
            Some(l) => &l[idx as usize * 32..idx as usize * 32 + 32],
            None => &e.chunk_table_hash,
        };
        let mut w = [0u8; 32];
        w.copy_from_slice(want);
        if !ct_eq(&leaf, &w) {
            return Err(self.fault(ErrorCode::StreamChunkHashMismatch));
        }
        if idx == 0 && is_exec_or_index(e.seg_type) {
            // Authenticated bytes, but an unauthenticated checksum claim must
            // not feed an executor or index consumer (ADR-353 section 2.6).
            let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if magic != SEGMENT_MAGIC || bytes[5] != e.seg_type {
                return Err(self.poison(ErrorCode::StreamTableInvalid));
            }
            if bytes[0x20] <= 1 {
                return Err(self.poison(ErrorCode::StreamWeakChecksum));
            }
        }
        if idx == 0
            && e.seg_type == STREAM_SEG_TYPE_ROOT_PAGE
            && bytes[..4] != ROOT_MANIFEST_MAGIC_BYTES
        {
            return Err(self.poison(ErrorCode::StreamTableInvalid));
        }
        let body_offset = e.body_offset + idx as u64 * h.chunk_size().unwrap_or(0);
        if let Some(t) = &mut self.identity {
            if t.feasible && t.pos == body_offset {
                t.hasher.update(bytes);
                t.pos += bytes.len() as u64;
            } else {
                t.feasible = false;
            }
        }
        // Identity is checked before the final chunk is handed out, so a
        // mismatch releases nothing from that chunk.
        let mut segment_verified = false;
        let mut stage_verified = None;
        let mut complete = None;
        let s = &mut self.segs[i];
        s.next += 1;
        if s.next == s.n {
            s.done = true;
            // Free the finished segment's chunk table: the retained bound is
            // about OPEN segments (F3-06). Late duplicates of this segment
            // can no longer be hash-checked, but they release nothing and are
            // charged to the duplicate cap.
            if let Some(l) = s.leaves.take() {
                self.table_bytes -= l.len() as u64;
            }
            segment_verified = true;
            self.left -= 1;
            let st = e.stage as usize;
            self.stage_remaining[st] -= 1;
            if self.stage_remaining[st] == 0 {
                stage_verified = Some(e.stage);
            }
        }
        self.verified_bytes += bytes.len() as u64;
        if self.left == 0 {
            match self.finish() {
                Ok(st) => complete = Some(st),
                Err(c) => return Err(self.poison(c)),
            }
        }
        Ok(Event::Chunk {
            chunk: VerifiedChunk {
                seg_id,
                chunk_index: idx,
                body_offset,
                bytes,
            },
            segment_verified,
            stage_verified,
            complete,
        })
    }

    /// A chunk below the next expected index: idempotent if it hashes to the
    /// verified leaf, otherwise an ordinary hash-mismatch fault.
    fn on_duplicate_chunk<'a>(
        &mut self,
        i: usize,
        idx: u32,
        f: &'a [u8],
    ) -> Result<Event<'a>, ErrorCode> {
        let (seg_id, bytes) = (rd64(f, 1), &f[17..]);
        let (h, e) = {
            let d = self.desc.as_ref().expect("descriptor installed");
            (d.header, d.entries[i])
        };
        if h.chunk_len(e.stored_len, idx) != Some(bytes.len() as u64) {
            return Err(self.poison(ErrorCode::StreamFrameSizeLie));
        }
        let mut w = e.chunk_table_hash;
        if let Some(l) = &self.segs[i].leaves {
            w.copy_from_slice(&l[idx as usize * 32..idx as usize * 32 + 32]);
            if !ct_eq(&leaf_hash(&h.file_id, seg_id, idx, bytes), &w) {
                return Err(self.fault(ErrorCode::StreamChunkHashMismatch));
            }
        } else if self.segs[i].n <= 1 {
            // Single-chunk segment: the entry commits to the leaf itself.
            if !ct_eq(&leaf_hash(&h.file_id, seg_id, idx, bytes), &w) {
                return Err(self.fault(ErrorCode::StreamChunkHashMismatch));
            }
        } else {
            // A finished multi-chunk segment whose table was freed: the bytes
            // cannot be checked, so they are counted and reported as such.
            self.count_duplicate()?;
            return Ok(Event::Unverified {
                seg_id,
                chunk_index: idx,
            });
        }
        self.count_duplicate()?;
        Ok(Event::Duplicate {
            seg_id,
            chunk_index: idx,
        })
    }

    fn finish(&mut self) -> Result<IdentityStatus, ErrorCode> {
        let h = self.desc.as_ref().expect("descriptor installed").header;
        if self.verified_bytes != h.body_len {
            return Err(ErrorCode::StreamTableInvalid);
        }
        if let Some(store) = self.policy.pins {
            self.recheck_pin(store)?;
        }
        let st = match self.identity.take() {
            None => IdentityStatus::NotPresent,
            Some(t) if t.feasible && t.pos == h.body_len => {
                if !ct_eq(&t.hasher.finalize(), &h.rvf_identity) {
                    return Err(ErrorCode::StreamIdentityMismatch);
                }
                IdentityStatus::Verified
            }
            Some(_) => IdentityStatus::Unverifiable,
        };
        self.identity_status = Some(st);
        self.state = if st == IdentityStatus::Unverifiable {
            StreamState::CompleteUnverified
        } else {
            StreamState::Complete
        };
        Ok(st)
    }

    /// Re-check the epoch pin against `store` now (ADR-353 section 7). Consumers call
    /// this before publishing anything derived from the stream as final; it runs
    /// automatically at completion when the policy carries a store. A session whose
    /// lineage advanced while it was running gets `StreamStaleEpoch` (or
    /// `StreamEpochConflict` for an equal epoch with a different descriptor).
    /// Does not change the state.
    pub fn recheck_pin(&self, store: &dyn PinStore) -> Result<(), ErrorCode> {
        let d = self.desc.as_ref().ok_or(ErrorCode::StreamBadState)?;
        let h = &d.header;
        match store.get(&h.signer_key_id, &h.file_id) {
            Some(p) if h.epoch < p.epoch => Err(ErrorCode::StreamStaleEpoch),
            Some(p) if h.epoch == p.epoch && p.descriptor_hash != d.descriptor_hash => {
                Err(ErrorCode::StreamEpochConflict)
            }
            _ => Ok(()),
        }
    }

    /// Record this lineage's pin after the stream is `Complete` with the identity
    /// confirmed. Compare-and-set against the pin seen at admission, so a concurrent
    /// session that advanced first wins and this one gets a typed error.
    pub fn commit_pin(&self, store: &dyn PinStore) -> Result<(), ErrorCode> {
        if self.state != StreamState::Complete {
            return Err(ErrorCode::StreamBadState);
        }
        let d = self.desc.as_ref().ok_or(ErrorCode::StreamBadState)?;
        let h = &d.header;
        let new = EpochPin {
            epoch: h.epoch,
            descriptor_hash: d.descriptor_hash,
        };
        store
            .advance(&h.signer_key_id, &h.file_id, self.admitted_pin, new)
            .map_err(|PinConflict { current }| match current {
                Some(c) if c.epoch > h.epoch => ErrorCode::StreamStaleEpoch,
                _ => ErrorCode::StreamEpochConflict,
            })
    }

    /// For storage-capable consumers that received out of file order.
    ///
    /// Re-reads the persisted body in file order through `read_at(offset, buf)`
    /// (which must fill `buf` or return false) in 64 KiB pieces and **re-verifies
    /// it against the signed commitments chunk by chunk** (every leaf must equal the
    /// verified leaf), hashing it for `SHA-256 == rvf_identity` in the same pass.
    /// Only then, and after re-checking the pin, does `CompleteUnverified` become
    /// `Complete`. A host that supplies different bytes poisons the stream.
    pub fn finish_identity(
        &mut self,
        mut read_at: impl FnMut(u64, &mut [u8]) -> bool,
    ) -> Result<(), ErrorCode> {
        if self.state != StreamState::CompleteUnverified
            || self.identity_status != Some(IdentityStatus::Unverifiable)
        {
            return Err(ErrorCode::StreamBadState);
        }
        let (h, entries) = {
            let d = self.desc.as_ref().expect("descriptor installed");
            (d.header, d.entries.clone())
        };
        let mut ident = Sha256::new();
        let mut buf = alloc::vec![0u8; 65536];
        for e in entries.iter() {
            let n = h.chunk_count(e.stored_len).unwrap_or(1);
            let mut seg_leaves: Vec<u8> = Vec::new();
            for idx in 0..n {
                let len = h.chunk_len(e.stored_len, idx).unwrap_or(0);
                let mut lh = rvf_crypto::LeafHasher::new(&h.file_id, e.seg_id, idx, len);
                let mut off = e.body_offset + idx as u64 * h.chunk_size().unwrap_or(0);
                let mut left = len;
                while left > 0 {
                    let k = left.min(buf.len() as u64) as usize;
                    if !read_at(off, &mut buf[..k]) {
                        return Err(ErrorCode::StreamBadState);
                    }
                    lh.update(&buf[..k]);
                    ident.update(&buf[..k]);
                    off += k as u64;
                    left -= k as u64;
                }
                let Some(leaf) = lh.finalize() else {
                    self.state = StreamState::Poisoned(ErrorCode::StreamChunkHashMismatch);
                    return Err(ErrorCode::StreamChunkHashMismatch);
                };
                if n == 1 {
                    // The entry commits to the leaf itself.
                    if !ct_eq(&leaf, &e.chunk_table_hash) {
                        self.state = StreamState::Poisoned(ErrorCode::StreamChunkHashMismatch);
                        return Err(ErrorCode::StreamChunkHashMismatch);
                    }
                } else {
                    seg_leaves.extend_from_slice(&leaf);
                }
            }
            if n > 1 {
                // Retained tables are freed once a segment verifies (F3-06), so
                // the commitment is re-derived from the re-read bytes: the
                // leaves must hash to the entry's `chunk_table_hash`.
                let ok = chunk_list_hash_flat(&h.file_id, e.seg_id, &seg_leaves)
                    .is_some_and(|got| ct_eq(&got, &e.chunk_table_hash));
                if !ok {
                    self.state = StreamState::Poisoned(ErrorCode::StreamChunkHashMismatch);
                    return Err(ErrorCode::StreamChunkHashMismatch);
                }
            }
        }
        if !ct_eq(&ident.finalize(), &h.rvf_identity) {
            // Explicit retroactive revocation of an unverified artifact.
            self.state = StreamState::Poisoned(ErrorCode::StreamIdentityMismatch);
            return Err(ErrorCode::StreamIdentityMismatch);
        }
        if let Some(store) = self.policy.pins {
            if let Err(e) = self.recheck_pin(store) {
                self.state = StreamState::Poisoned(e);
                return Err(e);
            }
        }
        self.identity_status = Some(IdentityStatus::Verified);
        self.state = StreamState::Complete;
        Ok(())
    }
}

fn is_exec_or_index(seg_type: u8) -> bool {
    matches!(
        StreamClass::of(seg_type),
        StreamClass::Executable | StreamClass::Index
    )
}
