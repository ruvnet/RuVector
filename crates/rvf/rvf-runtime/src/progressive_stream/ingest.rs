//! `StreamIngest`: feeds verified stream events into a `ProgressiveIndex`.
//!
//! Invariants (ADR-353 section 6):
//!
//! * Only bytes released by `StreamVerifier` as `VerifiedChunk`s are parsed;
//!   a vector becomes visible once all of its record bytes were released.
//! * A layer is published only when its segment is `SegmentVerified`; a layer
//!   is never searched before Layer A (entry points, centroids, anchors) is
//!   published. Layers B and C that arrive without A are a protocol violation
//!   and poison the stream.
//! * `Poisoned` drops every layer and every vector. `Suspended` keeps what was
//!   verified (it stays valid) but answers carry `sealed = false` regardless.
//! * Answers carry `sealed = true` only in state `Complete`.
//!
//! The reassembly buffer holds at most one frame plus the slice being fed.

mod query;
pub use query::QueryMode;

use super::layers::{decode_a, decode_b, MAGIC_A, MAGIC_B};
use super::vectors::{VecEvent, VecParse, VerifiedVectors};
use crate::index_path::decode_graph;
use crate::options::{QualityEnvelope, SearchResult};
use rvf_index::{LayerC, ProgressiveIndex};
use rvf_types::quality::{
    DegradationReason, DegradationReport, FallbackPath, IndexLayersUsed, ResponseQuality,
    RetrievalQuality, SearchEvidenceSummary,
};
use rvf_types::ErrorCode;
use rvf_wire::stream::{
    Event, IdentityStatus, StreamPolicy, StreamState, StreamVerifier, FRAME_CHUNK,
    FRAME_CHUNK_TABLE, FRAME_DESCRIPTOR,
};
use std::collections::{HashMap, HashSet};

const SEG_VEC: u8 = 0x01;
const SEG_INDEX: u8 = 0x02;
const SEG_JOURNAL: u8 = 0x04;
const MAX_WITNESSES: usize = 4096;
/// Default cap on bytes of INDEX / JOURNAL segments buffered at once.
const DEFAULT_BUFFER_BUDGET: u64 = 256 << 20;

/// Audit record of a state change (the RVM witness chain consumes these).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamWitness {
    /// Descriptor accepted.
    DescriptorVerified([u8; 32]),
    /// Descriptor refused.
    DescriptorRejected(ErrorCode),
    /// A layer became searchable (`b'A'`, `b'B'`, `b'C'`).
    LayerPublished(u8),
    /// Terminal failure; everything was unpublished.
    StreamPoisoned {
        /// Why.
        reason: ErrorCode,
        /// Bytes that had been verified.
        verified_bytes: u64,
    },
    /// Input ended early.
    Suspended,
    /// Resumed after suspension.
    Resumed,
    /// Every segment verified.
    Complete(IdentityStatus),
}

/// Result of feeding bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedOutcome {
    /// Frames released or accepted.
    pub frames: usize,
    /// The first refused frame, if any (the rest of the input was dropped).
    pub refused: Option<ErrorCode>,
    /// Stream offset of the first byte that was not consumed as a good frame;
    /// resume the transport here.
    pub resume_offset: u64,
}

/// A query answer: the ADR-033 envelope plus streaming state.
#[derive(Clone, Debug)]
pub struct StreamAnswer {
    /// Results and quality signals.
    pub envelope: QualityEnvelope,
    /// True only once the stream is `Complete`.
    pub sealed: bool,
    /// Receiver state when the answer was produced.
    pub state: StreamState,
    /// Verified bytes so far.
    pub verified_bytes: u64,
    /// Visible vectors.
    pub visible_vectors: usize,
    /// How the results were produced.
    pub path: QueryPath,
    /// Visible vectors as a fraction of the vectors the signed table says
    /// the stream will carry (1.0 once all arrived). Results from a scan or
    /// from partial layers cover only this fraction of the data.
    pub coverage_vectors: f64,
    /// Verified bytes as a fraction of the stream body.
    pub coverage_bytes: f64,
    /// The layer search stopped at its work budget: results are partial.
    pub work_truncated: bool,
    /// Steps the layer search spent (0 for the scan path).
    pub work_steps: u64,
}

/// How a query was answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryPath {
    /// Nothing to search yet.
    Empty,
    /// Exact scan over the verified vectors received so far.
    ArrivedScan,
    /// `rvf-index` `ProgressiveIndex` over the published layers.
    Layers,
}

enum Parse {
    Vec(VecParse),
    Buf(Vec<u8>),
    Ignore,
}

/// Verified-stream consumer feeding a progressive index.
pub struct StreamIngest<'p> {
    verifier: StreamVerifier<'p>,
    buf: Vec<u8>,
    pos: u64,
    vectors: VerifiedVectors,
    parses: HashMap<u64, Parse>,
    kinds: HashMap<u64, (u8, u64)>,
    index: ProgressiveIndex,
    deleted: HashSet<u64>,
    witnesses: Vec<StreamWitness>,
    unpublished: bool,
    dropped_witnesses: u64,
    /// `stored_len` of every VEC segment in the signed table.
    vec_stored_lens: Vec<u64>,
    work_budget: Option<u64>,
    scan_bound: std::time::Duration,
    scan_ns_per_vec: std::cell::Cell<f64>,
    buffered: u64,
    buffer_budget: u64,
    pins: Option<&'p dyn rvf_wire::stream::PinStore>,
}

impl<'p> StreamIngest<'p> {
    /// A consumer waiting for the descriptor frame.
    pub fn new(policy: StreamPolicy<'p>) -> Self {
        Self {
            pins: policy.pins,
            verifier: StreamVerifier::new(policy),
            buf: Vec::new(),
            pos: 0,
            vectors: VerifiedVectors::default(),
            parses: HashMap::new(),
            kinds: HashMap::new(),
            index: ProgressiveIndex::new(),
            deleted: HashSet::new(),
            witnesses: Vec::new(),
            unpublished: false,
            dropped_witnesses: 0,
            vec_stored_lens: Vec::new(),
            work_budget: None,
            scan_bound: std::time::Duration::from_millis(25),
            scan_ns_per_vec: std::cell::Cell::new(0.0),
            buffered: 0,
            buffer_budget: DEFAULT_BUFFER_BUDGET,
        }
    }

    /// Receiver state.
    pub fn state(&self) -> StreamState {
        self.verifier.state()
    }

    /// The underlying verifier (read-only).
    pub fn verifier(&self) -> &StreamVerifier<'p> {
        &self.verifier
    }

    /// Stream offset of the next byte to request.
    pub fn offset(&self) -> u64 {
        self.pos + self.buf.len() as u64
    }

    /// Visible vectors (verified only).
    pub fn vectors(&self) -> &VerifiedVectors {
        &self.vectors
    }

    /// Witness records so far.
    pub fn witnesses(&self) -> &[StreamWitness] {
        &self.witnesses
    }

    /// Layers currently searchable.
    pub fn layers(&self) -> IndexLayersUsed {
        IndexLayersUsed {
            layer_a: self.index.layer_a.is_some(),
            layer_b: self.index.layer_b.is_some(),
            layer_c: self.index.layer_c.is_some(),
            hot_cache: false,
        }
    }

    fn witness(&mut self, w: StreamWitness) {
        // Terminal and audit-critical records are always kept; only the
        // repetitive Suspended/Resumed/LayerPublished trail is capped.
        let routine = matches!(
            w,
            StreamWitness::Suspended | StreamWitness::Resumed | StreamWitness::LayerPublished(_)
        );
        if !routine || self.witnesses.len() < MAX_WITNESSES {
            self.witnesses.push(w);
        } else {
            self.dropped_witnesses += 1;
        }
    }

    /// Routine witness records dropped after the cap (terminal records are
    /// never dropped).
    pub fn dropped_witnesses(&self) -> u64 {
        self.dropped_witnesses
    }

    /// Cap bytes of INDEX / JOURNAL segments buffered at once (default
    /// 256 MiB). Exceeding it poisons the stream with `StreamLimitExceeded`.
    pub fn set_buffer_budget(&mut self, bytes: u64) {
        self.buffer_budget = bytes;
    }

    /// Input ended before `Complete`.
    pub fn end_of_input(&mut self) -> StreamState {
        let was = self.verifier.state();
        let s = self.verifier.end_of_input();
        if was == StreamState::Receiving && s == StreamState::Suspended {
            self.witness(StreamWitness::Suspended);
        }
        s
    }

    /// Continue after `Suspended` (progress is kept at chunk granularity).
    pub fn resume(&mut self) -> bool {
        let ok = self.verifier.resume();
        if ok {
            self.witness(StreamWitness::Resumed);
        }
        ok
    }

    /// Re-check the epoch pin against `store` (stale-epoch error if the lineage advanced).
    pub fn recheck_pin(&self, store: &dyn rvf_wire::stream::PinStore) -> Result<(), ErrorCode> {
        self.verifier.recheck_pin(store)
    }

    /// Re-read the persisted body in file order and settle an `Unverifiable`
    /// identity (see `StreamVerifier::finish_identity`).
    pub fn finish_identity(
        &mut self,
        read_at: impl FnMut(u64, &mut [u8]) -> bool,
    ) -> Result<(), ErrorCode> {
        let r = self.verifier.finish_identity(read_at);
        self.after_push();
        r
    }

    /// Feed transport bytes. Stops at the first refused frame and drops the
    /// remainder of `data`; `resume_offset` says where to continue.
    pub fn feed(&mut self, data: &[u8]) -> FeedOutcome {
        let mut out = FeedOutcome {
            frames: 0,
            refused: None,
            resume_offset: self.pos,
        };
        if matches!(
            self.verifier.state(),
            StreamState::Poisoned(_) | StreamState::Rejected(_)
        ) {
            out.refused = Some(ErrorCode::StreamPoisoned);
            return out;
        }
        let mut buf = std::mem::take(&mut self.buf);
        buf.extend_from_slice(data);
        let mut at = 0usize;
        loop {
            let rest = &buf[at..];
            let Some(&t) = rest.first() else { break };
            let hdr = match t {
                FRAME_DESCRIPTOR => 5,
                FRAME_CHUNK => 17,
                FRAME_CHUNK_TABLE => 13,
                _ => {
                    self.verifier.poison(ErrorCode::StreamUnexpectedFrame);
                    out.refused = Some(ErrorCode::StreamUnexpectedFrame);
                    break;
                }
            };
            if rest.len() < hdr {
                break;
            }
            let declared = match t {
                FRAME_DESCRIPTOR => u32::from_le_bytes(rest[1..5].try_into().unwrap()) as u64,
                FRAME_CHUNK => u32::from_le_bytes(rest[13..17].try_into().unwrap()) as u64,
                _ => 32 * u32::from_le_bytes(rest[9..13].try_into().unwrap()) as u64,
            };
            // Size lies are refused before any payload is buffered.
            if declared > self.verifier.frame_payload_limit(t).unwrap_or(0) {
                self.verifier.poison(ErrorCode::StreamFrameSizeLie);
                out.refused = Some(ErrorCode::StreamFrameSizeLie);
                break;
            }
            // Checked: on a 32-bit target `hdr + declared` can wrap (F2-11).
            let Some(total) = usize::try_from(declared)
                .ok()
                .and_then(|d| d.checked_add(hdr))
            else {
                self.verifier.poison(ErrorCode::StreamFrameSizeLie);
                out.refused = Some(ErrorCode::StreamFrameSizeLie);
                break;
            };
            if rest.len() < total {
                break;
            }
            let frame = &rest[..total];
            at += total;
            match self.verifier.push(frame) {
                Ok(ev) => {
                    out.frames += 1;
                    self.pos += total as u64;
                    self.apply(ev);
                }
                Err(code) => {
                    out.refused = Some(code);
                    at -= total; // the refused frame stays unconsumed
                    break;
                }
            }
            self.after_push();
        }
        if out.refused.is_some() {
            // Drop buffered input; the transport restarts at `pos`.
            buf.clear();
        } else {
            buf.drain(..at);
        }
        self.buf = buf;
        self.after_push();
        out.resume_offset = self.pos;
        out
    }

    fn after_push(&mut self) {
        if let StreamState::Poisoned(r) | StreamState::Rejected(r) = self.verifier.state() {
            if !self.unpublished {
                self.unpublished = true;
                self.unpublish();
                let w = match self.verifier.state() {
                    StreamState::Rejected(_) => StreamWitness::DescriptorRejected(r),
                    _ => StreamWitness::StreamPoisoned {
                        reason: r,
                        verified_bytes: self.verifier.verified_bytes(),
                    },
                };
                self.witness(w);
            }
        }
    }

    fn unpublish(&mut self) {
        self.index = ProgressiveIndex::new();
        self.vectors.clear();
        self.parses.clear();
        self.buffered = 0;
        self.deleted.clear();
    }

    fn apply(&mut self, ev: Event<'_>) {
        match ev {
            Event::DescriptorVerified {
                descriptor_hash, ..
            } => {
                if let Some(d) = self.verifier.descriptor() {
                    self.kinds = d
                        .entries
                        .iter()
                        .map(|e| (e.seg_id, (e.seg_type, e.stored_len)))
                        .collect();
                    self.vec_stored_lens = d
                        .entries
                        .iter()
                        .filter(|e| e.seg_type == SEG_VEC)
                        .map(|e| e.stored_len)
                        .collect();
                }
                self.witness(StreamWitness::DescriptorVerified(descriptor_hash));
            }
            Event::TableVerified { .. } => {}
            // Authentic repeats (retransmits, replays) are idempotent: the bytes were
            // already applied once and must never be applied again.
            Event::Duplicate { .. } | Event::DuplicateDescriptor | Event::Unverified { .. } => {}
            Event::Chunk {
                chunk,
                segment_verified,
                complete,
                ..
            } => {
                self.on_bytes(chunk.seg_id, chunk.bytes, segment_verified);
                if let Some(st) = complete {
                    self.witness(StreamWitness::Complete(st));
                }
            }
        }
    }

    fn on_bytes(&mut self, seg_id: u64, bytes: &[u8], done: bool) {
        let Some(&(ty, stored)) = self.kinds.get(&seg_id) else {
            return;
        };
        if matches!(ty, SEG_INDEX | SEG_JOURNAL) {
            // Whole-segment buffering is bounded by an ingest-level budget on
            // top of the signed table's per-segment sizes.
            if self.buffered + bytes.len() as u64 > self.buffer_budget {
                self.verifier.poison_consumer(ErrorCode::StreamLimitExceeded);
                return;
            }
            self.buffered += bytes.len() as u64;
        }
        let p = self.parses.entry(seg_id).or_insert_with(|| match ty {
            SEG_VEC => Parse::Vec(VecParse::new(stored)),
            SEG_INDEX | SEG_JOURNAL => Parse::Buf(Vec::new()),
            _ => Parse::Ignore,
        });
        match p {
            Parse::Vec(vp) => {
                let want = self.vectors.dimension_hint();
                let vecs = &mut self.vectors;
                vp.feed(bytes, want, &mut |ev| match ev {
                    VecEvent::Reserve { count, dim } => {
                        vecs.set_dim(dim);
                        vecs.reserve(count);
                    }
                    VecEvent::Record(id, v) => {
                        vecs.set_dim(v.len());
                        vecs.insert(id, v);
                    }
                });
            }
            Parse::Buf(b) => b.extend_from_slice(bytes),
            Parse::Ignore => {}
        }
        if done {
            if let Some(Parse::Buf(b)) = self.parses.remove(&seg_id) {
                self.buffered = self.buffered.saturating_sub(b.len() as u64);
                if ty == SEG_INDEX {
                    self.publish_index(&b);
                } else {
                    self.apply_journal(&b);
                }
            }
        }
    }

    fn apply_journal(&mut self, seg: &[u8]) {
        let Some(p) = payload_of(seg) else { return };
        if p.len() < 16 {
            return;
        }
        let n = u32::from_le_bytes(p[..4].try_into().unwrap()) as usize;
        for i in 0..n {
            let at = 16 + i * 12;
            let Some(e) = p.get(at..at + 12) else { break };
            if e[0] == 0x01 {
                self.deleted
                    .insert(u64::from_le_bytes(e[4..12].try_into().unwrap()));
            }
        }
    }

    fn publish_index(&mut self, seg: &[u8]) {
        let Some(p) = payload_of(seg) else { return };
        if p.len() >= 4 && p[..4] == MAGIC_A {
            let Some(a) = decode_a(p) else {
                self.verifier.poison_consumer(ErrorCode::StreamTableInvalid);
                return;
            };
            // Partition ranges must be well formed; their width is harmless
            // (queries visit only present ids) but an inverted range is a lie.
            if a.layer
                .partition_map
                .iter()
                .any(|p| p.vector_id_end < p.vector_id_start)
            {
                self.verifier.poison_consumer(ErrorCode::StreamTableInvalid);
                return;
            }
            self.vectors.set_dim(a.dim);
            for (id, v) in &a.anchors {
                self.vectors.insert(*id, v);
            }
            self.index.layer_a = Some(a.layer);
            self.witness(StreamWitness::LayerPublished(b'A'));
        } else if self.index.layer_a.is_none() {
            // B or C before A: nothing may be searched before its entry points.
            self.verifier.poison_consumer(ErrorCode::StreamRequirementsUnmet);
        } else if p.len() >= 4 && p[..4] == MAGIC_B {
            match decode_b(p) {
                Some(b) => {
                    self.index.layer_b = Some(b);
                    self.witness(StreamWitness::LayerPublished(b'B'));
                }
                None => {
                    self.verifier.poison_consumer(ErrorCode::StreamTableInvalid);
                }
            }
        } else {
            match decode_graph(p) {
                Some((g, _)) => {
                    self.index.layer_c = Some(LayerC {
                        full_adjacency: g.layers,
                    });
                    self.witness(StreamWitness::LayerPublished(b'C'));
                }
                None => {
                    self.verifier.poison_consumer(ErrorCode::StreamTableInvalid);
                }
            }
        }
    }
}

/// Payload slice of a buffered segment (`header + payload + ...`).
fn payload_of(seg: &[u8]) -> Option<&[u8]> {
    let len = u64::from_le_bytes(seg.get(0x10..0x18)?.try_into().ok()?) as usize;
    seg.get(64..64usize.checked_add(len)?)
}
