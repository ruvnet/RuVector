//! The streaming validator: a push-based state machine over untrusted bytes.
//!
//! Memory is bounded by the limits, never by the upload: it buffers one
//! 64-byte header, at most one manifest payload (`max_manifest_payload`), one
//! signature footer (`max_signature_len + 8`) and the 4096-byte root page.
//! Payload bytes are hashed as they pass. The result is independent of how
//! the input is chunked (property-tested and fuzzed).
//!
//! Wire shape accepted, per segment: `header(64) | payload | footer? | pad`,
//! where `footer` is present iff the `SIGNED` flag is set and `pad` is exactly
//! `header.alignment_pad` (< 64) zero bytes — `0` for the rvf runtime's packed
//! stream, the gap to the next 64-byte boundary for rvf-wire / RVForge
//! output. An optional 4096-byte Level-0 root page (`RVM0`) may follow the
//! last segment and must end the stream.

use super::error::ValidationError;
use super::hasher::ContentHasher;
use super::layout::{check_vec_head, parse_manifest, RuntimeManifest, VEC_HEAD};
use super::{finish_summary, SegmentEntry, ValidatedRvf, ValidationLimits};
use rvf_types::{
    ErrorCode, RvfError, SegmentFlags, SegmentHeader, SegmentType, MAX_SEGMENT_PAYLOAD,
    ROOT_MANIFEST_MAGIC, ROOT_MANIFEST_SIZE, SEGMENT_HEADER_SIZE,
};
use sha2::{Digest, Sha256};

/// Flag bits defined by rvf-types (`SegmentFlags::KNOWN_MASK`).
const KNOWN_FLAGS: u16 = 0x0FFF;
const HDR: usize = SEGMENT_HEADER_SIZE;

/// Per-payload capture: the manifest bytes, or a VEC_SEG's 6-byte prefix.
enum Capture {
    None,
    Manifest(Vec<u8>),
    VecHead([u8; VEC_HEAD], usize),
}

/// The segment being read.
struct Current {
    header: SegmentHeader,
    offset: u64,
    hasher: Option<ContentHasher>,
    sha: Sha256,
    capture: Capture,
    vec_head: Option<(u16, u32)>,
}

enum State {
    Header { buf: [u8; HDR], filled: usize },
    Payload { remaining: u64 },
    Footer { buf: Vec<u8>, need: usize },
    Pad { remaining: u32 },
    Root { buf: Vec<u8> },
    Done,
}

/// Push-based RVF validator. Feed chunks with [`StreamValidator::push`], then
/// call [`StreamValidator::finish`]. The first error is sticky.
pub struct StreamValidator {
    limits: ValidationLimits,
    state: State,
    offset: u64,
    file_sha: Sha256,
    current: Option<Current>,
    /// Offset of the last completed segment (for padding errors).
    last_offset: u64,
    segments: Vec<SegmentEntry>,
    vec_heads: Vec<(u32, u16, u32)>,
    manifest: Option<(u64, RuntimeManifest)>,
    root_dim: Option<u16>,
    failed: Option<ValidationError>,
}

impl StreamValidator {
    /// A validator enforcing `limits`.
    pub fn new(limits: ValidationLimits) -> Self {
        StreamValidator {
            limits,
            state: State::Header {
                buf: [0; HDR],
                filled: 0,
            },
            offset: 0,
            file_sha: Sha256::new(),
            current: None,
            last_offset: 0,
            segments: Vec::new(),
            vec_heads: Vec::new(),
            manifest: None,
            root_dim: None,
            failed: None,
        }
    }

    /// Bytes consumed so far.
    pub fn bytes_seen(&self) -> u64 {
        self.offset
    }

    /// Feed the next chunk. Once an error is returned every later call
    /// returns the same error.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), ValidationError> {
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        let r = self.push_inner(chunk);
        if let Err(e) = &r {
            self.failed = Some(e.clone());
        }
        r
    }

    fn push_inner(&mut self, chunk: &[u8]) -> Result<(), ValidationError> {
        // Consume up to the size limit first, so an earlier structural error
        // wins exactly as it would with any other chunking.
        let limit = self.limits.max_total_bytes;
        let room = limit.saturating_sub(self.offset);
        let take = usize::try_from(room).map_or(chunk.len(), |r| r.min(chunk.len()));
        self.file_sha.update(&chunk[..take]);
        let mut rest = &chunk[..take];
        while !rest.is_empty() {
            let used = self.step(rest)?;
            self.offset += used as u64;
            rest = &rest[used..];
        }
        if take < chunk.len() {
            return Err(ValidationError::TooLarge { limit });
        }
        Ok(())
    }

    /// Consume a prefix of `input`, returning how many bytes were used.
    fn step(&mut self, input: &[u8]) -> Result<usize, ValidationError> {
        match &mut self.state {
            State::Header { buf, filled } => {
                let n = (HDR - *filled).min(input.len());
                buf[*filled..*filled + n].copy_from_slice(&input[..n]);
                *filled += n;
                if *filled == HDR {
                    let buf = *buf;
                    let start = self.offset + n as u64 - HDR as u64;
                    self.on_header(buf, start)?;
                }
                Ok(n)
            }
            State::Payload { remaining } => {
                let n = (*remaining).min(input.len() as u64) as usize;
                *remaining -= n as u64;
                let done = *remaining == 0;
                let cur = self.current.as_mut().expect("payload has a segment");
                feed_payload(cur, &input[..n])?;
                if done {
                    self.end_payload()?;
                }
                Ok(n)
            }
            State::Footer { buf, need } => {
                let n = (*need - buf.len()).min(input.len());
                buf.extend_from_slice(&input[..n]);
                if buf.len() == *need {
                    self.on_footer_progress()?;
                }
                Ok(n)
            }
            State::Pad { remaining } => {
                let n = (*remaining as usize).min(input.len());
                if input[..n].iter().any(|b| *b != 0) {
                    let offset = self.current_offset();
                    return Err(ValidationError::BadPadding { offset });
                }
                *remaining -= n as u32;
                if *remaining == 0 {
                    self.state = State::Header {
                        buf: [0; HDR],
                        filled: 0,
                    };
                }
                Ok(n)
            }
            State::Root { buf } => {
                let n = (ROOT_MANIFEST_SIZE - buf.len()).min(input.len());
                buf.extend_from_slice(&input[..n]);
                if buf.len() == ROOT_MANIFEST_SIZE {
                    let offset = self.offset + n as u64 - ROOT_MANIFEST_SIZE as u64;
                    let root = rvf_wire::manifest_codec::read_root_manifest(buf)
                        .map_err(|_| ValidationError::RootManifestInvalid { offset })?;
                    self.root_dim = Some(root.dimension);
                    self.state = State::Done;
                }
                Ok(n)
            }
            State::Done => Err(ValidationError::TrailingBytes {
                offset: self.offset,
            }),
        }
    }

    fn current_offset(&self) -> u64 {
        self.current.as_ref().map_or(self.last_offset, |c| c.offset)
    }

    fn on_header(&mut self, buf: [u8; HDR], offset: u64) -> Result<(), ValidationError> {
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic == ROOT_MANIFEST_MAGIC && !self.segments.is_empty() {
            self.state = State::Root { buf: buf.to_vec() };
            return Ok(());
        }
        let header = rvf_wire::read_segment_header(&buf).map_err(|e| match e {
            RvfError::Code(ErrorCode::InvalidVersion) => ValidationError::BadVersion { offset },
            _ => ValidationError::BadMagic { offset },
        })?;
        let limits = &self.limits;
        if self.segments.len() >= limits.max_segments as usize {
            return Err(ValidationError::TooManySegments {
                limit: limits.max_segments,
            });
        }
        let seg_type = match SegmentType::try_from(header.seg_type) {
            Ok(SegmentType::Invalid) | Err(_) => {
                return Err(ValidationError::UnknownSegmentType {
                    offset,
                    seg_type: header.seg_type,
                })
            }
            Ok(t) => t,
        };
        let executable = matches!(
            seg_type,
            SegmentType::Kernel | SegmentType::Ebpf | SegmentType::Wasm
        );
        if executable && !limits.allow_executable {
            return Err(ValidationError::ExecutableSegment { offset });
        }
        check_header_fields(&header, offset)?;
        let limit = limits.max_segment_payload.min(MAX_SEGMENT_PAYLOAD);
        if header.payload_length > limit {
            return Err(ValidationError::SegmentTooLarge {
                offset,
                len: header.payload_length,
                limit,
            });
        }
        let end = offset + HDR as u64 + header.payload_length;
        if end > limits.max_total_bytes {
            return Err(ValidationError::TooLarge {
                limit: limits.max_total_bytes,
            });
        }
        let hasher = ContentHasher::for_algo(header.checksum_algo).ok_or(
            ValidationError::UnsupportedChecksum {
                offset,
                algo: header.checksum_algo,
            },
        )?;
        let capture = match seg_type {
            SegmentType::Manifest => {
                if header.payload_length > u64::from(limits.max_manifest_payload) {
                    return Err(ValidationError::ManifestTooLarge {
                        offset,
                        limit: limits.max_manifest_payload,
                    });
                }
                Capture::Manifest(Vec::with_capacity(header.payload_length as usize))
            }
            SegmentType::Vec => Capture::VecHead([0; VEC_HEAD], 0),
            _ => Capture::None,
        };
        let mut sha = Sha256::new();
        sha.update(buf);
        let remaining = header.payload_length;
        self.current = Some(Current {
            header,
            offset,
            hasher: Some(hasher),
            sha,
            capture,
            vec_head: None,
        });
        self.state = State::Payload { remaining };
        if remaining == 0 {
            self.end_payload()?;
        }
        Ok(())
    }

    fn end_payload(&mut self) -> Result<(), ValidationError> {
        let cur = self.current.as_mut().expect("payload has a segment");
        let offset = cur.offset;
        let hasher = cur.hasher.take().expect("hasher consumed once");
        // Under cargo-fuzz (`--cfg fuzzing`) a mismatch is ignored: libFuzzer
        // cannot repair a 16-byte digest after mutating a payload, so without
        // this every mutated MANIFEST_SEG / VEC_SEG stops here and the parsers
        // behind it are never exercised. Never set in real builds.
        if !hasher.matches(&cur.header.content_hash) && !cfg!(fuzzing) {
            return Err(ValidationError::ChecksumMismatch { offset });
        }
        if let Capture::VecHead(_, filled) = cur.capture {
            if filled < VEC_HEAD {
                return Err(ValidationError::VecSegMalformed {
                    offset,
                    reason: "shorter than the prefix",
                });
            }
        }
        if SegmentFlags::from_raw(cur.header.flags).contains(SegmentFlags::SIGNED) {
            self.state = State::Footer {
                buf: Vec::with_capacity(4),
                need: 4,
            };
            Ok(())
        } else {
            self.end_segment(0)
        }
    }

    fn on_footer_progress(&mut self) -> Result<(), ValidationError> {
        let offset = self.current_offset();
        let State::Footer { buf, need } = &mut self.state else {
            unreachable!("called in the footer state")
        };
        if *need == 4 {
            let sig_len = u16::from_le_bytes([buf[2], buf[3]]);
            if sig_len == 0 || sig_len > self.limits.max_signature_len {
                return Err(ValidationError::BadFooter { offset });
            }
            *need = 8 + usize::from(sig_len);
            return Ok(());
        }
        let n = buf.len();
        let declared = u32::from_le_bytes([buf[n - 4], buf[n - 3], buf[n - 2], buf[n - 1]]);
        if declared as usize != n {
            return Err(ValidationError::BadFooter { offset });
        }
        let footer = core::mem::take(buf);
        let cur = self.current.as_mut().expect("footer has a segment");
        cur.sha.update(&footer);
        self.end_segment(footer.len() as u64)
    }

    fn end_segment(&mut self, footer_len: u64) -> Result<(), ValidationError> {
        let cur = self.current.take().expect("segment in progress");
        let index = self.segments.len() as u32;
        self.last_offset = cur.offset;
        if let Some((dim, count)) = cur.vec_head {
            self.vec_heads.push((index, dim, count));
        }
        let h = &cur.header;
        let entry = SegmentEntry {
            index,
            seg_type: h.seg_type,
            segment_id: h.segment_id,
            offset: cur.offset,
            payload_length: h.payload_length,
            length: HDR as u64 + h.payload_length + footer_len,
            checksum_algo: h.checksum_algo,
            signed: SegmentFlags::from_raw(h.flags).contains(SegmentFlags::SIGNED),
            sha256: cur.sha.finalize().into(),
        };
        let pad = h.alignment_pad;
        if let Capture::Manifest(bytes) = cur.capture {
            let bound = (self.segments.len() as u64).min(u64::from(self.limits.max_segments));
            let parsed = parse_manifest(&bytes, cur.offset, bound)?;
            for d in &parsed.directory {
                let seen = self
                    .segments
                    .binary_search_by_key(&d.offset, |s| s.offset)
                    .ok()
                    .map(|i| &self.segments[i])
                    .filter(|s| {
                        s.segment_id == d.seg_id
                            && s.payload_length == d.payload_length
                            && s.seg_type == d.seg_type
                    });
                if seen.is_none() {
                    return Err(ValidationError::DirectoryMismatch {
                        offset: cur.offset,
                        target: d.offset,
                    });
                }
            }
            self.manifest = Some((cur.offset, parsed));
        }
        self.segments.push(entry);
        self.state = if pad == 0 {
            State::Header {
                buf: [0; HDR],
                filled: 0,
            }
        } else {
            State::Pad { remaining: pad }
        };
        Ok(())
    }

    /// Finish the stream: the input must end on a segment boundary (or after
    /// the root page) and hold a valid vector-store manifest.
    pub fn finish(self) -> Result<ValidatedRvf, ValidationError> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        if self.offset == 0 {
            return Err(ValidationError::Empty);
        }
        let seg = self.current_offset();
        let cut = match &self.state {
            State::Header { filled: 0, .. } | State::Done => None,
            State::Header { filled, .. } => Some((self.offset - *filled as u64, "segment header")),
            State::Payload { .. } => Some((seg, "segment payload")),
            State::Footer { .. } => Some((seg, "signature footer")),
            State::Pad { .. } => Some((seg, "alignment padding")),
            State::Root { buf } => Some((self.offset - buf.len() as u64, "root manifest page")),
        };
        if let Some((offset, what)) = cut {
            return Err(ValidationError::Truncated { offset, what });
        }
        let sha256: [u8; 32] = self.file_sha.finalize().into();
        finish_summary(
            &self.limits,
            self.offset,
            sha256,
            self.segments,
            &self.vec_heads,
            self.manifest,
            self.root_dim,
        )
    }
}

fn feed_payload(cur: &mut Current, bytes: &[u8]) -> Result<(), ValidationError> {
    cur.hasher.as_mut().expect("hasher live").update(bytes);
    cur.sha.update(bytes);
    match &mut cur.capture {
        Capture::None => {}
        Capture::Manifest(buf) => buf.extend_from_slice(bytes),
        Capture::VecHead(head, filled) => {
            let n = (VEC_HEAD - *filled).min(bytes.len());
            head[*filled..*filled + n].copy_from_slice(&bytes[..n]);
            *filled += n;
            if *filled == VEC_HEAD {
                let parsed = check_vec_head(head, cur.header.payload_length, cur.offset)?;
                cur.vec_head = Some(parsed);
                cur.capture = Capture::None;
            }
        }
    }
    Ok(())
}

fn check_header_fields(h: &SegmentHeader, offset: u64) -> Result<(), ValidationError> {
    if h.flags & !KNOWN_FLAGS != 0 {
        return Err(ValidationError::UnknownFlags { offset });
    }
    if h.compression != 0 || SegmentFlags::from_raw(h.flags).contains(SegmentFlags::COMPRESSED) {
        return Err(ValidationError::UnsupportedCompression { offset });
    }
    let reserved = [
        (h.reserved_0 != 0, "reserved_0"),
        (h.reserved_1 != 0, "reserved_1"),
        (h.uncompressed_len != 0, "uncompressed_len"),
    ];
    if let Some((_, field)) = reserved.iter().find(|(bad, _)| *bad) {
        return Err(ValidationError::ReservedNonZero { offset, field });
    }
    if h.alignment_pad >= 64 {
        return Err(ValidationError::BadPadding { offset });
    }
    Ok(())
}
