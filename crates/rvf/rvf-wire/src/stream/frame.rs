//! Frame encoding and byte-stream reassembly (ADR-353 section 4).
//!
//! * `0x00 DESCRIPTOR`: `type u8 | len u32 | descriptor bytes`
//! * `0x01 CHUNK`: `type u8 | seg_id u64 | chunk_index u32 | len u32 | bytes`
//! * `0x02 CHUNK_TABLE`: `type u8 | seg_id u64 | n u32 | n * 32-byte leaves`

use super::{Event, StreamVerifier};
use alloc::vec::Vec;
use rvf_types::ErrorCode;

extern crate alloc;

/// Frame type: inline descriptor.
pub const FRAME_DESCRIPTOR: u8 = 0x00;
/// Frame type: chunk.
pub const FRAME_CHUNK: u8 = 0x01;
/// Frame type: chunk table.
pub const FRAME_CHUNK_TABLE: u8 = 0x02;

/// Encode a descriptor frame.
pub fn descriptor_frame(descriptor: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(5 + descriptor.len());
    f.push(FRAME_DESCRIPTOR);
    f.extend_from_slice(&(descriptor.len() as u32).to_le_bytes());
    f.extend_from_slice(descriptor);
    f
}

/// Encode a chunk frame.
pub fn chunk_frame(seg_id: u64, chunk_index: u32, bytes: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(17 + bytes.len());
    f.push(FRAME_CHUNK);
    f.extend_from_slice(&seg_id.to_le_bytes());
    f.extend_from_slice(&chunk_index.to_le_bytes());
    f.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    f.extend_from_slice(bytes);
    f
}

/// Encode a chunk-table frame from `leaves`.
pub fn table_frame(seg_id: u64, leaves: &[[u8; 32]]) -> Vec<u8> {
    let mut f = Vec::with_capacity(13 + 32 * leaves.len());
    f.push(FRAME_CHUNK_TABLE);
    f.extend_from_slice(&seg_id.to_le_bytes());
    f.extend_from_slice(&(leaves.len() as u32).to_le_bytes());
    for l in leaves {
        f.extend_from_slice(l);
    }
    f
}

/// Fixed header length of a frame type.
fn header_len(t: u8) -> Option<usize> {
    match t {
        FRAME_DESCRIPTOR => Some(5),
        FRAME_CHUNK => Some(17),
        FRAME_CHUNK_TABLE => Some(13),
        _ => None,
    }
}

/// Total frame length once the header is known, from the declared sizes.
fn declared_total(h: &[u8]) -> (u64, u64) {
    let hdr = h.len() as u64;
    match h[0] {
        FRAME_DESCRIPTOR => (hdr, u32::from_le_bytes([h[1], h[2], h[3], h[4]]) as u64),
        FRAME_CHUNK => (hdr, u32::from_le_bytes([h[13], h[14], h[15], h[16]]) as u64),
        _ => (
            hdr,
            32 * u32::from_le_bytes([h[9], h[10], h[11], h[12]]) as u64,
        ),
    }
}

/// Reassembles frames from arbitrary byte splits and feeds a verifier.
///
/// The declared payload size is checked against the verifier's limits (derived
/// from the signed descriptor) as soon as the header is complete, before any
/// payload byte is buffered. At most one frame is buffered.
#[derive(Default)]
pub struct FrameReader {
    buf: Vec<u8>,
    need: Option<usize>,
}

impl FrameReader {
    /// An empty reader.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes currently buffered (a partial frame).
    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Consume `data`; every completed frame is pushed into `v` and its
    /// result handed to `sink`. Returns `Err` once the stream is terminal
    /// (the rest of `data` is discarded).
    pub fn feed<'p>(
        &mut self,
        v: &mut StreamVerifier<'p>,
        mut data: &[u8],
        sink: &mut dyn FnMut(Result<Event<'_>, ErrorCode>),
    ) -> Result<(), ErrorCode> {
        use super::StreamState;
        use StreamState::{Poisoned, Rejected};
        while !data.is_empty() {
            if matches!(v.state(), Poisoned(_) | Rejected(_)) {
                return Err(ErrorCode::StreamPoisoned);
            }
            // Bytes after a finished stream are reported; the state is not changed.
            if matches!(
                v.state(),
                StreamState::Complete | StreamState::CompleteUnverified
            ) {
                return Err(ErrorCode::StreamBadState);
            }
            if self.buf.is_empty() {
                // Fast path: a whole frame sits at the front of `data`.
                if let Some(total) = self.probe(v, data)? {
                    if data.len() >= total {
                        sink(v.push(&data[..total]));
                        data = &data[total..];
                        continue;
                    }
                }
            }
            // Slow path: accumulate. First the header, then the payload.
            let target = match self.need {
                Some(t) => t,
                None => match self.buf.first().copied().or_else(|| data.first().copied()) {
                    None => break,
                    Some(t) => match header_len(t) {
                        Some(h) => h,
                        None => {
                            // Unknown type: cannot frame the rest of the stream.
                            let e = v.poison(ErrorCode::StreamUnexpectedFrame);
                            sink(Err(e));
                            return Err(e);
                        }
                    },
                },
            };
            let take = (target - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.need.is_none() && self.buf.len() == target {
                let total = self.admit(v, &self.buf.clone())?;
                self.need = Some(total);
            }
            if Some(self.buf.len()) == self.need {
                let frame = core::mem::take(&mut self.buf);
                self.need = None;
                sink(v.push(&frame));
            }
        }
        Ok(())
    }

    /// Frame total from the front of `data` when its header is complete.
    fn probe(
        &mut self,
        v: &mut StreamVerifier<'_>,
        data: &[u8],
    ) -> Result<Option<usize>, ErrorCode> {
        let Some(&t) = data.first() else {
            return Ok(None);
        };
        let Some(h) = header_len(t) else {
            return Err(v.poison(ErrorCode::StreamUnexpectedFrame));
        };
        if data.len() < h {
            return Ok(None);
        }
        self.admit(v, &data[..h]).map(Some)
    }

    /// Check the declared payload size against the verifier before buffering.
    fn admit(&mut self, v: &mut StreamVerifier<'_>, header: &[u8]) -> Result<usize, ErrorCode> {
        let (hdr, payload) = declared_total(header);
        let limit = v.frame_payload_limit(header[0]).unwrap_or(0);
        if payload > limit {
            return Err(v.poison(ErrorCode::StreamFrameSizeLie));
        }
        match usize::try_from(hdr + payload) {
            Ok(t) => Ok(t),
            Err(_) => Err(v.poison(ErrorCode::StreamFrameSizeLie)),
        }
    }
}
