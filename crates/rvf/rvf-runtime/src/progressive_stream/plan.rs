//! Server-side frame plan: the deterministic frame stream of a bundle in a
//! chosen segment order, addressable by byte offset (HTTP `Range`, TCP offset).
//!
//! Stream layout: `[descriptor frame][per segment: table frame if chunked,
//! then chunk frames]`. Frame sizes are known arithmetically, so a range
//! request never materialises more than the frames it overlaps.

use rvf_wire::stream::{descriptor_frame, StreamBundle};

#[derive(Clone, Copy)]
enum Kind {
    Desc,
    Table(usize),
    Chunk(usize, u32),
}

#[derive(Clone, Copy)]
struct Span {
    start: u64,
    len: u64,
    kind: Kind,
}

/// A frame plan over one body.
pub struct StreamPlan {
    bundle: StreamBundle,
    desc_frame: Vec<u8>,
    tables: Vec<Option<Vec<u8>>>,
    spans: Vec<Span>,
    total: u64,
}

impl StreamPlan {
    /// Plan the stream of `bundle` over `body`, segments in `(stage, body
    /// order)`. Hashes the body once to materialise chunk tables.
    pub fn new(bundle: StreamBundle, body: &[u8]) -> Self {
        let mut order: Vec<usize> = (0..bundle.entries.len()).collect();
        order.sort_by_key(|&i| (bundle.entries[i].stage, i));
        let desc_frame = descriptor_frame(&bundle.descriptor);
        let mut spans = vec![Span {
            start: 0,
            len: desc_frame.len() as u64,
            kind: Kind::Desc,
        }];
        let mut at = desc_frame.len() as u64;
        let mut tables = vec![None; bundle.entries.len()];
        for &i in &order {
            let e = &bundle.entries[i];
            let n = bundle.header.chunk_count(e.stored_len).unwrap_or(1);
            if n > 1 {
                let f = bundle.segment_frames(body, i).swap_remove(0);
                spans.push(Span {
                    start: at,
                    len: f.len() as u64,
                    kind: Kind::Table(i),
                });
                at += f.len() as u64;
                tables[i] = Some(f);
            }
            for c in 0..n {
                let len = 17 + bundle.header.chunk_len(e.stored_len, c).unwrap_or(0);
                spans.push(Span {
                    start: at,
                    len,
                    kind: Kind::Chunk(i, c),
                });
                at += len;
            }
        }
        Self {
            bundle,
            desc_frame,
            tables,
            spans,
            total: at,
        }
    }

    /// Total bytes of the frame stream.
    pub fn total_len(&self) -> u64 {
        self.total
    }

    /// The bundle.
    pub fn bundle(&self) -> &StreamBundle {
        &self.bundle
    }

    /// Number of frames.
    pub fn frame_count(&self) -> usize {
        self.spans.len()
    }

    /// Write bytes `[start, end)` of the stream by calling `out` with pieces;
    /// `out` returns `false` to stop. Returns bytes written. `end` is clamped.
    pub fn write_range(
        &self,
        body: &[u8],
        start: u64,
        end: u64,
        out: &mut dyn FnMut(&[u8]) -> bool,
    ) -> u64 {
        let end = end.min(self.total);
        if start >= end {
            return 0;
        }
        let first = self.spans.partition_point(|s| s.start + s.len <= start);
        let mut written = 0u64;
        for s in &self.spans[first..] {
            if s.start >= end {
                break;
            }
            let from = start.max(s.start) - s.start;
            let to = end.min(s.start + s.len) - s.start;
            let mut emit = |bytes: &[u8]| -> bool { out(bytes) };
            let ok = match s.kind {
                Kind::Desc => emit(&self.desc_frame[from as usize..to as usize]),
                Kind::Table(i) => {
                    emit(&self.tables[i].as_ref().expect("table")[from as usize..to as usize])
                }
                Kind::Chunk(i, c) => {
                    let e = &self.bundle.entries[i];
                    let csz = self.bundle.header.chunk_size().unwrap_or(0);
                    let clen = s.len - 17;
                    let b0 = (e.body_offset + c as u64 * csz) as usize;
                    let mut hdr = [0u8; 17];
                    hdr[0] = 0x01;
                    hdr[1..9].copy_from_slice(&e.seg_id.to_le_bytes());
                    hdr[9..13].copy_from_slice(&c.to_le_bytes());
                    hdr[13..17].copy_from_slice(&(clen as u32).to_le_bytes());
                    let mut ok = true;
                    if from < 17 {
                        ok = emit(&hdr[from as usize..(to.min(17)) as usize]);
                    }
                    if ok && to > 17 {
                        let a = b0 + (from.max(17) - 17) as usize;
                        let z = b0 + (to - 17) as usize;
                        ok = emit(&body[a..z]);
                    }
                    ok
                }
            };
            written += to - from;
            if !ok {
                break;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rvf_wire::stream::StreamDescriptorBuilder;
    use rvf_wire::{write_segment, writer::write_segment_with_algo};

    fn body() -> Vec<u8> {
        let mut b = Vec::new();
        for i in 0..4u64 {
            b.extend(write_segment_with_algo(
                0x01,
                &vec![i as u8; 5000 + i as usize * 3000],
                rvf_types::SegmentFlags::empty(),
                i + 1,
                2,
            ));
        }
        let _ = write_segment;
        b
    }

    #[test]
    fn every_range_matches_the_builder_frames() {
        let body = body();
        let key = SigningKey::from_bytes(&[3u8; 32]);
        let bundle = StreamDescriptorBuilder::new([1; 16], key)
            .chunk_log2(12)
            .stages(&[1, 0, 1, 0])
            .build(&body)
            .unwrap();
        let want: Vec<u8> = bundle.frames_stage_order(&body).flatten().collect();
        let plan = StreamPlan::new(bundle, &body);
        assert_eq!(plan.total_len(), want.len() as u64);
        let mut all = Vec::new();
        plan.write_range(&body, 0, u64::MAX, &mut |b| {
            all.extend_from_slice(b);
            true
        });
        assert_eq!(all, want);
        for (a, z) in [
            (0, 1),
            (3, 40),
            (17, 18),
            (100, 9000),
            (5000, 5001),
            (want.len() as u64 - 5, u64::MAX),
        ] {
            let mut got = Vec::new();
            plan.write_range(&body, a, z, &mut |b| {
                got.extend_from_slice(b);
                true
            });
            assert_eq!(
                got,
                want[a as usize..(z.min(want.len() as u64)) as usize],
                "{a}..{z}"
            );
        }
    }
}
