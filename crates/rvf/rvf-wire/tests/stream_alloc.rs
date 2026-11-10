//! P7: size lies are rejected before allocation; peak allocation stays within the
//! policy-derived bound. Own test binary because it installs a global allocator.
//! (The only `unsafe` in the RVSD work: a counting `GlobalAlloc` wrapper, test-only.)
mod common;
use common::*;
use rvf_types::{ErrorCode as E, StreamEntry, StreamHeader};
use rvf_wire::stream::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), SeqCst) + l.size();
            PEAK.fetch_max(c, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CUR.fetch_sub(l.size(), SeqCst);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = System.realloc(p, l, n);
        if !q.is_null() {
            CUR.fetch_sub(l.size(), SeqCst);
            let c = CUR.fetch_add(n, SeqCst) + n;
            PEAK.fetch_max(c, SeqCst);
        }
        q
    }
}
#[global_allocator]
static A: Counting = Counting;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Peak bytes allocated above the level at entry while `f` runs.
fn peak_of<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let base = CUR.load(SeqCst);
    PEAK.store(base, SeqCst);
    let r = f();
    (r, PEAK.load(SeqCst).saturating_sub(base))
}

#[test]
fn p7_lying_descriptors_are_rejected_within_the_policy_bound() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let body = sample_body(1);
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let sk = key(1);
    let keys = [pk(1)];
    let mut pol = StreamPolicy::new(&keys);
    pol.max_descriptor_bytes = 64 * 1024;
    let bound = pol.descriptor_alloc_bound();
    type M = Box<dyn Fn(&mut Vec<StreamEntry>, &mut StreamHeader)>;
    let lies: Vec<(&str, M, E)> = vec![
        (
            "seg_count 2^32-1",
            Box::new(|_, h| {
                h.seg_count = u32::MAX;
                h.table_len = 0;
            }),
            E::StreamTableInvalid,
        ),
        (
            "seg_count 2^24",
            Box::new(|_, h| {
                h.seg_count = 1 << 24;
                h.table_len = (1u32 << 24).wrapping_mul(72);
            }),
            E::StreamLimitExceeded,
        ),
        (
            "stored_len 2^63",
            Box::new(|es, _| es[0].stored_len = 1 << 63),
            E::StreamLimitExceeded,
        ),
        (
            "uncompressed_len huge",
            Box::new(|es, _| es[0].uncompressed_len = u64::MAX),
            E::StreamLimitExceeded,
        ),
        (
            "body_len 2^63",
            Box::new(|_, h| h.body_len = 1 << 63),
            E::StreamLimitExceeded,
        ),
        (
            "class total",
            Box::new(|es, _| {
                for e in es.iter_mut() {
                    e.uncompressed_len = 1 << 20;
                }
                es[0].stored_len = 1 << 29;
            }),
            E::StreamTableInvalid,
        ),
    ];
    for (name, m, want) in lies {
        let d = resign(&b, &sk, |es, h| m(es, h));
        let (r, peak) = peak_of(|| decode_descriptor(&d, &pol));
        assert_eq!(r.err(), Some(want), "{name}");
        assert!(peak <= bound, "{name}: peak {peak} > bound {bound}");
    }
    // seg_count = 2^32-1 with a consistent table_len wraps in u32; a header-only
    // descriptor (210 bytes) declaring 2^24 segments is refused at the length check
    let mut h = b.header;
    h.seg_count = 1 << 24;
    h.table_len = (1u32 << 24) * 72; // 1.2 GB table claimed
    let mut d = h.to_bytes().to_vec();
    d.extend_from_slice(&[0u8; 66]);
    let (r, peak) = peak_of(|| decode_descriptor(&d, &pol));
    assert!(r.is_err());
    assert!(peak < 1024, "header-only lie allocated {peak} bytes");
    // an oversized descriptor is refused before anything is read or allocated
    let big = vec![0u8; pol.max_descriptor_bytes + 1];
    let (r, peak) = peak_of(|| decode_descriptor(&big, &pol));
    assert_eq!(r.err(), Some(E::StreamLimitExceeded));
    assert_eq!(peak, 0);
}

#[test]
fn p7_a_valid_max_size_descriptor_stays_within_the_bound() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use rvf_types::SegmentFlags as F;
    let mut body = Vec::new();
    for i in 0..800u64 {
        body.extend(rvf_wire::writer::write_segment(
            0x01,
            &[],
            F::empty(),
            i + 1,
        ));
    }
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let keys = [pk(1)];
    let mut pol = StreamPolicy::new(&keys);
    pol.max_descriptor_bytes = b.descriptor.len();
    let (r, peak) = peak_of(|| decode_descriptor(&b.descriptor, &pol));
    assert!(r.is_ok());
    assert!(
        peak <= pol.descriptor_alloc_bound(),
        "peak {peak} bound {}",
        pol.descriptor_alloc_bound()
    );
    assert!(
        peak <= 2 * b.descriptor.len() + 4096,
        "peak {peak} for a {} B descriptor",
        b.descriptor.len()
    );
}

#[test]
fn p7_declared_huge_segments_and_lying_frames_allocate_nothing_proportional() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let body = sample_body(1);
    let keys = [pk(1)];
    // A signed 1 TiB segment is a policy question, not an allocation: with the cap raised the
    // descriptor is accepted and the verifier still allocates only per-segment state.
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let d = resign(&b, &key(1), |es, h| {
        let extra = (1u64 << 40) - es[0].stored_len;
        es[0].stored_len += extra;
        h.body_len += extra;
        for e in es.iter_mut().skip(1) {
            e.body_offset += extra;
        }
        h.flags &= !rvf_types::stream::STREAM_FLAG_IDENTITY_PRESENT;
        h.rvf_identity = [0; 32];
    });
    let mut pol = StreamPolicy::new(&keys);
    pol.max_segment_bytes = 1 << 41;
    pol.max_body_len = 1 << 42;
    pol.max_class_bytes = [1 << 42; 4];
    let (v, peak) = peak_of(|| StreamVerifier::from_descriptor(&d, pol));
    let mut v = v.unwrap();
    assert!(peak <= pol.descriptor_alloc_bound(), "peak {peak}");
    // lying frames after a valid descriptor: nothing is allocated
    let mut chunk = vec![1u8];
    chunk.extend_from_slice(&1u64.to_le_bytes());
    chunk.extend_from_slice(&0u32.to_le_bytes());
    chunk.extend_from_slice(&u32::MAX.to_le_bytes()); // 4 GiB claimed, 0 carried
    let (r, peak) = peak_of(|| v.push(&chunk));
    assert_eq!(r.err(), Some(E::StreamFrameSizeLie));
    assert_eq!(peak, 0);
    // FrameReader: a 4 GiB length claim is refused at the header, nothing is buffered
    let mut v = StreamVerifier::from_descriptor(&b.descriptor, StreamPolicy::new(&keys)).unwrap();
    let mut rd = FrameReader::new();
    let (r, peak) = peak_of(|| rd.feed(&mut v, &chunk, &mut |_| {}));
    assert_eq!(r, Err(E::StreamFrameSizeLie));
    assert_eq!(v.state(), StreamState::Poisoned(E::StreamFrameSizeLie));
    assert!(peak <= 64, "peak {peak}");
    assert_eq!(rd.buffered(), 0);
}

#[test]
fn p7_whole_stream_peak_is_descriptor_plus_one_frame_plus_tables() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let body = sample_body(4);
    let keys = [pk(1)];
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    let bytes: Vec<u8> = frames.iter().flatten().copied().collect();
    let pol = StreamPolicy::new(&keys);
    let (state, peak) = peak_of(|| {
        let mut v = StreamVerifier::new(pol);
        let mut rd = FrameReader::new();
        for part in bytes.chunks(1500) {
            rd.feed(&mut v, part, &mut |_| {}).unwrap();
        }
        v.state()
    });
    assert_eq!(state, StreamState::Complete);
    let max_table: usize = b
        .entries
        .iter()
        .map(|e| 32 * b.header.chunk_count(e.stored_len).unwrap() as usize)
        .max()
        .unwrap();
    let bound = pol.descriptor_alloc_bound() + 2 * (4096 + 17) + max_table + 64 * 1024;
    assert!(peak <= bound, "peak {peak} bound {bound}");
    eprintln!(
        "whole-stream peak {peak} B (bound {bound}, body {} B)",
        body.len()
    );
}

/// F-17: retained chunk tables (kept for duplicate checks) never exceed the policy cap.
#[test]
fn p7_outstanding_table_bytes_are_capped_before_allocation() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use rvf_types::SegmentFlags as F;
    let mut body = Vec::new();
    for i in 0..8u64 {
        // 20 KiB payload, 4 KiB chunks: 6 chunks, a 192-byte table each
        body.extend(rvf_wire::writer::write_segment(
            0x01,
            &vec![i as u8; 20_000],
            F::empty(),
            i + 1,
        ));
    }
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let keys = [pk(1)];
    let mut pol = StreamPolicy::new(&keys);
    pol.max_table_bytes = 3 * 32 * 6; // room for exactly 3 tables
    pol.fault_budget = 100;
    let mut v = StreamVerifier::from_descriptor(&b.descriptor, pol).unwrap();
    let tables: Vec<Vec<u8>> = (0..b.entries.len())
        .map(|i| b.segment_frames(&body, i).remove(0))
        .collect();
    let (_, base) = peak_of(|| {
        for t in &tables[..3] {
            v.push(t).unwrap();
        }
    });
    assert!(
        base <= 3 * 32 * 6 + 256,
        "accepted tables used {base} bytes"
    );
    for t in &tables[3..] {
        let (r, peak) = peak_of(|| v.push(t));
        assert_eq!(r.err(), Some(E::StreamLimitExceeded));
        assert_eq!(peak, 0, "a refused table allocated memory");
    }
    // A finished segment's table is freed (F3-06): the retained bound is about
    // OPEN segments, so completing segment 0 makes room for exactly one more.
    let first = b.segment_frames(&body, 0);
    for f in &first[1..] {
        v.push(f).unwrap();
    }
    assert!(
        v.push(&tables[3]).is_ok(),
        "room freed by the finished segment"
    );
    assert_eq!(v.push(&tables[4]).err(), Some(E::StreamLimitExceeded));
}

/// F3-06: tables are freed when their segment verifies, so streaming many
/// multi-chunk segments in sequence retains about one table, not all of them.
#[test]
fn p7_finished_segment_tables_are_freed() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use rvf_types::SegmentFlags as F;
    let mut body = Vec::new();
    for i in 0..16u64 {
        // 1 MiB payload, 4 KiB chunks: 256 chunks, an 8 KiB table per segment
        body.extend(rvf_wire::writer::write_segment(
            0x01,
            &vec![i as u8; 1 << 20],
            F::empty(),
            i + 1,
        ));
    }
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let keys = [pk(1)];
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    v.push(&frames[0]).unwrap();
    let (_, peak) = peak_of(|| {
        for f in &frames[1..] {
            v.push(f).unwrap();
        }
    });
    assert_eq!(v.state(), StreamState::Complete);
    // Retaining every table would hold 16 x 8 KiB = 128 KiB; one open table is 8 KiB.
    assert!(
        peak < 64 * 1024,
        "peak {peak} B: finished tables were retained"
    );
}

/// F3-06: the presets are strictly tighter than the defaults and keep the
/// derived table cap.
#[test]
fn presets_are_tighter_than_default() {
    let keys = [pk(1)];
    let d = StreamPolicy::new(&keys);
    for p in [
        StreamPolicy::conservative(&keys),
        StreamPolicy::embedded(&keys),
    ] {
        assert!(p.max_body_len < d.max_body_len);
        assert!(p.max_segment_bytes < d.max_segment_bytes);
        assert!(p.max_table_bytes < d.max_table_bytes);
        assert_eq!(p.max_table_bytes, p.max_body_len / 4096 * 32);
    }
}
