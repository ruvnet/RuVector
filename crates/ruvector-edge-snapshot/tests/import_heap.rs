//! Heap regression for the import path, measured with a counting global
//! allocator (own test binary; one test fn so cases run sequentially).
//!
//! Before the fix, one 8 MiB hostile sidecar (2.8M empty ids) peaked at
//! +135 MB and one legitimate dim-1 8 MiB `VEC_SEG` (699k rows) at +63 MB,
//! both inside a single `push`. Now a sidecar is validated in place without
//! allocating and rows are pulled one batch at a time, so both stay near the
//! encoded record size (≤ 2 × 8 MiB reassembly capacity + one batch).

use ruvector_edge_snapshot::rvf_format::{legacy_content_hash, DirEntry, RvfManifest};
use ruvector_edge_snapshot::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CURRENT.fetch_sub(l.size(), Relaxed);
    }
}

#[global_allocator]
static A: Counting = Counting;

const MAX: usize = 8 << 20;

fn seg(file: &mut Vec<u8>, t: u8, id: u64, payload: &[u8]) -> DirEntry {
    let offset = file.len() as u64;
    let mut h = [0u8; 64];
    h[0..4].copy_from_slice(&0x5256_4653u32.to_le_bytes());
    h[4] = 1;
    h[5] = t;
    h[8..16].copy_from_slice(&id.to_le_bytes());
    h[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    h[0x28..0x38].copy_from_slice(&legacy_content_hash(payload));
    file.extend_from_slice(&h);
    file.extend_from_slice(payload);
    DirEntry {
        seg_id: id,
        offset,
        payload_len: payload.len() as u64,
        seg_type: t,
    }
}

/// `[sidecar] + VEC(dim 1, n rows) + manifest`.
fn file(sidecar: Option<Vec<u8>>, n: usize) -> Vec<u8> {
    let mut f = Vec::with_capacity(2 * MAX + 4096);
    if let Some(s) = sidecar {
        seg(&mut f, 0x0B, 1, &s);
    }
    let mut v = 1u16.to_le_bytes().to_vec();
    v.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n as u64 {
        v.extend_from_slice(&i.to_le_bytes());
        v.extend_from_slice(&1f32.to_le_bytes());
    }
    let d = seg(&mut f, 0x01, 2, &v);
    let m = RvfManifest {
        epoch: 1,
        dim: 1,
        total_vectors: n as u64,
        profile: 0,
        metric_id: 0,
        dir: vec![d],
        deleted: vec![],
    };
    seg(&mut f, 0x05, 3, &m.encode());
    f
}

/// Stream `f` in 1 MiB pieces, dropping each batch; returns (result, rows, peak Δ).
fn measure(f: &[u8]) -> (Result<u64, ImportError>, u64, usize) {
    let spec = ImportSpec {
        dim: 1,
        metric: Metric::L2,
        limits: ImportLimits::default(),
    };
    let summary = inspect_tail(f, f.len() as u64, &spec.limits).unwrap();
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut rows = 0u64;
    let r = (|| {
        let mut imp = RvfImporter::new(spec, summary, ImportCursor::default())?;
        for piece in f.chunks(1 << 20) {
            imp.feed(piece)?;
            while let Some(b) = imp.next_batch()? {
                assert!(b.rows.len() <= 500);
                rows += b.rows.len() as u64;
            }
        }
        Ok(imp.finish()?.rows)
    })();
    (r, rows, PEAK.load(Relaxed) - base)
}

#[test]
fn import_heap_is_bounded_by_the_encoded_record_not_the_decoded_rows() {
    // Reassembly buffer (≤ 2× capacity growth of an 8 MiB record) + one
    // 1 MiB piece + one batch + slack.
    let bound = 2 * (MAX + 128) + (2 << 20);

    // 1. Hostile sidecar: 2.8M rows of 3 bytes (empty id, no metadata),
    //    paired with a one-row VEC. Refused, without decoding the rows.
    let count = (MAX - 9) / 3;
    let mut s = b"RVEI\x01".to_vec();
    s.extend_from_slice(&(count as u32).to_le_bytes());
    s.resize(9 + 3 * count, 0);
    let f = file(Some(s), 1);
    let (r, rows, peak) = measure(&f);
    eprintln!("hostile sidecar: {r:?}, peak +{peak} B");
    assert_eq!(r.unwrap_err(), ImportError::Malformed("sidecar count"));
    assert_eq!(rows, 0);
    assert!(peak <= bound, "hostile sidecar peak {peak} > {bound}");

    // Same shape but the count matches a VEC: empty ids are refused in place.
    let mut s = b"RVEI\x01".to_vec();
    s.extend_from_slice(&1u32.to_le_bytes());
    s.extend_from_slice(&[0, 0, 0]);
    assert_eq!(
        measure(&file(Some(s), 1)).0.unwrap_err(),
        ImportError::Malformed("sidecar row")
    );

    // 2. Legitimate dim-1 collection: one 8 MiB VEC_SEG of 699k rows,
    //    pulled one batch at a time.
    let n = (MAX - 6) / 12;
    let f = file(None, n);
    let (r, rows, peak) = measure(&f);
    eprintln!("dim-1 vec: {n} rows, peak +{peak} B");
    assert_eq!(r.unwrap(), n as u64);
    assert_eq!(rows, n as u64);
    assert!(peak <= bound, "dim-1 peak {peak} > {bound}");

    // 3. Same rows with a matching sidecar (string ids + metadata), read lazily.
    let n = 200_000;
    let mut s = b"RVEI\x01".to_vec();
    s.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        let id = format!("k{i}");
        s.extend_from_slice(&(id.len() as u16).to_le_bytes());
        s.extend_from_slice(id.as_bytes());
        s.push(1);
        s.extend_from_slice(&2u16.to_le_bytes());
        s.extend_from_slice(b"{}");
    }
    let f = file(Some(s), n);
    let (r, _, peak) = measure(&f);
    eprintln!("dim-1 vec + sidecar: {n} rows, peak +{peak} B");
    assert_eq!(r.unwrap(), n as u64);
    assert!(peak <= bound, "sidecar pair peak {peak} > {bound}");
}
