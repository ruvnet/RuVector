//! M3 acceptance: a 100k × 384 import stays within quota and a fixed memory
//! bound. Exporter → importer are piped through a sink, so the ~150 MB file
//! never exists; a counting global allocator measures real peak heap use.
//! (Own test binary: the allocator sees nothing but this test.)

mod common;
use common::Rng;
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

const N: u64 = 100_000;
const DIM: u16 = 384;
const PER_SEG: u32 = 1024;

fn gen(i: u64, rng: &mut Rng) -> Row {
    Row {
        id: format!("doc-{i:09}"),
        values: (0..DIM).map(|_| rng.f32()).collect(),
        metadata: i.is_multiple_of(4).then(|| format!("{{\"i\":{i}}}")),
    }
}

/// Pass 1: the exporter is deterministic, so regenerate the file into a
/// rolling 64 KiB tail (what the Worker range-reads from R2) and summarise it.
fn tail_summary(limits: &ImportLimits) -> RvfSummary {
    let (mut tail, mut len) = (Vec::new(), 0u64);
    let mut sink = |piece: &[u8]| {
        len += piece.len() as u64;
        tail.extend_from_slice(piece);
        if tail.len() > 128 << 10 {
            tail.drain(..tail.len() - (64 << 10));
        }
    };
    let mut x = RvfExporter::new(DIM, Metric::Cosine, PER_SEG).unwrap();
    let mut src = Rng(0xfeed);
    for i in 0..N {
        x.push(&gen(i, &mut src), &mut sink).unwrap();
    }
    x.finish(&mut sink);
    inspect_tail(&tail, len, limits).unwrap()
}

#[test]
fn import_100k_x_384_streams_within_quota_and_memory_bound() {
    let limits = ImportLimits {
        max_rows: N,
        max_batch_rows: 500,
        ..ImportLimits::default()
    };
    let spec = ImportSpec {
        dim: DIM,
        metric: Metric::Cosine,
        limits,
    };
    let summary = tail_summary(&limits);
    assert_eq!((summary.planned_rows, summary.total_vectors), (N, N));
    // One row over quota is refused before any batch.
    let tight = ImportSpec {
        limits: ImportLimits {
            max_rows: N - 1,
            ..limits
        },
        ..spec
    };
    assert_eq!(
        RvfImporter::new(tight, summary.clone(), ImportCursor::default()).unwrap_err(),
        ImportError::QuotaExceeded { limit: N - 1 }
    );

    let baseline = CURRENT.load(Relaxed);
    PEAK.store(baseline, Relaxed);

    let mut importer = RvfImporter::new(spec, summary, ImportCursor::default()).unwrap();
    let mut exporter = RvfExporter::new(DIM, Metric::Cosine, PER_SEG).unwrap();
    let mut src = Rng(0xfeed);
    let mut check = Rng(0xfeed);
    let (mut seen, mut batches, mut file_bytes) = (0u64, 0u64, 0u64);
    let mut err = None;
    {
        let mut sink = |piece: &[u8]| {
            file_bytes += piece.len() as u64;
            if err.is_some() {
                return;
            }
            if let Err(e) = importer.feed(piece) {
                err = Some(e);
                return;
            }
            // The "shard": verify each batch against the source, then drop it.
            loop {
                let b = match importer.next_batch() {
                    Ok(Some(b)) => b,
                    Ok(None) => break,
                    Err(e) => {
                        err = Some(e);
                        return;
                    }
                };
                assert!(b.rows.len() <= 500);
                assert_eq!(b.seq, batches);
                batches += 1;
                for r in &b.rows {
                    assert!(r.bitwise_eq(&gen(seen, &mut check)));
                    seen += 1;
                }
            }
        };
        for i in 0..N {
            exporter.push(&gen(i, &mut src), &mut sink).unwrap();
        }
        exporter.finish(&mut sink);
    }
    assert_eq!(err, None);
    let peak_buffer = importer.peak_buffered_bytes();
    let totals = importer.finish().unwrap();
    let peak = PEAK.load(Relaxed) - baseline;

    assert_eq!((totals.rows, seen), (N, N));
    assert_eq!(totals.batches, batches);
    assert!(totals.sha256.is_some());
    assert!(file_bytes > 150_000_000, "file was {file_bytes} bytes");

    // Derived bound. One VEC payload is 6 + 1024 × (8 + 4·384) ≈ 1.58 MB.
    // Live at once: exporter vec + sidecar buffers (1 seg), importer
    // reassembly buffer (sidecar + vec record, ≤ 2 seg after Vec growth),
    // one batch (500 rows × ~1.6 KB ≈ 0.5 seg), plus 1 MiB slack.
    let seg = 6 + PER_SEG as usize * (8 + 4 * DIM as usize);
    // A record is the sidecar (1024 × ≤ 32 B of ids + metadata) plus the vec.
    let record = seg + 9 + PER_SEG as usize * 32 + 128;
    let bound = 4 * seg + (1 << 20);
    assert!(
        peak_buffer <= 2 * record,
        "importer buffer peak {peak_buffer}"
    );
    assert!(peak <= bound, "peak heap {peak} > bound {bound}");
    eprintln!("100k import: file {file_bytes} B, peak heap {peak} B (bound {bound}), importer buffer peak {peak_buffer} B");
}
