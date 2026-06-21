//! Benchmark witness log operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rvm_types::{WitnessRecord, WitnessRecordV2};
use rvm_witness::{Blake3SealSigner, WitnessLog, WitnessLogV2};

fn bench_witness_append(c: &mut Criterion) {
    c.bench_function("witness_log_append_256", |b| {
        let mut log = WitnessLog::<256>::new();
        b.iter(|| {
            black_box(log.append(WitnessRecord::zeroed()));
        });
    });
}

/// ADR-134 v2 per-record append: one keyed BLAKE3 compression plus
/// bookkeeping. Target: < 1 us per append.
fn bench_witness_v2_append(c: &mut Criterion) {
    c.bench_function("witness_log_v2_append_256", |b| {
        let log = WitnessLogV2::<256, 256>::new();
        b.iter(|| {
            black_box(log.append(WitnessRecordV2::zeroed()));
        });
    });
}

/// Segment seal cost (256 leaf hashes + Merkle tree + one MAC),
/// paid once per 256 records, off the append path.
fn bench_witness_v2_seal_segment(c: &mut Criterion) {
    c.bench_function("witness_log_v2_seal_segment_256", |b| {
        let signer = Blake3SealSigner::new([0x42u8; 32]);
        b.iter_with_setup(
            || {
                let log = WitnessLogV2::<256, 256>::new();
                for _ in 0..256 {
                    log.append(WitnessRecordV2::zeroed());
                }
                log
            },
            |log| {
                black_box(log.seal_segment(&signer));
            },
        );
    });
}

criterion_group!(
    benches,
    bench_witness_append,
    bench_witness_v2_append,
    bench_witness_v2_seal_segment
);
criterion_main!(benches);
