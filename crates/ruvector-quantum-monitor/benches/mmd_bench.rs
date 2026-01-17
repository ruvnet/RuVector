//! MMD benchmark placeholder
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_mmd(_c: &mut Criterion) {}

criterion_group!(benches, bench_mmd);
criterion_main!(benches);
