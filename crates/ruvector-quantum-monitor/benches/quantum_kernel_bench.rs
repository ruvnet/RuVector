//! Quantum kernel benchmark placeholder  
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_kernel(_c: &mut Criterion) {}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
