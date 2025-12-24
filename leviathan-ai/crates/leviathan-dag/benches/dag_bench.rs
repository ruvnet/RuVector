use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn placeholder_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            black_box(1 + 1);
        });
    });
}

criterion_group!(benches, placeholder_benchmark);
criterion_main!(benches);
