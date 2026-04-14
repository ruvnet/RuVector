//! Stage 1: Micro-benchmarks for each EML HNSW optimization in isolation.
//!
//! Measures:
//! - Cosine decomposition: full 128-dim vs EML 16-dim selected-dim distance
//! - Adaptive ef: prediction overhead per query
//! - Path prediction: prediction overhead per query
//! - Rebuild prediction: prediction overhead per observation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_eml_hnsw::{
    cosine_distance_f32, AdaptiveEfModel, EmlDistanceModel, GraphStats, RebuildPredictor,
    SearchPathPredictor,
};

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator (no dependency on rand)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f32 / (u32::MAX as f32)
    }
    fn gen_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

// ---------------------------------------------------------------------------
// Stage 1a: Cosine decomposition — full vs selected-dim distance
// ---------------------------------------------------------------------------

fn bench_cosine_decomp(c: &mut Criterion) {
    let dim = 128;
    let selected_k = 16;
    let n_pairs = 500;

    let mut rng = Lcg::new(42);
    let vectors: Vec<Vec<f32>> = (0..n_pairs + 1).map(|_| rng.gen_vec(dim)).collect();

    let mut group = c.benchmark_group("cosine_decomp");

    // Baseline: full 128-dim cosine distance
    group.bench_function("full_128d_cosine", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for pair in vectors.windows(2) {
                sum += cosine_distance_f32(black_box(&pair[0]), black_box(&pair[1]));
            }
            black_box(sum)
        })
    });

    // EML: selected 16-dim L2 proxy (raw, no model — measures dimension
    // reduction speedup independent of EML overhead)
    let selected: Vec<usize> = (0..selected_k).collect();
    group.bench_function("selected_16d_l2_proxy", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for pair in vectors.windows(2) {
                let d: f32 = selected
                    .iter()
                    .map(|&i| (pair[0][i] - pair[1][i]).powi(2))
                    .sum();
                sum += d;
            }
            black_box(sum)
        })
    });

    // EML: trained model fast_distance
    // Train a model so we measure real EML prediction latency
    let mut model = EmlDistanceModel::new(dim, selected_k);
    for pair in vectors.windows(2) {
        let exact = cosine_distance_f32(&pair[0], &pair[1]);
        model.record(&pair[0], &pair[1], exact);
    }
    model.train();

    group.bench_function("eml_16d_fast_distance", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for pair in vectors.windows(2) {
                sum += model.fast_distance(black_box(&pair[0]), black_box(&pair[1]));
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 1b: Adaptive ef prediction overhead
// ---------------------------------------------------------------------------

fn bench_adaptive_ef(c: &mut Criterion) {
    let mut rng = Lcg::new(99);
    let queries: Vec<Vec<f32>> = (0..200).map(|_| rng.gen_vec(128)).collect();

    // Train the model
    let mut model = AdaptiveEfModel::new(64, 10, 200);
    for q in &queries {
        let t = q[0];
        let ef = (20.0 + t * 100.0) as usize;
        let recall = if ef < 80 { 0.98 } else { 0.92 };
        model.record(q, 10_000, ef, recall);
    }
    model.train();

    let mut group = c.benchmark_group("adaptive_ef");

    // Baseline: returning a fixed constant (zero overhead)
    group.bench_function("fixed_ef_100", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &queries {
                let _ = black_box(q);
                sum += 100;
            }
            black_box(sum)
        })
    });

    // EML: predict ef per query
    group.bench_function("eml_predict_ef", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &queries {
                sum += model.predict_ef(black_box(q), 10_000);
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 1c: Path prediction overhead
// ---------------------------------------------------------------------------

fn bench_path_prediction(c: &mut Criterion) {
    let dim = 32;
    let mut rng = Lcg::new(77);

    // Train the predictor with 3 regions
    let mut predictor = SearchPathPredictor::new(3, dim);

    // Region A: near origin
    for i in 0..150 {
        let v = i as f32 * 0.001;
        let q: Vec<f32> = (0..dim).map(|d| v + d as f32 * 0.001).collect();
        predictor.record_search(&q, &[100, 101, 102]);
    }
    // Region B: around 5.0
    for i in 0..150 {
        let v = 5.0 + i as f32 * 0.001;
        let q: Vec<f32> = (0..dim).map(|d| v + d as f32 * 0.001).collect();
        predictor.record_search(&q, &[200, 201, 202]);
    }
    // Region C: around 10.0
    for i in 0..150 {
        let v = 10.0 + i as f32 * 0.001;
        let q: Vec<f32> = (0..dim).map(|d| v + d as f32 * 0.001).collect();
        predictor.record_search(&q, &[300, 301, 302]);
    }
    predictor.train();

    let test_queries: Vec<Vec<f32>> = (0..200).map(|_| rng.gen_vec(dim)).collect();

    let mut group = c.benchmark_group("path_prediction");

    // Baseline: no prediction (empty vec return)
    group.bench_function("no_prediction", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &test_queries {
                let _ = black_box(q);
                sum += 0; // would start from root
            }
            black_box(sum)
        })
    });

    // EML: predict entry points
    group.bench_function("eml_predict_entries", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for q in &test_queries {
                let entries = predictor.predict_entries(black_box(q));
                sum += entries.len();
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Stage 1d: Rebuild prediction overhead
// ---------------------------------------------------------------------------

fn bench_rebuild_prediction(c: &mut Criterion) {
    // Train the predictor
    let mut predictor = RebuildPredictor::new();
    for i in 0..100 {
        let stats = GraphStats {
            inserts_since_rebuild: 100 + i * 50,
            deletes_since_rebuild: 10 + i * 5,
            total_entries: 10_000,
            graph_density: 0.7 - (i as f64) * 0.003,
            avg_recent_recall: 0.98 - (i as f64) * 0.003,
        };
        predictor.record(&stats, stats.avg_recent_recall);
    }
    predictor.train();

    let test_stats: Vec<GraphStats> = (0..200)
        .map(|i| GraphStats {
            inserts_since_rebuild: 500 + i * 100,
            deletes_since_rebuild: 50 + i * 10,
            total_entries: 10_000 + i * 100,
            graph_density: 0.5 + (i as f64 % 50.0) * 0.01,
            avg_recent_recall: 0.8 + (i as f64 % 20.0) * 0.01,
        })
        .collect();

    let mut group = c.benchmark_group("rebuild_prediction");

    // Baseline: fixed threshold check (every N inserts)
    group.bench_function("fixed_threshold_100", |b| {
        b.iter(|| {
            let mut rebuilds = 0usize;
            for s in &test_stats {
                if black_box(s.inserts_since_rebuild) > 5000 {
                    rebuilds += 1;
                }
            }
            black_box(rebuilds)
        })
    });

    // EML: learned prediction
    group.bench_function("eml_should_rebuild", |b| {
        b.iter(|| {
            let mut rebuilds = 0usize;
            for s in &test_stats {
                if predictor.should_rebuild(black_box(s)) {
                    rebuilds += 1;
                }
            }
            black_box(rebuilds)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_decomp,
    bench_adaptive_ef,
    bench_path_prediction,
    bench_rebuild_prediction,
);
criterion_main!(benches);
