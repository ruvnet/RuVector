//! Stages 2-4 of the EML HNSW proof chain.
//!
//! Stage 2: Synthetic end-to-end (10K vectors, 128-dim, QPS + Recall@10)
//! Stage 3: Real dataset (SIFT1M) — deferred if unavailable
//! Stage 4: Hypothesis test (Spearman rank correlation for 16-dim decomposition)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_eml_hnsw::{cosine_distance_f32, EmlDistanceModel};

// ---------------------------------------------------------------------------
// Deterministic PRNG
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
        (0..dim).map(|_| self.next_f32() * 2.0 - 1.0).collect()
    }
}

// ---------------------------------------------------------------------------
// Brute-force KNN (ground truth)
// ---------------------------------------------------------------------------

fn brute_force_knn(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_distance_f32(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

fn compute_recall(approx: &[usize], exact: &[usize]) -> f64 {
    let exact_set: std::collections::HashSet<usize> = exact.iter().cloned().collect();
    let hits = approx.iter().filter(|i| exact_set.contains(i)).count();
    hits as f64 / exact.len() as f64
}

// ---------------------------------------------------------------------------
// Simple linear-scan HNSW substitute for benchmarking
// (We don't have access to ruvector-core's HNSW from this crate, so we
//  implement a minimal flat-scan "index" and measure the EML distance
//  speedup on top of it. This is honest: we're measuring the distance
//  function speedup, not a full HNSW speedup.)
// ---------------------------------------------------------------------------

/// Flat-scan search using full cosine distance. Returns top-k indices.
fn flat_scan_full(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    brute_force_knn(data, query, k)
}

/// Flat-scan search using EML fast distance. Returns top-k indices.
fn flat_scan_eml(
    data: &[Vec<f32>],
    query: &[f32],
    k: usize,
    model: &EmlDistanceModel,
) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, model.fast_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

// ---------------------------------------------------------------------------
// Stage 2: Synthetic end-to-end benchmark
// ---------------------------------------------------------------------------

fn bench_e2e_synthetic(c: &mut Criterion) {
    let n = 10_000;
    let dim = 128;
    let n_queries = 100; // fewer queries for criterion (it iterates)
    let k = 10;

    let mut rng = Lcg::new(12345);
    let data: Vec<Vec<f32>> = (0..n).map(|_| rng.gen_vec(dim)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| rng.gen_vec(dim)).collect();

    // Train EML distance model
    let mut model = EmlDistanceModel::new(dim, 16);
    // Use first 500 pairs from data for training
    for pair in data[..501].windows(2) {
        let exact = cosine_distance_f32(&pair[0], &pair[1]);
        model.record(&pair[0], &pair[1], exact);
    }
    model.train();

    // Pre-compute ground truth for recall measurement
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_knn(&data, q, k))
        .collect();

    let mut group = c.benchmark_group("e2e_synthetic_10k_128d");

    // Baseline: full cosine flat scan
    group.bench_function("baseline_full_cosine", |b| {
        b.iter(|| {
            let mut total_recall = 0.0f64;
            for (i, q) in queries.iter().enumerate() {
                let results = flat_scan_full(black_box(&data), black_box(q), k);
                total_recall += compute_recall(&results, &ground_truth[i]);
            }
            black_box(total_recall / n_queries as f64)
        })
    });

    // EML: fast distance flat scan
    group.bench_function("eml_fast_distance", |b| {
        b.iter(|| {
            let mut total_recall = 0.0f64;
            for (i, q) in queries.iter().enumerate() {
                let results = flat_scan_eml(black_box(&data), black_box(q), k, &model);
                total_recall += compute_recall(&results, &ground_truth[i]);
            }
            black_box(total_recall / n_queries as f64)
        })
    });

    group.finish();

    // Print recall numbers outside criterion (for the proof report)
    let mut baseline_recall = 0.0f64;
    let mut eml_recall = 0.0f64;
    for (i, q) in queries.iter().enumerate() {
        let base_results = flat_scan_full(&data, q, k);
        let eml_results = flat_scan_eml(&data, q, k, &model);
        baseline_recall += compute_recall(&base_results, &ground_truth[i]);
        eml_recall += compute_recall(&eml_results, &ground_truth[i]);
    }
    eprintln!(
        "[PROOF] Stage 2 Recall@{}: baseline={:.4}, eml={:.4}",
        k,
        baseline_recall / n_queries as f64,
        eml_recall / n_queries as f64,
    );
}

// ---------------------------------------------------------------------------
// Stage 3: Real dataset (SIFT1M) — check if available
// ---------------------------------------------------------------------------

fn bench_sift_dataset(c: &mut Criterion) {
    let sift_path = std::path::Path::new("bench_data/sift/sift_base.fvecs");

    if !sift_path.exists() {
        eprintln!(
            "[PROOF] Stage 3: SIFT1M dataset not found at {:?}. \
             Skipping real-dataset benchmark. \
             Download from http://corpus-texmex.irisa.fr/ to enable.",
            sift_path,
        );
        // Register a no-op benchmark so criterion doesn't complain
        let mut group = c.benchmark_group("sift_dataset");
        group.bench_function("not_available", |b| {
            b.iter(|| black_box("SIFT1M not downloaded"))
        });
        group.finish();
        return;
    }

    // If we get here, SIFT data exists — load and benchmark
    eprintln!("[PROOF] Stage 3: Loading SIFT1M from {:?}", sift_path);
    let data = load_fvecs(sift_path.to_str().unwrap());
    let n = data.len().min(100_000); // cap at 100K for bench time
    let data = &data[..n];
    let dim = if data.is_empty() { 128 } else { data[0].len() };

    let mut rng = Lcg::new(777);
    let queries: Vec<Vec<f32>> = (0..100).map(|_| rng.gen_vec(dim)).collect();

    let mut model = EmlDistanceModel::new(dim, 16);
    for pair in data[..501.min(n)].windows(2) {
        let exact = cosine_distance_f32(&pair[0], &pair[1]);
        model.record(&pair[0], &pair[1], exact);
    }
    model.train();

    let mut group = c.benchmark_group("sift_dataset");
    group.sample_size(10); // SIFT is large, reduce iterations

    group.bench_function("sift_full_cosine_100q", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = flat_scan_full(black_box(data), black_box(q), 10);
            }
        })
    });

    group.bench_function("sift_eml_fast_100q", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = flat_scan_eml(black_box(data), black_box(q), 10, &model);
            }
        })
    });

    group.finish();
}

/// Load vectors in .fvecs format (used by SIFT1M, GIST, etc.)
fn load_fvecs(path: &str) -> Vec<Vec<f32>> {
    use std::io::Read;

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[PROOF] Failed to open {}: {}", path, e);
            return Vec::new();
        }
    };

    let mut data = Vec::new();
    let mut buf = [0u8; 4];

    loop {
        // Read dimension (4-byte int)
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(buf) as usize;

        // Read dim floats
        let mut vec = vec![0.0f32; dim];
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, dim * 4)
        };
        if file.read_exact(byte_slice).is_err() {
            break;
        }
        data.push(vec);

        // Cap at 1M vectors
        if data.len() >= 1_000_000 {
            break;
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Stage 4: Hypothesis test — Spearman rank correlation
// ---------------------------------------------------------------------------

fn bench_hypothesis_test(c: &mut Criterion) {
    let dim = 128;
    let selected_k = 16;
    let n_vectors = 1000;

    let mut rng = Lcg::new(54321);
    let vectors: Vec<Vec<f32>> = (0..n_vectors).map(|_| rng.gen_vec(dim)).collect();

    // Train an EML model
    let mut model = EmlDistanceModel::new(dim, selected_k);
    for pair in vectors[..501].windows(2) {
        let exact = cosine_distance_f32(&pair[0], &pair[1]);
        model.record(&pair[0], &pair[1], exact);
    }
    model.train();

    // Compute Spearman rank correlation:
    // For each query, rank all other vectors by full distance and by EML distance.
    // Compute correlation across all query pairs.
    let n_test_queries = 50;

    let mut all_rho = Vec::new();
    for qi in 0..n_test_queries {
        let query = &vectors[qi];

        // Full-distance rankings
        let mut full_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != qi)
            .map(|(i, v)| (i, cosine_distance_f32(query, v)))
            .collect();
        full_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // EML-distance rankings
        let mut eml_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != qi)
            .map(|(i, v)| (i, model.fast_distance(query, v)))
            .collect();
        eml_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        let n = full_dists.len();
        let mut full_rank = vec![0usize; n_vectors];
        let mut eml_rank = vec![0usize; n_vectors];
        for (rank, (idx, _)) in full_dists.iter().enumerate() {
            full_rank[*idx] = rank;
        }
        for (rank, (idx, _)) in eml_dists.iter().enumerate() {
            eml_rank[*idx] = rank;
        }

        // Spearman: 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
        let mut sum_d2 = 0.0f64;
        for (idx, _) in &full_dists {
            let d = full_rank[*idx] as f64 - eml_rank[*idx] as f64;
            sum_d2 += d * d;
        }
        let n_f = n as f64;
        let rho = 1.0 - (6.0 * sum_d2) / (n_f * (n_f * n_f - 1.0));
        all_rho.push(rho);
    }

    let mean_rho: f64 = all_rho.iter().sum::<f64>() / all_rho.len() as f64;
    let min_rho: f64 = all_rho.iter().cloned().fold(f64::MAX, f64::min);
    let max_rho: f64 = all_rho.iter().cloned().fold(f64::MIN, f64::max);

    let hypothesis_confirmed = mean_rho >= 0.95;

    eprintln!("[PROOF] Stage 4 — Hypothesis Test Results:");
    eprintln!("[PROOF]   Hypothesis: 16-dim decomposition preserves >95% ranking accuracy");
    eprintln!(
        "[PROOF]   Spearman rho: mean={:.4}, min={:.4}, max={:.4} (n={} queries)",
        mean_rho,
        min_rho,
        max_rho,
        n_test_queries,
    );
    eprintln!(
        "[PROOF]   Result: {} (mean rho {} 0.95)",
        if hypothesis_confirmed {
            "CONFIRMED"
        } else {
            "DISPROVEN"
        },
        if hypothesis_confirmed { ">=" } else { "<" },
    );

    // Also run a quick benchmark of the correlation computation itself
    let mut group = c.benchmark_group("hypothesis_test");

    group.bench_function("spearman_correlation_50q", |b| {
        b.iter(|| {
            let mut total_rho = 0.0f64;
            for qi in 0..n_test_queries.min(10) {
                let query = &vectors[qi];
                let mut full_dists: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != qi)
                    .map(|(i, v)| (i, cosine_distance_f32(query, v)))
                    .collect();
                full_dists.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut eml_dists: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != qi)
                    .map(|(i, v)| (i, model.fast_distance(query, v)))
                    .collect();
                eml_dists.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let n = full_dists.len();
                let mut full_rank = vec![0usize; n_vectors];
                let mut eml_rank = vec![0usize; n_vectors];
                for (rank, (idx, _)) in full_dists.iter().enumerate() {
                    full_rank[*idx] = rank;
                }
                for (rank, (idx, _)) in eml_dists.iter().enumerate() {
                    eml_rank[*idx] = rank;
                }

                let mut sum_d2 = 0.0f64;
                for (idx, _) in &full_dists {
                    let d = full_rank[*idx] as f64 - eml_rank[*idx] as f64;
                    sum_d2 += d * d;
                }
                let n_f = n as f64;
                total_rho += 1.0 - (6.0 * sum_d2) / (n_f * (n_f * n_f - 1.0));
            }
            black_box(total_rho)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_e2e_synthetic,
    bench_sift_dataset,
    bench_hypothesis_test,
);
criterion_main!(benches);
