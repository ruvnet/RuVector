use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_lvq::{FlatF32, FlatLvqIndex};

fn dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

fn measure_recall(
    truth: &[Vec<u32>],
    candidates: impl Iterator<Item = Vec<u32>>,
    k: usize,
) -> f64 {
    let mut hits = 0usize;
    let mut q = 0usize;
    for (t, c) in truth.iter().zip(candidates) {
        for id in &c {
            if t.contains(id) {
                hits += 1;
            }
        }
        q += 1;
    }
    hits as f64 / (k * q) as f64
}

#[test]
fn end_to_end_lvq8_recall_above_90() {
    let dim = 128;
    let n = 5_000;
    let nq = 32;
    let k = 10;

    let data = dataset(n, dim, 1);
    let queries = dataset(nq, dim, 2);

    let mut gt = FlatF32::new(dim);
    for v in data.chunks_exact(dim) {
        gt.push(v).unwrap();
    }
    let mut lvq8 = FlatLvqIndex::new_lvq8(dim);
    lvq8.extend_from_flat(&data).unwrap();

    let truth: Vec<Vec<u32>> = queries
        .chunks_exact(dim)
        .map(|q| {
            gt.search_l2(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();
    let approx = queries.chunks_exact(dim).map(|q| {
        lvq8.search_l2(q, k)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect()
    });

    let recall = measure_recall(&truth, approx, k);
    assert!(recall > 0.90, "lvq8 recall@10 = {recall:.3}");
}

#[test]
fn end_to_end_lvq8x8_rerank_recall_above_98() {
    let dim = 128;
    let n = 5_000;
    let nq = 32;
    let k = 10;

    let data = dataset(n, dim, 17);
    let queries = dataset(nq, dim, 18);

    let mut gt = FlatF32::new(dim);
    for v in data.chunks_exact(dim) {
        gt.push(v).unwrap();
    }
    let mut lvq8x8 = FlatLvqIndex::new_lvq8x8(dim);
    lvq8x8.extend_from_flat(&data).unwrap();

    let truth: Vec<Vec<u32>> = queries
        .chunks_exact(dim)
        .map(|q| {
            gt.search_l2(q, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();
    let approx = queries.chunks_exact(dim).map(|q| {
        lvq8x8
            .search_l2_reranked(q, k, k * 10)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect()
    });

    let recall = measure_recall(&truth, approx, k);
    assert!(recall > 0.98, "lvq8x8 reranked recall@10 = {recall:.3}");
}

#[test]
fn lvq8_byte_size_is_close_to_d_per_vector() {
    let dim = 128;
    let n = 1_000;
    let data = dataset(n, dim, 5);
    let mut lvq8 = FlatLvqIndex::new_lvq8(dim);
    lvq8.extend_from_flat(&data).unwrap();

    // Each vector: dim bytes of code + 12 bytes of stats (3 x f32).
    // Compare to 4*d for fp32 storage.
    let lvq_per_vec = lvq8.byte_size() as f64 / n as f64;
    let fp32_per_vec = (dim * 4) as f64;
    let ratio = lvq_per_vec / fp32_per_vec;
    assert!(
        ratio < 0.30,
        "expected <30% of fp32 footprint, got {ratio:.3}"
    );
}
