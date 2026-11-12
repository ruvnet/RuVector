//! Recall + latency for ruvnet/RuVector#773, on BOTH data distributions.
//!
//! The #773 fix changes which layer a symmetric edge is written on and removes
//! an early-return in `search_layer`, so the fair question is not only "are
//! points still dropped" but "did graph quality or speed regress". A single
//! distribution cannot answer it: clustered data stresses neighbour-list
//! contention, uniform data stresses long-range connectivity, and a change can
//! easily help one and hurt the other.
//!
//! Ground truth is exact brute force over the same vectors, so recall here is
//! recall@k against the true nearest neighbours -- not agreement with another
//! approximate index.
//!
//! ⚠️ Latency below is meaningful only within one build profile -- a debug run\n//! is several times slower than release. Compare like with like.\n//!\n//! Everything is deterministic: vectors come from a counter-based generator, so
//! a number in this report can be reproduced exactly. Only the HNSW level draw
//! stays random, which is the property under test.

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};
use std::time::Instant;

const DIMS: usize = 64;
const ROWS: usize = 2_000;
const QUERIES: usize = 100;
const K: usize = 10;

fn cfg() -> HnswConfig {
    HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 100,
        max_elements: 10_000,
    }
}

/// Counter-based deterministic float in [-1, 1).
fn rnd(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let x = (*state >> 33) as u32;
    (x as f32) / (1u32 << 30) as f32 - 1.0
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

/// Uniform on the sphere: stresses long-range connectivity.
fn uniform_set(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| normalise((0..DIMS).map(|_| rnd(&mut s)).collect()))
        .collect()
}

/// 20 tight clusters: stresses neighbour-list contention, where many points
/// compete for the same slots and edge bookkeeping matters most.
fn clustered_set(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    let centres: Vec<Vec<f32>> = (0..20)
        .map(|_| normalise((0..DIMS).map(|_| rnd(&mut s)).collect()))
        .collect();
    (0..n)
        .map(|i| {
            let c = &centres[i % centres.len()];
            normalise(c.iter().map(|x| x + 0.05 * rnd(&mut s)).collect())
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact top-k by cosine similarity -- the ground truth recall is measured against.
fn brute_force(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(q, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

struct Report {
    recall: f64,
    mean_us: f64,
}

fn measure(label: &str, data: &[Vec<f32>]) -> Report {
    let mut index = HnswIndex::new(DIMS, DistanceMetric::Cosine, cfg()).expect("index");
    for (i, v) in data.iter().enumerate() {
        index.add(i.to_string(), v.clone()).expect("insert");
    }
    assert_eq!(
        index.len(),
        data.len(),
        "{label}: the index lost rows before measuring"
    );

    let mut hit = 0usize;
    let mut total = 0usize;
    let mut elapsed = 0u128;
    for qi in 0..QUERIES {
        // Queries are drawn from the data's own distribution, not uniform noise
        // against clustered data -- otherwise the clustered number measures the
        // wrong question.
        let q = &data[(qi * 7 + 3) % data.len()];
        let truth = brute_force(data, q, K);
        let t0 = Instant::now();
        let hits = index.search(q, K).expect("search");
        elapsed += t0.elapsed().as_micros();
        let got: Vec<usize> = hits
            .iter()
            .filter_map(|h| h.id.parse::<usize>().ok())
            .collect();
        hit += got.iter().filter(|i| truth.contains(i)).count();
        total += truth.len();
    }
    let r = Report {
        recall: hit as f64 / total as f64,
        mean_us: elapsed as f64 / QUERIES as f64,
    };
    println!(
        "  {label:<10} recall@{K} = {:.4}   mean search = {:.1} us",
        r.recall, r.mean_us
    );
    r
}

/// The floor is placed from MEASUREMENT, between the two populations, not
/// guessed:
///
///   fixed build     5/5 independent level draws -> recall@10 exactly 1.0000
///                   on BOTH distributions
///   #773 defect     uniform 0.9910, clustered 0.9960
///
/// So 0.999 separates them. A looser floor (0.90 was the first draft) measures
/// the difference and gates nothing -- the defect sailed through it. A floor
/// that cannot distinguish the two populations it sits between is decoration.
///
/// ⚠️ The printed numbers are still the comparison ruvnet asked to retain; the
/// floor only stops a silent collapse reaching main.
const RECALL_FLOOR: f64 = 0.999;

#[test]
fn recall_and_latency_on_both_distributions() {
    println!("\nHNSW #773 -- {ROWS} rows, {DIMS}d, k={K}, {QUERIES} queries, exact ground truth");
    let uniform = measure("uniform", &uniform_set(ROWS, 0xA5A5_1234));
    let clustered = measure("clustered", &clustered_set(ROWS, 0x5A5A_9876));

    assert!(
        uniform.recall >= RECALL_FLOOR,
        "uniform recall@{K} collapsed to {:.4} (floor {RECALL_FLOOR})",
        uniform.recall
    );
    assert!(
        clustered.recall >= RECALL_FLOOR,
        "clustered recall@{K} collapsed to {:.4} (floor {RECALL_FLOOR})",
        clustered.recall
    );

    // Both distributions must be retained: a run that silently measured one of
    // them would satisfy the floors above while answering half the question.
    assert!(
        uniform.mean_us > 0.0 && clustered.mean_us > 0.0,
        "both distributions must actually have been measured"
    );
}
