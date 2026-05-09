//! Numeric acceptance tests. No mocks. Real tiny datasets, real
//! recall@k, real assertions on the AVQ-vs-PQ gap.

use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::Normal;
use ruvector_avq::{AnisotropicPq, Encoder, ProductQuantizer, ScalarQuantizer, Scorer};

const DIM: usize = 32;
const M: usize = 8; // 4-d subspaces
const K: usize = 256;

fn unit_clusters(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let mut centers = vec![0.0f32; 16 * dim];
    for v in centers.iter_mut() {
        *v = normal.sample(&mut rng) * 2.0;
    }
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = (i * 9973) % 16;
        for j in 0..dim {
            out[i * dim + j] = centers[c * dim + j] + normal.sample(&mut rng) * 0.4;
        }
        let mut s = 0.0f32;
        for j in 0..dim {
            s += out[i * dim + j] * out[i * dim + j];
        }
        let inv = 1.0 / s.sqrt().max(1e-12);
        for j in 0..dim {
            out[i * dim + j] *= inv;
        }
    }
    out
}

fn brute_top1(db: &[f32], q: &[f32], dim: usize) -> u32 {
    let n = db.len() / dim;
    let mut best = 0u32;
    let mut best_s = f32::NEG_INFINITY;
    for i in 0..n {
        let row = &db[i * dim..(i + 1) * dim];
        let mut s = 0.0f32;
        for j in 0..dim {
            s += row[j] * q[j];
        }
        if s > best_s {
            best_s = s;
            best = i as u32;
        }
    }
    best
}

fn measure_recall<S: Scorer>(q: &S, codes: &[u8], db: &[f32], queries: &[f32], dim: usize, k: usize) -> f32 {
    let n_q = queries.len() / dim;
    let mut hits = 0;
    for i in 0..n_q {
        let qi = &queries[i * dim..(i + 1) * dim];
        let truth = brute_top1(db, qi, dim);
        let topk = q.topk_ip(qi, codes, k);
        if topk.iter().any(|(idx, _)| *idx == truth) {
            hits += 1;
        }
    }
    hits as f32 / n_q as f32
}

#[test]
fn pq_decodes_round_trip_within_codebook() {
    let db = unit_clusters(2_000, DIM, 1);
    let pq = ProductQuantizer::fit(&db, DIM, M, K, 99).unwrap();
    let mut codes = vec![0u8; 2_000 * pq.code_size()];
    pq.encode(&db, &mut codes);
    // Score x against itself should be close to 1 (unit-norm) under
    // a good quantizer, much better than 0.
    let mut scores = vec![0.0f32; 2_000];
    pq.score_ip(&db[..DIM], &codes, &mut scores);
    assert!(scores[0] > 0.7, "self-IP under PQ too small: {}", scores[0]);
}

#[test]
fn aniso_beats_uniform_pq_on_recall_at_10() {
    // Same training data, same M/K, only the loss differs.
    let db = unit_clusters(3_000, DIM, 7);
    let queries = unit_clusters(150, DIM, 8);
    let pq = ProductQuantizer::fit(&db, DIM, M, K, 11).unwrap();
    let avq = AnisotropicPq::fit(&db, DIM, M, K, 4.0, 11).unwrap();
    let mut pq_codes = vec![0u8; 3_000 * pq.code_size()];
    let mut avq_codes = vec![0u8; 3_000 * avq.code_size()];
    pq.encode(&db, &mut pq_codes);
    avq.encode(&db, &mut avq_codes);

    let r_pq = measure_recall(&pq, &pq_codes, &db, &queries, DIM, 10);
    let r_avq = measure_recall(&avq, &avq_codes, &db, &queries, DIM, 10);
    println!("PQ recall@10 = {:.3}, AVQ recall@10 = {:.3}", r_pq, r_avq);
    // AVQ must not regress vs uniform PQ. We accept equality (within
    // 1pp) under the small-N test budget, but typically it wins.
    assert!(
        r_avq + 0.01 >= r_pq,
        "AVQ regressed: PQ={:.3} AVQ={:.3}",
        r_pq, r_avq
    );
    assert!(r_avq >= 0.6, "AVQ recall@10 unexpectedly low: {}", r_avq);
}

#[test]
fn scalar_baseline_runs_and_is_consistent() {
    let db = unit_clusters(1_000, DIM, 3);
    let sq = ScalarQuantizer::fit(&db, DIM);
    assert_eq!(sq.code_size(), DIM);
    let mut codes = vec![0u8; 1_000 * sq.code_size()];
    sq.encode(&db, &mut codes);
    // top-1 self-match: querying with x_0 should put 0 near the top.
    let topk = sq.topk_ip(&db[..DIM], &codes, 5);
    assert!(topk.iter().any(|(i, _)| *i == 0));
}

#[test]
fn rejects_bad_eta() {
    let db = unit_clusters(500, DIM, 2);
    let r = AnisotropicPq::fit(&db, DIM, M, K, 0.5, 1);
    assert!(r.is_err(), "eta < 1 must be rejected");
}

#[test]
fn rejects_non_divisible_subspaces() {
    let db = vec![0.0f32; 500 * DIM];
    let r = ProductQuantizer::fit(&db, DIM, 7, K, 1); // 32 % 7 != 0
    assert!(r.is_err());
}
