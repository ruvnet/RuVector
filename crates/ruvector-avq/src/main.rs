//! End-to-end demo: train scalar/PQ/AVQ on synthetic clustered
//! embeddings, run linear-scan top-k against ground truth, print
//! recall@10, code size, and per-query latency for each variant.

use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::Normal;
use ruvector_avq::{AnisotropicPq, Encoder, ProductQuantizer, ScalarQuantizer, Scorer};
use std::time::Instant;

const N: usize = 10_000;
const N_QUERIES: usize = 300;
const DIM: usize = 128;
const M: usize = 16; // 16 subspaces of 8 dims each
const K: usize = 256; // 8 bits per subspace
const TOPK: usize = 10;

/// Low-rank embeddings: points live in a `rank`-dim subspace embedded
/// in `dim` ambient dims, then l2-normalized. This mimics real
/// learned text/image embeddings (effective rank << ambient dim) and
/// is the regime where AVQ's parallel/orthogonal split is most
/// impactful: residuals along the data manifold dominate score error.
fn make_lowrank(n: usize, dim: usize, rank: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    // shared random basis (dim x rank)
    let mut basis = vec![0.0f32; dim * rank];
    for v in basis.iter_mut() {
        *v = normal.sample(&mut rng);
    }
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let mut coef = vec![0.0f32; rank];
        for c in coef.iter_mut() {
            *c = normal.sample(&mut rng);
        }
        for j in 0..dim {
            let mut acc = 0.0f32;
            for r in 0..rank {
                acc += basis[j * rank + r] * coef[r];
            }
            // small ambient noise so quantization isn't trivial
            out[i * dim + j] = acc + 0.05 * normal.sample(&mut rng);
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

fn brute_topk(db: &[f32], q: &[f32], dim: usize, k: usize) -> Vec<u32> {
    let n = db.len() / dim;
    let mut scored: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| {
            let row = &db[i as usize * dim..(i as usize + 1) * dim];
            let mut s = 0.0f32;
            for j in 0..dim {
                s += row[j] * q[j];
            }
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn recall(pred: &[(u32, f32)], truth: &[u32]) -> f32 {
    let mut hit = 0;
    for (i, _) in pred {
        if truth.contains(i) {
            hit += 1;
        }
    }
    hit as f32 / truth.len() as f32
}

fn run<S: Scorer>(
    name: &str,
    q: &S,
    db: &[f32],
    queries: &[f32],
    codes: &[u8],
    gt: &[Vec<u32>],
) {
    let n_q = queries.len() / DIM;
    // recall + score-MSE on a sample
    let n_db = db.len() / DIM;
    let mut score_se = 0.0f64;
    let mut score_n = 0usize;
    let mut q_buf = vec![0.0f32; n_db];
    let t0 = Instant::now();
    let mut total_recall = 0.0f32;
    for i in 0..n_q {
        let qi = &queries[i * DIM..(i + 1) * DIM];
        let topk = q.topk_ip(qi, codes, TOPK);
        total_recall += recall(&topk, &gt[i]);
        if i < 30 {
            // measure score MSE only on first 30 queries for cost
            q.score_ip(qi, codes, &mut q_buf);
            for j in 0..n_db {
                let row = &db[j * DIM..(j + 1) * DIM];
                let mut t = 0.0f32;
                for d in 0..DIM {
                    t += qi[d] * row[d];
                }
                let e = q_buf[j] - t;
                score_se += (e * e) as f64;
                score_n += 1;
            }
        }
    }
    let elapsed = t0.elapsed();
    let bytes_per_vec = q.code_size();
    let n_db_codes = codes.len() / bytes_per_vec;
    let score_rmse = (score_se / score_n as f64).sqrt();
    println!(
        "{name:>14} | code={bytes_per_vec:>3} B/vec | mem={:>6.1} KiB | recall@{TOPK}={:>5.3} | score-RMSE={:.4} | {:>6.1} µs/query",
        (n_db_codes * bytes_per_vec) as f32 / 1024.0,
        total_recall / n_q as f32,
        score_rmse,
        elapsed.as_secs_f64() * 1e6 / n_q as f64,
    );
}

fn main() {
    println!(
        "ruvector-avq demo: n={N}, dim={DIM}, m={M}, k={K}, queries={N_QUERIES}, top-{TOPK}\n"
    );

    // rank=24 in dim=128 — typical effective-rank ratio for trained
    // sentence/image embeddings.
    let db = make_lowrank(N, DIM, 24, 42);
    let queries = make_lowrank(N_QUERIES, DIM, 24, 1337);

    print!("computing brute-force ground truth ... ");
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..N_QUERIES)
        .map(|i| brute_topk(&db, &queries[i * DIM..(i + 1) * DIM], DIM, TOPK))
        .collect();
    println!("done in {:.2?}", t0.elapsed());

    // Train all three variants on the same data.
    let t = Instant::now();
    let sq = ScalarQuantizer::fit(&db, DIM);
    let mut sq_codes = vec![0u8; N * sq.code_size()];
    sq.encode(&db, &mut sq_codes);
    println!("trained ScalarQuantizer    in {:.2?}", t.elapsed());

    let t = Instant::now();
    let pq = ProductQuantizer::fit(&db, DIM, M, K, 7).unwrap();
    let mut pq_codes = vec![0u8; N * pq.code_size()];
    pq.encode(&db, &mut pq_codes);
    println!("trained ProductQuantizer   in {:.2?}", t.elapsed());

    let t = Instant::now();
    let avq = AnisotropicPq::fit(&db, DIM, M, K, 16.0, 7).unwrap();
    let mut avq_codes = vec![0u8; N * avq.code_size()];
    avq.encode(&db, &mut avq_codes);
    println!("trained AnisotropicPq η=16 in {:.2?}", t.elapsed());

    let t = Instant::now();
    let avq8 = AnisotropicPq::fit(&db, DIM, M, K, 64.0, 7).unwrap();
    let mut avq8_codes = vec![0u8; N * avq8.code_size()];
    avq8.encode(&db, &mut avq8_codes);
    println!("trained AnisotropicPq η=64 in {:.2?}\n", t.elapsed());

    println!(
        "AVQ aniso-loss under η=16 (lower=better score-preservation): {:.4}",
        avq.aniso_loss(&db),
    );

    println!(
        "       variant | bytes      |   memory   |   recall   |   latency"
    );
    println!("---------------+------------+------------+------------+-----------");
    run("scalar-int8", &sq, &db, &queries, &sq_codes, &gt);
    run("uniform-PQ", &pq, &db, &queries, &pq_codes, &gt);
    run("AVQ η=16", &avq, &db, &queries, &avq_codes, &gt);
    run("AVQ η=64", &avq8, &db, &queries, &avq8_codes, &gt);
}
