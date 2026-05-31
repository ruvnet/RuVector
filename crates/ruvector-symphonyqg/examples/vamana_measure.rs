//! Measure the Vamana α-pruning recall impact at the operating point
//! where it's expected to matter most: n=50K, dim=128, ef=100.
//!
//! Run with:
//!   cargo run -p ruvector-symphonyqg --release --example vamana_measure
//!
//! Expected (per ADR-193 §Consequences and the gist's iter-5 entry):
//!   without Vamana → recall@10 ≈ 31%
//!   with Vamana    → recall@10 ≥ 50%
//! Build cost roughly doubles when refinement is on (one extra
//! beam search + α-prune pass per vertex).

use ruvector_symphonyqg::vamana::VamanaConfig;
use ruvector_symphonyqg::*;
use std::time::Instant;

fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = (seed as u32) | 1;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s as f32 / u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn ground_truth(vecs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(f32, usize)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
            (dist, i)
        })
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.iter().take(k).map(|&(_, i)| i).collect()
}

fn main() {
    let n = 50_000;
    let dim = 128;
    let k = 10;
    let queries = 50;
    let ef = 100;

    println!(
        "ruvector-symphonyqg vamana measurement: n={} dim={} k={} ef={} queries={}",
        n, dim, k, ef, queries
    );
    println!("Building corpus ...");
    let vecs = gaussian_vecs(n, dim, 42);

    // Generate OUT-OF-CORPUS queries (different seed from corpus). Self-queries
    // (where `query ∈ vecs`) trivially benefit from any improved local
    // connectivity; out-of-corpus queries are the production-relevant test.
    let probe_vecs = gaussian_vecs(queries, dim, 99); // separate seed → not in `vecs`

    println!(
        "{:>16} | {:>9} | {:>11} | {:>11} | {:>9}",
        "config", "build_ms", "recall@10", "self-recall", "search_ms"
    );
    println!("{}", "-".repeat(72));

    for vamana in [None, Some(VamanaConfig::default())] {
        let label = if vamana.is_some() {
            "WITH Vamana"
        } else {
            "no Vamana"
        };
        let cfg = Config {
            dim,
            m_base: 16,
            ef_construction: 200,
            vamana: vamana.clone(),
            ..Config::default()
        };

        let t0 = Instant::now();
        let (_, _, symphony) = build_all(&vecs, &cfg);
        let build_ms = t0.elapsed().as_millis();

        // Out-of-corpus recall (production-relevant)
        let mut hit_oop = 0;
        let t0 = Instant::now();
        for q in 0..queries {
            let res = symphony.search(&probe_vecs[q], k, ef);
            let gt = ground_truth(&vecs, &probe_vecs[q], k);
            let res_set: std::collections::HashSet<usize> = res.iter().map(|r| r.idx).collect();
            hit_oop += gt.iter().filter(|&&g| res_set.contains(&g)).count();
        }
        let search_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let recall_oop = hit_oop as f64 / (queries * k) as f64;

        // Self-recall (queries IN corpus — easier; included to show
        // why earlier Vamana measurements looked optimistic).
        let mut hit_self = 0;
        for q in 0..queries {
            let query = &vecs[(q * 7 + 3) % n];
            let res = symphony.search(query, k, ef);
            let gt = ground_truth(&vecs, query, k);
            let res_set: std::collections::HashSet<usize> = res.iter().map(|r| r.idx).collect();
            hit_self += gt.iter().filter(|&&g| res_set.contains(&g)).count();
        }
        let recall_self = hit_self as f64 / (queries * k) as f64;

        println!(
            "{:>16} | {:>9} | {:>11.3} | {:>11.3} | {:>9.0}",
            label, build_ms, recall_oop, recall_self, search_ms
        );
    }
    println!();
    println!("recall@10  = OUT-OF-CORPUS queries (production-relevant test)");
    println!("self-recall = queries sampled FROM corpus (easier — not the right benchmark)");
}
