//! Native HNSW benchmark on a raw f32 dump (SIFT-style): build time, then recall@10 and single-thread QPS per ef.
//!   sift_bench BASE.f32 QUERY.f32 GT.i32 DIM [M] [EF_CONSTRUCTION]
//!
//! The three inputs are raw little-endian dumps, e.g. from numpy: `X.astype('float32').tofile('base.f32')`,
//! the same for the queries, and `GT.astype('int32').tofile('gt.i32')` where `GT` holds the exact top-10 base ids per query
//! (for example from `faiss.IndexFlatL2`). Run pinned to one core for stable numbers: `taskset -c 3 ...`.
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};
use std::time::Instant;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn read_i32(p: &str) -> Vec<i32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dim: usize = a[4].parse().unwrap();
    let m: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(16);
    let efc: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(200);
    let base = read_f32(&a[1]);
    let q = read_f32(&a[2]);
    let gt = read_i32(&a[3]);
    let n = base.len() / dim;
    let nq = q.len() / dim;
    let cfg = HnswConfig {
        m,
        ef_construction: efc,
        ef_search: 50,
        max_elements: n + 10,
    };
    let mut idx = HnswIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    let t = Instant::now();
    for i in 0..n {
        idx.add(i.to_string(), base[i * dim..(i + 1) * dim].to_vec())
            .unwrap();
    }
    println!("build_s {:.2}", t.elapsed().as_secs_f64());
    for ef in [10usize, 20, 40, 80, 160, 320] {
        let mut best = f64::MAX;
        let mut hits = 0usize;
        for pass in 0..3 {
            let t = Instant::now();
            let mut cnt = 0usize;
            for qi in 0..nq {
                let r = idx
                    .search_with_ef(&q[qi * dim..(qi + 1) * dim], 10, ef)
                    .unwrap();
                if pass == 0 {
                    for h in &r {
                        let id: i32 = h.id.parse().unwrap();
                        if gt[qi * 10..qi * 10 + 10].contains(&id) {
                            cnt += 1;
                        }
                    }
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
            if pass == 0 {
                hits = cnt;
            }
        }
        println!(
            "ef {ef:4} recall@10 {:.4} qps {:.0}",
            hits as f64 / (nq * 10) as f64,
            nq as f64 / best
        );
    }
}
