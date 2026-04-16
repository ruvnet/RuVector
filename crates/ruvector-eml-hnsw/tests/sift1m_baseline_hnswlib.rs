//! Plain `hnsw_rs` baseline on SIFT1M (no EML wrapper).
//!
//! Makes the SOTA claim honest by giving EmlHnsw a same-configuration
//! reference point:
//!
//!   * `sift1m_plain_hnsw_baseline_l2_sweep`    — DistL2 vs SIFT's published
//!     L2 ground truth. Standard community benchmark.
//!   * `sift1m_plain_hnsw_baseline_cosine_sweep`— DistCosine vs brute-force
//!     cosine GT computed on the loaded base. This is the apples-to-apples
//!     reference for EmlHnsw in `sift1m_full_1m.rs`.
//!
//! Config matches the published SIFT1M HNSW: m=16, ef_construction=200,
//! ef_search ∈ {32, 64, 100, 200}.

use hnsw_rs::prelude::{DistCosine, DistL2, Hnsw};
use rayon::prelude::*;
use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

fn read_fvecs(path: &PathBuf, limit: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut out = Vec::new();
    loop {
        let mut dbuf = [0u8; 4];
        if r.read_exact(&mut dbuf).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dbuf) as usize;
        let mut vec = vec![0f32; dim];
        let mut bytes = vec![0u8; dim * 4];
        r.read_exact(&mut bytes)?;
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            vec[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        out.push(vec);
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

fn read_ivecs(path: &PathBuf, limit: usize) -> std::io::Result<Vec<Vec<i32>>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut out = Vec::new();
    loop {
        let mut dbuf = [0u8; 4];
        if r.read_exact(&mut dbuf).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dbuf) as usize;
        let mut v = vec![0i32; dim];
        let mut bytes = vec![0u8; dim * 4];
        r.read_exact(&mut bytes)?;
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            v[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        out.push(v);
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

fn brute_force_cosine_top_k(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<i32>> {
    queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance_f32(q, v)))
                .collect();
            let pivot = k.min(scored.len() - 1);
            scored.select_nth_unstable_by(pivot, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter().map(|(i, _)| i as i32).collect()
        })
        .collect()
}

fn recall_at_k_ivecs(truth: &[i32], got_1based: &[usize], k: usize) -> f32 {
    // ivecs GT is 0-based; our engine ids are 1-based.
    let tset: std::collections::HashSet<i32> = truth.iter().take(k).copied().collect();
    got_1based
        .iter()
        .take(k)
        .filter(|&&id| tset.contains(&((id as i32) - 1)))
        .count() as f32
        / k as f32
}

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

#[test]
fn sift1m_plain_hnsw_baseline_l2_sweep() {
    let base_path = match std::env::var("RUVECTOR_EML_SIFT1M_BASE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_BASE to enable");
            return;
        }
    };
    let query_path = PathBuf::from(
        std::env::var("RUVECTOR_EML_SIFT1M_QUERY").expect("set RUVECTOR_EML_SIFT1M_QUERY"),
    );
    let gt_path = PathBuf::from(
        std::env::var("RUVECTOR_EML_SIFT1M_GT").expect("set RUVECTOR_EML_SIFT1M_GT"),
    );

    let n: usize = std::env::var("RUVECTOR_EML_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let nq: usize = std::env::var("RUVECTOR_EML_NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);

    eprintln!("baseline-L2: loading SIFT1M n={n} nq={nq}");
    let t = Instant::now();
    let base = read_fvecs(&base_path, n).expect("read base");
    eprintln!("  base: {} × {} in {:?}", base.len(), base[0].len(), t.elapsed());
    let t = Instant::now();
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    eprintln!("  queries: {} × {} in {:?}", queries.len(), queries[0].len(), t.elapsed());
    let t = Instant::now();
    let gt = read_ivecs(&gt_path, nq).expect("read gt");
    eprintln!("  groundtruth: {} × {} in {:?}", gt.len(), gt[0].len(), t.elapsed());

    let efs: Vec<usize> = vec![32, 64, 100, 200];

    eprintln!("\n----- plain hnsw_rs DistL2 (SIFT published metric) -----");
    let hnsw_l2: Hnsw<f32, DistL2> = Hnsw::<f32, DistL2>::new(16, base.len() + 16, 16, 200, DistL2);
    let data_l2: Vec<(&Vec<f32>, usize)> =
        base.iter().enumerate().map(|(i, v)| (v, i + 1)).collect();
    let t = Instant::now();
    hnsw_l2.parallel_insert(&data_l2);
    eprintln!(
        "  L2 parallel build: {:?} ({:.0} vec/s)",
        t.elapsed(),
        base.len() as f64 / t.elapsed().as_secs_f64()
    );

    eprintln!("\n=== Plain hnsw_rs DistL2 sweep (vs published GT) ===");
    eprintln!("  ef_search  recall@10     QPS        p50(µs)   p95(µs)");
    for &ef in &efs {
        for q in queries.iter().take(10) {
            let _ = hnsw_l2.search(q, 10, ef);
        }
        let mut lat = Vec::with_capacity(queries.len());
        let mut rec = 0.0f32;
        let t_total = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let hits = hnsw_l2.search(q, 10, ef);
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            let ids: Vec<usize> = hits.into_iter().map(|n| n.d_id).collect();
            rec += recall_at_k_ivecs(&gt[qi], &ids, 10);
        }
        let total = t_total.elapsed();
        let rec = rec / queries.len() as f32;
        let qps = queries.len() as f64 / total.as_secs_f64();
        let p50 = percentile(&mut lat.clone(), 0.5);
        let p95 = percentile(&mut lat.clone(), 0.95);
        eprintln!(
            "   {:>8}  {:>9.4}  {:>9.1}  {:>8.1}  {:>8.1}",
            ef, rec, qps, p50, p95
        );
    }
}

#[test]
fn sift1m_plain_hnsw_baseline_cosine_sweep() {
    let base_path = match std::env::var("RUVECTOR_EML_SIFT1M_BASE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_BASE to enable");
            return;
        }
    };
    let query_path = PathBuf::from(
        std::env::var("RUVECTOR_EML_SIFT1M_QUERY").expect("set RUVECTOR_EML_SIFT1M_QUERY"),
    );

    let n: usize = std::env::var("RUVECTOR_EML_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let nq: usize = std::env::var("RUVECTOR_EML_NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);

    eprintln!("baseline-cosine: loading SIFT1M n={n} nq={nq}");
    let base = read_fvecs(&base_path, n).expect("read base");
    let queries = read_fvecs(&query_path, nq).expect("read queries");

    eprintln!("  computing brute-force cosine GT (parallel)...");
    let t = Instant::now();
    let gt = brute_force_cosine_top_k(&base, &queries, 10);
    eprintln!("  GT: {} × 10 in {:?}", gt.len(), t.elapsed());

    let efs: Vec<usize> = vec![32, 64, 100, 200];

    eprintln!("\n----- plain hnsw_rs DistCosine (same metric as EmlHnsw) -----");
    let hnsw_cos: Hnsw<f32, DistCosine> =
        Hnsw::<f32, DistCosine>::new(16, base.len() + 16, 16, 200, DistCosine);
    let data_cos: Vec<(&Vec<f32>, usize)> =
        base.iter().enumerate().map(|(i, v)| (v, i + 1)).collect();
    let t = Instant::now();
    hnsw_cos.parallel_insert(&data_cos);
    eprintln!(
        "  Cosine parallel build: {:?} ({:.0} vec/s)",
        t.elapsed(),
        base.len() as f64 / t.elapsed().as_secs_f64()
    );

    eprintln!("\n=== Plain hnsw_rs DistCosine sweep (vs brute-force cosine GT) ===");
    eprintln!("  ef_search  recall@10     QPS        p50(µs)   p95(µs)");
    for &ef in &efs {
        for q in queries.iter().take(10) {
            let _ = hnsw_cos.search(q, 10, ef);
        }
        let mut lat = Vec::with_capacity(queries.len());
        let mut rec = 0.0f32;
        let t_total = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let hits = hnsw_cos.search(q, 10, ef);
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            let ids: Vec<usize> = hits.into_iter().map(|n| n.d_id).collect();
            rec += recall_at_k_ivecs(&gt[qi], &ids, 10);
        }
        let total = t_total.elapsed();
        let rec = rec / queries.len() as f32;
        let qps = queries.len() as f64 / total.as_secs_f64();
        let p50 = percentile(&mut lat.clone(), 0.5);
        let p95 = percentile(&mut lat.clone(), 0.95);
        eprintln!(
            "   {:>8}  {:>9.4}  {:>9.1}  {:>8.1}  {:>8.1}",
            ef, rec, qps, p50, p95
        );
    }

    eprintln!("\n(SOTA-gap: compare QPS@recall≥0.95 above against EmlHnsw in sift1m_full_1m.)");
}
