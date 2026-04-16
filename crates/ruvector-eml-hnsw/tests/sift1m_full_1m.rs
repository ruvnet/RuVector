//! SIFT1M full-corpus (1M vectors) SOTA benchmark for EmlHnsw.
//!
//! This test validates EmlHnsw (cosine metric) against an honest,
//! same-metric ground truth: brute-force cosine on the exact corpus we load.
//! Mixing SIFT's published L2 ground truth with a cosine engine makes recall
//! numbers meaningless — the rankings of L2 and cosine differ on
//! non-normalized SIFT vectors — so we compute our own GT here (parallel,
//! ~10-30s on 32 cores at 1M × 1k queries).
//!
//! Gates
//! -----
//! - `RUVECTOR_EML_SIFT1M_BASE`   → path to `sift_base.fvecs`       (1M × 128)
//! - `RUVECTOR_EML_SIFT1M_QUERY`  → path to `sift_query.fvecs`      (10k × 128)
//! - `RUVECTOR_EML_SIFT1M_GT`     → path to `sift_groundtruth.ivecs` (unused for recall, kept for compat)
//! - `RUVECTOR_EML_N`             → base size to index  (default 1_000_000)
//! - `RUVECTOR_EML_NQ`            → query count         (default 1_000)
//! - `RUVECTOR_EML_K`             → selected_k          (default 48)
//! - `RUVECTOR_EML_FETCH_K`       → rerank fetch        (default 500)
//!
//! Outputs
//! -------
//! - parallel vs serial rerank QPS+latency at fetch_k=500
//! - ef_search sweep: recall@10, QPS, p50, p95
//! - best QPS at recall ≥ 0.95

use rayon::prelude::*;
use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::hnsw_integration::{EmlHnsw, EmlMetric};
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

/// Parallel brute-force top-k cosine GT. Returns 0-based ids (matching
/// 0-based ivecs convention; EmlHnsw ids are 1-based and we subtract 1 in
/// recall_at_k).
fn brute_force_cosine_top_k(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<i32>> {
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

fn recall_at_k(truth_0based: &[i32], got_1based: &[usize], k: usize) -> f32 {
    let tset: std::collections::HashSet<i32> = truth_0based.iter().take(k).copied().collect();
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
fn sift1m_full_1m_parallel_rerank_and_sweep() {
    let base_path = match std::env::var("RUVECTOR_EML_SIFT1M_BASE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_BASE / _QUERY to enable");
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
    let selected_k: usize = std::env::var("RUVECTOR_EML_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let fetch_k: usize = std::env::var("RUVECTOR_EML_FETCH_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    eprintln!("loading SIFT1M n={n} nq={nq} selected_k={selected_k} fetch_k={fetch_k}");
    let t = Instant::now();
    let base = read_fvecs(&base_path, n).expect("read base");
    eprintln!("  base: {} × {} in {:?}", base.len(), base[0].len(), t.elapsed());
    let t = Instant::now();
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    eprintln!("  queries: {} × {} in {:?}", queries.len(), queries[0].len(), t.elapsed());

    // Compute honest same-metric cosine GT in parallel.
    eprintln!("  computing brute-force cosine GT (parallel)...");
    let t = Instant::now();
    let gt = brute_force_cosine_top_k(&base, &queries, 10);
    eprintln!("  GT: {} × 10 in {:?}", gt.len(), t.elapsed());

    // Train EML selector on a 5k slice.
    let train: Vec<Vec<f32>> = base.iter().take(5000).cloned().collect();
    let t = Instant::now();
    let mut idx = EmlHnsw::train_and_build(
        &train,
        selected_k,
        EmlMetric::Cosine,
        base.len() + 16,
        16,
        200,
    )
    .expect("build");
    eprintln!("  selector trained in {:?}", t.elapsed());

    let t = Instant::now();
    idx.add_batch_parallel(&base);
    let build_elapsed = t.elapsed();
    eprintln!(
        "  EmlHnsw parallel build: {:?} ({:.0} vec/s)",
        build_elapsed,
        base.len() as f64 / build_elapsed.as_secs_f64()
    );

    // Phase 1: parallel vs serial rerank at fetch_k=500.
    eprintln!("\n=== Phase 1: parallel vs serial rerank (ef=100, fetch_k={fetch_k}) ===");
    let ef_phase1: usize = 100;
    for q in queries.iter().take(20) {
        let _ = idx.search_with_rerank(q, 10, fetch_k, ef_phase1);
        let _ = idx.search_with_rerank_serial(q, 10, fetch_k, ef_phase1);
    }

    let mut par_lat = Vec::with_capacity(queries.len());
    let mut par_recall = 0.0f32;
    let t_total = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let hits = idx.search_with_rerank(q, 10, fetch_k, ef_phase1);
        par_lat.push(t.elapsed().as_secs_f64() * 1e6);
        let ids: Vec<usize> = hits.into_iter().map(|r| r.id).collect();
        par_recall += recall_at_k(&gt[qi], &ids, 10);
    }
    let par_total = t_total.elapsed();
    let par_recall = par_recall / queries.len() as f32;
    let par_qps = queries.len() as f64 / par_total.as_secs_f64();
    let par_p50 = percentile(&mut par_lat.clone(), 0.5);
    let par_p95 = percentile(&mut par_lat.clone(), 0.95);

    let mut ser_lat = Vec::with_capacity(queries.len());
    let mut ser_recall = 0.0f32;
    let t_total = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let hits = idx.search_with_rerank_serial(q, 10, fetch_k, ef_phase1);
        ser_lat.push(t.elapsed().as_secs_f64() * 1e6);
        let ids: Vec<usize> = hits.into_iter().map(|r| r.id).collect();
        ser_recall += recall_at_k(&gt[qi], &ids, 10);
    }
    let ser_total = t_total.elapsed();
    let ser_recall = ser_recall / queries.len() as f32;
    let ser_qps = queries.len() as f64 / ser_total.as_secs_f64();
    let ser_p50 = percentile(&mut ser_lat.clone(), 0.5);
    let ser_p95 = percentile(&mut ser_lat.clone(), 0.95);

    eprintln!("  parallel rerank : recall@10={par_recall:.4} QPS={par_qps:.1}  p50={par_p50:.1}µs p95={par_p95:.1}µs");
    eprintln!("  serial   rerank : recall@10={ser_recall:.4} QPS={ser_qps:.1}  p50={ser_p50:.1}µs p95={ser_p95:.1}µs");
    eprintln!("  parallel speedup: {:.2}×", par_qps / ser_qps);

    // Phase 2: ef_search sweep.
    eprintln!("\n=== Phase 2: ef_search sweep (parallel rerank, fetch_k={fetch_k}) ===");
    eprintln!("  ef_search  recall@10     QPS        p50(µs)   p95(µs)");
    let efs: Vec<usize> = vec![32, 64, 100, 200];
    let mut sweep: Vec<(usize, f32, f64, f64, f64)> = Vec::new();
    for &ef in &efs {
        for q in queries.iter().take(10) {
            let _ = idx.search_with_rerank(q, 10, fetch_k, ef);
        }
        let mut lat = Vec::with_capacity(queries.len());
        let mut rec = 0.0f32;
        let t_total = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let hits = idx.search_with_rerank(q, 10, fetch_k, ef);
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            let ids: Vec<usize> = hits.into_iter().map(|r| r.id).collect();
            rec += recall_at_k(&gt[qi], &ids, 10);
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
        sweep.push((ef, rec, qps, p50, p95));
    }

    let best = sweep
        .iter()
        .filter(|(_, r, _, _, _)| *r >= 0.95)
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .cloned();
    eprintln!("\n=== SOTA-B: EmlHnsw @ recall ≥ 0.95 ===");
    match best {
        Some((ef, r, qps, _, _)) => {
            eprintln!("  ef={ef}  recall={r:.4}  QPS={qps:.1}");
        }
        None => {
            let top = sweep
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            eprintln!(
                "  NO ef reached recall≥0.95. Best recall: ef={} recall={:.4} QPS={:.1}",
                top.0, top.1, top.2
            );
        }
    }

    // Regression floors.
    assert!(
        par_recall >= 0.60,
        "parallel rerank recall@10 {par_recall:.3} below floor 0.60 — pipeline broke"
    );
    assert!(
        par_qps >= ser_qps * 0.5,
        "parallel rerank QPS {par_qps:.1} regresses vs serial {ser_qps:.1} by more than 2×"
    );
}
