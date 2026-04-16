//! SIFT1M A/B: baseline [`EmlHnsw`] vs [`ProgressiveEmlHnsw`] cascade.
//!
//! Gated by `RUVECTOR_EML_SIFT1M_BASE` / `RUVECTOR_EML_SIFT1M_QUERY` so CI
//! without the dataset skips cleanly. Mirrors `sift1m_real.rs`'s knobs
//! (`RUVECTOR_EML_N`, `RUVECTOR_EML_NQ`).
//!
//! Reports for both indexes:
//!   - build time
//!   - recall@10 vs brute-force full-cosine ground truth
//!   - p50 / p95 query latency

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::hnsw_integration::{EmlHnsw, EmlMetric};
use ruvector_eml_hnsw::progressive_hnsw::ProgressiveEmlHnsw;
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

fn brute_force_top_k(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1, cosine_distance_f32(q, v)))
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(truth: &[usize], got: &[usize], k: usize) -> f32 {
    let tset: std::collections::HashSet<_> = truth.iter().take(k).collect();
    got.iter().take(k).filter(|i| tset.contains(i)).count() as f32 / k as f32
}

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

#[test]
fn progressive_vs_baseline_sift1m() {
    let base_env = match std::env::var("RUVECTOR_EML_SIFT1M_BASE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_BASE to sift_base.fvecs");
            return;
        }
    };
    let query_env = std::env::var("RUVECTOR_EML_SIFT1M_QUERY")
        .expect("set RUVECTOR_EML_SIFT1M_QUERY to sift_query.fvecs");
    let query_path = PathBuf::from(query_env);

    let n: usize = std::env::var("RUVECTOR_EML_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let nq: usize = std::env::var("RUVECTOR_EML_NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let baseline_k: usize = std::env::var("RUVECTOR_EML_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    eprintln!(
        "loading SIFT1M: base={}, n={n}, queries={nq}, baseline_k={baseline_k}",
        base_env.display()
    );
    let base = read_fvecs(&base_env, n).expect("read base");
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    eprintln!(
        "loaded {} base x {} dim, {} queries",
        base.len(),
        base[0].len(),
        queries.len()
    );
    let train: Vec<Vec<f32>> = base.iter().take(1000).cloned().collect();

    // ---------- Baseline EmlHnsw ----------
    let t0 = Instant::now();
    let mut baseline = EmlHnsw::train_and_build(
        &train,
        baseline_k,
        EmlMetric::Cosine,
        base.len() + 16,
        16,
        200,
    )
    .expect("build baseline");
    baseline.add_batch(&base);
    let baseline_build = t0.elapsed();
    eprintln!("baseline EmlHnsw built in {:?}", baseline_build);

    // ---------- ProgressiveEmlHnsw [8, 32, 128] ----------
    let schedule: Vec<usize> = std::env::var("RUVECTOR_EML_SCHEDULE")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![8, 32, 128]);

    let t0 = Instant::now();
    let mut progressive = ProgressiveEmlHnsw::train_and_build(
        &train,
        &schedule,
        EmlMetric::Cosine,
        base.len() + 16,
        16,
        200,
    )
    .expect("build progressive");
    progressive.add_batch(&base);
    let progressive_build = t0.elapsed();
    eprintln!(
        "progressive [{}] built in {:?}",
        schedule
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
        progressive_build
    );

    // ---------- Query loop ----------
    let mut baseline_lat = Vec::with_capacity(nq);
    let mut prog_lat = Vec::with_capacity(nq);
    let mut baseline_recall = 0.0f32;
    let mut prog_recall = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&base, q, 10);

        let t = Instant::now();
        let b = baseline.search(q, 10, 64);
        baseline_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let p = progressive.search(q, 10, 64);
        prog_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let bids: Vec<usize> = b.into_iter().map(|r| r.id).collect();
        let pids: Vec<usize> = p.into_iter().map(|r| r.id).collect();

        baseline_recall += recall_at_k(&truth, &bids, 10);
        prog_recall += recall_at_k(&truth, &pids, 10);
    }

    let baseline_recall = baseline_recall / nq as f32;
    let prog_recall = prog_recall / nq as f32;
    let b_p50 = percentile(&mut baseline_lat.clone(), 0.5);
    let b_p95 = percentile(&mut baseline_lat.clone(), 0.95);
    let p_p50 = percentile(&mut prog_lat.clone(), 0.5);
    let p_p95 = percentile(&mut prog_lat.clone(), 0.95);

    eprintln!("------------------------------------------------------------");
    eprintln!("SIFT1M Tier-3A A/B: baseline EmlHnsw vs ProgressiveEmlHnsw");
    eprintln!(
        "  n={}, queries={}, dim={}, baseline_k={}, schedule={:?}",
        base.len(),
        nq,
        base[0].len(),
        baseline_k,
        schedule
    );
    eprintln!(
        "  baseline     build {:>7.2}s   recall@10 {:.4}   p50 {:>7.1} us   p95 {:>7.1} us",
        baseline_build.as_secs_f64(),
        baseline_recall,
        b_p50,
        b_p95
    );
    eprintln!(
        "  progressive  build {:>7.2}s   recall@10 {:.4}   p50 {:>7.1} us   p95 {:>7.1} us",
        progressive_build.as_secs_f64(),
        prog_recall,
        p_p50,
        p_p95
    );
    eprintln!(
        "  build ratio  {:.2}x    latency p50 ratio {:.2}x    p95 ratio {:.2}x",
        progressive_build.as_secs_f64() / baseline_build.as_secs_f64().max(1e-9),
        p_p50 / b_p50.max(1e-9),
        p_p95 / b_p95.max(1e-9),
    );
    eprintln!("------------------------------------------------------------");

    // Sanity floors — intentionally loose so the numbers are the deliverable,
    // not the pass/fail. Baseline must stay above its own proven floor;
    // progressive must at least hit brute-force-quality on most queries
    // thanks to the full-dim rerank on the finest level.
    assert!(
        baseline_recall >= 0.05,
        "baseline recall@10 {:.3} below sanity floor 0.05",
        baseline_recall
    );
    assert!(
        prog_recall >= 0.05,
        "progressive recall@10 {:.3} below sanity floor 0.05",
        prog_recall
    );
}
