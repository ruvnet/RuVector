//! Real-data Stage 3 benchmark: SIFT1M subset.
//!
//! Gated behind the `RUVECTOR_EML_SIFT1M_PATH` env var so it only runs when a
//! dataset is available. The SIFT1M `.fvecs` format is 32-bit little-endian:
//! each record is `<i32 dim><f32 * dim>`.
//!
//! Measures:
//!   - recall@10 of the reduced-dim EmlHnsw vs brute-force full-cosine
//!   - recall@10 after exact re-rank
//!   - p50 / p95 query latency of the reduced index
//!
//! Prints results to stderr. Fails only if recall drops below a conservative
//! bar so regressions surface in CI.

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
    got.iter()
        .take(k)
        .filter(|i| tset.contains(i))
        .count() as f32
        / k as f32
}

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

#[test]
fn sift1m_reduced_hnsw_recall() {
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
    let selected_k: usize = std::env::var("RUVECTOR_EML_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    eprintln!("loading SIFT1M: base={}, n={n}, queries={nq}, k={selected_k}", base_env.display());
    let base = read_fvecs(&base_env, n).expect("read base");
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    eprintln!("loaded {} base × {} dim, {} queries", base.len(), base[0].len(), queries.len());

    let train: Vec<Vec<f32>> = base.iter().take(1000).cloned().collect();
    let mut idx = EmlHnsw::train_and_build(
        &train,
        selected_k,
        EmlMetric::Cosine,
        base.len() + 16,
        16,
        200,
    )
    .expect("build");

    let t0 = Instant::now();
    idx.add_batch(&base);
    eprintln!("built index in {:?}", t0.elapsed());

    let mut reduced_lat = Vec::with_capacity(nq);
    let mut rerank_lat = Vec::with_capacity(nq);
    let mut reduced_recall = 0.0f32;
    let mut rerank_recall = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&base, q, 10);

        let t = Instant::now();
        let red = idx.search(q, 10, 64);
        reduced_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let rr = idx.search_with_rerank(q, 10, 50, 64);
        rerank_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let rids: Vec<usize> = red.into_iter().map(|r| r.id).collect();
        let rrids: Vec<usize> = rr.into_iter().map(|r| r.id).collect();

        reduced_recall += recall_at_k(&truth, &rids, 10);
        rerank_recall += recall_at_k(&truth, &rrids, 10);
    }

    let reduced_recall = reduced_recall / nq as f32;
    let rerank_recall = rerank_recall / nq as f32;
    let red_p50 = percentile(&mut reduced_lat.clone(), 0.5);
    let red_p95 = percentile(&mut reduced_lat.clone(), 0.95);
    let rr_p50 = percentile(&mut rerank_lat.clone(), 0.5);
    let rr_p95 = percentile(&mut rerank_lat.clone(), 0.95);

    eprintln!("------------------------------------------------------------");
    eprintln!("SIFT1M Stage-3 real-data summary");
    eprintln!("  base n={}, queries={}, dim={}, selected_k={}", base.len(), nq, base[0].len(), selected_k);
    eprintln!("  recall@10 reduced  = {reduced_recall:.4}");
    eprintln!("  recall@10 +rerank  = {rerank_recall:.4}");
    eprintln!("  latency reduced    p50 {:.1} µs  p95 {:.1} µs", red_p50, red_p95);
    eprintln!("  latency +rerank    p50 {:.1} µs  p95 {:.1} µs", rr_p50, rr_p95);
    eprintln!("------------------------------------------------------------");

    // Conservative regression bars. Real SIFT1M has strong PCA structure, so
    // reduced-dim cosine must preserve at least ~60% recall@10 and re-rank
    // must recover ≥85%. Tight bars would be brittle; these are floors.
    assert!(
        reduced_recall >= 0.55,
        "recall@10 reduced {reduced_recall:.3} below floor 0.55"
    );
    assert!(
        rerank_recall >= 0.80,
        "recall@10 +rerank {rerank_recall:.3} below floor 0.80"
    );
}
