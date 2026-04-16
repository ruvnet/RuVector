//! Tier 3B SIFT1M benchmark: baseline EmlHnsw vs PqEmlHnsw.
//!
//! Compares two indexes side-by-side on the same 50k base / 200 query slice:
//!
//!   1. EmlHnsw at selected_k=32 (reduced-dim float projection)
//!   2. PqEmlHnsw at 8 subspaces x 256 centroids (8-byte PQ codes)
//!
//! Both use fetch_k=200 rerank against full-dim cosine.
//!
//! Reports:
//!   - recall@10 (both without and with exact rerank)
//!   - search latency p50 / p95
//!   - memory per vector (bytes)
//!   - PqDistanceCorrector MSE before and after training on ~2000 pairs
//!
//! Gated behind RUVECTOR_EML_SIFT1M_BASE / RUVECTOR_EML_SIFT1M_QUERY env
//! vars. Prints a markdown table to stderr. Fails only on the explicit
//! rerank-recall floor.

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::hnsw_integration::{EmlHnsw, EmlMetric};
use ruvector_eml_hnsw::pq_corrector::PqDistanceCorrector;
use ruvector_eml_hnsw::pq_hnsw::PqEmlHnsw;
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

fn sq_euclidean(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[test]
fn sift1m_pq_vs_eml_hnsw() {
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

    let n: usize = std::env::var("RUVECTOR_EML_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let nq: usize = std::env::var("RUVECTOR_EML_NQ").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let selected_k: usize = std::env::var("RUVECTOR_EML_K").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let n_subspaces: usize = std::env::var("RUVECTOR_EML_PQ_M").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let n_centroids: u16 = std::env::var("RUVECTOR_EML_PQ_NC").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let kmeans_iters: usize = std::env::var("RUVECTOR_EML_PQ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let fetch_k: usize = std::env::var("RUVECTOR_EML_FETCH_K").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let ef_search: usize = std::env::var("RUVECTOR_EML_EF_SEARCH").ok().and_then(|v| v.parse().ok()).unwrap_or(128);

    eprintln!(
        "Loading SIFT1M: base={}, n={n}, queries={nq}, selected_k={selected_k}, M={n_subspaces} x {n_centroids}, iters={kmeans_iters}",
        base_env.display()
    );
    let base = read_fvecs(&base_env, n).expect("read base");
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    let dim = base[0].len();
    eprintln!("Loaded {} base x {dim} dim, {} queries", base.len(), queries.len());

    // ---------------- Baseline: EmlHnsw (reduced-dim float) -----------------
    let train_n = 2000.min(base.len());
    let train: Vec<Vec<f32>> = base.iter().take(train_n).cloned().collect();

    let t0 = Instant::now();
    let mut eml = EmlHnsw::train_and_build(&train, selected_k, EmlMetric::Cosine, base.len() + 16, 16, 200)
        .expect("build EmlHnsw");
    eml.add_batch(&base);
    let eml_build = t0.elapsed();

    // ---------------- PqEmlHnsw (PQ codes + HNSW over reconstruction) -------
    let t0 = Instant::now();
    let mut pq = PqEmlHnsw::train_and_build(&train, n_subspaces, n_centroids, kmeans_iters, base.len() + 16, 16, 200);
    pq.add_batch(&base);
    let pq_build = t0.elapsed();
    let mean_mse = pq.codebook().mean_final_mse();
    let iters_used: Vec<usize> = pq.codebook().iters_per_subspace.clone();
    eprintln!(
        "Build times: EmlHnsw {:?}, PqEmlHnsw {:?}  |  k-means mean final MSE = {:.5}, iters/subspace = {:?}",
        eml_build, pq_build, mean_mse, iters_used
    );

    // ---------------- Train the corrector on ~2000 (PQ, exact) pairs --------
    //
    // SOTA-C fix: use per-query local-scale normalization. The legacy
    // record()/correct() path is scale-saturated on SIFT (10^5 squared-dist)
    // and actually *increases* MSE. See `pq_corrector.rs`.
    //
    // Scale per query = median PQ distance across that query's probe
    // samples. This matches what `PqEmlHnsw::search_with_rerank` has
    // available at inference time.
    let mut corrector = PqDistanceCorrector::new();
    let pair_budget = 2000usize;
    let pair_queries = 40usize.min(queries.len());
    let pairs_per_query = (pair_budget / pair_queries).max(1);
    let mut pre_sq_err = 0.0f64;
    let mut pre_n = 0u64;

    for q in queries.iter().take(pair_queries) {
        let table = pq.codebook().build_query_table(q);
        let stride = (base.len() / pairs_per_query).max(1);
        // First pass: compute PQ for probe samples to get per-query median.
        let mut this_query_pq: Vec<(usize, f32, f32)> = Vec::with_capacity(pairs_per_query);
        for i in (0..base.len()).step_by(stride).take(pairs_per_query) {
            let code = pq.code_of(i + 1);
            let pq_d = pq.codebook().asymmetric_distance_with_table(&table, code);
            let exact_d = sq_euclidean(q, &base[i]);
            this_query_pq.push((i, pq_d, exact_d));
        }
        let mut sorted_pq: Vec<f32> = this_query_pq.iter().map(|x| x.1).collect();
        sorted_pq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q_scale = sorted_pq[sorted_pq.len() / 2].max(1.0);
        // Second pass: record pairs with per-query scale.
        for (_i, pq_d, exact_d) in &this_query_pq {
            corrector.record_normalized(*pq_d, *exact_d, q_scale);
            let e = (*pq_d - *exact_d) as f64;
            pre_sq_err += e * e;
            pre_n += 1;
        }
    }
    let converged = corrector.train();
    assert!(
        corrector.is_locally_normalized(),
        "corrector should be in local-scale mode after record_normalized"
    );
    let pre_mse = pre_sq_err / pre_n.max(1) as f64;

    // Post-correction MSE on a held-out set of queries. Inference scale =
    // per-query median PQ, same as training.
    let mut post_sq_err = 0.0f64;
    let mut post_n = 0u64;
    let hold_queries = pair_queries.min(queries.len() - pair_queries);
    for q in queries.iter().skip(pair_queries).take(hold_queries) {
        let table = pq.codebook().build_query_table(q);
        let stride = (base.len() / pairs_per_query).max(1);
        let mut probe: Vec<(usize, f32)> = Vec::with_capacity(pairs_per_query);
        for i in (0..base.len()).step_by(stride).take(pairs_per_query) {
            let code = pq.code_of(i + 1);
            probe.push((i, pq.codebook().asymmetric_distance_with_table(&table, code)));
        }
        let mut sorted_pq: Vec<f32> = probe.iter().map(|x| x.1).collect();
        sorted_pq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q_scale = sorted_pq[sorted_pq.len() / 2].max(1.0);
        for (i, pq_d) in &probe {
            let corrected = corrector.correct_with_scale(*pq_d, 0.0, q_scale);
            let exact_d = sq_euclidean(q, &base[*i]);
            let e = (corrected - exact_d) as f64;
            post_sq_err += e * e;
            post_n += 1;
        }
    }
    let post_mse = post_sq_err / post_n.max(1) as f64;
    eprintln!(
        "Corrector: trained on {} pairs (local-scale, per-query median PQ), \
         converged={}, pre_MSE={:.4e}, post_MSE={:.4e} (held-out)  reduction={:+.1}%",
        pre_n,
        converged,
        pre_mse,
        post_mse,
        100.0 * (pre_mse - post_mse) / pre_mse.max(1e-12)
    );
    let mse_improved = post_mse < pre_mse * 0.98;
    eprintln!(
        "Corrector promotable to non-advisory step: {} (criterion: post_MSE < 0.98 * pre_MSE). \
         In pq_hnsw.rs, the non-advisory path is enabled whenever \
         corrector.is_locally_normalized(), so this corrector IS being used \
         as a pre-rerank filter (k*15 window) in search_with_rerank.",
        mse_improved
    );

    pq.set_corrector(corrector);

    // ---------------- Measure both indexes ----------------------------------
    let mut eml_red_lat = Vec::with_capacity(nq);
    let mut eml_rr_lat = Vec::with_capacity(nq);
    let mut pq_red_lat = Vec::with_capacity(nq);
    let mut pq_rr_lat = Vec::with_capacity(nq);

    let mut eml_red_recall = 0.0f32;
    let mut eml_rr_recall = 0.0f32;
    let mut pq_red_recall = 0.0f32;
    let mut pq_rr_recall = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&base, q, 10);

        let t = Instant::now();
        let er = eml.search(q, 10, ef_search);
        eml_red_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let err = eml.search_with_rerank(q, 10, fetch_k, ef_search);
        eml_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let pr = pq.search(q, 10, ef_search);
        pq_red_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let prr = pq.search_with_rerank(q, 10, fetch_k, ef_search);
        pq_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        eml_red_recall += recall_at_k(&truth, &er.iter().map(|r| r.id).collect::<Vec<_>>(), 10);
        eml_rr_recall += recall_at_k(&truth, &err.iter().map(|r| r.id).collect::<Vec<_>>(), 10);
        pq_red_recall += recall_at_k(&truth, &pr.iter().map(|r| r.id).collect::<Vec<_>>(), 10);
        pq_rr_recall += recall_at_k(&truth, &prr.iter().map(|r| r.id).collect::<Vec<_>>(), 10);
    }

    let eml_red_recall = eml_red_recall / nq as f32;
    let eml_rr_recall = eml_rr_recall / nq as f32;
    let pq_red_recall = pq_red_recall / nq as f32;
    let pq_rr_recall = pq_rr_recall / nq as f32;

    let eml_red_p50 = percentile(&mut eml_red_lat.clone(), 0.5);
    let eml_red_p95 = percentile(&mut eml_red_lat.clone(), 0.95);
    let eml_rr_p50 = percentile(&mut eml_rr_lat.clone(), 0.5);
    let eml_rr_p95 = percentile(&mut eml_rr_lat.clone(), 0.95);
    let pq_red_p50 = percentile(&mut pq_red_lat.clone(), 0.5);
    let pq_red_p95 = percentile(&mut pq_red_lat.clone(), 0.95);
    let pq_rr_p50 = percentile(&mut pq_rr_lat.clone(), 0.5);
    let pq_rr_p95 = percentile(&mut pq_rr_lat.clone(), 0.95);

    // Memory accounting (bytes per vector as payload, not counting HNSW edges).
    let eml_payload_bytes = dim * std::mem::size_of::<f32>();
    let pq_code_bytes = pq.code_bytes_per_vec();
    let pq_hnsw_graph_bytes = pq.hnsw_stored_bytes_per_vec();

    eprintln!("============================================================");
    eprintln!("Tier 3B: PQ + learned corrector vs EmlHnsw on SIFT1M subset");
    eprintln!(
        "  base_n={} queries={} dim={} selected_k={} M={}x{} fetch_k={} ef_search={}",
        base.len(),
        nq,
        dim,
        selected_k,
        n_subspaces,
        n_centroids,
        fetch_k,
        ef_search
    );
    eprintln!("  k-means iters per subspace: {:?}  mean_final_MSE={:.5}", iters_used, mean_mse);
    eprintln!(
        "  Corrector: pre_MSE={:.4}  post_MSE={:.4}  delta={:+.4}",
        pre_mse,
        post_mse,
        pre_mse - post_mse
    );
    eprintln!("");
    eprintln!("| index      | recall@10 | rerank@10 | p50 red (us) | p95 red (us) | p50 rr (us) | p95 rr (us) | bytes/vec (payload) |");
    eprintln!("|------------|-----------|-----------|--------------|--------------|-------------|-------------|---------------------|");
    eprintln!(
        "| EmlHnsw    |   {:.4}  |   {:.4}  |   {:>8.1}   |   {:>8.1}   |  {:>8.1}   |  {:>8.1}   | {:>6}              |",
        eml_red_recall, eml_rr_recall, eml_red_p50, eml_red_p95, eml_rr_p50, eml_rr_p95, eml_payload_bytes
    );
    eprintln!(
        "| PqEmlHnsw  |   {:.4}  |   {:.4}  |   {:>8.1}   |   {:>8.1}   |  {:>8.1}   |  {:>8.1}   | {:>6}              |",
        pq_red_recall, pq_rr_recall, pq_red_p50, pq_red_p95, pq_rr_p50, pq_rr_p95, pq_code_bytes
    );
    eprintln!(
        "  PqEmlHnsw HNSW graph holds reconstructed floats = {} bytes/vec (transient; deployed system keeps only codes)",
        pq_hnsw_graph_bytes
    );
    eprintln!(
        "  Memory reduction (payload): {:.1}x",
        eml_payload_bytes as f64 / pq_code_bytes as f64
    );
    eprintln!("============================================================");

    // Soft regression gates. PQ rerank should not collapse recall below 0.80
    // (the tier acceptance criterion). If it does, the failure message
    // explicitly asks for more codebooks / finer partitioning.
    assert!(
        pq_rr_recall >= 0.80,
        "PqEmlHnsw rerank recall@10 {:.3} below 0.80 floor — tier 3B target not met, \
         try more codebooks (increase n_centroids) or finer subspace partitioning (increase n_subspaces)",
        pq_rr_recall
    );
    // Baseline comparison is informational only.
    assert!(
        eml_rr_recall >= 0.60,
        "EmlHnsw rerank recall@10 {:.3} well below expected — environment issue?",
        eml_rr_recall
    );
}
