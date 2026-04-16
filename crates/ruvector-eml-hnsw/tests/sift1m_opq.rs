//! SIFT1M benchmark: OPQ-trained codebooks vs plain PQ at matched memory.
//!
//! Both use M=8 x 256 codebooks over the PQ-native HNSW, i.e. same 8-byte
//! codes and same graph topology. The only difference is whether an OPQ
//! orthonormal rotation is applied before training the codebook (and before
//! every encode/query at runtime). Because the rotation is orthonormal,
//! squared-L2 distances are preserved, so HNSW is none the wiser.
//!
//! Gated behind RUVECTOR_EML_SIFT1M_BASE / RUVECTOR_EML_SIFT1M_QUERY.

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
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

#[test]
fn sift1m_opq_vs_pq() {
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
    let n_subspaces: usize = std::env::var("RUVECTOR_EML_PQ_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let n_centroids: u16 = std::env::var("RUVECTOR_EML_PQ_NC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let kmeans_iters: usize = std::env::var("RUVECTOR_EML_PQ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let fetch_k: usize = std::env::var("RUVECTOR_EML_FETCH_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let ef_search: usize = std::env::var("RUVECTOR_EML_EF_SEARCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    eprintln!(
        "Loading SIFT1M: base={}, n={n}, queries={nq}, M={n_subspaces} x {n_centroids}",
        base_env.display()
    );
    let base = read_fvecs(&base_env, n).expect("read base");
    let queries = read_fvecs(&query_path, nq).expect("read queries");
    let dim = base[0].len();
    eprintln!(
        "Loaded {} base x {dim} dim, {} queries",
        base.len(),
        queries.len()
    );

    let train_n = 2000.min(base.len());
    let train: Vec<Vec<f32>> = base.iter().take(train_n).cloned().collect();

    // Plain PQ
    let t0 = Instant::now();
    let mut pq = PqEmlHnsw::train_and_build(
        &train,
        n_subspaces,
        n_centroids,
        kmeans_iters,
        base.len() + 16,
        16,
        200,
    );
    pq.add_batch(&base);
    let pq_build = t0.elapsed();
    let pq_mse = pq.codebook().mean_final_mse();

    // OPQ (full parametric: eigen + permutation)
    let t0 = Instant::now();
    let mut opq = PqEmlHnsw::train_and_build_opq(
        &train,
        n_subspaces,
        n_centroids,
        kmeans_iters,
        base.len() + 16,
        16,
        200,
    );
    opq.add_batch(&base);
    let opq_build = t0.elapsed();
    let opq_mse = opq.codebook().mean_final_mse();

    // OPQ-NP (permutation only, preserves natural feature grouping)
    let t0 = Instant::now();
    let mut opq_np = PqEmlHnsw::train_and_build_opq_np(
        &train,
        n_subspaces,
        n_centroids,
        kmeans_iters,
        base.len() + 16,
        16,
        200,
    );
    opq_np.add_batch(&base);
    let opq_np_build = t0.elapsed();
    let opq_np_mse = opq_np.codebook().mean_final_mse();

    eprintln!(
        "Build times: plain PQ {:?} (mse={:.3}), OPQ {:?} (mse={:.3}), OPQ-NP {:?} (mse={:.3})",
        pq_build, pq_mse, opq_build, opq_mse, opq_np_build, opq_np_mse
    );

    let mut pq_rr_lat = Vec::with_capacity(nq);
    let mut opq_rr_lat = Vec::with_capacity(nq);
    let mut opq_np_rr_lat = Vec::with_capacity(nq);

    let mut pq_red_recall = 0.0f32;
    let mut pq_rr_recall = 0.0f32;
    let mut opq_red_recall = 0.0f32;
    let mut opq_rr_recall = 0.0f32;
    let mut opq_np_red_recall = 0.0f32;
    let mut opq_np_rr_recall = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&base, q, 10);

        let pr = pq.search(q, 10, ef_search);
        let t = Instant::now();
        let prr = pq.search_with_rerank(q, 10, fetch_k, ef_search);
        pq_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let ore = opq.search(q, 10, ef_search);
        let t = Instant::now();
        let orr = opq.search_with_rerank(q, 10, fetch_k, ef_search);
        opq_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let ore_np = opq_np.search(q, 10, ef_search);
        let t = Instant::now();
        let orr_np = opq_np.search_with_rerank(q, 10, fetch_k, ef_search);
        opq_np_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        pq_red_recall += recall_at_k(
            &truth,
            &pr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        pq_rr_recall += recall_at_k(
            &truth,
            &prr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        opq_red_recall += recall_at_k(
            &truth,
            &ore.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        opq_rr_recall += recall_at_k(
            &truth,
            &orr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        opq_np_red_recall += recall_at_k(
            &truth,
            &ore_np.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        opq_np_rr_recall += recall_at_k(
            &truth,
            &orr_np.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
    }

    let pq_red_recall = pq_red_recall / nq as f32;
    let pq_rr_recall = pq_rr_recall / nq as f32;
    let opq_red_recall = opq_red_recall / nq as f32;
    let opq_rr_recall = opq_rr_recall / nq as f32;
    let opq_np_red_recall = opq_np_red_recall / nq as f32;
    let opq_np_rr_recall = opq_np_rr_recall / nq as f32;

    let pq_rr_p50 = percentile(&mut pq_rr_lat.clone(), 0.5);
    let opq_rr_p50 = percentile(&mut opq_rr_lat.clone(), 0.5);
    let opq_np_rr_p50 = percentile(&mut opq_np_rr_lat.clone(), 0.5);

    eprintln!("============================================================");
    eprintln!(
        "OPQ variants vs plain PQ on SIFT1M subset (matched memory M={} x {})",
        n_subspaces, n_centroids
    );
    eprintln!(
        "  n={} queries={} dim={} fetch_k={} ef_search={}",
        base.len(),
        nq,
        dim,
        fetch_k,
        ef_search
    );
    eprintln!("| index      | red recall@10 | rerank@10 | p50 rr (us) | k-means mean MSE |");
    eprintln!("|------------|---------------|-----------|-------------|------------------|");
    eprintln!(
        "| plain PQ   |    {:.4}     |   {:.4}  |   {:>8.1}  |      {:.3}       |",
        pq_red_recall, pq_rr_recall, pq_rr_p50, pq_mse
    );
    eprintln!(
        "| OPQ (full) |    {:.4}     |   {:.4}  |   {:>8.1}  |      {:.3}       |",
        opq_red_recall, opq_rr_recall, opq_rr_p50, opq_mse
    );
    eprintln!(
        "| OPQ-NP     |    {:.4}     |   {:.4}  |   {:>8.1}  |      {:.3}       |",
        opq_np_red_recall, opq_np_rr_recall, opq_np_rr_p50, opq_np_mse
    );
    eprintln!(
        "  OPQ (full) delta: red={:+.4} rerank={:+.4} | MSE delta={:+.3}",
        opq_red_recall - pq_red_recall,
        opq_rr_recall - pq_rr_recall,
        opq_mse - pq_mse
    );
    eprintln!(
        "  OPQ-NP     delta: red={:+.4} rerank={:+.4} | MSE delta={:+.3}",
        opq_np_red_recall - pq_red_recall,
        opq_np_rr_recall - pq_rr_recall,
        opq_np_mse - pq_mse
    );
    eprintln!("============================================================");

    // Floor on PQ-native quality itself.
    assert!(
        pq_rr_recall >= 0.75,
        "plain PQ rerank recall {:.3} below 0.75 floor",
        pq_rr_recall
    );
    // Print-only: do not fail on OPQ regression. SIFT is a known adversarial
    // case for PCA-rotation OPQ (gradient histograms have meaningful local
    // structure that PCA destroys). Numbers are reported for inspection.
}
