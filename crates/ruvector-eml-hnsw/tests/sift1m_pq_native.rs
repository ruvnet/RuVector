//! SIFT1M benchmark for the PQ-native HNSW (codes-in-graph).
//!
//! Compares PqEmlHnsw (codes-in-graph) vs PqEmlHnswLegacy (float-reconstruction
//! in graph) on the same slice. Both should hit the same recall ceiling
//! *after* rerank; the PQ-native version trades a small slice of graph-time
//! accuracy (symmetric PQ distance) for a real 64× reduction in per-node
//! payload.
//!
//! Gated behind RUVECTOR_EML_SIFT1M_BASE / RUVECTOR_EML_SIFT1M_QUERY.

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::pq_hnsw::{PqEmlHnsw, PqEmlHnswLegacy};
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
fn sift1m_pq_native() {
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
        "Loading SIFT1M: base={}, n={n}, queries={nq}, M={n_subspaces} x {n_centroids}, iters={kmeans_iters}",
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

    // PQ-native (codes in graph)
    let t0 = Instant::now();
    let mut pq_native = PqEmlHnsw::train_and_build(
        &train,
        n_subspaces,
        n_centroids,
        kmeans_iters,
        base.len() + 16,
        16,
        200,
    );
    pq_native.add_batch(&base);
    let pq_native_build = t0.elapsed();

    // Legacy (floats in graph)
    let t0 = Instant::now();
    let mut pq_legacy = PqEmlHnswLegacy::train_and_build(
        &train,
        n_subspaces,
        n_centroids,
        kmeans_iters,
        base.len() + 16,
        16,
        200,
    );
    pq_legacy.add_batch(&base);
    let pq_legacy_build = t0.elapsed();

    eprintln!(
        "Build times: PQ-native {:?}, PQ-legacy {:?}",
        pq_native_build, pq_legacy_build
    );

    let mut nat_red_lat = Vec::with_capacity(nq);
    let mut nat_rr_lat = Vec::with_capacity(nq);
    let mut leg_rr_lat = Vec::with_capacity(nq);

    let mut nat_red_recall = 0.0f32;
    let mut nat_rr_recall = 0.0f32;
    let mut leg_rr_recall = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&base, q, 10);

        let t = Instant::now();
        let nr = pq_native.search(q, 10, ef_search);
        nat_red_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let nrr = pq_native.search_with_rerank(q, 10, fetch_k, ef_search);
        nat_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let lrr = pq_legacy.search_with_rerank(q, 10, fetch_k, ef_search);
        leg_rr_lat.push(t.elapsed().as_secs_f64() * 1e6);

        nat_red_recall += recall_at_k(
            &truth,
            &nr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        nat_rr_recall += recall_at_k(
            &truth,
            &nrr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
        leg_rr_recall += recall_at_k(
            &truth,
            &lrr.iter().map(|r| r.id).collect::<Vec<_>>(),
            10,
        );
    }

    let nat_red_recall = nat_red_recall / nq as f32;
    let nat_rr_recall = nat_rr_recall / nq as f32;
    let leg_rr_recall = leg_rr_recall / nq as f32;

    let nat_red_p50 = percentile(&mut nat_red_lat.clone(), 0.5);
    let nat_red_p95 = percentile(&mut nat_red_lat.clone(), 0.95);
    let nat_rr_p50 = percentile(&mut nat_rr_lat.clone(), 0.5);
    let nat_rr_p95 = percentile(&mut nat_rr_lat.clone(), 0.95);
    let leg_rr_p50 = percentile(&mut leg_rr_lat.clone(), 0.5);
    let leg_rr_p95 = percentile(&mut leg_rr_lat.clone(), 0.95);

    let nat_payload = pq_native.hnsw_payload_bytes_per_vec();
    let leg_payload = pq_legacy.hnsw_stored_bytes_per_vec();

    eprintln!("============================================================");
    eprintln!("PQ-native vs PQ-legacy on SIFT1M subset");
    eprintln!(
        "  n={} queries={} dim={} M={}x{} fetch_k={} ef_search={}",
        base.len(),
        nq,
        dim,
        n_subspaces,
        n_centroids,
        fetch_k,
        ef_search
    );
    eprintln!(
        "| index       | recall@10 | rerank@10 | p50 red (us) | p95 red (us) | p50 rr (us) | p95 rr (us) | graph payload bytes/vec |"
    );
    eprintln!(
        "|-------------|-----------|-----------|--------------|--------------|-------------|-------------|-------------------------|"
    );
    eprintln!(
        "| PQ-native   |   {:.4}  |   {:.4}  |   {:>8.1}   |   {:>8.1}   |  {:>8.1}   |  {:>8.1}   | {:>6}                  |",
        nat_red_recall, nat_rr_recall, nat_red_p50, nat_red_p95, nat_rr_p50, nat_rr_p95, nat_payload
    );
    eprintln!(
        "| PQ-legacy   |   ----   |   {:.4}  |     ----     |     ----     |  {:>8.1}   |  {:>8.1}   | {:>6}                  |",
        leg_rr_recall, leg_rr_p50, leg_rr_p95, leg_payload
    );
    eprintln!(
        "  Graph-payload reduction: {:.1}x ({} -> {} bytes/vec)",
        leg_payload as f64 / nat_payload as f64,
        leg_payload,
        nat_payload
    );
    eprintln!("============================================================");

    // Graph payload is strictly 8 bytes for M=8.
    assert_eq!(
        nat_payload, n_subspaces,
        "PQ-native graph payload must equal n_subspaces"
    );
    // Floor: rerank recall should stay within 5pp of the legacy impl.
    assert!(
        nat_rr_recall + 0.05 >= leg_rr_recall,
        "PQ-native rerank {:.3} trails legacy {:.3} by more than 5pp",
        nat_rr_recall,
        leg_rr_recall
    );
    assert!(
        nat_rr_recall >= 0.75,
        "PQ-native rerank recall {:.3} below 0.75 floor",
        nat_rr_recall
    );
}
