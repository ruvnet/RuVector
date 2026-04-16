//! Tier-1C A/B: retention-objective selector vs Pearson selector on SIFT1M.
//!
//! Hypothesis: the Pearson-correlation selector in `EmlDistanceModel::train`
//! optimizes the wrong objective (correlation between pair-distance and exact
//! distance), so selected dimensions do not maximize retention of the true
//! top-k. A greedy forward selector that directly optimizes mean recall@k
//! against a held-out training corpus should beat it on the Stage-3 SIFT1M
//! bottleneck.
//!
//! Gated behind three env vars so the test skips cleanly when SIFT1M is
//! unavailable:
//!
//!   RUVECTOR_EML_SIFT1M_BASE   -> sift_base.fvecs  (evaluation corpus)
//!   RUVECTOR_EML_SIFT1M_LEARN  -> sift_learn.fvecs (selector training only)
//!   RUVECTOR_EML_SIFT1M_QUERY  -> sift_query.fvecs (evaluation queries)
//!
//! Selector training uses ONLY sift_learn to avoid leakage into evaluation.

use ruvector_eml_hnsw::cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
use ruvector_eml_hnsw::hnsw_integration::{EmlHnsw, EmlMetric};
use std::collections::HashSet;
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
    // 1-based ids to match EmlHnsw's convention.
    let mut s: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1, cosine_distance_f32(q, v)))
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(truth: &[usize], got: &[usize], k: usize) -> f32 {
    let tset: HashSet<_> = truth.iter().take(k).collect();
    got.iter().take(k).filter(|i| tset.contains(i)).count() as f32 / k as f32
}

fn build_with_selector(
    selector: EmlDistanceModel,
    corpus: &[Vec<f32>],
    m: usize,
    ef_c: usize,
) -> EmlHnsw {
    let mut idx = EmlHnsw::new(selector, EmlMetric::Cosine, corpus.len() + 16, m, ef_c)
        .expect("build index");
    idx.add_batch(corpus);
    idx
}

#[test]
fn retention_vs_pearson_sift1m() {
    let base_env = match std::env::var("RUVECTOR_EML_SIFT1M_BASE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_BASE");
            return;
        }
    };
    let learn_env = match std::env::var("RUVECTOR_EML_SIFT1M_LEARN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_LEARN");
            return;
        }
    };
    let query_env = match std::env::var("RUVECTOR_EML_SIFT1M_QUERY") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set RUVECTOR_EML_SIFT1M_QUERY");
            return;
        }
    };

    // Fixed sizes per the Tier-1C brief.
    const TRAIN_CORPUS_N: usize = 1000;
    const TRAIN_QUERIES_N: usize = 500;
    const EVAL_N: usize = 50_000;
    const EVAL_QUERIES: usize = 200;
    const SELECTED_K: usize = 32;
    const TARGET_K: usize = 10;
    const CANDIDATE_POOL: usize = 100;
    const HNSW_M: usize = 16;
    const HNSW_EF_C: usize = 200;
    const HNSW_EF_S: usize = 64;
    const FETCH_K: usize = 200;

    eprintln!("loading SIFT1M …");
    // Selector training data: first 1000 + next 500 from sift_learn.
    let learn = read_fvecs(&learn_env, TRAIN_CORPUS_N + TRAIN_QUERIES_N).expect("read learn");
    assert!(
        learn.len() >= TRAIN_CORPUS_N + TRAIN_QUERIES_N,
        "sift_learn too small: {}",
        learn.len()
    );
    let train_corpus: Vec<Vec<f32>> = learn[..TRAIN_CORPUS_N].to_vec();
    let train_queries: Vec<Vec<f32>> =
        learn[TRAIN_CORPUS_N..TRAIN_CORPUS_N + TRAIN_QUERIES_N].to_vec();

    let eval_corpus = read_fvecs(&base_env, EVAL_N).expect("read base");
    let eval_queries = read_fvecs(&query_env, EVAL_QUERIES).expect("read query");
    let dim = eval_corpus[0].len();
    eprintln!(
        "loaded: train_corpus={} train_queries={} eval_corpus={} eval_queries={} dim={}",
        train_corpus.len(),
        train_queries.len(),
        eval_corpus.len(),
        eval_queries.len(),
        dim
    );

    // Ground truth on eval set.
    eprintln!("computing exact top-{TARGET_K} ground truth on eval set …");
    let t_gt = Instant::now();
    let truths: Vec<Vec<usize>> = eval_queries
        .iter()
        .map(|q| brute_force_top_k(&eval_corpus, q, TARGET_K))
        .collect();
    eprintln!("ground truth done in {:?}", t_gt.elapsed());

    // --- A: Pearson selector ------------------------------------------------
    let t_p = Instant::now();
    let mut pearson = EmlDistanceModel::new(dim, SELECTED_K);
    // Record training pairs from train_corpus (disjoint from eval).
    // Use the same pair-recording heuristic as `train_and_build`: pair adjacent
    // entries, then top up with strided cross-pairs if needed.
    for chunk in train_corpus.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let d = cosine_distance_f32(&chunk[0], &chunk[1]);
        pearson.record(&chunk[0], &chunk[1], d);
    }
    let need = 600usize.saturating_sub(pearson.sample_count());
    if need > 0 {
        let stride = (train_corpus.len() / (need + 2)).max(1);
        let mut i = 0;
        let mut recorded = 0;
        while recorded < need && i + stride < train_corpus.len() {
            let d = cosine_distance_f32(&train_corpus[i], &train_corpus[i + stride]);
            pearson.record(&train_corpus[i], &train_corpus[i + stride], d);
            recorded += 1;
            i += 1;
        }
    }
    let _ = pearson.train();
    let pearson_train_s = t_p.elapsed().as_secs_f64();
    assert!(pearson.is_trained());
    assert_eq!(pearson.selected_dims().len(), SELECTED_K);
    eprintln!(
        "pearson selector trained in {:.3}s ({} samples), dims={:?}",
        pearson_train_s,
        pearson.sample_count(),
        pearson.selected_dims(),
    );

    // --- B: retention-objective selector ------------------------------------
    let t_r = Instant::now();
    let mut retention = EmlDistanceModel::new(dim, SELECTED_K);
    let ok = retention.train_for_retention(
        &train_corpus,
        &train_queries,
        TARGET_K,
        CANDIDATE_POOL,
    );
    let retention_train_s = t_r.elapsed().as_secs_f64();
    assert!(ok, "train_for_retention failed");
    assert!(retention.is_trained());
    assert_eq!(retention.selected_dims().len(), SELECTED_K);
    eprintln!(
        "retention selector trained in {:.3}s, dims={:?}",
        retention_train_s,
        retention.selected_dims(),
    );

    // --- Build two HNSWs and evaluate ---------------------------------------
    eprintln!("building HNSW (pearson) …");
    let t_b1 = Instant::now();
    let pearson_idx = build_with_selector(pearson, &eval_corpus, HNSW_M, HNSW_EF_C);
    eprintln!("pearson index built in {:?}", t_b1.elapsed());

    eprintln!("building HNSW (retention) …");
    let t_b2 = Instant::now();
    let retention_idx = build_with_selector(retention, &eval_corpus, HNSW_M, HNSW_EF_C);
    eprintln!("retention index built in {:?}", t_b2.elapsed());

    let mut pearson_recall = 0.0f32;
    let mut retention_recall = 0.0f32;
    for (qi, q) in eval_queries.iter().enumerate() {
        let p_hits: Vec<usize> = pearson_idx
            .search_with_rerank(q, TARGET_K, FETCH_K, HNSW_EF_S)
            .into_iter()
            .map(|r| r.id)
            .collect();
        let r_hits: Vec<usize> = retention_idx
            .search_with_rerank(q, TARGET_K, FETCH_K, HNSW_EF_S)
            .into_iter()
            .map(|r| r.id)
            .collect();
        pearson_recall += recall_at_k(&truths[qi], &p_hits, TARGET_K);
        retention_recall += recall_at_k(&truths[qi], &r_hits, TARGET_K);
    }
    pearson_recall /= EVAL_QUERIES as f32;
    retention_recall /= EVAL_QUERIES as f32;
    let delta = retention_recall - pearson_recall;

    // Rough binomial standard error for 200 queries at the observed rate.
    let se_p = ((pearson_recall * (1.0 - pearson_recall)) / EVAL_QUERIES as f32).sqrt();
    let se_r =
        ((retention_recall * (1.0 - retention_recall)) / EVAL_QUERIES as f32).sqrt();

    eprintln!("------------------------------------------------------------");
    eprintln!("Tier-1C: Pearson vs Retention selector (SIFT1M, selected_k={SELECTED_K})");
    eprintln!("");
    eprintln!("| selector   | recall@10 | selector_train_s |");
    eprintln!("|------------|-----------|------------------|");
    eprintln!(
        "| pearson    | {pearson_recall:.4}    | {pearson_train_s:.3}            |"
    );
    eprintln!(
        "| retention  | {retention_recall:.4}    | {retention_train_s:.3}            |"
    );
    eprintln!("");
    eprintln!("delta (retention - pearson) = {delta:+.4}");
    eprintln!("rough SE (200 queries): pearson ~{se_p:.4}, retention ~{se_r:.4}");
    eprintln!("fetch_k={FETCH_K}, ef_search={HNSW_EF_S}, target_k={TARGET_K}");
    eprintln!("------------------------------------------------------------");

    // Print the honest number either way and fail only on significant regression.
    assert!(
        retention_recall >= pearson_recall - 0.02,
        "retention selector regressed by more than 2pp: pearson={pearson_recall:.4} \
         retention={retention_recall:.4}"
    );
}
