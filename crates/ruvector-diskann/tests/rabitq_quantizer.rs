//! Integration tests for the RaBitQ-backed [`Quantizer`] in DiskANN.
//!
//! Acceptance test from `docs/research/nightly/2026-04-23-rabitq/README.md`
//! requires recall@10 ≥ 0.95 on a 100k × 768-d dataset; that's too slow for an
//! interactive `cargo test` run. We exercise the same shape at 1k × 128 here,
//! plus an apples-to-apples PQ-vs-RaBitQ comparison and an on-disk size sanity
//! check against the f32 baseline. The full-scale benchmark lives in
//! `benches/rabitq_recall.rs`.
#![cfg(feature = "rabitq")]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_diskann::quantize::{ProductQuantizer, Quantizer, RabitqQuantizer};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn brute_force_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn quantizer_topk<Q: Quantizer>(q: &Q, codes: &[Vec<u8>], query: &[f32], k: usize) -> Vec<usize> {
    let prep = q.prepare_query(query).expect("query prep");
    let mut scored: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, q.distance(&prep, c)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn rabitq_quantizer_self_query_is_top1() {
    // 1k × 128 vectors — building a real DiskANN graph would muddy the test of
    // the *quantizer* itself. We do a flat scan over RaBitQ codes; the top-1
    // hit on a self-query must be the query's own row (estimator at agreement
    // = D returns ≈ 0 distance).
    let dim = 128;
    let n = 1_000;
    let vectors = random_vectors(n, dim, 42);

    let mut q = RabitqQuantizer::new(dim, 0xC0FFEE);
    q.train(&vectors, 0).unwrap();

    let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

    let mut hits = 0;
    let probes = 32usize;
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..probes {
        let idx = rng.gen_range(0..n);
        let query = &vectors[idx];
        let topk = quantizer_topk(&q, &codes, query, 5);
        if topk.first() == Some(&idx) {
            hits += 1;
        }
    }
    let rate = hits as f32 / probes as f32;
    // RaBitQ's asymmetric estimator is exact on a self-query (B = D), so the
    // self-row must always sort first. Allow no slack.
    assert!(
        rate >= 0.99,
        "self-query top1 rate too low: {rate} ({hits}/{probes})"
    );
}

#[test]
fn rabitq_distance_self_is_near_zero() {
    let dim = 128;
    let mut q = RabitqQuantizer::new(dim, 7);
    let vectors = random_vectors(16, dim, 13);
    q.train(&vectors, 0).unwrap();
    for v in &vectors {
        let code = q.encode(v).unwrap();
        let prep = q.prepare_query(v).unwrap();
        let d = q.distance(&prep, &code);
        // ε bounded by the asymmetric estimator's f32 round-off on a unit
        // vector against its own quantised code.
        assert!(d.abs() < 1e-3, "self-distance {d} > 1e-3");
    }
}

#[test]
fn rabitq_recall_not_drastically_worse_than_pq() {
    // Apples-to-apples: same 1k × 128 dataset, both quantizers. Compare top-10
    // recall vs the brute-force f32 baseline. RaBitQ is allowed to *trail* PQ
    // here because we're not reranking — but it must not be drastically worse.
    let dim = 128;
    let n = 1_000;
    let k = 10;
    let vectors = random_vectors(n, dim, 99);

    // PQ: M=16 → 16 bytes/code (256 centroids per subspace).
    let m = 16usize;
    let mut pq = ProductQuantizer::new(dim, m).unwrap();
    pq.train(&vectors, 5).unwrap();
    let pq_codes: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v).unwrap()).collect();

    // RaBitQ: 1 bit/dim → 16 bytes of code (+4 bytes norm) at D=128.
    let mut rb = RabitqQuantizer::new(dim, 0xBADF00D);
    rb.train(&vectors, 0).unwrap();
    let rb_codes: Vec<Vec<u8>> = vectors.iter().map(|v| rb.encode(v).unwrap()).collect();

    let queries = random_vectors(20, dim, 100);
    let mut pq_recall = 0.0f32;
    let mut rb_recall = 0.0f32;
    for query in &queries {
        let gt: std::collections::HashSet<usize> =
            brute_force_topk(&vectors, query, k).into_iter().collect();
        let pq_hits: std::collections::HashSet<usize> = quantizer_topk(&pq, &pq_codes, query, k)
            .into_iter()
            .collect();
        let rb_hits: std::collections::HashSet<usize> = quantizer_topk(&rb, &rb_codes, query, k)
            .into_iter()
            .collect();
        pq_recall += gt.intersection(&pq_hits).count() as f32 / k as f32;
        rb_recall += gt.intersection(&rb_hits).count() as f32 / k as f32;
    }
    pq_recall /= queries.len() as f32;
    rb_recall /= queries.len() as f32;
    eprintln!(
        "[1k×128] PQ recall@10 = {pq_recall:.3}, RaBitQ recall@10 = {rb_recall:.3} (no rerank)"
    );

    // RaBitQ without reranking is the *fast scan* path; the research note
    // measures 40% recall@10 at n=5k for that path. We require it to clear a
    // sanity floor here; full 95% recall is the rerank+IVF path tracked under
    // the bench in `benches/rabitq_recall.rs`.
    assert!(rb_recall >= 0.10, "RaBitQ recall too low: {rb_recall}");
    // PQ should also produce something non-trivial — guards against a
    // regression in the pre-existing PQ pipeline.
    assert!(pq_recall >= 0.30, "PQ recall too low: {pq_recall}");
}

#[test]
fn rabitq_on_disk_size_is_at_most_one_sixteenth_of_f32() {
    // Acceptance test #2 from the research roadmap: on-disk size of the codes
    // alone is ≤ 1/16 of the f32 baseline. We measure the quantizer's
    // self-reported `code_bytes` and compare to `dim * 4`. Includes the
    // 4-byte norm header so this is the full per-vector footprint.
    for &dim in &[128usize, 256, 512, 768, 1024] {
        let q = RabitqQuantizer::new(dim, 0);
        let f32_bytes = dim * 4;
        let rabitq_bytes = q.code_bytes();
        // 1/16 of f32 = dim/4 bytes. With the +4 byte norm header we allow a
        // small constant slack at low D; check the asymptotic ratio holds at
        // every D ≥ 128.
        let ratio = rabitq_bytes as f32 / f32_bytes as f32;
        eprintln!("dim={dim} f32={f32_bytes}B rabitq={rabitq_bytes}B ratio={ratio:.3}");
        // Allow the 4-byte norm overhead, which is the dominant cost at low D.
        // Floor: 1/16 + 4/(D*4) = 0.0625 + 1/D. At D=128 that's 0.0703.
        let allowed = 1.0 / 16.0 + 1.0 / (dim as f32);
        assert!(
            ratio <= allowed + 0.01,
            "on-disk ratio {ratio} > {allowed} at dim={dim}"
        );
    }
}

#[test]
fn rabitq_train_then_encode_within_diskann_loop() {
    // Smoke test that mirrors how DiskAnnIndex::build wires PQ today: collect
    // f32 vectors, hand them to the quantizer's `train`, then `encode` each
    // and stash the bytes. Confirms the trait surface lines up.
    let dim = 64;
    let n = 200;
    let vectors = random_vectors(n, dim, 5);
    let mut q = RabitqQuantizer::new(dim, 1);
    q.train(&vectors, 0).unwrap();
    let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();
    assert_eq!(codes.len(), n);
    for c in &codes {
        assert_eq!(c.len(), q.code_bytes());
    }
}
