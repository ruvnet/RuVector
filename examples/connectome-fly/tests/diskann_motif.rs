#![allow(clippy::needless_range_loop)]
//! Tests for `analysis::diskann_motif` — the Vamana-style motif index.
//!
//! These tests cover, in order:
//!
//! 1. `build_query_roundtrip` — build + query on a small labelled
//!    fixture, with a sanity-check that the self-query returns the
//!    query point as its first hit.
//! 2. `determinism_two_queries` — two indexes built from the same
//!    corpus + params return bit-identical query results.
//! 3. `recall_at_5_vs_bruteforce` — ≥ 0.95 recall@5 on a 10 000-vector
//!    synthetic Gaussian-mixture corpus. Brute force is the ground
//!    truth; Vamana must recover 95 % of its top-5.
//!
//! The fourth acceptance test — AC-2 on ≥ 100 windows — lives in
//! `tests/acceptance_core.rs::ac_2_motif_emergence_diskann` so it runs
//! alongside the existing AC-2 and the BENCHMARK.md comparison row
//! stays co-located.

use connectome_fly::{DiskAnnMotifIndex, EmbeddingF32, VamanaParams};

// -----------------------------------------------------------------
// 1. Build + query round-trip
// -----------------------------------------------------------------

#[test]
fn build_query_roundtrip() {
    // Tiny labelled fixture: 8 points on the unit grid in 2-D. Query
    // each point; the nearest hit (at distance 0) MUST be the point
    // itself.
    let corpus: Vec<EmbeddingF32> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![0.0, 2.0],
        vec![1.0, 2.0],
    ];
    let idx = DiskAnnMotifIndex::new(
        corpus.clone(),
        VamanaParams {
            max_degree: 4,
            build_beam: 8,
            search_beam: 8,
            alpha: 1.2,
            seed: 1,
        },
    );
    assert_eq!(idx.len(), corpus.len());
    for (i, v) in corpus.iter().enumerate() {
        let hits = idx.query(v, 3);
        assert_eq!(hits.len(), 3, "k=3 should return 3 hits");
        assert_eq!(hits[0].0, i, "self-hit should come first");
        assert!(hits[0].1 < 1e-6, "self-hit should have ~0 distance");
        // Distances must be sorted ascending.
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "hits not sorted by distance");
        }
    }
}

// -----------------------------------------------------------------
// 2. Determinism
// -----------------------------------------------------------------

#[test]
fn determinism_two_queries() {
    let corpus = synthetic_corpus(256, 32, 0xFEED_FACE);
    let params = VamanaParams {
        max_degree: 24,
        build_beam: 48,
        search_beam: 48,
        alpha: 1.2,
        seed: 0xCAFEBEEF,
    };
    let idx_a = DiskAnnMotifIndex::new(corpus.clone(), params.clone());
    let idx_b = DiskAnnMotifIndex::new(corpus.clone(), params);
    for i in 0..8 {
        let q = &corpus[i];
        let a = idx_a.query(q, 10);
        let b = idx_b.query(q, 10);
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.0, rb.0, "result id differs");
            assert_eq!(
                ra.1.to_bits(),
                rb.1.to_bits(),
                "distance differs (non-deterministic build)"
            );
        }
    }
}

// -----------------------------------------------------------------
// 3. Recall@5 vs brute force on 10 000-vector synthetic corpus
// -----------------------------------------------------------------

#[test]
fn recall_at_5_vs_bruteforce_10k() {
    const N: usize = 10_000;
    const DIM: usize = 32;
    const K: usize = 5;
    const Q: usize = 100;

    let corpus = mixture_corpus(N, DIM, 16, 0xBEEF_0F00);
    let idx = DiskAnnMotifIndex::new(
        corpus.clone(),
        VamanaParams {
            max_degree: 48,
            build_beam: 96,
            search_beam: 96,
            alpha: 1.2,
            seed: 0x51DECAFE,
        },
    );

    // Pick Q query points deterministically from the corpus (stride
    // sampling keeps the test fully seeded).
    let stride = N / Q;
    let mut recalls: Vec<f32> = Vec::with_capacity(Q);
    for qi in 0..Q {
        let qid = qi * stride;
        let q = &corpus[qid];
        let gt = brute_force_topk(&corpus, q, K + 1);
        let ann = idx.query(q, K + 1);
        // Drop the self-hit from both (qid, distance ~= 0).
        let gt_ids: std::collections::HashSet<usize> = gt
            .iter()
            .filter(|(id, _)| *id != qid)
            .map(|(id, _)| *id)
            .take(K)
            .collect();
        let ann_ids: std::collections::HashSet<usize> = ann
            .iter()
            .filter(|(id, _)| *id != qid)
            .map(|(id, _)| *id)
            .take(K)
            .collect();
        let hit = gt_ids.intersection(&ann_ids).count();
        recalls.push(hit as f32 / K as f32);
    }
    let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
    eprintln!(
        "diskann recall@{K}: mean={mean:.3}  n={N}  dim={DIM}  queries={Q}"
    );
    assert!(
        mean >= 0.95,
        "recall@{K} {mean:.3} below target 0.95 (10 000-vector corpus)"
    );
}

// -----------------------------------------------------------------
// Helpers (deterministic PRNG; same xoroshiro as analysis module)
// -----------------------------------------------------------------

fn synthetic_corpus(n: usize, dim: usize, seed: u64) -> Vec<EmbeddingF32> {
    let mut rng = TestRng::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(rng.next_f32_unit() * 2.0 - 1.0);
        }
        out.push(v);
    }
    out
}

/// Gaussian-mixture corpus: `clusters` centres in a DIM-cube, each
/// point is its centre plus tight iid noise. Well-separated clusters
/// make brute-force ground truth clean so a 0.95 recall bound is a
/// real signal rather than a dense-sphere coincidence.
fn mixture_corpus(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<EmbeddingF32> {
    let mut rng = TestRng::new(seed);
    let mut centres: Vec<Vec<f32>> = Vec::with_capacity(clusters);
    for _ in 0..clusters {
        let mut c = Vec::with_capacity(dim);
        for _ in 0..dim {
            c.push((rng.next_f32_unit() - 0.5) * 8.0);
        }
        centres.push(c);
    }
    let sigma = 0.35_f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let centre = &centres[i % clusters];
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let g = rng.next_gauss() * sigma;
            v.push(centre[d] + g);
        }
        out.push(v);
    }
    out
}

fn brute_force_topk(corpus: &[EmbeddingF32], q: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2(v, q)))
        .collect();
    all.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Tiny xoroshiro128++ for test-local determinism (matches the PRNG
/// in `analysis::diskann_motif` but reimplemented to keep these tests
/// independent of the crate-internal API).
struct TestRng {
    s0: u64,
    s1: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let s0 = splitmix(&mut z);
        let s1 = splitmix(&mut z);
        let s0 = if s0 == 0 { 0xD1B5_4A32_D192_ED03 } else { s0 };
        let s1 = if s1 == 0 { 0x6A09_E667_BB67_AE85 } else { s1 };
        Self { s0, s1 }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s0.wrapping_add(self.s1).rotate_left(17).wrapping_add(self.s0);
        let s1 = self.s1 ^ self.s0;
        self.s0 = self.s0.rotate_left(49) ^ s1 ^ (s1 << 21);
        self.s1 = s1.rotate_left(28);
        r
    }

    fn next_f32_unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        u as f32
    }

    /// Box-Muller standard normal.
    fn next_gauss(&mut self) -> f32 {
        let u1 = (self.next_f32_unit()).max(1e-9);
        let u2 = self.next_f32_unit();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let th = 2.0_f32 * std::f32::consts::PI * u2;
        r * th.cos()
    }
}

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
