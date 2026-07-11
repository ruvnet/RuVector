use ruvector_sq_hnsw::{
    recall_at_k, FlatExact, FlatSq8, GraphSq4, GraphSq8, NnSearch, ScalarQuantizer, SqHnsw2,
};

/// Deterministic LCG for test data generation (no external deps).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(1e-9);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn vec(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| self.gaussian()).collect()
    }
}

fn build_corpus(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.vec(dims)).collect()
}

const N: usize = 2_000;
const DIMS: usize = 64;
const K: usize = 10;
const Q: usize = 50;

// ─── Correctness tests ───────────────────────────────────────────────────────

#[test]
fn flat_exact_returns_sorted_results() {
    let corpus = build_corpus(100, DIMS, 1);
    let mut idx = FlatExact::new();
    for v in &corpus {
        idx.insert(v.clone());
    }
    let q = corpus[0].clone();
    let res = idx.search(&q, 5);
    assert_eq!(res.len(), 5);
    // First result should be the query itself (id=0, distance≈0).
    assert_eq!(res[0].id, 0);
    assert!(
        res[0].distance < 1e-6,
        "self-distance not near zero: {}",
        res[0].distance
    );
    for w in res.windows(2) {
        assert!(w[0].distance <= w[1].distance, "results not sorted");
    }
}

#[test]
fn flat_sq8_recall_above_90pct() {
    let corpus = build_corpus(N, DIMS, 42);
    let sq = ScalarQuantizer::train(&corpus, 8);

    let mut exact = FlatExact::new();
    let mut sq8 = FlatSq8::new(K * 10);
    sq8.init_quantizer(ScalarQuantizer::train(&corpus, 8));

    for v in &corpus {
        exact.insert(v.clone());
        sq8.insert(v.clone());
    }

    let mut rng = Rng::new(99);
    let _ = sq; // keep train result
    let mut total_recall = 0.0f32;
    for _ in 0..Q {
        let q = rng.vec(DIMS);
        let gt = exact.search(&q, K);
        let approx = sq8.search(&q, K);
        total_recall += recall_at_k(&gt, &approx, K);
    }
    let mean_recall = total_recall / Q as f32;
    assert!(
        mean_recall >= 0.90,
        "FlatSq8 recall@{K} = {mean_recall:.3}, want >= 0.90"
    );
}

#[test]
fn graph_sq8_recall_above_85pct() {
    let corpus = build_corpus(N, DIMS, 7);
    let sq = ScalarQuantizer::train(&corpus, 8);

    let mut exact = FlatExact::new();
    let mut g = GraphSq8::new(sq, 16, 200, 50, 120);

    for v in &corpus {
        exact.insert(v.clone());
        g.insert(v.clone());
    }

    let mut rng = Rng::new(11);
    let mut total_recall = 0.0f32;
    for _ in 0..Q {
        let q = rng.vec(DIMS);
        let gt = exact.search(&q, K);
        let approx = g.search(&q, K);
        total_recall += recall_at_k(&gt, &approx, K);
    }
    let mean_recall = total_recall / Q as f32;
    assert!(
        mean_recall >= 0.85,
        "GraphSq8 recall@{K} = {mean_recall:.3}, want >= 0.85"
    );
}

#[test]
fn graph_sq4_recall_above_70pct() {
    let corpus = build_corpus(N, DIMS, 13);
    let sq = ScalarQuantizer::train(&corpus, 4);

    let mut exact = FlatExact::new();
    let mut g = GraphSq4::new(sq, 16, 200, 50, 150);

    for v in &corpus {
        exact.insert(v.clone());
        g.insert(v.clone());
    }

    let mut rng = Rng::new(17);
    let mut total_recall = 0.0f32;
    for _ in 0..Q {
        let q = rng.vec(DIMS);
        let gt = exact.search(&q, K);
        let approx = g.search(&q, K);
        total_recall += recall_at_k(&gt, &approx, K);
    }
    let mean_recall = total_recall / Q as f32;
    assert!(
        mean_recall >= 0.70,
        "GraphSq4 recall@{K} = {mean_recall:.3}, want >= 0.70"
    );
}

#[test]
fn memory_estimates_are_positive() {
    let corpus = build_corpus(100, 32, 5);
    let sq8 = ScalarQuantizer::train(&corpus, 8);
    let sq4 = ScalarQuantizer::train(&corpus, 4);

    let mut fe = FlatExact::new();
    let mut fs = FlatSq8::new(40);
    fs.init_quantizer(ScalarQuantizer::train(&corpus, 8));
    let mut g8 = GraphSq8::new(sq8, 8, 50, 20, 50);
    let mut g4 = GraphSq4::new(sq4, 8, 50, 20, 50);

    for v in &corpus {
        fe.insert(v.clone());
        fs.insert(v.clone());
        g8.insert(v.clone());
        g4.insert(v.clone());
    }

    let sq_h = ScalarQuantizer::train(&corpus, 8);
    let mut h2 = SqHnsw2::new(sq_h, 16, 8, 50, 32, 16, 50);
    for v in &corpus {
        h2.insert(v.clone());
    }

    assert!(fe.memory_bytes() > 0);
    assert!(fs.memory_bytes() > 0);
    assert!(g8.memory_bytes() > 0);
    assert!(g4.memory_bytes() > 0);
    assert!(h2.memory_bytes() > 0);
    assert!(
        g4.memory_bytes() < g8.memory_bytes() + 1,
        "4-bit should not use more than 8-bit"
    );
}

#[test]
fn sq_hnsw2_recall_above_88pct() {
    let corpus = build_corpus(N, DIMS, 99);
    let sq = ScalarQuantizer::train(&corpus, 8);

    let mut exact = FlatExact::new();
    // 2-layer HNSW: M0=32, M1=16, ef0=200, ef1=64, L1 every 16 nodes, ef_search=200
    let mut h2 = SqHnsw2::new(sq, 32, 16, 200, 64, 16, 200);

    for v in &corpus {
        exact.insert(v.clone());
        h2.insert(v.clone());
    }

    let mut rng = Rng::new(23);
    let mut total_recall = 0.0f32;
    for _ in 0..Q {
        let q = rng.vec(DIMS);
        let gt = exact.search(&q, K);
        let approx = h2.search(&q, K);
        total_recall += recall_at_k(&gt, &approx, K);
    }
    let mean_recall = total_recall / Q as f32;
    assert!(
        mean_recall >= 0.88,
        "SqHnsw2 recall@{K} = {mean_recall:.3}, want >= 0.88"
    );
}
