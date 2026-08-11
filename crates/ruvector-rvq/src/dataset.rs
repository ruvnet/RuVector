//! Deterministic synthetic dataset generation for RVQ benchmarks.
//!
//! Uses an LCG PRNG so runs are fully reproducible without external crates.

/// Lightweight LCG (Knuth constants).
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xcafe_babe_dead_beef,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f64() + 1e-12).ln();
        let u2 = self.next_f64() * std::f64::consts::TAU;
        ((-2.0 * u1).sqrt() * u2.cos()) as f32
    }

    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub n_vectors: usize,
    pub dims: usize,
    pub n_queries: usize,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            n_vectors: 10_000,
            dims: 128,
            n_queries: 500,
            seed: 0xC0DE_CAFE_BABE_7777,
        }
    }
}

pub struct Dataset {
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dims: usize,
}

impl Dataset {
    pub fn generate(cfg: &DatasetConfig) -> Self {
        let mut rng = Lcg::new(cfg.seed);
        let vectors = gen_unit_vecs(cfg.n_vectors, cfg.dims, &mut rng);
        let queries = gen_unit_vecs(cfg.n_queries, cfg.dims, &mut rng);
        Self {
            vectors,
            queries,
            dims: cfg.dims,
        }
    }
}

pub fn gen_unit_vecs(n: usize, dims: usize, rng: &mut Lcg) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| rng.next_normal()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}
