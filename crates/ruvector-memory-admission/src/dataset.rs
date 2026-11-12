//! Synthetic streaming agent-memory dataset.
//!
//! Generates `K` ground-truth semantic clusters (Gaussian blobs on the unit
//! sphere in R^dims), then emits them as a single interleaved stream — the
//! order an agent would actually see memories arrive in across a session
//! that jumps between topics, not grouped by topic. A configurable fraction
//! of points are drawn with much higher noise ("drift" points) to create
//! boundary cases that stress admission decisions.
//!
//! Ground-truth cluster ids are carried alongside the stream *only* for
//! measurement (purity, recall grading) — admission policies never see them.

pub struct StreamPoint {
    pub vector: Vec<f32>,
    /// Ground-truth source cluster, used only for evaluation.
    pub true_cluster: usize,
}

pub struct StreamDataset {
    pub dims: usize,
    pub k_true: usize,
    pub points: Vec<StreamPoint>,
}

pub struct StreamConfig {
    pub n_points: usize,
    pub k_true: usize,
    pub dims: usize,
    pub seed: u64,
    /// Std-dev of Gaussian noise for "clean" points.
    pub noise: f32,
    /// Std-dev of Gaussian noise for "drift" (boundary) points.
    pub drift_noise: f32,
    /// Fraction of points drawn as drift points.
    pub drift_frac: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            n_points: 4000,
            k_true: 8,
            dims: 64,
            seed: 0x5EED_1234_ABCD,
            noise: 0.22,
            drift_noise: 0.55,
            drift_frac: 0.20,
        }
    }
}

impl StreamDataset {
    pub fn generate(cfg: &StreamConfig) -> Self {
        let mut rng = Lcg64(cfg.seed);
        let centres: Vec<Vec<f32>> = (0..cfg.k_true)
            .map(|c| make_centre(cfg.dims, c, &mut rng))
            .collect();

        // Assign each point a source cluster up front, then shuffle the
        // *order* so arrivals interleave across topics (Fisher-Yates on the
        // assignment vector, not on already-materialised vectors, so the
        // per-cluster distribution stays exact).
        let mut assignments: Vec<usize> = (0..cfg.n_points).map(|i| i % cfg.k_true).collect();
        fisher_yates(&mut assignments, &mut rng);

        let points = assignments
            .into_iter()
            .map(|c| {
                let is_drift = rng.uniform() < cfg.drift_frac;
                let sigma = if is_drift { cfg.drift_noise } else { cfg.noise };
                let vector = sample_around(&mut rng, &centres[c], sigma);
                StreamPoint {
                    vector,
                    true_cluster: c,
                }
            })
            .collect();

        StreamDataset {
            dims: cfg.dims,
            k_true: cfg.k_true,
            points,
        }
    }

    /// Clean (low-noise) held-out queries drawn from the same `k_true`
    /// centres, for post-stream recall evaluation. Regenerates the same
    /// centres deterministically from `cfg`, independent of point order.
    pub fn held_out_queries(
        &self,
        cfg: &StreamConfig,
        n: usize,
        seed: u64,
    ) -> Vec<(Vec<f32>, usize)> {
        let mut centre_rng = Lcg64(cfg.seed);
        let centres: Vec<Vec<f32>> = (0..cfg.k_true)
            .map(|c| make_centre(cfg.dims, c, &mut centre_rng))
            .collect();

        let mut rng = Lcg64(seed);
        (0..n)
            .map(|i| {
                let c = i % cfg.k_true;
                (sample_around(&mut rng, &centres[c], cfg.noise), c)
            })
            .collect()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_centre(dims: usize, cluster: usize, rng: &mut Lcg64) -> Vec<f32> {
    // Deterministic per-cluster centre: distinct random unit vector, seeded
    // from `cluster` so `generate` and `held_out_queries` reproduce the same
    // centres from independent `Lcg64(cfg.seed)` instances.
    let mut v = vec![0f32; dims];
    let mut local = Lcg64(rng.0 ^ (cluster as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    for x in v.iter_mut() {
        *x = local.gaussian();
    }
    normalise(&mut v);
    v
}

fn sample_around(rng: &mut Lcg64, centre: &[f32], sigma: f32) -> Vec<f32> {
    let mut v: Vec<f32> = centre.iter().map(|&c| c + sigma * rng.gaussian()).collect();
    normalise(&mut v);
    v
}

pub fn normalise(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn fisher_yates(a: &mut [usize], rng: &mut Lcg64) {
    for i in (1..a.len()).rev() {
        let j = (rng.uniform() * (i as f32 + 1.0)) as usize;
        let j = j.min(i);
        a.swap(i, j);
    }
}

// ─── minimal LCG + Box-Muller RNG (no external deps, matches the
//     ruvector-namespace-merge convention for reproducible nightly
//     benchmarks without pulling in `rand`) ─────────────────────────────────

pub struct Lcg64(pub u64);

impl Lcg64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_point_count() {
        let cfg = StreamConfig {
            n_points: 500,
            ..StreamConfig::default()
        };
        let ds = StreamDataset::generate(&cfg);
        assert_eq!(ds.points.len(), 500);
    }

    #[test]
    fn clusters_are_balanced_before_shuffle_semantics() {
        let cfg = StreamConfig {
            n_points: 800,
            k_true: 8,
            ..StreamConfig::default()
        };
        let ds = StreamDataset::generate(&cfg);
        let mut counts = vec![0usize; cfg.k_true];
        for p in &ds.points {
            counts[p.true_cluster] += 1;
        }
        for c in counts {
            assert_eq!(c, 100, "each of 8 clusters should get exactly 100 points");
        }
    }

    #[test]
    fn vectors_are_unit_norm() {
        let ds = StreamDataset::generate(&StreamConfig {
            n_points: 50,
            ..StreamConfig::default()
        });
        for p in &ds.points {
            let norm: f32 = p.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
        }
    }

    #[test]
    fn held_out_queries_reuse_same_centres() {
        let cfg = StreamConfig::default();
        let ds = StreamDataset::generate(&cfg);
        let queries = ds.held_out_queries(&cfg, 20, 0xAAAA);
        // A clean query for cluster c should be closer (cosine) to at least
        // one true member of cluster c than to a random other cluster on
        // average — sanity check that centres line up between the two
        // independent generation paths.
        let mut same_cluster_closer = 0usize;
        for (q, c) in &queries {
            let same: f32 = ds
                .points
                .iter()
                .filter(|p| p.true_cluster == *c)
                .map(|p| crate::cosine_sim(q, &p.vector))
                .fold(f32::NEG_INFINITY, f32::max);
            let other: f32 = ds
                .points
                .iter()
                .filter(|p| p.true_cluster != *c)
                .map(|p| crate::cosine_sim(q, &p.vector))
                .fold(f32::NEG_INFINITY, f32::max);
            if same > other {
                same_cluster_closer += 1;
            }
        }
        assert!(
            same_cluster_closer >= queries.len() * 9 / 10,
            "expected most held-out queries to be nearest their true cluster, got {same_cluster_closer}/{}",
            queries.len()
        );
    }
}
