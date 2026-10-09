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

/// A regime-shift variant of [`StreamConfig`]: the stream runs in regime A
/// (the same geometry as [`StreamDataset::generate`]) for the first
/// `switch_at_frac` of its points, then switches to regime B, where the
/// same `k_true` cluster centres are pulled `regime_b_crowding` of the way
/// toward their shared mean — fewer, more confusable degrees of separation,
/// not a different number of topics. This models a realistic non-stationary
/// agent-memory session (topics becoming less mutually distinguishable
/// partway through, e.g. a conversation narrowing into closely related
/// sub-topics) rather than an arbitrary synthetic shock.
///
/// A fixed admission threshold tuned on regime A's wider cut-weight
/// distribution is, by construction, miscalibrated for regime B's tighter
/// one — this is what a self-calibrating policy is supposed to buy over a
/// fixed `tau`, and what this config exists to test.
pub struct DriftStreamConfig {
    pub n_points: usize,
    pub k_true: usize,
    pub dims: usize,
    pub seed: u64,
    pub noise: f32,
    pub drift_noise: f32,
    pub drift_frac: f32,
    /// Fraction of the stream (0.0..=1.0) in regime A before switching.
    pub switch_at_frac: f32,
    /// How far regime B's centres are pulled toward their shared mean
    /// relative to regime A (0.0 = identical geometry, 1.0 = collapsed to a
    /// single point).
    pub regime_b_crowding: f32,
}

impl Default for DriftStreamConfig {
    fn default() -> Self {
        let base = StreamConfig::default();
        DriftStreamConfig {
            n_points: base.n_points,
            k_true: base.k_true,
            dims: base.dims,
            seed: base.seed,
            noise: base.noise,
            drift_noise: base.drift_noise,
            drift_frac: base.drift_frac,
            switch_at_frac: 0.5,
            regime_b_crowding: 0.45,
        }
    }
}

impl StreamDataset {
    /// Generate both regimes' centre sets deterministically from `cfg`,
    /// independent of stream order — used by both [`Self::generate_drift`]
    /// and [`Self::held_out_queries_drift`] so query centres line up with
    /// the stream's.
    fn drift_centres(cfg: &DriftStreamConfig) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = Lcg64(cfg.seed);
        let centres_a: Vec<Vec<f32>> = (0..cfg.k_true)
            .map(|c| make_centre(cfg.dims, c, &mut rng))
            .collect();
        let centres_b = crowd_centres(&centres_a, cfg.regime_b_crowding);
        (centres_a, centres_b)
    }

    /// Like [`Self::generate`], but the cluster geometry switches from
    /// regime A to regime B partway through the stream (see
    /// [`DriftStreamConfig`]). Arrival order is interleaved first, then
    /// each position's regime (and therefore which centre set it samples
    /// from) is a function of its position in that interleaved order — so
    /// "drift" means drift *over time*, not merely a different shuffle.
    pub fn generate_drift(cfg: &DriftStreamConfig) -> (Self, usize) {
        let (centres_a, centres_b) = Self::drift_centres(cfg);
        let switch_at = ((cfg.n_points as f32) * cfg.switch_at_frac).round() as usize;

        let mut shuffle_rng = Lcg64(cfg.seed ^ 0x00D8_17F7_51A1);
        let mut assignments: Vec<usize> = (0..cfg.n_points).map(|i| i % cfg.k_true).collect();
        fisher_yates(&mut assignments, &mut shuffle_rng);

        let mut sample_rng = Lcg64(cfg.seed.wrapping_add(0xABCD_EF01));
        let points = assignments
            .into_iter()
            .enumerate()
            .map(|(pos, c)| {
                let centres = if pos < switch_at {
                    &centres_a
                } else {
                    &centres_b
                };
                let is_drift = sample_rng.uniform() < cfg.drift_frac;
                let sigma = if is_drift { cfg.drift_noise } else { cfg.noise };
                let vector = sample_around(&mut sample_rng, &centres[c], sigma);
                StreamPoint {
                    vector,
                    true_cluster: c,
                }
            })
            .collect();

        (
            StreamDataset {
                dims: cfg.dims,
                k_true: cfg.k_true,
                points,
            },
            switch_at,
        )
    }

    /// Clean held-out queries drawn from regime B's geometry specifically —
    /// used to measure how well a policy's *final* clustering (shaped by
    /// however it handled the regime switch) serves the post-drift regime,
    /// as opposed to the overall stream average.
    pub fn held_out_queries_drift(
        cfg: &DriftStreamConfig,
        n: usize,
        seed: u64,
        use_regime_b: bool,
    ) -> Vec<(Vec<f32>, usize)> {
        let (centres_a, centres_b) = Self::drift_centres(cfg);
        let centres = if use_regime_b { &centres_b } else { &centres_a };
        let mut rng = Lcg64(seed);
        (0..n)
            .map(|i| {
                let c = i % cfg.k_true;
                (sample_around(&mut rng, &centres[c], cfg.noise), c)
            })
            .collect()
    }
}

fn crowd_centres(centres: &[Vec<f32>], crowding: f32) -> Vec<Vec<f32>> {
    let dims = centres[0].len();
    let mut mean = vec![0f32; dims];
    for c in centres {
        for (m, &x) in mean.iter_mut().zip(c.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= centres.len() as f32;
    }
    normalise(&mut mean);
    centres
        .iter()
        .map(|c| {
            let mut v: Vec<f32> = c
                .iter()
                .zip(mean.iter())
                .map(|(&ci, &mi)| ci * (1.0 - crowding) + mi * crowding)
                .collect();
            normalise(&mut v);
            v
        })
        .collect()
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
    fn drift_stream_preserves_total_point_count() {
        let cfg = DriftStreamConfig {
            n_points: 600,
            ..DriftStreamConfig::default()
        };
        let (ds, switch_at) = StreamDataset::generate_drift(&cfg);
        assert_eq!(ds.points.len(), 600);
        assert!(switch_at > 0 && switch_at < 600);
    }

    #[test]
    fn regime_b_centres_are_more_crowded_than_regime_a() {
        // Mean pairwise cosine similarity among regime B's centres must be
        // strictly higher than regime A's — that is the whole point of
        // `regime_b_crowding`, and the property that makes a tau tuned on
        // regime A miscalibrated for regime B.
        let cfg = DriftStreamConfig::default();
        let (centres_a, centres_b) = StreamDataset::drift_centres(&cfg);

        let mean_pairwise_sim = |centres: &[Vec<f32>]| -> f32 {
            let mut sum = 0f32;
            let mut n = 0usize;
            for i in 0..centres.len() {
                for j in (i + 1)..centres.len() {
                    sum += crate::cosine_sim(&centres[i], &centres[j]);
                    n += 1;
                }
            }
            sum / n as f32
        };

        let sim_a = mean_pairwise_sim(&centres_a);
        let sim_b = mean_pairwise_sim(&centres_b);
        assert!(
            sim_b > sim_a,
            "regime B centres should be more mutually similar: sim_a={sim_a}, sim_b={sim_b}"
        );
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
