//! Synthetic dataset generation for namespace-merge benchmarks.
//!
//! Generates clustered namespaces so that mincut routing has a meaningful
//! structural advantage: two groups of namespaces (A and B) are semantically
//! distant from each other and from a third isolated group (C). Queries
//! targeted at group A should route to only A's namespaces.

/// A single namespace: a collection of normalised f32 vectors with a
/// precomputed centroid.
#[derive(Clone)]
pub struct Namespace {
    pub id: usize,
    pub label: String,
    /// Row-major: `vectors[i * dims .. (i+1) * dims]`.
    pub vectors: Vec<f32>,
    pub n: usize,
    pub dims: usize,
    pub centroid: Vec<f32>,
}

impl Namespace {
    pub fn new(id: usize, label: String, vectors: Vec<f32>, n: usize, dims: usize) -> Self {
        let centroid = compute_centroid(&vectors, n, dims);
        Namespace {
            id,
            label,
            vectors,
            n,
            dims,
            centroid,
        }
    }

    pub fn vector(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dims..(i + 1) * self.dims]
    }
}

/// Whole dataset: multiple namespaces + flat ground-truth index.
pub struct Dataset {
    pub namespaces: Vec<Namespace>,
    pub dims: usize,
    /// Global id = namespace_index * per_ns + local_index.
    pub per_ns: usize,
}

/// Parameters controlling the synthetic dataset.
pub struct DatasetConfig {
    /// Vectors per namespace.
    pub per_ns: usize,
    /// Vector dimensions.
    pub dims: usize,
    /// RNG seed.
    pub seed: u64,
    /// Noise magnitude around the cluster centre.
    pub noise: f32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            per_ns: 500,
            dims: 64,
            seed: 0xDEAD_BEEF_CAFE,
            noise: 0.30,
        }
    }
}

impl Dataset {
    /// Build a 5-namespace dataset with two semantic clusters:
    ///
    /// - **Group A** (NS 0, 1): centred around `(1, 0, 0, …)`.
    /// - **Group B** (NS 2, 3): centred around `(0, 1, 0, …)`.
    /// - **Group C** (NS 4): centred around `(-1, -1, 0, …)` (normalised).
    ///
    /// All vectors are L2-normalised so cosine = dot product.
    pub fn generate(cfg: &DatasetConfig) -> Self {
        let mut rng = Lcg64(cfg.seed);

        let centres: Vec<Vec<f32>> = vec![
            make_centre(cfg.dims, 0, &[1.0, 0.0]),
            make_centre(cfg.dims, 0, &[1.0, 0.2]),
            make_centre(cfg.dims, 1, &[0.0, 1.0]),
            make_centre(cfg.dims, 1, &[0.2, 1.0]),
            make_centre(cfg.dims, 2, &[-0.7, -0.7]),
        ];
        let labels = ["ns-A0", "ns-A1", "ns-B0", "ns-B1", "ns-C"];

        let mut namespaces = Vec::with_capacity(5);
        for (ns_idx, (centre, label)) in centres.iter().zip(labels.iter()).enumerate() {
            let mut vecs = Vec::with_capacity(cfg.per_ns * cfg.dims);
            for _ in 0..cfg.per_ns {
                let v = sample_around(&mut rng, centre, cfg.noise);
                vecs.extend_from_slice(&v);
            }
            namespaces.push(Namespace::new(
                ns_idx,
                label.to_string(),
                vecs,
                cfg.per_ns,
                cfg.dims,
            ));
        }

        Dataset {
            namespaces,
            dims: cfg.dims,
            per_ns: cfg.per_ns,
        }
    }

    pub fn total_vecs(&self) -> usize {
        self.namespaces.len() * self.per_ns
    }

    /// Generate `n_queries` queries targeted at Group A (NS 0, 1).
    /// Returns normalised query vectors; ground truth is all hits from NS 0+1.
    pub fn group_a_queries(&self, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg64(seed ^ 0x1234);
        // centre of group A
        let centre = make_centre(self.dims, 0, &[1.0, 0.0]);
        (0..n)
            .map(|_| sample_around(&mut rng, &centre, 0.20))
            .collect()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn compute_centroid(vecs: &[f32], n: usize, dims: usize) -> Vec<f32> {
    let mut c = vec![0f32; dims];
    for i in 0..n {
        for d in 0..dims {
            c[d] += vecs[i * dims + d];
        }
    }
    let scale = 1.0 / n as f32;
    for x in &mut c {
        *x *= scale;
    }
    normalise(&mut c);
    c
}

/// Build a unit-norm centre vector. `axis` selects which principal dimension
/// is dominant; `weights` provides the two leading coefficients.
fn make_centre(dims: usize, axis: usize, weights: &[f32]) -> Vec<f32> {
    let mut v = vec![0f32; dims];
    // set principal dimensions
    for (i, &w) in weights.iter().enumerate() {
        let d = (axis * 4 + i).min(dims - 1);
        v[d] = w;
    }
    normalise(&mut v);
    v
}

/// Sample a vector near `centre` with Gaussian noise `sigma`, then normalise.
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

// ─── minimal LCG + Box-Muller RNG (no external deps) ─────────────────────────

pub struct Lcg64(pub u64);

impl Lcg64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform float in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Standard normal via Box-Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        r * theta.cos()
    }
}
