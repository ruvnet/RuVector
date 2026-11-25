//! Sheaf-residual-gated per-vector quantization bit-width for agent memory.
//!
//! # Hypothesis under test
//!
//! Agent memories are not equally safe to compress. A memory that sits well
//! inside a stable, well-integrated neighborhood should tolerate aggressive
//! quantization; a memory that *disagrees* with its neighborhood — the
//! sheaf-Laplacian "contradiction energy" of ADR-CE-001/ADR-CE-011 — should
//! keep more precision, because it is exactly the kind of surprising,
//! boundary-case memory whose fine detail later retrieval is most likely to
//! need.
//!
//! This crate tests that hypothesis honestly by racing three bit-allocation
//! policies against each other at an **identical, matched byte budget** (so
//! none can win simply by spending more memory — the failure mode that sank
//! the ADR-305 mutual-kNN experiment):
//!
//! - [`Policy::Random`] — baseline. A fixed fraction of vectors gets extra
//!   bits, chosen uniformly at random.
//! - [`Policy::ResidualGated`] — candidate A, the novel signal. The same
//!   fraction gets extra bits, chosen by highest per-node sheaf residual
//!   energy (see [`node_residual_energies`]).
//! - [`Policy::DensityGated`] — candidate B, the control. The same fraction
//!   gets extra bits, chosen by local-density/outlier score (see
//!   [`node_density_outlier_scores`]) — the "plain local-density score,
//!   decoupled from min-cut" that ADR-353's own open questions flagged as
//!   untested.
//!
//! ## Restriction maps: an honest simplification
//!
//! ADR-CE-001 defines the residual on edge `e = (u, v)` as
//! `r_e = ρ_u(x_u) − ρ_v(x_v)` for independent per-endpoint affine
//! restriction maps `ρ_u, ρ_v`. This crate uses the shared special case
//! `ρ_u = ρ_v = A_e` (one random projection per edge, not per endpoint), so
//! `r_e = A_e(x_u − x_v)`. That is a real, legitimate sheaf consistency
//! constraint (edges assert agreement in a shared subspace), but it is
//! *not* the fully general form, and — because `A_e` is an unstructured
//! random Gaussian projection — by Johnson–Lindenstrauss concentration the
//! aggregate residual energy over many edges is expected to correlate with
//! plain squared neighbor distance. The benchmark binary reports the actual
//! measured correlation between the residual score and the density score
//! ([`pearson`]) so this is not asserted, it is measured.
//!
//! See `docs/adr/ADR-362-residual-adaptive-quantization.md` for the full
//! pre-registered hypothesis, acceptance thresholds, and prior-art summary
//! (ADR-305, ADR-353).

use std::collections::HashSet;

// ─── Deterministic RNG (no external dependency, mirrors ruvector-recall-bounded) ───

/// Minimal splitmix64-style deterministic generator. Not cryptographic;
/// chosen only for reproducible synthetic data and reproducible shuffles.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Standard normal via Box-Muller.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound.max(1)
    }

    /// Deterministic Fisher-Yates shuffle of `0..n`.
    pub fn shuffled_indices(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.next_index(i + 1);
            idx.swap(i, j);
        }
        idx
    }
}

// ─── Core types ──────────────────────────────────────────────────────────────

pub type Vector = Vec<f32>;

/// One synthetic memory record. `is_boundary` is ground truth, known only to
/// the dataset generator and the evaluation code, never to the policies
/// under test: a boundary vector was planted at the midpoint of two randomly
/// chosen cluster centers, i.e. it is deliberately ambiguous between two
/// "narratives" — the intended stand-in for a contradictory agent memory.
#[derive(Clone)]
pub struct Entry {
    pub id: u32,
    pub vec: Vector,
    pub is_boundary: bool,
}

pub struct DatasetConfig {
    pub n: usize,
    pub dim: usize,
    pub n_clusters: usize,
    pub cluster_std: f32,
    pub center_scale: f32,
    pub boundary_frac: f32,
    pub seed: u64,
}

pub fn gen_dataset(cfg: &DatasetConfig) -> Vec<Entry> {
    assert!(
        cfg.n_clusters >= 2,
        "need >=2 clusters to plant boundary vectors"
    );
    let mut rng = Lcg(cfg.seed);
    let centers: Vec<Vector> = (0..cfg.n_clusters)
        .map(|_| {
            let mut v: Vector = (0..cfg.dim).map(|_| rng.next_gaussian()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut v {
                *x = *x / norm * cfg.center_scale;
            }
            v
        })
        .collect();

    let n_boundary = ((cfg.n as f32) * cfg.boundary_frac).round() as usize;
    let mut entries = Vec::with_capacity(cfg.n);
    for i in 0..cfg.n {
        let is_boundary = i < n_boundary;
        let vec = if is_boundary {
            let a = rng.next_index(cfg.n_clusters);
            let mut b = rng.next_index(cfg.n_clusters);
            while b == a {
                b = rng.next_index(cfg.n_clusters);
            }
            centers[a]
                .iter()
                .zip(centers[b].iter())
                .map(|(x, y)| 0.5 * (x + y) + rng.next_gaussian() * cfg.cluster_std * 0.5)
                .collect()
        } else {
            let c = rng.next_index(cfg.n_clusters);
            centers[c]
                .iter()
                .map(|x| x + rng.next_gaussian() * cfg.cluster_std)
                .collect()
        };
        entries.push(Entry {
            id: i as u32,
            vec,
            is_boundary,
        });
    }
    // Shuffle so boundary vectors are not contiguous ids (id order must not
    // leak the label to anything downstream).
    let order = rng.shuffled_indices(cfg.n);
    let mut shuffled: Vec<Entry> = order.iter().map(|&i| entries[i].clone()).collect();
    for (new_id, e) in shuffled.iter_mut().enumerate() {
        e.id = new_id as u32;
    }
    shuffled
}

// ─── Similarity / distance ──────────────────────────────────────────────────

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ─── kNN consistency graph ──────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub weight: f32,
}

/// Brute-force symmetric kNN graph by cosine similarity. O(n^2 log n); fine
/// at the corpus sizes used here (see benchmark defaults).
pub fn build_knn_graph(vectors: &[Vector], k: usize) -> Vec<Edge> {
    let n = vectors.len();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut edges = Vec::new();
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&vectors[i], &vectors[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, sim) in sims.iter().take(k) {
            let key = if i < j { (i, j) } else { (j, i) };
            if seen.insert(key) {
                edges.push(Edge {
                    u: key.0,
                    v: key.1,
                    weight: sim.max(0.0) + 1e-3,
                });
            }
        }
    }
    edges
}

// ─── Sheaf-residual scoring (ADR-CE-001 shared-restriction-map special case) ───

/// Deterministic random restriction map A_e for one edge: `edge_dim x dim`,
/// iid N(0,1) scaled by `1/sqrt(edge_dim)`.
pub fn edge_projection(seed: u64, edge_idx: usize, edge_dim: usize, dim: usize) -> Vec<Vector> {
    let mut rng = Lcg(seed ^ (edge_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let scale = 1.0 / (edge_dim as f32).sqrt();
    (0..edge_dim)
        .map(|_| (0..dim).map(|_| rng.next_gaussian() * scale).collect())
        .collect()
}

fn apply_projection(proj: &[Vector], v: &[f32]) -> Vector {
    proj.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// `energy_e = weight * |A_e (x_u - x_v)|^2`
pub fn edge_residual_energy(xu: &[f32], xv: &[f32], proj: &[Vector], weight: f32) -> f32 {
    let diff: Vector = xu.iter().zip(xv.iter()).map(|(a, b)| a - b).collect();
    let r = apply_projection(proj, &diff);
    weight * r.iter().map(|x| x * x).sum::<f32>()
}

/// Per-node residual energy: mean incident-edge residual energy (the
/// per-node aggregation `prime-radiant::CoherenceEngine` does not expose
/// directly — see ADR-362 — built here from `compute_edge_energy`-equivalent
/// building blocks over `edges_incident_to`-equivalent adjacency).
pub fn node_residual_energies(
    n: usize,
    edges: &[Edge],
    vectors: &[Vector],
    seed: u64,
    edge_dim: usize,
    dim: usize,
) -> Vec<f32> {
    let mut sums = vec![0f32; n];
    let mut counts = vec![0u32; n];
    for (idx, e) in edges.iter().enumerate() {
        let proj = edge_projection(seed, idx, edge_dim, dim);
        let energy = edge_residual_energy(&vectors[e.u], &vectors[e.v], &proj, e.weight);
        sums[e.u] += energy;
        counts[e.u] += 1;
        sums[e.v] += energy;
        counts[e.v] += 1;
    }
    (0..n)
        .map(|i| {
            if counts[i] > 0 {
                sums[i] / counts[i] as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Local-density / outlier score: mean squared Euclidean distance to graph
/// neighbors. Higher = more isolated / lower local density. This is the
/// "plain local-density score, decoupled from min-cut" ADR-353 flagged as
/// untested.
pub fn node_density_outlier_scores(n: usize, edges: &[Edge], vectors: &[Vector]) -> Vec<f32> {
    let mut sums = vec![0f32; n];
    let mut counts = vec![0u32; n];
    for e in edges {
        let d = squared_euclidean(&vectors[e.u], &vectors[e.v]);
        sums[e.u] += d;
        counts[e.u] += 1;
        sums[e.v] += d;
        counts[e.v] += 1;
    }
    (0..n)
        .map(|i| {
            if counts[i] > 0 {
                sums[i] / counts[i] as f32
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Scalar quantization ────────────────────────────────────────────────────

#[derive(Clone)]
pub struct QuantizedVector {
    pub bits: u8,
    pub dim: usize,
    pub min: f32,
    pub scale: f32,
    pub codes: Vec<u8>,
}

/// Per-vector min-max uniform scalar quantization at `bits` (1..=8).
pub fn quantize_vector(v: &[f32], bits: u8) -> QuantizedVector {
    assert!((1..=8).contains(&bits), "bits must be in 1..=8");
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let levels = (1u32 << bits) - 1;
    let scale = if max > min {
        (max - min) / levels as f32
    } else {
        1.0
    };
    let codes: Vec<u8> = v
        .iter()
        .map(|&x| (((x - min) / scale).round().clamp(0.0, levels as f32)) as u8)
        .collect();
    QuantizedVector {
        bits,
        dim: v.len(),
        min,
        scale,
        codes,
    }
}

pub fn dequantize(qv: &QuantizedVector) -> Vector {
    qv.codes
        .iter()
        .map(|&c| qv.min + c as f32 * qv.scale)
        .collect()
}

/// Bytes a production bit-packed layout would use: `ceil(dim*bits/8)` for
/// codes plus 8 bytes (two f32) for the per-vector min/scale header. This
/// crate stores codes unpacked (one byte each) for implementation clarity;
/// memory accounting always uses this packed estimate, never the unpacked
/// `Vec<u8>` length, so the reported numbers reflect a real deployment.
pub fn packed_bytes(qv: &QuantizedVector) -> usize {
    (qv.dim * qv.bits as usize).div_ceil(8) + 8
}

// ─── Bit allocation policies ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Random,
    ResidualGated,
    DensityGated,
}

impl Policy {
    pub fn name(&self) -> &'static str {
        match self {
            Policy::Random => "random",
            Policy::ResidualGated => "residual_gated",
            Policy::DensityGated => "density_gated",
        }
    }
}

/// Assigns `hi_bits` to the top `hi_fraction` of vectors by `scores`
/// (descending) for the gated policies, or to a uniformly random
/// `hi_fraction` subset for [`Policy::Random`] — so every policy spends an
/// identical total byte budget, differing only in *which* vectors are
/// chosen.
pub fn allocate_bits(
    scores: &[f32],
    hi_bits: u8,
    lo_bits: u8,
    hi_fraction: f32,
    policy: Policy,
    seed: u64,
) -> Vec<u8> {
    let n = scores.len();
    let n_hi = ((n as f32) * hi_fraction).round() as usize;
    let mut bits = vec![lo_bits; n];
    match policy {
        Policy::Random => {
            let mut rng = Lcg(seed);
            let idx = rng.shuffled_indices(n);
            for &i in idx.iter().take(n_hi) {
                bits[i] = hi_bits;
            }
        }
        Policy::ResidualGated | Policy::DensityGated => {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            for &i in idx.iter().take(n_hi) {
                bits[i] = hi_bits;
            }
        }
    }
    bits
}

// ─── Retrieval evaluation ───────────────────────────────────────────────────

pub fn topk_ids(query_idx: usize, vectors: &[Vector], k: usize) -> Vec<u32> {
    let q = &vectors[query_idx];
    let mut sims: Vec<(usize, f32)> = (0..vectors.len())
        .filter(|&j| j != query_idx)
        .map(|j| (j, cosine_similarity(q, &vectors[j])))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.into_iter().take(k).map(|(j, _)| j as u32).collect()
}

pub fn recall_at_k(approx: &[u32], exact: &[u32]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let exact_set: HashSet<u32> = exact.iter().cloned().collect();
    let hits = approx.iter().filter(|id| exact_set.contains(id)).count();
    hits as f32 / exact.len() as f32
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// AUC via the Mann-Whitney U / rank-sum identity: `U / (n_pos * n_neg)`.
/// Ties are broken by stable sort order (dataset scores are continuous
/// floats, so exact ties are negligible).
pub fn auc(scores: &[f32], labels: &[bool]) -> f32 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    // Average-rank tie handling: a run of equal scores all receive the mean
    // of the ranks the run occupies, so fully-tied inputs score AUC = 0.5
    // exactly rather than being biased by sort order.
    let mut rank = vec![0f32; scores.len()];
    let n = idx.len();
    let mut start = 0usize;
    while start < n {
        let mut end = start + 1;
        while end < n && scores[idx[end]] == scores[idx[start]] {
            end += 1;
        }
        let avg_rank = (start as f32 + end as f32 - 1.0) / 2.0;
        for &i in &idx[start..end] {
            rank[i] = avg_rank;
        }
        start = end;
    }
    let n_pos = labels.iter().filter(|&&l| l).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let pos_rank_sum: f32 = (0..scores.len())
        .filter(|&i| labels[i])
        .map(|i| rank[i])
        .sum();
    let u = pos_rank_sum - n_pos * (n_pos - 1.0) / 2.0;
    u / (n_pos * n_neg)
}

pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0f32;
    let mut va = 0f32;
    let mut vb = 0f32;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

// ─── Witness log (hash-chained, keyless — mirrors the ADR-134 ledger shape) ───

#[derive(Clone)]
pub struct WitnessRecord {
    pub vector_id: u32,
    pub policy: &'static str,
    pub score: f32,
    pub bits: u8,
    pub prev_hash: u64,
    pub hash: u64,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const WITNESS_GENESIS: u64 = 0xcbf29ce484222325;

fn witness_hash(prev: u64, vector_id: u32, policy: &str, score: f32, bits: u8) -> u64 {
    let mut buf = Vec::with_capacity(8 + 4 + policy.len() + 4 + 1);
    buf.extend_from_slice(&prev.to_le_bytes());
    buf.extend_from_slice(&vector_id.to_le_bytes());
    buf.extend_from_slice(policy.as_bytes());
    buf.extend_from_slice(&score.to_le_bytes());
    buf.push(bits);
    fnv1a(&buf)
}

/// Append-only, hash-chained record of every compression decision: *which*
/// vector, *why* (the score that drove it), and *what* bit-width was chosen
/// — the auditability requirement from ADR-362 / ADR-134's witness ledger
/// shape. A production integration would route this through
/// `ruvector-agent-memory`'s `WitnessSink`/`SignedWitnessSink` (ADR-347) for
/// Ed25519 signing; this crate implements the unsigned hash chain only.
pub struct WitnessLog {
    pub records: Vec<WitnessRecord>,
    head: u64,
}

impl Default for WitnessLog {
    fn default() -> Self {
        Self::new()
    }
}

impl WitnessLog {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            head: WITNESS_GENESIS,
        }
    }

    pub fn append(&mut self, vector_id: u32, policy: &'static str, score: f32, bits: u8) {
        let hash = witness_hash(self.head, vector_id, policy, score, bits);
        self.records.push(WitnessRecord {
            vector_id,
            policy,
            score,
            bits,
            prev_hash: self.head,
            hash,
        });
        self.head = hash;
    }

    /// Recomputes the chain from genesis and checks every link.
    pub fn verify(&self) -> bool {
        let mut head = WITNESS_GENESIS;
        for r in &self.records {
            if r.prev_hash != head {
                return false;
            }
            let hash = witness_hash(head, r.vector_id, r.policy, r.score, r.bits);
            if hash != r.hash {
                return false;
            }
            head = hash;
        }
        true
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(seed: u64) -> DatasetConfig {
        DatasetConfig {
            n: 200,
            dim: 16,
            n_clusters: 4,
            cluster_std: 0.08,
            center_scale: 2.0,
            boundary_frac: 0.1,
            seed,
        }
    }

    #[test]
    fn dataset_is_deterministic() {
        let a = gen_dataset(&small_cfg(7));
        let b = gen_dataset(&small_cfg(7));
        assert_eq!(a.len(), b.len());
        for (ea, eb) in a.iter().zip(b.iter()) {
            assert_eq!(ea.is_boundary, eb.is_boundary);
            assert_eq!(ea.vec, eb.vec);
        }
    }

    #[test]
    fn dataset_has_expected_boundary_count() {
        let data = gen_dataset(&small_cfg(1));
        let n_boundary = data.iter().filter(|e| e.is_boundary).count();
        assert_eq!(n_boundary, 20); // 10% of 200
    }

    #[test]
    fn quantize_roundtrip_is_bounded_error() {
        let v: Vector = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        for bits in [2u8, 4, 8] {
            let q = quantize_vector(&v, bits);
            let dq = dequantize(&q);
            let max_err = v
                .iter()
                .zip(dq.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // Max error can never exceed one quantization step.
            assert!(
                max_err <= q.scale + 1e-4,
                "bits={bits} max_err={max_err} scale={}",
                q.scale
            );
        }
    }

    #[test]
    fn higher_bits_never_worse_reconstruction() {
        let v: Vector = (0..32).map(|i| (i as f32).sin()).collect();
        let q2 = quantize_vector(&v, 2);
        let q8 = quantize_vector(&v, 8);
        let err = |q: &QuantizedVector| -> f32 {
            let dq = dequantize(q);
            squared_euclidean(&v, &dq)
        };
        assert!(err(&q8) <= err(&q2));
    }

    #[test]
    fn packed_bytes_matches_bit_width() {
        let v = vec![0.0f32; 64];
        let q2 = quantize_vector(&v, 2);
        let q8 = quantize_vector(&v, 8);
        assert_eq!(packed_bytes(&q2), (64usize * 2).div_ceil(8) + 8);
        assert_eq!(packed_bytes(&q8), (64usize * 8).div_ceil(8) + 8);
        assert!(packed_bytes(&q2) < packed_bytes(&q8));
    }

    #[test]
    fn allocate_bits_matches_budget_across_policies() {
        let n = 1000;
        let scores_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let scores_b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let hi = 8u8;
        let lo = 2u8;
        let frac = 0.25;
        let r = allocate_bits(&scores_a, hi, lo, frac, Policy::Random, 42);
        let a = allocate_bits(&scores_a, hi, lo, frac, Policy::ResidualGated, 42);
        let b = allocate_bits(&scores_b, hi, lo, frac, Policy::DensityGated, 42);
        for bits in [&r, &a, &b] {
            let n_hi = bits.iter().filter(|&&x| x == hi).count();
            assert_eq!(n_hi, 250);
            let total: usize = bits.iter().map(|&x| x as usize).sum();
            assert_eq!(total, 250 * hi as usize + 750 * lo as usize);
        }
    }

    #[test]
    fn residual_gated_selects_highest_score_vectors() {
        let scores = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.05, 0.6, 0.15, 0.95];
        let bits = allocate_bits(&scores, 8, 2, 0.3, Policy::ResidualGated, 0);
        // Top 3 scores are indices 9 (0.95), 1 (0.9), 3 (0.8).
        for i in [9usize, 1, 3] {
            assert_eq!(bits[i], 8);
        }
        assert_eq!(bits.iter().filter(|&&b| b == 8).count(), 3);
    }

    #[test]
    fn edge_residual_energy_is_zero_for_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = edge_projection(1, 0, 4, 4);
        assert_eq!(edge_residual_energy(&v, &v, &proj, 1.0), 0.0);
    }

    #[test]
    fn edge_residual_energy_grows_with_disagreement() {
        let a = vec![0.0f32; 8];
        let near = vec![0.05f32; 8];
        let far = vec![5.0f32; 8];
        let proj = edge_projection(1, 0, 8, 8);
        let e_near = edge_residual_energy(&a, &near, &proj, 1.0);
        let e_far = edge_residual_energy(&a, &far, &proj, 1.0);
        assert!(e_far > e_near);
    }

    #[test]
    fn knn_graph_is_symmetric_and_deduplicated() {
        let data = gen_dataset(&small_cfg(3));
        let vectors: Vec<Vector> = data.iter().map(|e| e.vec.clone()).collect();
        let edges = build_knn_graph(&vectors, 5);
        let mut seen = HashSet::new();
        for e in &edges {
            let key = if e.u < e.v { (e.u, e.v) } else { (e.v, e.u) };
            assert!(seen.insert(key), "duplicate edge {:?}", key);
            assert_ne!(e.u, e.v);
        }
    }

    #[test]
    fn boundary_vectors_tend_to_have_higher_residual_energy() {
        // Not a hard guarantee for every seed (that's the whole point of
        // running a real benchmark), but on a reasonably sized, well
        // separated synthetic corpus the signal should be detectable above
        // chance (AUC > 0.5) often enough that a single seed sanity check
        // is a meaningful regression guard, not a flaky assertion.
        let cfg = DatasetConfig {
            n: 600,
            dim: 32,
            n_clusters: 6,
            cluster_std: 0.1,
            center_scale: 3.0,
            boundary_frac: 0.1,
            seed: 123,
        };
        let data = gen_dataset(&cfg);
        let vectors: Vec<Vector> = data.iter().map(|e| e.vec.clone()).collect();
        let labels: Vec<bool> = data.iter().map(|e| e.is_boundary).collect();
        let edges = build_knn_graph(&vectors, 10);
        let residual = node_residual_energies(data.len(), &edges, &vectors, 999, 12, cfg.dim);
        let a = auc(&residual, &labels);
        assert!(
            a > 0.5,
            "residual AUC was {a}, expected boundary vectors to score higher than chance"
        );
    }

    #[test]
    fn witness_log_detects_tampering() {
        let mut log = WitnessLog::new();
        log.append(1, "residual_gated", 0.42, 8);
        log.append(2, "residual_gated", 0.10, 2);
        log.append(3, "residual_gated", 0.37, 4);
        assert!(log.verify());

        log.records[1].score = 0.99; // tamper after the fact
        assert!(!log.verify());
    }

    #[test]
    fn auc_is_half_for_uninformative_scores() {
        let scores = vec![1.0f32; 100];
        let labels: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        assert!((auc(&scores, &labels) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pearson_perfect_positive_correlation() {
        let a: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..50).map(|i| 2.0 * i as f32 + 1.0).collect();
        assert!((pearson(&a, &b) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn recall_at_k_counts_overlap() {
        let exact = vec![1u32, 2, 3, 4, 5];
        let approx = vec![1u32, 2, 9, 4, 10];
        assert!((recall_at_k(&approx, &exact) - 0.6).abs() < 1e-6);
    }
}
