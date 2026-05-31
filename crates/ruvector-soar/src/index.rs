//! SOAR-IVF index: IVF with Orthogonality-Amplified Residual spilling.
//!
//! Reference: Sun et al., "SOAR: Improved Indexing for Approximate Nearest
//! Neighbor Search", NeurIPS 2023. arXiv:2404.00774.
//!
//! ## Algorithm
//!
//! 1. Train k-means on corpus → `nlist` centroids.
//! 2. Assign each vector to its **primary** centroid.
//! 3. (SOAR only) Assign a **secondary** centroid to each vector via the
//!    orthogonality-amplified loss:
//!        score(c') = ‖r'‖² + λ · (r·r')² / ‖r‖²
//!    where r = v − centroid[primary] and r' = v − c'.
//!    Penalising r'∥r means the secondary cluster compensates for exactly
//!    the query directions that the primary cluster handles poorly.
//! 4. Build PQ codebook, encode all vectors.
//! 5. At query time: probe `nprobe` closest centroids (checking both primary
//!    and secondary inverted lists), deduplicate, score via ADC, rerank.

use crate::error::{Result, SoarError};
use crate::kmeans::{dot, l2_sq, Kmeans};
use crate::pq::ProductQuantizer;

/// Single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Index variant selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexKind {
    /// Brute-force flat scan (exact baseline).
    Flat,
    /// IVF with ADC, no secondary spilling.
    IvfPq,
    /// IVF with ADC + SOAR secondary assignments.
    SoarIvfPq,
}

/// Build-time configuration.
#[derive(Debug, Clone)]
pub struct SoarConfig {
    /// Number of IVF clusters.
    pub nlist: usize,
    /// Clusters probed at query time.
    pub nprobe: usize,
    /// SOAR orthogonality penalty coefficient (paper uses λ = 1.0).
    pub lambda: f32,
    /// Number of secondary-assignment candidates to evaluate (paper uses 10).
    pub n_secondary_candidates: usize,
    /// PQ subspaces (must divide `dim`).
    pub m_pq: usize,
    /// K-means max iterations.
    pub kmeans_iter: usize,
    /// Index type.
    pub kind: IndexKind,
}

impl Default for SoarConfig {
    fn default() -> Self {
        Self {
            nlist: 64,
            nprobe: 8,
            lambda: 1.0,
            n_secondary_candidates: 10,
            m_pq: 8,
            kmeans_iter: 20,
            kind: IndexKind::SoarIvfPq,
        }
    }
}

pub struct SoarIndex {
    config: SoarConfig,
    dim: usize,
    n: usize,
    /// Original f32 vectors (for flat baseline and final reranking).
    vectors: Vec<Vec<f32>>,
    /// K-means model.
    kmeans: Option<Kmeans>,
    /// Primary inverted lists: primary_lists[centroid] = [vector_id…]
    primary_lists: Vec<Vec<u32>>,
    /// Secondary inverted lists (SOAR only): secondary_lists[centroid] = [vector_id…]
    secondary_lists: Vec<Vec<u32>>,
    /// PQ codes, one per vector.
    pq_codes: Vec<Vec<u8>>,
    /// Trained PQ.
    pq: ProductQuantizer,
}

impl SoarIndex {
    /// Build the index from `vectors` using `config`.
    pub fn build(vectors: Vec<Vec<f32>>, config: SoarConfig) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SoarError::Empty);
        }
        let n = vectors.len();
        let dim = vectors[0].len();
        if dim == 0 {
            return Err(SoarError::InvalidConfig("zero-dimensional vectors".into()));
        }
        if config.nlist == 0 || config.nlist > n {
            return Err(SoarError::InvalidConfig(format!(
                "nlist={} must be in 1..={n}",
                config.nlist
            )));
        }

        let mut primary_lists = vec![Vec::new(); config.nlist];
        let mut secondary_lists = vec![Vec::new(); config.nlist];
        let mut pq_codes = Vec::with_capacity(n);

        let (kmeans, pq) = match config.kind {
            IndexKind::Flat => {
                // No clustering or quantisation for brute-force.
                (None, ProductQuantizer::new(dim, config.m_pq)?)
            }
            _ => {
                // Train k-means
                let km = Kmeans::train(&vectors, config.nlist, config.kmeans_iter, 42)?;
                // Primary assignment
                for (id, v) in vectors.iter().enumerate() {
                    let primary = km.assign(v);
                    primary_lists[primary].push(id as u32);
                }
                // SOAR secondary assignment
                if config.kind == IndexKind::SoarIvfPq {
                    soar_secondary_assign(
                        &vectors,
                        &km,
                        &config,
                        &primary_lists,
                        &mut secondary_lists,
                    );
                }
                // Train PQ on a sample of vectors
                let mut pq = ProductQuantizer::new(dim, config.m_pq)?;
                pq.train(&vectors, 20, 99)?;
                // Encode all vectors
                for v in &vectors {
                    pq_codes.push(pq.encode(v));
                }
                (Some(km), pq)
            }
        };

        Ok(Self {
            config,
            dim,
            n,
            vectors,
            kmeans,
            primary_lists,
            secondary_lists,
            pq_codes,
            pq,
        })
    }

    /// Approximate k-NN search. Returns results sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(SoarError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        match self.config.kind {
            IndexKind::Flat => self.flat_search(query, k),
            IndexKind::IvfPq => self.ivf_search(query, k, false),
            IndexKind::SoarIvfPq => self.ivf_search(query, k, true),
        }
    }

    // ── flat exact baseline ───────────────────────────────────────────────────

    fn flat_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut dists: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_sq(query, v)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(dists
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    // ── IVF search (with or without SOAR secondary lists) ────────────────────

    fn ivf_search(&self, query: &[f32], k: usize, use_secondary: bool) -> Result<Vec<SearchResult>> {
        let km = self.kmeans.as_ref().ok_or(SoarError::NotTrained)?;
        let nprobe = self.config.nprobe.min(self.config.nlist);

        // Find the nprobe closest centroids.
        let probes = km.top_k(query, nprobe);

        // Precompute ADC lookup table once per query.
        let table = self.pq.distance_table(query);

        // Collect candidates from primary (and optionally secondary) lists.
        // Use a bitset-style seen array for O(1) dedup.
        let mut seen = vec![false; self.n];
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        for (centroid_id, _) in &probes {
            for &vid in &self.primary_lists[*centroid_id] {
                if !seen[vid as usize] {
                    seen[vid as usize] = true;
                    let dist = self.pq.adc_distance(&self.pq_codes[vid as usize], &table);
                    candidates.push((vid, dist));
                }
            }
            if use_secondary {
                for &vid in &self.secondary_lists[*centroid_id] {
                    if !seen[vid as usize] {
                        seen[vid as usize] = true;
                        let dist = self.pq.adc_distance(&self.pq_codes[vid as usize], &table);
                        candidates.push((vid, dist));
                    }
                }
            }
        }

        // Partial sort: keep top-k by ADC estimate, then exact rerank.
        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let rerank_n = (k * 4).min(candidates.len());

        let mut results: Vec<SearchResult> = candidates[..rerank_n]
            .iter()
            .map(|&(vid, _)| {
                let exact = l2_sq(query, &self.vectors[vid as usize]);
                SearchResult { id: vid as usize, distance: exact }
            })
            .collect();
        results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        Ok(results)
    }

    /// Total memory used by inverted lists + PQ codes (bytes, approximate).
    pub fn index_bytes(&self) -> usize {
        let lists: usize = self
            .primary_lists
            .iter()
            .chain(self.secondary_lists.iter())
            .map(|l| l.len() * 4)
            .sum();
        let codes: usize = self.pq_codes.iter().map(|c| c.len()).sum();
        let centroids: usize = self
            .kmeans
            .as_ref()
            .map(|km| km.centroids.len() * km.dim * 4)
            .unwrap_or(0);
        lists + codes + centroids
    }

    pub fn len(&self) -> usize {
        self.n
    }
}

// ── SOAR secondary assignment ────────────────────────────────────────────────

fn soar_secondary_assign(
    vectors: &[Vec<f32>],
    km: &Kmeans,
    config: &SoarConfig,
    primary_lists: &[Vec<u32>],
    secondary_lists: &mut [Vec<u32>],
) {
    // Build reverse map: vector_id → primary centroid id
    let mut primary_of = vec![0usize; vectors.len()];
    for (c, list) in primary_lists.iter().enumerate() {
        for &vid in list {
            primary_of[vid as usize] = c;
        }
    }

    let n_candidates = config.n_secondary_candidates.min(km.centroids.len().saturating_sub(1));
    if n_candidates == 0 {
        return;
    }

    for (vid, v) in vectors.iter().enumerate() {
        let primary = primary_of[vid];
        let cp = &km.centroids[primary];

        // Primary residual r = v − cp
        let r: Vec<f32> = v.iter().zip(cp.iter()).map(|(a, b)| a - b).collect();
        let r_norm_sq = dot(&r, &r);

        // Probe up to n_candidates+1 closest centroids, skip the primary.
        let candidates = km.top_k(v, n_candidates + 1);

        let secondary = candidates
            .iter()
            .filter(|(c, _)| *c != primary)
            .map(|(c, _)| {
                let c_centroid = &km.centroids[*c];
                // Secondary residual r' = v − c'
                let r_prime: Vec<f32> =
                    v.iter().zip(c_centroid.iter()).map(|(a, b)| a - b).collect();
                let r_prime_norm_sq = dot(&r_prime, &r_prime);

                // Orthogonality-amplified loss:
                //   score = ‖r'‖² + λ · (r·r')² / ‖r‖²
                // If r_norm_sq ≈ 0, skip the penalty (vector is at its centroid).
                let penalty = if r_norm_sq > 1e-9 {
                    let proj = dot(&r, &r_prime);
                    config.lambda * (proj * proj) / r_norm_sq
                } else {
                    0.0
                };
                let score = r_prime_norm_sq + penalty;
                (*c, score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(c, _)| c);

        if let Some(sec) = secondary {
            secondary_lists[sec].push(vid as u32);
        }
    }
}

// ── recall helper (used in tests and the demo binary) ─────────────────────────

/// Recall@k: fraction of true top-k that appear in retrieved top-k.
pub fn recall_at_k(truth: &[usize], got: &[SearchResult], k: usize) -> f64 {
    let take = k.min(truth.len()).min(got.len());
    if take == 0 {
        return 0.0;
    }
    use std::collections::HashSet;
    let truth_set: HashSet<usize> = truth.iter().take(take).copied().collect();
    got.iter().take(take).filter(|r| truth_set.contains(&r.id)).count() as f64
        / take as f64
}
