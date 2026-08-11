//! The three cascade ANN variants.
//!
//! # LinearFull
//! Brute-force f32 scan over all N vectors.  Ground-truth reference.
//!
//! # QuantizedCascade
//! INT8 scan → keep top-(k * ef_mult) candidates → exact f32 re-rank.
//! Equivalent to a simpler speculative-style cascade without graph expansion.
//!
//! # GraphNeighbourCascade
//! INT8 scan → collect top-(k * ef_mult) candidates → identify the
//! "uncertain zone" (candidates with score ≤ k-th score * (1 + delta)) →
//! add their graph neighbours to the pool → exact f32 re-rank of the full pool.
//!
//! The uncertain-zone expansion recovers true neighbours displaced by
//! quantization error without scanning the full corpus a second time.

use crate::graph::KnnGraph;
use crate::quantize::QuantizedCorpus;
use crate::{AnnVariant, Hit};

// ─── LinearFull ──────────────────────────────────────────────────────────────

pub struct LinearFull {
    vecs: Vec<Vec<f32>>,
}

impl LinearFull {
    pub fn build(vecs: &[Vec<f32>]) -> Self {
        LinearFull {
            vecs: vecs.to_vec(),
        }
    }
}

impl AnnVariant for LinearFull {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(id, v)| Hit {
                id,
                dist: crate::dist_sq(query, v),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "LinearFull"
    }

    fn memory_bytes(&self) -> usize {
        self.vecs.len() * self.vecs.first().map(|v| v.len() * 4).unwrap_or(0)
    }
}

// ─── QuantizedCascade ────────────────────────────────────────────────────────

/// Candidate multiplier: initial pool size = k * EF_MULT.
/// Set to 1 so the initial pool equals exactly k — quantization rank inversions
/// become visible and graph expansion has meaningful work to do.
pub const DEFAULT_EF_MULT: usize = 1;

pub struct QuantizedCascade {
    corpus: QuantizedCorpus,
    ef_mult: usize,
}

impl QuantizedCascade {
    pub fn build(vecs: &[Vec<f32>], ef_mult: usize) -> Self {
        QuantizedCascade {
            corpus: QuantizedCorpus::build(vecs),
            ef_mult,
        }
    }
}

impl AnnVariant for QuantizedCascade {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let pool_size = (k * self.ef_mult).min(self.corpus.n);

        // INT8 approximate scan.
        let mut approx: Vec<(usize, f32)> = (0..self.corpus.n)
            .map(|id| (id, self.corpus.dist_sq_approx(id, query)))
            .collect();
        approx.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        approx.truncate(pool_size);

        // Exact f32 re-rank of the pool.
        let mut hits: Vec<Hit> = approx
            .iter()
            .map(|&(id, _)| Hit {
                id,
                dist: self.corpus.dist_sq_exact(id, query),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "QuantizedCascade"
    }

    fn memory_bytes(&self) -> usize {
        self.corpus.quantized_bytes() + self.corpus.f32_bytes()
    }
}

// ─── GraphNeighbourCascade ───────────────────────────────────────────────────

/// Relative uncertainty margin.  Candidates with approx score ≤ k-th score *
/// (1 + DELTA) are considered "uncertain" and trigger graph expansion.
/// Wider margin (0.30) is appropriate for ef_mult=1 small pools.
pub const DEFAULT_DELTA: f32 = 0.30;

pub struct GraphNeighbourCascade {
    corpus: QuantizedCorpus,
    graph: KnnGraph,
    ef_mult: usize,
    delta: f32,
}

impl GraphNeighbourCascade {
    pub fn build(vecs: &[Vec<f32>], k_graph: usize, ef_mult: usize, delta: f32) -> Self {
        GraphNeighbourCascade {
            corpus: QuantizedCorpus::build(vecs),
            graph: KnnGraph::build(vecs, k_graph),
            ef_mult,
            delta,
        }
    }
}

impl AnnVariant for GraphNeighbourCascade {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let pool_size = (k * self.ef_mult).min(self.corpus.n);

        // Stage 1: INT8 approximate scan.
        let mut approx: Vec<(usize, f32)> = (0..self.corpus.n)
            .map(|id| (id, self.corpus.dist_sq_approx(id, query)))
            .collect();
        approx.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        approx.truncate(pool_size);

        // Stage 2: Identify uncertain zone — candidates within delta of k-th score.
        let k_approx_dist = approx
            .get(k.saturating_sub(1))
            .map(|&(_, d)| d)
            .unwrap_or(f32::MAX);
        let threshold = k_approx_dist * (1.0 + self.delta);

        let uncertain: Vec<usize> = approx
            .iter()
            .take(k)
            .filter(|&&(_, d)| d <= threshold)
            .map(|&(id, _)| id)
            .collect();

        // Stage 3: Expand via graph neighbours of uncertain candidates.
        let initial_ids: std::collections::HashSet<usize> =
            approx.iter().map(|&(id, _)| id).collect();

        let mut expanded: Vec<usize> = Vec::new();
        for uid in &uncertain {
            for &nb in self.graph.neighbours(*uid) {
                let nb = nb as usize;
                if !initial_ids.contains(&nb) {
                    expanded.push(nb);
                }
            }
        }
        expanded.sort_unstable();
        expanded.dedup();

        // Stage 4: Exact f32 re-rank of initial pool + expanded neighbours.
        let mut hits: Vec<Hit> = approx
            .iter()
            .map(|&(id, _)| Hit {
                id,
                dist: self.corpus.dist_sq_exact(id, query),
            })
            .chain(expanded.iter().map(|&id| Hit {
                id,
                dist: self.corpus.dist_sq_exact(id, query),
            }))
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.dedup_by_key(|h| h.id);
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "GraphNeighbourCascade"
    }

    fn memory_bytes(&self) -> usize {
        self.corpus.quantized_bytes() + self.corpus.f32_bytes() + self.graph.memory_bytes()
    }
}
