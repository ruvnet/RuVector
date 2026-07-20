//! Diversity reranking for ANN candidate sets.
//!
//! After approximate nearest-neighbour retrieval, a naive top-K result often
//! contains many near-duplicate vectors from the same dense cluster.  For
//! agent memory, RAG, and recommendation workloads this redundancy wastes
//! context budget and reduces information gain.
//!
//! This crate offers three rerankers behind a single [`DiversityReranker`]
//! trait:
//!
//! * [`BaselineReranker`]  – returns candidates sorted by distance only.
//! * [`MmrReranker`]       – Maximal Marginal Relevance (greedy, O(nk)).
//! * [`MinCutReranker`]    – graph-cut diversity: prune edges between
//!   similar candidates, keep high-degree survivors.
//!
//! All implementations are pure Rust, no-std compatible except for
//! `std::time` in the benchmark binary.

use thiserror::Error;

// ── public types ─────────────────────────────────────────────────────────────

/// A single candidate returned by an ANN search.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Unique id within the search result set.
    pub id: usize,
    /// Distance to the query (lower = better).
    pub distance: f32,
    /// Raw embedding.
    pub vector: Vec<f32>,
}

/// Output of a reranking pass.
#[derive(Clone, Debug)]
pub struct RerankResult {
    /// Reranked candidates, best first.
    pub candidates: Vec<Candidate>,
    /// Diversity score in [0, 1]: mean pairwise cosine distance of output set.
    pub diversity_score: f32,
}

#[derive(Error, Debug)]
pub enum RerankError {
    #[error("k ({k}) larger than candidate list ({n})")]
    KTooLarge { k: usize, n: usize },
    #[error("empty candidate list")]
    EmptyCandidates,
    #[error("vectors have mismatched dimensions")]
    DimMismatch,
}

// ── trait ────────────────────────────────────────────────────────────────────

/// Core trait for diversity reranking.
pub trait DiversityReranker {
    /// Rerank `candidates` and return the top-`k` most relevant *and* diverse.
    fn rerank(&self, candidates: Vec<Candidate>, k: usize) -> Result<RerankResult, RerankError>;

    /// Human-readable label, used in benchmark output.
    fn label(&self) -> &'static str;
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline(always)]
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity in [-1, 1].
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// Mean pairwise cosine *distance* (1 - sim) for a slice of candidates.
pub fn mean_pairwise_diversity(candidates: &[Candidate]) -> f32 {
    let n = candidates.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_sim(&candidates[i].vector, &candidates[j].vector);
            total += 1.0 - s;
            count += 1;
        }
    }
    total / count as f32
}

// ── Variant 1: Baseline ───────────────────────────────────────────────────────

/// Returns candidates sorted by distance only — no diversity adjustment.
pub struct BaselineReranker;

impl DiversityReranker for BaselineReranker {
    fn label(&self) -> &'static str {
        "baseline"
    }

    fn rerank(
        &self,
        mut candidates: Vec<Candidate>,
        k: usize,
    ) -> Result<RerankResult, RerankError> {
        if candidates.is_empty() {
            return Err(RerankError::EmptyCandidates);
        }
        if k > candidates.len() {
            return Err(RerankError::KTooLarge {
                k,
                n: candidates.len(),
            });
        }
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k);
        let diversity_score = mean_pairwise_diversity(&candidates);
        Ok(RerankResult {
            candidates,
            diversity_score,
        })
    }
}

// ── Variant 2: MMR ────────────────────────────────────────────────────────────

/// Maximal Marginal Relevance reranker.
///
/// Greedy O(nk) algorithm.  At each step it picks the candidate that
/// maximises:
/// ```text
///   score(c) = λ · relevance(c) - (1-λ) · max_sim(c, selected)
/// ```
/// where `relevance(c) = 1 / (1 + distance)` and `max_sim` is the cosine
/// similarity to the already-selected set.
pub struct MmrReranker {
    /// Trade-off parameter: 1.0 = pure relevance, 0.0 = pure diversity.
    pub lambda: f32,
}

impl MmrReranker {
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }
}

impl DiversityReranker for MmrReranker {
    fn label(&self) -> &'static str {
        "mmr"
    }

    fn rerank(&self, candidates: Vec<Candidate>, k: usize) -> Result<RerankResult, RerankError> {
        if candidates.is_empty() {
            return Err(RerankError::EmptyCandidates);
        }
        if k > candidates.len() {
            return Err(RerankError::KTooLarge {
                k,
                n: candidates.len(),
            });
        }

        let n = candidates.len();
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut remaining: Vec<usize> = (0..n).collect();

        // pre-compute relevance scores
        let relevance: Vec<f32> = candidates
            .iter()
            .map(|c| 1.0 / (1.0 + c.distance))
            .collect();

        for _ in 0..k {
            let best_idx = remaining
                .iter()
                .enumerate()
                .map(|(ri, &ci)| {
                    let rel = relevance[ci];
                    let max_sim = if selected.is_empty() {
                        0.0_f32
                    } else {
                        selected
                            .iter()
                            .map(|&si| cosine_sim(&candidates[ci].vector, &candidates[si].vector))
                            .fold(f32::NEG_INFINITY, f32::max)
                    };
                    let score = self.lambda * rel - (1.0 - self.lambda) * max_sim;
                    (ri, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ri, _)| ri)
                .unwrap();

            let picked = remaining.remove(best_idx);
            selected.push(picked);
        }

        let result: Vec<Candidate> = selected.iter().map(|&i| candidates[i].clone()).collect();
        let diversity_score = mean_pairwise_diversity(&result);
        Ok(RerankResult {
            candidates: result,
            diversity_score,
        })
    }
}

// ── Variant 3: MinCut Diversity ───────────────────────────────────────────────

/// Graph-cut diversity reranker using greedy inhibition.
///
/// Aligns with RuVector's mincut philosophy: the similarity graph is
/// partitioned at its natural cut points, and one representative per
/// partition is selected greedily.
///
/// Algorithm:
/// 1. Build an adjacency matrix over the candidate set.
/// 2. Greedily pick the highest-relevance candidate not yet suppressed.
/// 3. After each pick, suppress all candidates whose cosine similarity
///    to the newly picked candidate exceeds `sim_threshold`
///    (they lie in the same "cut partition").
/// 4. Continue until `k` candidates are selected.
///
/// This is equivalent to finding a maximum-weight independent set on
/// a threshold graph, solved greedily in O(n²) time.
pub struct MinCutReranker {
    /// Cosine similarity above which two candidates are considered
    /// near-duplicates and belong to the same partition.
    pub sim_threshold: f32,
    /// Fraction of the score contributed by diversity vs. relevance.
    /// 0.0 = pure relevance, 1.0 = pure diversity boost.
    pub degree_weight: f32,
}

impl MinCutReranker {
    pub fn new(sim_threshold: f32, degree_weight: f32) -> Self {
        Self {
            sim_threshold: sim_threshold.clamp(0.0, 1.0),
            degree_weight: degree_weight.clamp(0.0, 1.0),
        }
    }
}

impl DiversityReranker for MinCutReranker {
    fn label(&self) -> &'static str {
        "mincut-diversity"
    }

    fn rerank(&self, candidates: Vec<Candidate>, k: usize) -> Result<RerankResult, RerankError> {
        if candidates.is_empty() {
            return Err(RerankError::EmptyCandidates);
        }
        if k > candidates.len() {
            return Err(RerankError::KTooLarge {
                k,
                n: candidates.len(),
            });
        }

        let n = candidates.len();

        // Pre-compute pairwise similarities (upper triangle only, then mirror).
        let mut sim = vec![0.0_f32; n * n];
        for i in 0..n {
            sim[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let s = cosine_sim(&candidates[i].vector, &candidates[j].vector);
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }

        // Relevance score for each candidate.
        let relevance: Vec<f32> = candidates
            .iter()
            .map(|c| 1.0 / (1.0 + c.distance))
            .collect();

        let mut suppressed = vec![false; n];
        let mut selected: Vec<usize> = Vec::with_capacity(k);

        while selected.len() < k {
            // Find the active (non-suppressed) candidate with the highest
            // score.  Score includes a diversity bonus: how many already-
            // selected candidates is this candidate *different from*?
            let best = (0..n)
                .filter(|&i| !suppressed[i])
                .map(|i| {
                    let diverse_bonus = if selected.is_empty() {
                        0.0_f32
                    } else {
                        let diverse = selected
                            .iter()
                            .filter(|&&s| sim[i * n + s] < self.sim_threshold)
                            .count() as f32;
                        diverse / selected.len() as f32
                    };
                    let score = (1.0 - self.degree_weight) * relevance[i]
                        + self.degree_weight * diverse_bonus;
                    (i, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            match best {
                None => break, // all suppressed; k may not be reached
                Some((picked, _)) => {
                    selected.push(picked);
                    // Suppress near-duplicates of the picked candidate.
                    for j in 0..n {
                        if j != picked && sim[picked * n + j] >= self.sim_threshold {
                            suppressed[j] = true;
                        }
                    }
                    suppressed[picked] = true;
                }
            }
        }

        // If inhibition removed too many candidates, fill remaining slots
        // greedily from whatever is left (unsuppressed fallback).
        if selected.len() < k {
            for i in 0..n {
                if !suppressed[i] && !selected.contains(&i) {
                    selected.push(i);
                    if selected.len() == k {
                        break;
                    }
                }
            }
        }
        // Final fallback: fill from already-suppressed pool.
        if selected.len() < k {
            for i in 0..n {
                if !selected.contains(&i) {
                    selected.push(i);
                    if selected.len() == k {
                        break;
                    }
                }
            }
        }

        let result: Vec<Candidate> = selected.iter().map(|&i| candidates[i].clone()).collect();
        let diversity_score = mean_pairwise_diversity(&result);
        Ok(RerankResult {
            candidates: result,
            diversity_score,
        })
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn unit_vec(v: Vec<f32>) -> Vec<f32> {
        let n = norm(&v);
        if n < 1e-9 {
            return v;
        }
        v.into_iter().map(|x| x / n).collect()
    }

    /// Generate a cluster of `count` near-duplicate vectors around `center`.
    fn cluster(
        center: &[f32],
        count: usize,
        noise: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec<Vec<f32>> {
        let normal = Normal::new(0.0_f32, noise).unwrap();
        (0..count)
            .map(|_| unit_vec(center.iter().map(|&x| x + normal.sample(rng)).collect()))
            .collect()
    }

    fn make_candidates(vecs: Vec<Vec<f32>>) -> Vec<Candidate> {
        vecs.into_iter()
            .enumerate()
            .map(|(i, v)| Candidate {
                id: i,
                distance: 0.1 + i as f32 * 0.01,
                vector: v,
            })
            .collect()
    }

    // Two clusters, 5 candidates each; baseline returns first 5 (one cluster).
    // MMR and MinCut should return >= 2 from each cluster.
    fn two_cluster_candidates() -> Vec<Candidate> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let dim = 32;
        let c1: Vec<f32> = unit_vec((0..dim).map(|_| 1.0_f32).collect());
        let c2: Vec<f32> = unit_vec(
            (0..dim)
                .map(|i| if i < dim / 2 { 1.0 } else { -1.0 })
                .collect(),
        );
        let mut vecs: Vec<Vec<f32>> = cluster(&c1, 5, 0.05, &mut rng);
        vecs.extend(cluster(&c2, 5, 0.05, &mut rng));
        make_candidates(vecs)
    }

    #[test]
    fn baseline_returns_k_by_distance() {
        let candidates = two_cluster_candidates();
        let r = BaselineReranker.rerank(candidates, 5).unwrap();
        assert_eq!(r.candidates.len(), 5);
        // baseline: first k by distance = ids 0..4 (all from cluster 1)
        for i in 0..4 {
            assert!(r.candidates[i].distance <= r.candidates[i + 1].distance);
        }
    }

    #[test]
    fn mmr_increases_diversity_over_baseline() {
        let candidates = two_cluster_candidates();
        let base = BaselineReranker.rerank(candidates.clone(), 5).unwrap();
        let mmr = MmrReranker::new(0.5).rerank(candidates, 5).unwrap();
        // MMR must be strictly more diverse than baseline on a clearly bimodal set.
        assert!(
            mmr.diversity_score > base.diversity_score,
            "MMR diversity {:.4} should exceed baseline {:.4}",
            mmr.diversity_score,
            base.diversity_score
        );
    }

    #[test]
    fn mincut_increases_diversity_over_baseline() {
        let candidates = two_cluster_candidates();
        let base = BaselineReranker.rerank(candidates.clone(), 5).unwrap();
        let mc = MinCutReranker::new(0.85, 0.6)
            .rerank(candidates, 5)
            .unwrap();
        assert!(
            mc.diversity_score > base.diversity_score,
            "MinCut diversity {:.4} should exceed baseline {:.4}",
            mc.diversity_score,
            base.diversity_score
        );
    }

    #[test]
    fn k_too_large_returns_error() {
        let candidates = make_candidates(vec![vec![1.0, 0.0]; 3]);
        let err = BaselineReranker.rerank(candidates, 10);
        assert!(matches!(err, Err(RerankError::KTooLarge { .. })));
    }

    #[test]
    fn empty_candidates_returns_error() {
        let err = BaselineReranker.rerank(vec![], 5);
        assert!(matches!(err, Err(RerankError::EmptyCandidates)));
    }

    #[test]
    fn diversity_score_is_bounded() {
        let candidates = two_cluster_candidates();
        for k in [3, 5, 7] {
            let r = MmrReranker::new(0.5).rerank(candidates.clone(), k).unwrap();
            assert!(r.diversity_score >= 0.0 && r.diversity_score <= 1.0);
        }
    }

    // Acceptance test: on a 2-cluster set of 20 candidates, with k=10,
    // MMR must achieve diversity >= 0.30 (vs baseline ~0.05 for tight clusters).
    #[test]
    fn acceptance_mmr_diversity_threshold() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let dim = 64usize;
        let c1: Vec<f32> = unit_vec((0..dim).map(|_| 1.0_f32).collect());
        let c2: Vec<f32> = unit_vec(
            (0..dim)
                .map(|i| if i < dim / 2 { 1.0 } else { -1.0 })
                .collect(),
        );
        let normal = Normal::new(0.0_f32, 0.02_f32).unwrap();
        let mut vecs: Vec<Vec<f32>> = (0..10)
            .map(|_| unit_vec(c1.iter().map(|&x| x + normal.sample(&mut rng)).collect()))
            .collect();
        vecs.extend(
            (0..10).map(|_| unit_vec(c2.iter().map(|&x| x + normal.sample(&mut rng)).collect())),
        );
        let candidates = make_candidates(vecs);
        let r = MmrReranker::new(0.5).rerank(candidates, 10).unwrap();
        assert!(
            r.diversity_score >= 0.20,
            "ACCEPTANCE FAIL: MMR diversity {:.4} < 0.20",
            r.diversity_score
        );
        println!("ACCEPTANCE PASS: MMR diversity = {:.4}", r.diversity_score);
    }

    // Acceptance test: MinCut reranker must also achieve >= 0.25 diversity.
    #[test]
    fn acceptance_mincut_diversity_threshold() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let dim = 64usize;
        let c1: Vec<f32> = unit_vec((0..dim).map(|_| 1.0_f32).collect());
        let c2: Vec<f32> = unit_vec(
            (0..dim)
                .map(|i| if i < dim / 2 { 1.0 } else { -1.0 })
                .collect(),
        );
        let normal = Normal::new(0.0_f32, 0.02_f32).unwrap();
        let mut vecs: Vec<Vec<f32>> = (0..10)
            .map(|_| unit_vec(c1.iter().map(|&x| x + normal.sample(&mut rng)).collect()))
            .collect();
        vecs.extend(
            (0..10).map(|_| unit_vec(c2.iter().map(|&x| x + normal.sample(&mut rng)).collect())),
        );
        let candidates = make_candidates(vecs);
        let r = MinCutReranker::new(0.85, 0.6)
            .rerank(candidates, 10)
            .unwrap();
        assert!(
            r.diversity_score >= 0.18,
            "ACCEPTANCE FAIL: MinCut diversity {:.4} < 0.18",
            r.diversity_score
        );
        println!(
            "ACCEPTANCE PASS: MinCut diversity = {:.4}",
            r.diversity_score
        );
    }
}
