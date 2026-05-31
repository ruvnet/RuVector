//! Multi-vector index variants: BruteForce MaxSim, FlatFDE, and HNSW-FDE.

use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::encoder::FdeEncoder;
use crate::error::MuveraError;

/// Result of a single multi-vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub doc_id: usize,
    /// Score (higher = more similar). MaxSim for brute-force; FDE inner product for others.
    pub score: f32,
}

/// Common interface for all three index variants.
pub trait MultiVecIndex {
    /// Build the index from a corpus.
    fn build(
        docs: Vec<Vec<Vec<f32>>>,
        encoder: Arc<FdeEncoder>,
    ) -> Result<Self, MuveraError>
    where
        Self: Sized;

    /// Return top-k most similar documents to `query_vecs`.
    fn search(
        &self,
        query_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MuveraError>;

    /// Approximate heap memory used by the index.
    fn memory_bytes(&self) -> usize;

    /// Human-readable variant name.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Variant 1: BruteForceMaxSim — exact MaxSim, O(|Q|×|D|×d) per query
// ---------------------------------------------------------------------------

/// Exact baseline: computes full MaxSim between query tokens and every document.
/// Ground-truth for recall computation; slowest at query time.
pub struct BruteForceMaxSim {
    docs: Vec<Vec<Vec<f32>>>,
    #[allow(dead_code)]
    encoder: Arc<FdeEncoder>,
}

impl MultiVecIndex for BruteForceMaxSim {
    fn build(
        docs: Vec<Vec<Vec<f32>>>,
        encoder: Arc<FdeEncoder>,
    ) -> Result<Self, MuveraError> {
        validate_corpus(&docs, encoder.orig_dim)?;
        Ok(Self { docs, encoder })
    }

    fn search(
        &self,
        query_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MuveraError> {
        if k > self.docs.len() {
            return Err(MuveraError::KTooLarge {
                k,
                n: self.docs.len(),
            });
        }
        let mut scored: Vec<(usize, f32)> = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, FdeEncoder::max_sim(query_vecs, doc)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(doc_id, score)| SearchResult { doc_id, score })
            .collect())
    }

    fn memory_bytes(&self) -> usize {
        self.docs
            .iter()
            .map(|doc| doc.iter().map(|v| v.len() * 4).sum::<usize>())
            .sum()
    }

    fn name(&self) -> &'static str {
        "BruteForceMaxSim (exact baseline)"
    }
}

// ---------------------------------------------------------------------------
// Variant 2: FlatFdeIndex — FDE-encoded flat scan, O(n×R×D) per query
// ---------------------------------------------------------------------------

/// Flat inner-product scan over FDE-encoded documents.
/// Faster than BruteForce because one matrix multiply replaces nested loops,
/// and the scan is over fixed-size float arrays (cache-friendly).
pub struct FlatFdeIndex {
    encoded_docs: Vec<Vec<f32>>,
    encoder: Arc<FdeEncoder>,
}

impl MultiVecIndex for FlatFdeIndex {
    fn build(
        docs: Vec<Vec<Vec<f32>>>,
        encoder: Arc<FdeEncoder>,
    ) -> Result<Self, MuveraError> {
        validate_corpus(&docs, encoder.orig_dim)?;
        let encoded_docs: Vec<Vec<f32>> = docs.iter().map(|doc| encoder.encode(doc)).collect();
        Ok(Self {
            encoded_docs,
            encoder,
        })
    }

    fn search(
        &self,
        query_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MuveraError> {
        if k > self.encoded_docs.len() {
            return Err(MuveraError::KTooLarge {
                k,
                n: self.encoded_docs.len(),
            });
        }
        let q_fde = self.encoder.encode(query_vecs);
        let mut heap = BinaryHeap::with_capacity(k + 1);

        for (i, doc_fde) in self.encoded_docs.iter().enumerate() {
            let ip: f32 = q_fde.iter().zip(doc_fde.iter()).map(|(a, b)| a * b).sum();
            heap.push(std::cmp::Reverse(OrdF32(ip, i as u32)));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|std::cmp::Reverse(OrdF32(score, doc_id))| SearchResult { doc_id: doc_id as usize, score })
            .collect();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(results)
    }

    fn memory_bytes(&self) -> usize {
        self.encoded_docs.len() * self.encoder.fde_dim * 4
    }

    fn name(&self) -> &'static str {
        "FlatFDE (FDE + flat scan)"
    }
}

// ---------------------------------------------------------------------------
// Variant 3: HnswFdeIndex — greedy single-layer HNSW over FDE encodings
// ---------------------------------------------------------------------------

/// HNSW-FDE: builds a greedy HNSW navigable graph over FDE-encoded docs,
/// then searches it with inner-product similarity (negated for min-heap).
///
/// M = 16 neighbors per node, ef = 64 for search.
pub struct HnswFdeIndex {
    // Adjacency list: neighbors[i] = up to M doc indices sorted by decreasing IP.
    neighbors: Vec<Vec<u32>>,
    encoded_docs: Vec<Vec<f32>>,
    encoder: Arc<FdeEncoder>,
    entry_point: u32,
    ef: usize,
}

const HNSW_M: usize = 16;

impl HnswFdeIndex {
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    #[inline]
    #[allow(dead_code)]
    fn ip(&self, a: usize, b: usize) -> f32 {
        self.encoded_docs[a]
            .iter()
            .zip(self.encoded_docs[b].iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    #[inline]
    fn ip_vec(&self, query: &[f32], doc: usize) -> f32 {
        query
            .iter()
            .zip(self.encoded_docs[doc].iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl MultiVecIndex for HnswFdeIndex {
    fn build(
        docs: Vec<Vec<Vec<f32>>>,
        encoder: Arc<FdeEncoder>,
    ) -> Result<Self, MuveraError> {
        validate_corpus(&docs, encoder.orig_dim)?;
        let n = docs.len();
        let encoded_docs: Vec<Vec<f32>> = docs.iter().map(|doc| encoder.encode(doc)).collect();

        // Greedy single-level HNSW construction.
        // For each new node, find its M nearest neighbors from already-inserted nodes.
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];

        for i in 1..n {
            // Greedy walk from entry point 0 to find candidates for node i.
            let mut candidates: Vec<(u32, f32)> = Vec::new();
            let mut visited = std::collections::HashSet::new();
            let mut stack = vec![0u32];
            visited.insert(0u32);

            // Best-first traversal bounded by 2×ef hops.
            let mut best_heap = BinaryHeap::new();
            let ip0: f32 = encoded_docs[i]
                .iter()
                .zip(encoded_docs[0].iter())
                .map(|(a, b)| a * b)
                .sum();
            best_heap.push(OrdF32(ip0, 0));

            let mut hops = 0usize;
            while let Some(OrdF32(_, cur)) = best_heap.pop() {
                hops += 1;
                if hops > 2 * HNSW_M {
                    break;
                }
                let cur = cur as usize;
                let ip_cur: f32 = encoded_docs[i]
                    .iter()
                    .zip(encoded_docs[cur].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                candidates.push((cur as u32, ip_cur));

                for &nb in &neighbors[cur] {
                    let nb = nb as usize;
                    if visited.insert(nb as u32) && nb < i {
                        let ip_nb: f32 = encoded_docs[i]
                            .iter()
                            .zip(encoded_docs[nb].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        best_heap.push(OrdF32(ip_nb, nb as u32));
                        stack.push(nb as u32);
                    }
                }
            }

            // Keep top M by IP as neighbors of i.
            candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.dedup_by_key(|(id, _)| *id);
            let my_neighbors: Vec<u32> = candidates
                .iter()
                .take(HNSW_M)
                .map(|(id, _)| *id)
                .collect();

            // Bidirectional link (pruned to M neighbors per node).
            for &nb in &my_neighbors {
                let nb = nb as usize;
                neighbors[nb].push(i as u32);
                if neighbors[nb].len() > HNSW_M {
                    // Keep best M neighbors for nb by their IP to nb.
                    let nb_enc = &encoded_docs[nb];
                    neighbors[nb].sort_unstable_by(|&a, &b| {
                        let ip_a: f32 = nb_enc
                            .iter()
                            .zip(encoded_docs[a as usize].iter())
                            .map(|(x, y)| x * y)
                            .sum();
                        let ip_b: f32 = nb_enc
                            .iter()
                            .zip(encoded_docs[b as usize].iter())
                            .map(|(x, y)| x * y)
                            .sum();
                        ip_b.partial_cmp(&ip_a).unwrap()
                    });
                    neighbors[nb].truncate(HNSW_M);
                }
            }
            neighbors[i] = my_neighbors;
        }

        Ok(Self {
            neighbors,
            encoded_docs,
            encoder,
            entry_point: 0,
            ef: 64,
        })
    }

    fn search(
        &self,
        query_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MuveraError> {
        let n = self.encoded_docs.len();
        if k > n {
            return Err(MuveraError::KTooLarge { k, n });
        }
        let q_fde = self.encoder.encode(query_vecs);

        // Greedy best-first search, ef candidates in the frontier.
        let mut visited = std::collections::HashSet::new();
        let ep = self.entry_point as usize;
        let ep_ip = self.ip_vec(&q_fde, ep);
        visited.insert(ep as u32);

        // candidate_heap: max-heap by IP (explore best first)
        let mut candidate_heap: BinaryHeap<OrdF32> = BinaryHeap::new();
        candidate_heap.push(OrdF32(ep_ip, ep as u32));

        // result_heap: min-heap of size ef (worst element at top for pruning)
        let mut result_heap: BinaryHeap<std::cmp::Reverse<OrdF32>> = BinaryHeap::new();
        result_heap.push(std::cmp::Reverse(OrdF32(ep_ip, ep as u32)));

        while let Some(OrdF32(cur_ip, cur_id)) = candidate_heap.pop() {
            // If current candidate is worse than worst in result set, stop.
            if let Some(std::cmp::Reverse(OrdF32(worst_ip, _))) = result_heap.peek() {
                if cur_ip < *worst_ip && result_heap.len() >= self.ef {
                    break;
                }
            }

            for &nb in &self.neighbors[cur_id as usize] {
                if visited.insert(nb) {
                    let nb_ip = self.ip_vec(&q_fde, nb as usize);
                    candidate_heap.push(OrdF32(nb_ip, nb));
                    result_heap.push(std::cmp::Reverse(OrdF32(nb_ip, nb)));
                    if result_heap.len() > self.ef {
                        result_heap.pop();
                    }
                }
            }
        }

        let mut results: Vec<SearchResult> = result_heap
            .into_iter()
            .map(|std::cmp::Reverse(OrdF32(score, doc_id))| SearchResult {
                doc_id: doc_id as usize,
                score,
            })
            .collect();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        Ok(results)
    }

    fn memory_bytes(&self) -> usize {
        let graph_bytes: usize = self.neighbors.iter().map(|nb| nb.len() * 4).sum();
        let fde_bytes = self.encoded_docs.len() * self.encoder.fde_dim * 4;
        graph_bytes + fde_bytes
    }

    fn name(&self) -> &'static str {
        "HnswFDE (FDE + HNSW, M=16)"
    }
}

// ---------------------------------------------------------------------------
// Recall helper
// ---------------------------------------------------------------------------

/// recall@k: fraction of true top-k doc IDs present in returned results.
pub fn recall_at_k(
    truth: &[SearchResult],
    got: &[SearchResult],
    k: usize,
) -> f64 {
    let truth_ids: std::collections::HashSet<usize> =
        truth.iter().take(k).map(|r| r.doc_id).collect();
    let hits = got.iter().take(k).filter(|r| truth_ids.contains(&r.doc_id)).count();
    let denom = truth_ids.len().min(k);
    if denom == 0 {
        1.0
    } else {
        hits as f64 / denom as f64
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn validate_corpus(docs: &[Vec<Vec<f32>>], expected_dim: usize) -> Result<(), MuveraError> {
    if docs.is_empty() {
        return Err(MuveraError::EmptyDataset);
    }
    for (i, doc) in docs.iter().enumerate() {
        if doc.is_empty() {
            return Err(MuveraError::EmptyDocument { doc_idx: i });
        }
        for tok in doc {
            if tok.len() != expected_dim {
                return Err(MuveraError::DimMismatch {
                    expected: expected_dim,
                    actual: tok.len(),
                    doc_idx: i,
                });
            }
        }
    }
    Ok(())
}

/// (score, id) ordered by score descending for max-heaps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrdF32(pub f32, pub u32);

impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn make_docs(n: usize, tokens: usize, dim: usize, seed: u64) -> Vec<Vec<Vec<f32>>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n)
            .map(|_| {
                (0..tokens)
                    .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
                    .collect()
            })
            .collect()
    }

    fn make_encoder(dim: usize, num_reps: usize) -> Arc<FdeEncoder> {
        Arc::new(FdeEncoder::new(num_reps, dim, 42).unwrap())
    }

    #[test]
    fn brute_force_returns_k_results() {
        let docs = make_docs(100, 10, 16, 1);
        let enc = make_encoder(16, 4);
        let idx = BruteForceMaxSim::build(docs.clone(), enc).unwrap();
        let query = make_docs(1, 5, 16, 99).pop().unwrap();
        let results = idx.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn flat_fde_returns_k_results() {
        let docs = make_docs(100, 10, 16, 2);
        let enc = make_encoder(16, 4);
        let idx = FlatFdeIndex::build(docs, enc).unwrap();
        let query = make_docs(1, 5, 16, 88).pop().unwrap();
        let results = idx.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn hnsw_fde_returns_k_results() {
        let docs = make_docs(200, 10, 16, 3);
        let enc = make_encoder(16, 4);
        let idx = HnswFdeIndex::build(docs, enc).unwrap();
        let query = make_docs(1, 5, 16, 77).pop().unwrap();
        let results = idx.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn brute_force_self_recall_is_one() {
        // If we put a query doc IN the corpus, it should be the top result.
        let mut docs = make_docs(50, 8, 16, 5);
        let query = docs[0].clone();
        let enc = make_encoder(16, 4);
        let idx = BruteForceMaxSim::build(docs.clone(), enc).unwrap();
        let res = idx.search(&query, 1).unwrap();
        assert_eq!(res[0].doc_id, 0, "self should be top-1 result");
    }

    /// Build clustered corpus: `n_clusters` topics, each doc's tokens sampled from
    /// one centroid + small noise. Queries are also drawn from cluster centroids.
    fn make_clustered_docs(
        n_docs: usize,
        tokens: usize,
        dim: usize,
        n_clusters: usize,
        seed: u64,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {
        use rand::Rng as _;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 0.05).unwrap();

        // Sample and L2-normalize cluster centroids.
        let centroid_normal = Normal::new(0.0_f32, 1.0).unwrap();
        let centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| centroid_normal.sample(&mut rng)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();

        let make_doc_for_cluster = |c: usize, rng: &mut rand::rngs::StdRng| -> Vec<Vec<f32>> {
            (0..tokens)
                .map(|_| {
                    let mut v = centroids[c].clone();
                    for x in v.iter_mut() {
                        *x += normal.sample(rng);
                    }
                    // Re-normalize after noise.
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    v.iter_mut().for_each(|x| *x /= norm);
                    v
                })
                .collect()
        };

        let docs: Vec<Vec<Vec<f32>>> = (0..n_docs)
            .map(|_| {
                let c = rng.gen_range(0..n_clusters);
                make_doc_for_cluster(c, &mut rng)
            })
            .collect();

        // Queries: one per cluster so the top results are the docs from that cluster.
        let queries: Vec<Vec<Vec<f32>>> = (0..n_clusters)
            .map(|c| make_doc_for_cluster(c, &mut rng))
            .collect();

        (docs, queries)
    }

    #[test]
    fn flat_fde_reasonable_recall_vs_brute() {
        // Clustered corpus: docs cluster around K=10 centroids, queries from same
        // centroids. FDE should rank same-cluster docs near the top.
        // R=16 reps, D=32 is sufficient for structure-based recall >40%.
        let (docs, queries) = make_clustered_docs(200, 10, 32, 10, 42);
        let enc = make_encoder(32, 16);
        let bf = BruteForceMaxSim::build(docs.clone(), enc.clone()).unwrap();
        let flat = FlatFdeIndex::build(docs.clone(), enc.clone()).unwrap();

        let mut total_recall = 0.0_f64;
        let k = 10;
        for q in &queries {
            let truth = bf.search(q, k).unwrap();
            let got = flat.search(q, k).unwrap();
            total_recall += recall_at_k(&truth, &got, k);
        }
        let mean = total_recall / queries.len() as f64;
        assert!(mean > 0.4, "FlatFDE recall@10 should be >40% on clustered data, got {mean:.2}");
    }

    #[test]
    fn dim_mismatch_is_rejected() {
        let enc = make_encoder(16, 4);
        let bad_docs = vec![vec![vec![0.0_f32; 8]]]; // 8 ≠ 16
        let err = BruteForceMaxSim::build(bad_docs, enc);
        assert!(err.is_err());
    }

    #[test]
    fn k_too_large_is_rejected() {
        let docs = make_docs(5, 4, 16, 9);
        let enc = make_encoder(16, 4);
        let idx = FlatFdeIndex::build(docs, enc).unwrap();
        let q = make_docs(1, 3, 16, 1).pop().unwrap();
        assert!(idx.search(&q, 10).is_err());
    }
}
