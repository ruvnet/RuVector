//! Greedy proximity graph with SymphonyQG-style packed neighbor codes.
//!
//! Each node stores:
//!   - `neighbor_ids`: indices of its M nearest graph neighbors.
//!   - `pq_codes`: packed 4-bit codes for *each neighbor's vector*, stored
//!     contiguously so FastScan can stream through the entire neighbor block.
//!
//! Construction: O(n × n) greedy, adequate for the nightly PoC. A production
//! implementation would use NSW/HNSW entry-point seeding for O(n log n) build.

use crate::error::SqgError;
use crate::pq4::Pq4;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Per-node edge list with co-located quantized neighbor codes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEdges {
    /// Graph neighbor indices.
    pub neighbor_ids: Vec<u32>,
    /// Packed 4-bit PQ codes for each neighbor, row-major.
    /// Each row is `ceil(pq.m / 2)` bytes.
    pub pq_codes: Vec<u8>,
}

/// Proximity graph with quantized neighbor codes (the SymphonyQG structure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqgGraph {
    pub edges: Vec<NodeEdges>,
    /// Per-node PQ codes for initial scoring during search seeding.
    pub node_codes: Vec<Vec<u8>>,
    pub code_bytes: usize,
    pub m_neighbors: usize,
}

impl SqgGraph {
    /// Build the graph from raw float vectors `data` (shape [n, dim]).
    pub fn build(data: &[f32], dim: usize, m_neighbors: usize, pq: &Pq4) -> Result<Self, SqgError> {
        let n = data.len() / dim;
        if n == 0 {
            return Err(SqgError::Config("empty dataset".into()));
        }
        let degree = m_neighbors.min(n.saturating_sub(1));
        let code_bytes = (pq.m + 1) / 2;

        // Pre-encode every vector once.
        let all_codes: Vec<Vec<u8>> = (0..n)
            .map(|i| pq.encode(&data[i * dim..(i + 1) * dim]))
            .collect();

        // Build O(n²) exact-neighbor lists (forward edges).
        let mut forward: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            let vi = &data[i * dim..(i + 1) * dim];
            let mut dists: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let vj = &data[j * dim..(j + 1) * dim];
                    let d: f32 = vi.iter().zip(vj).map(|(a, b)| (a - b).powi(2)).sum();
                    (d, j as u32)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(degree);
            forward.push(dists.iter().map(|&(_, id)| id).collect());
        }

        // Add reverse (backlink) edges for navigability: if A→B, also add B→A.
        // Cap each node's total degree at 2×degree. This mirrors HNSW's `efConstruction`
        // bidirectional edge policy, which is key to NSW-style navigability.
        let max_degree = degree * 2;
        let mut adj: Vec<Vec<u32>> = forward.clone();
        for i in 0..n {
            for j in 0..forward[i].len() {
                let nid = forward[i][j] as usize;
                if !adj[nid].contains(&(i as u32)) && adj[nid].len() < max_degree {
                    adj[nid].push(i as u32);
                }
            }
        }

        // Pack neighbor codes contiguously (SymphonyQG layout).
        let mut edges: Vec<NodeEdges> = Vec::with_capacity(n);
        for i in 0..n {
            let neighbor_ids = adj[i].clone();
            let mut pq_codes = Vec::with_capacity(neighbor_ids.len() * code_bytes);
            for &nid in &neighbor_ids {
                pq_codes.extend_from_slice(&all_codes[nid as usize]);
            }
            edges.push(NodeEdges { neighbor_ids, pq_codes });
        }

        Ok(Self { edges, node_codes: all_codes, code_bytes, m_neighbors: degree })
    }

    /// Greedy beam search using FastScan for neighbor scoring.
    ///
    /// Candidates are explored nearest-first (min-heap). The result set is a
    /// max-heap of size `ef` that keeps the best-seen candidates; the top-k are
    /// returned after the traversal completes.
    ///
    /// `rerank_data`: if `Some`, re-rank result candidates with exact f32 L2
    /// (Variant C).
    pub fn search(
        &self,
        query: &[f32],
        data: &[f32],
        dim: usize,
        pq: &Pq4,
        k: usize,
        ef: usize,
        rerank_data: Option<&[f32]>,
    ) -> Vec<(u32, f32)> {
        use crate::fastscan::scan_neighbors;

        let n = self.edges.len();
        let lut = pq.build_lut(query);
        let mut visited = vec![false; n];

        // Min-heap for exploration (nearest candidate first).
        let mut candidates: BinaryHeap<Reverse<(u16, u32)>> = BinaryHeap::new();
        // Max-heap for results (pop worst to maintain top-ef set).
        let mut result: BinaryHeap<(u16, u32)> = BinaryHeap::new();

        // Seed with multiple random-ish entry points (spread across n).
        let n_seeds = (n as f64).sqrt().ceil() as usize;
        let stride = (n / n_seeds).max(1);
        for s in 0..n_seeds {
            let seed = (s * stride) as u32;
            if visited[seed as usize] { continue; }
            let est = lut_score_codes(&lut, &self.node_codes[seed as usize], pq.m);
            visited[seed as usize] = true;
            candidates.push(Reverse((est, seed)));
            result.push((est, seed));
        }

        // Beam search: explore nearest unvisited node, expand via FastScan.
        while let Some(Reverse((cur_est, cur))) = candidates.pop() {
            // Early termination: if this candidate is worse than the ef-th result, stop.
            if result.len() >= ef {
                if let Some(&(worst, _)) = result.peek() {
                    if cur_est > worst {
                        break;
                    }
                }
            }

            let edges = &self.edges[cur as usize];
            if edges.neighbor_ids.is_empty() {
                continue;
            }

            // Batch-score all neighbors with FastScan.
            let scores = scan_neighbors(&lut, &edges.pq_codes, edges.neighbor_ids.len(), pq.m);

            for (&nid, &score) in edges.neighbor_ids.iter().zip(scores.iter()) {
                let nid_usize = nid as usize;
                if visited[nid_usize] {
                    continue;
                }
                visited[nid_usize] = true;

                // Add to results if better than current worst or result not full.
                if result.len() < ef {
                    result.push((score, nid));
                    candidates.push(Reverse((score, nid)));
                } else if let Some(&(worst, _)) = result.peek() {
                    if score < worst {
                        result.pop();
                        result.push((score, nid));
                        candidates.push(Reverse((score, nid)));
                    }
                }
            }
        }

        // Drain result heap into sorted vec (ascending = nearest first).
        let mut top: Vec<(u16, u32)> = result.into_sorted_vec();
        top.truncate(k);

        if let Some(full_data) = rerank_data {
            let mut reranked: Vec<(f32, u32)> = top.iter()
                .map(|&(_, id)| {
                    let v = &full_data[id as usize * dim..(id as usize + 1) * dim];
                    let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                    (d, id)
                })
                .collect();
            reranked.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            reranked.truncate(k);
            return reranked.iter().map(|&(d, id)| (id, d)).collect();
        }

        top.iter().map(|&(d, id)| (id, d as f32)).collect()
    }
}

/// Score a single node using its own PQ codes and the query LUT.
fn lut_score_codes(lut: &[u8], codes: &[u8], m: usize) -> u16 {
    use crate::pq4::CENTROIDS_PER_SUBSPACE;
    let k = CENTROIDS_PER_SUBSPACE;
    let mut acc: u16 = 0;
    for sub in 0..m {
        let byte_idx = sub / 2;
        if byte_idx >= codes.len() { break; }
        let nibble = if sub % 2 == 0 {
            (codes[byte_idx] & 0x0F) as usize
        } else {
            ((codes[byte_idx] >> 4) & 0x0F) as usize
        };
        acc = acc.saturating_add(lut[sub * k + nibble] as u16);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    fn random_vecs(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        (0..n * d).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn graph_builds_without_panic() {
        let n = 50;
        let d = 16;
        let data = random_vecs(n, d, 123);
        let pq = Pq4::train(&data, d, 4, 10).unwrap();
        let graph = SqgGraph::build(&data, d, 8, &pq).unwrap();
        assert_eq!(graph.edges.len(), n);
        assert_eq!(graph.node_codes.len(), n);
        assert!(graph.edges[0].neighbor_ids.len() <= 8 * 2); // forward + backlinks
    }

    #[test]
    fn search_returns_k_results() {
        let n = 100;
        let d = 16;
        let data = random_vecs(n, d, 77);
        let pq = Pq4::train(&data, d, 4, 10).unwrap();
        let graph = SqgGraph::build(&data, d, 8, &pq).unwrap();
        let query = random_vecs(1, d, 999);
        let results = graph.search(&query, &data, d, &pq, 10, 30, None);
        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }
}
