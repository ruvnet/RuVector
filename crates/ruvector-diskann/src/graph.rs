//! Vamana graph construction with α-robust pruning
//!
//! The Vamana algorithm builds a navigable graph with bounded out-degree R.
//! Each node connects to its nearest neighbors, pruned by the α parameter
//! which controls the balance between short-range and long-range edges.

use crate::distance::l2_squared;
use crate::error::{DiskAnnError, Result};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::sync::Arc;

/// Neighbor entry for the priority queue
#[derive(Clone)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (for beam search eviction)
struct MaxCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Vamana graph with bounded out-degree
pub struct VamanaGraph {
    /// Adjacency list: neighbors[node_id] = vec of neighbor ids
    pub neighbors: Vec<Vec<u32>>,
    /// Medoid (start node for greedy search)
    pub medoid: u32,
    /// Maximum out-degree
    pub max_degree: usize,
    /// Search beam width during construction
    pub build_beam: usize,
    /// Alpha parameter for robust pruning (>= 1.0, typically 1.2)
    pub alpha: f32,
}

impl VamanaGraph {
    pub fn new(n: usize, max_degree: usize, build_beam: usize, alpha: f32) -> Self {
        Self {
            neighbors: vec![Vec::new(); n],
            medoid: 0,
            max_degree,
            build_beam,
            alpha,
        }
    }

    /// Build the Vamana graph over the given vectors
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        let n = vectors.len();
        if n == 0 {
            return Err(DiskAnnError::Empty);
        }

        // Step 1: Find medoid (point closest to centroid)
        self.medoid = self.find_medoid(vectors);

        // Step 2: Initialize with random graph
        self.init_random_graph(n);

        // Step 3: Vamana iterations — two passes with α=1 then α=self.alpha
        let passes = if self.alpha > 1.0 { 2 } else { 1 };
        for pass in 0..passes {
            let alpha = if pass == 0 { 1.0 } else { self.alpha };

            // Random permutation
            let mut order: Vec<u32> = (0..n as u32).collect();
            {
                use rand::prelude::*;
                let mut rng = rand::thread_rng();
                order.shuffle(&mut rng);
            }

            for &node in &order {
                // Greedy search from medoid to find candidate neighbors
                let (candidates, _visited) =
                    self.greedy_search(vectors, &vectors[node as usize], self.build_beam);

                // Robust prune: select R neighbors from candidates
                let pruned = self.robust_prune(vectors, node, &candidates, alpha);

                // Update forward edges
                self.neighbors[node as usize] = pruned.clone();

                // Update reverse edges (bidirectional)
                for &neighbor in &pruned {
                    let nid = neighbor as usize;
                    if !self.neighbors[nid].contains(&node) {
                        if self.neighbors[nid].len() < self.max_degree {
                            self.neighbors[nid].push(node);
                        } else {
                            // Reverse edge overflows — prune the neighbor's list
                            let mut combined: Vec<u32> = self.neighbors[nid].clone();
                            combined.push(node);
                            let repruned = self.robust_prune(vectors, neighbor, &combined, alpha);
                            self.neighbors[nid] = repruned;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Greedy beam search from medoid toward query
    /// Returns (sorted candidates, visited set)
    pub fn greedy_search(
        &self,
        vectors: &[Vec<f32>],
        query: &[f32],
        beam_width: usize,
    ) -> (Vec<u32>, HashSet<u32>) {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::<Candidate>::new(); // min-heap
        let mut best = BinaryHeap::<MaxCandidate>::new(); // max-heap (for eviction)

        let start = self.medoid;
        let start_dist = l2_squared(&vectors[start as usize], query);
        candidates.push(Candidate { id: start, distance: start_dist });
        best.push(MaxCandidate { id: start, distance: start_dist });
        visited.insert(start);

        while let Some(current) = candidates.pop() {
            // If current is farther than the worst in best (and best is full), stop
            if best.len() >= beam_width {
                if let Some(worst) = best.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            // Expand neighbors
            for &neighbor in &self.neighbors[current.id as usize] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist = l2_squared(&vectors[neighbor as usize], query);

                // Add to candidates if better than worst in best
                let dominated = best.len() >= beam_width
                    && best.peek().map_or(false, |w| dist >= w.distance);

                if !dominated {
                    candidates.push(Candidate { id: neighbor, distance: dist });
                    best.push(MaxCandidate { id: neighbor, distance: dist });
                    if best.len() > beam_width {
                        best.pop(); // evict farthest
                    }
                }
            }
        }

        // Collect best as sorted vec
        let mut result: Vec<(u32, f32)> = best.into_iter().map(|c| (c.id, c.distance)).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let ids: Vec<u32> = result.into_iter().map(|(id, _)| id).collect();

        (ids, visited)
    }

    /// α-robust pruning (Algorithm 2 from DiskANN paper)
    fn robust_prune(
        &self,
        vectors: &[Vec<f32>],
        node: u32,
        candidates: &[u32],
        alpha: f32,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let node_vec = &vectors[node as usize];

        // Sort candidates by distance to node
        let mut sorted: Vec<(u32, f32)> = candidates
            .iter()
            .filter(|&&c| c != node)
            .map(|&c| (c, l2_squared(&vectors[c as usize], node_vec)))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result = Vec::with_capacity(self.max_degree);

        for (cand_id, cand_dist) in &sorted {
            if result.len() >= self.max_degree {
                break;
            }

            // Check α-RNG condition: keep candidate if it's not α-dominated
            // by any already-selected neighbor
            let dominated = result.iter().any(|&selected: &u32| {
                let inter_dist = l2_squared(&vectors[selected as usize], &vectors[*cand_id as usize]);
                alpha * inter_dist <= *cand_dist
            });

            if !dominated {
                result.push(*cand_id);
            }
        }

        result
    }

    /// Find the medoid (point closest to the centroid of all vectors)
    fn find_medoid(&self, vectors: &[Vec<f32>]) -> u32 {
        let n = vectors.len();
        let dim = vectors[0].len();

        // Compute centroid
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let inv_n = 1.0 / n as f32;
        for c in &mut centroid {
            *c *= inv_n;
        }

        // Find point closest to centroid
        let mut best_id = 0u32;
        let mut best_dist = f32::MAX;
        for (i, v) in vectors.iter().enumerate() {
            let d = l2_squared(v, &centroid);
            if d < best_dist {
                best_dist = d;
                best_id = i as u32;
            }
        }
        best_id
    }

    /// Initialize with random edges (for bootstrap)
    fn init_random_graph(&mut self, n: usize) {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let degree = self.max_degree.min(n - 1);

        for i in 0..n {
            let mut neighbors = Vec::with_capacity(degree);
            let mut attempts = 0;
            while neighbors.len() < degree && attempts < degree * 3 {
                let j = rng.gen_range(0..n) as u32;
                if j != i as u32 && !neighbors.contains(&j) {
                    neighbors.push(j);
                }
                attempts += 1;
            }
            self.neighbors[i] = neighbors;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_vamana_build_and_search() {
        let vectors = random_vectors(200, 32);
        let mut graph = VamanaGraph::new(200, 32, 64, 1.2);
        graph.build(&vectors).unwrap();

        // Search for a known vector — should find itself
        let (results, _) = graph.greedy_search(&vectors, &vectors[42], 10);
        assert!(!results.is_empty());
        // The query vector itself should be in top results
        assert!(results.contains(&42));
    }

    #[test]
    fn test_vamana_bounded_degree() {
        let vectors = random_vectors(100, 16);
        let max_degree = 8;
        let mut graph = VamanaGraph::new(100, max_degree, 32, 1.2);
        graph.build(&vectors).unwrap();

        for neighbors in &graph.neighbors {
            assert!(neighbors.len() <= max_degree);
        }
    }
}
