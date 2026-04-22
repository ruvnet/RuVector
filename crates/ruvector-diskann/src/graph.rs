//! Vamana graph construction with α-robust pruning
//!
//! Optimized with:
//! - FlatVectors (contiguous memory, cache-friendly)
//! - VisitedSet (O(1) clear via generation counter)
//! - Rayon-parallel medoid finding

use crate::distance::{l2_squared, FlatVectors, VisitedSet};
use crate::error::{DiskAnnError, Result};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

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
    pub neighbors: Vec<Vec<u32>>,
    pub medoid: u32,
    pub max_degree: usize,
    pub build_beam: usize,
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

    /// Build the Vamana graph over flat vector storage
    pub fn build(&mut self, vectors: &FlatVectors) -> Result<()> {
        let n = vectors.len();
        if n == 0 {
            return Err(DiskAnnError::Empty);
        }

        self.medoid = self.find_medoid_parallel(vectors);
        self.init_random_graph(n);

        let passes = if self.alpha > 1.0 { 2 } else { 1 };
        for pass in 0..passes {
            let alpha = if pass == 0 { 1.0 } else { self.alpha };

            let mut order: Vec<u32> = (0..n as u32).collect();
            {
                use rand::prelude::*;
                order.shuffle(&mut rand::thread_rng());
            }

            // Reusable visited set (O(1) clear per search)
            let mut visited = VisitedSet::new(n);

            for &node in &order {
                let (candidates, _) = self.greedy_search_fast(
                    vectors,
                    vectors.get(node as usize),
                    self.build_beam,
                    &mut visited,
                );

                let pruned = self.robust_prune(vectors, node, &candidates, alpha);
                self.neighbors[node as usize] = pruned.clone();

                for &neighbor in &pruned {
                    let nid = neighbor as usize;
                    if !self.neighbors[nid].contains(&node) {
                        if self.neighbors[nid].len() < self.max_degree {
                            self.neighbors[nid].push(node);
                        } else {
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

    /// Greedy beam search with reusable VisitedSet (zero-alloc per query)
    pub fn greedy_search_fast(
        &self,
        vectors: &FlatVectors,
        query: &[f32],
        beam_width: usize,
        visited: &mut VisitedSet,
    ) -> (Vec<u32>, usize) {
        visited.clear();

        let mut candidates = BinaryHeap::<Candidate>::new();
        let mut best = BinaryHeap::<MaxCandidate>::new();

        let start = self.medoid;
        let start_dist = l2_squared(vectors.get(start as usize), query);
        candidates.push(Candidate {
            id: start,
            distance: start_dist,
        });
        best.push(MaxCandidate {
            id: start,
            distance: start_dist,
        });
        visited.insert(start);

        let mut visit_count = 1usize;

        while let Some(current) = candidates.pop() {
            if best.len() >= beam_width {
                if let Some(worst) = best.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }

            for &neighbor in &self.neighbors[current.id as usize] {
                if visited.contains(neighbor) {
                    continue;
                }
                visited.insert(neighbor);
                visit_count += 1;

                let dist = l2_squared(vectors.get(neighbor as usize), query);

                let dominated =
                    best.len() >= beam_width && best.peek().map_or(false, |w| dist >= w.distance);

                if !dominated {
                    candidates.push(Candidate {
                        id: neighbor,
                        distance: dist,
                    });
                    best.push(MaxCandidate {
                        id: neighbor,
                        distance: dist,
                    });
                    if best.len() > beam_width {
                        best.pop();
                    }
                }
            }
        }

        let mut result: Vec<(u32, f32)> = best.into_iter().map(|c| (c.id, c.distance)).collect();
        result.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let ids: Vec<u32> = result.into_iter().map(|(id, _)| id).collect();

        (ids, visit_count)
    }

    /// Public search entry point (allocates its own VisitedSet)
    pub fn greedy_search(
        &self,
        vectors: &FlatVectors,
        query: &[f32],
        beam_width: usize,
    ) -> (Vec<u32>, usize) {
        let mut visited = VisitedSet::new(vectors.len());
        self.greedy_search_fast(vectors, query, beam_width, &mut visited)
    }

    fn robust_prune(
        &self,
        vectors: &FlatVectors,
        node: u32,
        candidates: &[u32],
        alpha: f32,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let node_vec = vectors.get(node as usize);
        let mut sorted: Vec<(u32, f32)> = candidates
            .iter()
            .filter(|&&c| c != node)
            .map(|&c| (c, l2_squared(vectors.get(c as usize), node_vec)))
            .collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result = Vec::with_capacity(self.max_degree);
        for (cand_id, cand_dist) in &sorted {
            if result.len() >= self.max_degree {
                break;
            }
            let dominated = result.iter().any(|&selected: &u32| {
                let inter_dist = l2_squared(
                    vectors.get(selected as usize),
                    vectors.get(*cand_id as usize),
                );
                alpha * inter_dist <= *cand_dist
            });
            if !dominated {
                result.push(*cand_id);
            }
        }
        result
    }

    /// Parallel medoid finding using rayon
    fn find_medoid_parallel(&self, vectors: &FlatVectors) -> u32 {
        let n = vectors.len();
        let dim = vectors.dim;

        // Compute centroid in parallel
        let centroid: Vec<f32> = (0..dim)
            .into_par_iter()
            .map(|d| {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += vectors.get(i)[d];
                }
                sum / n as f32
            })
            .collect();

        // Find closest point to centroid in parallel
        (0..n as u32)
            .into_par_iter()
            .map(|i| (i, l2_squared(vectors.get(i as usize), &centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(id, _)| id)
            .unwrap_or(0)
    }

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

    fn random_flat(n: usize, dim: usize) -> FlatVectors {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let mut fv = FlatVectors::with_capacity(dim, n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            fv.push(&v);
        }
        fv
    }

    #[test]
    fn test_vamana_build_and_search() {
        let vectors = random_flat(200, 32);
        let mut graph = VamanaGraph::new(200, 32, 64, 1.2);
        graph.build(&vectors).unwrap();

        let (results, _) = graph.greedy_search(&vectors, vectors.get(42), 10);
        assert!(!results.is_empty());
        assert!(results.contains(&42));
    }

    #[test]
    fn test_vamana_bounded_degree() {
        let vectors = random_flat(100, 16);
        let mut graph = VamanaGraph::new(100, 8, 32, 1.2);
        graph.build(&vectors).unwrap();

        for neighbors in &graph.neighbors {
            assert!(neighbors.len() <= 8);
        }
    }
}
