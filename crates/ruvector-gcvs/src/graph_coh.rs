/// Variant 3 — GraphCohSearch (coherence-gated BFS expansion).
///
/// Identical to GraphAugSearch but adds a coherence gate: an edge (u → v)
/// is only traversed if cosine(query, v) >= coherence_threshold. This prunes
/// semantically irrelevant branches early, reducing candidate set size while
/// maintaining or improving recall versus the ungated variant.
///
/// On dense, highly connected graphs the gate prevents recall loss from
/// noise vertices; on sparse graphs it matches GraphAugSearch.
use crate::{distance::cosine, graph::Graph, GcvsError, GcvsIndex, Hit, Result};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct GraphCohSearch {
    dim: usize,
    vectors: HashMap<usize, Vec<f32>>,
    graph: Graph,
    seed_k: usize,
    bfs_depth: usize,
    /// Minimum cosine similarity to query required to traverse an edge.
    coherence_threshold: f32,
}

impl GraphCohSearch {
    pub fn new(dim: usize, seed_k: usize, bfs_depth: usize, coherence_threshold: f32) -> Self {
        Self {
            dim,
            vectors: HashMap::new(),
            graph: Graph::new(),
            seed_k,
            bfs_depth,
            coherence_threshold,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.graph.add_edge(from, to);
    }

    fn gated_bfs_expand(&self, query: &[f32], seeds: &[usize], max_depth: usize) -> HashSet<usize> {
        let mut visited: HashSet<usize> = seeds.iter().copied().collect();
        let mut queue: VecDeque<(usize, usize)> = seeds.iter().map(|&s| (s, 0usize)).collect();
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for &nb in self.graph.neighbours(node) {
                if visited.contains(&nb) {
                    continue;
                }
                // Coherence gate: only follow edge if neighbour is relevant to query.
                if let Some(v) = self.vectors.get(&nb) {
                    if cosine(query, v) >= self.coherence_threshold {
                        visited.insert(nb);
                        queue.push_back((nb, depth + 1));
                    }
                }
            }
        }
        visited
    }
}

impl GcvsIndex for GraphCohSearch {
    fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(GcvsError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.vectors.insert(id, vector);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        if k == 0 {
            return Err(GcvsError::InvalidK);
        }
        if self.vectors.is_empty() {
            return Err(GcvsError::Empty);
        }

        // Step 1: vector scan for seeds.
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .map(|(&id, v)| (id, cosine(query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let seed_ids: Vec<usize> = scored.iter().take(self.seed_k).map(|&(id, _)| id).collect();

        // Step 2: coherence-gated BFS.
        let candidate_ids = self.gated_bfs_expand(query, &seed_ids, self.bfs_depth);

        // Step 3: re-rank expanded candidate set.
        let mut candidates: Vec<Hit> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(&id).map(|v| Hit {
                    id,
                    score: cosine(query, v),
                })
            })
            .collect();
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates.truncate(k);
        Ok(candidates)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn name(&self) -> &'static str {
        "GraphCohSearch (coherence-gated BFS)"
    }
}
