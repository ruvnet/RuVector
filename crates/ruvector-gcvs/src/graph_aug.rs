/// Variant 2 — GraphAugSearch.
///
/// Step 1: vector scan for seed candidates (top seed_k by cosine similarity).
/// Step 2: BFS through the graph up to `bfs_depth` hops, collecting additional candidates.
/// Step 3: re-rank all candidates by exact cosine similarity, return top-k.
///
/// Graph edges encode semantic relationships beyond raw embedding distance —
/// e.g., cross-cluster knowledge graph edges or agent memory associations.
/// This finds results that are reachable through the graph but not the nearest
/// vectors in embedding space.
use crate::{distance::cosine, graph::Graph, GcvsError, GcvsIndex, Hit, Result};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct GraphAugSearch {
    dim: usize,
    vectors: HashMap<usize, Vec<f32>>,
    graph: Graph,
    seed_k: usize,
    bfs_depth: usize,
}

impl GraphAugSearch {
    /// `seed_k`   — number of vector-scan seeds before BFS expansion.
    /// `bfs_depth` — maximum BFS hop depth from each seed.
    pub fn new(dim: usize, seed_k: usize, bfs_depth: usize) -> Self {
        Self {
            dim,
            vectors: HashMap::new(),
            graph: Graph::new(),
            seed_k,
            bfs_depth,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.graph.add_edge(from, to);
    }

    fn bfs_expand(&self, seeds: &[usize], max_depth: usize) -> HashSet<usize> {
        let mut visited: HashSet<usize> = seeds.iter().copied().collect();
        let mut queue: VecDeque<(usize, usize)> = seeds.iter().map(|&s| (s, 0usize)).collect();
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for &nb in self.graph.neighbours(node) {
                if visited.insert(nb) {
                    queue.push_back((nb, depth + 1));
                }
            }
        }
        visited
    }
}

impl GcvsIndex for GraphAugSearch {
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

        // Step 2: BFS expansion.
        let candidate_ids = self.bfs_expand(&seed_ids, self.bfs_depth);

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
        "GraphAugSearch (BFS expansion)"
    }
}
