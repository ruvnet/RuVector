//! FrozenSegment: an immutable Navigable Small World (NSW) graph built over
//! a batch of vectors. Used for L1 and L2 tiers of the LSM-ANN index.
//!
//! Construction: greedy k-NN insertion — each new vector connects to the M
//! nearest already-inserted vectors (found via brute-force scan during build).
//! Search: beam search (priority-queue greedy walk) from a random entry point.
//!
//! This is a single-layer NSW (no hierarchy), which is appropriate for
//! segments of 500 – 50,000 vectors where build time matters more than
//! asymptotic search complexity.

use std::collections::BinaryHeap;

use crate::sq_dist;

/// An immutable NSW graph over a frozen set of vectors.
pub struct FrozenSegment {
    /// Stored vectors indexed by their position in this segment.
    pub(crate) vectors: Vec<(u64, Vec<f32>)>,
    /// Adjacency list: `graph[i]` contains indices into `vectors` for node i.
    pub(crate) graph: Vec<Vec<usize>>,
    /// Entry point for graph traversal (the first inserted node).
    entry: usize,
    /// Beam width for search.
    ef_search: usize,
}

impl FrozenSegment {
    /// Build a frozen NSW segment from `data` using the given parameters.
    ///
    /// `m` — max neighbours per node during construction.
    /// `ef_construction` — beam width used while adding each node.
    /// `ef_search` — beam width stored for query-time use.
    pub fn build(
        data: Vec<(u64, Vec<f32>)>,
        m: usize,
        _ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        let n = data.len();
        let mut graph: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Construction uses brute-force k-NN to guarantee correct neighbourhood
        // selection regardless of graph topology state. O(N²·D) but run once.
        for i in 1..n {
            let vi = &data[i].1;

            // Find m exact nearest among already-inserted nodes 0..i-1.
            let mut dists: Vec<(usize, f32)> =
                (0..i).map(|j| (j, sq_dist(vi, &data[j].1))).collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let m_actual = m.min(dists.len());
            let neighbours = &dists[..m_actual];

            for &(j, _) in neighbours {
                graph[i].push(j);
                graph[j].push(i);
                // Prune j's adjacency list to m if needed.
                if graph[j].len() > m {
                    let vj = &data[j].1;
                    let mut adj: Vec<(usize, f32)> = graph[j]
                        .iter()
                        .map(|&k| (k, sq_dist(vj, &data[k].1)))
                        .collect();
                    adj.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    adj.truncate(m);
                    graph[j] = adj.into_iter().map(|(k, _)| k).collect();
                }
            }
        }

        let _ = m;
        Self {
            vectors: data,
            graph,
            entry: 0,
            ef_search,
        }
    }

    /// Search for the k nearest neighbours of `query` within this segment.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let ef = self.ef_search.max(k);
        let raw = greedy_search_internal(&self.vectors, &self.graph, query, self.entry, ef);

        raw.into_iter()
            .take(k)
            .map(|(idx, dist)| (self.vectors[idx].0, dist))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let vec_bytes: usize = self.vectors.iter().map(|(_, v)| 8 + v.len() * 4).sum();
        let graph_bytes: usize = self.graph.iter().map(|adj| adj.len() * 8).sum();
        vec_bytes + graph_bytes
    }

    /// Number of graph edges (for diagnostics).
    pub fn edge_count(&self) -> usize {
        self.graph.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Min-heap entry: smallest distance pops first (for the exploration frontier).
#[derive(Clone, PartialEq)]
struct ClosestFirst {
    neg_dist: f32, // negated so BinaryHeap (max-heap) pops closest
    idx: usize,
}
impl Eq for ClosestFirst {}
impl PartialOrd for ClosestFirst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ClosestFirst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // larger neg_dist = smaller actual dist → front of max-heap = closest node
        self.neg_dist
            .partial_cmp(&other.neg_dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Max-heap entry: largest distance pops first (for the result set eviction).
#[derive(Clone, PartialEq)]
struct FarthestFirst {
    dist: f32,
    idx: usize,
}
impl Eq for FarthestFirst {}
impl PartialOrd for FarthestFirst {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FarthestFirst {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // larger dist → front of max-heap = farthest node, correct for eviction
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Greedy beam search within the NSW graph built over `vectors`.
///
/// - `frontier`: min-heap by distance (explore closest unvisited node first).
/// - `best`: max-heap by distance (size-bounded to `ef`; evicts the farthest when full).
///
/// Returns `(node_index, squared_distance)` pairs sorted by ascending distance.
fn greedy_search_internal(
    vectors: &[(u64, Vec<f32>)],
    graph: &[Vec<usize>],
    query: &[f32],
    entry_idx: usize,
    ef: usize,
) -> Vec<(usize, f32)> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let entry = entry_idx.min(vectors.len() - 1);

    let mut visited = std::collections::HashSet::new();
    visited.insert(entry);

    let entry_dist = sq_dist(&vectors[entry].1, query);

    let mut frontier: BinaryHeap<ClosestFirst> = BinaryHeap::new();
    let mut best: BinaryHeap<FarthestFirst> = BinaryHeap::new();

    frontier.push(ClosestFirst {
        neg_dist: -entry_dist,
        idx: entry,
    });
    best.push(FarthestFirst {
        dist: entry_dist,
        idx: entry,
    });

    while let Some(curr) = frontier.pop() {
        let curr_dist = -curr.neg_dist;
        // Terminate early: current frontier node is farther than the worst in best.
        let worst_best_dist = best.peek().map(|e| e.dist).unwrap_or(f32::MAX);
        if curr_dist > worst_best_dist && best.len() >= ef {
            break;
        }

        for &neighbour in &graph[curr.idx] {
            if visited.contains(&neighbour) {
                continue;
            }
            visited.insert(neighbour);

            let d = sq_dist(&vectors[neighbour].1, query);
            let worst = best.peek().map(|e| e.dist).unwrap_or(f32::MAX);

            if d < worst || best.len() < ef {
                frontier.push(ClosestFirst {
                    neg_dist: -d,
                    idx: neighbour,
                });
                best.push(FarthestFirst {
                    dist: d,
                    idx: neighbour,
                });
                if best.len() > ef {
                    best.pop(); // evicts farthest — correct
                }
            }
        }
    }

    let mut results: Vec<(usize, f32)> = best.into_iter().map(|e| (e.idx, e.dist)).collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results
}
