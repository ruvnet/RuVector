use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Squared Euclidean distance — avoids sqrt for comparisons.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Wrapper so (distance, id) pairs are orderable in a BinaryHeap.
#[derive(PartialEq)]
struct HeapItem(f32, u32);

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // MAX-heap: larger distance → "greater" → sits at top of BinaryHeap.
        // results.peek() yields the worst (farthest) candidate for pruning.
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.1.cmp(&self.1))
    }
}

/// Flat Navigable Small-World graph — the base layer used by all ACORN variants.
///
/// During construction each inserted vector greedily finds its M nearest
/// current neighbours and links bidirectionally.  After all vectors are
/// inserted, `compress_neighbors` optionally expands every node's adjacency
/// list with neighbours-of-neighbours (the "ACORN-γ" trick) so that filtered
/// search can still navigate through rejected nodes.
pub struct NswGraph {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    /// adjacency list indexed by node id
    pub neighbors: Vec<Vec<u32>>,
    /// max neighbours per node after compression
    pub m_max: usize,
}

impl NswGraph {
    pub fn new(dim: usize, m: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            neighbors: Vec::new(),
            m_max: m,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Insert a vector and wire bidirectional edges using greedy candidate search.
    pub fn insert(&mut self, vec: Vec<f32>) -> u32 {
        let id = self.vectors.len() as u32;
        self.vectors.push(vec);
        self.neighbors.push(Vec::new());

        if id == 0 {
            return id;
        }

        // Find up to m_max nearest existing nodes
        let candidates = self.greedy_search_all(&self.vectors[id as usize].clone(), self.m_max * 2);

        let my_neighbors: Vec<u32> = candidates.iter().take(self.m_max).map(|(n, _)| *n).collect();

        // Bidirectional edges
        for &nb in &my_neighbors {
            let nb_vec = self.vectors[nb as usize].clone();
            let nb_list = &mut self.neighbors[nb as usize];
            if !nb_list.contains(&id) {
                nb_list.push(id);
                // Prune to m_max by distance from nb
                if nb_list.len() > self.m_max {
                    nb_list.sort_by(|&a, &b| {
                        l2_sq(&self.vectors[a as usize], &nb_vec)
                            .partial_cmp(&l2_sq(&self.vectors[b as usize], &nb_vec))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    nb_list.truncate(self.m_max);
                }
            }
        }

        self.neighbors[id as usize] = my_neighbors;
        id
    }

    /// ACORN-γ neighbour compression: add neighbours-of-neighbours so that
    /// filtered graph traversal remains connected even under selective filters.
    ///
    /// After compression each node stores up to `m_max * gamma` neighbours,
    /// sorted by distance to the node itself.
    pub fn compress_neighbors(&mut self, gamma: usize) {
        if gamma <= 1 {
            return;
        }
        let n = self.vectors.len();
        let target = self.m_max * gamma;

        // Collect second-hop candidates independently before mutating
        let second_hop: Vec<Vec<u32>> = (0..n)
            .map(|node| {
                let mut extras: Vec<u32> = Vec::new();
                for &nb in &self.neighbors[node] {
                    for &nn in &self.neighbors[nb as usize] {
                        if nn != node as u32 {
                            extras.push(nn);
                        }
                    }
                }
                extras
            })
            .collect();

        for node in 0..n {
            let mut all: Vec<u32> = self.neighbors[node].clone();
            all.extend(second_hop[node].iter().copied());
            all.sort_unstable();
            all.dedup();
            all.retain(|&id| id != node as u32);

            // Keep the closest `target` by L2² to this node
            let nv = self.vectors[node].clone();
            all.sort_by(|&a, &b| {
                l2_sq(&self.vectors[a as usize], &nv)
                    .partial_cmp(&l2_sq(&self.vectors[b as usize], &nv))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all.truncate(target);
            self.neighbors[node] = all;
        }
    }

    // ── Search algorithms ────────────────────────────────────────────────────

    /// Unfiltered greedy search: collects up to `ef` nearest candidates.
    fn greedy_search_all(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
        let entry = 0u32;
        let entry_dist = l2_sq(query, &self.vectors[0]);

        let mut visited = HashSet::new();
        visited.insert(entry);

        // min-heap of (dist, id) — exploration frontier
        let mut candidates: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        candidates.push(Reverse(HeapItem(entry_dist, entry)));

        // max-heap of (dist, id) — results window size ef
        let mut results: BinaryHeap<HeapItem> = BinaryHeap::new();
        results.push(HeapItem(entry_dist, entry));

        while let Some(Reverse(HeapItem(d, node))) = candidates.pop() {
            if results.len() >= ef {
                if let Some(HeapItem(worst, _)) = results.peek() {
                    if d > *worst {
                        break;
                    }
                }
            }
            for &nb in &self.neighbors[node as usize] {
                if visited.insert(nb) {
                    let nd = l2_sq(query, &self.vectors[nb as usize]);
                    candidates.push(Reverse(HeapItem(nd, nb)));
                    results.push(HeapItem(nd, nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> =
            results.into_iter().map(|HeapItem(d, id)| (id, d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// **PostFilter (baseline)**: search without any filter, then retain only
    /// candidates that satisfy the predicate.  Recall degrades sharply when
    /// the filter is selective (few matching nodes).
    pub fn search_postfilter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        let all = self.greedy_search_all(query, ef);
        all.into_iter()
            .filter(|(id, _)| filter(*id))
            .take(k)
            .collect()
    }

    /// **ACORN-1 (strict)**: only expands nodes that satisfy the filter.
    /// Fast when selectivity is high (50 %), degrades when filter is tight (1 %).
    pub fn search_acorn1(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }

        // Find first filter-passing entry
        let entry = match (0..self.vectors.len() as u32).find(|&id| filter(id)) {
            Some(e) => e,
            None => return vec![],
        };

        let entry_dist = l2_sq(query, &self.vectors[entry as usize]);
        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        candidates.push(Reverse(HeapItem(entry_dist, entry)));

        let mut results: BinaryHeap<HeapItem> = BinaryHeap::new();
        results.push(HeapItem(entry_dist, entry));

        while let Some(Reverse(HeapItem(d, node))) = candidates.pop() {
            if results.len() >= ef {
                if let Some(HeapItem(worst, _)) = results.peek() {
                    if d > *worst {
                        break;
                    }
                }
            }
            for &nb in &self.neighbors[node as usize] {
                if visited.insert(nb) && filter(nb) {
                    let nd = l2_sq(query, &self.vectors[nb as usize]);
                    candidates.push(Reverse(HeapItem(nd, nb)));
                    results.push(HeapItem(nd, nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> =
            results.into_iter().map(|HeapItem(d, id)| (id, d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    /// **ACORN-γ (full)**: navigates through all nodes but only counts
    /// filter-passing nodes in the result window.  With neighbour compression
    /// the graph remains connected under any predicate.
    pub fn search_acorn_gamma(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }

        let entry = 0u32;
        let entry_dist = l2_sq(query, &self.vectors[0]);

        let mut visited = HashSet::new();
        visited.insert(entry);

        // All nodes go into candidates (for navigation)
        let mut candidates: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        candidates.push(Reverse(HeapItem(entry_dist, entry)));

        // Only filter-passing nodes go into results
        let mut results: BinaryHeap<HeapItem> = BinaryHeap::new();
        if filter(entry) {
            results.push(HeapItem(entry_dist, entry));
        }

        while let Some(Reverse(HeapItem(d, node))) = candidates.pop() {
            // Stop when frontier is worse than the ef-th result
            if results.len() >= ef {
                if let Some(HeapItem(worst, _)) = results.peek() {
                    if d > *worst {
                        break;
                    }
                }
            }
            for &nb in &self.neighbors[node as usize] {
                if visited.insert(nb) {
                    let nd = l2_sq(query, &self.vectors[nb as usize]);
                    // Always add to candidates for navigation
                    candidates.push(Reverse(HeapItem(nd, nb)));
                    // Only add to results if it passes the filter
                    if filter(nb) {
                        results.push(HeapItem(nd, nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> =
            results.into_iter().map(|HeapItem(d, id)| (id, d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    /// Brute-force filtered scan — used to compute ground-truth recall.
    pub fn brute_force(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = (0..self.vectors.len() as u32)
            .filter(|&id| filter(id))
            .map(|id| (id, l2_sq(query, &self.vectors[id as usize])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
