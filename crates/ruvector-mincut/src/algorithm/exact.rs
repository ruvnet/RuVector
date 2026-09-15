//! Sparse Stoer-Wagner global minimum cut with a matching vertex partition.
//!
//! Each phase uses a maximum-adjacency heap. Storage is O(n + m), avoiding
//! a dense n-by-n matrix. Full recomputation is polynomial, not subpolynomial.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

#[derive(Clone, Copy)]
struct Candidate {
    weight: f64,
    vertex: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.weight.total_cmp(&other.weight) == Ordering::Equal && self.vertex == other.vertex
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
        self.weight
            .total_cmp(&other.weight)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

/// Endpoints index `0..n`; callers supply finite, nonnegative weights.
/// Returns infinity for fewer than two vertices, matching DynamicMinCut.
pub(super) fn minimum_cut(n: usize, edges: &[(usize, usize, f64)]) -> (f64, Vec<usize>) {
    if n < 2 {
        return (f64::INFINITY, (0..n).collect());
    }
    let mut adjacency = vec![BTreeMap::<usize, f64>::new(); n];
    for &(u, v, weight) in edges {
        *adjacency[u].entry(v).or_default() += weight;
        *adjacency[v].entry(u).or_default() += weight;
    }
    // Disconnected graphs and trees are common in failure monitoring. Resolve
    // them in linear traversal work rather than running contraction phases.
    let mut seen = vec![false; n];
    let mut stack = vec![0];
    seen[0] = true;
    let mut component = Vec::new();
    while let Some(v) = stack.pop() {
        component.push(v);
        for &neighbor in adjacency[v].keys() {
            if !seen[neighbor] {
                seen[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    if component.len() != n {
        return (0.0, component);
    }
    if edges.len() == n - 1 {
        let &(u, v, weight) = edges.iter().min_by(|a, b| {
            a.2.total_cmp(&b.2).then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        }).expect("connected tree has an edge");
        seen.fill(false);
        stack.push(u);
        seen[u] = true;
        component.clear();
        while let Some(x) = stack.pop() {
            component.push(x);
            for &y in adjacency[x].keys() {
                if (x == u && y == v) || (x == v && y == u) { continue; }
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        return (weight, component);
    }
    let mut active = vec![true; n];
    let mut groups: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
    let mut best = f64::INFINITY;
    let mut best_side = vec![0];
    // Reuse per-phase work buffers rather than allocating a traversal per cut.
    let mut weights = vec![0.0; n];
    let mut added = vec![false; n];
    let mut heap = BinaryHeap::new();

    for remaining in (2..=n).rev() {
        weights.fill(0.0);
        added.fill(false);
        heap.clear();
        for v in 0..n {
            if active[v] {
                heap.push(Candidate { weight: 0.0, vertex: v });
            }
        }
        let mut previous = 0;
        for step in 0..remaining {
            let v = loop {
                let candidate = heap.pop().expect("active vertex has a heap entry");
                if !added[candidate.vertex] && candidate.weight == weights[candidate.vertex] {
                    break candidate.vertex;
                }
            };
            if step + 1 == remaining {
                if weights[v] < best {
                    best = weights[v];
                    best_side.clone_from(&groups[v]);
                }
                if best == 0.0 {
                    return (best, best_side);
                }
                // Contract the last vertex into the penultimate vertex.
                let neighbors = std::mem::take(&mut adjacency[v]);
                for (neighbor, weight) in neighbors {
                    adjacency[neighbor].remove(&v);
                    if neighbor != previous {
                        *adjacency[previous].entry(neighbor).or_default() += weight;
                        *adjacency[neighbor].entry(previous).or_default() += weight;
                    }
                }
                let merged = std::mem::take(&mut groups[v]);
                groups[previous].extend(merged);
                active[v] = false;
                break;
            }
            added[v] = true;
            for (&neighbor, &weight) in &adjacency[v] {
                if !added[neighbor] {
                    weights[neighbor] += weight;
                    heap.push(Candidate { weight: weights[neighbor], vertex: neighbor });
                }
            }
            previous = v;
        }
    }
    (best, best_side)
}
