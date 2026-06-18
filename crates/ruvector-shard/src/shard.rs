use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::ShardError;
use crate::graph::{cosine_distance, KnnGraph};

/// Which strategy was used to extract this shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShardVariant {
    /// BFS expansion from anchor nodes — geographic locality in graph space.
    Bfs,
    /// Highest cosine-similarity to anchor centroid — semantic locality.
    Coherence,
    /// Highest incoming-degree nodes — topological hubs (upper-layer simulation).
    Hub,
}

/// Metadata recorded at extraction time.
#[derive(Debug, Clone)]
pub struct ShardMeta {
    pub variant: ShardVariant,
    pub extraction_us: u64,
}

/// A self-contained slice of a proximity graph, portable and standalone.
pub struct Shard {
    pub variant: ShardVariant,
    pub dim: usize,
    /// Global node IDs included in this shard, in stable order.
    pub node_ids: Vec<u32>,
    /// Row-major vector data: `vectors[i * dim .. (i+1) * dim]` is `node_ids[i]`.
    pub vectors: Vec<f32>,
    /// Neighbor lists using *local* indices into `node_ids`.
    pub local_neighbors: Vec<Vec<u32>>,
    pub meta: ShardMeta,
}

impl Shard {
    /// Slice the vector for local index `local_idx`.
    #[inline]
    pub fn get_vector(&self, local_idx: usize) -> &[f32] {
        let start = local_idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Approximate heap bytes consumed by this shard.
    pub fn memory_bytes(&self) -> usize {
        let vec_bytes = self.node_ids.len() * self.dim * 4;
        let nb_bytes: usize = self.local_neighbors.iter().map(|nb| nb.len() * 4).sum();
        vec_bytes + nb_bytes
    }
}

/// Trait implemented by all shard extraction strategies.
pub trait ShardExtractor {
    fn extract(&self, graph: &KnnGraph, anchors: &[u32], budget: usize) -> Shard;
}

// ─── BFS Shard ───────────────────────────────────────────────────────────────

/// Geographic shard: BFS expansion from anchor nodes through graph edges.
pub struct BfsShard;

impl ShardExtractor for BfsShard {
    fn extract(&self, graph: &KnnGraph, anchors: &[u32], budget: usize) -> Shard {
        let t0 = std::time::Instant::now();
        let node_ids = bfs_extract(graph, anchors, budget);
        let extraction_us = t0.elapsed().as_micros() as u64;
        build_shard(ShardVariant::Bfs, graph, node_ids, extraction_us)
    }
}

fn bfs_extract(graph: &KnnGraph, anchors: &[u32], budget: usize) -> Vec<u32> {
    let mut visited: HashSet<u32> = HashSet::with_capacity(budget);
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut result: Vec<u32> = Vec::with_capacity(budget);

    for &a in anchors {
        if (a as usize) < graph.n && visited.insert(a) {
            queue.push_back(a);
        }
    }

    while let Some(node) = queue.pop_front() {
        if result.len() >= budget {
            break;
        }
        result.push(node);
        for &nb in &graph.neighbors[node as usize] {
            if visited.insert(nb) {
                queue.push_back(nb);
            }
        }
    }

    // Fill any remaining budget with unseen nodes (handles disconnected graphs).
    if result.len() < budget {
        for i in 0..graph.n as u32 {
            if !visited.contains(&i) {
                result.push(i);
                if result.len() >= budget {
                    break;
                }
            }
        }
    }

    result
}

// ─── Coherence Shard ─────────────────────────────────────────────────────────

/// Semantic shard: nodes most similar to the centroid of anchor vectors.
pub struct CoherenceShard;

impl ShardExtractor for CoherenceShard {
    fn extract(&self, graph: &KnnGraph, anchors: &[u32], budget: usize) -> Shard {
        let t0 = std::time::Instant::now();
        let node_ids = coherence_extract(graph, anchors, budget);
        let extraction_us = t0.elapsed().as_micros() as u64;
        build_shard(ShardVariant::Coherence, graph, node_ids, extraction_us)
    }
}

fn coherence_extract(graph: &KnnGraph, anchors: &[u32], budget: usize) -> Vec<u32> {
    let dim = graph.dim;
    let mut centroid = vec![0.0f32; dim];

    let valid_anchors: Vec<u32> = anchors
        .iter()
        .copied()
        .filter(|&a| (a as usize) < graph.n)
        .collect();
    let count = valid_anchors.len().max(1) as f32;

    for &a in &valid_anchors {
        let v = graph.get_vector(a as usize);
        for (c, &f) in centroid.iter_mut().zip(v) {
            *c += f / count;
        }
    }

    // Score each node by cosine similarity to centroid (= 1 - cosine_distance).
    let mut scores: Vec<(f32, u32)> = (0..graph.n as u32)
        .map(|i| {
            let sim = 1.0 - cosine_distance(graph.get_vector(i as usize), &centroid);
            (sim, i)
        })
        .collect();

    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(budget);
    scores.into_iter().map(|(_, id)| id).collect()
}

// ─── Hub Shard ───────────────────────────────────────────────────────────────

/// Topological shard: nodes with the highest incoming degree (graph hubs).
/// Hubs correspond to the upper layers of HNSW and provide broad coverage.
pub struct HubShard;

impl ShardExtractor for HubShard {
    fn extract(&self, graph: &KnnGraph, _anchors: &[u32], budget: usize) -> Shard {
        let t0 = std::time::Instant::now();
        let node_ids = hub_extract(graph, budget);
        let extraction_us = t0.elapsed().as_micros() as u64;
        build_shard(ShardVariant::Hub, graph, node_ids, extraction_us)
    }
}

fn hub_extract(graph: &KnnGraph, budget: usize) -> Vec<u32> {
    let degrees = graph.incoming_degree();
    let mut indexed: Vec<(u32, u32)> = degrees
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i as u32))
        .collect();
    indexed.sort_by(|a, b| b.0.cmp(&a.0));
    indexed.truncate(budget);
    indexed.into_iter().map(|(_, id)| id).collect()
}

// ─── Shared builder ──────────────────────────────────────────────────────────

fn build_shard(
    variant: ShardVariant,
    graph: &KnnGraph,
    node_ids: Vec<u32>,
    extraction_us: u64,
) -> Shard {
    let dim = graph.dim;

    // Copy vectors for shard nodes.
    let mut vectors = Vec::with_capacity(node_ids.len() * dim);
    for &id in &node_ids {
        vectors.extend_from_slice(graph.get_vector(id as usize));
    }

    // Build local → global index map for remapping neighbor lists.
    let id_to_local: HashMap<u32, u32> = node_ids
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local as u32))
        .collect();

    let local_neighbors: Vec<Vec<u32>> = node_ids
        .iter()
        .map(|&global_id| {
            graph.neighbors[global_id as usize]
                .iter()
                .filter_map(|&nb| id_to_local.get(&nb).copied())
                .collect()
        })
        .collect();

    Shard {
        variant,
        dim,
        node_ids,
        vectors,
        local_neighbors,
        meta: ShardMeta {
            variant,
            extraction_us,
        },
    }
}

// ─── Error convenience ───────────────────────────────────────────────────────

/// Validate anchors are within graph bounds.
pub fn validate_anchors(graph: &KnnGraph, anchors: &[u32]) -> Result<(), ShardError> {
    for &a in anchors {
        if a as usize >= graph.n {
            return Err(ShardError::AnchorOutOfRange(a));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphConfig;

    fn tiny_graph() -> KnnGraph {
        let vectors: Vec<f32> = (0..20).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        KnnGraph::build(vectors, 3, &GraphConfig { k_neighbors: 4 }).unwrap()
    }

    #[test]
    fn bfs_shard_exact_budget() {
        let g = tiny_graph();
        let shard = BfsShard.extract(&g, &[0], 5);
        assert_eq!(shard.node_ids.len(), 5);
        assert_eq!(shard.vectors.len(), 5 * 3);
        assert_eq!(shard.local_neighbors.len(), 5);
    }

    #[test]
    fn coherence_shard_exact_budget() {
        let g = tiny_graph();
        let shard = CoherenceShard.extract(&g, &[0], 5);
        assert_eq!(shard.node_ids.len(), 5);
    }

    #[test]
    fn hub_shard_exact_budget() {
        let g = tiny_graph();
        let shard = HubShard.extract(&g, &[], 5);
        assert_eq!(shard.node_ids.len(), 5);
    }

    #[test]
    fn local_neighbors_in_bounds() {
        let g = tiny_graph();
        let shard = BfsShard.extract(&g, &[0], 8);
        for nlist in &shard.local_neighbors {
            for &local in nlist {
                assert!((local as usize) < shard.node_ids.len());
            }
        }
    }
}
