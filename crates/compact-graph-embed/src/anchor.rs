use std::collections::VecDeque;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::error::EmbedError;
use crate::graph::CsrGraph;
use crate::traits::{NodeTokenizer, Token};

/// Tokenizes graph nodes using BFS from selected anchor nodes.
///
/// Each node is represented by a short list of (anchor_idx, dist) tokens,
/// where dist is the BFS hop distance from the node to each anchor (clamped
/// to 1..=max_dist so that anchor self-tokens map to a valid embedding slot).
///
/// Token storage is compacted: tokens are stored as a flat `Vec<(u8, u8)>`
/// (anchor_idx as u8, dist as u8) with per-node offsets, giving a much
/// smaller memory footprint than `Vec<Vec<Token>>`.
pub struct AnchorTokenizer {
    /// Flat packed token storage: (anchor_as_u8, dist_as_u8) pairs.
    /// Supports up to 255 anchors.
    packed: Vec<(u8, u8)>,
    /// Per-node start index into `packed` (len = num_nodes + 1 for easy range).
    offsets: Vec<u32>,
    num_nodes: usize,
    num_anchors: usize,
    max_dist: u8,
}

impl AnchorTokenizer {
    /// Build an AnchorTokenizer by selecting `k` anchors and running BFS.
    ///
    /// Anchor selection: nodes sorted by degree descending, ties broken by a
    /// deterministic seeded shuffle. BFS explores up to `max_dist` hops.
    ///
    /// Dist=0 tokens (anchor nodes themselves) are clamped to dist=1 so that
    /// all tokens map to a valid embedding slot in the storage table.
    pub fn new(graph: &CsrGraph, k: usize, max_dist: u8, seed: u64) -> Self {
        assert!(k <= 255, "num_anchors must fit in u8 (<=255)");
        assert!(max_dist >= 1, "max_dist must be at least 1");

        let num_nodes = graph.num_nodes();
        let k = k.min(num_nodes);

        // --- Step 1: Select anchors ---
        let anchors = select_anchors(graph, k, seed);

        // --- Step 2: BFS from each anchor up to max_dist hops ---
        // dist_table[node * k + anchor_idx] = shortest BFS distance (u8::MAX = unreached)
        let mut dist_table: Vec<u8> = vec![u8::MAX; num_nodes * k];

        for (anchor_idx, &anchor_node) in anchors.iter().enumerate() {
            bfs_from_anchor(graph, anchor_node, anchor_idx, k, max_dist, &mut dist_table);
        }

        // --- Step 3: Build compact token lists ---
        let mut packed: Vec<(u8, u8)> = Vec::with_capacity(num_nodes * k.min(8));
        let mut offsets: Vec<u32> = Vec::with_capacity(num_nodes + 1);
        offsets.push(0);

        for node in 0..num_nodes {
            let start = packed.len();
            let row = &dist_table[node * k..(node + 1) * k];

            let mut any_reachable = false;
            for (anchor_idx, &d) in row.iter().enumerate() {
                if d != u8::MAX && d <= max_dist {
                    // Clamp dist=0 to 1 (anchor self-token maps to nearest slot)
                    let effective_dist = d.max(1);
                    packed.push((anchor_idx as u8, effective_dist));
                    any_reachable = true;
                }
            }

            // Fallback: if unreachable, assign all anchors at dist=max_dist
            if !any_reachable {
                for anchor_idx in 0..k {
                    packed.push((anchor_idx as u8, max_dist));
                }
            }

            // Sort by anchor_idx for determinism (tokens added in anchor order so already sorted)
            // They are already in anchor_idx order since we iterate 0..k, but sort to be safe.
            packed[start..].sort_unstable_by_key(|t| t.0);
            offsets.push(packed.len() as u32);
        }

        AnchorTokenizer {
            packed,
            offsets,
            num_nodes,
            num_anchors: k,
            max_dist,
        }
    }

    /// Returns a reference to the raw packed tokens for a node.
    pub fn tokens_packed(&self, node_id: usize) -> Option<&[(u8, u8)]> {
        if node_id >= self.num_nodes {
            return None;
        }
        let start = self.offsets[node_id] as usize;
        let end = self.offsets[node_id + 1] as usize;
        Some(&self.packed[start..end])
    }

    /// Estimated RAM consumed by the compact token storage (bytes).
    pub fn byte_size(&self) -> usize {
        self.packed.len() * 2              // 2 bytes per (u8, u8) token
            + self.offsets.len() * 4       // 4 bytes per u32 offset
            + std::mem::size_of::<Self>()  // struct fields
    }
}

impl NodeTokenizer for AnchorTokenizer {
    fn tokenize(&self, node_id: usize) -> Result<Vec<Token>, EmbedError> {
        if node_id >= self.num_nodes {
            return Err(EmbedError::NodeOutOfRange(node_id));
        }
        let start = self.offsets[node_id] as usize;
        let end = self.offsets[node_id + 1] as usize;
        let tokens = self.packed[start..end]
            .iter()
            .map(|&(a, d)| Token {
                anchor_idx: a as u32,
                dist: d,
            })
            .collect();
        Ok(tokens)
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn num_anchors(&self) -> usize {
        self.num_anchors
    }

    fn max_dist(&self) -> u8 {
        self.max_dist
    }
}

// ---------------------------------------------------------------------------
// Helper: select k anchor nodes
// ---------------------------------------------------------------------------

fn select_anchors(graph: &CsrGraph, k: usize, seed: u64) -> Vec<usize> {
    let num_nodes = graph.num_nodes();

    // Collect (degree, node_id) pairs
    let mut nodes: Vec<(usize, usize)> = (0..num_nodes)
        .map(|n| (graph.degree(n), n))
        .collect();

    // Sort descending by degree
    nodes.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Group nodes by degree, shuffle within each group using seeded RNG
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut i = 0;
    while i < nodes.len() {
        let deg = nodes[i].0;
        let mut j = i + 1;
        while j < nodes.len() && nodes[j].0 == deg {
            j += 1;
        }
        nodes[i..j].shuffle(&mut rng);
        i = j;
    }

    nodes.iter().take(k).map(|&(_, n)| n).collect()
}

// ---------------------------------------------------------------------------
// Helper: BFS from a single anchor
// ---------------------------------------------------------------------------

fn bfs_from_anchor(
    graph: &CsrGraph,
    anchor_node: usize,
    anchor_idx: usize,
    k: usize,
    max_dist: u8,
    dist_table: &mut [u8],
) {
    let num_nodes = graph.num_nodes();
    let mut visited: Vec<bool> = vec![false; num_nodes];
    let mut queue: VecDeque<(usize, u8)> = VecDeque::new();

    // Anchor itself at dist=0 (will be clamped to 1 in token building)
    dist_table[anchor_node * k + anchor_idx] = 0;
    visited[anchor_node] = true;
    queue.push_back((anchor_node, 0));

    while let Some((node, dist)) = queue.pop_front() {
        if dist >= max_dist {
            continue;
        }
        for &nbr in graph.neighbors(node) {
            if !visited[nbr] {
                visited[nbr] = true;
                let new_dist = dist + 1;
                dist_table[nbr * k + anchor_idx] = new_dist;
                queue.push_back((nbr, new_dist));
            }
        }
    }
}
