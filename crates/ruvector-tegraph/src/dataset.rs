use crate::{
    normalize,
    types::{EdgeType, Node, TypedEdge},
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Dataset generation parameters.
pub struct DatasetConfig {
    /// Number of synthetic documents (clusters).
    pub n_docs: usize,
    /// Nodes per document.
    pub nodes_per_doc: usize,
    /// Vector dimensionality.
    pub dims: usize,
    /// Within-document SameDocument edges per node.
    pub same_doc_edges: usize,
    /// Cross-document References edges per node.
    pub ref_edges: usize,
    /// CoOccurs edges per node (any pair).
    pub cooccur_edges: usize,
    /// PRNG seed for full reproducibility.
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            n_docs: 100,
            nodes_per_doc: 50,
            dims: 128,
            same_doc_edges: 4,
            ref_edges: 2,
            cooccur_edges: 2,
            seed: 0xc0de_cafe_1234_5678,
        }
    }
}

impl DatasetConfig {
    pub fn total_nodes(&self) -> usize {
        self.n_docs * self.nodes_per_doc
    }
}

/// Build a deterministic node list with typed edges.
///
/// Each document gets a random unit-vector centroid; per-document nodes are
/// centroid + Gaussian noise (std=0.15), then re-normalised.  Four edge types
/// are seeded: SameDocument (intra-cluster), References (cross-cluster),
/// CoOccurs (arbitrary pairs), and Temporal (sequential within-cluster).
pub fn generate(cfg: &DatasetConfig) -> Vec<Node> {
    let total = cfg.total_nodes();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let noise  = Normal::new(0.0f32, 0.15).unwrap();

    // Document centroids — random unit vectors.
    let centroids: Vec<Vec<f32>> = (0..cfg.n_docs)
        .map(|_| {
            let mut v: Vec<f32> = (0..cfg.dims).map(|_| normal.sample(&mut rng)).collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Per-node vectors: centroid + noise, then normalised.
    let mut nodes: Vec<Node> = (0..total)
        .map(|id| {
            let doc_id = id / cfg.nodes_per_doc;
            let mut v: Vec<f32> = centroids[doc_id]
                .iter()
                .map(|&c| c + noise.sample(&mut rng))
                .collect();
            normalize(&mut v);
            Node { id, vector: v, typed_edges: Vec::new(), doc_id }
        })
        .collect();

    // ── SameDocument edges (intra-cluster) ────────────────────────────────────
    for doc in 0..cfg.n_docs {
        let start = doc * cfg.nodes_per_doc;
        let end   = start + cfg.nodes_per_doc;
        for nid in start..end {
            for _ in 0..cfg.same_doc_edges {
                let mut tgt = rng.gen_range(start..end);
                while tgt == nid { tgt = rng.gen_range(start..end); }
                nodes[nid].typed_edges.push(TypedEdge {
                    target: tgt,
                    edge_type: EdgeType::SameDocument,
                    weight: 0.90,
                });
            }
        }
    }

    // ── References edges (cross-cluster) ─────────────────────────────────────
    for nid in 0..total {
        for _ in 0..cfg.ref_edges {
            let tgt = rng.gen_range(0..total);
            if tgt != nid && nodes[tgt].doc_id != nodes[nid].doc_id {
                nodes[nid].typed_edges.push(TypedEdge {
                    target: tgt,
                    edge_type: EdgeType::References,
                    weight: rng.gen_range(0.50_f32..0.90),
                });
            }
        }
    }

    // ── CoOccurs edges (arbitrary) ───────────────────────────────────────────
    for nid in 0..total {
        for _ in 0..cfg.cooccur_edges {
            let tgt = rng.gen_range(0..total);
            if tgt != nid {
                nodes[nid].typed_edges.push(TypedEdge {
                    target: tgt,
                    edge_type: EdgeType::CoOccurs,
                    weight: rng.gen_range(0.30_f32..0.80),
                });
            }
        }
    }

    // ── Temporal edges (sequential within cluster) ───────────────────────────
    for doc in 0..cfg.n_docs {
        let start = doc * cfg.nodes_per_doc;
        let end   = start + cfg.nodes_per_doc - 1;
        for i in start..end {
            nodes[i].typed_edges.push(TypedEdge {
                target: i + 1,
                edge_type: EdgeType::Temporal,
                weight: 0.70,
            });
        }
    }

    nodes
}

/// Generate n random unit-vector query vectors.
pub fn generate_queries(dims: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xffff_ffff);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| normal.sample(&mut rng)).collect();
            normalize(&mut v);
            v
        })
        .collect()
}
