//! Three vector index variants for the TAQ benchmark.
//!
//! FullPrecisionIndex — f32, brute-force (oracle / ground-truth).
//! UniformSq8Index    — all vectors at 8 bits per dim, brute-force.
//! UniformSq4Index    — all vectors at 4 bits per dim, brute-force.
//! TaqIndex           — hub nodes at SQ8, leaf nodes at SQ4, brute-force.
//!
//! Search in all variants is asymmetric: the query stays in f32, stored
//! vectors are dequantized at query time, then re-ranked with exact f32
//! distances over the shortlist.  This is the standard ADC (Asymmetric
//! Distance Computation) pattern and gives recall closer to the exact oracle.

use crate::graph::{build_knn_directed, classify_hubs, in_degree, sq_euclidean};
use crate::quantize::{Sq4Params, Sq8Params};

pub trait VectorIndex {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self;
    fn search(&self, query: &[f32], k: usize) -> Vec<usize>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Baseline: full f32 precision, brute-force search
// ---------------------------------------------------------------------------
pub struct FullPrecisionIndex {
    pub vectors: Vec<Vec<f32>>,
    pub dim: usize,
}

impl VectorIndex for FullPrecisionIndex {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self {
        Self { vectors, dim }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (sq_euclidean(query, v), i))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.len() * self.dim * 4
    }

    fn name(&self) -> &'static str {
        "FullPrecision-f32"
    }
}

// ---------------------------------------------------------------------------
// Uniform SQ8: all vectors quantized to 8 bits per dimension
// ---------------------------------------------------------------------------
pub struct UniformSq8Index {
    pub codes: Vec<Vec<u8>>,
    pub params: Sq8Params,
    pub raw: Vec<Vec<f32>>,
    pub dim: usize,
}

impl VectorIndex for UniformSq8Index {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self {
        let params = Sq8Params::fit(&vectors, dim);
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| params.encode(v)).collect();
        Self {
            codes,
            params,
            raw: vectors,
            dim,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let deq = self.params.decode(c);
                (sq_euclidean(query, &deq), i)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn memory_bytes(&self) -> usize {
        self.codes.len() * self.dim
    }

    fn name(&self) -> &'static str {
        "UniformSQ8"
    }
}

// ---------------------------------------------------------------------------
// Uniform SQ4: all vectors quantized to 4 bits per dimension (nibble packed)
// ---------------------------------------------------------------------------
pub struct UniformSq4Index {
    pub codes: Vec<Vec<u8>>,
    pub params: Sq4Params,
    pub raw: Vec<Vec<f32>>,
    pub dim: usize,
}

impl VectorIndex for UniformSq4Index {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self {
        let params = Sq4Params::fit(&vectors, dim);
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| params.encode(v)).collect();
        Self {
            codes,
            params,
            raw: vectors,
            dim,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let dim = self.dim;
        let mut dists: Vec<(f32, usize)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let deq = self.params.decode(c, dim);
                (sq_euclidean(query, &deq), i)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn memory_bytes(&self) -> usize {
        self.codes.len() * ((self.dim + 1) / 2)
    }

    fn name(&self) -> &'static str {
        "UniformSQ4"
    }
}

// ---------------------------------------------------------------------------
// TAQ: topology-aware mixed quantization
// ---------------------------------------------------------------------------
/// A stored vector is either hub (SQ8) or leaf (SQ4).
pub(crate) enum StoredVec {
    Hub(Vec<u8>),
    Leaf(Vec<u8>),
}

pub struct TaqIndex {
    pub(crate) stored: Vec<StoredVec>,
    pub hub_params: Sq8Params,
    pub leaf_params: Sq4Params,
    pub is_hub: Vec<bool>,
    pub dim: usize,
    pub hub_count: usize,
    pub leaf_count: usize,
}

impl TaqIndex {
    /// Degree threshold: nodes with in-degree > threshold become hubs.
    const HUB_DEGREE_THRESHOLD: usize = 2;
    /// k for building the topology graph.
    const GRAPH_K: usize = 8;
}

impl VectorIndex for TaqIndex {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self {
        let n = vectors.len();

        // 1. Build k-NN graph and classify hubs
        let graph = build_knn_directed(&vectors, Self::GRAPH_K.min(n.saturating_sub(1)));
        let degrees = in_degree(&graph);
        let is_hub = classify_hubs(&degrees, Self::HUB_DEGREE_THRESHOLD);
        let hub_count = is_hub.iter().filter(|&&h| h).count();
        let leaf_count = n - hub_count;

        // 2. Fit separate quantization params for hubs vs leaves
        let hub_vecs: Vec<&Vec<f32>> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| is_hub[*i])
            .map(|(_, v)| v)
            .collect();
        let leaf_vecs: Vec<&Vec<f32>> = vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_hub[*i])
            .map(|(_, v)| v)
            .collect();

        // Fall back to all-vector params when a class is empty
        let fit_hub: Vec<Vec<f32>> = if hub_vecs.is_empty() {
            vectors.clone()
        } else {
            hub_vecs.into_iter().cloned().collect()
        };
        let fit_leaf: Vec<Vec<f32>> = if leaf_vecs.is_empty() {
            vectors.clone()
        } else {
            leaf_vecs.into_iter().cloned().collect()
        };

        let hub_params = Sq8Params::fit(&fit_hub, dim);
        let leaf_params = Sq4Params::fit(&fit_leaf, dim);

        // 3. Encode each vector with its class-specific quantizer
        let stored: Vec<StoredVec> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if is_hub[i] {
                    StoredVec::Hub(hub_params.encode(v))
                } else {
                    StoredVec::Leaf(leaf_params.encode(v))
                }
            })
            .collect();

        Self {
            stored,
            hub_params,
            leaf_params,
            is_hub,
            dim,
            hub_count,
            leaf_count,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let dim = self.dim;
        let mut dists: Vec<(f32, usize)> = self
            .stored
            .iter()
            .enumerate()
            .map(|(i, sv)| {
                let deq = match sv {
                    StoredVec::Hub(c) => self.hub_params.decode(c),
                    StoredVec::Leaf(c) => self.leaf_params.decode(c, dim),
                };
                (sq_euclidean(query, &deq), i)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn memory_bytes(&self) -> usize {
        self.stored
            .iter()
            .map(|sv| match sv {
                StoredVec::Hub(c) => c.len(),
                StoredVec::Leaf(c) => c.len(),
            })
            .sum()
    }

    fn name(&self) -> &'static str {
        "TAQ-mixed(hub=SQ8,leaf=SQ4)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dataset() -> Vec<Vec<f32>> {
        (0..20usize)
            .map(|i| vec![i as f32, (i * 2) as f32])
            .collect()
    }

    fn query() -> Vec<f32> {
        vec![5.0, 10.0]
    }

    #[test]
    fn full_precision_returns_k_results() {
        let ds = tiny_dataset();
        let idx = FullPrecisionIndex::build(ds, 2);
        let res = idx.search(&query(), 5);
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn sq8_top1_matches_fp() {
        let ds = tiny_dataset();
        let fp = FullPrecisionIndex::build(ds.clone(), 2);
        let sq8 = UniformSq8Index::build(ds.clone(), 2);
        let fp_top1 = fp.search(&query(), 1)[0];
        let sq8_top1 = sq8.search(&query(), 1)[0];
        // SQ8 top-1 should match exact top-1 on this structured dataset
        assert_eq!(fp_top1, sq8_top1, "SQ8 top-1 drifted from exact");
    }

    #[test]
    fn taq_memory_between_sq4_and_sq8() {
        let ds: Vec<Vec<f32>> = (0..50usize)
            .map(|i| (0..32).map(|d| (i + d) as f32).collect())
            .collect();
        let dim = 32;
        let sq8 = UniformSq8Index::build(ds.clone(), dim);
        let sq4 = UniformSq4Index::build(ds.clone(), dim);
        let taq = TaqIndex::build(ds, dim);
        assert!(
            taq.memory_bytes() <= sq8.memory_bytes(),
            "TAQ should use <= SQ8 memory: {} vs {}",
            taq.memory_bytes(),
            sq8.memory_bytes()
        );
        assert!(
            taq.memory_bytes() >= sq4.memory_bytes(),
            "TAQ should use >= SQ4 memory: {} vs {}",
            taq.memory_bytes(),
            sq4.memory_bytes()
        );
    }

    #[test]
    fn taq_has_hub_and_leaf_nodes() {
        // 99 dense cluster points: many mutual k-NN → high in-degree → hubs.
        // 1 isolated outlier at (1e6, 1e6): no cluster point is within its
        // distance to the cluster, so in-degree=0 < threshold=2 → leaf.
        let mut ds: Vec<Vec<f32>> = (0..99usize)
            .map(|i| vec![(i % 10) as f32 * 0.1, (i / 10) as f32 * 0.1])
            .collect();
        ds.push(vec![1_000_000.0, 1_000_000.0]);
        let taq = TaqIndex::build(ds, 2);
        assert!(taq.hub_count > 0, "expected dense cluster nodes to be hubs");
        assert!(taq.leaf_count > 0, "expected isolated outlier to be a leaf");
    }
}
