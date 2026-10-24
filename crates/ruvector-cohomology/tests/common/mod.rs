//! Shared deterministic test helpers.
#![allow(dead_code)]

use ruvector_cohomology::block_sparse::CoboundaryBuilder;
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology, DynamicSheafEngine, EngineConfig};
use ruvector_cohomology::operator::{DenseMatrix, RestrictionMap};
use ruvector_cohomology::{AffineSheafSystem, EdgeField, EdgeWeights, GaugePolicy};

/// Deterministic SplitMix64 stream.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in (-1, 1).
    pub fn signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Deterministic random orthogonal matrix from a product of Givens rotations.
pub fn random_orthogonal(dim: usize, rng: &mut Rng) -> DenseMatrix {
    let mut q = DenseMatrix::identity(dim);
    for i in 0..dim {
        for j in (i + 1)..dim {
            let theta = rng.signed_unit() * std::f64::consts::PI;
            let (c, s) = (theta.cos(), theta.sin());
            for r in 0..dim {
                let qi = q.get(r, i);
                let qj = q.get(r, j);
                q.set(r, i, c * qi - s * qj);
                q.set(r, j, s * qi + c * qj);
            }
        }
    }
    q
}

/// Identity-sheaf system on the given edges with observations `b`.
pub fn identity_system(
    n: u64,
    dim: usize,
    edges: &[(u64, u64)],
    mut observations: impl FnMut(usize) -> Vec<f64>,
    weights: Option<Vec<f64>>,
) -> AffineSheafSystem {
    let mut builder = CoboundaryBuilder::new();
    for v in 0..n {
        builder.add_vertex(v, dim).unwrap();
    }
    for &(a, b) in edges {
        builder
            .add_edge(
                a,
                b,
                RestrictionMap::Identity { dim },
                RestrictionMap::Identity { dim },
            )
            .unwrap();
    }
    let op = builder.build();
    let mut data = vec![0.0; op.dim_c1()];
    for (i, e) in op.edge_blocks().iter().enumerate() {
        let block = observations(i);
        data[e.offset..e.offset + e.dim].copy_from_slice(&block);
    }
    let w = EdgeWeights { values: weights.unwrap_or_else(|| vec![1.0; op.edge_count()]) };
    AffineSheafSystem::new(op, EdgeField { data }, w, GaugePolicy::MinimumNorm).unwrap()
}

/// Ring engine with identity maps and coherent (zero) observations.
pub fn ring_engine(n: u64, dim: usize) -> DynamicSheafEngine {
    let mut engine = DynamicSheafEngine::new(EngineConfig::default());
    for i in 0..n {
        assert!(engine.apply_edit(CohomologyEdit::InsertVertex { id: i, dim }).accepted);
    }
    for i in 0..n {
        let receipt = engine.apply_edit(CohomologyEdit::InsertEdge {
            a: i,
            b: (i + 1) % n,
            rho_a: RestrictionMap::Identity { dim },
            rho_b: RestrictionMap::Identity { dim },
            weight: 1.0,
            observation: vec![0.0; dim],
        });
        assert!(receipt.accepted, "{:?}", receipt.error);
    }
    engine
}
