//! Affine sheaf system: observations `b`, weights `W`, and gauge.
//!
//! Adds the affine layer missing from the pure-topology sheaf model
//! (issue #669, implementation gaps 5–6): observed edge relationships and
//! confidence weights, hashed deterministically for witnesses.

use crate::block_sparse::BlockCoboundary;
use crate::{CohomologyError, EdgeKey, GaugePolicy};
use sha2::{Digest, Sha256};

/// A field over `C¹`: one value block per edge, laid out to match the
/// canonical edge order of a [`BlockCoboundary`].
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeField {
    /// Concatenated per-edge blocks.
    pub data: Vec<f64>,
}

impl EdgeField {
    /// Zero field matching an operator layout.
    pub fn zeros(op: &BlockCoboundary) -> Self {
        Self { data: vec![0.0; op.dim_c1()] }
    }

    /// Weighted energy `‖f‖²_W = Σ_e w_e ‖f_e‖²`.
    pub fn weighted_energy(&self, op: &BlockCoboundary, weights: &EdgeWeights) -> f64 {
        let mut acc = 0.0;
        for (i, e) in op.edge_blocks().iter().enumerate() {
            let mut block = 0.0;
            for v in &self.data[e.offset..e.offset + e.dim] {
                block += v * v;
            }
            acc += weights.values[i] * block;
        }
        acc
    }

    /// Deterministic hash over the exact bit patterns in canonical order.
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector-cohomology/edge-field/v1");
        h.update((self.data.len() as u64).to_le_bytes());
        for v in &self.data {
            h.update(v.to_bits().to_le_bytes());
        }
        h.finalize().into()
    }
}

/// Per-edge positive scalar confidence/trust/severity weights, laid out in
/// canonical edge order.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeWeights {
    /// One weight per edge.
    pub values: Vec<f64>,
}

impl EdgeWeights {
    /// Unit weights matching an operator.
    pub fn ones(op: &BlockCoboundary) -> Self {
        Self { values: vec![1.0; op.edge_count()] }
    }

    /// Validate positivity and finiteness.
    pub fn validate(&self, max_value: f64) -> Result<(), CohomologyError> {
        for (i, w) in self.values.iter().enumerate() {
            if !w.is_finite() || *w <= 0.0 || *w > max_value {
                return Err(CohomologyError::ValidationFailed(format!(
                    "edge weight #{i} = {w} must be finite, positive, and ≤ {max_value}"
                )));
            }
        }
        Ok(())
    }

    /// Deterministic hash over the exact bit patterns in canonical order.
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ruvector-cohomology/edge-weights/v1");
        h.update((self.values.len() as u64).to_le_bytes());
        for v in &self.values {
            h.update(v.to_bits().to_le_bytes());
        }
        h.finalize().into()
    }
}

/// The full affine sheaf system (issue #669 core type).
pub struct AffineSheafSystem {
    /// Epoch of the topology (vertices, edges, restriction maps).
    pub topology_epoch: u64,
    /// Epoch of the observation/weight data.
    pub observation_epoch: u64,
    /// The block coboundary operator `δ`.
    pub operator: BlockCoboundary,
    /// Observed edge relationships `b ∈ C¹`.
    pub observations: EdgeField,
    /// Confidence weights `W` (per-edge scalars).
    pub weights: EdgeWeights,
    /// Gauge fixing policy for the least-squares representative.
    pub gauge: GaugePolicy,
}

impl AffineSheafSystem {
    /// Assemble and validate a system.
    pub fn new(
        operator: BlockCoboundary,
        observations: EdgeField,
        weights: EdgeWeights,
        gauge: GaugePolicy,
    ) -> Result<Self, CohomologyError> {
        if observations.data.len() != operator.dim_c1() {
            return Err(CohomologyError::DimensionMismatch {
                expected: operator.dim_c1(),
                actual: observations.data.len(),
                context: "observations vs C¹ layout".into(),
            });
        }
        if weights.values.len() != operator.edge_count() {
            return Err(CohomologyError::DimensionMismatch {
                expected: operator.edge_count(),
                actual: weights.values.len(),
                context: "weights vs edge count".into(),
            });
        }
        if observations.data.iter().any(|v| !v.is_finite()) {
            return Err(CohomologyError::ValidationFailed(
                "observations contain non-finite values".into(),
            ));
        }
        weights.validate(f64::MAX)?;
        Ok(Self {
            topology_epoch: 0,
            observation_epoch: 0,
            operator,
            observations,
            weights,
            gauge,
        })
    }

    /// Set the observation block for one edge.
    pub fn set_observation(&mut self, key: &EdgeKey, value: &[f64]) -> Result<(), CohomologyError> {
        let (off, dim) = self
            .operator
            .edge_slice(key)
            .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
        if value.len() != dim {
            return Err(CohomologyError::DimensionMismatch {
                expected: dim,
                actual: value.len(),
                context: format!("observation on edge ({}, {})", key.u, key.v),
            });
        }
        if value.iter().any(|v| !v.is_finite()) {
            return Err(CohomologyError::ValidationFailed(
                "observation contains non-finite values".into(),
            ));
        }
        self.observations.data[off..off + dim].copy_from_slice(value);
        self.observation_epoch += 1;
        Ok(())
    }

    /// Set the weight for one edge.
    pub fn set_weight(&mut self, key: &EdgeKey, weight: f64) -> Result<(), CohomologyError> {
        let pos = self
            .operator
            .edge_position(key)
            .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(CohomologyError::ValidationFailed(format!(
                "weight {weight} must be finite and positive"
            )));
        }
        self.weights.values[pos] = weight;
        self.observation_epoch += 1;
        Ok(())
    }
}
