//! Canonical contradiction witnesses (issue #669, §7).
//!
//! A witness is reproducible across machines and execution orders: every
//! field is either an exact hash, a sorted identifier list, or an explicitly
//! quantized value. Solver floating-point output never enters the hash
//! directly — only its quantization does.

use crate::{EdgeKey, GaugePolicy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Default quantization scale for witness values (values are rounded to
/// integer multiples of this scale before hashing).
pub const DEFAULT_QUANTIZATION_SCALE: f64 = 1e-9;

/// Quantize a value onto the witness lattice.
pub fn quantize(value: f64, scale: f64) -> i64 {
    let q = (value / scale).round();
    if q >= i64::MAX as f64 {
        i64::MAX
    } else if q <= i64::MIN as f64 {
        i64::MIN
    } else {
        q as i64
    }
}

/// One ranked support entry: an edge and its quantized syndrome energy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportEntry {
    /// The edge.
    pub edge: EdgeKey,
    /// Quantized contribution `w_e ‖s_e‖²`.
    pub energy_q: i64,
}

/// Canonical harmonic metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HarmonicMeta {
    /// Numerical rank of `W^{1/2}δ`.
    pub rank: u64,
    /// Dimension of the harmonic (contradiction) subspace that was
    /// canonicalized, when the dense basis path ran.
    pub harmonic_dim: u64,
    /// Quantized harmonic coordinates of the syndrome (dense path only).
    pub coordinates_q: Vec<i64>,
    /// Whether the basis was canonicalized against the hashed probe.
    pub canonicalized: bool,
}

/// Summary of a proposed repair embedded in a witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairSummary {
    /// Edges carrying nonzero correction, canonically sorted.
    pub support: Vec<EdgeKey>,
    /// Quantized repair objective.
    pub objective_q: i64,
    /// Quantized syndrome energy before repair.
    pub pre_energy_q: i64,
    /// Quantized syndrome energy after repair.
    pub post_energy_q: i64,
}

/// Canonical, signed contradiction witness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohomologyWitness {
    /// Topology epoch at computation time.
    pub topology_epoch: u64,
    /// Observation epoch at computation time.
    pub observation_epoch: u64,
    /// Number of vertices.
    pub vertex_count: u64,
    /// Number of edges.
    pub edge_count: u64,
    /// Hash of the full operator (layout, orientation, restriction maps).
    pub operator_hash: [u8; 32],
    /// Hash of the observation field.
    pub observation_hash: [u8; 32],
    /// Hash of the weights.
    pub weight_hash: [u8; 32],
    /// Gauge policy used.
    pub gauge: GaugePolicy,
    /// Solver identifier (e.g. `"cg/normal-equations"`).
    pub solver_id: String,
    /// Solver tolerance.
    pub solver_tol: f64,
    /// Quantization scale for all quantized values below.
    pub quantization_scale: f64,
    /// Quantized syndrome energy `‖s‖²_W`.
    pub energy_q: i64,
    /// Ranked support, highest contribution first (ties by edge key).
    pub support: Vec<SupportEntry>,
    /// Harmonic canonicalization metadata.
    pub harmonic: HarmonicMeta,
    /// Optional embedded repair summary.
    pub repair: Option<RepairSummary>,
    /// SHA-256 over every canonical field above.
    pub hash: [u8; 32],
}

impl CohomologyWitness {
    /// Compute the canonical hash from the current field values and stamp it.
    pub fn seal(mut self) -> Self {
        let mut h = Sha256::new();
        h.update(b"ruvector-cohomology/witness/v1");
        h.update(self.topology_epoch.to_le_bytes());
        h.update(self.observation_epoch.to_le_bytes());
        h.update(self.vertex_count.to_le_bytes());
        h.update(self.edge_count.to_le_bytes());
        h.update(self.operator_hash);
        h.update(self.observation_hash);
        h.update(self.weight_hash);
        h.update([match self.gauge {
            GaugePolicy::MinimumNorm => 0u8,
        }]);
        h.update(self.solver_id.as_bytes());
        h.update(self.solver_tol.to_bits().to_le_bytes());
        h.update(self.quantization_scale.to_bits().to_le_bytes());
        h.update(self.energy_q.to_le_bytes());
        h.update((self.support.len() as u64).to_le_bytes());
        for s in &self.support {
            h.update(s.edge.u.to_le_bytes());
            h.update(s.edge.v.to_le_bytes());
            h.update(s.energy_q.to_le_bytes());
        }
        h.update(self.harmonic.rank.to_le_bytes());
        h.update(self.harmonic.harmonic_dim.to_le_bytes());
        h.update((self.harmonic.coordinates_q.len() as u64).to_le_bytes());
        for c in &self.harmonic.coordinates_q {
            h.update(c.to_le_bytes());
        }
        h.update([self.harmonic.canonicalized as u8]);
        match &self.repair {
            None => h.update([0u8]),
            Some(r) => {
                h.update([1u8]);
                h.update((r.support.len() as u64).to_le_bytes());
                for e in &r.support {
                    h.update(e.u.to_le_bytes());
                    h.update(e.v.to_le_bytes());
                }
                h.update(r.objective_q.to_le_bytes());
                h.update(r.pre_energy_q.to_le_bytes());
                h.update(r.post_energy_q.to_le_bytes());
            }
        }
        self.hash = h.finalize().into();
        self
    }

    /// Hex encoding of the witness hash.
    pub fn hash_hex(&self) -> String {
        self.hash.iter().map(|b| format!("{b:02x}")).collect()
    }
}
