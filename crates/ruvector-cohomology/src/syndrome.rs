//! Affine syndrome engine (issue #669, Phase 2).
//!
//! Solves the weighted least-squares problem
//! `x* = argmin ‖W^{1/2}(δx − b)‖²`, extracts the canonical syndrome
//! `s = b − δx*` (the weighted projection of `b` onto `(im δ)^{⊥_W}`),
//! ranks edge contributions, and produces a sealed canonical witness.

use crate::affine::{AffineSheafSystem, EdgeField};
use crate::harmonic::HarmonicBasis;
use crate::solvers::cg::solve_psd;
use crate::solvers::sparse_qr::PivotedQr;
use crate::solvers::{ResidualCertificate, SolverConfig};
use crate::witness::{
    quantize, CohomologyWitness, HarmonicMeta, SupportEntry, DEFAULT_QUANTIZATION_SCALE,
};
use crate::{CohomologyError, EdgeKey};

/// Configuration for syndrome extraction.
#[derive(Debug, Clone)]
pub struct SyndromeConfig {
    /// Linear solver configuration.
    pub solver: SolverConfig,
    /// Witness quantization scale.
    pub quantization_scale: f64,
    /// Compute the canonical dense harmonic basis when `dim C¹` is at most
    /// this threshold (the basis is O(dim²) memory).
    pub dense_harmonic_threshold: usize,
    /// Number of ranked support entries stored in the witness.
    pub top_k_support: usize,
    /// Relative rank tolerance for the rank-revealing QR.
    pub rank_tol_rel: f64,
}

impl Default for SyndromeConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            quantization_scale: DEFAULT_QUANTIZATION_SCALE,
            dense_harmonic_threshold: 256,
            top_k_support: 32,
            rank_tol_rel: 1e-10,
        }
    }
}

/// Contribution of one edge to the syndrome energy.
#[derive(Debug, Clone)]
pub struct EdgeContribution {
    /// The edge.
    pub edge: EdgeKey,
    /// `w_e ‖s_e‖²`.
    pub energy: f64,
    /// Fraction of the total syndrome energy (statistical leverage proxy).
    pub fraction: f64,
}

/// Full syndrome result (issue #669 core type).
pub struct SyndromeResult {
    /// Total syndrome energy `‖s‖²_W`.
    pub energy: f64,
    /// The syndrome `s = b − δx*` as an edge field.
    pub residual: EdgeField,
    /// Gauged least-squares explanation `x*`.
    pub x_star: Vec<f64>,
    /// Coordinates of `W^{1/2}s` in the canonical harmonic basis (dense
    /// path; empty when the system exceeds the dense threshold).
    pub harmonic_coordinates: Vec<f64>,
    /// Edge contributions, highest energy first (ties by edge key).
    pub ranked_support: Vec<EdgeContribution>,
    /// Sealed canonical witness.
    pub witness: CohomologyWitness,
    /// Residual certificate of the linear solve.
    pub certificate: ResidualCertificate,
}

fn rank_support(system: &AffineSheafSystem, residual: &[f64]) -> (f64, Vec<EdgeContribution>) {
    let op = &system.operator;
    let mut total = 0.0;
    let mut entries: Vec<EdgeContribution> = op
        .edge_blocks()
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut block = 0.0;
            for v in &residual[e.offset..e.offset + e.dim] {
                block += v * v;
            }
            let energy = system.weights.values[i] * block;
            total += energy;
            EdgeContribution { edge: e.key, energy, fraction: 0.0 }
        })
        .collect();
    for e in &mut entries {
        e.fraction = if total > 0.0 { e.energy / total } else { 0.0 };
    }
    entries.sort_by(|a, b| {
        b.energy
            .partial_cmp(&a.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.edge.cmp(&b.edge))
    });
    (total, entries)
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    system: &AffineSheafSystem,
    x_star: Vec<f64>,
    certificate: ResidualCertificate,
    solver_id: &str,
    rank_hint: Option<usize>,
    cfg: &SyndromeConfig,
) -> SyndromeResult {
    let op = &system.operator;
    let mut dx = vec![0.0; op.dim_c1()];
    op.apply(&x_star, &mut dx);
    let mut residual = system.observations.data.clone();
    for (r, d) in residual.iter_mut().zip(dx.iter()) {
        *r -= d;
    }
    let (energy, ranked_support) = rank_support(system, &residual);

    // Canonical harmonic coordinates on the dense path.
    let (harmonic_coordinates, harmonic_meta) = if op.dim_c1() <= cfg.dense_harmonic_threshold
        && op.dim_c1() > 0
    {
        let basis = HarmonicBasis::dense(op, &system.weights.values, cfg.rank_tol_rel);
        let mut scaled = residual.clone();
        for (i, e) in op.edge_blocks().iter().enumerate() {
            let sw = system.weights.values[i].sqrt();
            for v in &mut scaled[e.offset..e.offset + e.dim] {
                *v *= sw;
            }
        }
        let coords = basis.coordinates(&scaled);
        let meta = HarmonicMeta {
            rank: basis.rank as u64,
            harmonic_dim: basis.basis.cols as u64,
            coordinates_q: coords
                .iter()
                .map(|c| quantize(*c, cfg.quantization_scale))
                .collect(),
            canonicalized: true,
        };
        (coords, meta)
    } else {
        (
            Vec::new(),
            HarmonicMeta {
                rank: rank_hint.unwrap_or(0) as u64,
                ..HarmonicMeta::default()
            },
        )
    };

    // Witness support is ordered on *quantized* energies (ties by edge key)
    // so solver-level float noise can never change the canonical ordering.
    let mut support: Vec<SupportEntry> = ranked_support
        .iter()
        .map(|c| SupportEntry {
            edge: c.edge,
            energy_q: quantize(c.energy, cfg.quantization_scale),
        })
        .collect();
    support.sort_by(|a, b| b.energy_q.cmp(&a.energy_q).then(a.edge.cmp(&b.edge)));
    support.truncate(cfg.top_k_support);

    let witness = CohomologyWitness {
        topology_epoch: system.topology_epoch,
        observation_epoch: system.observation_epoch,
        vertex_count: op.vertex_count() as u64,
        edge_count: op.edge_count() as u64,
        operator_hash: op.canonical_hash(),
        observation_hash: system.observations.canonical_hash(),
        weight_hash: system.weights.canonical_hash(),
        gauge: system.gauge,
        solver_id: solver_id.to_string(),
        solver_tol: cfg.solver.tol,
        quantization_scale: cfg.quantization_scale,
        energy_q: quantize(energy, cfg.quantization_scale),
        support,
        harmonic: harmonic_meta,
        repair: None,
        hash: [0u8; 32],
    }
    .seal();

    SyndromeResult {
        energy,
        residual: EdgeField { data: residual },
        x_star,
        harmonic_coordinates,
        ranked_support,
        witness,
        certificate,
    }
}

/// Compute the syndrome with the production path: matrix-free CG on the
/// gauged normal equations `δᵀWδ x = δᵀW b` (minimum-norm gauge).
pub fn compute_syndrome(
    system: &AffineSheafSystem,
    cfg: &SyndromeConfig,
) -> Result<SyndromeResult, CohomologyError> {
    compute_syndrome_warm(system, cfg, None)
}

/// [`compute_syndrome`] with an optional warm start for the dynamic engine.
pub fn compute_syndrome_warm(
    system: &AffineSheafSystem,
    cfg: &SyndromeConfig,
    warm_start: Option<&[f64]>,
) -> Result<SyndromeResult, CohomologyError> {
    let op = &system.operator;
    let weights = &system.weights.values;

    // rhs = δᵀ W b.
    let mut weighted = system.observations.data.clone();
    for (i, e) in op.edge_blocks().iter().enumerate() {
        let w = weights[i];
        for v in &mut weighted[e.offset..e.offset + e.dim] {
            *v *= w;
        }
    }
    let mut rhs = vec![0.0; op.dim_c0()];
    op.apply_transpose(&weighted, &mut rhs);

    let (x_star, certificate) = solve_psd(
        |x, y| op.apply_weighted_laplacian(weights, x, y),
        &rhs,
        warm_start,
        &cfg.solver,
    );
    if !certificate.converged && certificate.residual_norm > cfg.solver.tol.sqrt() {
        return Err(CohomologyError::SolverFailure(format!(
            "CG did not converge: residual {:.3e} after {} iterations",
            certificate.residual_norm, certificate.iterations
        )));
    }
    Ok(build_result(
        system,
        x_star,
        certificate,
        "cg/normal-equations",
        None,
        cfg,
    ))
}

/// Compute the syndrome with the exact dense reference oracle
/// (rank-revealing QR of `W^{1/2}δ`; issue #669, Phase 0).
pub fn compute_syndrome_dense(
    system: &AffineSheafSystem,
    cfg: &SyndromeConfig,
) -> Result<SyndromeResult, CohomologyError> {
    let op = &system.operator;
    let a = op.assemble_scaled_dense(&system.weights.values);
    let mut b_scaled = system.observations.data.clone();
    for (i, e) in op.edge_blocks().iter().enumerate() {
        let sw = system.weights.values[i].sqrt();
        for v in &mut b_scaled[e.offset..e.offset + e.dim] {
            *v *= sw;
        }
    }
    let qr = PivotedQr::factor(&a, cfg.rank_tol_rel);
    let x_star = qr.solve_least_squares(&b_scaled);

    // Exact residual certificate.
    let mut ax = vec![0.0; op.dim_c1()];
    a.matvec(&x_star, &mut ax);
    let mut res = 0.0;
    for (axi, bi) in ax.iter().zip(b_scaled.iter()) {
        res += (bi - axi).powi(2);
    }
    let certificate = ResidualCertificate {
        residual_norm: res.sqrt(),
        rhs_norm: b_scaled.iter().map(|v| v * v).sum::<f64>().sqrt(),
        iterations: 1,
        converged: true,
        tolerance: 0.0,
    };
    Ok(build_result(
        system,
        x_star,
        certificate,
        "qr/dense-oracle",
        Some(qr.rank()),
        cfg,
    ))
}
