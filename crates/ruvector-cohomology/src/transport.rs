//! Orthogonal transport and latent-drift guards (issue #669, §6).
//!
//! The main false-positive risk of the whole engine is treating coordinate
//! drift as semantic contradiction. Orthogonal frame changes must transport
//! stored state exactly (`x_new = Q_new Q_oldᵀ x_old`) and restriction maps
//! must be re-expressed (`ρ ← ρ Rᵀ`) so every coboundary — and therefore
//! the syndrome — is invariant. Map updates are guarded by cycle-consistency
//! and coherent-control tests before acceptance.

use crate::affine::AffineSheafSystem;
use crate::operator::{DenseMatrix, LinearRestriction, RestrictionMap};
use crate::syndrome::{compute_syndrome, SyndromeConfig};
use crate::{CohomologyError, ValidationLimits};

/// Validate `q` as a square orthogonal matrix of the given dimension.
fn validate_orthogonal(
    q: &DenseMatrix,
    dim: usize,
    tol: f64,
    name: &str,
) -> Result<(), CohomologyError> {
    if q.rows != dim || q.cols != dim {
        return Err(CohomologyError::DimensionMismatch {
            expected: dim,
            actual: q.rows,
            context: format!("frame {name}"),
        });
    }
    if !q.is_finite() {
        return Err(CohomologyError::ValidationFailed(format!(
            "frame {name} has non-finite entries"
        )));
    }
    let defect = q.orthogonality_defect();
    if defect > tol {
        return Err(CohomologyError::ValidationFailed(format!(
            "frame {name} orthogonality defect {defect:.3e} exceeds {tol:.3e}"
        )));
    }
    Ok(())
}

/// Compute the frame-change matrix `R = Q_new Q_oldᵀ` after validating both
/// frames as orthogonal.
pub fn frame_change(
    q_old: &DenseMatrix,
    q_new: &DenseMatrix,
    limits: &ValidationLimits,
) -> Result<DenseMatrix, CohomologyError> {
    let dim = q_old.rows;
    validate_orthogonal(q_old, dim, limits.orthogonality_tol, "Q_old")?;
    validate_orthogonal(q_new, dim, limits.orthogonality_tol, "Q_new")?;
    q_new.matmul(&q_old.transpose())
}

/// Transport a stored stalk state through a frame change:
/// `x_new = R x_old`.
pub fn transport_state(r_frame: &DenseMatrix, x_old: &[f64]) -> Result<Vec<f64>, CohomologyError> {
    if x_old.len() != r_frame.cols {
        return Err(CohomologyError::DimensionMismatch {
            expected: r_frame.cols,
            actual: x_old.len(),
            context: "transport_state".into(),
        });
    }
    let mut out = vec![0.0; r_frame.rows];
    r_frame.matvec(x_old, &mut out);
    Ok(out)
}

/// Re-express a restriction map after its source vertex changed frame:
/// `ρ' = ρ Rᵀ`, so that `ρ'(x_new) = ρ(x_old)` exactly.
pub fn transport_restriction(
    rho: &RestrictionMap,
    r_frame: &DenseMatrix,
) -> Result<RestrictionMap, CohomologyError> {
    rho.compose_transpose_frame(r_frame)
}

/// Cycle-consistency defect of a closed path of square maps: the Frobenius
/// distance of their ordered product from the identity. A legitimate
/// transport family must keep this near zero on held-out cycles.
pub fn cycle_consistency_defect(maps: &[&RestrictionMap]) -> Result<f64, CohomologyError> {
    let first = maps
        .first()
        .ok_or_else(|| CohomologyError::InvalidValue("empty cycle".into()))?;
    let dim = first.input_dim();
    let mut product = DenseMatrix::identity(dim);
    for m in maps {
        if m.input_dim() != product.rows || m.output_dim() != m.input_dim() {
            return Err(CohomologyError::DimensionMismatch {
                expected: product.rows,
                actual: m.input_dim(),
                context: "cycle_consistency_defect: maps must be square and chained".into(),
            });
        }
        product = m.to_dense().matmul(&product)?;
    }
    let mut defect = 0.0;
    for r in 0..dim {
        for c in 0..dim {
            let target = if r == c { 1.0 } else { 0.0 };
            defect += (product.get(r, c) - target).powi(2);
        }
    }
    Ok(defect.sqrt())
}

/// Report from the map-update guard.
#[derive(Debug, Clone)]
pub struct GuardReport {
    /// Syndrome energy of the control system before the update.
    pub energy_before: f64,
    /// Syndrome energy of the control system after the update.
    pub energy_after: f64,
    /// Whether the update stays within the allowed increase.
    pub accepted: bool,
}

/// Negative-control guard for restriction-map updates: on a **coherent**
/// control system, an admissible transport update must not create syndrome
/// energy beyond `max_increase`. Learned maps stay advisory until they pass
/// this and cycle-consistency checks (issue #669 principal uncertainty).
pub fn guard_map_update(
    control_before: &AffineSheafSystem,
    control_after: &AffineSheafSystem,
    max_increase: f64,
    cfg: &SyndromeConfig,
) -> Result<GuardReport, CohomologyError> {
    let before = compute_syndrome(control_before, cfg)?;
    let after = compute_syndrome(control_after, cfg)?;
    let accepted = after.energy <= before.energy + max_increase;
    Ok(GuardReport {
        energy_before: before.energy,
        energy_after: after.energy,
        accepted,
    })
}
