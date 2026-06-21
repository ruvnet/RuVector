//! Wheeler–DeWitt timeless constraint.
//!
//! The quantum state of a closed universe obeys `Ĥ|Ψ> = 0` — there is no
//! external time parameter. "Time" must be found *inside* the state. This
//! module builds bipartite constraint operators `Ĵ = H_C ⊗ I + I ⊗ H_R` and
//! locates their physical (kernel) states.
//!
//! ## What is trivial here vs. what is discriminating
//!
//! The constraint `Ĵ = H_C ⊗ I + I ⊗ H_R` has eigenvalues `{aᵢ + bⱼ}` over all
//! pairs of eigenvalues `aᵢ` of `H_C` and `bⱼ` of `H_R`. A kernel (a physical
//! timeless state) exists **iff** some `aᵢ = −bⱼ`, i.e. iff the clock spectrum
//! and the (negated) rest spectrum overlap.
//!
//! In the Page–Wootters construction the clock is *built* with `H_C = diag(−Eₖ)`,
//! so the spectra match by construction and the kernel's existence is therefore
//! **trivial-by-construction** — verifying it is a *consistency check*, not a
//! discovery. The discriminating physical content of the module is the
//! complementary statement: for a *generic* clock Hamiltonian whose spectrum is
//! not `−spectrum(H_R)`, the constraint has **no zero eigenvalue at all** — the
//! physical Hilbert space is *empty*. That is what makes the constraint a real
//! constraint rather than a tautology, and it is tested in
//! [`tests::generic_clock_yields_empty_physical_space`].

use crate::complex::Complex;
use crate::complex_matrix::CMatrix;
use crate::real_matrix::RealMatrix;
use crate::state::idx;

/// Build the bipartite constraint `Ĵ = H_C ⊗ I_{dr} + I_{dc} ⊗ H_R`.
///
/// Physical states `|Ψ>` of the joint clock+rest system satisfy `Ĵ|Ψ> = 0`:
/// the total "energy" (clock + rest) is constrained to vanish, which is what
/// removes the external time parameter.
pub fn bipartite_constraint(h_c: &RealMatrix, h_r: &RealMatrix) -> RealMatrix {
    let dc = h_c.n;
    let dr = h_r.n;
    let n = dc * dr;
    let mut j = RealMatrix::zeros(n);
    for c in 0..dc {
        for cp in 0..dc {
            let hc = h_c.get(c, cp);
            for r in 0..dr {
                for rp in 0..dr {
                    let mut v = 0.0;
                    if r == rp {
                        v += hc; // H_C ⊗ I
                    }
                    if c == cp {
                        v += h_r.get(r, rp); // I ⊗ H_R
                    }
                    if v != 0.0 {
                        j.set(idx(c, r, dr), idx(cp, rp, dr), v);
                    }
                }
            }
        }
    }
    j
}

/// A physical (timeless) state: the eigenvector of the constraint with the
/// eigenvalue closest to zero.
pub struct PhysicalState {
    /// The constraint eigenvalue actually achieved (≈ 0 for a true kernel).
    pub eigenvalue: f64,
    /// The normalized physical state vector.
    pub state: Vec<f64>,
}

/// Find the physical state `|Ψ>` solving `Ĵ|Ψ> ≈ 0` — the kernel direction of
/// the constraint operator.
pub fn solve_constraint(j: &RealMatrix) -> PhysicalState {
    let (vals, vecs) = j.symmetric_eigen();
    let mut best = 0usize;
    for k in 1..vals.len() {
        if vals[k].abs() < vals[best].abs() {
            best = k;
        }
    }
    PhysicalState {
        eigenvalue: vals[best],
        state: vecs.column(best),
    }
}

/// Residual `‖Ĵ|Ψ>‖` for a (possibly complex) state vector — the degree to
/// which the timeless equation `Ĥ|Ψ> = 0` is satisfied.
pub fn constraint_residual(j: &RealMatrix, psi: &[Complex]) -> f64 {
    let jc = CMatrix::from_real(j);
    let out = jc.matvec(psi);
    out.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_wootters::PageWootters;

    fn sample_h() -> RealMatrix {
        RealMatrix::from_fn(3, |r, c| if r == c { (r as f64) - 1.0 } else { 0.3 })
    }

    #[test]
    fn constructed_page_wootters_state_lies_in_kernel() {
        // CONSISTENCY CHECK (not a discovery): with the energy-matched clock
        // `H_C = diag(−Eₖ)`, the Page–Wootters state is *built* term-by-term to
        // be annihilated by Ĵ, so it is in the kernel by construction. This test
        // confirms the construction is internally consistent — it cannot fail
        // for a correct implementation and is not evidence that the constraint
        // constrains. The discriminating test is below.
        let pw = PageWootters::new(sample_h());
        let j = bipartite_constraint(&pw.clock_hamiltonian(), &pw.h_r);
        let psi = pw.global_static_state();
        let residual = constraint_residual(&j, &psi);
        assert!(residual < 1e-8, "residual {residual} should be ~0");
    }

    #[test]
    fn energy_matched_clock_has_zero_eigenvalue_by_construction() {
        // CONSISTENCY CHECK: because `H_C = diag(−Eₖ)`, every diagonal pair
        // (a = b) contributes an eigenvalue `−Eₖ + Eₖ = 0`, so a kernel is
        // guaranteed to exist. Verifying it is a sanity check on the eigensolver,
        // not a physical discovery.
        let pw = PageWootters::new(sample_h());
        let j = bipartite_constraint(&pw.clock_hamiltonian(), &pw.h_r);
        let phys = solve_constraint(&j);
        assert!(phys.eigenvalue.abs() < 1e-8);
        // Kernel state is a unit vector.
        let norm: f64 = phys.state.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn generic_clock_yields_empty_physical_space() {
        // DISCRIMINATING TEST — this is the one that can actually fail and that
        // proves the constraint constrains. Build Ĵ from a GENERIC clock
        // Hamiltonian `H_C` that is NOT `−H_R`, and whose spectrum does not
        // match `−spectrum(H_R)`. Then Ĵ has eigenvalues {aᵢ + bⱼ} with no
        // (aᵢ = −bⱼ) coincidence, so there is NO eigenvalue near zero: the
        // physical (timeless) Hilbert space is EMPTY.
        //
        // Why this is the real content: finding the kernel for the energy-matched
        // clock is trivial-by-construction (see the consistency checks above).
        // Emptiness for a generic clock is what distinguishes a genuine
        // constraint from a tautology — it shows the kernel's existence is
        // *special* to the energy-matched clock, not automatic.
        let h_r = sample_h();
        let (energies, _v) = h_r.symmetric_eigen();

        // A deterministic generic real-symmetric clock Hamiltonian, chosen so its
        // spectrum is well separated from −spectrum(H_R). We start from a base
        // and, if any accidental near-degeneracy aᵢ ≈ −bⱼ shows up, perturb the
        // diagonal by a fixed offset until the minimum |aᵢ + bⱼ| clears a margin.
        let min_gap = 0.25;
        let mut offset = 0.0f64;
        let (h_c, min_sum) = loop {
            let off = offset;
            // Generic: distinct diagonal, non-trivial off-diagonal, NOT diag(−Eₖ).
            let h_c = RealMatrix::from_fn(3, |r, c| {
                if r == c {
                    // 7, 8, 9 (+ offset): far from −E (which are ≈ −2..1 here),
                    // and not a permutation of −Eₖ.
                    (7 + r) as f64 + off
                } else {
                    0.2
                }
            });
            let (a_vals, _) = h_c.symmetric_eigen();
            let mut min_sum = f64::INFINITY;
            for &a in &a_vals {
                for &b in &energies {
                    min_sum = min_sum.min((a + b).abs());
                }
            }
            if min_sum > min_gap {
                break (h_c, min_sum);
            }
            // Deterministic perturbation to escape any accidental coincidence.
            offset += 0.5;
            assert!(
                offset < 100.0,
                "failed to find a generic non-matching clock"
            );
        };

        // Sanity: the chosen clock is genuinely not the energy-matched one.
        let matched = RealMatrix::diag(&energies.iter().map(|e| -e).collect::<Vec<_>>());
        let max_diff = (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| (h_c.get(r, c) - matched.get(r, c)).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1.0,
            "test clock must differ from the matched clock"
        );

        let j = bipartite_constraint(&h_c, &h_r);
        let (j_vals, _) = j.symmetric_eigen();
        let nearest_zero = j_vals.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);

        // No eigenvalue within 1e-9 of zero ⇒ empty physical Hilbert space.
        assert!(
            nearest_zero > 1e-9,
            "generic clock must leave NO kernel; nearest eigenvalue to zero was {nearest_zero} \
             (predicted lower bound min|aᵢ+bⱼ| = {min_sum})"
        );
        // The predicted bound and the measured spectrum agree.
        assert!(
            (nearest_zero - min_sum).abs() < 1e-6,
            "measured nearest-zero {nearest_zero} should match the eigenvalue-sum prediction {min_sum}"
        );
    }
}
