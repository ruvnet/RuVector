//! Bipartite pure states and the partial traces used to split a closed system
//! into a *clock* subsystem `C` and the *rest* `R`:  `H = H_C ⊗ H_R`.
//!
//! A bipartite state vector is stored in row-major `C ⊗ R` order, so the
//! amplitude for clock index `c` and rest index `r` lives at `psi[c * dr + r]`.

use crate::complex::Complex;
use crate::complex_matrix::CMatrix;

/// Index of amplitude `|c>_C ⊗ |r>_R` inside a bipartite state vector.
#[inline]
pub fn idx(c: usize, r: usize, dr: usize) -> usize {
    c * dr + r
}

/// Reduced density matrix of the *rest* sector,
/// `ρ_R = Tr_C |Ψ><Ψ|`, given a bipartite pure state.
pub fn reduced_rest(psi: &[Complex], dc: usize, dr: usize) -> CMatrix {
    assert_eq!(psi.len(), dc * dr);
    let mut rho = CMatrix::zeros(dr);
    for r in 0..dr {
        for rp in 0..dr {
            let mut acc = Complex::ZERO;
            for c in 0..dc {
                acc += psi[idx(c, r, dr)] * psi[idx(c, rp, dr)].conj();
            }
            rho.set(r, rp, acc);
        }
    }
    rho
}

/// Reduced density matrix of the *clock* sector, `ρ_C = Tr_R |Ψ><Ψ|`.
pub fn reduced_clock(psi: &[Complex], dc: usize, dr: usize) -> CMatrix {
    assert_eq!(psi.len(), dc * dr);
    let mut rho = CMatrix::zeros(dc);
    for c in 0..dc {
        for cp in 0..dc {
            let mut acc = Complex::ZERO;
            for r in 0..dr {
                acc += psi[idx(c, r, dr)] * psi[idx(cp, r, dr)].conj();
            }
            rho.set(c, cp, acc);
        }
    }
    rho
}

/// Project the clock onto a (not necessarily normalized) clock vector
/// `clock_bra` and return the resulting **unnormalized** state of the rest
/// sector — the Page–Wootters conditional state before renormalization.
///
/// `out[r] = Σ_c conj(clock_bra[c]) · psi[c, r]`.
pub fn condition_on_clock(psi: &[Complex], clock_bra: &[Complex], dr: usize) -> Vec<Complex> {
    let dc = clock_bra.len();
    assert_eq!(psi.len(), dc * dr);
    let mut out = vec![Complex::ZERO; dr];
    for r in 0..dr {
        let mut acc = Complex::ZERO;
        for c in 0..dc {
            acc += clock_bra[c].conj() * psi[idx(c, r, dr)];
        }
        out[r] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::vec_norm;

    #[test]
    fn product_state_traces_to_pure() {
        // |0>_C ⊗ (|0> + |1>)/√2 in dc=2, dr=2.
        let s = 1.0 / 2.0f64.sqrt();
        let psi = vec![
            Complex::real(s), // c0 r0
            Complex::real(s), // c0 r1
            Complex::ZERO,    // c1 r0
            Complex::ZERO,    // c1 r1
        ];
        let rho_r = reduced_rest(&psi, 2, 2);
        // trace = 1
        let tr = rho_r.get(0, 0) + rho_r.get(1, 1);
        assert!((tr.re - 1.0).abs() < 1e-12 && tr.im.abs() < 1e-12);
    }

    #[test]
    fn conditioning_extracts_branch() {
        // Bell-like: (|0>|0> + |1>|1>)/√2. Conditioning clock on |0> yields |0>_R.
        let s = 1.0 / 2.0f64.sqrt();
        let psi = vec![
            Complex::real(s),
            Complex::ZERO,
            Complex::ZERO,
            Complex::real(s),
        ];
        let bra0 = vec![Complex::ONE, Complex::ZERO];
        let out = condition_on_clock(&psi, &bra0, 2);
        assert!((vec_norm(&out) - s).abs() < 1e-12);
        assert!(out[0].modulus() > out[1].modulus());
    }
}
