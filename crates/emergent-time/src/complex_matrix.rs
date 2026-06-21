//! Dense complex matrices plus the spectral functions that turn a real
//! symmetric Hamiltonian into complex unitary dynamics.
//!
//! All evolution operators (`e^{-iHt}`, modular flow `e^{isK}`) are built by
//! diagonalizing the underlying real symmetric generator and exponentiating its
//! eigenvalues — accurate and free of truncated power series.

use crate::complex::Complex;
use crate::real_matrix::RealMatrix;

/// Row-major dense `n x n` complex matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CMatrix {
    pub n: usize,
    pub data: Vec<Complex>,
}

impl CMatrix {
    pub fn zeros(n: usize) -> Self {
        CMatrix {
            n,
            data: vec![Complex::ZERO; n * n],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.set(i, i, Complex::ONE);
        }
        m
    }

    /// Embed a real matrix into the complex matrices.
    pub fn from_real(m: &RealMatrix) -> Self {
        CMatrix {
            n: m.n,
            data: m.data.iter().map(|&x| Complex::real(x)).collect(),
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Complex {
        self.data[r * self.n + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: Complex) {
        self.data[r * self.n + c] = v;
    }

    /// Conjugate transpose `A†`.
    pub fn dagger(&self) -> CMatrix {
        let n = self.n;
        CMatrix {
            n,
            data: {
                let mut d = vec![Complex::ZERO; n * n];
                for r in 0..n {
                    for c in 0..n {
                        d[c * n + r] = self.get(r, c).conj();
                    }
                }
                d
            },
        }
    }

    /// Matrix product `self * other`.
    pub fn matmul(&self, other: &CMatrix) -> CMatrix {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let mut out = CMatrix::zeros(n);
        for r in 0..n {
            for k in 0..n {
                let a = self.get(r, k);
                if a == Complex::ZERO {
                    continue;
                }
                for c in 0..n {
                    out.data[r * n + c] += a * other.get(k, c);
                }
            }
        }
        out
    }

    /// Matrix-vector product `self * v`.
    pub fn matvec(&self, v: &[Complex]) -> Vec<Complex> {
        assert_eq!(self.n, v.len());
        let n = self.n;
        let mut out = vec![Complex::ZERO; n];
        for r in 0..n {
            let mut acc = Complex::ZERO;
            for c in 0..n {
                acc += self.get(r, c) * v[c];
            }
            out[r] = acc;
        }
        out
    }

    pub fn sub(&self, other: &CMatrix) -> CMatrix {
        CMatrix {
            n: self.n,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }

    pub fn scale(&self, s: Complex) -> CMatrix {
        CMatrix {
            n: self.n,
            data: self.data.iter().map(|z| *z * s).collect(),
        }
    }

    /// Commutator `[A, B] = AB - BA`.
    pub fn commutator(a: &CMatrix, b: &CMatrix) -> CMatrix {
        a.matmul(b).sub(&b.matmul(a))
    }

    /// Frobenius norm `sqrt(Σ |a_ij|²)`.
    pub fn frob_norm(&self) -> f64 {
        self.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
    }
}

/// `exp(i * theta * H)` for a generator supplied by its **precomputed** real
/// spectral decomposition `(eigvals, V)` with `H = V diag(eigvals) Vᵀ`.
///
/// This is the cache-reuse entry point: callers who already hold the spectrum
/// (e.g. [`crate::page_wootters::PageWootters`], which diagonalizes once in
/// `new`) build any number of evolution operators `e^{iθH}` without paying for
/// re-diagonalization. The result is unitary.
pub fn exp_i_from_spectrum(eigvals: &[f64], v: &RealMatrix, theta: f64) -> CMatrix {
    let n = v.n;
    debug_assert_eq!(
        eigvals.len(),
        n,
        "spectrum length must match matrix dimension"
    );
    // phases[k] = e^{i*theta*E_k}
    let phases: Vec<Complex> = eigvals.iter().map(|&e| Complex::phase(theta * e)).collect();
    let mut out = CMatrix::zeros(n);
    for r in 0..n {
        for c in 0..n {
            let mut acc = Complex::ZERO;
            for k in 0..n {
                let w = v.get(r, k) * v.get(c, k);
                acc += phases[k].scale(w);
            }
            out.set(r, c, acc);
        }
    }
    out
}

/// Apply `exp(i * theta * H)` to a complex vector `psi` directly, using the
/// **precomputed** spectral decomposition `(eigvals, V)` — without ever forming
/// the propagator matrix. This is `O(n²)` work per call and the natural way to
/// evolve a state in its own energy eigenbasis:
///
/// ```text
///   e^{iθH} |ψ> = Σ_k e^{iθE_k} <E_k|ψ> |E_k>,   |E_k> = column k of V.
/// ```
///
/// Equivalent (to round-off) to `exp_i_from_spectrum(eigvals, v, theta).matvec(psi)`
/// but allocates no `n × n` matrix.
pub fn exp_i_apply_from_spectrum(
    eigvals: &[f64],
    v: &RealMatrix,
    theta: f64,
    psi: &[Complex],
) -> Vec<Complex> {
    let n = v.n;
    debug_assert_eq!(
        eigvals.len(),
        n,
        "spectrum length must match matrix dimension"
    );
    debug_assert_eq!(psi.len(), n, "state length must match matrix dimension");
    // Coefficients c_k = <E_k|ψ> (V is real, so the bra is just the column).
    let mut coeffs = vec![Complex::ZERO; n];
    for k in 0..n {
        let mut acc = Complex::ZERO;
        for r in 0..n {
            acc += psi[r].scale(v.get(r, k));
        }
        coeffs[k] = Complex::phase(theta * eigvals[k]) * acc;
    }
    // Reconstruct in the standard basis: out[r] = Σ_k V[r][k] * (phase_k c_k).
    let mut out = vec![Complex::ZERO; n];
    for r in 0..n {
        let mut acc = Complex::ZERO;
        for k in 0..n {
            acc += coeffs[k].scale(v.get(r, k));
        }
        out[r] = acc;
    }
    out
}

/// `exp(i * theta * H)` for a real **symmetric** generator `H`, via its
/// spectral decomposition. The result is unitary. From-scratch convenience for
/// callers who hold only `H`; reuse [`exp_i_from_spectrum`] when the spectrum is
/// already cached.
pub fn exp_i_symmetric(h: &RealMatrix, theta: f64) -> CMatrix {
    let (eigvals, v) = h.symmetric_eigen();
    exp_i_from_spectrum(&eigvals, &v, theta)
}

/// Schrödinger propagator `U(t) = e^{-iHt}` for a real symmetric Hamiltonian.
pub fn schrodinger_propagator(h: &RealMatrix, t: f64) -> CMatrix {
    exp_i_symmetric(h, -t)
}

/// Eigenvalues of a complex **Hermitian** matrix, obtained from the real
/// symmetric `2n x 2n` embedding
///
/// ```text
///   M  ->  [ Re(M)  -Im(M) ]
///          [ Im(M)   Re(M) ]
/// ```
///
/// whose spectrum is each Hermitian eigenvalue repeated twice. We sort the `2n`
/// values and keep every second one. This lets us take the von Neumann entropy
/// of an arbitrary complex reduced density matrix without a dedicated Hermitian
/// eigensolver.
pub fn hermitian_eigenvalues(m: &CMatrix) -> Vec<f64> {
    let n = m.n;
    let mut big = RealMatrix::zeros(2 * n);
    for r in 0..n {
        for c in 0..n {
            let z = m.get(r, c);
            // top-left = Re, bottom-right = Re
            big.set(r, c, z.re);
            big.set(r + n, c + n, z.re);
            // top-right = -Im, bottom-left = +Im
            big.set(r, c + n, -z.im);
            big.set(r + n, c, z.im);
        }
    }
    let (mut vals, _v) = big.symmetric_eigen();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Each true eigenvalue appears twice; take the even-indexed representatives.
    (0..n).map(|i| vals[2 * i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_is_unitary() {
        let h = RealMatrix::from_fn(2, |r, c| if r == c { 1.0 } else { 0.4 });
        let u = schrodinger_propagator(&h, 0.9);
        let prod = u.matmul(&u.dagger());
        let id = CMatrix::identity(2);
        assert!(prod.sub(&id).frob_norm() < 1e-9);
    }

    #[test]
    fn group_property() {
        // U(t1) U(t2) = U(t1 + t2)
        let h = RealMatrix::from_fn(3, |r, c| if r == c { (r as f64) - 1.0 } else { 0.3 });
        let a = schrodinger_propagator(&h, 0.5);
        let b = schrodinger_propagator(&h, 0.7);
        let ab = a.matmul(&b);
        let c = schrodinger_propagator(&h, 1.2);
        assert!(ab.sub(&c).frob_norm() < 1e-9);
    }

    #[test]
    fn hermitian_eig_matches_real_diag() {
        // A real diagonal density matrix embedded as complex.
        let mut m = CMatrix::zeros(3);
        m.set(0, 0, Complex::real(0.5));
        m.set(1, 1, Complex::real(0.3));
        m.set(2, 2, Complex::real(0.2));
        let mut e = hermitian_eigenvalues(&m);
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] - 0.2).abs() < 1e-9);
        assert!((e[2] - 0.5).abs() < 1e-9);
    }
}
