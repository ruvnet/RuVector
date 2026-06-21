//! Minimal complex-scalar arithmetic.
//!
//! Self-contained so the crate pulls in no external linear-algebra deps. Only
//! the operations needed by the emergent-time formalisms are provided.

use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// A complex number `re + im*i` over `f64`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// Real scalar embedded into the complex plane.
    #[inline]
    pub const fn real(re: f64) -> Self {
        Complex { re, im: 0.0 }
    }

    /// `e^{i*theta}` — a unit phasor.
    #[inline]
    pub fn phase(theta: f64) -> Self {
        Complex {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Complex conjugate `re - im*i`.
    #[inline]
    pub fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus `|z|^2`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus `|z|`.
    #[inline]
    pub fn modulus(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Scale by a real factor.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Complex) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Neg for Complex {
    type Output = Complex;
    #[inline]
    fn neg(self) -> Complex {
        Complex {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: f64) -> Complex {
        self.scale(rhs)
    }
}

/// `<a|b>` inner product of two complex vectors (conjugate-linear in the first
/// argument, matching the physics convention).
pub fn inner(a: &[Complex], b: &[Complex]) -> Complex {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = Complex::ZERO;
    for i in 0..a.len() {
        acc += a[i].conj() * b[i];
    }
    acc
}

/// L2 norm of a complex vector.
pub fn vec_norm(v: &[Complex]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// Return a unit-norm copy of `v` (unchanged if it is the zero vector).
pub fn normalized(v: &[Complex]) -> Vec<Complex> {
    let n = vec_norm(v);
    if n < 1e-300 {
        return v.to_vec();
    }
    v.iter().map(|z| z.scale(1.0 / n)).collect()
}

/// Quantum fidelity `|<a|b>|` between two normalized state vectors. Equals 1.0
/// when they coincide up to a global phase.
pub fn fidelity(a: &[Complex], b: &[Complex]) -> f64 {
    let a = normalized(a);
    let b = normalized(b);
    inner(&a, &b).modulus()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_and_conj() {
        let z = Complex::new(2.0, 3.0);
        assert_eq!(z * z.conj(), Complex::real(13.0));
    }

    #[test]
    fn phase_is_unit() {
        let p = Complex::phase(0.7);
        assert!((p.modulus() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fidelity_phase_invariant() {
        let a = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        // global phase rotation by e^{i*1.1}
        let ph = Complex::phase(1.1);
        let b: Vec<_> = a.iter().map(|z| *z * ph).collect();
        assert!((fidelity(&a, &b) - 1.0).abs() < 1e-12);
    }
}
