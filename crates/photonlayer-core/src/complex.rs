//! Minimal dependency-free complex number type used across the optical core.
//!
//! We deliberately avoid pulling an external `num-complex` version into the
//! workspace so that `photonlayer-core` stays small, deterministic, and
//! `no_std`-friendly. Only the operations the propagation engine needs are
//! implemented.

use core::ops::{Add, Mul, Sub};

/// A single-precision complex number `re + i*im`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `exp(i * theta)` — a unit phasor. Core of phase-mask application.
    #[inline]
    pub fn from_phase(theta: f32) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Squared magnitude `re^2 + im^2` (what an intensity detector records).
    #[inline]
    pub fn norm_sqr(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude `sqrt(re^2 + im^2)`.
    #[inline]
    pub fn abs(&self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Phase angle in radians, `atan2(im, re)`.
    #[inline]
    pub fn arg(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phasor_unit_magnitude() {
        for k in 0..16 {
            let theta = k as f32 * 0.4;
            let p = Complex::from_phase(theta);
            assert!((p.abs() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mul_matches_definition() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        let c = a * b;
        // (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 3 + 5i + 2 = 5 + 5i
        assert!((c.re - 5.0).abs() < 1e-6);
        assert!((c.im - 5.0).abs() < 1e-6);
    }
}
