//! INT8 Kernels - Scalar and SIMD implementations

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod simd;
