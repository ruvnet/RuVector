//! INT8 Quantized Kernels Module
//!
//! Provides SIMD-optimized INT8 kernels for:
//! - 2D Convolution (standard and depthwise)
//! - Matrix multiplication (GEMM)
//! - Dot product operations
//!
//! ## Architecture Support
//!
//! - **x86_64**: AVX2 with `_mm256_maddubs_epi16` and `_mm256_madd_epi16`
//! - **aarch64**: ARM NEON with `vmull_s8` and `vpadalq_s16`
//! - **wasm32**: WebAssembly SIMD128 with `i8x16` operations

pub mod int8_avx2;
pub mod int8_neon;
pub mod int8_wasm;
