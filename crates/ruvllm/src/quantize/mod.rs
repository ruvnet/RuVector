//! Quantization Pipeline for RuvLTRA Models
//!
//! This module provides quantization capabilities for converting full-precision
//! models to optimized quantized formats suitable for edge inference on Apple Silicon.
//!
//! ## Supported Quantization Formats
//!
//! | Format | Bits | Memory (0.5B) | Quality | Use Case |
//! |--------|------|---------------|---------|----------|
//! | Q4_K_M | 4.5  | ~300 MB       | Good    | Best quality/size tradeoff |
//! | Q5_K_M | 5.5  | ~375 MB       | Better  | Higher quality, still compact |
//! | Q8_0   | 8.5  | ~500 MB       | Best    | Near-lossless quantization |
//! | PiQ3   | 3.0  | ~187 MB       | Good    | Ultra-low-bit with pi-scaling |
//!
//! ## Pi-Quantization (PiQ3)
//!
//! Pi-constant quantization uses irrational step sizes (pi/k) for better
//! information preservation at ultra-low bit-widths. Benefits include:
//! - Non-uniform grid aligned with Fourier transform properties
//! - Reduced quantization resonance (no rational harmonic buildup)
//! - ~5% lower MSE than uniform 3-bit quantization
//!
//! SIMD kernels provide high-performance dequantization:
//! - ARM NEON: >10 GB/s on Apple Silicon
//! - x86_64 AVX-512: >12 GB/s on Intel Ice Lake+ / AMD Zen4+
//! - x86_64 AVX2: >8 GB/s on modern Intel/AMD (fallback)
//!
//! ## Incoherence Processing (ADR-090 Phase 3)
//!
//! For ultra-low-bit quantization, this module provides incoherence transforms
//! using the Walsh-Hadamard algorithm. The incoherence transform spreads
//! outliers uniformly across all coefficients, reducing quantization error.
//!
//! Key property: H x H^T = n x I (orthogonal, self-inverse up to scaling)
//!
//! ```rust,ignore
//! use ruvllm::quantize::{IncoherenceTransform, IncoherenceConfig};
//!
//! // Apply incoherence before quantization
//! let mut transform = IncoherenceTransform::with_defaults()?;
//! let padded_dim = transform.apply_before_quantization(&mut weights)?;
//!
//! // ... quantize weights ...
//!
//! // Restore after dequantization
//! transform.restore_after_dequantization(&mut weights, Some(original_len))?;
//! ```
//!
//! ## Apple Neural Engine (ANE) Optimization
//!
//! The quantization pipeline produces weights optimized for ANE inference:
//! - 16-byte aligned weight layouts
//! - Blocked quantization compatible with ANE tile operations
//! - Optimized memory access patterns for M4 Pro's unified memory
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::quantize::{RuvltraQuantizer, QuantConfig, TargetFormat};
//! use std::path::Path;
//!
//! // Create quantizer for Q4_K_M format
//! let config = QuantConfig::default()
//!     .with_format(TargetFormat::Q4_K_M)
//!     .with_ane_optimization(true);
//!
//! let quantizer = RuvltraQuantizer::new(config)?;
//!
//! // Quantize a model
//! quantizer.quantize_model(
//!     Path::new("qwen-0.5b.safetensors"),
//!     Path::new("ruvltra-small-q4.gguf"),
//! )?;
//! ```

pub mod hadamard;
pub mod incoherence;
pub mod pi_quant;
pub mod pi_quant_simd;
pub mod quip;
mod ruvltra_quant;
pub mod security;
pub mod turbo_quant;

pub use ruvltra_quant::{
    dequantize_for_ane,

    // Memory estimation
    estimate_memory_q4,
    estimate_memory_q5,
    estimate_memory_q8,
    // Quantization functions
    quantize_ruvltra_q4,
    quantize_ruvltra_q5,
    quantize_ruvltra_q8,
    MemoryEstimate,

    // Block types
    Q4KMBlock,
    Q5KMBlock,
    Q8Block,

    QuantConfig,
    // Progress tracking
    QuantProgress,
    QuantStats,
    // Core quantizer
    RuvltraQuantizer,
    TargetFormat,
};

// Pi-Quantization SIMD kernels
pub use pi_quant_simd::{
    // Utility functions
    extract_pi3_value,
    // Runtime dispatch (selects best kernel)
    pi_dequantize,
    pi_dequantize_kernel_name,
    // Scalar reference (always available)
    pi_dequantize_scalar,
    pi_quantize,
    pi_quantize_kernel_name,
    pi_quantize_scalar,
    pi_quantize_value,
    pi_scale,
    pi_scale_adaptive,
    pi_scale_from_max,
    // Constants
    DEFAULT_K,
    PI3_BYTES_PER_GROUP,
    PI3_VALUES_PER_GROUP,
    PI_F32,
};

// Architecture-specific SIMD kernels (conditionally exported)
#[cfg(target_arch = "aarch64")]
pub use pi_quant_simd::pi_dequantize_neon;

#[cfg(target_arch = "x86_64")]
pub use pi_quant_simd::{pi_dequantize_avx2, pi_dequantize_avx512, pi_quantize_avx512};

// High-performance quantization (ADR-090 >1 GB/s target)
pub use pi_quant::{
    batch_quantize_3bit, quantize_2bit, quantize_2bit_fast, quantize_3bit, quantize_3bit_fast,
    quantize_kernel_name,
};

// Architecture-specific quantization kernels
#[cfg(target_arch = "aarch64")]
pub use pi_quant::{quantize_2bit_neon, quantize_3bit_neon};

#[cfg(target_arch = "x86_64")]
pub use pi_quant::{quantize_2bit_avx2, quantize_3bit_avx2};

// Hadamard transform (ADR-090 Phase 3)
pub use hadamard::{
    hadamard_batch_inverse, hadamard_batch_transform, log2_exact, next_power_of_2,
    pad_to_power_of_2, HadamardTransform, MAX_LOG_DIM, SIMD_LANES,
};

// Incoherence transform (ADR-090 Phase 3)
pub use incoherence::{
    apply_incoherence, restore_incoherence, IncoherenceConfig, IncoherenceEvent, IncoherencePhase,
    IncoherenceStats, IncoherenceTransform,
};

// QuIP 2-bit quantization (ADR-090 Phase 3)
pub use quip::{
    Q2QuipBlock, Q2QuipSuperBlock, QuipCodebook, QuipConfig, QuipMetadata, QuipQuantizer,
};

// TurboQuant data-oblivious compression (ICLR 2026)
pub use turbo_quant::{
    TurboQuantBits, TurboQuantCacheTier, TurboQuantCompressor, TurboQuantConfig,
    TurboQuantEmbeddingStore, TurboQuantKvPair, TurboQuantStats, TurboQuantized,
};
