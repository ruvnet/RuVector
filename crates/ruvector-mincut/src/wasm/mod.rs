//! WASM bindings and optimizations for agentic chip
//!
//! Provides:
//! - SIMD-accelerated boundary computation
//! - Agentic chip interface
//! - Inter-core messaging
//! - Canonical min-cut FFI (ADR-117)

pub mod agentic;
pub mod simd;

#[cfg(feature = "canonical")]
pub mod canonical;

pub use agentic::*;
pub use simd::*;
