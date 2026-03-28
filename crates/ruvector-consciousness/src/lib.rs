//! # ruvector-consciousness — SOTA Consciousness Metrics
//!
//! Ultra-optimized Rust implementation of consciousness computation:
//!
//! | Module | Algorithm | Complexity |
//! |--------|-----------|-----------|
//! | [`phi`] | IIT Φ (exact) | O(2^n · n²) |
//! | [`phi`] | IIT Φ (spectral) | O(n² log n) |
//! | [`phi`] | IIT Φ (stochastic) | O(k · n²) |
//! | [`emergence`] | Causal emergence / EI | O(n³) |
//! | [`collapse`] | Quantum-inspired MIP search | O(√N · n²) |
//!
//! # Features
//!
//! - **SIMD-accelerated** KL-divergence, entropy, dense matvec (AVX2)
//! - **Zero-alloc** hot paths via bump arena
//! - **Sublinear** partition search via spectral and quantum-collapse methods
//! - **Auto-selecting** algorithm based on system size
//!
//! # Example
//!
//! ```rust
//! use ruvector_consciousness::types::{TransitionMatrix, ComputeBudget};
//! use ruvector_consciousness::phi::auto_compute_phi;
//!
//! // 4-state system (2 binary elements)
//! let tpm = TransitionMatrix::new(4, vec![
//!     0.5, 0.25, 0.25, 0.0,
//!     0.5, 0.25, 0.25, 0.0,
//!     0.5, 0.25, 0.25, 0.0,
//!     0.0, 0.0,  0.0,  1.0,
//! ]);
//!
//! let result = auto_compute_phi(&tpm, Some(0), &ComputeBudget::exact()).unwrap();
//! println!("Φ = {:.6}, algorithm = {}", result.phi, result.algorithm);
//! ```

pub mod arena;
pub mod error;
pub mod simd;
pub mod traits;
pub mod types;

#[cfg(feature = "phi")]
pub mod phi;

#[cfg(feature = "phi")]
pub mod geomip;

#[cfg(feature = "emergence")]
pub mod emergence;

#[cfg(feature = "emergence")]
pub mod rsvd_emergence;

#[cfg(feature = "collapse")]
pub mod collapse;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "solver-accel")]
pub mod sparse_accel;

#[cfg(feature = "mincut-accel")]
pub mod mincut_phi;

#[cfg(feature = "math-accel")]
pub mod chebyshev_phi;

#[cfg(feature = "coherence-accel")]
pub mod coherence_phi;

#[cfg(feature = "witness")]
pub mod witness_phi;

// IIT 4.0 / SOTA modules
#[cfg(feature = "phi")]
pub mod iit4;

#[cfg(feature = "phi")]
pub mod ces;

#[cfg(feature = "phi")]
pub mod phi_id;

#[cfg(feature = "phi")]
pub mod pid;

#[cfg(feature = "phi")]
pub mod streaming;

#[cfg(feature = "phi")]
pub mod bounds;
