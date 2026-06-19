//! # PhotonLayer Core
//!
//! Pure-Rust optical-computing simulator implementing the learned-optical-
//! frontend pipeline of **ADR-260**:
//!
//! ```text
//! input image
//!   -> scalar optical field      (field)
//!   -> learned phase mask         (mask)
//!   -> diffraction propagation    (propagate)
//!   -> sensor intensity frame     (detector)
//!   -> metrics + RVF receipt      (metrics, receipt)
//! ```
//!
//! The precise claim, per ADR-260: *light performs the first trained
//! transformation; a smaller digital backend reads the result.* This crate
//! owns the optical half. It is dependency-light (`serde`, `blake3`,
//! `thiserror`), deterministic (in-house FFT + seeded RNG), and the same
//! input + mask + config + seed always produce the same output hash
//! (the determinism invariant, ADR-260 §21).
//!
//! ## Example
//! ```
//! use photonlayer_core::prelude::*;
//!
//! let n = 32;
//! let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
//! let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
//! let mask = PhaseMask::random(n, n, 42);
//! let cfg = OpticalConfig::demo(n, n);
//!
//! let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
//! assert_eq!(frame.width, n);
//! // Re-running is bit-identical.
//! let frame2 = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
//! assert_eq!(frame.frame_hash, frame2.frame_hash);
//! ```

pub mod complex;
pub mod config;
pub mod detector;
pub mod error;
pub mod fft;
pub mod field;
pub mod hash;
pub mod mask;
pub mod metrics;
pub mod propagate;
pub mod receipt;
pub mod rng;
pub mod simulator;

/// Commonly used types, re-exported for ergonomic downstream use.
pub mod prelude {
    pub use crate::complex::Complex;
    pub use crate::config::{DetectorConfig, OpticalConfig, PropagationMode};
    pub use crate::detector::{capture, capture_with, OpticalFrame};
    pub use crate::error::{PhotonError, Result};
    pub use crate::field::{InputImage, OpticalField};
    pub use crate::mask::PhaseMask;
    pub use crate::metrics::{
        accuracy, compression_ratio, frame_spectrum_embedding, input_frame_similarity, mse, psnr,
        MetricReport,
    };
    pub use crate::propagate::{propagate, Propagator};
    pub use crate::receipt::{build_receipt, verify_receipt, ExperimentReceipt, Provenance};
    pub use crate::simulator::{OpticalSimulator, ScalarSimulator, SimulationTrace};
}

pub use prelude::*;
