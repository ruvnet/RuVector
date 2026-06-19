//! # photonlayer-ruvector
//!
//! Experiment memory and verification substrate for PhotonLayer optical
//! simulations (ADR-260 §5, §11–§15).
//!
//! RuVector serves as the *experiment memory* layer — not a generic data
//! store. Concretely, this crate provides:
//!
//! * **Embeddings** ([`embedding`]) — 32-dim L2-normalised experiment vectors
//!   built from mask phase-histograms and detector frame spectra.
//! * **Memory** ([`memory`]) — in-memory store with cosine-similarity
//!   nearest-experiment recall.
//! * **Boundary analysis** ([`boundary`]) — Fiedler spectral partitioning
//!   that identifies which `OpticalConfig` variable best separates pass/fail
//!   experiment outcomes.
//! * **Coherence** ([`coherence`]) — spectral gap of the mask family similarity
//!   graph; families above the threshold qualify for demo promotion.
//! * **Receipts** ([`receipts`]) — JSON persistence and binding-digest
//!   verification of RVF-style experiment receipts.
//!
//! ## Quick Start
//! ```no_run
//! use photonlayer_core::prelude::*;
//! use photonlayer_ruvector::{
//!     embedding::experiment_embedding,
//!     memory::{ExperimentMemory, ExperimentRecord},
//!     receipts::ReceiptStore,
//! };
//!
//! let n = 16;
//! let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
//! let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
//! let mask = PhaseMask::random(n, n, 42);
//! let cfg = OpticalConfig::demo(n, n);
//! let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
//! let metrics = MetricReport::default();
//! let receipt = build_receipt("e1", &img, &mask, &cfg, &frame, &metrics, &Provenance::default());
//!
//! let embedding = experiment_embedding(&mask, &frame);
//!
//! let mut mem = ExperimentMemory::new();
//! // ... store and recall experiments ...
//!
//! let mut store = ReceiptStore::new();
//! store.insert(&receipt).unwrap();
//! assert!(store.verify("e1"));
//! ```

pub mod boundary;
pub mod coherence;
pub mod embedding;
pub mod memory;
pub mod receipts;

// Convenience re-exports for the most common types.
pub use boundary::{explain_boundary, BoundaryReport};
pub use coherence::{mask_family_coherence, FamilyCoherence};
pub use embedding::{experiment_embedding, mask_embedding};
pub use memory::{ExperimentMemory, ExperimentRecord, MaskSearchHit, NearestHit};
pub use receipts::ReceiptStore;

/// Crate version, driven from `Cargo.toml`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
