//! # connectome-fly — RuVector connectome-driven embodied brain demonstrator
//!
//! This crate is a self-contained example implementing ADR-154. It ships a
//! synthetic fly-like connectome generator, an event-driven leaky
//! integrate-and-fire (LIF) kernel, a deterministic current-injection
//! stimulus stub (embodiment is deferred), a spike observer with a Fiedler
//! coherence-collapse detector, and an analysis layer that plugs
//! `ruvector-mincut`, `ruvector-sparsifier`, and `ruvector-attention` into a
//! live simulation.
//!
//! This is **not** consciousness upload, mind upload, or a digital-person
//! claim. It is a graph-native runtime with auditable structural analysis.
//! See `docs/research/connectome-ruvector/07-positioning.md` for the
//! hype-avoidance rubric binding on every public artifact of this crate.
//!
//! ## Quick start
//!
//! ```no_run
//! use connectome_fly::{Connectome, ConnectomeConfig, Engine, EngineConfig,
//!     Stimulus, Observer, Report};
//!
//! // Build the synthetic connectome (N=1024, ~50k synapses by default).
//! let cfg = ConnectomeConfig::default();
//! let conn = Connectome::generate(&cfg);
//!
//! // Construct the LIF engine.
//! let mut engine = Engine::new(&conn, EngineConfig::default());
//!
//! // Inject deterministic stimulus into sensory neurons.
//! let stim = Stimulus::pulse_train(
//!     conn.sensory_neurons(),
//!     /* onset_ms = */ 100.0,
//!     /* duration_ms = */ 200.0,
//!     /* amplitude_pa = */ 80.0,
//!     /* rate_hz = */ 100.0,
//! );
//!
//! // Observe spikes + coherence events.
//! let mut obs = Observer::new(conn.num_neurons());
//!
//! // Run 500 ms.
//! engine.run_with(&stim, &mut obs, /* t_end_ms = */ 500.0);
//!
//! let report: Report = obs.finalize();
//! println!("spikes = {}, coherence_events = {}",
//!          report.total_spikes, report.coherence_events.len());
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
// Demo-crate ergonomics: clippy's pedantic cast-truncation / f64 precision
// warnings fire on every u32→f32 scale math we use. Explicit allows keep
// `cargo clippy -- -D warnings` usable without papering over real issues.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::module_inception)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::collapsible_if)]

pub mod analysis;
pub mod connectome;
pub mod lif;
pub mod observer;
pub mod stimulus;

pub use analysis::{
    Analysis, AnalysisConfig, DiskAnnMotifIndex, EmbeddingF32, FunctionalPartition, MotifEmbedding,
    MotifHit, MotifIndex, MotifSignature, VamanaParams,
};
pub use connectome::{
    load_flywire, Connectome, ConnectomeConfig, ConnectomeError, FlyWireNeuronId, FlywireError,
    NeuronClass, NeuronId, NeuronMeta, Sign, Synapse,
};
pub use lif::{Engine, EngineConfig, LifError, NeuronParams, Spike, SpikeEvent};
pub use observer::{CoherenceEvent, Observer, Report};
pub use stimulus::{CurrentInjection, Stimulus};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
