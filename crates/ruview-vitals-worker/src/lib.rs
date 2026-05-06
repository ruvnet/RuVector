//! `ruview-vitals-worker` — per-Pi WiFi-CSI vital-signs worker (ADR-183
//! Tier 1).
//!
//! Listens on UDP `:5005` for [ADR-018] binary CSI frames, runs a
//! sliding-window vital-signs pipeline (breathing 0.1–0.5 Hz, heart
//! rate 0.8–2.0 Hz), exposes the readings on a gRPC service at
//! `:50054`, and posts spatial-vital memories to the cognitum-v0 brain
//! at `:9876` reusing RuView's `/memories` POST shape.
//!
//! ## Module layout
//!
//! - [`frame`] — ADR-018 binary frame parser; keeps the I/Q payload
//!   (the iter-123 telemetry bridge intentionally dropped it).
//! - [`csi`] — antenna-folded [`CsiFrame`] + `from_adr018` boundary.
//! - [`preprocessor`] — EMA static-component suppressor.
//! - [`window`] — per-subcarrier sliding ring buffer.
//! - [`biquad`] — RBJ-cookbook 2nd-order bandpass filter.
//! - [`breathing`] — bandpass + zero-crossing respiratory rate.
//! - [`heartrate`] — bandpass + autocorrelation cardiac rate.
//! - [`pipeline`] — orchestrator that wires the above into a
//!   per-frame `step(Adr018Frame) -> Option<PipelineStep>` loop.
//! - [`types`] — `VitalEstimate`, `VitalReading`, `VitalStatus`. Mirrors
//!   the upstream RuView shape so the optional `--features
//!   ruview-integration` swap is mechanical.
//! - [`error`] — crate-wide [`Error`] enum + [`Result`] alias.
//! - [`config`] — environment-variable parser ([`Config::from_env`]).
//!
//! Tier 1 follow-ups (next iters): brain POST shim, gRPC `:50054`
//! service, systemd unit + idempotent install script.
//!
//! [ADR-018]: ../../../docs/adr/ADR-018-binary-csi-frame.md

pub mod biquad;
pub mod brain;
pub mod breathing;
pub mod config;
pub mod csi;
pub mod error;
pub mod frame;
pub mod grpc;
pub mod heartrate;
pub mod mcp_brain;
pub mod pipeline;
pub mod preprocessor;
pub mod sona;
pub mod state;
pub mod types;
pub mod window;

pub use brain::{BrainClient, MemoryPost};
pub use breathing::BreathingExtractor;
pub use config::Config;
pub use csi::CsiFrame;
pub use error::{Error, Result};
pub use frame::{
    Adr018Frame, Adr018Header, CsiPayload, ADR018_HEADER_SIZE, CSI_MAGIC_V1, CSI_MAGIC_V6,
};
pub use grpc::VitalsService;
pub use heartrate::HeartRateExtractor;
pub use pipeline::{PipelineStep, VitalsPipeline};
pub use preprocessor::CsiVitalPreprocessor;
pub use state::{WorkerState, WorkerStats, WorkerStatsSnapshot};
pub use types::{NodeId, VitalEstimate, VitalReading, VitalStatus};
pub use window::CsiSlidingWindow;

/// Generated tonic stubs from `proto/vitals.proto`. Both client + server
/// sides are emitted so the same crate can be linked from coordinator
/// tooling later (e.g. a future `ruvector-vitals-stats` binary).
pub mod proto {
    tonic::include_proto!("cognitum.ruview.vitals.v1");
}

/// Crate version — surfaced on the gRPC `Health` RPC response.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
