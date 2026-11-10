//! Verified progressive streaming for stores (project Ofer, ADR-353).
//!
//! * [`layers`]: wire payloads of Layer A and B.
//! * [`StreamIngest`]: consumes stream frames, publishes only verified data.
//! * [`build`]: produces a stream-ready store file and its stage plan.
//! * [`plan`]: server-side frame plan with byte-range addressing.
//! * [`client`]: resumable client over a [`client::Transport`].

pub mod build;
pub mod client;
pub mod ingest;
pub mod layers;
pub mod plan;
pub mod vectors;

pub use ingest::{FeedOutcome, QueryMode, QueryPath, StreamAnswer, StreamIngest, StreamWitness};
pub use vectors::VerifiedVectors;
