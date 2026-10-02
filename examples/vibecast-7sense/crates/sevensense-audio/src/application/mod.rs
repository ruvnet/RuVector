//! Application layer for the audio ingestion bounded context.
//!
//! This module contains application services that orchestrate
//! domain operations and infrastructure components.

pub mod error;
/// Async ingestion orchestration. Requires the `full` feature.
#[cfg(feature = "full")]
pub mod services;

pub use error::*;
#[cfg(feature = "full")]
pub use services::*;
