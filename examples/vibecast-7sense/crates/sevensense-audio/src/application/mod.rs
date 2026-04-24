//! Application layer for the audio ingestion bounded context.
//!
//! This module contains application services that orchestrate
//! domain operations and infrastructure components.

pub mod error;
pub mod services;

pub use error::*;
pub use services::*;
