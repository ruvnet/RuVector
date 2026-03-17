//! Integration with external services
//!
//! This module provides:
//! - PiRuvIoClient for π.ruv.io cloud brain submission
//! - GeminiGoapReasoner for GOAP planning with Gemini 2.5 Flash
//! - Google Secret Manager integration

mod pi_ruvio;
mod gemini;
mod secrets;

pub use pi_ruvio::PiRuvIoClient;
pub use gemini::GeminiGoapReasoner;
pub use secrets::SecretManager;
