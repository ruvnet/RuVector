//! # RuvLLM - Self-Learning LLM
//!
//! A self-learning language model system integrating LFM2 with Ruvector.
//!
//! ## Architecture
//!
//! The system is built on a three-layer architecture:
//!
//! - **LFM2** (Frozen core): Stable reasoning engine (350M-2.6B parameters)
//! - **Ruvector** (Living memory): Adaptive synaptic mesh that learns continuously
//! - **FastGRNN** (Control circuit): Intelligent router for resource allocation
//!
//! > "The intelligence is not in one model anymore. It is in the loop."
//!
//! ## Self-Learning Loops
//!
//! The system learns through three feedback loops:
//!
//! ### Loop A: Memory Growth & Refinement
//! - Every interaction writes to ruvector (Q&A, context, outcome)
//! - Graph edges strengthen/weaken based on success patterns
//! - Same LFM2 checkpoint → different answers over time
//!
//! ### Loop B: Router Learning
//! - FastGRNN learns optimal model selection
//! - Prefers cheaper routes when quality holds
//! - Escalates only when necessary
//!
//! ### Loop C: Compression & Abstraction
//! - Periodic summarization creates concept hierarchies
//! - Prevents unbounded memory growth
//! - Old nodes archived, concepts stay accessible
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ruvllm::{RuvLLM, Config};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = Config::builder()
//!         .db_path("./memory.db")
//!         .build()?;
//!
//!     let llm = RuvLLM::new(config).await?;
//!
//!     let response = llm.query("What is machine learning?").await?;
//!     println!("Response: {}", response.text);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::excessive_precision)]

pub mod attention;
pub mod compression;
pub mod config;
pub mod embedding;
pub mod error;
pub mod inference;
pub mod learning;
pub mod memory;
pub mod orchestrator;
pub mod router;
pub mod simd_inference;
pub mod sona;
pub mod training;
pub mod types;

#[cfg(feature = "real-inference")]
pub mod inference_real;

#[cfg(feature = "napi")]
pub mod napi;

// Re-exports
pub use config::{Config, ConfigBuilder};
pub use error::{Error, Result};
pub use inference::{GenerationConfig, GenerationResult, InferenceMode, InferencePool};
pub use orchestrator::RuvLLM;
pub use simd_inference::{SimdGenerationConfig, SimdInferenceEngine, SimdOps};
pub use sona::{BackgroundLoop, InstantLoop, LoopCoordinator, SonaConfig};
pub use types::{Feedback, Request, Response, RoutingInfo, Session};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
