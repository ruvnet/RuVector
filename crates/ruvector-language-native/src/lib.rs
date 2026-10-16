//! Native adapters. Model scores express plausibility, never probability.
pub mod artifact;
pub mod graph;
pub mod model_cli;
pub mod training;

pub use artifact::{load_models, ModelArtifact, ModelBundle, NativePredictor, ScorerKind};
pub use graph::capture_quiescent;

use ruvector_language::error::{Error, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub fn content_hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|e| Error::new("SERIALIZE", e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn native_error(error: impl std::fmt::Display) -> Error {
    Error::new("NATIVE", error.to_string())
}

pub(crate) fn decode<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    let value = ruvector_language::parser::strict_json(json)?;
    serde_json::from_value(value).map_err(|e| Error::new("MODEL_FORMAT", e.to_string()))
}
