//! The engine: validates a request, embeds `state` and the question texts,
//! runs the per-question head (ADR-003), calibrates, and returns a
//! Jev-shaped response with the additive fields. The public signature below
//! is the contract the bindings compile against; the body is filled in by
//! the heads implementation (see lib.rs module ownership).

use crate::{DecisionRequest, DecisionResponse, Embedder, Result, TypesafeError};

/// Labeled example used by `train` (text, option key / legend bucket / "yes"|"no").
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LabeledExample {
    pub text: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainReport {
    pub question: String,
    pub accepted: usize,
    pub rejected: usize,
    pub head: crate::Head,
    pub calibrated: bool,
}

pub struct Engine<E: Embedder> {
    embedder: E,
}

impl<E: Embedder> Engine<E> {
    pub fn new(embedder: E) -> Self {
        Self { embedder }
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Answer every question in `req` against `req.state`.
    pub fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse> {
        crate::limits::validate(req)?;
        let _ = &self.embedder;
        Err(TypesafeError::Invalid("engine not implemented yet".into()))
    }

    /// Admit labeled examples for one question (ADR-004 loop 1). Examples are
    /// keyed by question id so a later `decide` with the same question id and
    /// the same criteria keys uses the trained head.
    pub fn train(&mut self, question: &str, examples: &[LabeledExample]) -> Result<TrainReport> {
        let _ = (question, examples);
        Err(TypesafeError::Invalid("train not implemented yet".into()))
    }
}
