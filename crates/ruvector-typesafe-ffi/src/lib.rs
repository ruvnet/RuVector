//! napi-rs surface for `@ruvector/typesafe`. Decisions run synchronously via
//! `decideJson` and asynchronously as an `AsyncTask` via `decide` (ADR-002 §4);
//! the JSON contract is `ruvector_typesafe_core::types`.
//!
//! Error discipline (spec + ADR-005): request-level failures never throw across
//! the boundary — they are returned as `{"error":{"kind":"limit"|"invalid"|
//! "embedder","message":"..."}}`. Only programmer errors (malformed options
//! JSON, unsupported embedder) throw a JS `Error`.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi::Task;
use napi_derive::napi;
use ruvector_typesafe_core::engine::{Engine as CoreEngine, LabeledExample};
use ruvector_typesafe_core::hash_embedder::HashEmbedder;
use ruvector_typesafe_core::{DecisionRequest, Embedder, TypesafeError};
use serde::Deserialize;
use std::sync::{Arc, RwLock};

/// Crate (== Cargo workspace) version. Mirrors the router binding's `version()`.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ---- options JSON ---------------------------------------------------------

#[derive(Deserialize)]
struct EngineOptions {
    embedder: EmbedderSpec,
    #[serde(default = "default_dims")]
    dims: usize,
}

/// `"hash"` or `{"kind":"onnx", ...}`. onnx is parsed but rejected for now.
#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedderSpec {
    Named(String),
    Kinded { kind: String },
}

fn default_dims() -> usize {
    256
}

// ---- train JSON -----------------------------------------------------------

#[derive(Deserialize)]
struct TrainInput {
    question: String,
    #[serde(default)]
    examples: Vec<ExampleInput>,
}

#[derive(Deserialize)]
struct ExampleInput {
    text: String,
    label: String,
}

// ---- the engine -----------------------------------------------------------

/// A compiled decision engine over one embedder. `Send + Sync` (the core
/// `Engine` is, and the hash embedder holds only a `usize` + `String`), so the
/// async path can hand an `Arc` clone to the libuv thread pool. `train` needs
/// `&mut`, hence the `RwLock`.
#[napi]
pub struct Engine {
    inner: Arc<RwLock<CoreEngine<HashEmbedder>>>,
}

#[napi]
impl Engine {
    /// `optionsJson`: `{"embedder":"hash","dims":256}` or
    /// `{"embedder":{"kind":"onnx",...}}` (onnx not yet built — throws).
    /// Throws on malformed options JSON or an unsupported embedder.
    #[napi(constructor)]
    pub fn new(options_json: String) -> Result<Self> {
        let opts: EngineOptions = serde_json::from_str(&options_json)
            .map_err(|e| Error::from_reason(format!("invalid options JSON: {e}")))?;
        match opts.embedder {
            EmbedderSpec::Named(ref s) if s == "hash" => {
                let engine = CoreEngine::new(HashEmbedder::new(opts.dims));
                Ok(Self {
                    inner: Arc::new(RwLock::new(engine)),
                })
            }
            EmbedderSpec::Kinded { ref kind } if kind == "onnx" => Err(Error::from_reason(
                "onnx embedder backend is not built into this binary yet \
                 (ruvector-embed-core integration pending)"
                    .to_string(),
            )),
            EmbedderSpec::Named(s) => Err(Error::from_reason(format!("unknown embedder \"{s}\""))),
            EmbedderSpec::Kinded { kind } => Err(Error::from_reason(format!(
                "unknown embedder kind \"{kind}\""
            ))),
        }
    }

    /// Synchronous decide. Returns a `DecisionResponse` JSON, or the documented
    /// error JSON — never throws for a request-level failure.
    #[napi(js_name = "decideJson")]
    pub fn decide_json(&self, request_json: String) -> String {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        decide_to_json(&guard, &request_json)
    }

    /// Asynchronous decide, implemented as a napi `AsyncTask` so the CPU work
    /// runs off the JS thread (ADR-002 §4). Resolves to the same JSON string
    /// `decideJson` returns.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn decide(&self, request_json: String) -> AsyncTask<DecideTask> {
        AsyncTask::new(DecideTask {
            engine: self.inner.clone(),
            request_json,
        })
    }

    /// Admit labeled examples for one question. Returns a `TrainReport` JSON or
    /// the documented error JSON.
    #[napi(js_name = "trainJson")]
    pub fn train_json(&self, train_json: String) -> String {
        let input: TrainInput = match serde_json::from_str(&train_json) {
            Ok(v) => v,
            Err(e) => return invalid_json(&format!("train JSON parse error: {e}")),
        };
        let examples: Vec<LabeledExample> = input
            .examples
            .into_iter()
            .map(|e| LabeledExample {
                text: e.text,
                label: e.label,
            })
            .collect();
        let mut guard = self.inner.write().unwrap_or_else(|p| p.into_inner());
        match guard.train(&input.question, &examples) {
            Ok(report) => {
                serde_json::to_string(&report).unwrap_or_else(|e| embedder_json(&e.to_string()))
            }
            Err(e) => error_json(&e),
        }
    }

    /// Introspection: `{"embedderId","dims","questionsCompiled","examples"}`.
    /// The core does not yet expose compiled/example counts, so those are 0.
    #[napi(js_name = "statsJson")]
    pub fn stats_json(&self) -> String {
        let guard = self.inner.read().unwrap_or_else(|p| p.into_inner());
        let embedder = guard.embedder();
        serde_json::json!({
            "embedderId": embedder.id(),
            "dims": embedder.dims(),
            "questionsCompiled": 0,
            "examples": 0,
        })
        .to_string()
    }
}

/// The off-thread half of `Engine::decide`.
pub struct DecideTask {
    engine: Arc<RwLock<CoreEngine<HashEmbedder>>>,
    request_json: String,
}

impl Task for DecideTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        let guard = self.engine.read().unwrap_or_else(|p| p.into_inner());
        Ok(decide_to_json(&guard, &self.request_json))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

// ---- shared helpers -------------------------------------------------------

fn decide_to_json(engine: &CoreEngine<HashEmbedder>, request_json: &str) -> String {
    let req: DecisionRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => return invalid_json(&format!("request JSON parse error: {e}")),
    };
    match engine.decide(&req) {
        Ok(resp) => serde_json::to_string(&resp).unwrap_or_else(|e| embedder_json(&e.to_string())),
        Err(e) => error_json(&e),
    }
}

fn error_json(e: &TypesafeError) -> String {
    let (kind, message) = match e {
        TypesafeError::Limit(m) => ("limit", (*m).to_string()),
        TypesafeError::Invalid(m) => ("invalid", m.clone()),
        TypesafeError::Embedder(m) => ("embedder", m.clone()),
    };
    serde_json::json!({ "error": { "kind": kind, "message": message } }).to_string()
}

fn invalid_json(message: &str) -> String {
    serde_json::json!({ "error": { "kind": "invalid", "message": message } }).to_string()
}

fn embedder_json(message: &str) -> String {
    serde_json::json!({ "error": { "kind": "embedder", "message": message } }).to_string()
}
