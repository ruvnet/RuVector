//! wasm-bindgen surface for `@ruvector/typesafe` — the fallback backend, kept
//! byte-for-byte behind the same JSON contract as the native binding.
//!
//! ADR-005: the built module must import NO WASI fs/net symbols;
//! `scripts/check-wasm-imports.mjs` enforces it in CI. Nothing here touches the
//! filesystem, the network, or logging of `state`.
//!
//! Error discipline: request-level failures are returned as
//! `{"error":{"kind":"limit"|"invalid"|"embedder","message":"..."}}`; only a
//! bad options JSON / unsupported embedder throws (a JS exception from the
//! constructor).

use ruvector_typesafe_core::engine::{Engine as CoreEngine, LabeledExample};
use ruvector_typesafe_core::hash_embedder::HashEmbedder;
use ruvector_typesafe_core::{DecisionRequest, Embedder, TypesafeError};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Crate (== Cargo workspace) version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[derive(Deserialize)]
struct EngineOptions {
    embedder: EmbedderSpec,
    #[serde(default = "default_dims")]
    dims: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedderSpec {
    Named(String),
    Kinded { kind: String },
}

fn default_dims() -> usize {
    256
}

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

/// A compiled decision engine. WASM is single-threaded, so the core engine is
/// held directly (no `Arc`/lock); `trainJson` takes `&mut self`.
#[wasm_bindgen]
pub struct Engine {
    inner: CoreEngine<HashEmbedder>,
}

#[wasm_bindgen]
impl Engine {
    /// `optionsJson`: `{"embedder":"hash","dims":256}`. The onnx arm
    /// (`{"embedder":{"kind":"onnx",...}}`) is rejected until the tract backend
    /// lands — use `Engine.fromBytes` for the eventual model-bytes path.
    #[wasm_bindgen(constructor)]
    pub fn new(options_json: &str) -> std::result::Result<Engine, JsValue> {
        let opts: EngineOptions = serde_json::from_str(options_json)
            .map_err(|e| JsValue::from_str(&format!("invalid options JSON: {e}")))?;
        match opts.embedder {
            EmbedderSpec::Named(ref s) if s == "hash" => Ok(Engine {
                inner: CoreEngine::new(HashEmbedder::new(opts.dims)),
            }),
            EmbedderSpec::Kinded { ref kind } if kind == "onnx" => {
                Err(JsValue::from_str(&onnx_error_json()))
            }
            EmbedderSpec::Named(s) => Err(JsValue::from_str(&format!("unknown embedder \"{s}\""))),
            EmbedderSpec::Kinded { kind } => Err(JsValue::from_str(&format!(
                "unknown embedder kind \"{kind}\""
            ))),
        }
    }

    /// Construct an onnx-backed engine from in-memory model + tokenizer bytes.
    /// Not yet implemented (needs ruvector-embed-core's tract backend); returns
    /// the documented embedder error JSON as the thrown value.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(
        _options_json: &str,
        _model_bytes: &[u8],
        _tokenizer_bytes: &[u8],
    ) -> std::result::Result<Engine, JsValue> {
        Err(JsValue::from_str(&onnx_error_json()))
    }

    /// Returns a `DecisionResponse` JSON, or the documented error JSON.
    #[wasm_bindgen(js_name = decideJson)]
    pub fn decide_json(&self, request_json: &str) -> String {
        let req: DecisionRequest = match serde_json::from_str(request_json) {
            Ok(r) => r,
            Err(e) => return invalid_json(&format!("request JSON parse error: {e}")),
        };
        match self.inner.decide(&req) {
            Ok(resp) => {
                serde_json::to_string(&resp).unwrap_or_else(|e| embedder_json(&e.to_string()))
            }
            Err(e) => error_json(&e),
        }
    }

    /// Returns a `TrainReport` JSON, or the documented error JSON.
    #[wasm_bindgen(js_name = trainJson)]
    pub fn train_json(&mut self, train_json: &str) -> String {
        let input: TrainInput = match serde_json::from_str(train_json) {
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
        match self.inner.train(&input.question, &examples) {
            Ok(report) => {
                serde_json::to_string(&report).unwrap_or_else(|e| embedder_json(&e.to_string()))
            }
            Err(e) => error_json(&e),
        }
    }

    /// `{"embedderId","dims","questionsCompiled","examples"}` (counts are 0
    /// until the core exposes them).
    #[wasm_bindgen(js_name = statsJson)]
    pub fn stats_json(&self) -> String {
        let embedder = self.inner.embedder();
        serde_json::json!({
            "embedderId": embedder.id(),
            "dims": embedder.dims(),
            "questionsCompiled": 0,
            "examples": 0,
        })
        .to_string()
    }
}

fn onnx_error_json() -> String {
    embedder_json("onnx backend requires ruvector-embed-core (tract, wasm) — not yet integrated")
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
