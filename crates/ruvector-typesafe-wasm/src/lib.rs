//! wasm-bindgen surface for `@ruvector/typesafe` — the fallback backend, kept
//! byte-for-byte behind the same JSON contract as the native binding.
//!
//! ADR-005: the built module must import NO WASI fs/net symbols;
//! `scripts/check-wasm-imports.mjs` enforces it in CI. Nothing here touches the
//! filesystem, the network, or logging of `state`.
//!
//! Error discipline: request-level failures are returned as
//! `{"error":{"kind":"limit"|"invalid"|"embedder","message":"..."}}`; only a
//! bad options JSON / unsupported embedder / failed model load throws (a JS
//! exception from the constructor or `fromBytes`).

use ruvector_typesafe_core::engine::{Engine as CoreEngine, LabeledExample};
use ruvector_typesafe_core::hash_embedder::HashEmbedder;
use ruvector_typesafe_core::{DecisionRequest, Embedder, TypesafeError};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// The engine over a runtime-chosen backend (hash today, onnx via `fromBytes`
/// behind `wasm-onnx`).
type BoxedEngine = CoreEngine<Box<dyn Embedder>>;

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
    inner: BoxedEngine,
}

#[wasm_bindgen]
impl Engine {
    /// `optionsJson`: `{"embedder":"hash","dims":256}`. The onnx arm
    /// (`{"embedder":{"kind":"onnx",...}}`) needs model bytes, so it is rejected
    /// here — use `Engine.fromBytes` instead.
    #[wasm_bindgen(constructor)]
    pub fn new(options_json: &str) -> std::result::Result<Engine, JsValue> {
        let opts: EngineOptions = serde_json::from_str(options_json)
            .map_err(|e| JsValue::from_str(&format!("invalid options JSON: {e}")))?;
        match opts.embedder {
            EmbedderSpec::Named(ref s) if s == "hash" => Ok(Engine {
                inner: CoreEngine::new(Box::new(HashEmbedder::new(opts.dims))),
            }),
            EmbedderSpec::Kinded { ref kind } if kind == "onnx" => Err(JsValue::from_str(
                "onnx embedder requires Engine.fromBytes(optionsJson, modelBytes, tokenizerBytes)",
            )),
            EmbedderSpec::Named(s) => Err(JsValue::from_str(&format!("unknown embedder \"{s}\""))),
            EmbedderSpec::Kinded { kind } => Err(JsValue::from_str(&format!(
                "unknown embedder kind \"{kind}\""
            ))),
        }
    }

    /// Construct an onnx-backed engine (tract) from in-memory model + tokenizer
    /// bytes. `optionsJson` carries the model `manifest` (a JSON string or an
    /// inline object). Requires the `wasm-onnx` build feature; otherwise returns
    /// the documented embedder error JSON as the thrown value. Throws on a hash
    /// mismatch or a model tract cannot load (fail closed, ADR-005).
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(
        options_json: &str,
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> std::result::Result<Engine, JsValue> {
        build_onnx(options_json, model_bytes, tokenizer_bytes)
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

/// Build the tract-backed engine from bytes + a manifest carried in the options.
#[cfg(feature = "wasm-onnx")]
#[derive(Deserialize)]
struct FromBytesOptions {
    /// The model manifest: a JSON string, or an inline object.
    manifest: serde_json::Value,
}

#[cfg(feature = "wasm-onnx")]
fn build_onnx(
    options_json: &str,
    model_bytes: &[u8],
    tokenizer_bytes: &[u8],
) -> std::result::Result<Engine, JsValue> {
    use ruvector_embed_core::{ModelManifest, TractEmbedder};
    let opts: FromBytesOptions = serde_json::from_str(options_json)
        .map_err(|e| JsValue::from_str(&format!("invalid options JSON: {e}")))?;
    let manifest_json = match opts.manifest {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };
    let manifest = ModelManifest::from_json(&manifest_json)
        .map_err(|e| JsValue::from_str(&format!("manifest: {e}")))?;
    let embedder = TractEmbedder::from_bytes(model_bytes, tokenizer_bytes, &manifest)
        .map_err(|e| JsValue::from_str(&format!("onnx load: {e}")))?;
    Ok(Engine {
        inner: CoreEngine::new(Box::new(embedder)),
    })
}

#[cfg(not(feature = "wasm-onnx"))]
fn build_onnx(
    _options_json: &str,
    _model_bytes: &[u8],
    _tokenizer_bytes: &[u8],
) -> std::result::Result<Engine, JsValue> {
    Err(JsValue::from_str(&onnx_error_json()))
}

#[cfg(not(feature = "wasm-onnx"))]
fn onnx_error_json() -> String {
    embedder_json(
        "onnx backend requires ruvector-embed-core (tract, wasm) — rebuild with --features wasm-onnx",
    )
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
