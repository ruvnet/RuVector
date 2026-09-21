//! wasm-bindgen surface for `@ruvector/kge` — the fallback backend, kept
//! byte-for-byte behind the same JSON contract as the native binding. The pure
//! logic lives in [`model`] and [`ops`], shared verbatim with the ffi crate.
//!
//! ADR-005: the built module must import NO WASI fs/net symbols;
//! `scripts/check-wasm-imports.mjs` enforces it. Nothing here touches the
//! filesystem, the network, or logging of triple text.
//!
//! Error discipline: request-level failures return `{"error":{"kind","message"}}`
//! (`kind` in `limit | invalid | unavailable | unsupported | scorer`); only a
//! bad options JSON, an odd/zero `dims`, or a tampered model throws — from the
//! constructor or `fromJson`. WASM is single-threaded, so there is no async
//! `train`; `trainJson` is synchronous.

#![deny(clippy::all)]

mod model;
mod ops;
mod pipeline;

use model::KgeModel;
use wasm_bindgen::prelude::*;

/// Crate (== Cargo workspace) version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A knowledge-graph embedding model. Held directly (no lock): the wasm build
/// is single-threaded, so mutating methods take `&mut self`.
#[wasm_bindgen]
pub struct Model {
    inner: KgeModel,
}

#[wasm_bindgen]
impl Model {
    /// `optionsJson`: `{"scorer":"hole"|"rotate","dims":256,"seed":42}`. Throws
    /// on malformed JSON or a `dims` that is zero or odd.
    #[wasm_bindgen(constructor)]
    pub fn new(options_json: &str) -> std::result::Result<Model, JsValue> {
        KgeModel::new(options_json)
            .map(|inner| Model { inner })
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Rebuild a model from a `toJson` envelope; throws on a hash mismatch.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(model_json: &str) -> std::result::Result<Model, JsValue> {
        KgeModel::from_json(model_json)
            .map(|inner| Model { inner })
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Serialize to a hash-carrying `{"sha256","model"}` envelope.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Admit `[{"s","r","o"}]`. Returns a summary or error JSON.
    #[wasm_bindgen(js_name = addTriplesJson)]
    pub fn add_triples_json(&mut self, triples_json: &str) -> String {
        self.inner.add_triples_json(triples_json)
    }

    /// Link prediction: `{"s","r","k"}` or `{"r","o","k"}`.
    #[wasm_bindgen(js_name = predictJson)]
    pub fn predict_json(&mut self, query_json: &str) -> String {
        self.inner.predict_json(query_json)
    }

    /// Relation similarity: `{"r","k"}`.
    #[wasm_bindgen(js_name = similarRelationsJson)]
    pub fn similar_relations_json(&mut self, query_json: &str) -> String {
        self.inner.similar_relations_json(query_json)
    }

    /// 2-hop composition: `{"r1","r2","s","k"}` (RotatE only).
    #[wasm_bindgen(js_name = composeJson)]
    pub fn compose_json(&mut self, query_json: &str) -> String {
        self.inner.compose_json(query_json)
    }

    /// Build the ANN index over entities.
    #[wasm_bindgen(js_name = buildIndexJson)]
    pub fn build_index_json(&mut self) -> String {
        self.inner.build_index_json()
    }

    /// Synchronous train (the only train form on wasm).
    #[wasm_bindgen(js_name = trainJson)]
    pub fn train_json(&mut self, config_json: &str) -> String {
        self.inner.train_json(config_json)
    }

    /// Filtered evaluation: `{"split":"test", ...}`.
    #[wasm_bindgen(js_name = evalJson)]
    pub fn eval_json(&mut self, config_json: &str) -> String {
        self.inner.eval_json(config_json)
    }

    /// Self-optimization campaign.
    #[wasm_bindgen(js_name = optimizeJson)]
    pub fn optimize_json(&mut self, campaign_json: &str) -> String {
        self.inner.optimize_json(campaign_json)
    }

    /// `{"scorer","dims","seed","entities","relations","triples","indexed"}`.
    #[wasm_bindgen(js_name = statsJson)]
    pub fn stats_json(&self) -> String {
        self.inner.stats_json()
    }
}
