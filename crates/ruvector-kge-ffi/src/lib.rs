//! napi-rs surface for `@ruvector/kge`. The `Model` class exposes the binding
//! contract as JSON-in/JSON-out methods (identical to the wasm build); the pure
//! logic lives in [`model`] and [`ops`], shared verbatim with the wasm crate.
//!
//! Error discipline (ADR-005): request-level failures never throw — they return
//! `{"error":{"kind","message"}}` with `kind` in
//! `limit | invalid | unavailable | unsupported | scorer`. Only programmer
//! errors (malformed options JSON, an odd/zero `dims`, a tampered model) throw
//! a JS `Error`, from the constructor or `fromJson`.

#![deny(clippy::all)]

mod model;
mod ops;
mod optimize;
mod pipeline;

use model::KgeModel;
use napi::bindgen_prelude::*;
use napi::Task;
use napi_derive::napi;
use std::sync::{Arc, RwLock};

/// Crate (== Cargo workspace) version. Mirrors the router/typesafe bindings.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

type Shared = Arc<RwLock<KgeModel>>;

fn read(shared: &Shared) -> std::sync::RwLockReadGuard<'_, KgeModel> {
    shared.read().unwrap_or_else(|p| p.into_inner())
}
fn write(shared: &Shared) -> std::sync::RwLockWriteGuard<'_, KgeModel> {
    shared.write().unwrap_or_else(|p| p.into_inner())
}

/// A knowledge-graph embedding model. Mutating methods take an internal write
/// lock so the async `train` task can share the instance across the libuv pool.
#[napi]
pub struct Model {
    inner: Shared,
}

#[napi]
impl Model {
    /// `optionsJson`: `{"scorer":"hole"|"rotate","dims":256,"seed":42}`. Throws
    /// on malformed JSON or a `dims` that is zero or odd.
    #[napi(constructor)]
    pub fn new(options_json: String) -> Result<Self> {
        let m = KgeModel::new(&options_json).map_err(Error::from_reason)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(m)),
        })
    }

    /// Rebuild a model from a `toJson` envelope; throws on a hash mismatch.
    #[napi(factory, js_name = "fromJson")]
    pub fn from_json(model_json: String) -> Result<Self> {
        let m = KgeModel::from_json(&model_json).map_err(Error::from_reason)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(m)),
        })
    }

    /// Serialize to a hash-carrying `{"sha256","model"}` envelope.
    #[napi(js_name = "toJson")]
    pub fn to_json(&self) -> String {
        read(&self.inner).to_json()
    }

    /// Admit `[{"s","r","o"}]`. Returns a summary or error JSON.
    #[napi(js_name = "addTriplesJson")]
    pub fn add_triples_json(&self, triples_json: String) -> String {
        write(&self.inner).add_triples_json(&triples_json)
    }

    /// Link prediction: `{"s","r","k"}` or `{"r","o","k"}`.
    #[napi(js_name = "predictJson")]
    pub fn predict_json(&self, query_json: String) -> String {
        write(&self.inner).predict_json(&query_json)
    }

    /// Relation similarity: `{"r","k"}`.
    #[napi(js_name = "similarRelationsJson")]
    pub fn similar_relations_json(&self, query_json: String) -> String {
        write(&self.inner).similar_relations_json(&query_json)
    }

    /// 2-hop composition: `{"r1","r2","s","k"}` (RotatE only).
    #[napi(js_name = "composeJson")]
    pub fn compose_json(&self, query_json: String) -> String {
        write(&self.inner).compose_json(&query_json)
    }

    /// Build the ANN index over entities.
    #[napi(js_name = "buildIndexJson")]
    pub fn build_index_json(&self) -> String {
        write(&self.inner).build_index_json()
    }

    /// Synchronous train. The async `train` runs the same work off-thread.
    #[napi(js_name = "trainJson")]
    pub fn train_json(&self, config_json: String) -> String {
        write(&self.inner).train_json(&config_json)
    }

    /// Filtered evaluation: `{"split":"test", ...}`.
    #[napi(js_name = "evalJson")]
    pub fn eval_json(&self, config_json: String) -> String {
        write(&self.inner).eval_json(&config_json)
    }

    /// Self-optimization campaign.
    #[napi(js_name = "optimizeJson")]
    pub fn optimize_json(&self, campaign_json: String) -> String {
        write(&self.inner).optimize_json(&campaign_json)
    }

    /// `{"scorer","dims","seed","entities","relations","triples","indexed"}`.
    #[napi(js_name = "statsJson")]
    pub fn stats_json(&self) -> String {
        read(&self.inner).stats_json()
    }

    /// Native-only async training as a napi `AsyncTask` (ADR-003), so the CPU
    /// work runs off the JS thread. Resolves to the same JSON `trainJson`
    /// returns.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn train(&self, config_json: String) -> AsyncTask<TrainTask> {
        AsyncTask::new(TrainTask {
            model: self.inner.clone(),
            config_json,
        })
    }
}

/// The off-thread half of `Model::train`.
pub struct TrainTask {
    model: Shared,
    config_json: String,
}

impl Task for TrainTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(write(&self.model).train_json(&self.config_json))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}
