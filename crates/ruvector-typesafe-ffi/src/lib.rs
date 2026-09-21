//! napi-rs surface for @ruvector/typesafe. Decisions run as `AsyncTask`s
//! (ADR-002 §4); the JSON contract is the one in `ruvector-typesafe-core::types`.

use napi_derive::napi;

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
