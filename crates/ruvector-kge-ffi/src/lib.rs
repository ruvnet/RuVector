//! napi-rs surface for @ruvector/kge (filled in by the bindings agent).
use napi_derive::napi;

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
