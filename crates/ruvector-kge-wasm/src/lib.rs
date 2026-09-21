//! wasm-bindgen surface for @ruvector/kge (filled in by the bindings agent).
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
