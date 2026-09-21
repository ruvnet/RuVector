//! wasm-bindgen surface for @ruvector/typesafe. No WASI fs/net imports may
//! appear in the built module (ADR-005: CI inspects the import section).

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
