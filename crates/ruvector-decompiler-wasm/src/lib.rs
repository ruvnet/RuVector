//! WASM bindings for the RuVector JavaScript bundle decompiler.
//!
//! Exposes the full Louvain graph-partitioning decompiler pipeline
//! (parse -> graph -> partition -> infer -> witness) to Node.js / browser.
//!
//! ## Usage from Node.js
//!
//! ```javascript
//! const wasm = require('./ruvector_decompiler_wasm');
//! const result = JSON.parse(wasm.decompile(source, '{}'));
//! console.log(result.modules.length);
//! ```

use wasm_bindgen::prelude::*;

/// Initialize the WASM module (sets up panic hook for better error messages).
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Decompile a minified JavaScript bundle using the full Louvain pipeline.
///
/// # Arguments
///
/// * `source` - The minified JavaScript source code.
/// * `config_json` - JSON string of `DecompileConfig` fields. Pass `"{}"` for defaults.
///
/// # Returns
///
/// A JSON string containing the `DecompileResult` (modules, witness, inferred names, etc.)
/// or a JSON object with an `"error"` field on failure.
#[wasm_bindgen]
pub fn decompile(source: &str, config_json: &str) -> String {
    let config: ruvector_decompiler::DecompileConfig =
        serde_json::from_str(config_json).unwrap_or_default();
    match ruvector_decompiler::decompile(source, &config) {
        Ok(result) => serde_json::to_string(&result).unwrap_or_else(|e| {
            serde_json::json!({"error": format!("serialization failed: {}", e)}).to_string()
        }),
        Err(e) => serde_json::json!({"error": e.to_string()}).to_string(),
    }
}

/// Return the version of the decompiler WASM module.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
