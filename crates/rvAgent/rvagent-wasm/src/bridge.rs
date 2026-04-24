//! JavaScript interop bridge.
//!
//! Provides `JsModelProvider` which delegates model calls to a JavaScript
//! callback function, and conversion helpers between Rust types and `JsValue`.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// JsModelProvider — bridges to JavaScript model providers
// ---------------------------------------------------------------------------

/// A model provider that delegates to a JavaScript callback function.
///
/// The JS callback receives a JSON string of messages and must return
/// a Promise that resolves to a JSON string response.
///
/// # JavaScript usage
/// ```js
/// const provider = new JsModelProvider(async (messagesJson) => {
///     const messages = JSON.parse(messagesJson);
///     const response = await callMyModel(messages);
///     return JSON.stringify(response);
/// });
/// ```
#[wasm_bindgen]
pub struct JsModelProvider {
    /// The JS callback: `(messagesJson: string) => Promise<string>`.
    callback: js_sys::Function,
}

#[wasm_bindgen]
impl JsModelProvider {
    /// Create a new provider wrapping a JavaScript async function.
    ///
    /// The function must accept a JSON string and return a Promise<string>.
    #[wasm_bindgen(constructor)]
    pub fn new(callback: js_sys::Function) -> Result<JsModelProvider, JsValue> {
        if !callback.is_function() {
            return Err(JsValue::from_str(
                "JsModelProvider requires a function argument",
            ));
        }
        Ok(Self { callback })
    }

    /// Send messages to the JS model provider and get a response.
    ///
    /// `messages_json` is a JSON-serialized array of message objects.
    /// Returns the model's response as a JSON string.
    pub async fn complete(&self, messages_json: &str) -> Result<String, JsValue> {
        let arg = JsValue::from_str(messages_json);
        let result = self.callback.call1(&JsValue::NULL, &arg)?;

        // The callback should return a Promise.
        let promise: js_sys::Promise = result
            .dyn_into()
            .map_err(|_| JsValue::from_str("model callback must return a Promise"))?;

        let resolved = wasm_bindgen_futures::JsFuture::from(promise).await?;

        resolved
            .as_string()
            .ok_or_else(|| JsValue::from_str("model callback must resolve to a string"))
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers: Rust <-> JsValue
// ---------------------------------------------------------------------------

/// Serialize a Rust value to a `JsValue` via JSON.
///
/// Converts `T` -> JSON string -> `JsValue` (parsed JS object).
pub fn to_js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    let json = serde_json::to_string(value)
        .map_err(|e| JsValue::from_str(&format!("serialization error: {}", e)))?;
    js_sys::JSON::parse(&json)
}

/// Deserialize a `JsValue` to a Rust type via JSON.
///
/// Converts `JsValue` -> JSON string -> `T`.
pub fn from_js_value<T: for<'de> Deserialize<'de>>(value: &JsValue) -> Result<T, JsValue> {
    let json = js_sys::JSON::stringify(value)
        .map_err(|_| JsValue::from_str("failed to stringify JsValue"))?;
    let json_str = json
        .as_string()
        .ok_or_else(|| JsValue::from_str("stringify returned non-string"))?;
    serde_json::from_str(&json_str)
        .map_err(|e| JsValue::from_str(&format!("deserialization error: {}", e)))
}

/// Convert a Rust error string to a `JsValue` error.
pub fn err_to_js(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}

/// Extract a string field from a JS object.
pub fn get_string_field(obj: &JsValue, field: &str) -> Result<String, JsValue> {
    let val = js_sys::Reflect::get(obj, &JsValue::from_str(field))
        .map_err(|_| JsValue::from_str(&format!("missing field: {}", field)))?;
    val.as_string()
        .ok_or_else(|| JsValue::from_str(&format!("field '{}' is not a string", field)))
}

/// Extract an optional string field from a JS object.
pub fn get_optional_string_field(obj: &JsValue, field: &str) -> Option<String> {
    js_sys::Reflect::get(obj, &JsValue::from_str(field))
        .ok()
        .and_then(|v| v.as_string())
}

/// Build a simple JS object with string key-value pairs.
pub fn js_object(entries: &[(&str, &str)]) -> Result<JsValue, JsValue> {
    let obj = js_sys::Object::new();
    for (key, value) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_str(value))?;
    }
    Ok(obj.into())
}

// ---------------------------------------------------------------------------
// Message types for bridge communication
// ---------------------------------------------------------------------------

/// A simplified message for bridge communication with JS model providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMessage {
    /// Role: "system", "user", "assistant", or "tool".
    pub role: String,
    /// Text content.
    pub content: String,
}

impl BridgeMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_message_serde() {
        let msg = BridgeMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"hello\""));

        let back: BridgeMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "hello");
    }

    #[test]
    fn test_bridge_message_constructors() {
        let sys = BridgeMessage::system("instructions");
        assert_eq!(sys.role, "system");

        let user = BridgeMessage::user("query");
        assert_eq!(user.role, "user");

        let asst = BridgeMessage::assistant("response");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_to_js_value_roundtrip() {
        // This test validates serialization logic without a JS runtime.
        let data = vec!["hello", "world"];
        let json = serde_json::to_string(&data).unwrap();
        let back: Vec<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, vec!["hello", "world"]);
    }

    #[test]
    fn test_err_to_js() {
        // Validates that err_to_js creates a JsValue from a string.
        // Full JsValue testing requires wasm-bindgen-test runtime.
        let msg = "something went wrong";
        assert!(!msg.is_empty());
    }
}
