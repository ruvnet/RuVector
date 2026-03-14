//! WitnessMiddleware — logs each tool call to a witness chain (ADR-103 B3).
//! Records tool_name, arguments_hash (SHA3-256), timestamp.
//! Thread-safe via `Arc<Mutex<WitnessBuilder>>`.

use async_trait::async_trait;
use chrono::Utc;
use sha3::{Digest, Sha3_256};
use std::sync::{Arc, Mutex};

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// A single entry in the witness chain.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Name of the tool that was called.
    pub tool_name: String,
    /// SHA3-256 hash of the serialized arguments.
    pub arguments_hash: String,
    /// ISO 8601 timestamp of the call.
    pub timestamp: String,
    /// Sequential index in the witness chain.
    pub sequence: u64,
}

/// Builder that accumulates witness entries in a thread-safe manner.
#[derive(Debug)]
pub struct WitnessBuilder {
    entries: Vec<WitnessEntry>,
    next_sequence: u64,
}

impl WitnessBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Add a tool call entry to the witness chain.
    pub fn add_tool_call_entry(&mut self, tool_name: &str, args: &serde_json::Value) {
        let arguments_hash = compute_arguments_hash(args);
        let entry = WitnessEntry {
            tool_name: tool_name.to_string(),
            arguments_hash,
            timestamp: Utc::now().to_rfc3339(),
            sequence: self.next_sequence,
        };
        self.next_sequence += 1;
        self.entries.push(entry);
    }

    /// Get all entries in the witness chain.
    pub fn entries(&self) -> &[WitnessEntry] {
        &self.entries
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for WitnessBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SHA3-256 hash of tool call arguments.
pub fn compute_arguments_hash(args: &serde_json::Value) -> String {
    let serialized = serde_json::to_vec(args).unwrap_or_default();
    let mut hasher = Sha3_256::new();
    hasher.update(&serialized);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Middleware that records tool calls to a witness chain for auditability.
///
/// After the model handler returns, each tool call in the response is logged
/// with its name, argument hash, and timestamp.
pub struct WitnessMiddleware {
    builder: Arc<Mutex<WitnessBuilder>>,
}

impl WitnessMiddleware {
    pub fn new() -> Self {
        Self {
            builder: Arc::new(Mutex::new(WitnessBuilder::new())),
        }
    }

    pub fn with_builder(builder: Arc<Mutex<WitnessBuilder>>) -> Self {
        Self { builder }
    }

    /// Get a reference to the witness builder for inspection.
    pub fn builder(&self) -> &Arc<Mutex<WitnessBuilder>> {
        &self.builder
    }
}

impl Default for WitnessMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for WitnessMiddleware {
    fn name(&self) -> &str {
        "witness"
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let response = handler.call(request);

        // Log each tool call to the witness chain
        if !response.tool_calls.is_empty() {
            let mut builder = self.builder.lock().unwrap();
            for tc in &response.tool_calls {
                builder.add_tool_call_entry(&tc.name, &tc.args);
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ToolCall};

    struct ToolCallHandler {
        tool_calls: Vec<ToolCall>,
    }
    impl ModelHandler for ToolCallHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            let mut response = ModelResponse::text("done");
            response.tool_calls = self.tool_calls.clone();
            response
        }
    }

    #[test]
    fn test_middleware_name() {
        let mw = WitnessMiddleware::new();
        assert_eq!(mw.name(), "witness");
    }

    #[test]
    fn test_compute_arguments_hash() {
        let args = serde_json::json!({"path": "test.txt"});
        let hash = compute_arguments_hash(&args);
        assert_eq!(hash.len(), 64); // SHA3-256 = 32 bytes = 64 hex
        // Deterministic
        assert_eq!(hash, compute_arguments_hash(&args));
        // Different args -> different hash
        let other = serde_json::json!({"path": "other.txt"});
        assert_ne!(hash, compute_arguments_hash(&other));
    }

    #[test]
    fn test_witness_builder() {
        let mut builder = WitnessBuilder::new();
        assert!(builder.is_empty());

        builder.add_tool_call_entry("read_file", &serde_json::json!({"path": "a.txt"}));
        assert_eq!(builder.len(), 1);
        assert_eq!(builder.entries()[0].tool_name, "read_file");
        assert_eq!(builder.entries()[0].sequence, 0);

        builder.add_tool_call_entry("write_file", &serde_json::json!({}));
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.entries()[1].sequence, 1);
    }

    #[test]
    fn test_wrap_model_call_records_tool_calls() {
        let mw = WitnessMiddleware::new();
        let handler = ToolCallHandler {
            tool_calls: vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": "test.txt"}),
                },
                ToolCall {
                    id: "call-2".into(),
                    name: "execute".into(),
                    args: serde_json::json!({"command": "ls"}),
                },
            ],
        };

        let request = ModelRequest::new(vec![Message::user("test")]);
        let _response = mw.wrap_model_call(request, &handler);

        let builder = mw.builder().lock().unwrap();
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.entries()[0].tool_name, "read_file");
        assert_eq!(builder.entries()[1].tool_name, "execute");
    }

    #[test]
    fn test_wrap_model_call_no_tool_calls() {
        let mw = WitnessMiddleware::new();
        let handler = ToolCallHandler {
            tool_calls: vec![],
        };

        let request = ModelRequest::new(vec![]);
        let _response = mw.wrap_model_call(request, &handler);

        let builder = mw.builder().lock().unwrap();
        assert!(builder.is_empty());
    }

    #[test]
    fn test_thread_safety() {
        let builder = Arc::new(Mutex::new(WitnessBuilder::new()));
        let mw1 = WitnessMiddleware::with_builder(builder.clone());
        let mw2 = WitnessMiddleware::with_builder(builder.clone());

        let handler1 = ToolCallHandler {
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "tool_a".into(),
                args: serde_json::json!({}),
            }],
        };
        let handler2 = ToolCallHandler {
            tool_calls: vec![ToolCall {
                id: "c2".into(),
                name: "tool_b".into(),
                args: serde_json::json!({}),
            }],
        };

        let req1 = ModelRequest::new(vec![]);
        let req2 = ModelRequest::new(vec![]);
        mw1.wrap_model_call(req1, &handler1);
        mw2.wrap_model_call(req2, &handler2);

        let builder = builder.lock().unwrap();
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_witness_entry_has_timestamp() {
        let mut builder = WitnessBuilder::new();
        builder.add_tool_call_entry("test", &serde_json::json!({}));
        let entry = &builder.entries()[0];
        assert!(!entry.timestamp.is_empty());
        // Should be valid ISO 8601
        assert!(entry.timestamp.contains('T'));
    }
}
