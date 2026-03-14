//! Witness chain middleware (ADR-103 B3).
//!
//! Records tool call entries for audit provenance. Each tool call is hashed
//! and added to the witness chain for later verification.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// A single entry in the witness chain.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Tool name that was called.
    pub tool_name: String,
    /// Hash of the arguments (placeholder — would use SHAKE-256).
    pub arguments_hash: Vec<u8>,
    /// Timestamp of the call.
    pub timestamp: u64,
}

/// Middleware that builds a witness chain of all tool calls for audit provenance.
pub struct WitnessMiddleware {
    entries: Mutex<Vec<WitnessEntry>>,
}

impl WitnessMiddleware {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Get the current witness chain entries.
    pub fn entries(&self) -> Vec<WitnessEntry> {
        self.entries.lock().unwrap().clone()
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

        // Record each tool call in the witness chain.
        let mut entries = self.entries.lock().unwrap();
        for tc in &response.tool_calls {
            let args_bytes = serde_json::to_vec(&tc.args).unwrap_or_default();
            // Placeholder hash — in production use SHAKE-256.
            let hash: Vec<u8> = {
                let mut h = 0u64;
                for b in &args_bytes {
                    h = h.wrapping_mul(31).wrapping_add(*b as u64);
                }
                h.to_le_bytes().to_vec()
            };

            entries.push(WitnessEntry {
                tool_name: tc.name.clone(),
                arguments_hash: hash,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_middleware_name() {
        let mw = WitnessMiddleware::new();
        assert_eq!(mw.name(), "witness");
        assert!(mw.entries().is_empty());
    }
}
