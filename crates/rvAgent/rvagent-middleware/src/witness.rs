//! Witness chain middleware (ADR-103 B3).
use async_trait::async_trait;
use crate::{Middleware, ModelRequest, ModelResponse, ModelHandler};
use std::sync::{Arc, Mutex};

/// A single tool call entry in the witness chain.
#[derive(Debug, Clone)]
pub struct ToolCallEntry {
    pub tool_name: String,
    pub arguments_hash: Vec<u8>,
}

pub struct WitnessMiddleware {
    entries: Arc<Mutex<Vec<ToolCallEntry>>>,
}

impl WitnessMiddleware {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return a snapshot of all recorded entries.
    pub fn entries(&self) -> Vec<ToolCallEntry> {
        self.entries.lock().unwrap().clone()
    }
}

#[async_trait]
impl Middleware for WitnessMiddleware {
    fn name(&self) -> &str { "witness" }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let response = handler.call(request);
        // Record all tool calls from the response
        let mut entries = self.entries.lock().unwrap();
        for tc in &response.tool_calls {
            entries.push(ToolCallEntry {
                tool_name: tc.name.clone(),
                arguments_hash: tc.args.to_string().into_bytes(),
            });
        }
        response
    }
}
