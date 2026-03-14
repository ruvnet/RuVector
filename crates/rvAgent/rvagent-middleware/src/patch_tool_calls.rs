//! PatchToolCalls middleware — validates and normalizes tool call IDs.
//!
//! Ensures tool call IDs conform to security constraints (ADR-103 C12):
//! max 128 chars, ASCII alphanumeric + hyphens + underscores only.

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Middleware that patches and validates tool call IDs in model responses.
pub struct PatchToolCallsMiddleware;

impl PatchToolCallsMiddleware {
    pub fn new() -> Self {
        Self
    }

    /// Validate a tool call ID per ADR-103 C12.
    pub fn is_valid_tool_call_id(id: &str) -> bool {
        if id.is_empty() || id.len() > 128 {
            return false;
        }
        id.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }
}

#[async_trait]
impl Middleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str {
        "patch_tool_calls"
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let mut response = handler.call(request);
        // Validate and sanitize tool call IDs.
        for tc in &mut response.tool_calls {
            if !Self::is_valid_tool_call_id(&tc.id) {
                // Replace invalid ID with a sanitized version.
                tc.id = tc
                    .id
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                    .take(128)
                    .collect();
                if tc.id.is_empty() {
                    tc.id = format!("tc_{}", uuid::Uuid::new_v4().as_simple());
                }
            }
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tool_call_ids() {
        assert!(PatchToolCallsMiddleware::is_valid_tool_call_id("abc-123"));
        assert!(PatchToolCallsMiddleware::is_valid_tool_call_id("call_1"));
        assert!(!PatchToolCallsMiddleware::is_valid_tool_call_id(""));
        assert!(!PatchToolCallsMiddleware::is_valid_tool_call_id(
            "has spaces"
        ));
        assert!(!PatchToolCallsMiddleware::is_valid_tool_call_id(
            &"x".repeat(129)
        ));
    }
}
