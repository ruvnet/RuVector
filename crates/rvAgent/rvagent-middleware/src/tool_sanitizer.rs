//! Tool result sanitizer middleware (ADR-103 C3).
//!
//! Wraps all tool results in clearly delimited blocks to defend against
//! indirect prompt injection via file contents, grep results, or command output.

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Middleware that sanitizes tool results by wrapping them in delimited blocks.
pub struct ToolResultSanitizerMiddleware;

impl ToolResultSanitizerMiddleware {
    pub fn new() -> Self {
        Self
    }

    /// Wrap tool output in a delimited block.
    pub fn sanitize_tool_output(tool_name: &str, tool_call_id: &str, content: &str) -> String {
        format!(
            "<tool_output tool=\"{}\" id=\"{}\">\n{}\n</tool_output>",
            tool_name, tool_call_id, content
        )
    }
}

#[async_trait]
impl Middleware for ToolResultSanitizerMiddleware {
    fn name(&self) -> &str {
        "tool_result_sanitizer"
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        // Sanitize tool messages in the request before passing to the model.
        let mut sanitized = request;
        for msg in &mut sanitized.messages {
            if msg.role == crate::Role::Tool {
                if let (Some(ref id), Some(ref name)) = (&msg.tool_call_id, &msg.tool_name) {
                    msg.content = Self::sanitize_tool_output(name, id, &msg.content);
                }
            }
        }
        handler.call(sanitized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_tool_output() {
        let result = ToolResultSanitizerMiddleware::sanitize_tool_output(
            "read_file",
            "tc_1",
            "file contents here",
        );
        assert!(result.starts_with("<tool_output"));
        assert!(result.contains("read_file"));
        assert!(result.contains("tc_1"));
        assert!(result.contains("file contents here"));
        assert!(result.ends_with("</tool_output>"));
    }
}
