//! Tool result sanitizer middleware (ADR-103 C3).
use async_trait::async_trait;
use crate::{Middleware, ModelRequest, ModelResponse, ModelHandler, Message, Role};

pub struct ToolResultSanitizerMiddleware;

impl ToolResultSanitizerMiddleware {
    pub fn new() -> Self { Self }

    /// Wrap a tool result in XML delimiters for defense-in-depth against
    /// indirect prompt injection (ADR-103 C3 / SEC-009).
    pub fn sanitize(tool_name: &str, tool_call_id: &str, content: &str) -> String {
        format!(
            "<tool_output tool=\"{}\" id=\"{}\">\n{}\n</tool_output>",
            tool_name, tool_call_id, content
        )
    }
}

#[async_trait]
impl Middleware for ToolResultSanitizerMiddleware {
    fn name(&self) -> &str { "tool_sanitizer" }

    fn wrap_model_call(
        &self,
        mut request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        // Sanitize all tool messages in the request
        for msg in &mut request.messages {
            if msg.role == Role::Tool {
                if let (Some(ref id), Some(ref name)) = (&msg.tool_call_id, &msg.tool_name) {
                    msg.content = Self::sanitize(name, id, &msg.content);
                }
            }
        }
        handler.call(request)
    }
}
