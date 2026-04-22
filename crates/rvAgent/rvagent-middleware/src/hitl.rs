//! HumanInTheLoopMiddleware — intercepts tool calls matching interrupt patterns,
//! pausing execution awaiting human approval.

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse, ToolCall};

/// Approval decision from a human reviewer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    Deny,
    ApproveWithModification(String),
}

/// Middleware that intercepts tool calls matching configurable interrupt patterns.
///
/// - `wrap_model_call`: after the model returns, checks if any tool calls match
///   the interrupt patterns. If so, pauses execution awaiting human approval.
pub struct HumanInTheLoopMiddleware {
    /// Tool name patterns that trigger human approval.
    interrupt_patterns: Vec<String>,
}

impl HumanInTheLoopMiddleware {
    pub fn new(interrupt_patterns: Vec<String>) -> Self {
        Self { interrupt_patterns }
    }

    /// Check if a tool call matches any interrupt pattern.
    pub fn should_interrupt(&self, tool_name: &str) -> bool {
        self.interrupt_patterns.iter().any(|pattern| {
            if pattern == "*" {
                return true;
            }
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                return tool_name.starts_with(prefix);
            }
            pattern == tool_name
        })
    }
}

#[async_trait]
impl Middleware for HumanInTheLoopMiddleware {
    fn name(&self) -> &str {
        "hitl"
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        let mut response = handler.call(request);

        // Filter out tool calls that require approval
        let (needs_approval, approved): (Vec<ToolCall>, Vec<ToolCall>) = response
            .tool_calls
            .drain(..)
            .partition(|tc| self.should_interrupt(&tc.name));

        // Keep approved tool calls
        response.tool_calls = approved;

        // For tool calls needing approval, mark them as pending in the response.
        if !needs_approval.is_empty() {
            let pending_names: Vec<String> =
                needs_approval.iter().map(|tc| tc.name.clone()).collect();
            tracing::info!(
                "HITL: {} tool calls require approval: {:?}",
                pending_names.len(),
                pending_names
            );

            if !response.message.content.is_empty() {
                response.message.content.push_str("\n\n");
            }
            response.message.content.push_str(&format!(
                "[HITL] Awaiting approval for: {}",
                pending_names.join(", ")
            ));
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    struct EchoHandler;
    impl ModelHandler for EchoHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            let mut response = ModelResponse::text("response");
            response.tool_calls = vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "execute".into(),
                    args: serde_json::json!({"command": "rm -rf /"}),
                },
                ToolCall {
                    id: "call-2".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": "safe.txt"}),
                },
            ];
            response
        }
    }

    #[test]
    fn test_middleware_name() {
        let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
        assert_eq!(mw.name(), "hitl");
    }

    #[test]
    fn test_should_interrupt_exact_match() {
        let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
        assert!(mw.should_interrupt("execute"));
        assert!(!mw.should_interrupt("read_file"));
    }

    #[test]
    fn test_should_interrupt_wildcard() {
        let mw = HumanInTheLoopMiddleware::new(vec!["*".into()]);
        assert!(mw.should_interrupt("any_tool"));
        assert!(mw.should_interrupt("execute"));
    }

    #[test]
    fn test_should_interrupt_prefix_wildcard() {
        let mw = HumanInTheLoopMiddleware::new(vec!["write_*".into()]);
        assert!(mw.should_interrupt("write_file"));
        assert!(mw.should_interrupt("write_anything"));
        assert!(!mw.should_interrupt("read_file"));
    }

    #[test]
    fn test_wrap_model_call_filters_tool_calls() {
        let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
        let request = ModelRequest::new(vec![Message::user("do something")]);
        let handler = EchoHandler;
        let response = mw.wrap_model_call(request, &handler);

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "read_file");
        assert!(response.message.content.contains("[HITL]"));
        assert!(response.message.content.contains("execute"));
    }

    #[test]
    fn test_wrap_model_call_no_interrupt() {
        let mw = HumanInTheLoopMiddleware::new(vec!["dangerous_tool".into()]);
        let request = ModelRequest::new(vec![Message::user("safe")]);
        let handler = EchoHandler;
        let response = mw.wrap_model_call(request, &handler);

        assert_eq!(response.tool_calls.len(), 2);
        assert!(!response.message.content.contains("[HITL]"));
    }

    #[test]
    fn test_multiple_patterns() {
        let mw = HumanInTheLoopMiddleware::new(vec!["execute".into(), "write_file".into()]);
        assert!(mw.should_interrupt("execute"));
        assert!(mw.should_interrupt("write_file"));
        assert!(!mw.should_interrupt("read_file"));
    }

    #[test]
    fn test_empty_patterns() {
        let mw = HumanInTheLoopMiddleware::new(vec![]);
        assert!(!mw.should_interrupt("anything"));
    }
}
