//! Human-in-the-Loop (HITL) middleware.
//!
//! Interrupts the agent pipeline when specified tool patterns are matched,
//! requiring human approval before execution proceeds.

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Middleware that interrupts on matching tool call patterns for human approval.
pub struct HumanInTheLoopMiddleware {
    /// Tool name patterns that require human approval.
    interrupt_patterns: Vec<String>,
}

impl HumanInTheLoopMiddleware {
    /// Create a new HITL middleware with the given interrupt patterns.
    pub fn new(interrupt_patterns: Vec<String>) -> Self {
        Self { interrupt_patterns }
    }

    /// Check if a tool name matches any interrupt pattern.
    pub fn should_interrupt(&self, tool_name: &str) -> bool {
        self.interrupt_patterns
            .iter()
            .any(|p| tool_name.contains(p.as_str()))
    }
}

#[async_trait]
impl Middleware for HumanInTheLoopMiddleware {
    fn name(&self) -> &str {
        "hitl"
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let response = handler.call(request);
        // In a full implementation, check tool_calls against interrupt_patterns
        // and prompt the user for approval. For now, pass through.
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hitl_should_interrupt() {
        let mw = HumanInTheLoopMiddleware::new(vec!["execute".into(), "write".into()]);
        assert!(mw.should_interrupt("execute"));
        assert!(mw.should_interrupt("write_file"));
        assert!(!mw.should_interrupt("read_file"));
    }
}
