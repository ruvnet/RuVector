//! tool_sanitizer middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct Tool_sanitizerMiddleware;

impl Tool_sanitizerMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for Tool_sanitizerMiddleware {
    fn name(&self) -> &str { "tool_sanitizer" }
}
