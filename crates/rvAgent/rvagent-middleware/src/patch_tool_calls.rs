//! patch_tool_calls middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct Patch_tool_callsMiddleware;

impl Patch_tool_callsMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for Patch_tool_callsMiddleware {
    fn name(&self) -> &str { "patch_tool_calls" }
}
