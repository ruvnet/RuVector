//! subagents middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct SubagentsMiddleware;

impl SubagentsMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for SubagentsMiddleware {
    fn name(&self) -> &str { "subagents" }
}
