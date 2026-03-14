//! memory middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct MemoryMiddleware;

impl MemoryMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for MemoryMiddleware {
    fn name(&self) -> &str { "memory" }
}
