//! hitl middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct HitlMiddleware;

impl HitlMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for HitlMiddleware {
    fn name(&self) -> &str { "hitl" }
}
