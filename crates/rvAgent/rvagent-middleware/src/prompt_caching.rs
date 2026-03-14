//! prompt_caching middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct Prompt_cachingMiddleware;

impl Prompt_cachingMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for Prompt_cachingMiddleware {
    fn name(&self) -> &str { "prompt_caching" }
}
