//! summarization middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct SummarizationMiddleware;

impl SummarizationMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for SummarizationMiddleware {
    fn name(&self) -> &str { "summarization" }
}
