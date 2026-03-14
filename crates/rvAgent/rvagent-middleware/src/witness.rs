//! witness middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct WitnessMiddleware;

impl WitnessMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for WitnessMiddleware {
    fn name(&self) -> &str { "witness" }
}
