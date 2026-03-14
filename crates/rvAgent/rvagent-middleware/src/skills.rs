//! skills middleware stub.

use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig, ModelRequest, ModelResponse, ModelHandler};

pub struct SkillsMiddleware;

impl SkillsMiddleware {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Middleware for SkillsMiddleware {
    fn name(&self) -> &str { "skills" }
}
