//! SubAgent middleware stub.
use async_trait::async_trait;
use crate::Middleware;

pub struct SubAgentMiddleware;
impl SubAgentMiddleware {
    pub fn new() -> Self { Self }
}
#[async_trait]
impl Middleware for SubAgentMiddleware {
    fn name(&self) -> &str { "subagent" }
}
