//! Human-in-the-loop middleware stub.
use async_trait::async_trait;
use crate::Middleware;

pub struct HumanInTheLoopMiddleware {
    patterns: Vec<String>,
}
impl HumanInTheLoopMiddleware {
    pub fn new(patterns: Vec<String>) -> Self { Self { patterns } }
}
#[async_trait]
impl Middleware for HumanInTheLoopMiddleware {
    fn name(&self) -> &str { "hitl" }
}
