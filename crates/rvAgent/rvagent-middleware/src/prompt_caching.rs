//! Prompt caching middleware stub.
use async_trait::async_trait;
use crate::Middleware;

pub struct PromptCachingMiddleware;
impl PromptCachingMiddleware {
    pub fn new() -> Self { Self }
}
#[async_trait]
impl Middleware for PromptCachingMiddleware {
    fn name(&self) -> &str { "prompt_caching" }
}
