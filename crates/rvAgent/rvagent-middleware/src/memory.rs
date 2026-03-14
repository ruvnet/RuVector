//! Memory middleware stub.
use async_trait::async_trait;
use crate::Middleware;

pub struct MemoryMiddleware {
    sources: Vec<String>,
}
impl MemoryMiddleware {
    pub fn new(sources: Vec<String>) -> Self { Self { sources } }
}
#[async_trait]
impl Middleware for MemoryMiddleware {
    fn name(&self) -> &str { "memory" }
}
