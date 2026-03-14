//! Filesystem middleware stub.
use async_trait::async_trait;
use crate::Middleware;

pub struct FilesystemMiddleware;
impl FilesystemMiddleware {
    pub fn new() -> Self { Self }
}
#[async_trait]
impl Middleware for FilesystemMiddleware {
    fn name(&self) -> &str { "filesystem" }
}
