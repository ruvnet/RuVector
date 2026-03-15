//! MCP middleware for the rvagent request pipeline.

use async_trait::async_trait;

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::Result;

/// Middleware that can intercept and transform MCP requests/responses.
#[async_trait]
pub trait McpMiddleware: Send + Sync {
    /// Process a request before it reaches the server handler.
    /// Return `None` to let the request pass through, or `Some(response)` to short-circuit.
    async fn on_request(&self, request: &JsonRpcRequest) -> Result<Option<JsonRpcResponse>>;

    /// Process a response before it is sent to the client.
    async fn on_response(
        &self,
        _request: &JsonRpcRequest,
        response: JsonRpcResponse,
    ) -> Result<JsonRpcResponse> {
        Ok(response)
    }
}

/// Logging middleware that traces all requests.
pub struct LoggingMiddleware;

#[async_trait]
impl McpMiddleware for LoggingMiddleware {
    async fn on_request(&self, request: &JsonRpcRequest) -> Result<Option<JsonRpcResponse>> {
        tracing::debug!(method = %request.method, id = %request.id, "MCP request");
        Ok(None)
    }

    async fn on_response(
        &self,
        request: &JsonRpcRequest,
        response: JsonRpcResponse,
    ) -> Result<JsonRpcResponse> {
        let has_error = response.error.is_some();
        tracing::debug!(
            method = %request.method,
            id = %request.id,
            error = has_error,
            "MCP response"
        );
        Ok(response)
    }
}

/// Rate-limiting middleware that blocks requests exceeding a threshold.
pub struct RateLimitMiddleware {
    max_requests: usize,
    counter: std::sync::atomic::AtomicUsize,
}

impl RateLimitMiddleware {
    /// Create a new rate limit middleware.
    pub fn new(max_requests: usize) -> Self {
        Self {
            max_requests,
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Current request count.
    pub fn count(&self) -> usize {
        self.counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the counter.
    pub fn reset(&self) {
        self.counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[async_trait]
impl McpMiddleware for RateLimitMiddleware {
    async fn on_request(&self, request: &JsonRpcRequest) -> Result<Option<JsonRpcResponse>> {
        let count = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count >= self.max_requests {
            return Ok(Some(JsonRpcResponse::error(
                request.id.clone(),
                crate::protocol::JsonRpcError::internal_error("rate limit exceeded"),
            )));
        }
        Ok(None)
    }
}

/// Middleware pipeline that runs multiple middlewares in sequence.
pub struct McpMiddlewarePipeline {
    middlewares: Vec<Box<dyn McpMiddleware>>,
}

impl McpMiddlewarePipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the pipeline.
    pub fn push(&mut self, middleware: Box<dyn McpMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Number of middlewares.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Process a request through all middlewares.
    /// Returns `Some(response)` if any middleware short-circuits, else `None`.
    pub async fn process_request(
        &self,
        request: &JsonRpcRequest,
    ) -> Result<Option<JsonRpcResponse>> {
        for mw in &self.middlewares {
            if let Some(response) = mw.on_request(request).await? {
                return Ok(Some(response));
            }
        }
        Ok(None)
    }

    /// Process a response through all middlewares (in reverse order).
    pub async fn process_response(
        &self,
        request: &JsonRpcRequest,
        mut response: JsonRpcResponse,
    ) -> Result<JsonRpcResponse> {
        for mw in self.middlewares.iter().rev() {
            response = mw.on_response(request, response).await?;
        }
        Ok(response)
    }
}

impl Default for McpMiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

    #[tokio::test]
    async fn test_logging_middleware_passes_through() {
        let mw = LoggingMiddleware;
        let req = JsonRpcRequest::new(1, "ping");
        let result = mw.on_request(&req).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_logging_middleware_on_response() {
        let mw = LoggingMiddleware;
        let req = JsonRpcRequest::new(1, "ping");
        let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({}));
        let result = mw.on_response(&req, resp).await.unwrap();
        assert!(result.result.is_some());
    }

    #[tokio::test]
    async fn test_rate_limit_allows_under_threshold() {
        let mw = RateLimitMiddleware::new(5);
        let req = JsonRpcRequest::new(1, "ping");
        for _ in 0..5 {
            let result = mw.on_request(&req).await.unwrap();
            assert!(result.is_none());
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_over_threshold() {
        let mw = RateLimitMiddleware::new(2);
        let req = JsonRpcRequest::new(1, "ping");
        mw.on_request(&req).await.unwrap(); // 0 -> ok
        mw.on_request(&req).await.unwrap(); // 1 -> ok
        let result = mw.on_request(&req).await.unwrap(); // 2 -> blocked
        assert!(result.is_some());
        assert!(result.unwrap().error.is_some());
    }

    #[tokio::test]
    async fn test_rate_limit_reset() {
        let mw = RateLimitMiddleware::new(1);
        let req = JsonRpcRequest::new(1, "ping");
        mw.on_request(&req).await.unwrap();
        assert_eq!(mw.count(), 1);
        mw.reset();
        assert_eq!(mw.count(), 0);
    }

    #[tokio::test]
    async fn test_pipeline_empty() {
        let pipeline = McpMiddlewarePipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        let req = JsonRpcRequest::new(1, "ping");
        assert!(pipeline.process_request(&req).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_pipeline_with_logging() {
        let mut pipeline = McpMiddlewarePipeline::new();
        pipeline.push(Box::new(LoggingMiddleware));
        assert_eq!(pipeline.len(), 1);
        let req = JsonRpcRequest::new(1, "ping");
        assert!(pipeline.process_request(&req).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_pipeline_short_circuit() {
        let mut pipeline = McpMiddlewarePipeline::new();
        pipeline.push(Box::new(RateLimitMiddleware::new(0)));
        pipeline.push(Box::new(LoggingMiddleware));
        let req = JsonRpcRequest::new(1, "ping");
        let result = pipeline.process_request(&req).await.unwrap();
        assert!(result.is_some()); // first middleware blocks
    }

    #[tokio::test]
    async fn test_pipeline_process_response() {
        let mut pipeline = McpMiddlewarePipeline::new();
        pipeline.push(Box::new(LoggingMiddleware));
        let req = JsonRpcRequest::new(1, "ping");
        let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({}));
        let result = pipeline.process_response(&req, resp).await.unwrap();
        assert!(result.result.is_some());
    }

    #[tokio::test]
    async fn test_pipeline_default() {
        let pipeline = McpMiddlewarePipeline::default();
        assert!(pipeline.is_empty());
    }

    #[tokio::test]
    async fn test_logging_middleware_with_error_response() {
        let mw = LoggingMiddleware;
        let req = JsonRpcRequest::new(1, "bad");
        let resp = JsonRpcResponse::error(
            serde_json::json!(1),
            JsonRpcError::method_not_found("bad"),
        );
        let result = mw.on_response(&req, resp).await.unwrap();
        assert!(result.error.is_some());
    }

    #[test]
    fn test_rate_limit_count() {
        let mw = RateLimitMiddleware::new(10);
        assert_eq!(mw.count(), 0);
    }
}
