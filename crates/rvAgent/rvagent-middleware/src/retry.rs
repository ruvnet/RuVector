//! Retry middleware with exponential backoff for transient model errors.
//!
//! Intercepts `wrap_model_call` and retries when the response content indicates
//! a transient error (e.g., content starts with `"error:"` or is empty).

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use async_trait::async_trait;

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Determines whether a `ModelResponse` represents a transient error worth retrying.
///
/// Heuristic: the response is considered an error if its content is empty or
/// starts with the prefix `"error:"` (case-insensitive).
fn is_transient_error(response: &ModelResponse) -> bool {
    let content = &response.message.content;
    content.is_empty() || content.to_ascii_lowercase().starts_with("error:")
}

/// Retry middleware that wraps model calls with exponential backoff.
///
/// # Configuration
///
/// | Field              | Default | Description                          |
/// |--------------------|---------|--------------------------------------|
/// | `max_retries`      | 3       | Maximum number of retry attempts     |
/// | `initial_delay_ms` | 100     | Delay before the first retry (ms)    |
///
/// The delay doubles after each attempt: `initial_delay_ms * 2^attempt`.
///
/// # Metrics
///
/// `retry_count` and `total_retries` are exposed as atomic counters so callers
/// can observe retry behaviour without locking.
pub struct RetryMiddleware {
    max_retries: u32,
    initial_delay_ms: u64,
    /// Number of model calls that required at least one retry.
    retry_count: AtomicU64,
    /// Cumulative number of individual retry attempts across all calls.
    total_retries: AtomicU64,
}

impl RetryMiddleware {
    /// Create a new `RetryMiddleware` with the given configuration.
    pub fn new(max_retries: u32, initial_delay_ms: u64) -> Self {
        Self {
            max_retries,
            initial_delay_ms,
            retry_count: AtomicU64::new(0),
            total_retries: AtomicU64::new(0),
        }
    }

    /// Number of model calls that needed at least one retry.
    pub fn retry_count(&self) -> u64 {
        self.retry_count.load(Ordering::Relaxed)
    }

    /// Total number of individual retry attempts.
    pub fn total_retries(&self) -> u64 {
        self.total_retries.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset_metrics(&self) {
        self.retry_count.store(0, Ordering::Relaxed);
        self.total_retries.store(0, Ordering::Relaxed);
    }
}

impl Default for RetryMiddleware {
    fn default() -> Self {
        Self::new(3, 100)
    }
}

#[async_trait]
impl Middleware for RetryMiddleware {
    fn name(&self) -> &str {
        "retry"
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        let mut response = handler.call(request.clone());

        if !is_transient_error(&response) {
            return response;
        }

        // At least one retry will happen — increment the call-level counter once.
        self.retry_count.fetch_add(1, Ordering::Relaxed);

        for attempt in 0..self.max_retries {
            let delay_ms = self.initial_delay_ms * 2u64.pow(attempt);
            thread::sleep(Duration::from_millis(delay_ms));

            self.total_retries.fetch_add(1, Ordering::Relaxed);

            response = handler.call(request.clone());

            if !is_transient_error(&response) {
                return response;
            }
        }

        // All retries exhausted — return the last (error) response.
        response
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ModelRequest, ModelResponse};
    use std::sync::atomic::AtomicU32;

    /// A handler that fails `n` times then succeeds.
    struct FailNHandler {
        remaining_failures: AtomicU32,
    }

    impl FailNHandler {
        fn new(n: u32) -> Self {
            Self {
                remaining_failures: AtomicU32::new(n),
            }
        }
    }

    impl ModelHandler for FailNHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            let remaining = self.remaining_failures.load(Ordering::SeqCst);
            if remaining > 0 {
                self.remaining_failures.fetch_sub(1, Ordering::SeqCst);
                ModelResponse::text("error: transient failure")
            } else {
                ModelResponse::text("success")
            }
        }
    }

    /// A handler that always succeeds.
    struct SuccessHandler;
    impl ModelHandler for SuccessHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            ModelResponse::text("ok")
        }
    }

    /// A handler that always fails with an error response.
    struct AlwaysFailHandler;
    impl ModelHandler for AlwaysFailHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            ModelResponse::text("error: permanent failure")
        }
    }

    fn make_request() -> ModelRequest {
        ModelRequest::new(vec![Message::user("hello")])
    }

    #[test]
    fn test_no_retry_on_success() {
        let mw = RetryMiddleware::default();
        let handler = SuccessHandler;
        let resp = mw.wrap_model_call(make_request(), &handler);

        assert_eq!(resp.message.content, "ok");
        assert_eq!(mw.retry_count(), 0);
        assert_eq!(mw.total_retries(), 0);
    }

    #[test]
    fn test_retry_succeeds_after_failures() {
        let mw = RetryMiddleware::new(3, 1); // 1ms delay for fast tests
        let handler = FailNHandler::new(2); // fails twice, then succeeds
        let resp = mw.wrap_model_call(make_request(), &handler);

        assert_eq!(resp.message.content, "success");
        assert_eq!(mw.retry_count(), 1);
        assert_eq!(mw.total_retries(), 2);
    }

    #[test]
    fn test_retries_exhausted() {
        let mw = RetryMiddleware::new(2, 1);
        let handler = AlwaysFailHandler;
        let resp = mw.wrap_model_call(make_request(), &handler);

        assert!(resp.message.content.starts_with("error:"));
        assert_eq!(mw.retry_count(), 1);
        assert_eq!(mw.total_retries(), 2);
    }

    #[test]
    fn test_default_config() {
        let mw = RetryMiddleware::default();
        assert_eq!(mw.max_retries, 3);
        assert_eq!(mw.initial_delay_ms, 100);
    }

    #[test]
    fn test_reset_metrics() {
        let mw = RetryMiddleware::new(3, 1);
        let handler = FailNHandler::new(1);
        let _ = mw.wrap_model_call(make_request(), &handler);

        assert!(mw.retry_count() > 0);
        mw.reset_metrics();
        assert_eq!(mw.retry_count(), 0);
        assert_eq!(mw.total_retries(), 0);
    }

    #[test]
    fn test_name() {
        let mw = RetryMiddleware::default();
        assert_eq!(mw.name(), "retry");
    }

    #[test]
    fn test_is_transient_error_empty_content() {
        let resp = ModelResponse::text("");
        assert!(is_transient_error(&resp));
    }

    #[test]
    fn test_is_transient_error_error_prefix() {
        let resp = ModelResponse::text("Error: something went wrong");
        assert!(is_transient_error(&resp));
    }

    #[test]
    fn test_is_transient_error_normal_response() {
        let resp = ModelResponse::text("Here is the answer.");
        assert!(!is_transient_error(&resp));
    }

    #[test]
    fn test_retry_first_attempt_succeeds() {
        // Edge case: handler fails on first call but succeeds on first retry (attempt 0).
        let mw = RetryMiddleware::new(5, 1);
        let handler = FailNHandler::new(1);
        let resp = mw.wrap_model_call(make_request(), &handler);

        assert_eq!(resp.message.content, "success");
        assert_eq!(mw.retry_count(), 1);
        assert_eq!(mw.total_retries(), 1);
    }

    #[test]
    fn test_zero_max_retries() {
        // With max_retries = 0, the initial call is made but no retries happen.
        let mw = RetryMiddleware::new(0, 1);
        let handler = AlwaysFailHandler;
        let resp = mw.wrap_model_call(make_request(), &handler);

        assert!(resp.message.content.starts_with("error:"));
        assert_eq!(mw.retry_count(), 1);
        assert_eq!(mw.total_retries(), 0);
    }

    #[test]
    fn test_metrics_accumulate_across_calls() {
        let mw = RetryMiddleware::new(3, 1);

        // First call: 1 failure then success
        let handler1 = FailNHandler::new(1);
        let _ = mw.wrap_model_call(make_request(), &handler1);

        // Second call: 2 failures then success
        let handler2 = FailNHandler::new(2);
        let _ = mw.wrap_model_call(make_request(), &handler2);

        assert_eq!(mw.retry_count(), 2); // two calls needed retries
        assert_eq!(mw.total_retries(), 3); // 1 + 2 retries
    }
}
