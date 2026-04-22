//! Authentication and rate-limiting middleware for the ACP server.
//!
//! Implements ADR-103 C6:
//! - API key authentication via `Authorization: Bearer` header
//! - Token-bucket rate limiting per IP address
//! - Request body size enforcement

use axum::{
    extract::{ConnectInfo, Request},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::types::ErrorResponse;

// ---------------------------------------------------------------------------
// API key authentication
// ---------------------------------------------------------------------------

/// Shared state holding the optional API key.
#[derive(Debug, Clone)]
pub struct ApiKeyState {
    /// When `None`, authentication is disabled.
    pub api_key: Option<String>,
}

/// Axum middleware that validates `Authorization: Bearer <key>`.
///
/// Skips validation for the `/health` endpoint and when no API key is configured.
pub async fn require_api_key(request: Request, next: Next) -> Result<Response, Response> {
    // Extract API key state from extensions.
    let api_key_state = request.extensions().get::<ApiKeyState>().cloned();

    let expected_key = match api_key_state {
        Some(state) => state.api_key,
        None => None,
    };

    // Skip auth if no key is configured.
    let expected_key = match expected_key {
        Some(k) => k,
        None => return Ok(next.run(request).await),
    };

    // Skip auth for /health.
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    // Extract and validate the Bearer token.
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected_key {
                Ok(next.run(request).await)
            } else {
                Err(unauthorized_response("invalid API key"))
            }
        }
        Some(_) => Err(unauthorized_response("malformed Authorization header")),
        None => Err(unauthorized_response("missing Authorization header")),
    }
}

fn unauthorized_response(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse::unauthorized(message)),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Rate limiter (token bucket per IP)
// ---------------------------------------------------------------------------

/// Per-IP token bucket state.
#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Shared rate limiter state.
#[derive(Debug, Clone)]
pub struct RateLimiterState {
    /// Maximum requests per minute.
    pub rate_limit: u32,
    /// Per-IP buckets.
    buckets: Arc<Mutex<HashMap<String, Bucket>>>,
}

impl RateLimiterState {
    /// Create a new rate limiter with the given requests-per-minute limit.
    pub fn new(rate_limit: u32) -> Self {
        Self {
            rate_limit,
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Try to consume one token for the given IP. Returns `true` if allowed.
    pub fn try_acquire(&self, ip: &str) -> bool {
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        let max_tokens = self.rate_limit as f64;
        let refill_rate = max_tokens / 60.0; // tokens per second

        let bucket = buckets.entry(ip.to_string()).or_insert(Bucket {
            tokens: max_tokens,
            last_refill: Instant::now(),
        });

        // Refill tokens based on elapsed time.
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * refill_rate).min(max_tokens);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Axum middleware that enforces per-IP rate limiting.
///
/// Skips rate limiting for the `/health` endpoint.
pub async fn rate_limiter(request: Request, next: Next) -> Result<Response, Response> {
    // Skip for /health.
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let limiter = request.extensions().get::<RateLimiterState>().cloned();

    let limiter = match limiter {
        Some(l) => l,
        None => return Ok(next.run(request).await),
    };

    // Extract client IP from ConnectInfo or fall back to "unknown".
    let ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    if limiter.try_acquire(&ip) {
        Ok(next.run(request).await)
    } else {
        Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse::too_many_requests("rate limit exceeded")),
        )
            .into_response())
    }
}

// ---------------------------------------------------------------------------
// Request size limit
// ---------------------------------------------------------------------------

/// Axum middleware that enforces a maximum request body size.
///
/// This is a defense-in-depth layer on top of tower-http's `RequestBodyLimit`.
/// Checks the `Content-Length` header; actual body limiting is done by the
/// tower-http layer configured in `AcpServer::router()`.
pub async fn request_size_limit(request: Request, next: Next) -> Result<Response, Response> {
    // Skip for /health.
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let max_size = request
        .extensions()
        .get::<MaxBodySize>()
        .map(|m| m.0)
        .unwrap_or(1024 * 1024); // 1 MB default

    if let Some(content_length) = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
    {
        if content_length > max_size {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorResponse::payload_too_large(format!(
                    "request body exceeds maximum size of {} bytes",
                    max_size
                ))),
            )
                .into_response());
        }
    }

    Ok(next.run(request).await)
}

/// Extension type carrying the configured max body size.
#[derive(Debug, Clone, Copy)]
pub struct MaxBodySize(pub usize);

// ---------------------------------------------------------------------------
// TLS requirement middleware
// ---------------------------------------------------------------------------

/// Extension type carrying the TLS requirement flag.
#[derive(Debug, Clone, Copy)]
pub struct RequireTls(pub bool);

/// Axum middleware that enforces TLS for non-localhost connections.
///
/// Checks the `x-forwarded-proto` header and the `Host` header to determine
/// if the connection is secure. Allows localhost connections without TLS.
pub async fn require_tls_middleware(request: Request, next: Next) -> Result<Response, Response> {
    // Skip for /health endpoint.
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let require_tls = request
        .extensions()
        .get::<RequireTls>()
        .map(|r| r.0)
        .unwrap_or(false);

    if !require_tls {
        return Ok(next.run(request).await);
    }

    // Check if this is a localhost connection.
    let is_localhost = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(|host| {
            host.starts_with("localhost")
                || host.starts_with("127.0.0.1")
                || host.starts_with("[::1]")
        })
        .unwrap_or(false);

    if is_localhost {
        return Ok(next.run(request).await);
    }

    // Check if the connection is using HTTPS via reverse proxy.
    let is_https = request
        .headers()
        .get("x-forwarded-proto")
        .and_then(|p| p.to_str().ok())
        .map(|proto| proto.eq_ignore_ascii_case("https"))
        .unwrap_or(false);

    if is_https {
        Ok(next.run(request).await)
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse::forbidden(
                "TLS is required for non-localhost connections",
            )),
        )
            .into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiterState::new(60);
        // Should allow up to 60 requests.
        for _ in 0..60 {
            assert!(limiter.try_acquire("127.0.0.1"));
        }
    }

    #[test]
    fn test_rate_limiter_blocks_excess() {
        let limiter = RateLimiterState::new(5);
        for _ in 0..5 {
            assert!(limiter.try_acquire("10.0.0.1"));
        }
        // 6th request should be blocked.
        assert!(!limiter.try_acquire("10.0.0.1"));
    }

    #[test]
    fn test_rate_limiter_per_ip_isolation() {
        let limiter = RateLimiterState::new(2);
        assert!(limiter.try_acquire("1.1.1.1"));
        assert!(limiter.try_acquire("1.1.1.1"));
        assert!(!limiter.try_acquire("1.1.1.1"));

        // Different IP should still have tokens.
        assert!(limiter.try_acquire("2.2.2.2"));
        assert!(limiter.try_acquire("2.2.2.2"));
        assert!(!limiter.try_acquire("2.2.2.2"));
    }

    #[test]
    fn test_error_response_unauthorized() {
        let resp = ErrorResponse::unauthorized("bad key");
        assert_eq!(resp.status, 401);
        assert_eq!(resp.error, "unauthorized");
    }

    #[test]
    fn test_max_body_size_clone() {
        let m = MaxBodySize(1024);
        let m2 = m;
        assert_eq!(m2.0, 1024);
    }

    #[test]
    fn test_require_tls_clone() {
        let r = RequireTls(true);
        let r2 = r;
        assert_eq!(r2.0, true);
    }

    #[test]
    fn test_error_response_forbidden() {
        let resp = ErrorResponse::forbidden("TLS required");
        assert_eq!(resp.status, 403);
        assert_eq!(resp.error, "forbidden");
    }
}
