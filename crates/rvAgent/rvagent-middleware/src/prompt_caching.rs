//! Prompt caching middleware.
//!
//! Marks system messages and tool definitions with cache control hints
//! for providers that support prompt caching (e.g., Anthropic).

use async_trait::async_trait;

use crate::{CacheControl, Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Middleware that adds cache control annotations to requests.
pub struct PromptCachingMiddleware;

impl PromptCachingMiddleware {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Middleware for PromptCachingMiddleware {
    fn name(&self) -> &str {
        "prompt_caching"
    }

    fn modify_request(&self, mut request: ModelRequest) -> ModelRequest {
        // Mark system message for caching if present.
        if request.system_message.is_some() {
            request.cache_control.insert(
                "system".to_string(),
                CacheControl {
                    cache_type: "ephemeral".to_string(),
                },
            );
        }
        // Mark tool definitions for caching if present.
        if !request.tools.is_empty() {
            request.cache_control.insert(
                "tools".to_string(),
                CacheControl {
                    cache_type: "ephemeral".to_string(),
                },
            );
        }
        request
    }
}
