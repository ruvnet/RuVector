//! PromptCachingMiddleware — adds cache control headers for Anthropic prompt caching.

use async_trait::async_trait;

use crate::{CacheControl, Middleware, ModelRequest};

/// Middleware that marks system prompt and tool definitions as cacheable
/// for Anthropic prompt caching.
pub struct PromptCachingMiddleware {
    cache_type: String,
}

impl PromptCachingMiddleware {
    pub fn new() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }

    pub fn with_cache_type(cache_type: impl Into<String>) -> Self {
        Self {
            cache_type: cache_type.into(),
        }
    }
}

impl Default for PromptCachingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for PromptCachingMiddleware {
    fn name(&self) -> &str {
        "prompt_caching"
    }

    fn modify_request(&self, mut request: ModelRequest) -> ModelRequest {
        if request.system_message.is_some() {
            request.cache_control.insert(
                "system".to_string(),
                CacheControl {
                    cache_type: self.cache_type.clone(),
                },
            );
        }

        if !request.tools.is_empty() {
            request.cache_control.insert(
                "tools".to_string(),
                CacheControl {
                    cache_type: self.cache_type.clone(),
                },
            );
        }

        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn test_middleware_name() {
        let mw = PromptCachingMiddleware::new();
        assert_eq!(mw.name(), "prompt_caching");
    }

    #[test]
    fn test_modify_request_with_system() {
        let mw = PromptCachingMiddleware::new();
        let request = ModelRequest::new(vec![Message::user("hi")])
            .with_system(Some("You are helpful.".into()));

        let modified = mw.modify_request(request);
        assert!(modified.cache_control.contains_key("system"));
        assert_eq!(modified.cache_control["system"].cache_type, "ephemeral");
    }

    #[test]
    fn test_modify_request_without_system() {
        let mw = PromptCachingMiddleware::new();
        let request = ModelRequest::new(vec![Message::user("hi")]);

        let modified = mw.modify_request(request);
        assert!(!modified.cache_control.contains_key("system"));
    }

    #[test]
    fn test_modify_request_with_tools() {
        let mw = PromptCachingMiddleware::new();
        let mut request = ModelRequest::new(vec![Message::user("hi")]);
        request.tools.push(crate::ToolDefinition {
            name: "test".into(),
            description: "test tool".into(),
            parameters: serde_json::json!({}),
        });

        let modified = mw.modify_request(request);
        assert!(modified.cache_control.contains_key("tools"));
    }

    #[test]
    fn test_modify_request_without_tools() {
        let mw = PromptCachingMiddleware::new();
        let request = ModelRequest::new(vec![]);

        let modified = mw.modify_request(request);
        assert!(!modified.cache_control.contains_key("tools"));
    }

    #[test]
    fn test_custom_cache_type() {
        let mw = PromptCachingMiddleware::with_cache_type("persistent");
        let request = ModelRequest::new(vec![]).with_system(Some("sys".into()));

        let modified = mw.modify_request(request);
        assert_eq!(modified.cache_control["system"].cache_type, "persistent");
    }
}
