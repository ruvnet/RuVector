//! Error types for rvAgent core.

/// Top-level error enum for rvAgent operations.
#[derive(Debug, thiserror::Error)]
pub enum RvAgentError {
    /// Configuration error (invalid config, missing required fields).
    #[error("config error: {0}")]
    Config(String),

    /// Model resolution or invocation error.
    #[error("model error: {0}")]
    Model(String),

    /// Tool execution error.
    #[error("tool error: {0}")]
    Tool(String),

    /// Backend operation error.
    #[error("backend error: {0}")]
    Backend(String),

    /// Middleware pipeline error.
    #[error("middleware error: {0}")]
    Middleware(String),

    /// State manipulation error.
    #[error("state error: {0}")]
    State(String),

    /// Security policy violation.
    #[error("security error: {0}")]
    Security(String),

    /// Operation timed out.
    #[error("timeout: {0}")]
    Timeout(String),

    /// Wraps a serde_json error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Wraps a generic I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RvAgentError>;

impl RvAgentError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }
    pub fn tool(msg: impl Into<String>) -> Self {
        Self::Tool(msg.into())
    }
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
    pub fn middleware(msg: impl Into<String>) -> Self {
        Self::Middleware(msg.into())
    }
    pub fn state(msg: impl Into<String>) -> Self {
        Self::State(msg.into())
    }
    pub fn security(msg: impl Into<String>) -> Self {
        Self::Security(msg.into())
    }
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = RvAgentError::config("bad value");
        assert_eq!(e.to_string(), "config error: bad value");
    }

    #[test]
    fn test_error_variants() {
        let cases: Vec<RvAgentError> = vec![
            RvAgentError::config("c"),
            RvAgentError::model("m"),
            RvAgentError::tool("t"),
            RvAgentError::backend("b"),
            RvAgentError::middleware("mw"),
            RvAgentError::state("s"),
            RvAgentError::security("sec"),
            RvAgentError::timeout("to"),
        ];
        assert_eq!(cases.len(), 8);
        for e in &cases {
            // All should produce a non-empty display string.
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn test_from_json_error() {
        let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{invalid");
        let rv_err: RvAgentError = bad.unwrap_err().into();
        assert!(matches!(rv_err, RvAgentError::Json(_)));
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);
        let err: Result<i32> = Err(RvAgentError::config("oops"));
        assert!(err.is_err());
    }
}
