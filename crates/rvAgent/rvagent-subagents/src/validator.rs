//! SubAgent result validation (ADR-103 C8 / SEC-011).

/// Default maximum response length in bytes (100KB).
pub const DEFAULT_MAX_RESPONSE_LENGTH: usize = 100 * 1024;

/// Known prompt injection patterns to detect in subagent results.
const INJECTION_PATTERNS: &[&str] = &[
    "ignore previous instructions",
    "disregard all prior",
    "you are now",
    "new system prompt",
    "override your instructions",
    "<|im_start|>system",
];

/// Validates subagent results for security concerns.
pub struct SubAgentResultValidator {
    max_response_length: usize,
}

impl SubAgentResultValidator {
    /// Create with default settings.
    pub fn new() -> Self {
        Self {
            max_response_length: DEFAULT_MAX_RESPONSE_LENGTH,
        }
    }

    /// Create with custom max response length.
    pub fn with_max_length(max_response_length: usize) -> Self {
        Self {
            max_response_length,
        }
    }

    /// Validate a result message. Returns Ok(()) if valid, Err with reason if not.
    pub fn validate(&self, result: &str) -> Result<(), String> {
        // Check length
        if result.len() > self.max_response_length {
            return Err(format!(
                "Response too long: {} bytes (max {})",
                result.len(),
                self.max_response_length
            ));
        }

        // Check for injection patterns
        let lower = result.to_lowercase();
        for pattern in INJECTION_PATTERNS {
            if lower.contains(pattern) {
                return Err(format!(
                    "Potential prompt injection detected: '{}'",
                    pattern
                ));
            }
        }

        Ok(())
    }

    /// Strip control characters from a result string.
    pub fn sanitize(&self, result: &str) -> String {
        result
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t' || *c == '\r')
            .collect()
    }
}

impl Default for SubAgentResultValidator {
    fn default() -> Self {
        Self::new()
    }
}
