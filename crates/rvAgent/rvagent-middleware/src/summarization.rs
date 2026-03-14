//! Summarization middleware stub (ADR-098).
use async_trait::async_trait;
use crate::Middleware;

/// Trigger configuration for auto-compaction.
pub enum TriggerConfig {
    Fraction(f64),
    Tokens(u64),
}

pub struct SummarizationMiddleware {
    max_tokens: u64,
    trigger_fraction: f64,
    keep_fraction: f64,
}

impl SummarizationMiddleware {
    pub fn new(max_tokens: u64, trigger_fraction: f64, keep_fraction: f64) -> Self {
        Self { max_tokens, trigger_fraction, keep_fraction }
    }

    /// Check if compaction should trigger given a token count.
    pub fn should_compact(&self, token_count: u64) -> bool {
        let threshold = (self.max_tokens as f64 * self.trigger_fraction) as u64;
        token_count > threshold
    }
}

#[async_trait]
impl Middleware for SummarizationMiddleware {
    fn name(&self) -> &str { "summarization" }
}
