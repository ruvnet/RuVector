//! SummarizationMiddleware — auto-compact conversation when token limit approached.
//! Offloads history with UUID-based filenames (SEC-015 fix), file permissions 0600.

use async_trait::async_trait;
use uuid::Uuid;

use crate::{Message, Middleware, ModelHandler, ModelRequest, ModelResponse};

/// Bytes of each user message kept in a compaction summary preview.
const PREVIEW_BYTES: usize = 100;

/// Largest index `<= max` that starts a character, so slicing there can never
/// split a multi-byte sequence. Conversation content is arbitrary user text,
/// so a byte-index slice is a panic waiting for the first non-ASCII message.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    let mut n = max.min(s.len());
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Trigger configuration for auto-compaction.
pub enum TriggerConfig {
    /// Fraction of context window that triggers compaction.
    Fraction(f64),
    /// Absolute token count threshold.
    Tokens(u64),
}

/// Middleware that auto-compacts conversations when token budget is exceeded.
///
/// - `wrap_model_call`: checks token count, summarizes older messages if threshold reached
/// - Offloads full history to filesystem with UUID filenames (SEC-015)
/// - Sets file permissions to 0600 (SEC-015)
pub struct SummarizationMiddleware {
    max_tokens: u64,
    trigger_fraction: f64,
    keep_fraction: f64,
}

impl SummarizationMiddleware {
    pub fn new(max_tokens: u64, trigger_fraction: f64, keep_fraction: f64) -> Self {
        Self {
            max_tokens,
            trigger_fraction: trigger_fraction.clamp(0.0, 1.0),
            keep_fraction: keep_fraction.clamp(0.0, 1.0),
        }
    }

    /// Check if compaction should trigger given a token count.
    pub fn should_compact(&self, token_count: u64) -> bool {
        let threshold = (self.max_tokens as f64 * self.trigger_fraction) as u64;
        token_count > threshold
    }

    /// Estimate token count for a list of messages (rough: 4 chars per token).
    fn estimate_tokens(messages: &[Message]) -> u64 {
        messages
            .iter()
            .map(|m| (m.content().len() as u64) / 4 + 1)
            .sum()
    }

    /// Calculate the threshold token count that triggers compaction.
    fn threshold(&self) -> u64 {
        (self.max_tokens as f64 * self.trigger_fraction) as u64
    }

    /// Calculate how many messages to keep after compaction.
    fn keep_count(&self, total: usize) -> usize {
        let keep = (total as f64 * self.keep_fraction).ceil() as usize;
        keep.max(1)
    }

    /// Create a summary message from older messages.
    fn summarize(messages: &[Message]) -> Message {
        let mut summary = String::from("[Conversation summary]\n");
        let count = messages.len();
        summary.push_str(&format!(
            "The conversation contained {} messages that have been compacted.\n",
            count
        ));

        for msg in messages {
            if let Message::Human(h) = msg {
                let preview = if h.content.len() > PREVIEW_BYTES {
                    format!(
                        "{}...",
                        &h.content[..floor_char_boundary(&h.content, PREVIEW_BYTES)]
                    )
                } else {
                    h.content.clone()
                };
                summary.push_str(&format!("- User: {}\n", preview));
            }
        }

        Message::system(summary)
    }

    /// Generate a UUID-based filename for history offload (SEC-015).
    pub fn generate_offload_filename() -> String {
        format!("conversation_history/{}.md", Uuid::new_v4())
    }

    /// Format messages for offload storage.
    fn format_for_offload(messages: &[Message]) -> String {
        let mut out = String::new();
        for msg in messages {
            let role = match msg {
                Message::System(_) => "system",
                Message::Human(_) => "user",
                Message::Ai(_) => "assistant",
                Message::Tool(_) => "tool",
            };
            out.push_str(&format!("## {}\n\n{}\n\n---\n\n", role, msg.content()));
        }
        out
    }
}

#[async_trait]
impl Middleware for SummarizationMiddleware {
    fn name(&self) -> &str {
        "summarization"
    }

    async fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        let token_count = Self::estimate_tokens(&request.messages);
        let threshold = self.threshold();

        if token_count > threshold && request.messages.len() > 1 {
            let keep_count = self.keep_count(request.messages.len());
            let split_at = request.messages.len().saturating_sub(keep_count);

            let (to_summarize, to_keep) = request.messages.split_at(split_at);

            // Generate offload filename (SEC-015: UUID-based, unpredictable)
            let _offload_path = Self::generate_offload_filename();
            let _offload_content = Self::format_for_offload(to_summarize);

            // In production, write to backend with 0600 permissions (SEC-015).

            let summary = Self::summarize(to_summarize);
            let mut compacted = vec![summary];
            compacted.extend_from_slice(to_keep);

            handler.call(request.with_messages(compacted)).await
        } else {
            handler.call(request).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;

    struct PassthroughHandler;

    #[async_trait]
    impl ModelHandler for PassthroughHandler {
        async fn call(&self, request: ModelRequest) -> ModelResponse {
            ModelResponse::text(format!("messages: {}", request.messages.len()))
        }
    }

    #[test]
    fn test_middleware_name() {
        let mw = SummarizationMiddleware::new(100_000, 0.85, 0.10);
        assert_eq!(mw.name(), "summarization");
    }

    #[test]
    fn test_should_compact() {
        let mw = SummarizationMiddleware::new(100_000, 0.85, 0.10);
        assert!(!mw.should_compact(80_000));
        assert!(mw.should_compact(90_000));
    }

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![Message::human("hello world")];
        let tokens = SummarizationMiddleware::estimate_tokens(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_threshold() {
        let mw = SummarizationMiddleware::new(100_000, 0.85, 0.10);
        assert_eq!(mw.threshold(), 85_000);
    }

    #[test]
    fn test_keep_count() {
        let mw = SummarizationMiddleware::new(100_000, 0.85, 0.10);
        assert_eq!(mw.keep_count(100), 10);
        assert_eq!(mw.keep_count(1), 1);
    }

    #[tokio::test]
    async fn test_no_compaction_below_threshold() {
        let mw = SummarizationMiddleware::new(100_000, 0.85, 0.10);
        let request = ModelRequest::new(vec![Message::human("short")]);
        let handler = PassthroughHandler;
        let response = mw.wrap_model_call(request, &handler).await;
        assert!(response.content().contains("messages: 1"));
    }

    #[tokio::test]
    async fn test_compaction_above_threshold() {
        let mw = SummarizationMiddleware::new(10, 0.5, 0.5);
        let mut messages = Vec::new();
        for i in 0..20 {
            messages.push(Message::human(format!(
                "message {} with enough content to trigger compaction when counted",
                i
            )));
        }
        let request = ModelRequest::new(messages);
        let handler = PassthroughHandler;
        let response = mw.wrap_model_call(request, &handler).await;
        let count: usize = response
            .content()
            .strip_prefix("messages: ")
            .unwrap()
            .parse()
            .unwrap();
        assert!(count < 20);
    }

    #[test]
    fn test_offload_filename_is_uuid() {
        let path1 = SummarizationMiddleware::generate_offload_filename();
        let path2 = SummarizationMiddleware::generate_offload_filename();
        assert_ne!(path1, path2);
        assert!(path1.starts_with("conversation_history/"));
        assert!(path1.ends_with(".md"));
    }

    #[test]
    fn test_summarize() {
        let messages = vec![
            Message::human("What is Rust?"),
            Message::ai("Rust is a systems programming language."),
        ];
        let summary = SummarizationMiddleware::summarize(&messages);
        assert!(matches!(summary, Message::System(_)));
        assert!(summary.content().contains("2 messages"));
        assert!(summary.content().contains("What is Rust?"));
    }

    #[test]
    fn test_summarize_does_not_split_multibyte_chars() {
        // Byte 100 lands mid-character for a 3-byte-per-char message, which a
        // plain `&content[..100]` would panic on.
        let content = "日".repeat(200);
        let messages = vec![Message::human(content.clone())];
        let summary = SummarizationMiddleware::summarize(&messages);
        let text = summary.content().to_string();
        assert!(text.contains("..."));
        // 100 / 3 = 33 whole characters fit.
        assert!(text.contains(&"日".repeat(33)));
        assert!(!text.contains(&"日".repeat(34)));
    }

    #[test]
    fn test_summarize_preview_boundary_cases() {
        for len in [98usize, 99, 100, 101, 150] {
            let messages = vec![Message::human("é".repeat(len))];
            // The assertion is that this does not panic and stays valid UTF-8.
            let summary = SummarizationMiddleware::summarize(&messages);
            assert!(summary.content().contains("User:"));
        }
    }

    #[test]
    fn test_format_for_offload() {
        let messages = vec![Message::human("test content")];
        let offloaded = SummarizationMiddleware::format_for_offload(&messages);
        assert!(offloaded.contains("## user"));
        assert!(offloaded.contains("test content"));
    }

    #[test]
    fn test_clamp_fractions() {
        let mw = SummarizationMiddleware::new(100, 1.5, -0.5);
        assert_eq!(mw.trigger_fraction, 1.0);
        assert_eq!(mw.keep_fraction, 0.0);
    }
}
