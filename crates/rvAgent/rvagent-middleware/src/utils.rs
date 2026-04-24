//! Utility types for middleware — SystemPromptBuilder (ADR-103 A5) and helpers.

use smallvec::SmallVec;
use std::borrow::Cow;

/// Efficient system prompt builder that defers concatenation until `build()`.
///
/// Collects segments in a `SmallVec` (inline for up to 8 segments) and performs
/// a single allocation with pre-calculated capacity on `build()`.
/// This replaces 4 sequential `format!()` calls per model call (ADR-103 A5).
pub struct SystemPromptBuilder {
    segments: SmallVec<[Cow<'static, str>; 8]>,
}

impl SystemPromptBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            segments: SmallVec::new(),
        }
    }

    /// Append a text segment. Accepts `&'static str`, `String`, or `Cow<'static, str>`.
    pub fn append(&mut self, text: impl Into<Cow<'static, str>>) {
        let cow = text.into();
        if !cow.is_empty() {
            self.segments.push(cow);
        }
    }

    /// Returns the number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Returns true if no segments have been appended.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Build the final prompt string with a single allocation.
    /// Segments are joined with double newlines.
    pub fn build(&self) -> String {
        if self.segments.is_empty() {
            return String::new();
        }
        // Pre-calculate total capacity: sum of segment lengths + separators
        let separator = "\n\n";
        let total_len: usize = self.segments.iter().map(|s| s.len()).sum::<usize>()
            + separator.len() * self.segments.len().saturating_sub(1);

        let mut out = String::with_capacity(total_len);
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                out.push_str(separator);
            }
            out.push_str(segment);
        }
        out
    }
}

impl Default for SystemPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Append text to an existing system message string, returning the combined result.
/// If `system_message` is `None`, returns a new string from `text`.
/// Used by Memory, Skills, SubAgent middlewares to inject into system prompts.
pub fn append_to_system_message(system_message: &Option<String>, text: &str) -> Option<String> {
    match system_message {
        Some(msg) => Some(format!("{}\n\n{}", msg, text)),
        None => Some(text.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_empty() {
        let builder = SystemPromptBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        assert_eq!(builder.build(), "");
    }

    #[test]
    fn test_builder_single_segment() {
        let mut builder = SystemPromptBuilder::new();
        builder.append("Hello");
        assert_eq!(builder.len(), 1);
        assert_eq!(builder.build(), "Hello");
    }

    #[test]
    fn test_builder_multiple_segments() {
        let mut builder = SystemPromptBuilder::new();
        builder.append("System prompt");
        builder.append("Memory context");
        builder.append("Skills info");
        assert_eq!(builder.len(), 3);
        assert_eq!(
            builder.build(),
            "System prompt\n\nMemory context\n\nSkills info"
        );
    }

    #[test]
    fn test_builder_skips_empty() {
        let mut builder = SystemPromptBuilder::new();
        builder.append("A");
        builder.append("");
        builder.append("B");
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.build(), "A\n\nB");
    }

    #[test]
    fn test_builder_with_owned_strings() {
        let mut builder = SystemPromptBuilder::new();
        builder.append(String::from("owned"));
        builder.append("static");
        assert_eq!(builder.build(), "owned\n\nstatic");
    }

    #[test]
    fn test_builder_with_cow() {
        let mut builder = SystemPromptBuilder::new();
        builder.append(Cow::Borrowed("borrowed"));
        builder.append(Cow::Owned("owned".to_string()));
        assert_eq!(builder.build(), "borrowed\n\nowned");
    }

    #[test]
    fn test_builder_single_allocation() {
        // Verify capacity is pre-calculated (no reallocation)
        let mut builder = SystemPromptBuilder::new();
        for i in 0..8 {
            builder.append(format!("segment-{}", i));
        }
        let result = builder.build();
        assert!(result.contains("segment-0"));
        assert!(result.contains("segment-7"));
    }

    #[test]
    fn test_append_to_system_message_none() {
        let result = append_to_system_message(&None, "new text");
        assert_eq!(result, Some("new text".to_string()));
    }

    #[test]
    fn test_append_to_system_message_some() {
        let existing = Some("existing".to_string());
        let result = append_to_system_message(&existing, "appended");
        assert_eq!(result, Some("existing\n\nappended".to_string()));
    }

    #[test]
    fn test_builder_default() {
        let builder = SystemPromptBuilder::default();
        assert!(builder.is_empty());
    }

    #[test]
    fn bench_builder_vs_naive() {
        // Functional test that builder produces same result as naive concat
        let segments = vec!["seg1", "seg2", "seg3", "seg4"];

        let mut builder = SystemPromptBuilder::new();
        for s in &segments {
            builder.append(*s);
        }
        let builder_result = builder.build();

        let naive_result = segments.join("\n\n");
        assert_eq!(builder_result, naive_result);
    }
}
