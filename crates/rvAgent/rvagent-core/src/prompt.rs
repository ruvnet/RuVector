//! System prompt constants and the `SystemPromptBuilder` (ADR-103 A5).
//!
//! The builder uses `SmallVec<[Cow<'static, str>; 8]>` to defer concatenation
//! until a single `build()` call, reducing O(n) copies from 4 to 1 per model call.

use smallvec::SmallVec;
use std::borrow::Cow;

/// The default base agent prompt used when no custom instructions are provided.
///
/// This is a comprehensive coding-assistant system prompt that establishes the
/// agent's identity, capabilities, behavioral guidelines, and output format.
pub const BASE_AGENT_PROMPT: &str = r#"You are rvAgent, a highly capable AI coding assistant powered by RuVector.

You have access to a set of tools that allow you to interact with the user's
codebase, filesystem, and development environment. Use these tools to accomplish
the tasks the user requests.

## Core Principles

1. **Accuracy** — Always produce correct, working code. Verify your changes
   compile and pass tests before reporting completion.
2. **Minimalism** — Do what was asked; nothing more, nothing less. Prefer the
   smallest change that solves the problem.
3. **Safety** — Never execute destructive operations without confirmation.
   Never expose secrets, credentials, or sensitive environment variables.
4. **Transparency** — Explain your reasoning when it aids understanding. Report
   errors honestly rather than guessing.

## Tool Usage

- Read files before editing them.
- Prefer editing existing files over creating new ones.
- Use grep/glob for searching; do not guess file locations.
- Run tests after making changes.
- Use absolute file paths.

## Output Format

- Keep responses concise and focused on the task.
- Include relevant file paths (absolute) in your response.
- Show code snippets only when the exact text is important.
- Do not create documentation files unless explicitly asked.

## Security

- Never hardcode API keys, secrets, or credentials.
- Never commit .env files or credential stores.
- Validate all user input at system boundaries.
- Sanitize file paths to prevent directory traversal.
- Strip sensitive environment variables before spawning child processes.

## Conversation Style

- Be direct and professional.
- Avoid unnecessary filler, emoji, or decoration.
- When uncertain, ask for clarification rather than making assumptions.
- Summarize what you did after completing multi-step tasks."#;

/// A builder for efficiently constructing system prompts from multiple segments.
///
/// Instead of concatenating strings 4+ times per model call (each O(n)), this
/// builder collects segments and concatenates once in `build()` with a single
/// pre-calculated allocation.
///
/// Per ADR-103 A5: uses `SmallVec<[Cow<'static, str>; 8]>` to avoid heap
/// allocation for typical prompt compositions (≤ 8 segments).
#[derive(Debug, Clone)]
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

    /// Create a builder initialized with the base agent prompt.
    pub fn with_base_prompt() -> Self {
        let mut b = Self::new();
        b.append(Cow::Borrowed(BASE_AGENT_PROMPT));
        b
    }

    /// Append a segment to the prompt.
    pub fn append(&mut self, text: impl Into<Cow<'static, str>>) {
        self.segments.push(text.into());
    }

    /// Append a segment with a leading blank line separator.
    pub fn append_section(&mut self, text: impl Into<Cow<'static, str>>) {
        self.segments.push(Cow::Borrowed("\n\n"));
        self.segments.push(text.into());
    }

    /// Number of segments currently held.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Build the final prompt string with a single allocation.
    pub fn build(&self) -> String {
        let total_len: usize = self.segments.iter().map(|s| s.len()).sum();
        let mut out = String::with_capacity(total_len);
        for seg in &self.segments {
            out.push_str(seg);
        }
        out
    }
}

impl Default for SystemPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_prompt_not_empty() {
        assert!(!BASE_AGENT_PROMPT.is_empty());
        assert!(BASE_AGENT_PROMPT.contains("rvAgent"));
        // Should be a substantial prompt (~50+ lines).
        let line_count = BASE_AGENT_PROMPT.lines().count();
        assert!(line_count >= 40, "base prompt has {} lines", line_count);
    }

    #[test]
    fn test_builder_empty() {
        let b = SystemPromptBuilder::new();
        assert_eq!(b.build(), "");
        assert_eq!(b.segment_count(), 0);
    }

    #[test]
    fn test_builder_with_base() {
        let b = SystemPromptBuilder::with_base_prompt();
        let result = b.build();
        assert_eq!(result, BASE_AGENT_PROMPT);
    }

    #[test]
    fn test_builder_append_sections() {
        let mut b = SystemPromptBuilder::with_base_prompt();
        b.append_section("## Memory\nYou have access to memory.");
        b.append_section("## Skills\nAvailable skills: foo, bar.");
        let result = b.build();
        assert!(result.starts_with(BASE_AGENT_PROMPT));
        assert!(result.contains("## Memory"));
        assert!(result.contains("## Skills"));
    }

    #[test]
    fn test_builder_single_allocation() {
        let mut b = SystemPromptBuilder::new();
        b.append("a");
        b.append("b");
        b.append("c");
        let result = b.build();
        assert_eq!(result, "abc");
        // Capacity should be exactly 3 (pre-calculated).
        assert_eq!(result.capacity(), 3);
    }

    #[test]
    fn test_builder_cow_borrowed_vs_owned() {
        let mut b = SystemPromptBuilder::new();
        // Borrowed (static)
        b.append(Cow::Borrowed("static segment"));
        // Owned (dynamic)
        let dynamic = format!("dynamic {}", 42);
        b.append(Cow::Owned(dynamic));
        let result = b.build();
        assert_eq!(result, "static segmentdynamic 42");
    }

    #[test]
    fn test_builder_default() {
        let b = SystemPromptBuilder::default();
        assert_eq!(b.segment_count(), 0);
    }
}
