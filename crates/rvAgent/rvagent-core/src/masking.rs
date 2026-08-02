//! Observation masking — the default context strategy (ADR-274).
//!
//! Old tool observations are replaced with compact placeholders before the
//! history is sent to the model. Reasoning steps and actions are kept verbatim;
//! only observations are masked.
//!
//! This is deliberately *not* summarization. Measured comparisons put simple
//! masking at or above LLM summarization on solve rate at roughly half the
//! cost, and show that summarization inflates trajectories 13–15% by destroying
//! the natural stopping signals an agent uses to notice it has finished. A
//! placeholder that says `[read_file output elided: 2847 bytes, recall id tc7]`
//! preserves the *shape* of history without fabricating its contents.
//!
//! Masking is a projection, not a mutation: `AgentState::messages` remains the
//! complete append-only log, which is what makes the elided content
//! addressable — the `tool_call_id` in each placeholder is the recall handle.

use crate::messages::{Message, ToolMessage};

/// Configuration for the observation-masking projection.
#[derive(Debug, Clone)]
pub struct MaskConfig {
    /// How many of the most recent observations to keep in full.
    ///
    /// Set to `usize::MAX` to disable masking.
    pub keep_last_observations: usize,
    /// Hard cap on a single tool result's size, applied at write time.
    ///
    /// An uncapped tool result can consume the whole context window in one
    /// call. Truncation is explicit and marked so the model knows output was
    /// cut rather than silently ending.
    pub max_tool_result_bytes: usize,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            keep_last_observations: 8,
            // ~25k tokens at ~4 bytes/token, matching the industry default.
            max_tool_result_bytes: 100_000,
        }
    }
}

impl MaskConfig {
    /// Whether masking is active at all.
    pub fn masking_enabled(&self) -> bool {
        self.keep_last_observations != usize::MAX
    }
}

/// Truncate `content` to at most `max_bytes`, on a character boundary,
/// appending an explicit marker when anything was removed.
///
/// Operates on bytes rather than chars because the cap exists to bound memory
/// and context cost, but never splits a UTF-8 sequence.
pub fn truncate_tool_result(content: String, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content;
    }
    // Reserve room for the marker so the result still respects the budget.
    let marker = "\n... [output truncated]";

    // A cap smaller than the marker itself cannot carry the marker and stay
    // within budget. The cap wins: it is what bounds context cost, and
    // announcing the truncation is the part that can be given up.
    if max_bytes < marker.len() {
        return content[..floor_char_boundary(&content, max_bytes)].to_string();
    }

    let end = floor_char_boundary(&content, max_bytes - marker.len());
    let mut out = String::with_capacity(end + marker.len());
    out.push_str(&content[..end]);
    out.push_str(marker);
    out
}

/// Largest index `<= max` that starts a character, so slicing there never
/// splits a multi-byte sequence.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    let mut n = max.min(s.len());
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// The placeholder substituted for an elided observation.
fn placeholder(msg: &ToolMessage) -> String {
    let name = msg.tool_name.as_deref().unwrap_or("tool");
    format!(
        "[{} output elided: {} bytes, recall id {}]",
        name,
        msg.content.len(),
        msg.tool_call_id
    )
}

/// Project `messages` into the view sent to the model, masking all but the
/// most recent `keep_last_observations` tool results.
///
/// Non-tool messages pass through untouched — masking reasoning or actions
/// would destroy exactly the trail the model needs to stay coherent.
pub fn mask_observations(messages: &[Message], config: &MaskConfig) -> Vec<Message> {
    if !config.masking_enabled() {
        return messages.to_vec();
    }

    let total_observations = messages
        .iter()
        .filter(|m| matches!(m, Message::Tool(_)))
        .count();
    if total_observations <= config.keep_last_observations {
        return messages.to_vec();
    }
    let mask_before = total_observations - config.keep_last_observations;

    let mut seen = 0usize;
    messages
        .iter()
        .map(|msg| match msg {
            Message::Tool(tool) => {
                let index = seen;
                seen += 1;
                if index < mask_before {
                    Message::Tool(ToolMessage {
                        tool_call_id: tool.tool_call_id.clone(),
                        content: placeholder(tool),
                        tool_name: tool.tool_name.clone(),
                        metadata: tool.metadata.clone(),
                    })
                } else {
                    msg.clone()
                }
            }
            other => other.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Addressable recall (ADR-274 §3.2)
// ---------------------------------------------------------------------------

/// The reserved tool name used to dereference a masked observation.
///
/// Handled by the agent loop itself rather than a `ToolExecutor`: recall reads
/// the message log, which executors do not have, and reserving it in the loop
/// means a workspace tool cannot shadow it.
pub const RECALL_TOOL: &str = "recall";

/// Schema for the recall tool, advertised whenever masking is active.
pub fn recall_definition() -> crate::models::ToolDefinition {
    crate::models::ToolDefinition {
        name: RECALL_TOOL.to_string(),
        description:
            "Retrieve the full content of an earlier tool result that was elided from the \
             conversation. Pass the recall id shown in the placeholder, e.g. \
             '[read_file output elided: 2847 bytes, recall id tc7]' -> recall_id \"tc7\"."
                .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "recall_id": {
                    "type": "string",
                    "description": "The recall id from an elided observation placeholder."
                }
            },
            "required": ["recall_id"]
        }),
    }
}

/// Dereference a recall id against the full message log.
///
/// `messages` must be the complete log, never a masked projection — recalling
/// from a masked view would return the placeholder rather than the content.
pub fn recall(messages: &[Message], recall_id: &str) -> String {
    for msg in messages {
        if let Message::Tool(t) = msg {
            if t.tool_call_id == recall_id {
                return t.content.clone();
            }
        }
    }
    // Actionable rather than opaque: tell the model where valid ids come from.
    format!(
        "Error: no observation found with recall id '{recall_id}'. Recall ids appear \
         in elided-output placeholders in this conversation; they are not tool names \
         or file paths."
    )
}

/// Extract the `recall_id` argument from a recall tool call.
pub fn recall_id_from_args(args: &serde_json::Value) -> Result<&str, String> {
    args.get("recall_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            "Error: recall requires a string 'recall_id' argument, taken from an \
             elided-output placeholder."
                .to_string()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(id: &str, name: &str, content: &str) -> Message {
        Message::tool_with_name(id, content, name)
    }

    #[test]
    fn keeps_everything_when_under_the_limit() {
        let msgs = vec![
            Message::ai("thinking"),
            tool("t1", "read_file", "a"),
            tool("t2", "read_file", "b"),
        ];
        let out = mask_observations(&msgs, &MaskConfig::default());
        assert_eq!(out, msgs);
    }

    #[test]
    fn masks_all_but_the_last_n_observations() {
        let config = MaskConfig {
            keep_last_observations: 2,
            ..MaskConfig::default()
        };
        let msgs = vec![
            tool("t1", "read_file", "oldest"),
            tool("t2", "grep", "older"),
            tool("t3", "read_file", "recent"),
            tool("t4", "ls", "newest"),
        ];
        let out = mask_observations(&msgs, &config);

        let contents: Vec<&str> = out
            .iter()
            .map(|m| match m {
                Message::Tool(t) => t.content.as_str(),
                _ => unreachable!(),
            })
            .collect();

        assert!(contents[0].contains("read_file output elided"));
        assert!(contents[0].contains("recall id t1"));
        assert!(contents[1].contains("grep output elided"));
        // The most recent two survive verbatim.
        assert_eq!(contents[2], "recent");
        assert_eq!(contents[3], "newest");
    }

    #[test]
    fn never_masks_reasoning_or_actions() {
        let config = MaskConfig {
            keep_last_observations: 0,
            ..MaskConfig::default()
        };
        let msgs = vec![
            Message::system("rules"),
            Message::human("do the thing"),
            Message::ai("here is my plan"),
            tool("t1", "read_file", "contents"),
        ];
        let out = mask_observations(&msgs, &config);

        assert_eq!(out[0], msgs[0]);
        assert_eq!(out[1], msgs[1]);
        assert_eq!(out[2], msgs[2], "AI reasoning must survive masking");
        match &out[3] {
            Message::Tool(t) => assert!(t.content.contains("elided")),
            _ => panic!("expected a tool message"),
        }
    }

    #[test]
    fn masking_can_be_disabled() {
        let config = MaskConfig {
            keep_last_observations: usize::MAX,
            ..MaskConfig::default()
        };
        let msgs: Vec<Message> = (0..50)
            .map(|i| tool(&format!("t{i}"), "read_file", "body"))
            .collect();
        assert_eq!(mask_observations(&msgs, &config), msgs);
    }

    #[test]
    fn placeholder_preserves_the_recall_handle() {
        let config = MaskConfig {
            keep_last_observations: 0,
            ..MaskConfig::default()
        };
        let msgs = vec![tool("call-42", "grep", "many matches")];
        let out = mask_observations(&msgs, &config);
        match &out[0] {
            Message::Tool(t) => {
                // The id must survive so the full content stays addressable,
                // and must still pair with the model's tool_use block.
                assert_eq!(t.tool_call_id, "call-42");
                assert!(t.content.contains("recall id call-42"));
                assert!(t.content.contains("12 bytes"));
            }
            _ => panic!("expected a tool message"),
        }
    }

    #[test]
    fn truncation_marks_what_it_removed() {
        let out = truncate_tool_result("x".repeat(1000), 100);
        assert!(out.len() <= 100);
        assert!(out.ends_with("[output truncated]"));
    }

    #[test]
    fn truncation_leaves_short_content_alone() {
        let out = truncate_tool_result("short".into(), 100);
        assert_eq!(out, "short");
    }

    #[test]
    fn truncation_never_splits_a_multibyte_char() {
        // Every char is 4 bytes, so a naive byte cut would split one.
        let content = "🙂".repeat(100);
        for cap in [33usize, 50, 77, 99] {
            let out = truncate_tool_result(content.clone(), cap);
            // The result is valid UTF-8 by construction; the assertion is that
            // this did not panic and stayed inside the cap.
            assert!(out.ends_with("[output truncated]"), "cap {cap}");
            assert!(out.len() <= cap, "cap {cap} exceeded: {} bytes", out.len());
        }
    }

    #[test]
    fn truncation_respects_caps_too_small_for_the_marker() {
        // The cap bounds context cost, so it wins over announcing the cut.
        for cap in [0usize, 1, 5, 10, 22] {
            let out = truncate_tool_result("🙂".repeat(100), cap);
            assert!(out.len() <= cap, "cap {cap} exceeded: {} bytes", out.len());
        }
    }
}
