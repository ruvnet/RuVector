//! PatchToolCallsMiddleware — detects dangling tool calls from AI messages and
//! creates synthetic ToolMessage responses. Validates tool call IDs (ADR-103 C12).

use async_trait::async_trait;

use crate::{
    AgentState, AgentStateUpdate, Message, Middleware, Role, RunnableConfig, Runtime,
};

/// Maximum length for tool call IDs (ADR-103 C12).
pub const MAX_TOOL_CALL_ID_LENGTH: usize = 128;

/// Validate a tool call ID: max 128 chars, ASCII alphanumeric + hyphens + underscores only.
pub fn validate_tool_call_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("tool call ID is empty".into());
    }
    if id.len() > MAX_TOOL_CALL_ID_LENGTH {
        return Err(format!(
            "tool call ID exceeds {} characters (len={})",
            MAX_TOOL_CALL_ID_LENGTH,
            id.len()
        ));
    }
    for c in id.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            continue;
        }
        return Err(format!(
            "tool call ID contains invalid character '{}'",
            c
        ));
    }
    Ok(())
}

/// Middleware that patches dangling tool calls by creating synthetic tool responses.
pub struct PatchToolCallsMiddleware;

impl PatchToolCallsMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PatchToolCallsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str {
        "patch_tool_calls"
    }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if state.messages.is_empty() {
            return None;
        }

        let mut patched = Vec::with_capacity(state.messages.len());
        let mut modified = false;

        for (i, msg) in state.messages.iter().enumerate() {
            patched.push(msg.clone());

            if msg.role == Role::Assistant && !msg.tool_calls.is_empty() {
                for tc in &msg.tool_calls {
                    // Validate tool call ID (ADR-103 C12)
                    if let Err(err) = validate_tool_call_id(&tc.id) {
                        tracing::warn!("Invalid tool call ID '{}': {}", tc.id, err);
                        continue;
                    }

                    let has_response = state.messages[i + 1..].iter().any(|m| {
                        m.role == Role::Tool
                            && m.tool_call_id.as_deref() == Some(&*tc.id)
                    });

                    if !has_response {
                        patched.push(Message::tool(
                            format!(
                                "Tool call {} with id {} was cancelled — another message came in before it could be completed.",
                                tc.name, tc.id
                            ),
                            &tc.id,
                            &tc.name,
                        ));
                        modified = true;
                    }
                }
            }
        }

        if modified {
            let mut update = AgentStateUpdate::default();
            update.messages = Some(patched);
            Some(update)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolCall;

    #[test]
    fn test_middleware_name() {
        let mw = PatchToolCallsMiddleware::new();
        assert_eq!(mw.name(), "patch_tool_calls");
    }

    #[test]
    fn test_validate_tool_call_id_valid() {
        assert!(validate_tool_call_id("call-abc123").is_ok());
        assert!(validate_tool_call_id("toolu_01XYZ").is_ok());
        assert!(validate_tool_call_id("a").is_ok());
        assert!(validate_tool_call_id("abc-123_def").is_ok());
    }

    #[test]
    fn test_validate_tool_call_id_empty() {
        assert!(validate_tool_call_id("").is_err());
    }

    #[test]
    fn test_validate_tool_call_id_too_long() {
        let long_id = "a".repeat(129);
        assert!(validate_tool_call_id(&long_id).is_err());
    }

    #[test]
    fn test_validate_tool_call_id_max_length() {
        let max_id = "a".repeat(128);
        assert!(validate_tool_call_id(&max_id).is_ok());
    }

    #[test]
    fn test_validate_tool_call_id_invalid_chars() {
        assert!(validate_tool_call_id("call id").is_err());
        assert!(validate_tool_call_id("call.id").is_err());
        assert!(validate_tool_call_id("call@id").is_err());
        assert!(validate_tool_call_id("call/id").is_err());
    }

    #[test]
    fn test_no_patch_needed() {
        let mw = PatchToolCallsMiddleware::new();
        let state = AgentState {
            messages: vec![
                Message::user("hi"),
                Message::assistant("hello"),
            ],
            ..Default::default()
        };
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_patch_dangling_tool_call() {
        let mw = PatchToolCallsMiddleware::new();

        let mut assistant_msg = Message::assistant("I'll use a tool");
        assistant_msg.tool_calls.push(ToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            args: serde_json::json!({"path": "test.txt"}),
        });

        let state = AgentState {
            messages: vec![
                Message::user("help me"),
                assistant_msg,
                Message::user("never mind"),
            ],
            ..Default::default()
        };
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());

        let messages = update.unwrap().messages.unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].role, Role::Tool);
        assert!(messages[2].content.contains("cancelled"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn test_no_patch_when_response_exists() {
        let mw = PatchToolCallsMiddleware::new();

        let mut assistant_msg = Message::assistant("Using tool");
        assistant_msg.tool_calls.push(ToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            args: serde_json::json!({}),
        });

        let state = AgentState {
            messages: vec![
                assistant_msg,
                Message::tool("file content", "call-1", "read_file"),
            ],
            ..Default::default()
        };
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_patch_multiple_dangling() {
        let mw = PatchToolCallsMiddleware::new();

        let mut assistant_msg = Message::assistant("Using tools");
        assistant_msg.tool_calls.push(ToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            args: serde_json::json!({}),
        });
        assistant_msg.tool_calls.push(ToolCall {
            id: "call-2".into(),
            name: "write_file".into(),
            args: serde_json::json!({}),
        });

        let state = AgentState {
            messages: vec![assistant_msg],
            ..Default::default()
        };
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());

        let messages = update.unwrap().messages.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, Role::Tool);
        assert_eq!(messages[2].role, Role::Tool);
    }

    #[test]
    fn test_empty_messages() {
        let mw = PatchToolCallsMiddleware::new();
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }
}
