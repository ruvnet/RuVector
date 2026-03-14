//! Message types for agent communication.
//!
//! Maps the Python `langchain_core.messages` hierarchy to Rust enums.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tool invocation requested by the AI model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned tool call identifier.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments as a JSON value (typically an object).
    pub args: serde_json::Value,
}

/// Content from the system / instructions layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMessage {
    pub content: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A message from the human user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanMessage {
    pub content: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A response from the AI model, possibly containing tool calls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AiMessage {
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// The result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolMessage {
    /// The id of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The tool's output content.
    pub content: String,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Unified message enum used throughout the agent pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    System(SystemMessage),
    Human(HumanMessage),
    Ai(AiMessage),
    Tool(ToolMessage),
}

impl Message {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemMessage {
            content: content.into(),
            metadata: HashMap::new(),
        })
    }

    /// Create a human message.
    pub fn human(content: impl Into<String>) -> Self {
        Self::Human(HumanMessage {
            content: content.into(),
            metadata: HashMap::new(),
        })
    }

    /// Create an AI message without tool calls.
    pub fn ai(content: impl Into<String>) -> Self {
        Self::Ai(AiMessage {
            content: content.into(),
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    /// Create an AI message with tool calls.
    pub fn ai_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Ai(AiMessage {
            content: content.into(),
            tool_calls,
            metadata: HashMap::new(),
        })
    }

    /// Create a tool result message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool(ToolMessage {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        })
    }

    /// Get the text content of any message variant.
    #[inline]
    pub fn content(&self) -> &str {
        match self {
            Self::System(m) => &m.content,
            Self::Human(m) => &m.content,
            Self::Ai(m) => &m.content,
            Self::Tool(m) => &m.content,
        }
    }

    /// Returns true if this is an AI message with pending tool calls.
    #[inline]
    pub fn has_tool_calls(&self) -> bool {
        matches!(self, Self::Ai(m) if !m.tool_calls.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("you are helpful");
        assert_eq!(sys.content(), "you are helpful");
        assert!(!sys.has_tool_calls());

        let human = Message::human("hello");
        assert_eq!(human.content(), "hello");

        let ai = Message::ai("sure");
        assert_eq!(ai.content(), "sure");
        assert!(!ai.has_tool_calls());

        let tool = Message::tool("tc_1", "result");
        assert_eq!(tool.content(), "result");
    }

    #[test]
    fn test_ai_with_tool_calls() {
        let tc = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            args: serde_json::json!({"path": "/tmp/f.txt"}),
        };
        let msg = Message::ai_with_tools("", vec![tc.clone()]);
        assert!(msg.has_tool_calls());
        if let Message::Ai(ai) = &msg {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "read_file");
        } else {
            panic!("expected Ai variant");
        }
    }

    #[test]
    fn test_message_serialization_roundtrip() {
        let messages = vec![
            Message::system("sys"),
            Message::human("hi"),
            Message::ai("hello"),
            Message::tool("id1", "output"),
            Message::ai_with_tools(
                "let me check",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "ls".into(),
                    args: serde_json::json!({"dir": "."}),
                }],
            ),
        ];
        let json = serde_json::to_string(&messages).unwrap();
        let roundtrip: Vec<Message> = serde_json::from_str(&json).unwrap();
        assert_eq!(messages, roundtrip);
    }

    #[test]
    fn test_tool_call_serde() {
        let tc = ToolCall {
            id: "abc".into(),
            name: "grep".into(),
            args: serde_json::json!({"pattern": "foo", "path": "."}),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let back: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(tc, back);
    }

    #[test]
    fn test_message_metadata() {
        let msg = Message::System(SystemMessage {
            content: "test".into(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("cache_control".into(), serde_json::json!("ephemeral"));
                m
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("cache_control"));
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }
}
