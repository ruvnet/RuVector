//! Integration tests for the message types in `rvagent_core::messages`.
//!
//! Covers Message enum variants, ToolCall serialization, AI messages with
//! tool calls, and message ordering semantics.

use rvagent_core::messages::{Message, ToolCall};

/// All four Message variants should be constructible and distinguishable.
#[test]
fn test_message_variants() {
    let sys = Message::system("You are helpful.");
    let human = Message::human("Hello");
    let ai = Message::ai("Hi there");
    let tool = Message::tool("call_1", "result data");

    // Content extraction works for every variant.
    assert_eq!(sys.content(), "You are helpful.");
    assert_eq!(human.content(), "Hello");
    assert_eq!(ai.content(), "Hi there");
    assert_eq!(tool.content(), "result data");

    // Only AI messages with tool calls report has_tool_calls = true.
    assert!(!sys.has_tool_calls());
    assert!(!human.has_tool_calls());
    assert!(!ai.has_tool_calls());
    assert!(!tool.has_tool_calls());

    // Pattern matching on enum variants.
    assert!(matches!(sys, Message::System(_)));
    assert!(matches!(human, Message::Human(_)));
    assert!(matches!(ai, Message::Ai(_)));
    assert!(matches!(tool, Message::Tool(_)));
}

/// ToolCall should serialize to JSON and deserialize back identically.
#[test]
fn test_tool_call_serialization() {
    let tc = ToolCall {
        id: "call_abc123".to_string(),
        name: "read_file".to_string(),
        args: serde_json::json!({
            "path": "/src/main.rs",
            "offset": 0,
            "limit": 100
        }),
    };

    let json = serde_json::to_string(&tc).unwrap();

    // JSON should contain all fields.
    assert!(json.contains("call_abc123"));
    assert!(json.contains("read_file"));
    assert!(json.contains("/src/main.rs"));

    let back: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(tc, back);

    // Args should preserve nested structure.
    assert_eq!(back.args["path"], "/src/main.rs");
    assert_eq!(back.args["offset"], 0);
}

/// AI messages with tool calls should correctly report has_tool_calls
/// and preserve tool call data through serialization.
#[test]
fn test_ai_message_with_tool_calls() {
    let tool_calls = vec![
        ToolCall {
            id: "tc_1".to_string(),
            name: "ls".to_string(),
            args: serde_json::json!({"path": "."}),
        },
        ToolCall {
            id: "tc_2".to_string(),
            name: "grep".to_string(),
            args: serde_json::json!({"pattern": "TODO", "path": "src/"}),
        },
    ];

    let msg = Message::ai_with_tools("Let me check the files.", tool_calls);

    assert!(msg.has_tool_calls());
    assert_eq!(msg.content(), "Let me check the files.");

    if let Message::Ai(ref ai) = msg {
        assert_eq!(ai.tool_calls.len(), 2);
        assert_eq!(ai.tool_calls[0].name, "ls");
        assert_eq!(ai.tool_calls[1].name, "grep");
    } else {
        panic!("expected Ai variant");
    }

    // Round-trip through JSON.
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(msg, restored);

    // AI message without tool calls should not report has_tool_calls.
    let no_tools = Message::ai("Just a plain response.");
    assert!(!no_tools.has_tool_calls());
}

/// Messages in a conversation should maintain their insertion order
/// and serialize/deserialize as a Vec preserving that order.
#[test]
fn test_message_ordering() {
    let conversation = vec![
        Message::system("You are an assistant."),
        Message::human("What is 2+2?"),
        Message::ai_with_tools(
            "Let me calculate.",
            vec![ToolCall {
                id: "calc_1".to_string(),
                name: "calculate".to_string(),
                args: serde_json::json!({"expr": "2+2"}),
            }],
        ),
        Message::tool("calc_1", "4"),
        Message::ai("The answer is 4."),
    ];

    assert_eq!(conversation.len(), 5);

    // Verify ordering by variant.
    assert!(matches!(conversation[0], Message::System(_)));
    assert!(matches!(conversation[1], Message::Human(_)));
    assert!(matches!(conversation[2], Message::Ai(_)));
    assert!(matches!(conversation[3], Message::Tool(_)));
    assert!(matches!(conversation[4], Message::Ai(_)));

    // The third message (index 2) should have tool calls.
    assert!(conversation[2].has_tool_calls());
    // The fifth message (index 4) should not.
    assert!(!conversation[4].has_tool_calls());

    // Round-trip the entire conversation.
    let json = serde_json::to_string(&conversation).unwrap();
    let restored: Vec<Message> = serde_json::from_str(&json).unwrap();
    assert_eq!(conversation, restored);

    // Verify content order is preserved.
    assert_eq!(restored[0].content(), "You are an assistant.");
    assert_eq!(restored[1].content(), "What is 2+2?");
    assert_eq!(restored[4].content(), "The answer is 4.");
}
