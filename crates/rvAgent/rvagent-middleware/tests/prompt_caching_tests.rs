//! Integration tests for the prompt caching middleware.

use rvagent_middleware::{
    Message, Middleware, ModelRequest, ToolDefinition,
};
use rvagent_middleware::prompt_caching::PromptCachingMiddleware;

// ---------------------------------------------------------------------------
// Tests: Construction
// ---------------------------------------------------------------------------

#[test]
fn test_middleware_name() {
    let mw = PromptCachingMiddleware::new();
    assert_eq!(mw.name(), "prompt_caching");
}

#[test]
fn test_default_cache_type_is_ephemeral() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_system(Some("system prompt".into()));

    let modified = mw.modify_request(request);
    assert_eq!(modified.cache_control["system"].cache_type, "ephemeral");
}

#[test]
fn test_custom_cache_type() {
    let mw = PromptCachingMiddleware::with_cache_type("persistent");
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_system(Some("system prompt".into()));

    let modified = mw.modify_request(request);
    assert_eq!(modified.cache_control["system"].cache_type, "persistent");
}

#[test]
fn test_default_trait_implementation() {
    let mw = PromptCachingMiddleware::default();
    assert_eq!(mw.name(), "prompt_caching");
}

// ---------------------------------------------------------------------------
// Tests: System message cache control
// ---------------------------------------------------------------------------

#[test]
fn test_adds_cache_control_for_system_message() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hello")])
        .with_system(Some("You are a helpful assistant.".into()));

    let modified = mw.modify_request(request);

    assert!(
        modified.cache_control.contains_key("system"),
        "should add cache control for system message"
    );
    assert_eq!(modified.cache_control["system"].cache_type, "ephemeral");
}

#[test]
fn test_no_cache_control_without_system_message() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hello")]);

    let modified = mw.modify_request(request);

    assert!(
        !modified.cache_control.contains_key("system"),
        "should not add system cache control when no system message"
    );
}

// ---------------------------------------------------------------------------
// Tests: Tools cache control
// ---------------------------------------------------------------------------

#[test]
fn test_adds_cache_control_for_tools() {
    let mw = PromptCachingMiddleware::new();
    let mut request = ModelRequest::new(vec![Message::user("hello")]);
    request.tools.push(ToolDefinition {
        name: "read_file".into(),
        description: "Read a file".into(),
        parameters: serde_json::json!({"type": "object"}),
    });

    let modified = mw.modify_request(request);

    assert!(
        modified.cache_control.contains_key("tools"),
        "should add cache control for tools"
    );
    assert_eq!(modified.cache_control["tools"].cache_type, "ephemeral");
}

#[test]
fn test_no_cache_control_without_tools() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![Message::user("hello")]);

    let modified = mw.modify_request(request);

    assert!(
        !modified.cache_control.contains_key("tools"),
        "should not add tools cache control when no tools defined"
    );
}

// ---------------------------------------------------------------------------
// Tests: Combined scenarios
// ---------------------------------------------------------------------------

#[test]
fn test_both_system_and_tools_get_cache_control() {
    let mw = PromptCachingMiddleware::new();
    let mut request = ModelRequest::new(vec![Message::user("hello")])
        .with_system(Some("system".into()));
    request.tools.push(ToolDefinition {
        name: "ls".into(),
        description: "List files".into(),
        parameters: serde_json::json!({}),
    });

    let modified = mw.modify_request(request);

    assert!(modified.cache_control.contains_key("system"));
    assert!(modified.cache_control.contains_key("tools"));
}

#[test]
fn test_neither_system_nor_tools_no_cache_control() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![]);

    let modified = mw.modify_request(request);

    assert!(
        modified.cache_control.is_empty(),
        "should have no cache control entries"
    );
}

#[test]
fn test_custom_cache_type_applies_to_both() {
    let mw = PromptCachingMiddleware::with_cache_type("long_lived");
    let mut request = ModelRequest::new(vec![Message::user("hi")])
        .with_system(Some("sys".into()));
    request.tools.push(ToolDefinition {
        name: "tool".into(),
        description: "desc".into(),
        parameters: serde_json::json!({}),
    });

    let modified = mw.modify_request(request);

    assert_eq!(modified.cache_control["system"].cache_type, "long_lived");
    assert_eq!(modified.cache_control["tools"].cache_type, "long_lived");
}

#[test]
fn test_messages_are_preserved_after_modify() {
    let mw = PromptCachingMiddleware::new();
    let request = ModelRequest::new(vec![
        Message::user("first"),
        Message::assistant("second"),
    ])
    .with_system(Some("sys".into()));

    let modified = mw.modify_request(request);

    assert_eq!(modified.messages.len(), 2);
    assert_eq!(modified.messages[0].content, "first");
    assert_eq!(modified.messages[1].content, "second");
    assert_eq!(modified.system_message, Some("sys".to_string()));
}

#[test]
fn test_multiple_tools_get_single_cache_entry() {
    let mw = PromptCachingMiddleware::new();
    let mut request = ModelRequest::new(vec![]);
    request.tools.push(ToolDefinition {
        name: "tool_a".into(),
        description: "a".into(),
        parameters: serde_json::json!({}),
    });
    request.tools.push(ToolDefinition {
        name: "tool_b".into(),
        description: "b".into(),
        parameters: serde_json::json!({}),
    });

    let modified = mw.modify_request(request);

    assert!(modified.cache_control.contains_key("tools"));
    // Only one "tools" cache entry, not per-tool
    assert_eq!(
        modified.cache_control.len(),
        1,
        "should have exactly one cache control entry for tools"
    );
}
