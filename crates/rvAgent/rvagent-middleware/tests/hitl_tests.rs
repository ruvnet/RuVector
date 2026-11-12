//! Integration tests for the Human-in-the-Loop (HITL) middleware.

use async_trait::async_trait;
use rvagent_middleware::hitl::{ApprovalDecision, HumanInTheLoopMiddleware};
use rvagent_middleware::{
    Message, Middleware, ModelHandler, ModelRequest, ModelResponse, ToolCall,
};

// ---------------------------------------------------------------------------
// Test handler
// ---------------------------------------------------------------------------

/// Handler that returns a response with configurable tool calls.
struct ToolCallHandler {
    tool_calls: Vec<ToolCall>,
}

impl ToolCallHandler {
    fn new(tool_calls: Vec<ToolCall>) -> Self {
        Self { tool_calls }
    }

    fn with_names(names: &[&str]) -> Self {
        let calls = names
            .iter()
            .enumerate()
            .map(|(i, name)| ToolCall {
                id: format!("call-{}", i),
                name: name.to_string(),
                args: serde_json::json!({}),
            })
            .collect();
        Self::new(calls)
    }
}

#[async_trait]
impl ModelHandler for ToolCallHandler {
    async fn call(&self, _request: ModelRequest) -> ModelResponse {
        let mut response = ModelResponse::text("model response");
        response.tool_calls = self.tool_calls.clone();
        response
    }
}

// ---------------------------------------------------------------------------
// Tests: Construction
// ---------------------------------------------------------------------------

#[test]
fn test_middleware_name() {
    let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
    assert_eq!(mw.name(), "hitl");
}

// ---------------------------------------------------------------------------
// Tests: should_interrupt — exact match
// ---------------------------------------------------------------------------

#[test]
fn test_exact_match_interrupts() {
    let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
    assert!(mw.should_interrupt("execute"));
}

#[test]
fn test_exact_match_does_not_interrupt_other() {
    let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
    assert!(!mw.should_interrupt("read_file"));
    assert!(!mw.should_interrupt("ls"));
}

// ---------------------------------------------------------------------------
// Tests: should_interrupt — wildcard (*)
// ---------------------------------------------------------------------------

#[test]
fn test_wildcard_star_interrupts_everything() {
    let mw = HumanInTheLoopMiddleware::new(vec!["*".into()]);
    assert!(mw.should_interrupt("execute"));
    assert!(mw.should_interrupt("read_file"));
    assert!(mw.should_interrupt("glob"));
    assert!(mw.should_interrupt("any_tool_name"));
}

// ---------------------------------------------------------------------------
// Tests: should_interrupt — prefix wildcard
// ---------------------------------------------------------------------------

#[test]
fn test_prefix_wildcard_matches_prefix() {
    let mw = HumanInTheLoopMiddleware::new(vec!["write_*".into()]);
    assert!(mw.should_interrupt("write_file"));
    assert!(mw.should_interrupt("write_todos"));
    assert!(mw.should_interrupt("write_anything_else"));
}

#[test]
fn test_prefix_wildcard_does_not_match_other() {
    let mw = HumanInTheLoopMiddleware::new(vec!["write_*".into()]);
    assert!(!mw.should_interrupt("read_file"));
    assert!(!mw.should_interrupt("execute"));
}

// ---------------------------------------------------------------------------
// Tests: should_interrupt — multiple patterns
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_patterns() {
    let mw =
        HumanInTheLoopMiddleware::new(vec!["execute".into(), "write_*".into(), "delete".into()]);
    assert!(mw.should_interrupt("execute"));
    assert!(mw.should_interrupt("write_file"));
    assert!(mw.should_interrupt("write_todos"));
    assert!(mw.should_interrupt("delete"));
    assert!(!mw.should_interrupt("read_file"));
    assert!(!mw.should_interrupt("ls"));
}

// ---------------------------------------------------------------------------
// Tests: should_interrupt — empty patterns
// ---------------------------------------------------------------------------

#[test]
fn test_empty_patterns_interrupts_nothing() {
    let mw = HumanInTheLoopMiddleware::new(vec![]);
    assert!(!mw.should_interrupt("execute"));
    assert!(!mw.should_interrupt("anything"));
}

// ---------------------------------------------------------------------------
// Tests: wrap_model_call
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_wrap_filters_matching_tool_calls() {
    let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);
    let handler = ToolCallHandler::with_names(&["execute", "read_file"]);
    let request = ModelRequest::new(vec![Message::human("do something")]);

    let response = mw.wrap_model_call(request, &handler).await;

    // Only read_file should remain
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "read_file");

    // HITL message should be appended
    assert!(
        response.content().contains("[HITL]"),
        "should contain HITL marker"
    );
    assert!(
        response.content().contains("execute"),
        "should mention the interrupted tool"
    );
}

#[tokio::test]
async fn test_wrap_no_matching_tools_passes_all_through() {
    let mw = HumanInTheLoopMiddleware::new(vec!["dangerous_tool".into()]);
    let handler = ToolCallHandler::with_names(&["read_file", "ls", "glob"]);
    let request = ModelRequest::new(vec![Message::human("safe operation")]);

    let response = mw.wrap_model_call(request, &handler).await;

    assert_eq!(response.tool_calls.len(), 3);
    assert!(
        !response.content().contains("[HITL]"),
        "should not contain HITL marker when nothing is interrupted"
    );
}

#[tokio::test]
async fn test_wrap_all_tools_interrupted() {
    let mw = HumanInTheLoopMiddleware::new(vec!["*".into()]);
    let handler = ToolCallHandler::with_names(&["execute", "write_file"]);
    let request = ModelRequest::new(vec![Message::human("do things")]);

    let response = mw.wrap_model_call(request, &handler).await;

    assert!(
        response.tool_calls.is_empty(),
        "all tool calls should be intercepted"
    );
    assert!(response.content().contains("[HITL]"));
    assert!(response.content().contains("execute"));
    assert!(response.content().contains("write_file"));
}

#[tokio::test]
async fn test_wrap_no_tool_calls_from_handler() {
    let mw = HumanInTheLoopMiddleware::new(vec!["execute".into()]);

    struct NoToolHandler;

    #[async_trait]
    impl ModelHandler for NoToolHandler {
        async fn call(&self, _request: ModelRequest) -> ModelResponse {
            ModelResponse::text("just text, no tools")
        }
    }

    let request = ModelRequest::new(vec![Message::human("question")]);
    let response = mw.wrap_model_call(request, &NoToolHandler).await;

    assert!(response.tool_calls.is_empty());
    assert!(
        !response.content().contains("[HITL]"),
        "should not add HITL marker when no tool calls"
    );
}

#[tokio::test]
async fn test_wrap_preserves_original_response_content() {
    let mw = HumanInTheLoopMiddleware::new(vec!["dangerous".into()]);
    let handler = ToolCallHandler::with_names(&["read_file"]);
    let request = ModelRequest::new(vec![Message::human("hi")]);

    let response = mw.wrap_model_call(request, &handler).await;

    assert!(
        response.content().contains("model response"),
        "should preserve original model response content"
    );
}

#[tokio::test]
async fn test_wrap_prefix_pattern_filters_correctly() {
    let mw = HumanInTheLoopMiddleware::new(vec!["write_*".into()]);
    let handler =
        ToolCallHandler::with_names(&["write_file", "write_todos", "read_file", "execute"]);
    let request = ModelRequest::new(vec![Message::human("do writes")]);

    let response = mw.wrap_model_call(request, &handler).await;

    assert_eq!(
        response.tool_calls.len(),
        2,
        "read_file and execute should pass through"
    );
    let names: Vec<&str> = response
        .tool_calls
        .iter()
        .map(|tc| tc.name.as_str())
        .collect();
    assert!(names.contains(&"read_file"));
    assert!(names.contains(&"execute"));
    assert!(!names.contains(&"write_file"));
    assert!(!names.contains(&"write_todos"));
}

// ---------------------------------------------------------------------------
// Tests: ApprovalDecision enum
// ---------------------------------------------------------------------------

#[test]
fn test_approval_decision_variants() {
    let approve = ApprovalDecision::Approve;
    let deny = ApprovalDecision::Deny;
    let modify = ApprovalDecision::ApproveWithModification("changed args".into());

    assert_eq!(approve, ApprovalDecision::Approve);
    assert_eq!(deny, ApprovalDecision::Deny);
    assert_ne!(approve, deny);

    match modify {
        ApprovalDecision::ApproveWithModification(s) => {
            assert_eq!(s, "changed args");
        }
        _ => panic!("expected ApproveWithModification"),
    }
}
