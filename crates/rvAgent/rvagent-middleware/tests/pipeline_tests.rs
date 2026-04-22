//! Integration tests for the middleware pipeline — ordering, before_agent chain,
//! wrap_model_call chain, and tool injection.

use async_trait::async_trait;
use rvagent_middleware::{
    append_to_system_message, AgentState, AgentStateUpdate, Message, Middleware,
    MiddlewarePipeline, ModelHandler, ModelRequest, ModelResponse, Role, RunnableConfig, Runtime,
    Tool, ToolDefinition,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A middleware that records its name when before_agent is called.
struct RecordingMiddleware {
    label: String,
    extension_key: String,
}

impl RecordingMiddleware {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            extension_key: format!("visited_{}", label),
        }
    }
}

#[async_trait]
impl Middleware for RecordingMiddleware {
    fn name(&self) -> &str {
        &self.label
    }

    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        let mut update = AgentStateUpdate::default();
        update
            .extensions
            .insert(self.extension_key.clone(), serde_json::json!(true));
        Some(update)
    }
}

/// A middleware that appends text to the system message.
struct SystemAppender {
    label: String,
    text: String,
}

impl SystemAppender {
    fn new(label: &str, text: &str) -> Self {
        Self {
            label: label.to_string(),
            text: text.to_string(),
        }
    }
}

#[async_trait]
impl Middleware for SystemAppender {
    fn name(&self) -> &str {
        &self.label
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        let new_sys = append_to_system_message(&request.system_message, &self.text);
        handler.call(request.with_system(new_sys))
    }
}

/// A middleware that injects a tool.
struct ToolInjectorMw {
    label: String,
    tool_name: String,
}

impl ToolInjectorMw {
    fn new(label: &str, tool_name: &str) -> Self {
        Self {
            label: label.to_string(),
            tool_name: tool_name.to_string(),
        }
    }
}

struct NamedTool {
    name: String,
}

impl Tool for NamedTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "test tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
        Ok("ok".into())
    }
}

#[async_trait]
impl Middleware for ToolInjectorMw {
    fn name(&self) -> &str {
        &self.label
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(NamedTool {
            name: self.tool_name.clone(),
        })]
    }
}

/// Handler that captures the final system message.
struct CaptureSystemHandler;

impl ModelHandler for CaptureSystemHandler {
    fn call(&self, request: ModelRequest) -> ModelResponse {
        ModelResponse::text(request.system_message.unwrap_or_default())
    }
}

/// Handler that returns the number of tool definitions.
struct CountToolsHandler;

impl ModelHandler for CountToolsHandler {
    fn call(&self, request: ModelRequest) -> ModelResponse {
        ModelResponse::text(format!("tools:{}", request.tools.len()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_ordering() {
    let mut pipeline = MiddlewarePipeline::empty();
    pipeline.push(Box::new(RecordingMiddleware::new("alpha")));
    pipeline.push(Box::new(RecordingMiddleware::new("beta")));
    pipeline.push(Box::new(RecordingMiddleware::new("gamma")));

    let names = pipeline.names();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    assert_eq!(pipeline.len(), 3);
    assert!(!pipeline.is_empty());
}

#[tokio::test]
async fn test_pipeline_before_agent_chain() {
    let pipeline = MiddlewarePipeline::new(vec![
        Box::new(RecordingMiddleware::new("first")),
        Box::new(RecordingMiddleware::new("second")),
        Box::new(RecordingMiddleware::new("third")),
    ]);

    let mut state = AgentState::default();
    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    pipeline
        .run_before_agent(&mut state, &runtime, &config)
        .await;

    // All three middlewares should have set their extension key.
    assert_eq!(
        state.extensions.get("visited_first"),
        Some(&serde_json::json!(true)),
        "first middleware should have run"
    );
    assert_eq!(
        state.extensions.get("visited_second"),
        Some(&serde_json::json!(true)),
        "second middleware should have run"
    );
    assert_eq!(
        state.extensions.get("visited_third"),
        Some(&serde_json::json!(true)),
        "third middleware should have run"
    );
}

#[test]
fn test_pipeline_wrap_model_call_chain() {
    // Two appenders: "A" then "B". Both should appear in the final system message.
    let pipeline = MiddlewarePipeline::new(vec![
        Box::new(SystemAppender::new("appender_a", "<<A>>")),
        Box::new(SystemAppender::new("appender_b", "<<B>>")),
    ]);

    let request = ModelRequest::new(vec![Message::user("hi")]).with_system(Some("base".into()));

    let response = pipeline.run_wrap_model_call(request, &CaptureSystemHandler);

    let sys = response.message.content;
    assert!(sys.contains("base"), "should preserve base system message");
    assert!(sys.contains("<<A>>"), "should include appender A");
    assert!(sys.contains("<<B>>"), "should include appender B");
}

#[test]
fn test_pipeline_tool_injection() {
    let pipeline = MiddlewarePipeline::new(vec![
        Box::new(ToolInjectorMw::new("injector_1", "tool_alpha")),
        Box::new(ToolInjectorMw::new("injector_2", "tool_beta")),
    ]);

    let tools = pipeline.collect_tools();
    assert_eq!(tools.len(), 2);

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"tool_alpha"));
    assert!(names.contains(&"tool_beta"));
}
