//! `PipelineModel` — a `rvagent_core::models::ChatModel` adapter that runs
//! every model call through a `MiddlewarePipeline` (P0.3 wiring).
//!
//! This lets `AgentGraph` stay unchanged while gaining the full middleware
//! stack: the graph calls `ChatModel::complete`, and this adapter routes the
//! call through `modify_request` and the async `wrap_model_call` chain before
//! delegating to the wrapped inner model.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use rvagent_core::error::{Result as CoreResult, RvAgentError};
use rvagent_core::models::{ChatModel, ToolDefinition};

use crate::{Message, MiddlewarePipeline, ModelHandler, ModelRequest, ModelResponse};

/// A `ChatModel` that wraps an inner model with a middleware pipeline.
///
/// Notes:
/// - A leading `Message::System` in the input is hoisted into the request's
///   `system_message` slot so middleware can append to it; the base handler
///   reassembles it before calling the inner model.
/// - Inner model errors surface to the chain as a `"error: …"` text response
///   (so e.g. `RetryMiddleware` can retry them); if the *final* inner call
///   still failed, the original error is propagated to the caller.
/// - Middleware-provided tools are NOT auto-advertised here; the tool set is
///   whatever the graph's `ToolExecutor` advertises. Use
///   `MiddlewarePipeline::collect_tools` to register middleware tools with an
///   executor if desired.
pub struct PipelineModel<M: ChatModel> {
    inner: M,
    pipeline: Arc<MiddlewarePipeline>,
}

impl<M: ChatModel> PipelineModel<M> {
    /// Wrap `inner` with the given middleware pipeline.
    pub fn new(inner: M, pipeline: Arc<MiddlewarePipeline>) -> Self {
        Self { inner, pipeline }
    }

    /// Access the pipeline.
    pub fn pipeline(&self) -> &Arc<MiddlewarePipeline> {
        &self.pipeline
    }

    /// Access the wrapped inner model.
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

/// Innermost handler: calls the real model.
struct BaseHandler<'a, M: ChatModel> {
    model: &'a M,
    /// Error from the most recent inner call (cleared at the start of each
    /// call, so a successful retry clears a previous failure).
    last_error: Mutex<Option<RvAgentError>>,
}

#[async_trait]
impl<M: ChatModel> ModelHandler for BaseHandler<'_, M> {
    async fn call(&self, request: ModelRequest) -> ModelResponse {
        *self.last_error.lock().unwrap() = None;

        let ModelRequest {
            system_message,
            mut messages,
            tools,
            ..
        } = request;
        if let Some(sys) = system_message {
            messages.insert(0, Message::system(sys));
        }

        match self.model.complete(&messages, &tools).await {
            Ok(message) => {
                let tool_calls = match &message {
                    Message::Ai(ai) => ai.tool_calls.clone(),
                    _ => Vec::new(),
                };
                ModelResponse {
                    message,
                    tool_calls,
                    usage: None,
                }
            }
            Err(e) => {
                // Feed the failure into the chain as an "error: …" response so
                // retry-style middleware can react; remember it for propagation.
                let resp = ModelResponse::text(format!("error: {e}"));
                *self.last_error.lock().unwrap() = Some(e);
                resp
            }
        }
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for PipelineModel<M> {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> CoreResult<Message> {
        // Hoist a leading system message into the request's system slot.
        let mut msgs = messages.to_vec();
        let system_message = match msgs.first() {
            Some(Message::System(sys)) => {
                let content = sys.content.clone();
                msgs.remove(0);
                Some(content)
            }
            _ => None,
        };

        let mut request = ModelRequest::new(msgs).with_system(system_message);
        request.tools = tools.to_vec();
        let request = self.pipeline.run_modify_request(request);

        let handler = BaseHandler {
            model: &self.inner,
            last_error: Mutex::new(None),
        };
        let response = self.pipeline.run_wrap_model_call(request, &handler).await;

        // If the final inner call failed, propagate the original error.
        if let Some(err) = handler.last_error.lock().unwrap().take() {
            return Err(err);
        }

        // Reattach the (possibly middleware-filtered) tool calls to the message.
        let ModelResponse {
            mut message,
            tool_calls,
            ..
        } = response;
        if let Message::Ai(ai) = &mut message {
            ai.tool_calls = tool_calls;
        }
        Ok(message)
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> CoreResult<Vec<Message>> {
        let msg = self.complete(messages, tools).await?;
        Ok(vec![msg])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{append_to_system_message, Middleware};
    use rvagent_core::messages::ToolCall;

    struct EchoModel;

    #[async_trait]
    impl ChatModel for EchoModel {
        async fn complete(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> CoreResult<Message> {
            // Echo the system message content (if any) for inspection.
            let sys = messages
                .iter()
                .find_map(|m| match m {
                    Message::System(s) => Some(s.content.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            Ok(Message::ai(format!("sys:[{sys}] n={}", messages.len())))
        }

        async fn stream(
            &self,
            messages: &[Message],
            tools: &[ToolDefinition],
        ) -> CoreResult<Vec<Message>> {
            Ok(vec![self.complete(messages, tools).await?])
        }
    }

    struct FailingModel;

    #[async_trait]
    impl ChatModel for FailingModel {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> CoreResult<Message> {
            Err(RvAgentError::model("boom"))
        }

        async fn stream(
            &self,
            messages: &[Message],
            tools: &[ToolDefinition],
        ) -> CoreResult<Vec<Message>> {
            Ok(vec![self.complete(messages, tools).await?])
        }
    }

    struct AppendMw(&'static str);

    #[async_trait]
    impl Middleware for AppendMw {
        fn name(&self) -> &str {
            "append"
        }
        async fn wrap_model_call(
            &self,
            request: ModelRequest,
            handler: &dyn ModelHandler,
        ) -> ModelResponse {
            let sys = append_to_system_message(&request.system_message, self.0);
            handler.call(request.with_system(sys)).await
        }
    }

    /// Middleware that drops all tool calls (HITL-style filtering).
    struct DropToolCalls;

    #[async_trait]
    impl Middleware for DropToolCalls {
        fn name(&self) -> &str {
            "drop_tool_calls"
        }
        async fn wrap_model_call(
            &self,
            request: ModelRequest,
            handler: &dyn ModelHandler,
        ) -> ModelResponse {
            let mut response = handler.call(request).await;
            response.tool_calls.clear();
            response
        }
    }

    struct ToolCallingModel;

    #[async_trait]
    impl ChatModel for ToolCallingModel {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
        ) -> CoreResult<Message> {
            Ok(Message::ai_with_tools(
                "calling",
                vec![ToolCall {
                    id: "tc1".into(),
                    name: "ls".into(),
                    args: serde_json::json!({}),
                }],
            ))
        }

        async fn stream(
            &self,
            messages: &[Message],
            tools: &[ToolDefinition],
        ) -> CoreResult<Vec<Message>> {
            Ok(vec![self.complete(messages, tools).await?])
        }
    }

    #[tokio::test]
    async fn test_pipeline_model_appends_system() {
        let pipeline = Arc::new(MiddlewarePipeline::new(vec![Box::new(AppendMw("EXTRA"))]));
        let model = PipelineModel::new(EchoModel, pipeline);

        let messages = vec![Message::system("base"), Message::human("hi")];
        let out = model.complete(&messages, &[]).await.unwrap();
        assert!(out.content().contains("base"));
        assert!(out.content().contains("EXTRA"));
    }

    #[tokio::test]
    async fn test_pipeline_model_propagates_errors() {
        let pipeline = Arc::new(MiddlewarePipeline::empty());
        let model = PipelineModel::new(FailingModel, pipeline);
        let err = model.complete(&[Message::human("hi")], &[]).await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_model_middleware_filters_tool_calls() {
        let pipeline = Arc::new(MiddlewarePipeline::new(vec![Box::new(DropToolCalls)]));
        let model = PipelineModel::new(ToolCallingModel, pipeline);
        let out = model.complete(&[Message::human("hi")], &[]).await.unwrap();
        assert!(!out.has_tool_calls(), "middleware must filter tool calls");
    }

    #[tokio::test]
    async fn test_pipeline_model_empty_pipeline_passthrough() {
        let pipeline = Arc::new(MiddlewarePipeline::empty());
        let model = PipelineModel::new(ToolCallingModel, pipeline);
        let out = model.complete(&[Message::human("hi")], &[]).await.unwrap();
        assert!(out.has_tool_calls());
    }
}
