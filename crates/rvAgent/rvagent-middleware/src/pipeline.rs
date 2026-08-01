//! Middleware pipeline execution (ADR-095) — fully async model-call chain (P0.3).

use std::future::Future;
use std::pin::Pin;

use crate::{
    AgentState, Middleware, ModelHandler, ModelRequest, ModelResponse, RunnableConfig, Runtime,
    Tool,
};

/// Executes the middleware pipeline in order.
/// Mirrors LangChain's `create_agent` middleware composition.
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    /// Create a new pipeline from an ordered list of middlewares.
    pub fn new(middlewares: Vec<Box<dyn Middleware>>) -> Self {
        Self { middlewares }
    }

    /// Create an empty pipeline.
    pub fn empty() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the end of the pipeline.
    pub fn push(&mut self, middleware: Box<dyn Middleware>) {
        self.middlewares.push(middleware);
    }

    /// Number of middlewares in the pipeline.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Get middleware names in order.
    pub fn names(&self) -> Vec<&str> {
        self.middlewares.iter().map(|mw| mw.name()).collect()
    }

    /// Run `before_agent` hooks in order, accumulating state updates.
    pub async fn run_before_agent(
        &self,
        state: &mut AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) {
        for mw in &self.middlewares {
            if let Some(update) = mw.before_agent(state, runtime, config).await {
                update.apply_to(state);
            }
        }
    }

    /// Collect all tools from all middlewares.
    pub fn collect_tools(&self) -> Vec<Box<dyn Tool>> {
        self.middlewares.iter().flat_map(|mw| mw.tools()).collect()
    }

    /// Run `modify_request` through all middlewares in order.
    pub fn run_modify_request(&self, mut request: ModelRequest) -> ModelRequest {
        for mw in &self.middlewares {
            request = mw.modify_request(request);
        }
        request
    }

    /// Run the async `wrap_model_call` chain.
    /// Middlewares are chained so the outermost (first) wraps the innermost (last).
    pub async fn run_wrap_model_call(
        &self,
        request: ModelRequest,
        base_handler: &dyn ModelHandler,
    ) -> ModelResponse {
        chain_call(&self.middlewares, request, base_handler).await
    }

    /// Full pipeline run: before_agent, collect tools, modify_request, wrap_model_call.
    pub async fn run(
        &self,
        state: &mut AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
        mut request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        // 1. Run before_agent hooks
        self.run_before_agent(state, runtime, config).await;

        // 2. Collect tools from all middlewares
        for tool in self.collect_tools() {
            request.tools.push(tool.definition());
        }

        // 3. Run modify_request
        request = self.run_modify_request(request);

        // 4. Run the async wrap_model_call chain
        self.run_wrap_model_call(request, handler).await
    }
}

type BoxResponseFuture<'a> = Pin<Box<dyn Future<Output = ModelResponse> + Send + 'a>>;

/// Recursively chain `wrap_model_call` futures from the outside in.
fn chain_call<'a>(
    middlewares: &'a [Box<dyn Middleware>],
    request: ModelRequest,
    handler: &'a dyn ModelHandler,
) -> BoxResponseFuture<'a> {
    Box::pin(async move {
        match middlewares.split_first() {
            None => handler.call(request).await,
            Some((first, rest)) => {
                let inner = ChainedHandler { rest, handler };
                first.wrap_model_call(request, &inner).await
            }
        }
    })
}

/// Handler that forwards to the remainder of the middleware chain.
struct ChainedHandler<'a> {
    rest: &'a [Box<dyn Middleware>],
    handler: &'a dyn ModelHandler,
}

#[async_trait::async_trait]
impl ModelHandler for ChainedHandler<'_> {
    async fn call(&self, request: ModelRequest) -> ModelResponse {
        chain_call(self.rest, request, self.handler).await
    }
}
