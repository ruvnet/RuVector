# ADR-095: Middleware Pipeline Architecture

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crate**   | `ruvector-deep-middleware`                       |

## Context

DeepAgents uses LangChain's `AgentMiddleware[StateT, ContextT, ResponseT]` generic class with these hooks:

1. `before_agent(state, runtime, config) -> StateUpdate | None` — Pre-execution state injection
2. `wrap_model_call(request, handler) -> response` — Model call interception
3. `awrap_model_call(request, handler) -> response` — Async model call interception
4. `modify_request(request) -> request` — Request transformation
5. `tools: list[BaseTool]` — Additional tools injected by middleware
6. `state_schema` — State schema extension (via class attribute)

The default middleware stack order in `create_deep_agent()`:

```
1. TodoListMiddleware
2. MemoryMiddleware (if memory configured)
3. SkillsMiddleware (if skills configured)
4. FilesystemMiddleware
5. SubAgentMiddleware
6. SummarizationMiddleware
7. AnthropicPromptCachingMiddleware
8. PatchToolCallsMiddleware
9. [User middleware...]
10. HumanInTheLoopMiddleware (if interrupt_on configured)
```

## Decision

### Core Middleware Trait

```rust
// crates/ruvector-deep-middleware/src/lib.rs

use async_trait::async_trait;

/// Agent state — extensible via middleware state schemas.
pub type AgentState = HashMap<String, serde_json::Value>;

/// Model request wrapping messages and state.
pub struct ModelRequest<C> {
    pub system_message: Option<SystemMessage>,
    pub messages: Vec<Message>,
    pub state: AgentState,
    pub context: C,
    pub tools: Vec<Box<dyn Tool>>,
}

impl<C> ModelRequest<C> {
    pub fn override_system(&self, system_message: Option<SystemMessage>) -> Self { ... }
}

/// Model response from LLM call.
pub struct ModelResponse<R> {
    pub message: Message,
    pub response: R,
}

/// Runtime context passed to middleware hooks.
pub struct Runtime {
    pub context: serde_json::Value,
    pub stream_writer: Option<Box<dyn StreamWriter>>,
    pub store: Option<Box<dyn Store>>,
    pub config: RunnableConfig,
}

/// Core middleware trait — mirrors Python's AgentMiddleware exactly.
/// Generic over State (S), Context (C), and Response (R).
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before agent execution. Returns state update or None.
    /// Python: before_agent(state, runtime, config)
    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentState> {
        None
    }

    /// Async version of before_agent.
    async fn abefore_agent(
        &self,
        state: &AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) -> Option<AgentState> {
        self.before_agent(state, runtime, config)
    }

    /// Wrap a model call — intercept request/response.
    /// Python: wrap_model_call(request, handler)
    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        handler(request)
    }

    /// Async wrap model call.
    async fn awrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> BoxFuture<ModelResponse<()>>,
    ) -> ModelResponse<()> {
        handler(request).await
    }

    /// Transform request before model call.
    /// Python: modify_request(request)
    fn modify_request(&self, request: ModelRequest<()>) -> ModelRequest<()> {
        request
    }

    /// Additional tools provided by this middleware.
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![]
    }

    /// State schema extensions (keys this middleware manages).
    fn state_keys(&self) -> Vec<&str> {
        vec![]
    }
}
```

### Middleware Pipeline Executor

```rust
/// Executes the middleware pipeline in order.
/// Mirrors LangChain's create_agent middleware composition.
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    pub fn new(middlewares: Vec<Box<dyn Middleware>>) -> Self {
        Self { middlewares }
    }

    /// Run before_agent hooks in order, accumulating state updates.
    pub async fn run_before_agent(
        &self,
        state: &mut AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) {
        for mw in &self.middlewares {
            if let Some(update) = mw.abefore_agent(state, runtime, config).await {
                for (k, v) in update {
                    state.insert(k, v);
                }
            }
        }
    }

    /// Collect all tools from all middlewares.
    pub fn collect_tools(&self) -> Vec<Box<dyn Tool>> {
        self.middlewares.iter().flat_map(|mw| mw.tools()).collect()
    }

    /// Chain wrap_model_call handlers (innermost first, outermost wraps).
    pub async fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        base_handler: impl Fn(ModelRequest<()>) -> BoxFuture<ModelResponse<()>>,
    ) -> ModelResponse<()> {
        // Build handler chain from inside out
        let mut handler: Box<dyn Fn(ModelRequest<()>) -> BoxFuture<ModelResponse<()>>> =
            Box::new(base_handler);

        for mw in self.middlewares.iter().rev() {
            let prev = handler;
            handler = Box::new(move |req| {
                Box::pin(mw.awrap_model_call(req, &*prev))
            });
        }

        handler(request).await
    }
}
```

### Concrete Middleware Implementations

Each Python middleware maps 1:1:

| Python Middleware | Rust Struct | Purpose |
|---|---|---|
| `TodoListMiddleware` | `TodoListMiddleware` | `write_todos` tool + state |
| `FilesystemMiddleware` | `FilesystemMiddleware` | File operation tools (ls, read, write, edit, glob, grep, execute) |
| `SubAgentMiddleware` | `SubAgentMiddleware` | `task` tool for subagent spawning |
| `SummarizationMiddleware` | `SummarizationMiddleware` | Auto-compact + `compact_conversation` tool |
| `MemoryMiddleware` | `MemoryMiddleware` | AGENTS.md loading into system prompt |
| `SkillsMiddleware` | `SkillsMiddleware` | SKILL.md progressive disclosure |
| `PatchToolCallsMiddleware` | `PatchToolCallsMiddleware` | Dangling tool call repair |
| `AnthropicPromptCachingMiddleware` | `PromptCachingMiddleware` | Cache control block injection |
| `HumanInTheLoopMiddleware` | `HumanInTheLoopMiddleware` | Interrupt on specific tools |

### System Message Composition

```rust
/// Python: append_to_system_message(system_message, text)
/// Used by Memory, Skills, SubAgent middlewares to inject into system prompt.
pub fn append_to_system_message(
    system_message: &Option<SystemMessage>,
    text: &str,
) -> Option<SystemMessage> {
    match system_message {
        Some(msg) => Some(SystemMessage {
            content: format!("{}\n\n{}", msg.content, text),
        }),
        None => Some(SystemMessage {
            content: text.to_string(),
        }),
    }
}
```

### State Schema Extension

Python uses class-level `state_schema` and `PrivateStateAttr` annotations. In Rust:

```rust
/// Private state attributes (not propagated to parent agents).
/// Python: Annotated[T, PrivateStateAttr]
pub struct PrivateState<T> {
    inner: T,
    private: bool, // Always true — excluded from serialization to parent
}

/// Memory middleware state extension
/// Python: MemoryState(AgentState) with memory_contents: PrivateStateAttr
pub struct MemoryStateExt {
    pub memory_contents: PrivateState<HashMap<String, String>>,
}

/// Skills middleware state extension
/// Python: SkillsState(AgentState) with skills_metadata: PrivateStateAttr
pub struct SkillsStateExt {
    pub skills_metadata: PrivateState<Vec<SkillMetadata>>,
}
```

## Default Pipeline Construction

```rust
/// Python: create_deep_agent() middleware assembly
pub fn build_default_pipeline(config: &DeepAgentConfig) -> MiddlewarePipeline {
    let mut middlewares: Vec<Box<dyn Middleware>> = vec![
        Box::new(TodoListMiddleware::new()),
    ];

    if let Some(memory_sources) = &config.memory {
        middlewares.push(Box::new(MemoryMiddleware::new(
            config.backend.clone(),
            memory_sources.clone(),
        )));
    }

    if let Some(skill_sources) = &config.skills {
        middlewares.push(Box::new(SkillsMiddleware::new(
            config.backend.clone(),
            skill_sources.clone(),
        )));
    }

    middlewares.extend([
        Box::new(FilesystemMiddleware::new(config.backend.clone())),
        Box::new(SubAgentMiddleware::new(config.backend.clone(), config.subagents.clone())),
        Box::new(SummarizationMiddleware::new(config.model.clone(), config.backend.clone())),
        Box::new(PromptCachingMiddleware::new()),
        Box::new(PatchToolCallsMiddleware::new()),
    ]);

    middlewares.extend(config.extra_middleware.drain(..));

    if let Some(interrupt_on) = &config.interrupt_on {
        middlewares.push(Box::new(HumanInTheLoopMiddleware::new(interrupt_on.clone())));
    }

    MiddlewarePipeline::new(middlewares)
}
```

## Consequences

- Middleware pipeline is fully type-safe with compile-time guarantees
- Same ordering semantics as Python (sequential before_agent, nested wrap_model_call)
- `PrivateState<T>` prevents state leakage to parent agents (same as `PrivateStateAttr`)
- Tools collected from all middlewares match Python's tool aggregation behavior
