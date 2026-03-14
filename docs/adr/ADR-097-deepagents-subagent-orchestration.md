# ADR-097: SubAgent & Task Orchestration

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crate**   | `ruvector-deep-subagents`                       |

## Context

DeepAgents' `SubAgentMiddleware` provides a `task` tool that spawns ephemeral subagents with isolated context. Key behaviors:

1. **SubAgent spec** — `TypedDict` with name, description, system_prompt, optional model/tools/middleware
2. **CompiledSubAgent** — Pre-built runnable with name and description
3. **General-purpose agent** — Default subagent with same tools as parent
4. **State isolation** — Excluded keys: `messages`, `todos`, `structured_response`, `skills_metadata`, `memory_contents`
5. **Task tool** — Accepts `description` and `subagent_type`, returns subagent's final message
6. **System prompt injection** — TASK_SYSTEM_PROMPT appended to parent's system prompt
7. **Parallel execution** — LLM can invoke multiple task tools in one message

## Decision

### SubAgent Types

```rust
// crates/ruvector-deep-subagents/src/lib.rs

use serde::{Deserialize, Serialize};

/// SubAgent specification (not yet compiled).
/// Python: SubAgent(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    #[serde(default)]
    pub tools: Option<Vec<ToolSpec>>,
    #[serde(default)]
    pub model: Option<ModelSpec>,
    #[serde(default)]
    pub middleware: Option<Vec<MiddlewareSpec>>,
    #[serde(default)]
    pub interrupt_on: Option<HashMap<String, InterruptConfig>>,
    #[serde(default)]
    pub skills: Option<Vec<String>>,
}

/// Pre-compiled subagent with a runnable graph.
/// Python: CompiledSubAgent(TypedDict)
pub struct CompiledSubAgent {
    pub name: String,
    pub description: String,
    pub runnable: Box<dyn AgentRunnable>,
}

/// Trait for runnable agent graphs.
/// Python: langgraph Runnable with 'messages' in state
#[async_trait]
pub trait AgentRunnable: Send + Sync {
    fn invoke(&self, state: AgentState) -> AgentState;
    async fn ainvoke(&self, state: AgentState) -> AgentState;
}

/// Model specification — either a string ("provider:model") or configured instance.
/// Python: str | BaseChatModel
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelSpec {
    String(String),
    Config(ModelConfig),
}
```

### General-Purpose SubAgent

```rust
/// Python: GENERAL_PURPOSE_SUBAGENT constant
pub const GENERAL_PURPOSE_NAME: &str = "general-purpose";

pub const GENERAL_PURPOSE_DESCRIPTION: &str =
    "General-purpose agent for researching complex questions, searching for files \
     and content, and executing multi-step tasks. When you are searching for a keyword \
     or file and are not confident that you will find the right match in the first few \
     tries use this agent to perform the search for you. This agent has access to all \
     tools as the main agent.";

pub const DEFAULT_SUBAGENT_PROMPT: &str =
    "In order to complete the objective that the user asks of you, you have access \
     to a number of standard tools.";
```

### State Isolation

```rust
/// Keys excluded when passing state to/from subagents.
/// Python: _EXCLUDED_STATE_KEYS
const EXCLUDED_STATE_KEYS: &[&str] = &[
    "messages",
    "todos",
    "structured_response",
    "skills_metadata",
    "memory_contents",
];

/// Filter state for subagent invocation.
fn prepare_subagent_state(parent_state: &AgentState, task_description: &str) -> AgentState {
    let mut state: AgentState = parent_state
        .iter()
        .filter(|(k, _)| !EXCLUDED_STATE_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Replace messages with single HumanMessage containing the task description
    state.insert(
        "messages".to_string(),
        serde_json::json!([{"type": "human", "content": task_description}]),
    );

    state
}

/// Extract result from subagent state.
/// Python: _return_command_with_state_update
fn extract_subagent_result(
    result: AgentState,
    tool_call_id: &str,
) -> ToolResult {
    let messages = result.get("messages")
        .expect("CompiledSubAgent must return state with 'messages' key");

    let final_message = messages.as_array().unwrap().last().unwrap();
    let message_text = final_message["content"].as_str().unwrap_or("").trim_end();

    // Collect non-excluded state updates
    let state_update: AgentState = result
        .into_iter()
        .filter(|(k, _)| !EXCLUDED_STATE_KEYS.contains(&k.as_str()))
        .collect();

    ToolResult::Command(StateUpdate::SubAgentResult {
        state_update,
        tool_message: ToolMessage {
            content: message_text.to_string(),
            tool_call_id: tool_call_id.to_string(),
        },
    })
}
```

### Task Tool Construction

```rust
/// Build the task tool from subagent specs.
/// Python: _build_task_tool(subagents, task_description)
pub fn build_task_tool(
    subagents: &[CompiledSubAgent],
    task_description: Option<&str>,
) -> Box<dyn Tool> {
    let graphs: HashMap<String, &dyn AgentRunnable> = subagents
        .iter()
        .map(|s| (s.name.clone(), s.runnable.as_ref()))
        .collect();

    let agents_desc = subagents
        .iter()
        .map(|s| format!("- {}: {}", s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n");

    let description = match task_description {
        Some(desc) if desc.contains("{available_agents}") => {
            desc.replace("{available_agents}", &agents_desc)
        }
        Some(desc) => desc.to_string(),
        None => TASK_TOOL_DESCRIPTION.replace("{available_agents}", &agents_desc),
    };

    Box::new(TaskTool {
        graphs,
        description,
    })
}

struct TaskTool {
    graphs: HashMap<String, Box<dyn AgentRunnable>>,
    description: String,
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str { "task" }
    fn description(&self) -> &str { &self.description }

    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let description = args["description"].as_str().unwrap();
        let subagent_type = args["subagent_type"].as_str().unwrap();

        // Validate subagent type exists
        let runnable = match self.graphs.get(subagent_type) {
            Some(r) => r,
            None => {
                let allowed = self.graphs.keys()
                    .map(|k| format!("`{}`", k))
                    .collect::<Vec<_>>()
                    .join(", ");
                return ToolResult::Text(format!(
                    "We cannot invoke subagent {} because it does not exist, \
                     the only allowed types are {}",
                    subagent_type, allowed
                ));
            }
        };

        let tool_call_id = runtime.tool_call_id.as_ref()
            .expect("Tool call ID is required for subagent invocation");

        let subagent_state = prepare_subagent_state(&runtime.state, description);
        let result = runnable.invoke(subagent_state);
        extract_subagent_result(result, tool_call_id)
    }

    async fn ainvoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        // Same logic but with ainvoke on the runnable
        let description = args["description"].as_str().unwrap();
        let subagent_type = args["subagent_type"].as_str().unwrap();

        let runnable = match self.graphs.get(subagent_type) {
            Some(r) => r,
            None => {
                let allowed = self.graphs.keys()
                    .map(|k| format!("`{}`", k))
                    .collect::<Vec<_>>()
                    .join(", ");
                return ToolResult::Text(format!(
                    "We cannot invoke subagent {} because it does not exist, \
                     the only allowed types are {}",
                    subagent_type, allowed
                ));
            }
        };

        let tool_call_id = runtime.tool_call_id.as_ref()
            .expect("Tool call ID is required for subagent invocation");

        let subagent_state = prepare_subagent_state(&runtime.state, description);
        let result = runnable.ainvoke(subagent_state).await;
        extract_subagent_result(result, tool_call_id)
    }
}
```

### SubAgentMiddleware

```rust
/// Python: SubAgentMiddleware(AgentMiddleware)
pub struct SubAgentMiddleware {
    task_tool: Box<dyn Tool>,
    system_prompt: Option<String>,
}

impl SubAgentMiddleware {
    pub fn new(
        backend: BackendRef,
        subagents: Vec<SubAgentSpec>,
        system_prompt: Option<String>,
    ) -> Self {
        // Build compiled subagents from specs
        // Each subagent gets its own middleware pipeline:
        //   TodoList, Filesystem, Summarization, PromptCaching, PatchToolCalls
        let compiled = compile_subagents(backend, subagents);
        let task_tool = build_task_tool(&compiled, None);

        // Build system prompt with agent descriptions
        let prompt = system_prompt.unwrap_or_else(|| TASK_SYSTEM_PROMPT.to_string());
        let agents_desc = compiled.iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");
        let full_prompt = format!("{}\n\nAvailable subagent types:\n{}", prompt, agents_desc);

        Self {
            task_tool,
            system_prompt: Some(full_prompt),
        }
    }
}

impl Middleware for SubAgentMiddleware {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![self.task_tool.clone()]
    }

    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        if let Some(ref prompt) = self.system_prompt {
            let new_system = append_to_system_message(&request.system_message, prompt);
            handler(request.override_system(new_system))
        } else {
            handler(request)
        }
    }
}
```

### System Prompts (Exact Fidelity)

The following prompts are preserved verbatim from Python:

- `TASK_TOOL_DESCRIPTION` — 237 lines of tool description with examples
- `TASK_SYSTEM_PROMPT` — Instructions for when/how to use task tool
- `BASE_AGENT_PROMPT` — Core agent behavior instructions

All stored as `const &str` in Rust with identical content.

## Consequences

- Task tool behavior is identical: same validation, same error messages, same state isolation
- Subagent compilation mirrors Python's `create_agent()` with same middleware stack
- General-purpose subagent is auto-included unless overridden by name
- Parallel task invocation supported (LLM sends multiple tool_calls)
- State isolation prevents leakage of todos, skills, memory between agents
