//! SubAgentMiddleware — compiles subagent specs and provides the task tool.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    AgentState, AgentStateUpdate, Middleware, ModelHandler, ModelRequest, ModelResponse,
    RunnableConfig, Runtime, Tool,
};

/// Specification for a subagent that can be spawned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub tools: Vec<String>,
}

/// Middleware that manages subagent spawning.
///
/// - `before_agent`: compiles subagent specs from configuration
/// - `tools()`: returns the `task` tool for spawning subagents
pub struct SubAgentMiddleware {
    specs: Vec<SubAgentSpec>,
}

impl SubAgentMiddleware {
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    pub fn with_specs(specs: Vec<SubAgentSpec>) -> Self {
        Self { specs }
    }

    fn format_subagent_descriptions(&self) -> String {
        if self.specs.is_empty() {
            return String::new();
        }
        let mut out = String::from("Available subagents:\n");
        for spec in &self.specs {
            out.push_str(&format!("- {}: {}\n", spec.name, spec.description));
        }
        out
    }
}

impl Default for SubAgentMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for SubAgentMiddleware {
    fn name(&self) -> &str {
        "subagent"
    }

    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if self.specs.is_empty() {
            return None;
        }

        let mut update = AgentStateUpdate::default();
        update.extensions.insert(
            "subagent_specs".into(),
            serde_json::to_value(&self.specs).unwrap_or_default(),
        );
        Some(update)
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        if self.specs.is_empty() {
            return handler.call(request);
        }

        let descriptions = self.format_subagent_descriptions();
        let new_system = crate::append_to_system_message(&request.system_message, &descriptions);
        handler.call(request.with_system(new_system))
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(TaskTool)]
    }
}

/// Tool for spawning subagents.
struct TaskTool;

impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific task. The subagent runs independently and returns its result."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the task for the subagent"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt/instructions for the subagent"
                },
                "agent": {
                    "type": "string",
                    "description": "Name of the subagent type to spawn (optional)"
                }
            },
            "required": ["description", "prompt"]
        })
    }

    fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
        Err("task tool must be invoked through the agent runtime".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_name() {
        let mw = SubAgentMiddleware::new();
        assert_eq!(mw.name(), "subagent");
    }

    #[test]
    fn test_tools() {
        let mw = SubAgentMiddleware::new();
        let tools = mw.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "task");
    }

    #[test]
    fn test_before_agent_no_specs() {
        let mw = SubAgentMiddleware::new();
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_before_agent_with_specs() {
        let specs = vec![SubAgentSpec {
            name: "coder".into(),
            description: "A coding agent".into(),
            model: None,
            system_prompt: None,
            tools: vec!["read_file".into()],
        }];
        let mw = SubAgentMiddleware::with_specs(specs);
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());
        assert!(update.unwrap().extensions.contains_key("subagent_specs"));
    }

    #[test]
    fn test_format_subagent_descriptions() {
        let specs = vec![
            SubAgentSpec {
                name: "coder".into(),
                description: "Writes code".into(),
                model: None,
                system_prompt: None,
                tools: vec![],
            },
            SubAgentSpec {
                name: "reviewer".into(),
                description: "Reviews code".into(),
                model: None,
                system_prompt: None,
                tools: vec![],
            },
        ];
        let mw = SubAgentMiddleware::with_specs(specs);
        let desc = mw.format_subagent_descriptions();
        assert!(desc.contains("coder: Writes code"));
        assert!(desc.contains("reviewer: Reviews code"));
    }

    #[test]
    fn test_task_tool_schema() {
        let tool = TaskTool;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("description")));
        assert!(required.contains(&serde_json::json!("prompt")));
    }
}
