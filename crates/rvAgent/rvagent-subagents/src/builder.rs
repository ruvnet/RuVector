//! Subagent compilation — converts `SubAgentSpec` into `CompiledSubAgent`.
//!
//! Each compiled subagent receives:
//! - Its own isolated agent graph
//! - A middleware pipeline subset based on its capabilities
//! - State isolation via `EXCLUDED_STATE_KEYS`

use crate::{CompiledSubAgent, RvAgentConfig, SubAgentSpec, EXCLUDED_STATE_KEYS};

/// Default middleware applied to all subagents regardless of capabilities.
const BASE_MIDDLEWARE: &[&str] = &["prompt_caching", "patch_tool_calls"];

/// Additional middleware for subagents with file-system read access.
const READ_MIDDLEWARE: &[&str] = &["filesystem"];

/// Additional middleware for subagents with write access.
const WRITE_MIDDLEWARE: &[&str] = &["todo_list", "summarization"];

/// Additional middleware for subagents with execute access.
const EXECUTE_MIDDLEWARE: &[&str] = &["execution_guard"];

/// Compile a list of subagent specs into runnable `CompiledSubAgent`s.
///
/// Each subagent gets:
/// - An isolated graph with nodes: `start -> agent -> tools -> end`
/// - A middleware subset based on `can_read`, `can_write`, `can_execute`
/// - The parent's backend identifier (from config)
///
/// State isolation is enforced by `EXCLUDED_STATE_KEYS` — the parent's
/// messages, todos, and completion state are never visible to subagents.
pub fn compile_subagents(
    specs: &[SubAgentSpec],
    parent_config: &RvAgentConfig,
) -> Vec<CompiledSubAgent> {
    specs
        .iter()
        .map(|spec| compile_single(spec, parent_config))
        .collect()
}

/// Compile a single spec into a `CompiledSubAgent`.
fn compile_single(spec: &SubAgentSpec, parent_config: &RvAgentConfig) -> CompiledSubAgent {
    let graph = build_graph(spec);
    let middleware_pipeline = build_middleware_pipeline(spec, parent_config);
    let backend = resolve_backend(spec, parent_config);

    CompiledSubAgent {
        spec: spec.clone(),
        graph,
        middleware_pipeline,
        backend,
    }
}

/// Build the graph node list for a subagent.
///
/// The graph follows the standard agent loop:
/// `start -> agent_loop -> tool_dispatch -> end`
fn build_graph(spec: &SubAgentSpec) -> Vec<String> {
    let mut nodes = vec![
        "start".to_string(),
        format!("agent:{}", spec.name),
    ];

    if !spec.tools.is_empty() || spec.can_read || spec.can_write || spec.can_execute {
        nodes.push("tool_dispatch".to_string());
    }

    nodes.push("end".to_string());
    nodes
}

/// Build the middleware pipeline for a subagent based on its capabilities.
///
/// Always includes base middleware. Adds filesystem, todo_list, summarization,
/// and execution_guard based on the spec's capability flags.
fn build_middleware_pipeline(spec: &SubAgentSpec, parent_config: &RvAgentConfig) -> Vec<String> {
    let mut pipeline: Vec<String> = BASE_MIDDLEWARE.iter().map(|s| s.to_string()).collect();

    if spec.can_read {
        pipeline.extend(READ_MIDDLEWARE.iter().map(|s| s.to_string()));
    }

    if spec.can_write {
        pipeline.extend(WRITE_MIDDLEWARE.iter().map(|s| s.to_string()));
    }

    if spec.can_execute {
        pipeline.extend(EXECUTE_MIDDLEWARE.iter().map(|s| s.to_string()));
    }

    // Only include parent middleware that the subagent is allowed to use
    for mw in &parent_config.middleware {
        if !pipeline.contains(mw) && is_safe_middleware(mw) {
            pipeline.push(mw.clone());
        }
    }

    pipeline
}

/// Check if a middleware is safe to propagate to subagents.
///
/// Some middleware (like subagent middleware itself) should not be
/// recursively applied to prevent infinite nesting.
fn is_safe_middleware(name: &str) -> bool {
    !matches!(name, "subagent" | "hitl" | "human_in_the_loop")
}

/// Resolve the backend identifier for a subagent.
fn resolve_backend(spec: &SubAgentSpec, parent_config: &RvAgentConfig) -> String {
    if spec.can_execute {
        "local_shell".to_string()
    } else if spec.can_write {
        parent_config.cwd.clone().unwrap_or_else(|| "filesystem".to_string())
    } else {
        "read_only".to_string()
    }
}

/// Return the list of state keys that must be excluded during subagent
/// state preparation and result merging.
pub fn excluded_state_keys() -> &'static [&'static str] {
    EXCLUDED_STATE_KEYS
}

/// Resolve tools for a subagent — uses spec tools if specified,
/// otherwise inherits from parent config.
pub fn resolve_tools(spec: &SubAgentSpec, parent_config: &RvAgentConfig) -> Vec<String> {
    if spec.tools.is_empty() {
        // Inherit parent tools, filtered by capability
        parent_config
            .tools
            .iter()
            .filter(|t| {
                if !spec.can_write && is_write_tool(t) {
                    return false;
                }
                if !spec.can_execute && is_execute_tool(t) {
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    } else {
        spec.tools.clone()
    }
}

fn is_write_tool(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "write_todos")
}

fn is_execute_tool(name: &str) -> bool {
    matches!(name, "execute" | "shell" | "run_command")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SubAgentSpec;

    fn test_config() -> RvAgentConfig {
        RvAgentConfig {
            default_model: Some("anthropic:claude-sonnet-4-20250514".into()),
            tools: vec![
                "read_file".into(),
                "write_file".into(),
                "grep".into(),
                "execute".into(),
            ],
            middleware: vec!["prompt_caching".into(), "summarization".into()],
            cwd: Some("/tmp/project".into()),
        }
    }

    #[test]
    fn test_compile_subagent_from_spec() {
        let spec = SubAgentSpec::new("researcher", "Find information.");
        let config = test_config();
        let compiled = compile_subagents(&[spec], &config);

        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].spec.name, "researcher");
        assert!(!compiled[0].graph.is_empty());
        assert!(compiled[0].graph.contains(&"start".to_string()));
        assert!(compiled[0].graph.contains(&"end".to_string()));
    }

    #[test]
    fn test_compile_multiple_specs() {
        let specs = vec![
            SubAgentSpec::new("a", "Do A"),
            SubAgentSpec::new("b", "Do B"),
            SubAgentSpec::general_purpose(),
        ];
        let compiled = compile_subagents(&specs, &test_config());
        assert_eq!(compiled.len(), 3);
    }

    #[test]
    fn test_middleware_pipeline_read_only() {
        let spec = SubAgentSpec {
            can_read: true,
            can_write: false,
            can_execute: false,
            ..SubAgentSpec::new("reader", "Read stuff")
        };
        let pipeline = build_middleware_pipeline(&spec, &test_config());

        assert!(pipeline.contains(&"prompt_caching".to_string()));
        assert!(pipeline.contains(&"filesystem".to_string()));
        assert!(!pipeline.contains(&"execution_guard".to_string()));
        // summarization from parent config should be added since it's safe
        assert!(pipeline.contains(&"summarization".to_string()));
    }

    #[test]
    fn test_middleware_pipeline_full_access() {
        let spec = SubAgentSpec::general_purpose();
        let pipeline = build_middleware_pipeline(&spec, &test_config());

        assert!(pipeline.contains(&"prompt_caching".to_string()));
        assert!(pipeline.contains(&"filesystem".to_string()));
        assert!(pipeline.contains(&"todo_list".to_string()));
        assert!(pipeline.contains(&"execution_guard".to_string()));
    }

    #[test]
    fn test_subagent_middleware_not_propagated() {
        let spec = SubAgentSpec::new("child", "Do child work");
        let mut config = test_config();
        config.middleware.push("subagent".into());

        let pipeline = build_middleware_pipeline(&spec, &config);
        assert!(!pipeline.contains(&"subagent".to_string()));
    }

    #[test]
    fn test_resolve_tools_inherits_parent() {
        let spec = SubAgentSpec {
            can_read: true,
            can_write: false,
            can_execute: false,
            ..SubAgentSpec::new("reader", "Read files")
        };
        let tools = resolve_tools(&spec, &test_config());
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"grep".to_string()));
        assert!(!tools.contains(&"write_file".to_string()));
        assert!(!tools.contains(&"execute".to_string()));
    }

    #[test]
    fn test_resolve_tools_explicit_list() {
        let mut spec = SubAgentSpec::new("custom", "Custom tools");
        spec.tools = vec!["my_tool".into()];
        let tools = resolve_tools(&spec, &test_config());
        assert_eq!(tools, vec!["my_tool".to_string()]);
    }

    #[test]
    fn test_backend_resolution() {
        let config = test_config();

        let read_spec = SubAgentSpec::new("reader", "Read");
        assert_eq!(resolve_backend(&read_spec, &config), "read_only");

        let mut write_spec = SubAgentSpec::new("writer", "Write");
        write_spec.can_write = true;
        assert_eq!(resolve_backend(&write_spec, &config), "/tmp/project");

        let exec_spec = SubAgentSpec::general_purpose();
        assert_eq!(resolve_backend(&exec_spec, &config), "local_shell");
    }

    #[test]
    fn test_graph_structure() {
        let spec = SubAgentSpec::general_purpose();
        let graph = build_graph(&spec);
        assert_eq!(graph[0], "start");
        assert!(graph[1].starts_with("agent:"));
        assert!(graph.contains(&"tool_dispatch".to_string()));
        assert_eq!(*graph.last().unwrap(), "end");
    }

    #[test]
    fn test_graph_no_tools_node_for_toolless_agent() {
        let spec = SubAgentSpec {
            can_read: false,
            can_write: false,
            can_execute: false,
            tools: Vec::new(),
            ..SubAgentSpec::new("thinker", "Just think")
        };
        let graph = build_graph(&spec);
        assert!(!graph.contains(&"tool_dispatch".to_string()));
        assert_eq!(graph, vec!["start", "agent:thinker", "end"]);
    }
}
