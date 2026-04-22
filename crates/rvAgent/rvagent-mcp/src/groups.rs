//! Tool groups for organizing MCP tools into logical categories.
//!
//! Groups allow selective exposure of tools via CLI arguments,
//! reducing attack surface and improving discoverability.

use std::collections::HashSet;
use std::str::FromStr;

/// Tool group categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolGroup {
    /// File system operations: read, write, edit, ls, glob, grep
    File,
    /// Command execution: execute, bash
    Shell,
    /// Vector memory: semantic_search, store, retrieve
    Memory,
    /// Multi-agent: spawn, status, orchestrate
    Agent,
    /// Version control: git_status, git_commit, git_diff, git_log
    Git,
    /// Web operations: web_fetch, web_search
    Web,
    /// π Brain integration: brain_search, brain_share, brain_vote
    Brain,
    /// Task management: create_task, list_tasks, complete_task
    Task,
    /// Core utilities: ping, echo, version
    Core,
}

impl ToolGroup {
    /// Get the list of tool names in this group.
    pub fn tools(&self) -> &'static [&'static str] {
        match self {
            Self::File => &[
                "read_file",
                "write_file",
                "edit_file",
                "ls",
                "glob",
                "grep",
                "multi_edit",
            ],
            Self::Shell => &["execute", "bash", "run_command"],
            Self::Memory => &[
                "semantic_search",
                "store_memory",
                "retrieve_memory",
                "list_memories",
                "delete_memory",
            ],
            Self::Agent => &[
                "spawn_agent",
                "agent_status",
                "orchestrate",
                "terminate_agent",
                "list_agents",
            ],
            Self::Git => &[
                "git_status",
                "git_commit",
                "git_diff",
                "git_log",
                "git_add",
                "git_push",
                "git_pull",
            ],
            Self::Web => &["web_fetch", "web_search", "http_request"],
            Self::Brain => &[
                "brain_search",
                "brain_share",
                "brain_vote",
                "brain_get",
                "brain_list",
                "brain_status",
            ],
            Self::Task => &[
                "create_task",
                "list_tasks",
                "complete_task",
                "update_task",
                "cancel_task",
            ],
            Self::Core => &["ping", "echo", "version", "health"],
        }
    }

    /// Get all available groups.
    pub fn all() -> &'static [ToolGroup] {
        &[
            Self::File,
            Self::Shell,
            Self::Memory,
            Self::Agent,
            Self::Git,
            Self::Web,
            Self::Brain,
            Self::Task,
            Self::Core,
        ]
    }

    /// Get all tool names across all groups.
    pub fn all_tools() -> Vec<&'static str> {
        Self::all()
            .iter()
            .flat_map(|g| g.tools().iter().copied())
            .collect()
    }

    /// Get the string representation of the group.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Shell => "shell",
            Self::Memory => "memory",
            Self::Agent => "agent",
            Self::Git => "git",
            Self::Web => "web",
            Self::Brain => "brain",
            Self::Task => "task",
            Self::Core => "core",
        }
    }
}

impl FromStr for ToolGroup {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "file" | "files" | "fs" => Ok(Self::File),
            "shell" | "sh" | "exec" => Ok(Self::Shell),
            "memory" | "mem" | "vector" => Ok(Self::Memory),
            "agent" | "agents" | "swarm" => Ok(Self::Agent),
            "git" | "vcs" => Ok(Self::Git),
            "web" | "http" | "net" => Ok(Self::Web),
            "brain" | "pi" | "π" => Ok(Self::Brain),
            "task" | "tasks" | "todo" => Ok(Self::Task),
            "core" | "util" | "utils" => Ok(Self::Core),
            _ => Err(format!("unknown tool group: {}", s)),
        }
    }
}

impl std::fmt::Display for ToolGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Tool filter based on selected groups.
#[derive(Debug, Clone)]
pub struct ToolFilter {
    /// Set of allowed tool names.
    allowed: HashSet<String>,
    /// Whether all tools are allowed.
    allow_all: bool,
}

impl ToolFilter {
    /// Create a filter that allows all tools.
    pub fn all() -> Self {
        Self {
            allowed: HashSet::new(),
            allow_all: true,
        }
    }

    /// Create a filter from a list of groups.
    pub fn from_groups(groups: &[ToolGroup]) -> Self {
        let allowed: HashSet<String> = groups
            .iter()
            .flat_map(|g| g.tools().iter().map(|s| s.to_string()))
            .collect();
        Self {
            allowed,
            allow_all: false,
        }
    }

    /// Create a filter from group names (strings).
    pub fn from_group_names(names: &[String]) -> Result<Self, String> {
        let groups: Result<Vec<ToolGroup>, _> = names.iter().map(|n| n.parse()).collect();
        groups.map(|g| Self::from_groups(&g))
    }

    /// Check if a tool is allowed by this filter.
    pub fn is_allowed(&self, tool_name: &str) -> bool {
        self.allow_all || self.allowed.contains(tool_name)
    }

    /// Get the number of allowed tools (0 means all).
    pub fn count(&self) -> usize {
        if self.allow_all {
            0
        } else {
            self.allowed.len()
        }
    }

    /// Check if all tools are allowed.
    pub fn allows_all(&self) -> bool {
        self.allow_all
    }

    /// Get the set of allowed tool names.
    pub fn allowed_tools(&self) -> &HashSet<String> {
        &self.allowed
    }
}

impl Default for ToolFilter {
    fn default() -> Self {
        // Default: allow core + file groups
        Self::from_groups(&[ToolGroup::Core, ToolGroup::File])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_group_tools() {
        assert!(ToolGroup::File.tools().contains(&"read_file"));
        assert!(ToolGroup::Shell.tools().contains(&"execute"));
        assert!(ToolGroup::Memory.tools().contains(&"semantic_search"));
    }

    #[test]
    fn test_tool_group_from_str() {
        assert_eq!("file".parse::<ToolGroup>().unwrap(), ToolGroup::File);
        assert_eq!("shell".parse::<ToolGroup>().unwrap(), ToolGroup::Shell);
        assert_eq!("brain".parse::<ToolGroup>().unwrap(), ToolGroup::Brain);
        assert_eq!("π".parse::<ToolGroup>().unwrap(), ToolGroup::Brain);
    }

    #[test]
    fn test_tool_group_from_str_invalid() {
        assert!("invalid".parse::<ToolGroup>().is_err());
    }

    #[test]
    fn test_tool_filter_all() {
        let filter = ToolFilter::all();
        assert!(filter.is_allowed("anything"));
        assert!(filter.is_allowed("read_file"));
        assert!(filter.allows_all());
    }

    #[test]
    fn test_tool_filter_from_groups() {
        let filter = ToolFilter::from_groups(&[ToolGroup::File]);
        assert!(filter.is_allowed("read_file"));
        assert!(filter.is_allowed("write_file"));
        assert!(!filter.is_allowed("execute"));
        assert!(!filter.allows_all());
    }

    #[test]
    fn test_tool_filter_from_group_names() {
        let filter =
            ToolFilter::from_group_names(&["file".to_string(), "shell".to_string()]).unwrap();
        assert!(filter.is_allowed("read_file"));
        assert!(filter.is_allowed("execute"));
        assert!(!filter.is_allowed("brain_search"));
    }

    #[test]
    fn test_tool_filter_from_group_names_invalid() {
        let result = ToolFilter::from_group_names(&["invalid".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_filter_default() {
        let filter = ToolFilter::default();
        assert!(filter.is_allowed("ping")); // core
        assert!(filter.is_allowed("read_file")); // file
        assert!(!filter.is_allowed("execute")); // shell (not in default)
    }

    #[test]
    fn test_tool_group_all() {
        let all = ToolGroup::all();
        assert!(all.len() >= 9);
    }

    #[test]
    fn test_tool_group_all_tools() {
        let tools = ToolGroup::all_tools();
        assert!(tools.contains(&"ping"));
        assert!(tools.contains(&"read_file"));
        assert!(tools.contains(&"brain_search"));
    }

    #[test]
    fn test_tool_group_display() {
        assert_eq!(format!("{}", ToolGroup::File), "file");
        assert_eq!(format!("{}", ToolGroup::Brain), "brain");
    }
}
