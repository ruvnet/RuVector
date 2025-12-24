//! Agent Specification Module
//!
//! Defines the core types for agent specifications, roles, capabilities, and tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Complete specification of an AI agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Unique identifier for this agent instance
    pub id: Uuid,

    /// Human-readable name
    pub name: String,

    /// The role this agent plays
    pub role: AgentRole,

    /// Capabilities this agent possesses
    pub capabilities: Vec<Capability>,

    /// Tools available to this agent
    pub tools: Vec<ToolSpec>,

    /// Detailed instructions for the agent
    pub instructions: String,

    /// Knowledge items this agent has access to
    pub knowledge_base: Vec<KnowledgeItem>,

    /// Hash of parent spec for lineage tracking (if spawned from another agent)
    pub parent_spec_hash: Option<[u8; 32]>,
}

impl AgentSpec {
    /// Calculate the hash of this specification
    pub fn hash(&self) -> [u8; 32] {
        let json = serde_json::to_string(self).unwrap_or_default();
        blake3::hash(json.as_bytes()).into()
    }

    /// Validate that all capabilities have required tools
    pub fn validate(&self) -> anyhow::Result<()> {
        let tool_names: Vec<String> = self.tools.iter().map(|t| t.name.clone()).collect();

        for cap in &self.capabilities {
            for required_tool in &cap.required_tools {
                if !tool_names.contains(required_tool) {
                    anyhow::bail!(
                        "Capability '{}' requires tool '{}' which is not available",
                        cap.name, required_tool
                    );
                }
            }
        }

        Ok(())
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Check if agent has a specific capability
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }
}

/// Roles that agents can play
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentRole {
    /// Data engineering and pipeline development
    DataEngineer,

    /// Machine learning model development
    MLEngineer,

    /// Research and analysis
    Researcher,

    /// Testing and quality assurance
    Tester,

    /// Code review and quality control
    Reviewer,

    /// High-level task orchestration
    Orchestrator,

    /// Custom role with specific name
    Custom(String),
}

impl AgentRole {
    /// Get a human-readable description of this role
    pub fn description(&self) -> &str {
        match self {
            AgentRole::DataEngineer => "Designs and implements data pipelines, ETL processes, and data infrastructure",
            AgentRole::MLEngineer => "Develops, trains, and deploys machine learning models",
            AgentRole::Researcher => "Conducts research, analyzes problems, and proposes solutions",
            AgentRole::Tester => "Writes and executes tests to ensure quality",
            AgentRole::Reviewer => "Reviews code, documentation, and processes for quality",
            AgentRole::Orchestrator => "Coordinates multiple agents and manages complex workflows",
            AgentRole::Custom(name) => name,
        }
    }
}

/// A capability that an agent possesses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Name of the capability (e.g., "python_development")
    pub name: String,

    /// Detailed description of what this capability enables
    pub description: String,

    /// Tools required to use this capability
    pub required_tools: Vec<String>,
}

/// Specification of a tool that an agent can use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name (e.g., "python", "git", "docker")
    pub name: String,

    /// Base command to execute
    pub command: String,

    /// Handlebars template for command arguments
    /// Variables available: {input}, {workspace}, {output_file}, etc.
    pub args_template: String,

    /// How to parse the tool's output
    pub output_parser: OutputParser,

    /// Optional working directory for tool execution
    pub working_dir: Option<String>,

    /// Environment variables to set
    pub env_vars: HashMap<String, String>,
}

impl ToolSpec {
    /// Expand the argument template with given variables
    pub fn expand_args(&self, variables: &HashMap<String, String>) -> Result<String, handlebars::RenderError> {
        let handlebars = handlebars::Handlebars::new();
        handlebars.render_template(&self.args_template, variables)
    }
}

/// How to parse tool output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputParser {
    /// Raw text output
    Raw,

    /// JSON output
    Json,

    /// Line-based output
    Lines,

    /// Regex pattern to extract specific data
    Regex { pattern: String, groups: Vec<String> },

    /// Custom parser (name reference)
    Custom(String),
}

/// Knowledge items in an agent's knowledge base
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KnowledgeItem {
    /// A framework or library (e.g., "LangChain", "PyTorch")
    Framework(String),

    /// A concept or technique (e.g., "RAG", "Fine-tuning")
    Concept(String),

    /// A best practice or pattern
    BestPractice(String),

    /// A reference document or URL
    Reference { title: String, url: String },

    /// Custom knowledge item
    Custom { category: String, content: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_spec_hash() {
        let spec = AgentSpec {
            id: Uuid::new_v4(),
            name: "Test".into(),
            role: AgentRole::Tester,
            capabilities: vec![],
            tools: vec![],
            instructions: "Test".into(),
            knowledge_base: vec![],
            parent_spec_hash: None,
        };

        let hash1 = spec.hash();
        let hash2 = spec.hash();
        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_validate_missing_tool() {
        let spec = AgentSpec {
            id: Uuid::new_v4(),
            name: "Test".into(),
            role: AgentRole::MLEngineer,
            capabilities: vec![Capability {
                name: "python_dev".into(),
                description: "Python development".into(),
                required_tools: vec!["python".into()],
            }],
            tools: vec![],
            instructions: "".into(),
            knowledge_base: vec![],
            parent_spec_hash: None,
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validate_with_tools() {
        let spec = AgentSpec {
            id: Uuid::new_v4(),
            name: "Test".into(),
            role: AgentRole::MLEngineer,
            capabilities: vec![Capability {
                name: "python_dev".into(),
                description: "Python development".into(),
                required_tools: vec!["python".into()],
            }],
            tools: vec![ToolSpec {
                name: "python".into(),
                command: "python3".into(),
                args_template: "-c '{{code}}'".into(),
                output_parser: OutputParser::Raw,
                working_dir: None,
                env_vars: HashMap::new(),
            }],
            instructions: "".into(),
            knowledge_base: vec![],
            parent_spec_hash: None,
        };

        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_tool_expand_args() {
        let tool = ToolSpec {
            name: "python".into(),
            command: "python3".into(),
            args_template: "-c '{{code}}' --output {{output}}".into(),
            output_parser: OutputParser::Raw,
            working_dir: None,
            env_vars: HashMap::new(),
        };

        let mut vars = HashMap::new();
        vars.insert("code".into(), "print('hello')".into());
        vars.insert("output".into(), "result.txt".into());

        let result = tool.expand_args(&vars).unwrap();
        println!("Expanded result: {}", result);
        assert!(result.contains("print(&#x27;hello&#x27;)") || result.contains("print('hello')"));
        assert!(result.contains("result.txt"));
    }
}
