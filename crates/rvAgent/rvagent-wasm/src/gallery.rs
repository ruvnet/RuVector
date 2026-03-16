//! RVF App Gallery — pre-built agent templates and configurations.
//!
//! Provides a gallery of RVF containers that can be loaded, configured,
//! and used in the chat system. Templates include complete agent setups
//! with tools, prompts, skills, and orchestrator configurations.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::bridge::to_js_value;
use crate::rvf::{
    AgentNode, AgentPrompt, CapabilityDef, McpToolEntry, OrchestratorConfig,
    SkillDefinition, ToolDefinition, WasmRvfBuilder,
};

// ---------------------------------------------------------------------------
// Gallery Constants
// ---------------------------------------------------------------------------

/// Maximum number of custom templates in the gallery
pub const MAX_CUSTOM_TEMPLATES: usize = 100;

/// Maximum template name length
pub const MAX_TEMPLATE_NAME_LENGTH: usize = 64;

/// Maximum template description length
pub const MAX_TEMPLATE_DESC_LENGTH: usize = 512;

// ---------------------------------------------------------------------------
// Gallery Types
// ---------------------------------------------------------------------------

/// Template category for organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TemplateCategory {
    /// Development and coding agents
    Development,
    /// Research and analysis agents
    Research,
    /// Testing and QA agents
    Testing,
    /// Documentation agents
    Documentation,
    /// DevOps and deployment agents
    DevOps,
    /// Security-focused agents
    Security,
    /// Multi-agent orchestration
    Orchestration,
    /// Custom user-defined templates
    Custom,
}

/// A gallery template entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryTemplate {
    /// Unique template identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Template description
    pub description: String,
    /// Category for organization
    pub category: TemplateCategory,
    /// Version string
    pub version: String,
    /// Author/maintainer
    pub author: String,
    /// Tags for search
    pub tags: Vec<String>,
    /// Tool definitions
    pub tools: Vec<ToolDefinition>,
    /// Agent prompts
    pub prompts: Vec<AgentPrompt>,
    /// Skill definitions
    pub skills: Vec<SkillDefinition>,
    /// MCP tool entries
    pub mcp_tools: Vec<McpToolEntry>,
    /// Capability definitions
    pub capabilities: Vec<CapabilityDef>,
    /// Optional orchestrator config for multi-agent setups
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestrator: Option<OrchestratorConfig>,
    /// Whether this is a built-in template
    #[serde(default)]
    pub builtin: bool,
}

impl GalleryTemplate {
    /// Build an RVF container from this template
    pub fn to_rvf(&self) -> Vec<u8> {
        let mut builder = WasmRvfBuilder::new();

        // Add tools
        if !self.tools.is_empty() {
            let tools_json = serde_json::to_string(&self.tools).unwrap();
            let _ = builder.add_tools(&tools_json);
        }

        // Add prompts
        if !self.prompts.is_empty() {
            let prompts_json = serde_json::to_string(&self.prompts).unwrap();
            let _ = builder.add_prompts(&prompts_json);
        }

        // Add skills
        if !self.skills.is_empty() {
            let skills_json = serde_json::to_string(&self.skills).unwrap();
            let _ = builder.add_skills(&skills_json);
        }

        // Add MCP tools
        if !self.mcp_tools.is_empty() {
            let mcp_json = serde_json::to_string(&self.mcp_tools).unwrap();
            let _ = builder.add_mcp_tools(&mcp_json);
        }

        // Add capabilities
        if !self.capabilities.is_empty() {
            let caps_json = serde_json::to_string(&self.capabilities).unwrap();
            let _ = builder.add_capabilities(&caps_json);
        }

        // Add orchestrator if present
        if let Some(ref orch) = self.orchestrator {
            let orch_json = serde_json::to_string(orch).unwrap();
            let _ = builder.set_orchestrator(&orch_json);
        }

        builder.build_internal()
    }
}

/// Gallery search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: TemplateCategory,
    pub tags: Vec<String>,
    pub relevance: f32,
}

// ---------------------------------------------------------------------------
// Built-in Templates
// ---------------------------------------------------------------------------

/// Create the coder template
fn coder_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "coder".to_string(),
        name: "Coder Agent".to_string(),
        description: "A coding assistant that can read, write, and edit files with best practices".to_string(),
        category: TemplateCategory::Development,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["code".into(), "development".into(), "programming".into(), "files".into()],
        tools: vec![
            ToolDefinition {
                name: "analyze_code".to_string(),
                description: "Analyze code for issues and improvements".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "code": { "type": "string" },
                        "language": { "type": "string" }
                    },
                    "required": ["code"]
                }),
                returns: Some("Analysis results".to_string()),
            },
        ],
        prompts: vec![
            AgentPrompt {
                name: "coder".to_string(),
                system_prompt: "You are an expert software engineer. Write clean, efficient, and well-documented code. Follow best practices and design patterns. Always consider edge cases and error handling.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![
            SkillDefinition {
                name: "refactor".to_string(),
                description: "Refactor code for better readability and performance".to_string(),
                trigger: "/refactor".to_string(),
                content: "Analyze the code and suggest improvements for readability, performance, and maintainability.".to_string(),
            },
            SkillDefinition {
                name: "explain".to_string(),
                description: "Explain code in detail".to_string(),
                trigger: "/explain".to_string(),
                content: "Provide a detailed explanation of how the code works, including its purpose, logic flow, and key concepts.".to_string(),
            },
        ],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read file contents".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
            McpToolEntry {
                name: "write_file".to_string(),
                description: "Write file contents".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}, "content": {"type": "string"}}),
                group: Some("file".to_string()),
            },
            McpToolEntry {
                name: "edit_file".to_string(),
                description: "Edit file with string replacement".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 0,
            },
            CapabilityDef {
                name: "file_write".to_string(),
                rights: vec!["write".to_string(), "create".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 0,
            },
        ],
        orchestrator: None,
        builtin: true,
    }
}

/// Create the researcher template
fn researcher_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "researcher".to_string(),
        name: "Research Agent".to_string(),
        description: "A research assistant that gathers information, analyzes data, and synthesizes findings".to_string(),
        category: TemplateCategory::Research,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["research".into(), "analysis".into(), "information".into(), "synthesis".into()],
        tools: vec![
            ToolDefinition {
                name: "web_search".to_string(),
                description: "Search the web for information".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "max_results": { "type": "integer", "default": 10 }
                    },
                    "required": ["query"]
                }),
                returns: Some("Search results".to_string()),
            },
            ToolDefinition {
                name: "summarize".to_string(),
                description: "Summarize long content".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "max_length": { "type": "integer" }
                    },
                    "required": ["content"]
                }),
                returns: Some("Summary".to_string()),
            },
        ],
        prompts: vec![
            AgentPrompt {
                name: "researcher".to_string(),
                system_prompt: "You are a meticulous research assistant. Gather comprehensive information from multiple sources, verify facts, identify patterns, and synthesize findings into clear, well-organized reports. Always cite sources and acknowledge limitations.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![
            SkillDefinition {
                name: "deep_dive".to_string(),
                description: "Perform deep research on a topic".to_string(),
                trigger: "/deepdive".to_string(),
                content: "Conduct comprehensive research including background, current state, key players, trends, and future outlook.".to_string(),
            },
        ],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read file for analysis".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 1,
            },
            CapabilityDef {
                name: "web_access".to_string(),
                rights: vec!["fetch".to_string()],
                scope: "network".to_string(),
                delegation_depth: 0,
            },
        ],
        orchestrator: None,
        builtin: true,
    }
}

/// Create the tester template
fn tester_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "tester".to_string(),
        name: "Testing Agent".to_string(),
        description: "A QA and testing agent that writes tests, analyzes coverage, and finds bugs".to_string(),
        category: TemplateCategory::Testing,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["testing".into(), "qa".into(), "coverage".into(), "bugs".into(), "tdd".into()],
        tools: vec![
            ToolDefinition {
                name: "generate_tests".to_string(),
                description: "Generate test cases for code".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "code": { "type": "string" },
                        "framework": { "type": "string", "enum": ["jest", "pytest", "rust", "go"] }
                    },
                    "required": ["code"]
                }),
                returns: Some("Generated tests".to_string()),
            },
        ],
        prompts: vec![
            AgentPrompt {
                name: "tester".to_string(),
                system_prompt: "You are a thorough QA engineer. Write comprehensive tests covering edge cases, error conditions, and happy paths. Analyze code coverage and identify untested paths. Follow testing best practices and TDD principles.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![
            SkillDefinition {
                name: "coverage".to_string(),
                description: "Analyze test coverage gaps".to_string(),
                trigger: "/coverage".to_string(),
                content: "Analyze the codebase for untested code paths and suggest tests to improve coverage.".to_string(),
            },
        ],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read code files".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
            McpToolEntry {
                name: "write_file".to_string(),
                description: "Write test files".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}, "content": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 0,
            },
            CapabilityDef {
                name: "file_write".to_string(),
                rights: vec!["write".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 0,
            },
        ],
        orchestrator: None,
        builtin: true,
    }
}

/// Create the reviewer template
fn reviewer_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "reviewer".to_string(),
        name: "Code Review Agent".to_string(),
        description: "A code review agent that analyzes code quality, security, and best practices".to_string(),
        category: TemplateCategory::Development,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["review".into(), "quality".into(), "security".into(), "best-practices".into()],
        tools: vec![
            ToolDefinition {
                name: "review_diff".to_string(),
                description: "Review a code diff".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "diff": { "type": "string" },
                        "context": { "type": "string" }
                    },
                    "required": ["diff"]
                }),
                returns: Some("Review comments".to_string()),
            },
        ],
        prompts: vec![
            AgentPrompt {
                name: "reviewer".to_string(),
                system_prompt: "You are a senior code reviewer. Analyze code for quality, security vulnerabilities, performance issues, and adherence to best practices. Provide constructive feedback with specific suggestions for improvement. Be thorough but concise.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![
            SkillDefinition {
                name: "security_audit".to_string(),
                description: "Security-focused code review".to_string(),
                trigger: "/security".to_string(),
                content: "Perform a security-focused review checking for OWASP top 10, injection vulnerabilities, authentication issues, and data exposure risks.".to_string(),
            },
        ],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read code for review".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 2,
            },
        ],
        orchestrator: None,
        builtin: true,
    }
}

/// Create the security agent template
fn security_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "security".to_string(),
        name: "Security Agent".to_string(),
        description: "A security-focused agent for vulnerability scanning and threat analysis".to_string(),
        category: TemplateCategory::Security,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["security".into(), "vulnerabilities".into(), "audit".into(), "threats".into()],
        tools: vec![
            ToolDefinition {
                name: "scan_vulnerabilities".to_string(),
                description: "Scan code for security vulnerabilities".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "code": { "type": "string" },
                        "language": { "type": "string" }
                    },
                    "required": ["code"]
                }),
                returns: Some("Vulnerability report".to_string()),
            },
        ],
        prompts: vec![
            AgentPrompt {
                name: "security".to_string(),
                system_prompt: "You are a security expert. Identify vulnerabilities, analyze attack vectors, and recommend mitigations. Follow OWASP guidelines and security best practices. Be thorough and prioritize findings by severity.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![
            SkillDefinition {
                name: "threat_model".to_string(),
                description: "Create a threat model".to_string(),
                trigger: "/threatmodel".to_string(),
                content: "Analyze the system architecture and identify potential threats, attack vectors, and security controls.".to_string(),
            },
        ],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read code for analysis".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 0,
            },
        ],
        orchestrator: None,
        builtin: true,
    }
}

/// Create the swarm orchestrator template
fn swarm_orchestrator_template() -> GalleryTemplate {
    GalleryTemplate {
        id: "swarm-orchestrator".to_string(),
        name: "Swarm Orchestrator".to_string(),
        description: "Multi-agent swarm with coder, tester, and reviewer working together".to_string(),
        category: TemplateCategory::Orchestration,
        version: "1.0.0".to_string(),
        author: "RuVector".to_string(),
        tags: vec!["swarm".into(), "multi-agent".into(), "orchestration".into(), "team".into()],
        tools: vec![],
        prompts: vec![
            AgentPrompt {
                name: "queen".to_string(),
                system_prompt: "You are the swarm coordinator. Decompose complex tasks, delegate to specialized workers, and synthesize results. Maintain consistency and resolve conflicts.".to_string(),
                version: "1.0.0".to_string(),
            },
            AgentPrompt {
                name: "coder".to_string(),
                system_prompt: "You are a coder in the swarm. Implement features as directed by the coordinator.".to_string(),
                version: "1.0.0".to_string(),
            },
            AgentPrompt {
                name: "tester".to_string(),
                system_prompt: "You are a tester in the swarm. Write tests for code produced by coders.".to_string(),
                version: "1.0.0".to_string(),
            },
            AgentPrompt {
                name: "reviewer".to_string(),
                system_prompt: "You are a reviewer in the swarm. Review code and tests for quality.".to_string(),
                version: "1.0.0".to_string(),
            },
        ],
        skills: vec![],
        mcp_tools: vec![
            McpToolEntry {
                name: "read_file".to_string(),
                description: "Read files".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}}),
                group: Some("file".to_string()),
            },
            McpToolEntry {
                name: "write_file".to_string(),
                description: "Write files".to_string(),
                input_schema: serde_json::json!({"path": {"type": "string"}, "content": {"type": "string"}}),
                group: Some("file".to_string()),
            },
        ],
        capabilities: vec![
            CapabilityDef {
                name: "file_read".to_string(),
                rights: vec!["read".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 3,
            },
            CapabilityDef {
                name: "file_write".to_string(),
                rights: vec!["write".to_string()],
                scope: "sandbox".to_string(),
                delegation_depth: 2,
            },
        ],
        orchestrator: Some(OrchestratorConfig {
            topology: "hierarchical".to_string(),
            agents: vec![
                AgentNode {
                    id: "queen".to_string(),
                    agent_type: "coordinator".to_string(),
                    prompt_ref: "queen".to_string(),
                },
                AgentNode {
                    id: "coder-1".to_string(),
                    agent_type: "coder".to_string(),
                    prompt_ref: "coder".to_string(),
                },
                AgentNode {
                    id: "tester-1".to_string(),
                    agent_type: "tester".to_string(),
                    prompt_ref: "tester".to_string(),
                },
                AgentNode {
                    id: "reviewer-1".to_string(),
                    agent_type: "reviewer".to_string(),
                    prompt_ref: "reviewer".to_string(),
                },
            ],
            connections: vec![
                ("queen".to_string(), "coder-1".to_string()),
                ("queen".to_string(), "tester-1".to_string()),
                ("queen".to_string(), "reviewer-1".to_string()),
                ("coder-1".to_string(), "tester-1".to_string()),
                ("tester-1".to_string(), "reviewer-1".to_string()),
            ],
        }),
        builtin: true,
    }
}

/// Get all built-in templates
fn builtin_templates() -> Vec<GalleryTemplate> {
    vec![
        coder_template(),
        researcher_template(),
        tester_template(),
        reviewer_template(),
        security_template(),
        swarm_orchestrator_template(),
    ]
}

// ---------------------------------------------------------------------------
// WasmGallery — WASM-exported gallery manager
// ---------------------------------------------------------------------------

/// RVF App Gallery — browse, load, and configure agent templates.
///
/// # Example (JavaScript)
/// ```js
/// const gallery = new WasmGallery();
///
/// // List all templates
/// const templates = gallery.list();
///
/// // Search by tags
/// const results = gallery.search("security testing");
///
/// // Get template details
/// const template = gallery.get("coder");
///
/// // Load as RVF container
/// const rvfBytes = gallery.loadRvf("coder");
///
/// // Configure template
/// gallery.configure("coder", { maxTurns: 100 });
/// ```
#[wasm_bindgen]
pub struct WasmGallery {
    /// Built-in templates
    builtin: Vec<GalleryTemplate>,
    /// User-added custom templates
    custom: Vec<GalleryTemplate>,
    /// Currently loaded template
    active: Option<String>,
    /// Configuration overrides for active template
    config_overrides: serde_json::Value,
}

#[wasm_bindgen]
impl WasmGallery {
    /// Create a new gallery with built-in templates.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builtin: builtin_templates(),
            custom: Vec::new(),
            active: None,
            config_overrides: serde_json::json!({}),
        }
    }

    /// List all available templates.
    pub fn list(&self) -> Result<JsValue, JsValue> {
        let all: Vec<_> = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .map(|t| serde_json::json!({
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "version": t.version,
                "author": t.author,
                "tags": t.tags,
                "builtin": t.builtin,
            }))
            .collect();
        to_js_value(&all)
    }

    /// List templates by category.
    #[wasm_bindgen(js_name = listByCategory)]
    pub fn list_by_category(&self, category: &str) -> Result<JsValue, JsValue> {
        let cat: TemplateCategory = serde_json::from_str(&format!("\"{}\"", category))
            .map_err(|_| JsValue::from_str("invalid category"))?;

        let filtered: Vec<_> = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .filter(|t| t.category == cat)
            .map(|t| serde_json::json!({
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "tags": t.tags,
            }))
            .collect();
        to_js_value(&filtered)
    }

    /// Search templates by query (matches name, description, tags).
    pub fn search(&self, query: &str) -> Result<JsValue, JsValue> {
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<SearchResult> = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .filter_map(|t| {
                let mut score = 0.0f32;
                let name_lower = t.name.to_lowercase();
                let desc_lower = t.description.to_lowercase();

                for term in &terms {
                    if name_lower.contains(term) {
                        score += 0.4;
                    }
                    if desc_lower.contains(term) {
                        score += 0.3;
                    }
                    if t.tags.iter().any(|tag| tag.to_lowercase().contains(term)) {
                        score += 0.3;
                    }
                }

                if score > 0.0 {
                    Some(SearchResult {
                        id: t.id.clone(),
                        name: t.name.clone(),
                        description: t.description.clone(),
                        category: t.category,
                        tags: t.tags.clone(),
                        relevance: score.min(1.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by relevance
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());

        to_js_value(&results)
    }

    /// Get a template by ID.
    pub fn get(&self, id: &str) -> Result<JsValue, JsValue> {
        let template = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .find(|t| t.id == id)
            .ok_or_else(|| JsValue::from_str(&format!("template not found: {}", id)))?;

        to_js_value(template)
    }

    /// Load a template as an RVF container (returns Uint8Array).
    #[wasm_bindgen(js_name = loadRvf)]
    pub fn load_rvf(&self, id: &str) -> Result<js_sys::Uint8Array, JsValue> {
        let template = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .find(|t| t.id == id)
            .ok_or_else(|| JsValue::from_str(&format!("template not found: {}", id)))?;

        let bytes = template.to_rvf();
        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// Set a template as active for use.
    #[wasm_bindgen(js_name = setActive)]
    pub fn set_active(&mut self, id: &str) -> Result<(), JsValue> {
        // Verify template exists
        let exists = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .any(|t| t.id == id);

        if !exists {
            return Err(JsValue::from_str(&format!("template not found: {}", id)));
        }

        self.active = Some(id.to_string());
        self.config_overrides = serde_json::json!({});
        Ok(())
    }

    /// Get the currently active template ID.
    #[wasm_bindgen(js_name = getActive)]
    pub fn get_active(&self) -> Option<String> {
        self.active.clone()
    }

    /// Configure the active template with overrides.
    pub fn configure(&mut self, config_json: &str) -> Result<(), JsValue> {
        if self.active.is_none() {
            return Err(JsValue::from_str("no active template"));
        }

        let config: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid config JSON: {}", e)))?;

        self.config_overrides = config;
        Ok(())
    }

    /// Get configuration overrides for active template.
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.config_overrides)
    }

    /// Add a custom template to the gallery.
    #[wasm_bindgen(js_name = addCustom)]
    pub fn add_custom(&mut self, template_json: &str) -> Result<(), JsValue> {
        // Security: Check custom template limit
        if self.custom.len() >= MAX_CUSTOM_TEMPLATES {
            return Err(JsValue::from_str(&format!(
                "custom template limit ({}) exceeded",
                MAX_CUSTOM_TEMPLATES
            )));
        }

        let mut template: GalleryTemplate = serde_json::from_str(template_json)
            .map_err(|e| JsValue::from_str(&format!("invalid template JSON: {}", e)))?;

        // Security: Validate name length
        if template.name.len() > MAX_TEMPLATE_NAME_LENGTH {
            return Err(JsValue::from_str(&format!(
                "template name exceeds {} characters",
                MAX_TEMPLATE_NAME_LENGTH
            )));
        }

        // Security: Validate description length
        if template.description.len() > MAX_TEMPLATE_DESC_LENGTH {
            return Err(JsValue::from_str(&format!(
                "template description exceeds {} characters",
                MAX_TEMPLATE_DESC_LENGTH
            )));
        }

        // Check for duplicate ID
        let exists = self
            .builtin
            .iter()
            .chain(self.custom.iter())
            .any(|t| t.id == template.id);

        if exists {
            return Err(JsValue::from_str(&format!(
                "template ID already exists: {}",
                template.id
            )));
        }

        // Mark as custom (not builtin)
        template.builtin = false;
        template.category = TemplateCategory::Custom;

        self.custom.push(template);
        Ok(())
    }

    /// Remove a custom template by ID.
    #[wasm_bindgen(js_name = removeCustom)]
    pub fn remove_custom(&mut self, id: &str) -> Result<(), JsValue> {
        let idx = self
            .custom
            .iter()
            .position(|t| t.id == id)
            .ok_or_else(|| JsValue::from_str(&format!("custom template not found: {}", id)))?;

        self.custom.remove(idx);

        // Clear active if it was removed
        if self.active.as_deref() == Some(id) {
            self.active = None;
        }

        Ok(())
    }

    /// Get all categories with template counts.
    #[wasm_bindgen(js_name = getCategories)]
    pub fn get_categories(&self) -> Result<JsValue, JsValue> {
        use std::collections::HashMap;

        let mut counts: HashMap<String, usize> = HashMap::new();

        for template in self.builtin.iter().chain(self.custom.iter()) {
            let cat = serde_json::to_string(&template.category)
                .unwrap()
                .trim_matches('"')
                .to_string();
            *counts.entry(cat).or_insert(0) += 1;
        }

        to_js_value(&counts)
    }

    /// Get the number of templates in the gallery.
    pub fn count(&self) -> usize {
        self.builtin.len() + self.custom.len()
    }

    /// Export all custom templates as JSON.
    #[wasm_bindgen(js_name = exportCustom)]
    pub fn export_custom(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.custom)
    }

    /// Import custom templates from JSON.
    #[wasm_bindgen(js_name = importCustom)]
    pub fn import_custom(&mut self, templates_json: &str) -> Result<u32, JsValue> {
        let templates: Vec<GalleryTemplate> = serde_json::from_str(templates_json)
            .map_err(|e| JsValue::from_str(&format!("invalid JSON: {}", e)))?;

        let mut imported = 0u32;
        for mut template in templates {
            if self.custom.len() >= MAX_CUSTOM_TEMPLATES {
                break;
            }

            // Skip duplicates
            let exists = self
                .builtin
                .iter()
                .chain(self.custom.iter())
                .any(|t| t.id == template.id);

            if !exists {
                template.builtin = false;
                self.custom.push(template);
                imported += 1;
            }
        }

        Ok(imported)
    }
}

impl Default for WasmGallery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rust-only accessor methods (not exported to WASM)
// ---------------------------------------------------------------------------

impl WasmGallery {
    /// Get an iterator over all templates (builtin + custom).
    pub fn all_templates(&self) -> impl Iterator<Item = &GalleryTemplate> {
        self.builtin.iter().chain(self.custom.iter())
    }

    /// Get a reference to builtin templates.
    pub fn builtin_templates(&self) -> &[GalleryTemplate] {
        &self.builtin
    }

    /// Get a reference to custom templates.
    pub fn custom_templates(&self) -> &[GalleryTemplate] {
        &self.custom
    }

    /// Set the active template ID (internal use).
    pub fn set_active_id(&mut self, id: Option<String>) {
        self.active = id;
    }

    /// Get the active template ID reference.
    pub fn active_id(&self) -> Option<&String> {
        self.active.as_ref()
    }

    /// Set configuration overrides (internal use).
    pub fn set_config_overrides(&mut self, config: serde_json::Value) {
        self.config_overrides = config;
    }

    /// Get configuration overrides reference.
    pub fn config_overrides(&self) -> &serde_json::Value {
        &self.config_overrides
    }

    /// Find a template by ID.
    pub fn find_template(&self, id: &str) -> Option<&GalleryTemplate> {
        self.all_templates().find(|t| t.id == id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gallery_new() {
        let gallery = WasmGallery::new();
        assert_eq!(gallery.builtin.len(), 6);
        assert!(gallery.custom.is_empty());
        assert!(gallery.active.is_none());
    }

    #[test]
    fn test_gallery_count() {
        let gallery = WasmGallery::new();
        assert_eq!(gallery.count(), 6);
    }

    #[test]
    fn test_builtin_templates() {
        let templates = builtin_templates();
        assert!(!templates.is_empty());

        // Check coder template
        let coder = templates.iter().find(|t| t.id == "coder").unwrap();
        assert_eq!(coder.name, "Coder Agent");
        assert_eq!(coder.category, TemplateCategory::Development);
        assert!(coder.builtin);
    }

    #[test]
    fn test_template_to_rvf() {
        let coder = coder_template();
        let rvf_bytes = coder.to_rvf();

        // Should start with RVF magic
        assert_eq!(&rvf_bytes[0..4], b"RVF\x01");

        // Should end with SHA3-256 checksum (32 bytes)
        assert!(rvf_bytes.len() > 40);
    }

    #[test]
    fn test_search() {
        let gallery = WasmGallery::new();

        // Search should find coder template
        let results: Vec<SearchResult> = gallery
            .builtin
            .iter()
            .filter(|t| t.name.to_lowercase().contains("coder"))
            .map(|t| SearchResult {
                id: t.id.clone(),
                name: t.name.clone(),
                description: t.description.clone(),
                category: t.category,
                tags: t.tags.clone(),
                relevance: 1.0,
            })
            .collect();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "coder");
    }

    #[test]
    fn test_swarm_orchestrator_template() {
        let swarm = swarm_orchestrator_template();
        assert!(swarm.orchestrator.is_some());

        let orch = swarm.orchestrator.unwrap();
        assert_eq!(orch.topology, "hierarchical");
        assert_eq!(orch.agents.len(), 4);
        assert!(!orch.connections.is_empty());
    }

    #[test]
    fn test_add_custom_template() {
        let mut gallery = WasmGallery::new();
        let initial_count = gallery.count();

        // Add a valid custom template
        let custom = GalleryTemplate {
            id: "my-custom".to_string(),
            name: "My Custom Agent".to_string(),
            description: "A custom agent".to_string(),
            category: TemplateCategory::Custom,
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            tags: vec!["custom".to_string()],
            tools: vec![],
            prompts: vec![],
            skills: vec![],
            mcp_tools: vec![],
            capabilities: vec![],
            orchestrator: None,
            builtin: false,
        };

        let json = serde_json::to_string(&custom).unwrap();
        gallery.add_custom(&json).unwrap();

        assert_eq!(gallery.count(), initial_count + 1);
    }

    #[test]
    fn test_set_active() {
        let mut gallery = WasmGallery::new();

        // Verify coder template exists
        assert!(gallery.find_template("coder").is_some());

        // Set coder as active using internal accessor
        gallery.set_active_id(Some("coder".to_string()));
        assert_eq!(gallery.get_active(), Some("coder".to_string()));

        // Verify nonexistent template doesn't exist
        assert!(gallery.find_template("nonexistent").is_none());
    }

    #[test]
    fn test_category_enum() {
        let cat = TemplateCategory::Development;
        let json = serde_json::to_string(&cat).unwrap();
        assert_eq!(json, "\"development\"");

        let back: TemplateCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(back, TemplateCategory::Development);
    }
}
