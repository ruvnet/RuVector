//! Configuration types for rvAgent.
//!
//! `RvAgentConfig` is the top-level configuration, renamed from `DeepAgentConfig`.

use serde::{Deserialize, Serialize};

use crate::prompt::BASE_AGENT_PROMPT;
use crate::rvf_bridge::RvfBridgeConfig;

// ---------------------------------------------------------------------------
// Security policy (ADR-103 C1 — virtual_mode default true)
// ---------------------------------------------------------------------------

/// Sensitive environment variable patterns that must be stripped
/// before passing env to child processes (ADR-103 C2).
pub const SENSITIVE_ENV_PATTERNS: &[&str] = &[
    "SECRET",
    "KEY",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "AWS_",
    "AZURE_",
    "GCP_",
    "DATABASE_URL",
    "PRIVATE",
];

/// Security policy controlling sandbox, allowlists, and trust settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// When true, filesystem operations run in a virtual sandbox (default: true per ADR-103 C1).
    #[serde(default = "default_true")]
    pub virtual_mode: bool,

    /// Optional allowlist of shell commands permitted for execution.
    #[serde(default)]
    pub command_allowlist: Vec<String>,

    /// Env variable name patterns considered sensitive (stripped before child processes).
    #[serde(default = "default_sensitive_env_patterns")]
    pub sensitive_env_patterns: Vec<String>,

    /// Maximum response length in bytes from sub-agents (default: 100 KB per ADR-103 C8).
    #[serde(default = "default_max_response_length")]
    pub max_response_length: usize,

    /// Whether to trust AGENTS.md files found in the working directory.
    #[serde(default)]
    pub trust_agents_md: bool,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            virtual_mode: true,
            command_allowlist: Vec::new(),
            sensitive_env_patterns: default_sensitive_env_patterns(),
            max_response_length: default_max_response_length(),
            trust_agents_md: false,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_sensitive_env_patterns() -> Vec<String> {
    SENSITIVE_ENV_PATTERNS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn default_max_response_length() -> usize {
    100 * 1024 // 100 KB
}

// ---------------------------------------------------------------------------
// Resource budget (ADR-103 B4)
// ---------------------------------------------------------------------------

/// Resource budget enforcement limits per agent invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum wall-clock seconds for the agent run.
    #[serde(default)]
    pub max_time_secs: u32,

    /// Maximum total tokens (input + output).
    #[serde(default)]
    pub max_tokens: u64,

    /// Maximum cost in micro-dollars (1 USD = 1_000_000).
    #[serde(default)]
    pub max_cost_microdollars: u64,

    /// Maximum number of tool calls.
    #[serde(default)]
    pub max_tool_calls: u32,

    /// Maximum external (non-sandbox) writes.
    #[serde(default)]
    pub max_external_writes: u32,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        Self {
            max_time_secs: 300,
            max_tokens: 200_000,
            max_cost_microdollars: 5_000_000, // $5
            max_tool_calls: 500,
            max_external_writes: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-configs
// ---------------------------------------------------------------------------

/// Configuration for a middleware entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiddlewareConfig {
    /// Middleware identifier (e.g. "filesystem", "memory", "skills").
    pub name: String,
    /// Middleware-specific settings.
    #[serde(default)]
    pub settings: serde_json::Value,
}

/// Configuration for a tool registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool name.
    pub name: String,
    /// Tool-specific settings.
    #[serde(default)]
    pub settings: serde_json::Value,
}

/// Backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type identifier (e.g. "filesystem", "local_shell", "composite").
    #[serde(default = "default_backend_type")]
    pub backend_type: String,
    /// Working directory for filesystem/shell backends.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Extra backend-specific settings.
    #[serde(default)]
    pub settings: serde_json::Value,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: default_backend_type(),
            cwd: None,
            settings: serde_json::Value::Null,
        }
    }
}

fn default_backend_type() -> String {
    "local_shell".into()
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Default model identifier.
pub const DEFAULT_MODEL: &str = "anthropic:claude-sonnet-4-20250514";

/// Top-level agent configuration.
///
/// Renamed from `DeepAgentConfig` to `RvAgentConfig` for the RuVector rebrand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvAgentConfig {
    /// Model identifier in "provider:model" format.
    #[serde(default = "default_model")]
    pub model: String,

    /// Optional agent name for logging/tracing.
    #[serde(default)]
    pub name: Option<String>,

    /// System instructions / base prompt.
    #[serde(default = "default_instructions")]
    pub instructions: String,

    /// Ordered middleware pipeline configuration.
    #[serde(default)]
    pub middleware: Vec<MiddlewareConfig>,

    /// Additional tool registrations.
    #[serde(default)]
    pub tools: Vec<ToolConfig>,

    /// Backend configuration.
    #[serde(default)]
    pub backend: BackendConfig,

    /// Security policy (virtual_mode defaults true per ADR-103 C1).
    #[serde(default)]
    pub security_policy: SecurityPolicy,

    /// Optional resource budget for cost/time/token limits.
    #[serde(default)]
    pub resource_budget: Option<ResourceBudget>,

    /// RVF bridge configuration (ADR-106 integration).
    #[serde(default)]
    pub rvf_bridge: RvfBridgeConfig,
}

impl Default for RvAgentConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            name: None,
            instructions: default_instructions(),
            middleware: Vec::new(),
            tools: Vec::new(),
            backend: BackendConfig::default(),
            security_policy: SecurityPolicy::default(),
            resource_budget: None,
            rvf_bridge: RvfBridgeConfig::default(),
        }
    }
}

fn default_model() -> String {
    DEFAULT_MODEL.to_string()
}

fn default_instructions() -> String {
    BASE_AGENT_PROMPT.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = RvAgentConfig::default();
        assert_eq!(cfg.model, DEFAULT_MODEL);
        assert!(cfg.name.is_none());
        assert!(!cfg.instructions.is_empty());
        assert!(cfg.middleware.is_empty());
        assert!(cfg.tools.is_empty());
        assert_eq!(cfg.backend.backend_type, "local_shell");
        assert!(cfg.security_policy.virtual_mode);
        assert!(cfg.resource_budget.is_none());
    }

    #[test]
    fn test_security_policy_defaults() {
        let sp = SecurityPolicy::default();
        assert!(sp.virtual_mode);
        assert!(sp.command_allowlist.is_empty());
        assert!(!sp.sensitive_env_patterns.is_empty());
        assert!(sp.sensitive_env_patterns.contains(&"SECRET".to_string()));
        assert_eq!(sp.max_response_length, 100 * 1024);
        assert!(!sp.trust_agents_md);
    }

    #[test]
    fn test_resource_budget_defaults() {
        let rb = ResourceBudget::default();
        assert_eq!(rb.max_time_secs, 300);
        assert!(rb.max_tokens > 0);
        assert!(rb.max_cost_microdollars > 0);
        assert!(rb.max_tool_calls > 0);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let cfg = RvAgentConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RvAgentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, cfg.model);
        assert_eq!(back.security_policy.virtual_mode, true);
    }

    #[test]
    fn test_config_from_partial_json() {
        let json = r#"{"model": "openai:gpt-4o"}"#;
        let cfg: RvAgentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.model, "openai:gpt-4o");
        // Everything else should get defaults.
        assert!(cfg.security_policy.virtual_mode);
        assert!(!cfg.instructions.is_empty());
    }

    #[test]
    fn test_sensitive_env_patterns() {
        assert!(SENSITIVE_ENV_PATTERNS.contains(&"AWS_"));
        assert!(SENSITIVE_ENV_PATTERNS.contains(&"TOKEN"));
        assert!(SENSITIVE_ENV_PATTERNS.len() >= 10);
    }
}
