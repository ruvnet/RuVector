//! Integration tests for RvAgentConfig, SecurityPolicy, and ResourceBudget.
//!
//! These tests exercise the public configuration API from `rvagent_core::config`.

use rvagent_core::config::{
    ResourceBudget, RvAgentConfig, SecurityPolicy, SENSITIVE_ENV_PATTERNS,
};

/// Default config must have virtual_mode=true (ADR-103 C1).
#[test]
fn test_default_config_has_virtual_mode_true() {
    let cfg = RvAgentConfig::default();
    assert!(
        cfg.security_policy.virtual_mode,
        "virtual_mode must default to true per ADR-103 C1"
    );
}

/// SecurityPolicy defaults should match the ADR-103 C2 requirements.
#[test]
fn test_security_policy_defaults() {
    let sp = SecurityPolicy::default();

    // virtual_mode true (ADR-103 C1)
    assert!(sp.virtual_mode);

    // command_allowlist starts empty (no commands allowed by default)
    assert!(sp.command_allowlist.is_empty());

    // sensitive_env_patterns contains all required patterns from ADR-103 C2
    for pattern in SENSITIVE_ENV_PATTERNS {
        assert!(
            sp.sensitive_env_patterns
                .iter()
                .any(|p| p == pattern),
            "missing sensitive env pattern: {}",
            pattern
        );
    }

    // max_response_length 100 KB (ADR-103 C8)
    assert_eq!(sp.max_response_length, 100 * 1024);

    // trust_agents_md defaults to false (ADR-103 C4)
    assert!(!sp.trust_agents_md);
}

/// ResourceBudget defaults should have reasonable non-zero values.
#[test]
fn test_resource_budget_enforcement() {
    let rb = ResourceBudget::default();

    assert!(rb.max_time_secs > 0, "max_time_secs should be positive");
    assert!(rb.max_tokens > 0, "max_tokens should be positive");
    assert!(
        rb.max_cost_microdollars > 0,
        "max_cost_microdollars should be positive"
    );
    assert!(
        rb.max_tool_calls > 0,
        "max_tool_calls should be positive"
    );
    assert!(
        rb.max_external_writes > 0,
        "max_external_writes should be positive"
    );

    // Specific defaults from ADR-103 B4
    assert_eq!(rb.max_time_secs, 300);
    assert_eq!(rb.max_tokens, 200_000);
    assert_eq!(rb.max_cost_microdollars, 5_000_000);
    assert_eq!(rb.max_tool_calls, 500);
    assert_eq!(rb.max_external_writes, 100);
}

/// Config should survive a JSON serialization round-trip with defaults intact.
#[test]
fn test_config_serialization() {
    let cfg = RvAgentConfig::default();

    let json = serde_json::to_string_pretty(&cfg).unwrap();
    let restored: RvAgentConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.model, cfg.model);
    assert_eq!(
        restored.security_policy.virtual_mode,
        cfg.security_policy.virtual_mode
    );
    assert_eq!(restored.backend.backend_type, cfg.backend.backend_type);
    assert_eq!(restored.name, cfg.name);

    // Partial JSON should fill in defaults.
    let partial = r#"{"model": "openai:gpt-4o"}"#;
    let partial_cfg: RvAgentConfig = serde_json::from_str(partial).unwrap();
    assert_eq!(partial_cfg.model, "openai:gpt-4o");
    assert!(partial_cfg.security_policy.virtual_mode);
    assert!(!partial_cfg.instructions.is_empty());

    // SecurityPolicy round-trip
    let sp = SecurityPolicy::default();
    let sp_json = serde_json::to_string(&sp).unwrap();
    let sp_back: SecurityPolicy = serde_json::from_str(&sp_json).unwrap();
    assert_eq!(sp_back.virtual_mode, sp.virtual_mode);
    assert_eq!(
        sp_back.sensitive_env_patterns.len(),
        sp.sensitive_env_patterns.len()
    );

    // ResourceBudget round-trip
    let rb = ResourceBudget::default();
    let rb_json = serde_json::to_string(&rb).unwrap();
    let rb_back: ResourceBudget = serde_json::from_str(&rb_json).unwrap();
    assert_eq!(rb_back.max_time_secs, rb.max_time_secs);
    assert_eq!(rb_back.max_tokens, rb.max_tokens);
}
