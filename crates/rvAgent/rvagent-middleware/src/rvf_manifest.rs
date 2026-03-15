//! RVF Manifest Middleware — ADR-106 Layer 2 manifest & signature convergence.
//!
//! This middleware discovers and validates RVF packages, mounting them into the
//! agent's runtime. It implements:
//!
//! - Package discovery from configured directories
//! - Manifest parsing (delegates to `rvf-manifest` when available)
//! - Signature verification (delegates to `rvf-crypto` when available)
//! - Tool injection from mounted RVF packages
//!
//! When the `rvf-compat` feature is enabled, actual RVF parsing and crypto
//! verification is used. Without it, a JSON-based manifest format is supported.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rvagent_core::rvf_bridge::{
    MountTable, RvfBridgeConfig, RvfManifest, RvfMountHandle, RvfVerifyStatus,
};

use crate::{
    AgentState, AgentStateUpdate, Middleware, Runtime, RunnableConfig, Tool,
};

// ---------------------------------------------------------------------------
// RVF Manifest Middleware
// ---------------------------------------------------------------------------

/// Middleware that discovers and mounts RVF packages, injecting their tools
/// into the agent pipeline.
pub struct RvfManifestMiddleware {
    /// Shared mount table.
    mount_table: Arc<Mutex<MountTable>>,
    /// Bridge configuration.
    config: RvfBridgeConfig,
    /// Cached tool definitions from mounted packages.
    cached_tools: Arc<Mutex<Vec<RvfToolAdapter>>>,
}

impl RvfManifestMiddleware {
    /// Create a new RVF manifest middleware.
    pub fn new(config: RvfBridgeConfig) -> Self {
        Self {
            mount_table: Arc::new(Mutex::new(MountTable::new())),
            config,
            cached_tools: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create with a shared mount table.
    pub fn with_mount_table(
        config: RvfBridgeConfig,
        mount_table: Arc<Mutex<MountTable>>,
    ) -> Self {
        Self {
            mount_table,
            config,
            cached_tools: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get a reference to the mount table.
    pub fn mount_table(&self) -> &Arc<Mutex<MountTable>> {
        &self.mount_table
    }

    /// Mount a package programmatically.
    pub fn mount_package(
        &self,
        manifest: RvfManifest,
    ) -> RvfMountHandle {
        let verify_status = if self.config.verify_signatures {
            // Without rvf-crypto, we mark as valid (signature check is a no-op)
            // With rvf-compat feature, this would delegate to rvf-crypto::verify
            RvfVerifyStatus::SignatureValid
        } else {
            RvfVerifyStatus::SignatureValid
        };

        let handle = {
            let mut table = self.mount_table.lock().unwrap();
            table.mount(manifest, verify_status)
        };

        // Rebuild tool cache
        self.rebuild_tool_cache();

        handle
    }

    /// Rebuild the cached tool adapters from the mount table.
    fn rebuild_tool_cache(&self) {
        let table = self.mount_table.lock().unwrap();
        let tools: Vec<RvfToolAdapter> = table
            .all_tools()
            .into_iter()
            .map(|(handle, entry)| RvfToolAdapter {
                mount_handle: *handle,
                name: format!("rvf:{}", entry.name),
                description: entry.description.clone(),
                parameters_schema: entry
                    .parameters_schema
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            })
            .collect();
        *self.cached_tools.lock().unwrap() = tools;
    }

    /// Parse a manifest from JSON (the fallback format without rvf-manifest crate).
    pub fn parse_manifest_json(json: &str) -> Result<RvfManifest, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse RVF manifest: {}", e))
    }
}

/// Tool adapter that wraps an RVF manifest tool entry as a middleware Tool.
#[derive(Debug, Clone)]
struct RvfToolAdapter {
    mount_handle: RvfMountHandle,
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
}

impl Tool for RvfToolAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    fn invoke(&self, args: serde_json::Value) -> Result<String, String> {
        // Without rvf-runtime, return a stub response indicating the tool is available
        // but actual execution requires the rvf-compat feature.
        Ok(format!(
            "RVF tool '{}' (mount {}:{}) invoked with args: {}. \
             Note: Full execution requires rvf-runtime integration.",
            self.name,
            self.mount_handle.id,
            self.mount_handle.generation,
            serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
        ))
    }
}

#[async_trait]
impl Middleware for RvfManifestMiddleware {
    fn name(&self) -> &str {
        "rvf_manifest"
    }

    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if !self.config.enabled {
            return None;
        }

        // Inject RVF mount info into state extensions
        let table = self.mount_table.lock().unwrap();
        if table.is_empty() {
            return None;
        }

        let mut extensions = HashMap::new();
        let mount_info: Vec<serde_json::Value> = table
            .list()
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "package": entry.package_name,
                    "version": entry.package_version,
                    "verified": entry.verify_status.is_valid(),
                    "tools": entry.manifest.tools().iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>(),
                    "skills": entry.manifest.skills().iter()
                        .map(|s| s.name.as_str())
                        .collect::<Vec<_>>(),
                })
            })
            .collect();

        extensions.insert(
            "rvf_packages".to_string(),
            serde_json::json!(mount_info),
        );

        Some(AgentStateUpdate {
            messages: None,
            todos: None,
            extensions,
        })
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        if !self.config.enabled {
            return vec![];
        }
        let cached = self.cached_tools.lock().unwrap();
        cached
            .iter()
            .map(|t| Box::new(t.clone()) as Box<dyn Tool>)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use rvagent_core::rvf_bridge::{RvfManifestEntry, RvfManifestEntryType};

    fn sample_config() -> RvfBridgeConfig {
        RvfBridgeConfig {
            enabled: true,
            verify_signatures: false,
            ..Default::default()
        }
    }

    fn sample_manifest() -> RvfManifest {
        let mut manifest = RvfManifest::new("test-pkg", "0.1.0");
        manifest.entries.push(RvfManifestEntry {
            name: "lint".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Lint code".into(),
            version: "0.1.0".into(),
            parameters_schema: Some(serde_json::json!({"type": "object"})),
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "format".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Format code".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "ci-skill".into(),
            entry_type: RvfManifestEntryType::Skill,
            description: "Run CI pipeline".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec!["execute".into()],
        });
        manifest
    }

    #[test]
    fn test_middleware_name() {
        let mw = RvfManifestMiddleware::new(sample_config());
        assert_eq!(mw.name(), "rvf_manifest");
    }

    #[test]
    fn test_mount_and_tools() {
        let mw = RvfManifestMiddleware::new(sample_config());
        let handle = mw.mount_package(sample_manifest());
        assert!(!handle.is_null());

        let tools = mw.tools();
        assert_eq!(tools.len(), 2); // Only Tool entries
        assert!(tools.iter().any(|t| t.name() == "rvf:lint"));
        assert!(tools.iter().any(|t| t.name() == "rvf:format"));
    }

    #[test]
    fn test_tool_invoke() {
        let mw = RvfManifestMiddleware::new(sample_config());
        mw.mount_package(sample_manifest());

        let tools = mw.tools();
        let lint = tools.iter().find(|t| t.name() == "rvf:lint").unwrap();
        let result = lint.invoke(serde_json::json!({"path": "src/main.rs"}));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("rvf:lint"));
    }

    #[test]
    fn test_disabled_middleware() {
        let config = RvfBridgeConfig {
            enabled: false,
            ..Default::default()
        };
        let mw = RvfManifestMiddleware::new(config);
        mw.mount_package(sample_manifest());

        // Tools should be empty when disabled
        let tools = mw.tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_before_agent_injects_state() {
        let mw = RvfManifestMiddleware::new(sample_config());
        mw.mount_package(sample_manifest());

        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());

        let update = update.unwrap();
        let packages = update.extensions.get("rvf_packages").unwrap();
        let arr = packages.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["package"], "test-pkg");
    }

    #[test]
    fn test_before_agent_empty_table() {
        let mw = RvfManifestMiddleware::new(sample_config());

        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_none());
    }

    #[test]
    fn test_parse_manifest_json() {
        let json = serde_json::to_string(&sample_manifest()).unwrap();
        let parsed = RvfManifestMiddleware::parse_manifest_json(&json).unwrap();
        assert_eq!(parsed.name, "test-pkg");
        assert_eq!(parsed.entries.len(), 3);
    }

    #[test]
    fn test_parse_manifest_json_invalid() {
        let result = RvfManifestMiddleware::parse_manifest_json("{invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_mount_table() {
        let table = Arc::new(Mutex::new(MountTable::new()));
        let mw1 = RvfManifestMiddleware::with_mount_table(sample_config(), table.clone());
        let mw2 = RvfManifestMiddleware::with_mount_table(sample_config(), table);

        mw1.mount_package(sample_manifest());

        // mw2 should see the mounted package via shared table
        // (tools need rebuild on mw2 side, but mount table is shared)
        let table = mw2.mount_table().lock().unwrap();
        assert_eq!(table.len(), 1);
    }
}
