//! Configuration management for Leviathan CLI
//!
//! Handles loading and validation of leviathan.toml configuration files
//! with environment variable overrides.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Main Leviathan configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeviathanConfig {
    /// Data directory for storing models and caches
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Audit configuration
    #[serde(default)]
    pub audit: AuditConfig,

    /// Swarm configuration
    #[serde(default)]
    pub swarm: SwarmConfig,

    /// Agent configuration
    #[serde(default)]
    pub agent: AgentConfig,

    /// Training configuration
    #[serde(default)]
    pub training: TrainingConfig,

    /// UI configuration
    #[serde(default)]
    pub ui: UiConfig,
}

impl Default for LeviathanConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            audit: AuditConfig::default(),
            swarm: SwarmConfig::default(),
            agent: AgentConfig::default(),
            training: TrainingConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

impl LeviathanConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let mut config: Self = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        // Apply environment variable overrides
        config.apply_env_overrides();

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Load configuration from default locations
    pub fn load_default() -> Result<Self> {
        // Try current directory first
        let local_config = PathBuf::from("leviathan.toml");
        if local_config.exists() {
            return Self::load(&local_config);
        }

        // Try home directory
        if let Some(home_dir) = dirs::home_dir() {
            let home_config = home_dir.join(".config/leviathan/config.toml");
            if home_config.exists() {
                return Self::load(&home_config);
            }
        }

        // Fall back to default configuration
        Ok(Self::default())
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) {
        if let Ok(data_dir) = std::env::var("LEVIATHAN_DATA_DIR") {
            self.data_dir = PathBuf::from(data_dir);
        }

        if let Ok(audit_enabled) = std::env::var("LEVIATHAN_AUDIT_ENABLED") {
            self.audit.enabled = audit_enabled.parse().unwrap_or(true);
        }

        if let Ok(swarm_topology) = std::env::var("LEVIATHAN_SWARM_TOPOLOGY") {
            self.swarm.default_topology = swarm_topology;
        }
    }

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        // Ensure data directory is writable
        if let Some(parent) = self.data_dir.parent() {
            if parent.exists() && parent.metadata()?.permissions().readonly() {
                anyhow::bail!(
                    "Data directory parent is read-only: {}",
                    parent.display()
                );
            }
        }

        // Validate audit log path
        if self.audit.enabled {
            if let Some(parent) = self.audit.log_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("Failed to create audit log directory: {}", parent.display()))?;
                }
            }
        }

        Ok(())
    }

    /// Save configuration to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;

        std::fs::write(path, contents)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        Ok(())
    }
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Audit log file path
    #[serde(default = "default_audit_log_path")]
    pub log_path: PathBuf,

    /// Maximum audit log size in MB
    #[serde(default = "default_max_log_size")]
    pub max_size_mb: usize,

    /// Enable cryptographic verification
    #[serde(default = "default_true")]
    pub crypto_verify: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_path: default_audit_log_path(),
            max_size_mb: default_max_log_size(),
            crypto_verify: true,
        }
    }
}

/// Swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Default topology (mesh, hierarchical, adaptive)
    #[serde(default = "default_topology")]
    pub default_topology: String,

    /// Maximum number of agents
    #[serde(default = "default_max_agents")]
    pub max_agents: usize,

    /// Task timeout in seconds
    #[serde(default = "default_task_timeout")]
    pub task_timeout_secs: u64,

    /// Enable auto-healing
    #[serde(default = "default_true")]
    pub auto_heal: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            default_topology: default_topology(),
            max_agents: default_max_agents(),
            task_timeout_secs: default_task_timeout(),
            auto_heal: true,
        }
    }
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default agent type
    #[serde(default = "default_agent_type")]
    pub default_type: String,

    /// Agent memory limit in MB
    #[serde(default = "default_agent_memory_mb")]
    pub memory_limit_mb: usize,

    /// Enable agent persistence
    #[serde(default = "default_true")]
    pub persist_state: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_type: default_agent_type(),
            memory_limit_mb: default_agent_memory_mb(),
            persist_state: true,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size for training
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Learning rate
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Maximum epochs
    #[serde(default = "default_max_epochs")]
    pub max_epochs: usize,

    /// Enable GPU acceleration
    #[serde(default = "default_false")]
    pub use_gpu: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            max_epochs: default_max_epochs(),
            use_gpu: false,
        }
    }
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Refresh rate in milliseconds
    #[serde(default = "default_refresh_rate")]
    pub refresh_rate_ms: u64,

    /// Enable colors
    #[serde(default = "default_true")]
    pub colors: bool,

    /// Show help on startup
    #[serde(default = "default_true")]
    pub show_help: bool,

    /// Maximum history items
    #[serde(default = "default_max_history")]
    pub max_history: usize,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: default_refresh_rate(),
            colors: true,
            show_help: true,
            max_history: default_max_history(),
        }
    }
}

// Default value functions
fn default_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("leviathan")
}

fn default_audit_log_path() -> PathBuf {
    default_data_dir().join("audit.log")
}

fn default_max_log_size() -> usize {
    100
}

fn default_topology() -> String {
    "mesh".to_string()
}

fn default_max_agents() -> usize {
    10
}

fn default_task_timeout() -> u64 {
    300
}

fn default_agent_type() -> String {
    "general".to_string()
}

fn default_agent_memory_mb() -> usize {
    512
}

fn default_batch_size() -> usize {
    32
}

fn default_learning_rate() -> f32 {
    0.001
}

fn default_max_epochs() -> usize {
    100
}

fn default_refresh_rate() -> u64 {
    250
}

fn default_max_history() -> usize {
    100
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = LeviathanConfig::default();
        assert!(config.audit.enabled);
        assert_eq!(config.swarm.default_topology, "mesh");
    }

    #[test]
    fn test_save_and_load() -> Result<()> {
        let config = LeviathanConfig::default();
        let temp_file = NamedTempFile::new()?;

        config.save(temp_file.path())?;
        let loaded = LeviathanConfig::load(temp_file.path())?;

        assert_eq!(config.swarm.max_agents, loaded.swarm.max_agents);
        Ok(())
    }

    #[test]
    fn test_env_override() {
        std::env::set_var("LEVIATHAN_DATA_DIR", "/tmp/test");
        let mut config = LeviathanConfig::default();
        config.apply_env_overrides();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/test"));
        std::env::remove_var("LEVIATHAN_DATA_DIR");
    }
}
