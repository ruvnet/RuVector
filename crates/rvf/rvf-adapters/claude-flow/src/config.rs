//! Configuration for the Claude-Flow RVF adapter.  
  
use std::path::PathBuf;  
  
/// Configuration for the RVF-backed Claude-Flow memory store.  
#[derive(Clone, Debug)]  
pub struct ClaudeFlowConfig {  
    /// Directory where RVF data files are stored.  
    pub data_dir: PathBuf,  
    /// Vector embedding dimension (must match embeddings used by the agent).  
    pub dimension: u16,  
    /// Unique identifier for this agent.  
    pub agent_id: String,  
    /// Whether to enable WITNESS_SEG audit trails.  
    pub enable_witness: bool,  
    /// Optional memory namespace (e.g., session or context scope).  
    pub namespace: Option<String>,  
}  
  
impl ClaudeFlowConfig {  
    /// Create a new configuration with required parameters.  
    ///  
    /// Defaults: dimension=384, witness enabled, no namespace.  
    pub fn new(data_dir: impl Into<PathBuf>, agent_id: impl Into<String>) -> Self {  
        Self {  
            data_dir: data_dir.into(),  
            dimension: 384,  
            agent_id: agent_id.into(),  
            enable_witness: true,  
            namespace: None,  
        }  
    }  
  
    /// Set the embedding dimension.  
    pub fn with_dimension(mut self, dimension: u16) -> Self {  
        self.dimension = dimension;  
        self  
    }  
  
    /// Enable or disable witness audit trails.  
    pub fn with_witness(mut self, enable: bool) -> Self {  
        self.enable_witness = enable;  
        self  
    }  
  
    /// Set the memory namespace.  
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {  
        self.namespace = Some(namespace.into());  
        self  
    }  
  
    /// Return the path to the main RVF memory store file.  
    pub fn store_path(&self) -> PathBuf {  
        self.data_dir.join("memory.rvf")  
    }  
  
    /// Return the path to the witness chain file.  
    pub fn witness_path(&self) -> PathBuf {  
        self.data_dir.join("witness.bin")  
    }  
  
    /// Ensure the data directory exists.  
    pub fn ensure_dirs(&self) -> std::io::Result<()> {  
        std::fs::create_dir_all(&self.data_dir)  
    }  
  
    /// Validate the configuration.  
    pub fn validate(&self) -> Result<(), ConfigError> {  
        if self.dimension == 0 {  
            return Err(ConfigError::InvalidDimension);  
        }  
        if self.agent_id.is_empty() {  
            return Err(ConfigError::EmptyAgentId);  
        }  
        Ok(())  
    }  
}  
  
/// Errors specific to adapter configuration.  
#[derive(Clone, Debug, PartialEq, Eq)]  
pub enum ConfigError {  
    /// Dimension must be > 0.  
    InvalidDimension,  
    /// Agent ID must not be empty.  
    EmptyAgentId,  
}  
  
impl std::fmt::Display for ConfigError {  
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {  
        match self {  
            Self::InvalidDimension => write!(f, "vector dimension must be > 0"),  
            Self::EmptyAgentId => write!(f, "agent_id must not be empty"),  
        }  
    }  
}  
  
impl std::error::Error for ConfigError {}  
  
#[cfg(test)]  
mod tests {  
    use super::*;  
  
    #[test]  
    fn config_defaults() {  
        let cfg = ClaudeFlowConfig::new("/tmp/test", "agent-1");  
        assert_eq!(cfg.dimension, 384);  
        assert!(cfg.enable_witness);  
        assert!(cfg.namespace.is_none());  
        assert_eq!(cfg.agent_id, "agent-1");  
    }  
  
    #[test]  
    fn config_paths() {  
        let cfg = ClaudeFlowConfig::new("/data/memory", "a1");  
        assert_eq!(cfg.store_path(), Path::new("/data/memory/memory.rvf"));  
        assert_eq!(cfg.witness_path(), Path::new("/data/memory/witness.bin"));  
    }  
  
    #[test]  
    fn validate_zero_dimension() {  
        let cfg = ClaudeFlowConfig::new("/tmp", "a1").with_dimension(0);  
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidDimension));  
    }  
  
    #[test]  
    fn validate_empty_agent_id() {  
        let cfg = ClaudeFlowConfig::new("/tmp", "");  
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyAgentId));  
    }  
  
    #[test]  
    fn validate_ok() {  
        let cfg = ClaudeFlowConfig::new("/tmp", "agent-1").with_dimension(64);  
        assert!(cfg.validate().is_ok());  
    }  
  
    #[test]  
    fn builder_methods() {  
        let cfg = ClaudeFlowConfig::new("/tmp", "a1")  
            .with_dimension(128)  
            .with_witness(false)  
            .with_namespace("session-42");  
        assert_eq!(cfg.dimension, 128);  
        assert!(!cfg.enable_witness);  
        assert_eq!(cfg.namespace.as_deref(), Some("session-42"));  
    }  
}
