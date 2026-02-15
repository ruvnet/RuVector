//! Error types for the Claude-Flow RVF adapter.  
  
use std::fmt;  
  
use rvf_types::RvfError;  
  
/// Errors that can occur in Claude-Flow adapter operations.  
#[derive(Debug, Clone)]  
pub enum ClaudeFlowError {  
    /// An error originating from the RVF runtime or types layer.  
    Rvf(RvfError),  
    /// An I/O error described by a message string.  
    Io(String),  
    /// Configuration validation error.  
    Config(ConfigError),  
    /// Witness/audit trail error.  
    Witness(RvfError),  
    /// Embedding dimension mismatch.  
    DimensionMismatch {  
        /// Expected dimension.  
        expected: usize,  
        /// Provided dimension.  
        got: usize,  
    },  
}  
  
impl fmt::Display for ClaudeFlowError {  
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {  
        match self {  
            Self::Rvf(e) => write!(f, "RVF error: {e}"),  
            Self::Io(msg) => write!(f, "I/O error: {msg}"),  
            Self::Config(e) => write!(f, "config error: {e}"),  
            Self::Witness(e) => write!(f, "witness error: {e}"),  
            Self::DimensionMismatch { expected, got } => {  
                write!(f, "dimension mismatch: expected {expected}, got {got}")  
            }  
        }  
    }  
}  
  
impl std::error::Error for ClaudeFlowError {}  
  
impl From<RvfError> for ClaudeFlowError {  
    fn from(e: RvfError) -> Self {  
        Self::Rvf(e)  
    }  
}  
  
impl From<std::io::Error> for ClaudeFlowError {  
    fn from(e: std::io::Error) -> Self {  
        Self::Io(e.to_string())  
    }  
}  
  
impl From<ConfigError> for ClaudeFlowError {  
    fn from(e: ConfigError) -> Self {  
        Self::Config(e)  
    }  
}  
  
/// Configuration-specific errors.  
#[derive(Debug, Clone, PartialEq, Eq)]  
pub enum ConfigError {  
    /// Dimension must be > 0.  
    InvalidDimension,  
    /// Agent ID must not be empty.  
    EmptyAgentId,  
}  
  
impl fmt::Display for ConfigError {  
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {  
        match self {  
            Self::InvalidDimension => write!(f, "vector dimension must be > 0"),  
            Self::EmptyAgentId => write!(f, "agent_id must not be empty"),  
        }  
    }  
}  
  
impl std::error::Error for ConfigError {}  
  
/// Convenience alias used throughout the claude-flow adapter.  
pub type Result<T> = std::result::Result<T, ClaudeFlowError>;  
  
#[cfg(test)]  
mod tests {  
    use super::*;  
    use rvf_types::ErrorCode;  
  
    #[test]  
    fn display_rvf_variant() {  
        let err = ClaudeFlowError::Rvf(RvfError::Code(ErrorCode::DimensionMismatch));  
        let msg = format!("{err}");  
        assert!(msg.contains("RVF error"));  
    }  
  
    #[test]  
    fn display_io_variant() {  
        let err = ClaudeFlowError::Io("file not found".into());  
        let msg = format!("{err}");  
        assert!(msg.contains("I/O error"));  
        assert!(msg.contains("file not found"));  
    }  
  
    #[test]  
    fn display_config_variant() {  
        let err = ClaudeFlowError::Config(ConfigError::InvalidDimension);  
        let msg = format!("{err}");  
        assert!(msg.contains("config error"));  
        assert!(msg.contains("dimension must be > 0"));  
    }  
  
    #[test]  
    fn display_witness_variant() {  
        let err = ClaudeFlowError::Witness(RvfError::Code(ErrorCode::InvalidChecksum));  
        let msg = format!("{err}");  
        assert!(msg.contains("witness error"));  
    }  
  
    #[test]  
    fn display_dimension_mismatch() {  
        let err = ClaudeFlowError::DimensionMismatch {  
            expected: 384,  
            got: 128,  
        };  
        let msg = format!("{err}");  
        assert!(msg.contains("expected 384"));  
        assert!(msg.contains("got 128"));  
    }  
  
    #[test]  
    fn from_rvf_error() {  
        let rvf = RvfError::Code(ErrorCode::FsyncFailed);  
        let err: ClaudeFlowError = rvf.into();  
        matches!(err, ClaudeFlowError::Rvf(_));  
    }  
  
    #[test]  
    fn from_io_error() {  
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");  
        let err: ClaudeFlowError = io_err.into();  
        match err {  
            ClaudeFlowError::Io(msg) => assert!(msg.contains("missing")),  
            _ => panic!("expected Io variant"),  
        }  
    }  
  
    #[test]  
    fn from_config_error() {  
        let cfg_err = ConfigError::EmptyAgentId;  
        let err: ClaudeFlowError = cfg_err.into();  
        match err {  
            ClaudeFlowError::Config(_) => {} // expected  
            _ => panic!("expected Config variant"),  
        }  
    }  
}
