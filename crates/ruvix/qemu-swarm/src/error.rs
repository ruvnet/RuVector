//! Error types for the QEMU swarm simulation system.

use std::path::PathBuf;
use thiserror::Error;

/// Result type for swarm operations.
pub type SwarmResult<T> = Result<T, SwarmError>;

/// Errors that can occur during swarm operations.
#[derive(Error, Debug)]
pub enum SwarmError {
    /// QEMU process failed to start.
    #[error("Failed to spawn QEMU process: {0}")]
    QemuSpawnFailed(String),

    /// QEMU process exited unexpectedly.
    #[error("QEMU process exited with code {code}: {message}")]
    QemuExited { code: i32, message: String },

    /// Node not found.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Node already exists.
    #[error("Node already exists: {0}")]
    NodeExists(String),

    /// Invalid node configuration.
    #[error("Invalid node configuration: {0}")]
    InvalidNodeConfig(String),

    /// Cluster not initialized.
    #[error("Cluster not initialized")]
    ClusterNotInitialized,

    /// Cluster already running.
    #[error("Cluster already running")]
    ClusterAlreadyRunning,

    /// Timeout waiting for operation.
    #[error("Timeout waiting for {operation}: {details}")]
    Timeout { operation: String, details: String },

    /// Network configuration error.
    #[error("Network configuration error: {0}")]
    NetworkConfig(String),

    /// Socket connection failed.
    #[error("Socket connection failed: {path}")]
    SocketConnection { path: PathBuf },

    /// Console I/O error.
    #[error("Console I/O error: {0}")]
    ConsoleIO(String),

    /// RVF deployment failed.
    #[error("RVF deployment failed: {0}")]
    RvfDeployment(String),

    /// Fault injection failed.
    #[error("Fault injection failed: {0}")]
    FaultInjection(String),

    /// Metrics collection failed.
    #[error("Metrics collection failed: {0}")]
    MetricsCollection(String),

    /// Configuration parsing error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// TOML parsing error.
    #[error("TOML parsing error: {0}")]
    TomlParse(#[from] toml::de::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Maximum swarm size exceeded.
    #[error("Maximum swarm size ({max}) exceeded: requested {requested}")]
    MaxSwarmSizeExceeded { max: usize, requested: usize },

    /// Invalid topology.
    #[error("Invalid topology: {0}")]
    InvalidTopology(String),

    /// Resource exhausted.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Consensus error.
    #[error("Consensus error: {0}")]
    Consensus(String),
}

impl SwarmError {
    /// Create a timeout error.
    pub fn timeout(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::Timeout {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create a QEMU spawn error.
    pub fn qemu_spawn(msg: impl Into<String>) -> Self {
        Self::QemuSpawnFailed(msg.into())
    }

    /// Create a network config error.
    pub fn network(msg: impl Into<String>) -> Self {
        Self::NetworkConfig(msg.into())
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}
