//! Configuration types for swarm simulation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::network::Topology;

/// Complete swarm configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Cluster configuration section.
    pub cluster: ClusterSection,

    /// Node defaults section.
    pub node: NodeSection,

    /// Network configuration section.
    pub network: NetworkConfig,

    /// Optional test scenarios.
    #[serde(default)]
    pub scenarios: Vec<TestScenario>,
}

impl SwarmConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &PathBuf) -> Result<Self, crate::SwarmError> {
        let content = std::fs::read_to_string(path)
            .map_err(|_| crate::SwarmError::FileNotFound(path.clone()))?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_str(content: &str) -> Result<Self, crate::SwarmError> {
        toml::from_str(content).map_err(crate::SwarmError::TomlParse)
    }

    /// Create a minimal default configuration.
    pub fn minimal(node_count: usize) -> Self {
        Self {
            cluster: ClusterSection {
                name: "minimal-cluster".to_string(),
                node_count,
                topology: Topology::Mesh,
                startup_delay_ms: 500,
            },
            node: NodeSection {
                defaults: NodeDefaults::default(),
                overrides: Vec::new(),
            },
            network: NetworkConfig::default(),
            scenarios: Vec::new(),
        }
    }
}

/// Cluster configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSection {
    /// Cluster name for identification.
    pub name: String,

    /// Number of nodes in the cluster.
    pub node_count: usize,

    /// Network topology.
    #[serde(default)]
    pub topology: Topology,

    /// Delay between node startups in milliseconds.
    #[serde(default = "default_startup_delay")]
    pub startup_delay_ms: u64,
}

fn default_startup_delay() -> u64 {
    500
}

/// Node configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSection {
    /// Default settings for all nodes.
    pub defaults: NodeDefaults,

    /// Per-node overrides.
    #[serde(default)]
    pub overrides: Vec<NodeOverride>,
}

/// Default settings for all nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefaults {
    /// Number of CPUs per node.
    #[serde(default = "default_cpu_count")]
    pub cpu_count: u32,

    /// Memory in MB per node.
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u32,

    /// Path to the kernel binary.
    pub kernel: Option<PathBuf>,

    /// Path to device tree blob.
    pub dtb: Option<PathBuf>,

    /// Machine type for QEMU.
    #[serde(default = "default_machine")]
    pub machine: String,

    /// CPU model for QEMU.
    #[serde(default = "default_cpu_model")]
    pub cpu_model: String,

    /// Additional QEMU arguments.
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Enable GDB server.
    #[serde(default)]
    pub enable_gdb: bool,

    /// Enable QEMU monitor.
    #[serde(default = "default_true")]
    pub enable_monitor: bool,
}

impl Default for NodeDefaults {
    fn default() -> Self {
        Self {
            cpu_count: default_cpu_count(),
            memory_mb: default_memory_mb(),
            kernel: None,
            dtb: None,
            machine: default_machine(),
            cpu_model: default_cpu_model(),
            extra_args: Vec::new(),
            enable_gdb: false,
            enable_monitor: true,
        }
    }
}

fn default_cpu_count() -> u32 {
    crate::DEFAULT_CPU_COUNT
}

fn default_memory_mb() -> u32 {
    crate::DEFAULT_MEMORY_MB
}

fn default_machine() -> String {
    "virt".to_string()
}

fn default_cpu_model() -> String {
    "cortex-a72".to_string()
}

fn default_true() -> bool {
    true
}

/// Per-node configuration override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOverride {
    /// Node index (0-based).
    pub index: usize,

    /// Override CPU count.
    pub cpu_count: Option<u32>,

    /// Override memory.
    pub memory_mb: Option<u32>,

    /// Override kernel path.
    pub kernel: Option<PathBuf>,

    /// Additional extra args for this node.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

/// Network configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Base MAC address (last byte incremented per node).
    #[serde(default = "default_base_mac")]
    pub base_mac: String,

    /// Multicast group for inter-node communication.
    #[serde(default = "default_multicast_group")]
    pub multicast_group: String,

    /// Network MTU.
    #[serde(default = "default_mtu")]
    pub mtu: u16,

    /// Enable network isolation between groups.
    #[serde(default)]
    pub isolation_groups: Vec<IsolationGroup>,

    /// Network latency simulation in microseconds.
    #[serde(default)]
    pub latency_us: u32,

    /// Packet loss percentage (0-100).
    #[serde(default)]
    pub packet_loss_pct: u8,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            base_mac: default_base_mac(),
            multicast_group: default_multicast_group(),
            mtu: default_mtu(),
            isolation_groups: Vec::new(),
            latency_us: 0,
            packet_loss_pct: 0,
        }
    }
}

fn default_base_mac() -> String {
    "52:54:00:12:34:00".to_string()
}

fn default_multicast_group() -> String {
    "239.0.0.1:5000".to_string()
}

fn default_mtu() -> u16 {
    1500
}

/// Network isolation group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationGroup {
    /// Group name.
    pub name: String,

    /// Node indices in this group.
    pub nodes: Vec<usize>,

    /// Multicast group for this isolation group.
    pub multicast_group: String,
}

/// Test scenario definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    /// Scenario name.
    pub name: String,

    /// Description.
    pub description: Option<String>,

    /// Steps to execute.
    pub steps: Vec<ScenarioStep>,

    /// Expected outcomes.
    #[serde(default)]
    pub assertions: Vec<Assertion>,
}

/// A step in a test scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum ScenarioStep {
    /// Wait for a duration.
    #[serde(rename = "wait")]
    Wait { duration_ms: u64 },

    /// Deploy an RVF package.
    #[serde(rename = "deploy")]
    Deploy { rvf_path: PathBuf, nodes: Option<Vec<usize>> },

    /// Inject a fault.
    #[serde(rename = "fault")]
    Fault { fault_type: String, node: usize },

    /// Broadcast a message.
    #[serde(rename = "broadcast")]
    Broadcast { message: String },

    /// Collect metrics.
    #[serde(rename = "metrics")]
    CollectMetrics { output: PathBuf },

    /// Custom command to a node.
    #[serde(rename = "command")]
    Command { node: usize, command: String },
}

/// Assertion for test scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    /// Assertion type.
    #[serde(rename = "type")]
    pub assertion_type: String,

    /// Expected value.
    pub expected: String,

    /// Target node (optional).
    pub node: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
[cluster]
name = "test"
node_count = 3

[node.defaults]
cpu_count = 2
memory_mb = 512

[network]
"#;
        let config: SwarmConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.cluster.name, "test");
        assert_eq!(config.cluster.node_count, 3);
    }

    #[test]
    fn test_default_values() {
        let defaults = NodeDefaults::default();
        assert_eq!(defaults.cpu_count, 2);
        assert_eq!(defaults.memory_mb, 512);
        assert_eq!(defaults.machine, "virt");
    }
}
