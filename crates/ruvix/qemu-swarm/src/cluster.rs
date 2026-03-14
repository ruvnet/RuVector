//! Multi-node QEMU cluster management.
//!
//! This module orchestrates multiple QEMU nodes as a cohesive cluster.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::config::{NodeDefaults, SwarmConfig};
use crate::console::ConsoleIO;
use crate::error::{SwarmError, SwarmResult};
use crate::network::{NetworkTopology, Topology, VirtualNetwork};
use crate::node::{NodeConfig, NodeId, NodeStatus, QemuNode};
use crate::orchestrator::SwarmOrchestrator;
use crate::{DEFAULT_TIMEOUT, MAX_SWARM_SIZE};

/// Cluster status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterStatus {
    /// Cluster is created but not started.
    Created,
    /// Cluster is starting up.
    Starting,
    /// Cluster is running.
    Running,
    /// Cluster is partially running (some nodes failed).
    Degraded,
    /// Cluster is stopping.
    Stopping,
    /// Cluster is stopped.
    Stopped,
    /// Cluster has failed.
    Failed,
}

impl Default for ClusterStatus {
    fn default() -> Self {
        Self::Created
    }
}

/// Configuration for the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster name.
    pub name: String,

    /// Number of nodes.
    pub node_count: usize,

    /// Network topology.
    pub topology: Topology,

    /// Default node configuration.
    pub node_defaults: NodeDefaults,

    /// Per-node overrides.
    pub node_overrides: HashMap<usize, NodeOverrides>,

    /// Base working directory.
    pub work_dir: PathBuf,

    /// Delay between node startups (ms).
    pub startup_delay_ms: u64,

    /// Multicast group for networking.
    pub multicast_group: String,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            name: "ruvix-cluster".to_string(),
            node_count: 3,
            topology: Topology::Mesh,
            node_defaults: NodeDefaults::default(),
            node_overrides: HashMap::new(),
            work_dir: std::env::temp_dir().join("ruvix-swarm"),
            startup_delay_ms: 500,
            multicast_group: "239.0.0.1:5000".to_string(),
        }
    }
}

impl ClusterConfig {
    /// Create a new cluster configuration.
    pub fn new(name: impl Into<String>, node_count: usize) -> Self {
        Self {
            name: name.into(),
            node_count,
            ..Default::default()
        }
    }

    /// Create a builder for cluster configuration.
    pub fn builder() -> ClusterConfigBuilder {
        ClusterConfigBuilder::new()
    }

    /// Load from TOML config file.
    pub fn from_file(path: &PathBuf) -> SwarmResult<Self> {
        let config = SwarmConfig::from_file(path)?;
        Self::from_swarm_config(config)
    }

    /// Convert from SwarmConfig.
    pub fn from_swarm_config(config: SwarmConfig) -> SwarmResult<Self> {
        let mut overrides = HashMap::new();
        for node_override in &config.node.overrides {
            overrides.insert(
                node_override.index,
                NodeOverrides {
                    cpu_count: node_override.cpu_count,
                    memory_mb: node_override.memory_mb,
                    kernel_path: node_override.kernel.clone(),
                    extra_args: node_override.extra_args.clone(),
                },
            );
        }

        Ok(Self {
            name: config.cluster.name,
            node_count: config.cluster.node_count,
            topology: config.cluster.topology,
            node_defaults: config.node.defaults,
            node_overrides: overrides,
            work_dir: std::env::temp_dir().join("ruvix-swarm"),
            startup_delay_ms: config.cluster.startup_delay_ms,
            multicast_group: config.network.multicast_group,
        })
    }

    /// Generate node configurations.
    pub fn generate_node_configs(&self) -> SwarmResult<Vec<NodeConfig>> {
        if self.node_count > MAX_SWARM_SIZE {
            return Err(SwarmError::MaxSwarmSizeExceeded {
                max: MAX_SWARM_SIZE,
                requested: self.node_count,
            });
        }

        let mut configs = Vec::with_capacity(self.node_count);

        for i in 0..self.node_count {
            let mut config = NodeConfig::new(i);

            // Apply defaults
            config.cpu_count = self.node_defaults.cpu_count;
            config.memory_mb = self.node_defaults.memory_mb;
            config.kernel_path = self.node_defaults.kernel.clone();
            config.dtb_path = self.node_defaults.dtb.clone();
            config.machine = self.node_defaults.machine.clone();
            config.cpu_model = self.node_defaults.cpu_model.clone();
            config.extra_args = self.node_defaults.extra_args.clone();
            config.enable_gdb = self.node_defaults.enable_gdb;
            config.enable_monitor = self.node_defaults.enable_monitor;
            config.multicast_group = self.multicast_group.clone();

            // Set work directory
            config.work_dir = self.work_dir.join(format!("node-{}", i));
            config.console_socket = config.work_dir.join("console.sock");

            // Apply overrides
            if let Some(overrides) = self.node_overrides.get(&i) {
                if let Some(cpu) = overrides.cpu_count {
                    config.cpu_count = cpu;
                }
                if let Some(mem) = overrides.memory_mb {
                    config.memory_mb = mem;
                }
                if let Some(ref kernel) = overrides.kernel_path {
                    config.kernel_path = Some(kernel.clone());
                }
                config.extra_args.extend(overrides.extra_args.clone());
            }

            configs.push(config);
        }

        Ok(configs)
    }
}

/// Per-node configuration overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeOverrides {
    /// Override CPU count.
    pub cpu_count: Option<u32>,

    /// Override memory.
    pub memory_mb: Option<u32>,

    /// Override kernel path.
    pub kernel_path: Option<PathBuf>,

    /// Additional extra args.
    pub extra_args: Vec<String>,
}

/// Builder for ClusterConfig.
pub struct ClusterConfigBuilder {
    config: ClusterConfig,
}

impl ClusterConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: ClusterConfig::default(),
        }
    }

    /// Set cluster name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set node count.
    pub fn node_count(mut self, count: usize) -> Self {
        self.config.node_count = count;
        self
    }

    /// Set topology.
    pub fn topology(mut self, topology: Topology) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set kernel path.
    pub fn kernel_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.node_defaults.kernel = Some(path.into());
        self
    }

    /// Set DTB path.
    pub fn dtb_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.node_defaults.dtb = Some(path.into());
        self
    }

    /// Set CPU count per node.
    pub fn cpus_per_node(mut self, cpus: u32) -> Self {
        self.config.node_defaults.cpu_count = cpus;
        self
    }

    /// Set memory per node in MB.
    pub fn memory_per_node(mut self, mb: u32) -> Self {
        self.config.node_defaults.memory_mb = mb;
        self
    }

    /// Enable GDB servers.
    pub fn enable_gdb(mut self) -> Self {
        self.config.node_defaults.enable_gdb = true;
        self
    }

    /// Set work directory.
    pub fn work_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.work_dir = path.into();
        self
    }

    /// Set startup delay.
    pub fn startup_delay(mut self, ms: u64) -> Self {
        self.config.startup_delay_ms = ms;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> SwarmResult<ClusterConfig> {
        if self.config.node_count == 0 {
            return Err(SwarmError::InvalidNodeConfig("Node count cannot be 0".to_string()));
        }
        if self.config.node_count > MAX_SWARM_SIZE {
            return Err(SwarmError::MaxSwarmSizeExceeded {
                max: MAX_SWARM_SIZE,
                requested: self.config.node_count,
            });
        }
        Ok(self.config)
    }
}

impl Default for ClusterConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A multi-node QEMU cluster.
pub struct QemuCluster {
    /// Cluster configuration.
    config: ClusterConfig,

    /// Cluster status.
    status: Arc<RwLock<ClusterStatus>>,

    /// Nodes in the cluster.
    nodes: Vec<QemuNode>,

    /// Network topology and management.
    network: VirtualNetwork,

    /// Console I/O aggregator.
    console: ConsoleIO,

    /// Swarm orchestrator.
    orchestrator: SwarmOrchestrator,
}

impl QemuCluster {
    /// Create a new QEMU cluster.
    pub async fn new(config: ClusterConfig) -> SwarmResult<Self> {
        info!(
            name = %config.name,
            nodes = config.node_count,
            topology = %config.topology,
            "Creating QEMU cluster"
        );

        // Generate node configurations
        let node_configs = config.generate_node_configs()?;

        // Create nodes
        let mut nodes = Vec::with_capacity(config.node_count);
        for node_config in node_configs {
            nodes.push(QemuNode::new(node_config));
        }

        // Create network topology
        let network_topology = NetworkTopology::new(config.topology, config.node_count);
        let multicast_addr: std::net::SocketAddr = config
            .multicast_group
            .parse()
            .map_err(|e| SwarmError::network(format!("Invalid multicast group: {}", e)))?;
        let network = VirtualNetwork::new(network_topology, multicast_addr);

        // Create console aggregator
        let console = ConsoleIO::new(10000);

        // Create orchestrator
        let orchestrator = SwarmOrchestrator::new(config.node_count);

        Ok(Self {
            config,
            status: Arc::new(RwLock::new(ClusterStatus::Created)),
            nodes,
            network,
            console,
            orchestrator,
        })
    }

    /// Get cluster configuration.
    pub fn config(&self) -> &ClusterConfig {
        &self.config
    }

    /// Get cluster status.
    pub fn status(&self) -> ClusterStatus {
        *self.status.read()
    }

    /// Get number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a node by index.
    pub fn node(&self, index: usize) -> Option<&QemuNode> {
        self.nodes.get(index)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[QemuNode] {
        &self.nodes
    }

    /// Get the network manager.
    pub fn network(&self) -> &VirtualNetwork {
        &self.network
    }

    /// Get mutable network manager.
    pub fn network_mut(&mut self) -> &mut VirtualNetwork {
        &mut self.network
    }

    /// Get the console aggregator.
    pub fn console(&self) -> &ConsoleIO {
        &self.console
    }

    /// Get the orchestrator.
    pub fn orchestrator(&self) -> &SwarmOrchestrator {
        &self.orchestrator
    }

    /// Start all nodes in the cluster.
    pub async fn start_all(&mut self) -> SwarmResult<()> {
        let status = *self.status.read();
        if status == ClusterStatus::Running {
            return Err(SwarmError::ClusterAlreadyRunning);
        }

        {
            *self.status.write() = ClusterStatus::Starting;
        }

        info!(
            name = %self.config.name,
            nodes = self.nodes.len(),
            "Starting cluster"
        );

        // Create work directory
        tokio::fs::create_dir_all(&self.config.work_dir)
            .await
            .map_err(SwarmError::Io)?;

        let mut failed_nodes = Vec::new();
        let node_count = self.nodes.len();
        let startup_delay = self.config.startup_delay_ms;

        // Start nodes with delay
        for i in 0..node_count {
            let node = &mut self.nodes[i];
            match node.start().await {
                Ok(_) => {
                    info!(node = i, "Node started");

                    // Register console
                    let node_id = node.id();
                    let socket_path = node.config().console_socket.clone();
                    self.console
                        .register_node(node_id, i, socket_path)
                        .await?;
                }
                Err(e) => {
                    error!(node = i, error = %e, "Failed to start node");
                    failed_nodes.push(i);
                }
            }

            // Delay between startups
            if i < node_count - 1 {
                tokio::time::sleep(Duration::from_millis(startup_delay)).await;
            }
        }

        // Update status
        {
            let mut status = self.status.write();
            if failed_nodes.is_empty() {
                *status = ClusterStatus::Running;
            } else if failed_nodes.len() < self.nodes.len() {
                *status = ClusterStatus::Degraded;
            } else {
                *status = ClusterStatus::Failed;
            }
        }

        if !failed_nodes.is_empty() {
            warn!(
                failed = ?failed_nodes,
                "Some nodes failed to start"
            );
        }

        Ok(())
    }

    /// Stop all nodes in the cluster.
    pub async fn stop_all(&mut self) -> SwarmResult<()> {
        {
            *self.status.write() = ClusterStatus::Stopping;
        }

        info!(name = %self.config.name, "Stopping cluster");

        for (i, node) in self.nodes.iter().enumerate() {
            if let Err(e) = node.stop().await {
                warn!(node = i, error = %e, "Error stopping node");
            }
        }

        {
            *self.status.write() = ClusterStatus::Stopped;
        }

        Ok(())
    }

    /// Start a specific node.
    pub async fn start_node(&mut self, index: usize) -> SwarmResult<()> {
        let node = self
            .nodes
            .get_mut(index)
            .ok_or_else(|| SwarmError::NodeNotFound(format!("Node {}", index)))?;

        node.start().await?;

        // Register console
        self.console
            .register_node(node.id(), index, node.config().console_socket.clone())
            .await?;

        Ok(())
    }

    /// Stop a specific node.
    pub async fn stop_node(&mut self, index: usize) -> SwarmResult<()> {
        let node = self
            .nodes
            .get(index)
            .ok_or_else(|| SwarmError::NodeNotFound(format!("Node {}", index)))?;

        node.stop().await
    }

    /// Wait for all nodes to be ready.
    pub async fn wait_for_ready(&self, timeout: Duration) -> SwarmResult<()> {
        let start = Instant::now();
        let check_interval = Duration::from_millis(500);

        info!(
            timeout = ?timeout,
            "Waiting for cluster to be ready"
        );

        loop {
            if start.elapsed() > timeout {
                return Err(SwarmError::timeout(
                    "wait_for_ready",
                    "Cluster did not become ready in time",
                ));
            }

            let mut all_ready = true;
            for (i, node) in self.nodes.iter().enumerate() {
                let status = node.status();
                if status != NodeStatus::Running {
                    debug!(node = i, status = ?status, "Node not ready");
                    all_ready = false;
                    break;
                }
            }

            if all_ready {
                info!("Cluster is ready");
                return Ok(());
            }

            tokio::time::sleep(check_interval).await;
        }
    }

    /// Wait for a specific pattern in console output from all nodes.
    pub async fn wait_for_boot_message(
        &self,
        pattern: &str,
        timeout: Duration,
    ) -> SwarmResult<()> {
        self.console
            .wait_for_all_nodes(self.nodes.len(), pattern, timeout)
            .await
    }

    /// Perform health check on all nodes.
    pub async fn health_check(&self) -> HashMap<usize, bool> {
        let mut results = HashMap::new();

        for (i, node) in self.nodes.iter().enumerate() {
            results.insert(i, node.health_check().await);
        }

        results
    }

    /// Get cluster statistics.
    pub fn stats(&self) -> ClusterStats {
        let mut running = 0;
        let mut stopped = 0;
        let mut failed = 0;

        for node in &self.nodes {
            match node.status() {
                NodeStatus::Running => running += 1,
                NodeStatus::Stopped | NodeStatus::Created => stopped += 1,
                NodeStatus::Failed => failed += 1,
                _ => {}
            }
        }

        ClusterStats {
            name: self.config.name.clone(),
            status: *self.status.read(),
            node_count: self.nodes.len(),
            running_nodes: running,
            stopped_nodes: stopped,
            failed_nodes: failed,
            topology: self.config.topology,
            network_diameter: self.network.topology().diameter(),
            console_stats: self.console.stats(),
        }
    }

    /// Print cluster status.
    pub fn print_status(&self) {
        let stats = self.stats();
        println!("Cluster: {}", stats.name);
        println!("Status: {:?}", stats.status);
        println!(
            "Nodes: {} total ({} running, {} stopped, {} failed)",
            stats.node_count, stats.running_nodes, stats.stopped_nodes, stats.failed_nodes
        );
        println!("Topology: {} (diameter: {})", stats.topology, stats.network_diameter);
        println!("\nNetwork topology:");
        println!("{}", self.network.topology().ascii_diagram());
    }
}

/// Cluster statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Cluster name.
    pub name: String,

    /// Cluster status.
    pub status: ClusterStatus,

    /// Total node count.
    pub node_count: usize,

    /// Running nodes.
    pub running_nodes: usize,

    /// Stopped nodes.
    pub stopped_nodes: usize,

    /// Failed nodes.
    pub failed_nodes: usize,

    /// Network topology.
    pub topology: Topology,

    /// Network diameter.
    pub network_diameter: usize,

    /// Console statistics.
    pub console_stats: crate::console::ConsoleStats,
}

impl Drop for QemuCluster {
    fn drop(&mut self) {
        // Stop all nodes synchronously
        for node in &self.nodes {
            let _ = std::thread::spawn({
                let node_id = node.id();
                move || {
                    debug!(node = %node_id, "Cleaning up node");
                }
            })
            .join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_builder() {
        let config = ClusterConfig::builder()
            .name("test")
            .node_count(4)
            .topology(Topology::Ring)
            .cpus_per_node(2)
            .memory_per_node(1024)
            .build()
            .unwrap();

        assert_eq!(config.name, "test");
        assert_eq!(config.node_count, 4);
        assert_eq!(config.topology, Topology::Ring);
        assert_eq!(config.node_defaults.cpu_count, 2);
        assert_eq!(config.node_defaults.memory_mb, 1024);
    }

    #[test]
    fn test_generate_node_configs() {
        let config = ClusterConfig::new("test", 3);
        let node_configs = config.generate_node_configs().unwrap();

        assert_eq!(node_configs.len(), 3);
        assert_ne!(node_configs[0].mac_address, node_configs[1].mac_address);
    }

    #[test]
    fn test_max_swarm_size() {
        let result = ClusterConfig::builder()
            .node_count(MAX_SWARM_SIZE + 1)
            .build();

        assert!(result.is_err());
    }
}
