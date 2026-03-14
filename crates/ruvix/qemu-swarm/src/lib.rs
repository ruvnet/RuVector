//! # RuVix QEMU Swarm Simulation System
//!
//! This crate provides a comprehensive simulation environment for testing
//! distributed RuVix clusters using multiple QEMU instances. It enables:
//!
//! - **Multi-node cluster simulation**: Spawn N QEMU instances as cluster nodes
//! - **Virtual networking**: Configurable topologies (mesh, ring, star)
//! - **Console multiplexing**: Aggregate output from all nodes
//! - **Swarm orchestration**: Deploy RVF packages, inject faults, collect metrics
//! - **Deterministic replay**: Reproducible test scenarios
//!
//! ## Architecture
//!
//! ```text
//!                          +------------------------+
//!                          |   SwarmOrchestrator    |
//!                          |  (coordination layer)  |
//!                          +----------+-------------+
//!                                     |
//!                   +-----------------+-----------------+
//!                   |                 |                 |
//!           +-------v------+  +-------v------+  +-------v------+
//!           | QemuCluster  |  | VirtualNetwork|  | ConsoleIO   |
//!           | (node mgmt)  |  | (networking) |  | (I/O mux)   |
//!           +-------+------+  +-------+------+  +-------+------+
//!                   |                 |                 |
//!       +-----------+-----------+     |                 |
//!       |           |           |     |                 |
//!   +---v---+   +---v---+   +---v---+ |                 |
//!   | Node0 |   | Node1 |   | NodeN | |                 |
//!   | QEMU  |   | QEMU  |   | QEMU  | |                 |
//!   +---+---+   +---+---+   +---+---+ |                 |
//!       |           |           |     |                 |
//!       +-----------+-----------+-----+-----------------+
//!                   |
//!           +-------v--------+
//!           | VirtIO Network |
//!           | (multicast)    |
//!           +----------------+
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ruvix_qemu_swarm::{QemuCluster, ClusterConfig, Topology};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a 3-node mesh cluster
//!     let config = ClusterConfig::builder()
//!         .node_count(3)
//!         .topology(Topology::Mesh)
//!         .kernel_path("/path/to/ruvix-kernel")
//!         .build()?;
//!
//!     let mut cluster = QemuCluster::new(config).await?;
//!     cluster.start_all().await?;
//!
//!     // Wait for nodes to be ready
//!     cluster.wait_for_ready(Duration::from_secs(30)).await?;
//!
//!     // Deploy RVF package to all nodes
//!     let orchestrator = cluster.orchestrator();
//!     orchestrator.deploy_rvf("/path/to/package.rvf").await?;
//!
//!     // Run tests...
//!
//!     cluster.stop_all().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Configuration
//!
//! Clusters can be configured via TOML files:
//!
//! ```toml
//! [cluster]
//! name = "test-cluster"
//! node_count = 3
//! topology = "mesh"
//!
//! [node.defaults]
//! cpu_count = 2
//! memory_mb = 512
//! kernel = "path/to/kernel"
//!
//! [network]
//! base_mac = "52:54:00:12:34:00"
//! multicast_group = "239.0.0.1:5000"
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod cluster;
pub mod consensus;
pub mod console;
pub mod network;
pub mod node;
pub mod orchestrator;

mod config;
mod error;

pub use cluster::{QemuCluster, ClusterConfig, ClusterStatus};
pub use config::{SwarmConfig, NodeDefaults, NetworkConfig};
pub use consensus::{
    PbftMessage, PbftReplica, PbftConfig, PbftStats,
    Request, PrePrepare, Prepare, Commit, Reply, ViewChange, NewView, Checkpoint,
    Operation, OperationResult, ReplicaState,
};
pub use console::{ConsoleIO, ConsoleMessage, ConsoleFilter};
pub use error::{SwarmError, SwarmResult};
pub use network::{VirtualNetwork, NetworkTopology, Topology, MacAddress};
pub use node::{QemuNode, NodeConfig, NodeStatus, NodeId};
pub use orchestrator::{SwarmOrchestrator, SwarmMetrics, FaultType, DeploymentResult};

use std::time::Duration;

/// Default timeout for cluster operations.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Default QEMU binary path.
pub const DEFAULT_QEMU_BINARY: &str = "qemu-system-aarch64";

/// Default memory per node in MB.
pub const DEFAULT_MEMORY_MB: u32 = 512;

/// Default CPU count per node.
pub const DEFAULT_CPU_COUNT: u32 = 2;

/// Maximum number of nodes in a swarm.
pub const MAX_SWARM_SIZE: usize = 64;

/// Base port for GDB servers.
pub const GDB_BASE_PORT: u16 = 1234;

/// Base port for QEMU monitors.
pub const MONITOR_BASE_PORT: u16 = 4444;

/// Base port for console sockets.
pub const CONSOLE_BASE_PORT: u16 = 5555;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        QemuCluster, ClusterConfig, QemuNode, NodeConfig,
        SwarmOrchestrator, VirtualNetwork, ConsoleIO,
        Topology, FaultType, SwarmResult,
        PbftReplica, PbftConfig, PbftMessage, Operation, Request,
    };
}
