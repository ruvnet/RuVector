//! Cluster-wide fan-out: query all nodes concurrently and aggregate results.

use std::collections::HashMap;
use std::time::Duration;

use futures::future;

use crate::client::VitalsClient;
use crate::error::Result;
use crate::proto::{HealthResponse, VitalReading};

/// Address of a single cluster node.
#[derive(Debug, Clone)]
pub struct NodeAddr {
    pub name: String,
    pub endpoint: String,
}

impl NodeAddr {
    pub fn new(name: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            endpoint: endpoint.into(),
        }
    }
}

/// Per-node health snapshot.
#[derive(Debug)]
pub struct NodeHealth {
    pub name: String,
    pub reachable: bool,
    pub health: Option<HealthResponse>,
}

/// Cluster-wide snapshot: latest reading per node + health of all nodes.
#[derive(Debug)]
pub struct ClusterSnapshot {
    /// Latest `VitalReading` per node name. Absent when unreachable.
    pub readings: HashMap<String, VitalReading>,
    /// Health probe results for all configured nodes.
    pub health: Vec<NodeHealth>,
    /// Number of nodes that responded to the health probe.
    pub nodes_up: usize,
}

impl ClusterSnapshot {
    /// True when every configured node is reachable.
    pub fn all_healthy(&self) -> bool {
        self.nodes_up == self.health.len()
    }
}

/// Fan-out client for the full ruview-vitals-worker cluster.
///
/// Spawns concurrent requests to all nodes; partial failures are tolerated —
/// unreachable nodes contribute a `None` reading and `reachable: false` health.
pub struct ClusterClient {
    nodes: Vec<NodeAddr>,
    connect_timeout: Duration,
}

impl ClusterClient {
    pub fn new(nodes: Vec<NodeAddr>) -> Self {
        Self {
            nodes,
            connect_timeout: Duration::from_secs(5),
        }
    }

    pub fn with_connect_timeout(mut self, t: Duration) -> Self {
        self.connect_timeout = t;
        self
    }

    /// Concurrently probe health and fetch the latest reading from every node.
    /// Never returns an error — partial failures surface in `ClusterSnapshot`.
    pub async fn snapshot(&self) -> Result<ClusterSnapshot> {
        let futs = self.nodes.iter().map(|n| {
            let name = n.name.clone();
            let endpoint = n.endpoint.clone();
            async move {
                match VitalsClient::connect(&name, &endpoint).await {
                    Err(e) => {
                        tracing::warn!(node = %name, error = %e, "cluster: connect failed");
                        (name, None, None)
                    }
                    Ok(mut c) => {
                        let health = c.health().await.ok();
                        let reading = c.latest(0).await.ok();
                        (name, health, reading)
                    }
                }
            }
        });

        let results = future::join_all(futs).await;

        let mut readings = HashMap::new();
        let mut health_vec = Vec::new();
        let mut nodes_up = 0usize;

        for (name, health, reading) in results {
            let reachable = health.is_some();
            if reachable {
                nodes_up += 1;
            }
            if let Some(r) = reading {
                readings.insert(name.clone(), r);
            }
            health_vec.push(NodeHealth {
                name,
                reachable,
                health,
            });
        }

        Ok(ClusterSnapshot {
            readings,
            health: health_vec,
            nodes_up,
        })
    }

    /// Concurrently fetch the latest reading from every node, returning a map
    /// of node names to readings for only the reachable nodes.
    pub async fn latest_all(&self) -> HashMap<String, VitalReading> {
        let futs = self.nodes.iter().map(|n| {
            let name = n.name.clone();
            let endpoint = n.endpoint.clone();
            async move {
                match VitalsClient::connect(&name, &endpoint).await {
                    Err(e) => {
                        tracing::debug!(node = %name, error = %e, "latest_all: skipping");
                        None
                    }
                    Ok(mut c) => c.latest(0).await.ok().map(|r| (name, r)),
                }
            }
        });
        future::join_all(futs)
            .await
            .into_iter()
            .flatten()
            .collect()
    }
}

/// Default cognitum cluster node addresses (Tailscale IPs).
pub fn default_cluster_nodes() -> Vec<NodeAddr> {
    vec![
        NodeAddr::new("cognitum-cluster-1", "http://100.80.54.16:50055"),
        NodeAddr::new("cognitum-cluster-2", "http://100.77.220.24:50055"),
        NodeAddr::new("cognitum-cluster-3", "http://100.73.75.53:50055"),
        NodeAddr::new("cognitum-v0", "http://100.77.59.83:50054"),
    ]
}
