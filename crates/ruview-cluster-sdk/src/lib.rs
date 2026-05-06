//! `ruview-cluster-sdk` — gRPC client for the cognitum ruview-vitals-worker
//! cluster (ADR-183). Provides typed access to all four nodes' vitals streams
//! with concurrent fan-out and health aggregation.
//!
//! ## Quick start
//!
//! ```ignore
//! use ruview_cluster_sdk::{ClusterClient, NodeAddr, cluster::default_cluster_nodes};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = ClusterClient::new(default_cluster_nodes());
//!     let snapshot = client.snapshot().await.unwrap();
//!     println!("{}/{} nodes up", snapshot.nodes_up, snapshot.health.len());
//!     for (name, r) in &snapshot.readings {
//!         let br = r.breathing.as_ref().map_or(0.0, |e| e.value_bpm);
//!         println!("{name}: breathing {br:.1} bpm");
//!     }
//! }
//! ```

pub mod client;
pub mod cluster;
pub mod error;

pub use client::VitalsClient;
pub use cluster::{ClusterClient, ClusterSnapshot, NodeAddr, NodeHealth};
pub use error::{Error, Result};

/// Generated tonic stubs (client-side only; server disabled in build.rs).
pub mod proto {
    tonic::include_proto!("cognitum.ruview.vitals.v1");
}
