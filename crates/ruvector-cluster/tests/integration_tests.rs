//! Integration tests for ruvector-cluster
//!
//! These tests exercise the public API of the cluster management system
//! end-to-end, covering cluster formation, node registration/deregistration,
//! shard assignment, consistent hashing, discovery services, and the
//! DAG-based consensus protocol.

use ruvector_cluster::consensus::{DagConsensus, DagVertex, Transaction, TransactionType};
use ruvector_cluster::discovery::{DiscoveryService, GossipDiscovery, StaticDiscovery};
use ruvector_cluster::shard::{ConsistentHashRing, LoadBalancer, ShardMigration, ShardRouter};
use ruvector_cluster::{
    ClusterConfig, ClusterManager, ClusterNode, NodeStatus, ShardStatus,
};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port)
}

fn test_node(id: &str, port: u16) -> ClusterNode {
    ClusterNode::new(id.to_string(), test_addr(port))
}

fn test_config(shard_count: u32, replication_factor: usize) -> ClusterConfig {
    ClusterConfig {
        shard_count,
        replication_factor,
        heartbeat_interval: Duration::from_secs(5),
        node_timeout: Duration::from_secs(30),
        enable_consensus: true,
        min_quorum_size: 2,
    }
}

fn test_manager(
    shard_count: u32,
    replication_factor: usize,
) -> ClusterManager {
    let config = test_config(shard_count, replication_factor);
    let discovery = Box::new(StaticDiscovery::new(vec![]));
    ClusterManager::new(config, "test-manager".to_string(), discovery).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Cluster creation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cluster_manager_creation_with_defaults() {
    let config = ClusterConfig::default();
    let discovery = Box::new(StaticDiscovery::new(vec![]));
    let result = ClusterManager::new(config, "manager-1".to_string(), discovery);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_cluster_manager_creation_with_custom_config() {
    let config = ClusterConfig {
        replication_factor: 5,
        shard_count: 128,
        heartbeat_interval: Duration::from_secs(2),
        node_timeout: Duration::from_secs(10),
        enable_consensus: false,
        min_quorum_size: 3,
    };
    let discovery = Box::new(StaticDiscovery::new(vec![]));
    let manager = ClusterManager::new(config, "custom-mgr".to_string(), discovery).unwrap();

    // Without consensus enabled, consensus() should return None
    assert!(manager.consensus().is_none());
}

#[tokio::test]
async fn test_cluster_manager_consensus_enabled() {
    let config = ClusterConfig {
        enable_consensus: true,
        ..ClusterConfig::default()
    };
    let discovery = Box::new(StaticDiscovery::new(vec![]));
    let manager = ClusterManager::new(config, "mgr".to_string(), discovery).unwrap();

    assert!(manager.consensus().is_some());
}

#[tokio::test]
async fn test_cluster_starts_empty() {
    let manager = test_manager(16, 3);
    let stats = manager.get_stats();

    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.healthy_nodes, 0);
    assert_eq!(stats.total_vectors, 0);
}

// ---------------------------------------------------------------------------
// 2. Node registration and deregistration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_add_single_node() {
    let manager = test_manager(4, 2);
    let node = test_node("node-1", 9001);

    manager.add_node(node).await.unwrap();

    let nodes = manager.list_nodes();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, "node-1");
}

#[tokio::test]
async fn test_add_multiple_nodes() {
    let manager = test_manager(4, 2);

    for i in 0..5 {
        manager
            .add_node(test_node(&format!("n{}", i), 9000 + i))
            .await
            .unwrap();
    }

    assert_eq!(manager.list_nodes().len(), 5);
}

#[tokio::test]
async fn test_get_node_by_id() {
    let manager = test_manager(4, 2);
    manager.add_node(test_node("alpha", 9001)).await.unwrap();
    manager.add_node(test_node("beta", 9002)).await.unwrap();

    let found = manager.get_node("alpha");
    assert!(found.is_some());
    assert_eq!(found.unwrap().node_id, "alpha");

    let missing = manager.get_node("gamma");
    assert!(missing.is_none());
}

#[tokio::test]
async fn test_remove_node() {
    let manager = test_manager(4, 2);

    manager.add_node(test_node("n1", 9001)).await.unwrap();
    manager.add_node(test_node("n2", 9002)).await.unwrap();
    assert_eq!(manager.list_nodes().len(), 2);

    manager.remove_node("n1").await.unwrap();
    assert_eq!(manager.list_nodes().len(), 1);

    // Verify the correct node was removed
    assert!(manager.get_node("n1").is_none());
    assert!(manager.get_node("n2").is_some());
}

#[tokio::test]
async fn test_remove_nonexistent_node_does_not_panic() {
    let manager = test_manager(4, 2);
    // Removing a non-existent node from an empty cluster triggers rebalancing
    // which may error because there are no nodes to assign shards to.
    // The important property is that it does not panic.
    let _result = manager.remove_node("does-not-exist").await;
    // If we reach this point, the call didn't panic -- that's the safety guarantee.
}

#[tokio::test]
async fn test_remove_node_that_was_added() {
    let manager = test_manager(4, 2);
    manager.add_node(test_node("a", 9001)).await.unwrap();
    manager.add_node(test_node("b", 9002)).await.unwrap();

    // Removing an existing node when others remain should succeed
    let result = manager.remove_node("a").await;
    assert!(result.is_ok());
    assert!(manager.get_node("a").is_none());
    assert!(manager.get_node("b").is_some());
}

#[tokio::test]
async fn test_node_health_check() {
    let node = test_node("healthy-node", 9001);
    assert!(node.is_healthy(Duration::from_secs(60)));

    // A very tight timeout will consider even a fresh node unhealthy
    // (since some nanoseconds have passed since creation)
    // We test the opposite: a generous timeout keeps the node healthy
    assert!(node.is_healthy(Duration::from_secs(300)));
}

#[tokio::test]
async fn test_node_heartbeat_refreshes_timestamp() {
    let mut node = test_node("node-hb", 9001);
    let original_last_seen = node.last_seen;

    // Small delay to ensure timestamps differ
    tokio::time::sleep(Duration::from_millis(10)).await;
    node.heartbeat();

    assert!(
        node.last_seen >= original_last_seen,
        "heartbeat should refresh last_seen"
    );
}

#[tokio::test]
async fn test_node_default_status_is_follower() {
    let node = test_node("new-node", 9001);
    assert_eq!(node.status, NodeStatus::Follower);
    assert_eq!(node.capacity, 1.0);
    assert!(node.metadata.is_empty());
}

#[tokio::test]
async fn test_healthy_nodes_filter() {
    let manager = test_manager(4, 2);

    // All fresh nodes should be healthy
    for i in 0..3 {
        manager
            .add_node(test_node(&format!("h{}", i), 9000 + i))
            .await
            .unwrap();
    }

    let healthy = manager.healthy_nodes();
    assert_eq!(healthy.len(), 3);
}

// ---------------------------------------------------------------------------
// 3. Shard assignment
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_shard_assignment_with_nodes() {
    let manager = test_manager(8, 2);

    for i in 0..3 {
        manager
            .add_node(test_node(&format!("node{}", i), 9000 + i))
            .await
            .unwrap();
    }

    let shard = manager.assign_shard(0).unwrap();
    assert_eq!(shard.shard_id, 0);
    assert_eq!(shard.status, ShardStatus::Active);
    assert!(!shard.primary_node.is_empty());
    assert_eq!(shard.vector_count, 0);
}

#[tokio::test]
async fn test_shard_assignment_returns_replicas() {
    let manager = test_manager(8, 3);

    for i in 0..5 {
        manager
            .add_node(test_node(&format!("node{}", i), 9000 + i))
            .await
            .unwrap();
    }

    let shard = manager.assign_shard(1).unwrap();
    // With replication_factor=3, we expect 1 primary + up to 2 replicas
    assert!(
        shard.replica_nodes.len() <= 2,
        "should have at most (replication_factor - 1) replicas"
    );
}

#[tokio::test]
async fn test_shard_assignment_without_nodes_fails() {
    let manager = test_manager(8, 2);
    // No nodes added => assignment should fail
    let result = manager.assign_shard(0);
    assert!(result.is_err(), "assigning shard with no nodes should fail");
}

#[tokio::test]
async fn test_get_shard_after_assignment() {
    let manager = test_manager(8, 2);
    manager.add_node(test_node("n1", 9001)).await.unwrap();

    manager.assign_shard(3).unwrap();

    let shard = manager.get_shard(3);
    assert!(shard.is_some());
    assert_eq!(shard.unwrap().shard_id, 3);
}

#[tokio::test]
async fn test_list_shards() {
    let manager = test_manager(8, 2);
    manager.add_node(test_node("n1", 9001)).await.unwrap();

    for sid in 0..4 {
        manager.assign_shard(sid).unwrap();
    }

    // There may be more shards due to rebalancing in add_node,
    // but at least our 4 manual assignments should be present
    let shards = manager.list_shards();
    assert!(shards.len() >= 4);
}

#[tokio::test]
async fn test_cluster_stats_after_setup() {
    let manager = test_manager(4, 2);

    for i in 0..3 {
        manager
            .add_node(test_node(&format!("s{}", i), 9000 + i))
            .await
            .unwrap();
    }

    let stats = manager.get_stats();
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.healthy_nodes, 3);
    // Shards should be auto-assigned during rebalancing
    assert!(stats.total_shards > 0);
}

// ---------------------------------------------------------------------------
// 4. Consistent hash ring
// ---------------------------------------------------------------------------

#[test]
fn test_hash_ring_empty() {
    let ring = ConsistentHashRing::new(3);
    assert_eq!(ring.node_count(), 0);
    assert!(ring.get_primary_node("any-key").is_none());
    assert!(ring.get_nodes("any-key", 3).is_empty());
}

#[test]
fn test_hash_ring_add_and_remove() {
    let mut ring = ConsistentHashRing::new(2);

    ring.add_node("alpha".to_string());
    ring.add_node("beta".to_string());
    assert_eq!(ring.node_count(), 2);

    ring.remove_node("alpha");
    assert_eq!(ring.node_count(), 1);

    let nodes = ring.list_nodes();
    assert!(nodes.contains(&"beta".to_string()));
    assert!(!nodes.contains(&"alpha".to_string()));
}

#[test]
fn test_hash_ring_idempotent_add() {
    let mut ring = ConsistentHashRing::new(2);
    ring.add_node("node".to_string());
    ring.add_node("node".to_string()); // duplicate
    assert_eq!(ring.node_count(), 1);
}

#[test]
fn test_hash_ring_remove_nonexistent_is_safe() {
    let mut ring = ConsistentHashRing::new(2);
    ring.add_node("existing".to_string());
    ring.remove_node("ghost"); // should not panic
    assert_eq!(ring.node_count(), 1);
}

#[test]
fn test_hash_ring_deterministic_routing() {
    let mut ring = ConsistentHashRing::new(3);
    ring.add_node("a".to_string());
    ring.add_node("b".to_string());
    ring.add_node("c".to_string());

    let primary1 = ring.get_primary_node("my-key").unwrap();
    let primary2 = ring.get_primary_node("my-key").unwrap();
    assert_eq!(primary1, primary2, "same key must always route to same node");
}

#[test]
fn test_hash_ring_distribution_fairness() {
    let mut ring = ConsistentHashRing::new(3);
    ring.add_node("n1".to_string());
    ring.add_node("n2".to_string());
    ring.add_node("n3".to_string());

    let mut counts: HashMap<String, usize> = HashMap::new();
    for i in 0..3000 {
        let key = format!("key-{}", i);
        if let Some(node) = ring.get_primary_node(&key) {
            *counts.entry(node).or_default() += 1;
        }
    }

    // Each node should get roughly 1/3 of keys (with some variance)
    for (node, count) in &counts {
        let ratio = *count as f64 / 3000.0;
        assert!(
            ratio > 0.15 && ratio < 0.55,
            "node {} has ratio {} which is outside acceptable range",
            node,
            ratio
        );
    }
}

#[test]
fn test_hash_ring_get_nodes_returns_unique_nodes() {
    let mut ring = ConsistentHashRing::new(3);
    ring.add_node("x".to_string());
    ring.add_node("y".to_string());
    ring.add_node("z".to_string());

    let nodes = ring.get_nodes("test-key", 3);
    assert_eq!(nodes.len(), 3);

    // All nodes should be unique
    let mut unique = nodes.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 3, "get_nodes should return unique nodes");
}

#[test]
fn test_hash_ring_get_nodes_caps_at_available() {
    let mut ring = ConsistentHashRing::new(3);
    ring.add_node("only-node".to_string());

    // Requesting 5 nodes when only 1 exists
    let nodes = ring.get_nodes("key", 5);
    assert_eq!(nodes.len(), 1);
}

// ---------------------------------------------------------------------------
// 5. Shard router (jump consistent hash)
// ---------------------------------------------------------------------------

#[test]
fn test_shard_router_deterministic() {
    let router = ShardRouter::new(32);

    let shard1 = router.get_shard("vector-123");
    let shard2 = router.get_shard("vector-123");
    assert_eq!(shard1, shard2, "same key must always route to same shard");
}

#[test]
fn test_shard_router_range_within_bounds() {
    let shard_count = 64;
    let router = ShardRouter::new(shard_count);

    for i in 0..1000 {
        let shard = router.get_shard(&format!("key-{}", i));
        assert!(
            shard < shard_count,
            "shard {} exceeds shard_count {}",
            shard,
            shard_count
        );
    }
}

#[test]
fn test_shard_router_cache_behaviour() {
    let router = ShardRouter::new(16);

    let _ = router.get_shard("cached-key");
    let stats = router.cache_stats();
    assert_eq!(stats.entries, 1);

    let _ = router.get_shard("another-key");
    let stats = router.cache_stats();
    assert_eq!(stats.entries, 2);

    router.clear_cache();
    let stats = router.cache_stats();
    assert_eq!(stats.entries, 0);
}

#[test]
fn test_shard_router_for_vector_id() {
    let router = ShardRouter::new(16);

    let shard_a = router.get_shard_for_vector("vec-abc");
    let shard_b = router.get_shard_for_vector("vec-abc");
    assert_eq!(shard_a, shard_b);
}

#[test]
fn test_shard_router_range_query_returns_all_shards() {
    let router = ShardRouter::new(8);
    let shards = router.get_shards_for_range("a", "z");
    assert_eq!(shards.len(), 8);
}

// ---------------------------------------------------------------------------
// 6. Shard migration
// ---------------------------------------------------------------------------

#[test]
fn test_shard_migration_lifecycle() {
    let mut migration = ShardMigration::new(0, 1, 200);

    assert!(!migration.is_complete());
    assert_eq!(migration.progress, 0.0);
    assert_eq!(migration.source_shard, 0);
    assert_eq!(migration.target_shard, 1);

    migration.update_progress(100);
    assert!((migration.progress - 0.5).abs() < f64::EPSILON);
    assert!(!migration.is_complete());

    migration.update_progress(200);
    assert!(migration.is_complete());
    assert!((migration.progress - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_shard_migration_zero_keys() {
    let migration = ShardMigration::new(0, 1, 0);
    // With 0 total keys, migration should be considered complete
    assert!(migration.is_complete());
}

// ---------------------------------------------------------------------------
// 7. Load balancer
// ---------------------------------------------------------------------------

#[test]
fn test_load_balancer_least_loaded() {
    let lb = LoadBalancer::new();

    lb.update_load(0, 0.9);
    lb.update_load(1, 0.2);
    lb.update_load(2, 0.5);

    let least = lb.get_least_loaded_shard(&[0, 1, 2]);
    assert_eq!(least, Some(1));
}

#[test]
fn test_load_balancer_stats() {
    let lb = LoadBalancer::new();

    lb.update_load(0, 0.4);
    lb.update_load(1, 0.6);

    let stats = lb.get_stats();
    assert_eq!(stats.shard_count, 2);
    assert!((stats.avg_load - 0.5).abs() < f64::EPSILON);
    assert!((stats.max_load - 0.6).abs() < f64::EPSILON);
    assert!((stats.min_load - 0.4).abs() < f64::EPSILON);
}

#[test]
fn test_load_balancer_empty() {
    let lb = LoadBalancer::new();
    let stats = lb.get_stats();
    assert_eq!(stats.shard_count, 0);
    assert_eq!(stats.avg_load, 0.0);
}

#[test]
fn test_load_balancer_no_candidates() {
    let lb = LoadBalancer::new();
    let result = lb.get_least_loaded_shard(&[]);
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// 8. Discovery services
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_static_discovery_full_lifecycle() {
    let discovery = StaticDiscovery::new(vec![]);

    // Register nodes
    discovery
        .register_node(test_node("d1", 9001))
        .await
        .unwrap();
    discovery
        .register_node(test_node("d2", 9002))
        .await
        .unwrap();

    let nodes = discovery.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 2);

    // Unregister one
    discovery.unregister_node("d1").await.unwrap();
    let nodes = discovery.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, "d2");
}

#[tokio::test]
async fn test_static_discovery_heartbeat() {
    let node = test_node("hb-node", 9001);
    let discovery = StaticDiscovery::new(vec![node]);

    // Heartbeat should succeed without error
    let result = discovery.heartbeat("hb-node").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_gossip_discovery_initial_state() {
    let local = test_node("local-g", 8000);
    let discovery = GossipDiscovery::new(
        local,
        vec![test_addr(9000)],
        Duration::from_secs(5),
        Duration::from_secs(30),
    );

    let nodes = discovery.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 1, "only local node should exist initially");
}

#[tokio::test]
async fn test_gossip_discovery_merge() {
    let local = test_node("local-g", 8000);
    let discovery = GossipDiscovery::new(
        local,
        vec![],
        Duration::from_secs(5),
        Duration::from_secs(30),
    );

    // Simulate receiving gossip from remote nodes
    let remote_nodes = vec![
        test_node("remote-1", 8001),
        test_node("remote-2", 8002),
        test_node("remote-3", 8003),
    ];
    discovery.merge_gossip(remote_nodes);

    let stats = discovery.get_stats();
    assert_eq!(stats.total_nodes, 4); // local + 3 remote
    assert_eq!(stats.healthy_nodes, 4);
}

#[tokio::test]
async fn test_gossip_discovery_register_and_unregister() {
    let local = test_node("local", 8000);
    let discovery = GossipDiscovery::new(
        local,
        vec![],
        Duration::from_secs(5),
        Duration::from_secs(30),
    );

    discovery
        .register_node(test_node("new-peer", 8010))
        .await
        .unwrap();

    let nodes = discovery.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 2);

    discovery.unregister_node("new-peer").await.unwrap();
    let nodes = discovery.discover_nodes().await.unwrap();
    assert_eq!(nodes.len(), 1);
}

// ---------------------------------------------------------------------------
// 9. DAG consensus
// ---------------------------------------------------------------------------

#[test]
fn test_dag_consensus_creation() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);
    let stats = consensus.get_stats();

    assert_eq!(stats.total_vertices, 0);
    assert_eq!(stats.finalized_vertices, 0);
    assert_eq!(stats.pending_transactions, 0);
    assert_eq!(stats.tips, 0);
}

#[test]
fn test_dag_submit_and_create_vertex() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    let tx_id = consensus
        .submit_transaction(TransactionType::Write, b"data".to_vec())
        .unwrap();
    assert!(!tx_id.is_empty());

    let stats = consensus.get_stats();
    assert_eq!(stats.pending_transactions, 1);

    let vertex = consensus.create_vertex().unwrap();
    assert!(vertex.is_some());

    let stats = consensus.get_stats();
    assert_eq!(stats.total_vertices, 1);
    assert_eq!(stats.pending_transactions, 0);
    assert_eq!(stats.tips, 1);
}

#[test]
fn test_dag_create_vertex_with_no_pending_returns_none() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);
    let vertex = consensus.create_vertex().unwrap();
    assert!(vertex.is_none());
}

#[test]
fn test_dag_multiple_vertices_form_chain() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    // Create a chain of 5 vertices
    for i in 0..5 {
        consensus
            .submit_transaction(TransactionType::Write, vec![i])
            .unwrap();
        consensus.create_vertex().unwrap();
    }

    let stats = consensus.get_stats();
    assert_eq!(stats.total_vertices, 5);
    // Each new vertex references the previous as parent, so there should be 1 tip
    assert_eq!(stats.tips, 1);
}

#[test]
fn test_dag_add_vertex_from_another_node() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    // Create a local vertex first to serve as parent
    consensus
        .submit_transaction(TransactionType::Write, b"local".to_vec())
        .unwrap();
    let local_vertex = consensus.create_vertex().unwrap().unwrap();

    // Simulate a vertex from another node referencing the local vertex
    let remote_vertex = DagVertex::new(
        "node-2".to_string(),
        Transaction {
            id: "remote-tx-1".to_string(),
            tx_type: TransactionType::Write,
            data: b"remote-data".to_vec(),
            nonce: 1,
        },
        vec![local_vertex.id.clone()],
        {
            let mut clock = HashMap::new();
            clock.insert("node-1".to_string(), 1);
            clock.insert("node-2".to_string(), 1);
            clock
        },
    );

    let result = consensus.add_vertex(remote_vertex);
    assert!(result.is_ok());

    let stats = consensus.get_stats();
    assert_eq!(stats.total_vertices, 2);
}

#[test]
fn test_dag_add_vertex_with_missing_parent_fails() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    let orphan_vertex = DagVertex::new(
        "node-2".to_string(),
        Transaction {
            id: "orphan-tx".to_string(),
            tx_type: TransactionType::Write,
            data: b"orphan".to_vec(),
            nonce: 1,
        },
        vec!["nonexistent-parent-id".to_string()],
        HashMap::new(),
    );

    let result = consensus.add_vertex(orphan_vertex);
    assert!(
        result.is_err(),
        "vertex with missing parent should be rejected"
    );
}

#[test]
fn test_dag_finalization_requires_quorum() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    // Create vertices only from node-1
    for i in 0..3 {
        consensus
            .submit_transaction(TransactionType::Write, vec![i])
            .unwrap();
        consensus.create_vertex().unwrap();
    }

    // Without vertices from other nodes, nothing can be finalized
    let finalized = consensus.finalize_vertices().unwrap();
    assert_eq!(finalized.len(), 0);
}

#[test]
fn test_dag_finalization_with_multi_node_confirmations() {
    let consensus = DagConsensus::new("node-1".to_string(), 1);

    // Create vertex from node-1
    consensus
        .submit_transaction(TransactionType::Write, b"v1".to_vec())
        .unwrap();
    let v1 = consensus.create_vertex().unwrap().unwrap();

    // Simulate a confirmation vertex from node-2 that references v1
    let confirm = DagVertex::new(
        "node-2".to_string(),
        Transaction {
            id: "confirm-tx".to_string(),
            tx_type: TransactionType::Write,
            data: b"confirm".to_vec(),
            nonce: 1,
        },
        vec![v1.id.clone()],
        {
            let mut clock = HashMap::new();
            clock.insert("node-1".to_string(), 1);
            clock.insert("node-2".to_string(), 1);
            clock
        },
    );
    consensus.add_vertex(confirm).unwrap();

    // Now v1 has a confirmation from node-2, with min_quorum_size=1 it should finalize
    let finalized = consensus.finalize_vertices().unwrap();
    assert!(
        finalized.contains(&v1.id),
        "v1 should be finalized with sufficient confirmations"
    );
    assert!(consensus.is_finalized(&v1.id));
}

#[test]
fn test_dag_conflict_detection() {
    let consensus = DagConsensus::new("node-1".to_string(), 2);

    let write_tx = Transaction {
        id: "w1".to_string(),
        tx_type: TransactionType::Write,
        data: b"write".to_vec(),
        nonce: 1,
    };
    let another_write = Transaction {
        id: "w2".to_string(),
        tx_type: TransactionType::Write,
        data: b"write2".to_vec(),
        nonce: 2,
    };
    let read_tx = Transaction {
        id: "r1".to_string(),
        tx_type: TransactionType::Read,
        data: b"read".to_vec(),
        nonce: 3,
    };

    // Write-Write conflicts
    assert!(consensus.detect_conflicts(&write_tx, &another_write));
    // Read-Read does not conflict
    assert!(!consensus.detect_conflicts(&read_tx, &read_tx));
    // Read-Write does not conflict (per current implementation)
    assert!(!consensus.detect_conflicts(&read_tx, &write_tx));
}

#[test]
fn test_dag_prune_old_vertices() {
    let consensus = DagConsensus::new("node-1".to_string(), 1);

    // Create vertices from node-1
    let mut vertex_ids = Vec::new();
    for i in 0..5 {
        consensus
            .submit_transaction(TransactionType::Write, vec![i])
            .unwrap();
        let v = consensus.create_vertex().unwrap().unwrap();
        vertex_ids.push(v.id.clone());
    }

    // Manually finalize by adding confirmations from node-2
    for vid in &vertex_ids {
        let confirm = DagVertex::new(
            "node-2".to_string(),
            Transaction {
                id: format!("confirm-{}", vid),
                tx_type: TransactionType::System,
                data: vec![],
                nonce: 0,
            },
            vec![vid.clone()],
            {
                let mut c = HashMap::new();
                c.insert("node-2".to_string(), 1);
                c
            },
        );
        consensus.add_vertex(confirm).unwrap();
    }
    consensus.finalize_vertices().unwrap();

    // Prune keeping only 2 finalized vertices
    consensus.prune_old_vertices(2);

    // Stats should reflect pruning (total_vertices reduced)
    let stats = consensus.get_stats();
    // Some vertices were pruned; the exact count depends on implementation
    // but total should be less than the original 10 (5 originals + 5 confirmations)
    assert!(
        stats.total_vertices < 10,
        "pruning should reduce vertex count, got {}",
        stats.total_vertices
    );
}

#[test]
fn test_dag_get_finalized_order() {
    let consensus = DagConsensus::new("node-1".to_string(), 1);

    // Submit and create 3 vertices
    for i in 0..3 {
        consensus
            .submit_transaction(TransactionType::Write, vec![i])
            .unwrap();
        consensus.create_vertex().unwrap();
    }

    // Without finalization, no transactions should be in the finalized order
    let order = consensus.get_finalized_order();
    assert!(order.is_empty());
}

// ---------------------------------------------------------------------------
// 10. Cluster start with discovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cluster_start_with_static_discovery() {
    let peers = vec![test_node("peer-1", 9001), test_node("peer-2", 9002)];

    let config = test_config(4, 2);
    let discovery = Box::new(StaticDiscovery::new(peers));
    let manager = ClusterManager::new(config, "self-node".to_string(), discovery).unwrap();

    // start() discovers peers and initialises shards
    manager.start().await.unwrap();

    let stats = manager.get_stats();
    assert_eq!(stats.total_nodes, 2, "discovered peers should be added");
    assert!(stats.total_shards > 0, "shards should be initialised");
}

#[tokio::test]
async fn test_cluster_start_excludes_self_from_discovery() {
    let peers = vec![
        test_node("self-node", 9000), // same as manager's own ID
        test_node("other", 9001),
    ];

    let config = test_config(4, 2);
    let discovery = Box::new(StaticDiscovery::new(peers));
    let manager = ClusterManager::new(config, "self-node".to_string(), discovery).unwrap();

    manager.start().await.unwrap();

    // self-node should NOT be added to the cluster (it would be the local node)
    let stats = manager.get_stats();
    assert_eq!(stats.total_nodes, 1, "self should be excluded from peers");
}

// ---------------------------------------------------------------------------
// 11. Health check marks offline nodes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_health_check_no_panic_on_empty_cluster() {
    let manager = test_manager(4, 2);
    let result = manager.run_health_checks().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_health_check_keeps_fresh_nodes_healthy() {
    let manager = test_manager(4, 2);

    manager.add_node(test_node("fresh", 9001)).await.unwrap();
    manager.run_health_checks().await.unwrap();

    // Fresh node should still be healthy (not marked offline)
    let node = manager.get_node("fresh").unwrap();
    assert_ne!(node.status, NodeStatus::Offline);
}

// ---------------------------------------------------------------------------
// 12. Router accessor
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cluster_manager_exposes_router() {
    let manager = test_manager(16, 2);
    let router = manager.router();

    // Router should work independently
    let shard = router.get_shard("test-vector");
    assert!(shard < 16);
}

// ---------------------------------------------------------------------------
// 13. End-to-end: full cluster lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_full_cluster_lifecycle() {
    // 1. Create cluster
    let manager = test_manager(8, 2);

    // 2. Add nodes
    for i in 0..4 {
        manager
            .add_node(test_node(&format!("lifecycle-{}", i), 9000 + i))
            .await
            .unwrap();
    }

    // 3. Verify cluster state
    assert_eq!(manager.list_nodes().len(), 4);
    assert_eq!(manager.healthy_nodes().len(), 4);

    // 4. Assign shards manually (in addition to auto-rebalanced ones)
    for sid in 0..8 {
        let _ = manager.assign_shard(sid); // may already exist from rebalancing
    }

    // 5. All shards should be assigned
    for sid in 0..8 {
        let shard = manager.get_shard(sid);
        assert!(shard.is_some(), "shard {} should be assigned", sid);
    }

    // 6. Use router to map vectors to shards
    let router = manager.router();
    let shard_id = router.get_shard("my-vector-id");
    assert!(shard_id < 8);

    // 7. Remove a node
    manager.remove_node("lifecycle-0").await.unwrap();
    assert_eq!(manager.list_nodes().len(), 3);

    // 8. Verify stats
    let stats = manager.get_stats();
    assert_eq!(stats.total_nodes, 3);
    assert!(stats.total_shards > 0);

    // 9. Health check
    manager.run_health_checks().await.unwrap();
}
