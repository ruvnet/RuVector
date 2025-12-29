//! DAG integration tests

use ruvector_dag::dag::{QueryDag, OperatorNode, OperatorType};

#[test]
fn test_complex_query_dag() {
    // Build a realistic query DAG
    let mut dag = QueryDag::new();

    // Add scan nodes
    let scan1 = dag.add_node(OperatorNode::seq_scan(0, "users"));
    let scan2 = dag.add_node(OperatorNode::hnsw_scan(1, "vectors_idx", 64));

    // Add join
    let join = dag.add_node(OperatorNode::hash_join(2, "user_id"));
    dag.add_edge(scan1, join).unwrap();
    dag.add_edge(scan2, join).unwrap();

    // Add filter and result
    let filter = dag.add_node(OperatorNode::filter(3, "score > 0.5"));
    dag.add_edge(join, filter).unwrap();

    let result = dag.add_node(OperatorNode::new(4, OperatorType::Result));
    dag.add_edge(filter, result).unwrap();

    // Verify structure
    assert_eq!(dag.node_count(), 5);
    assert_eq!(dag.edge_count(), 4);

    // Verify topological order
    let order = dag.topological_sort().unwrap();
    assert_eq!(order.len(), 5);

    // Scans should come before join
    let scan1_pos = order.iter().position(|&x| x == scan1).unwrap();
    let scan2_pos = order.iter().position(|&x| x == scan2).unwrap();
    let join_pos = order.iter().position(|&x| x == join).unwrap();

    assert!(scan1_pos < join_pos);
    assert!(scan2_pos < join_pos);
}

#[test]
fn test_dag_serialization_roundtrip() {
    let mut dag = QueryDag::new();

    for i in 0..10 {
        dag.add_node(OperatorNode::new(i, OperatorType::SeqScan {
            table: format!("table_{}", i)
        }));
    }

    // Create chain
    for i in 0..9 {
        dag.add_edge(i, i + 1).unwrap();
    }

    // Serialize and deserialize
    let json = dag.to_json().unwrap();
    let restored = QueryDag::from_json(&json).unwrap();

    assert_eq!(dag.node_count(), restored.node_count());
    assert_eq!(dag.edge_count(), restored.edge_count());
}

#[test]
fn test_dag_depths() {
    let mut dag = QueryDag::new();

    // Create tree structure
    //       0
    //      / \
    //     1   2
    //    / \
    //   3   4

    for i in 0..5 {
        dag.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    dag.add_edge(3, 1).unwrap();
    dag.add_edge(4, 1).unwrap();
    dag.add_edge(1, 0).unwrap();
    dag.add_edge(2, 0).unwrap();

    let depths = dag.compute_depths();

    assert_eq!(depths[&3], 0);
    assert_eq!(depths[&4], 0);
    assert_eq!(depths[&2], 0);
    assert_eq!(depths[&1], 1);
    assert_eq!(depths[&0], 2);
}

#[test]
fn test_dag_cycle_detection() {
    let mut dag = QueryDag::new();

    for i in 0..3 {
        dag.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    // Create valid edges
    dag.add_edge(0, 1).unwrap();
    dag.add_edge(1, 2).unwrap();

    // Attempt to create cycle should fail
    let result = dag.add_edge(2, 0);
    assert!(result.is_err());
}

#[test]
fn test_dag_node_removal() {
    let mut dag = QueryDag::new();

    for i in 0..5 {
        dag.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    dag.add_edge(0, 1).unwrap();
    dag.add_edge(1, 2).unwrap();
    dag.add_edge(2, 3).unwrap();
    dag.add_edge(3, 4).unwrap();

    // Remove middle node
    dag.remove_node(2);

    assert_eq!(dag.node_count(), 4);
    // Edges connected to node 2 should be removed
    assert!(!dag.has_edge(1, 2));
    assert!(!dag.has_edge(2, 3));
}

#[test]
fn test_dag_subgraph_extraction() {
    let mut dag = QueryDag::new();

    // Create larger graph
    for i in 0..10 {
        dag.add_node(OperatorNode::new(i, OperatorType::SeqScan {
            table: format!("t{}", i)
        }));
    }

    // Create edges
    for i in 0..9 {
        dag.add_edge(i, i + 1).unwrap();
    }

    // Extract subgraph
    let nodes = vec![2, 3, 4, 5];
    let subgraph = dag.extract_subgraph(&nodes);

    assert_eq!(subgraph.node_count(), 4);
}

#[test]
fn test_dag_merge() {
    let mut dag1 = QueryDag::new();
    let mut dag2 = QueryDag::new();

    for i in 0..3 {
        dag1.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    for i in 3..6 {
        dag2.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    dag1.add_edge(0, 1).unwrap();
    dag1.add_edge(1, 2).unwrap();

    dag2.add_edge(3, 4).unwrap();
    dag2.add_edge(4, 5).unwrap();

    // Merge dag2 into dag1
    dag1.merge(&dag2);

    assert_eq!(dag1.node_count(), 6);
}

#[test]
fn test_dag_clone() {
    let mut dag = QueryDag::new();

    for i in 0..5 {
        dag.add_node(OperatorNode::new(i, OperatorType::Result));
    }

    for i in 0..4 {
        dag.add_edge(i, i + 1).unwrap();
    }

    let cloned = dag.clone();

    assert_eq!(dag.node_count(), cloned.node_count());
    assert_eq!(dag.edge_count(), cloned.edge_count());
}
