//! Integration tests for the leviathan-dag crate

use leviathan_dag::*;
use leviathan_dag::node::ValidationType;
use leviathan_dag::verify::generate_proof;
use uuid::Uuid;

#[test]
fn test_complete_audit_workflow() {
    let mut dag = AuditDag::new();

    // Step 1: Ingest raw data
    let raw_data_id = dag.add_data_node(
        b"raw transaction data",
        vec![],
        serde_json::json!({
            "source": "api",
            "timestamp": "2024-01-01T00:00:00Z"
        }),
    ).unwrap();

    // Step 2: Validate data
    let validation_id = dag.add_node(DagNode::new(
        b"validation result: passed".to_vec(),
        vec![raw_data_id],
        serde_json::json!({
            "rules": ["schema_check", "business_rules"]
        }),
        NodeType::Validation(ValidationNode::passed(ValidationType::Schema)),
    )).unwrap();

    // Step 3: Transform data
    let transform_id = dag.add_compute_node(
        "aggregate_by_customer".to_string(),
        b"aggregated data",
        vec![validation_id],
        serde_json::json!({
            "aggregation": "sum"
        }),
    ).unwrap();

    // Step 4: Create checkpoint
    let merkle_root = compute_merkle_root(&[
        dag.get_node(&raw_data_id).unwrap().hash.clone(),
        dag.get_node(&validation_id).unwrap().hash.clone(),
        dag.get_node(&transform_id).unwrap().hash.clone(),
    ]);

    let checkpoint_id = dag.add_node(DagNode::new(
        b"monthly checkpoint".to_vec(),
        vec![transform_id],
        serde_json::json!({}),
        NodeType::Checkpoint(CheckpointNode::new(
            "January 2024 Audit",
            merkle_root,
            3,
        )),
    )).unwrap();

    // Verify the complete chain
    assert!(dag.verify_chain(&checkpoint_id).unwrap());

    // Check lineage
    let lineage = dag.get_lineage(&checkpoint_id).unwrap();
    assert_eq!(lineage.len(), 4);
    assert!(lineage.contains(&raw_data_id));
    assert!(lineage.contains(&validation_id));
    assert!(lineage.contains(&transform_id));
    assert!(lineage.contains(&checkpoint_id));
}

#[test]
fn test_multi_parent_merge() {
    let mut dag = AuditDag::new();

    // Create two separate data streams
    let stream1_id = dag.add_data_node(
        b"customer data",
        vec![],
        serde_json::json!({"stream": "customers"}),
    ).unwrap();

    let stream2_id = dag.add_data_node(
        b"transaction data",
        vec![],
        serde_json::json!({"stream": "transactions"}),
    ).unwrap();

    // Merge them
    let merged_id = dag.add_compute_node(
        "join_on_customer_id".to_string(),
        b"joined data",
        vec![stream1_id, stream2_id],
        serde_json::json!({
            "join_type": "inner",
            "key": "customer_id"
        }),
    ).unwrap();

    // Verify lineage includes both parents
    let lineage = dag.get_lineage(&merged_id).unwrap();
    assert!(lineage.contains(&stream1_id));
    assert!(lineage.contains(&stream2_id));
    assert!(lineage.contains(&merged_id));
}

#[test]
fn test_data_lineage_tracking() {
    let mut dag = AuditDag::new();
    let tracker = dag.lineage_tracker_mut();

    // Create a data flow pipeline
    let source_id = Uuid::new_v4();
    let filter_id = Uuid::new_v4();
    let aggregate_id = Uuid::new_v4();

    // Track transformations
    tracker.record_flow(
        source_id,
        filter_id,
        TransformationType::Filter,
        Some("Remove invalid records".to_string()),
        serde_json::json!({"condition": "amount > 0"}),
    );

    tracker.record_flow(
        filter_id,
        aggregate_id,
        TransformationType::Aggregation,
        Some("Sum by category".to_string()),
        serde_json::json!({"group_by": "category"}),
    );

    // Query inputs
    let inputs = tracker.get_inputs(&aggregate_id);
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].source, filter_id);
    assert_eq!(inputs[0].transformation, TransformationType::Aggregation);

    // Query outputs
    let outputs = tracker.get_outputs(&source_id);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].destination, filter_id);

    // Get full lineage
    let lineage = tracker.get_full_lineage(&aggregate_id);
    assert!(lineage.contains(&source_id));
    assert!(lineage.contains(&filter_id));
    assert!(lineage.contains(&aggregate_id));
}

#[test]
fn test_bcbs_239_compliance() {
    let mut dag = AuditDag::new();

    // Simulate a regulatory reporting pipeline
    let source_id = dag.add_data_node(
        b"raw risk data",
        vec![],
        serde_json::json!({
            "source_system": "risk_engine",
            "data_quality": "high"
        }),
    ).unwrap();

    // Data quality validation
    let validation_id = dag.add_node(DagNode::new(
        b"validation passed".to_vec(),
        vec![source_id],
        serde_json::json!({
            "checks": ["completeness", "accuracy", "consistency"]
        }),
        NodeType::Validation(ValidationNode::passed(ValidationType::Compliance)),
    )).unwrap();

    // Risk calculation
    let calc_id = dag.add_compute_node(
        "calculate_var".to_string(),
        b"VaR: 1.23M",
        vec![validation_id],
        serde_json::json!({
            "method": "historical_simulation",
            "confidence_level": 0.99
        }),
    ).unwrap();

    // Record lineage
    let tracker = dag.lineage_tracker_mut();
    tracker.record_flow(
        source_id,
        validation_id,
        TransformationType::Validation,
        Some("BCBS 239 compliance check".to_string()),
        serde_json::json!({"standard": "bcbs_239"}),
    );

    tracker.record_flow(
        validation_id,
        calc_id,
        TransformationType::Calculation,
        Some("Value at Risk calculation".to_string()),
        serde_json::json!({"model": "historical"}),
    );

    // Generate audit report
    let report = tracker.generate_lineage_report(&calc_id);
    assert!(report.total_ancestors >= 3);
    assert!(report.transformation_counts.contains_key(&TransformationType::Validation));
    assert!(report.transformation_counts.contains_key(&TransformationType::Calculation));
}

#[test]
fn test_merkle_proof_verification() {
    // Create a set of hashes
    let hashes = vec![
        "hash1".to_string(),
        "hash2".to_string(),
        "hash3".to_string(),
        "hash4".to_string(),
        "hash5".to_string(),
    ];

    // Generate proofs for each leaf
    for i in 0..hashes.len() {
        let proof = generate_proof(&hashes, i).unwrap();
        assert!(proof.verify(), "Proof for index {} should verify", i);
    }
}

#[test]
fn test_tamper_detection() {
    use leviathan_dag::verify::TamperDetector;

    let mut detector = TamperDetector::new();

    // Establish baseline
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    let baseline = vec![
        (id1, "hash1".to_string()),
        (id2, "hash2".to_string()),
        (id3, "hash3".to_string()),
    ];

    detector.create_baseline(baseline);

    // Simulate tampering
    let current = vec![
        (id1, "hash1_tampered".to_string()), // Modified
        (id2, "hash2".to_string()),           // Unchanged
        // id3 removed
        (Uuid::new_v4(), "hash4".to_string()), // Added
    ];

    let report = detector.detect_tampering(&current);
    assert!(report.has_tampering());
    assert_eq!(report.modified.len(), 1);
    assert_eq!(report.removed.len(), 1);
    assert_eq!(report.added.len(), 1);
}

#[test]
fn test_dag_statistics() {
    let mut dag = AuditDag::new();

    // Create a small DAG
    let root = dag.add_data_node(
        b"root",
        vec![],
        serde_json::json!({}),
    ).unwrap();

    let child1 = dag.add_data_node(
        b"child1",
        vec![root],
        serde_json::json!({}),
    ).unwrap();

    let child2 = dag.add_data_node(
        b"child2",
        vec![root],
        serde_json::json!({}),
    ).unwrap();

    let _grandchild = dag.add_data_node(
        b"grandchild",
        vec![child1, child2],
        serde_json::json!({}),
    ).unwrap();

    let stats = dag.stats();
    assert_eq!(stats.total_nodes, 4);
    assert_eq!(stats.root_nodes, 1);
    assert_eq!(stats.leaf_nodes, 1);
}

#[test]
fn test_graphviz_export() {
    let mut dag = AuditDag::new();

    let root = dag.add_data_node(
        b"root",
        vec![],
        serde_json::json!({}),
    ).unwrap();

    let _child = dag.add_data_node(
        b"child",
        vec![root],
        serde_json::json!({}),
    ).unwrap();

    let dot = dag.export_graphviz();
    assert!(!dot.is_empty());
    assert!(dot.contains("digraph"));
}

#[test]
fn test_concurrent_lineage_queries() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let mut dag = AuditDag::new();

    // Create a complex DAG
    let mut nodes = Vec::new();
    for i in 0..10 {
        let parent_ids = if i > 0 { vec![nodes[i - 1]] } else { vec![] };
        let id = dag.add_data_node(
            format!("node_{}", i).as_bytes(),
            parent_ids,
            serde_json::json!({"index": i}),
        ).unwrap();
        nodes.push(id);
    }

    // Share DAG across threads for queries
    let dag = Arc::new(Mutex::new(dag));
    let mut handles = vec![];

    for node_id in nodes.iter().take(5) {
        let dag_clone = Arc::clone(&dag);
        let node_id = *node_id;

        let handle = thread::spawn(move || {
            let dag = dag_clone.lock().unwrap();
            let lineage = dag.get_lineage(&node_id).unwrap();
            lineage.len()
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let lineage_len = handle.join().unwrap();
        assert!(lineage_len > 0);
    }
}
