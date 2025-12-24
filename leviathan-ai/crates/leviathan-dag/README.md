# Leviathan DAG - Bank-Grade Auditability System

A comprehensive DAG (Directed Acyclic Graph) implementation for tracking data lineage, ensuring auditability, and maintaining BCBS 239 compliance in financial systems.

## Features

### Core Capabilities

- **Content-Addressable Storage**: Each node is uniquely identified by its BLAKE3 content hash
- **Cryptographic Verification**: Merkle tree proofs and tamper detection
- **Data Lineage Tracking**: Full provenance tracking for regulatory compliance (BCBS 239)
- **Multiple Parent Support**: Track merges and complex data flows
- **Graph Export**: Visualization support via GraphViz DOT format

### Node Types

1. **DataNode**: Raw or processed data storage
2. **ComputeNode**: Transformation and calculation tracking
3. **ValidationNode**: Compliance and quality checks
4. **CheckpointNode**: Audit snapshots with Merkle roots

### Regulatory Compliance

Built specifically for **BCBS 239** (Basel Committee on Banking Supervision 239) requirements:

- ✅ **Accuracy**: Cryptographic hash verification
- ✅ **Completeness**: Full lineage tracking
- ✅ **Timeliness**: Timestamp on every node
- ✅ **Adaptability**: Flexible metadata and node types
- ✅ **Lineage**: Complete forward/backward tracing

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
leviathan-dag = "0.1.0"
```

## Quick Start

### Basic DAG Operations

```rust
use leviathan_dag::{AuditDag, NodeType, DataNode};

// Create a new DAG
let mut dag = AuditDag::new();

// Add a root data node
let root_id = dag.add_data_node(
    b"transaction data",
    vec![], // no parents
    serde_json::json!({"type": "transaction", "amount": 1000})
).unwrap();

// Add a child node
let child_id = dag.add_data_node(
    b"processed data",
    vec![root_id],
    serde_json::json!({"status": "validated"})
).unwrap();

// Verify the chain
assert!(dag.verify_chain(&child_id).unwrap());

// Get lineage
let lineage = dag.get_lineage(&child_id).unwrap();
println!("Node has {} ancestors", lineage.len());
```

### Data Lineage Tracking

```rust
use leviathan_dag::{AuditDag, TransformationType};

let mut dag = AuditDag::new();

// Create source data
let source_id = dag.add_data_node(
    b"raw customer data",
    vec![],
    serde_json::json!({})
).unwrap();

// Apply transformation
let filtered_id = dag.add_compute_node(
    "filter_active_customers".to_string(),
    b"filtered data",
    vec![source_id],
    serde_json::json!({"filter": "status = 'active'"})
).unwrap();

// Track the transformation
let tracker = dag.lineage_tracker_mut();
tracker.record_flow(
    source_id,
    filtered_id,
    TransformationType::Filter,
    Some("Remove inactive customers".to_string()),
    serde_json::json!({"condition": "status = 'active'"})
);

// Query lineage
let inputs = tracker.get_inputs(&filtered_id);
println!("Transformation has {} inputs", inputs.len());

// Get full lineage
let full_lineage = tracker.get_full_lineage(&filtered_id);
println!("Full lineage: {} nodes", full_lineage.len());
```

### Compliance Validation

```rust
use leviathan_dag::{AuditDag, DagNode, NodeType, ValidationNode, ValidationType};

let mut dag = AuditDag::new();

// Source data
let data_id = dag.add_data_node(
    b"financial report",
    vec![],
    serde_json::json!({})
).unwrap();

// Add validation
let validation_id = dag.add_node(DagNode::new(
    b"BCBS 239 compliance check passed".to_vec(),
    vec![data_id],
    serde_json::json!({
        "rules": ["accuracy", "completeness", "timeliness"]
    }),
    NodeType::Validation(ValidationNode::passed(ValidationType::Compliance))
)).unwrap();

// Verify everything
assert!(dag.verify_chain(&validation_id).unwrap());
```

### Merkle Proofs

```rust
use leviathan_dag::verify::{compute_merkle_root, generate_proof};

// Collect node hashes
let hashes = vec![
    "hash1".to_string(),
    "hash2".to_string(),
    "hash3".to_string(),
    "hash4".to_string(),
];

// Compute Merkle root
let root = compute_merkle_root(&hashes);

// Generate proof for specific node
let proof = generate_proof(&hashes, 2).unwrap();

// Verify proof
assert!(proof.verify());
```

### Tamper Detection

```rust
use leviathan_dag::verify::TamperDetector;
use uuid::Uuid;

let mut detector = TamperDetector::new();

// Establish baseline
let id1 = Uuid::new_v4();
let id2 = Uuid::new_v4();

detector.create_baseline(vec![
    (id1, "hash1".to_string()),
    (id2, "hash2".to_string()),
]);

// Check for tampering
let current = vec![
    (id1, "hash1_modified".to_string()), // Modified!
    (id2, "hash2".to_string()),
];

let report = detector.detect_tampering(&current);
if report.has_tampering() {
    println!("Tampering detected!");
    println!("Modified: {}", report.modified.len());
    println!("Added: {}", report.added.len());
    println!("Removed: {}", report.removed.len());
}
```

## Architecture

### Module Structure

```
leviathan-dag/
├── src/
│   ├── lib.rs          # Main DAG structure (AuditDag, DagNode)
│   ├── node.rs         # Node types and hashing
│   ├── lineage.rs      # Data lineage tracking
│   └── verify.rs       # Cryptographic verification
└── tests/
    └── integration_tests.rs  # Comprehensive tests
```

### Key Components

#### 1. DagNode

```rust
pub struct DagNode {
    pub id: Uuid,                      // Unique identifier
    pub hash: String,                  // BLAKE3 content hash
    pub timestamp: DateTime<Utc>,      // Creation time
    pub parent_ids: Vec<Uuid>,         // Parent nodes
    pub data: Vec<u8>,                 // Node data
    pub metadata: serde_json::Value,   // Additional context
    pub node_type: NodeType,           // Type of node
}
```

#### 2. AuditDag

The main DAG structure with:
- Petgraph-based directed graph
- UUID to NodeIndex mapping
- Integrated lineage tracker
- Cycle detection
- Chain verification

#### 3. LineageTracker

Tracks data transformations:
- Forward trace (impact analysis)
- Backward trace (lineage)
- Transformation chain recording
- BCBS 239 compliance reports

#### 4. Cryptographic Verification

- Merkle tree construction
- Proof generation/verification
- Tamper detection
- Batch verification

## Testing

The crate includes **34 comprehensive tests**:

- 24 unit tests across all modules
- 9 integration tests for real-world scenarios
- 1 documentation test

Run tests:

```bash
cargo test
```

Run specific test:

```bash
cargo test test_bcbs_239_compliance
```

## Performance

- **Hashing**: BLAKE3 (faster than SHA-256)
- **Graph Operations**: Petgraph (optimized for DAGs)
- **Serialization**: rkyv (zero-copy deserialization)
- **Indexing**: O(1) node lookup via HashMap

## Use Cases

### Financial Services

- Regulatory reporting (BCBS 239)
- Transaction audit trails
- Risk calculation provenance
- Data quality tracking

### Healthcare

- Patient data lineage
- Clinical trial auditing
- HIPAA compliance tracking

### Supply Chain

- Product provenance
- Quality control tracking
- Compliance verification

### General

- Any system requiring:
  - Immutable audit logs
  - Data lineage tracking
  - Cryptographic verification
  - Regulatory compliance

## API Reference

### Main Types

- `AuditDag` - Main DAG structure
- `DagNode` - Node in the DAG
- `NodeType` - Data, Compute, Validation, Checkpoint
- `LineageTracker` - Data flow tracking
- `MerkleProof` - Cryptographic proofs

### Key Methods

#### AuditDag

- `new()` - Create empty DAG
- `add_node(node)` - Add node to DAG
- `add_data_node(data, parents, metadata)` - Convenience for data nodes
- `add_compute_node(op, result, parents, metadata)` - Convenience for compute nodes
- `get_node(id)` - Retrieve node
- `get_lineage(id)` - Get all ancestors
- `verify_chain(id)` - Verify integrity
- `export_graphviz()` - Export to DOT format
- `stats()` - Get DAG statistics

#### LineageTracker

- `record_flow(source, dest, transformation, desc, metadata)` - Record transformation
- `get_inputs(node_id)` - What produced this output?
- `get_outputs(node_id)` - What depends on this input?
- `get_full_lineage(node_id)` - All ancestors
- `get_full_impact(node_id)` - All descendants
- `generate_lineage_report(node_id)` - BCBS 239 report

## License

MIT

## Contributing

Issues and pull requests welcome at [https://github.com/ruvnet/leviathan-ai](https://github.com/ruvnet/leviathan-ai)

## Authors

Leviathan AI Team

---

**Built for bank-grade auditability and regulatory compliance.**
