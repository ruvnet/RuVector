//! Node.js bindings for RuVector Graph Database via NAPI-RS
//!
//! High-performance native graph database with Cypher-like query support,
//! hypergraph capabilities, async/await support, and zero-copy buffer sharing.

#![allow(clippy::all)]
#![allow(clippy::pedantic)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_core::advanced::hypergraph::{
    CausalMemory as CoreCausalMemory, Hyperedge as CoreHyperedge,
    HypergraphIndex as CoreHypergraphIndex,
};
use ruvector_core::DistanceMetric;
use ruvector_graph::cypher::{parse_cypher, Statement};
use ruvector_graph::edge::Edge as GraphEdge;
use ruvector_graph::hyperedge::Hyperedge as GraphHyperedge;

/// Extract an f32 vector from a persisted property. Tolerates both the
/// compact FloatArray form and the boxed Array/List form produced by the
/// generic From<Vec<T>> impl (which older writes used).
fn prop_to_f32_vec(p: Option<&PropertyValue>) -> Vec<f32> {
    match p {
        Some(PropertyValue::FloatArray(v)) => v.clone(),
        Some(PropertyValue::Array(items)) | Some(PropertyValue::List(items)) => items
            .iter()
            .filter_map(|x| match x {
                PropertyValue::Float(f) => Some(*f as f32),
                PropertyValue::Integer(i) => Some(*i as f32),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
use ruvector_graph::node::NodeBuilder;
use ruvector_graph::types::PropertyValue;
use ruvector_graph::storage::GraphStorage;
use ruvector_graph::GraphDB;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

mod streaming;
mod transactions;
mod types;

pub use streaming::*;
pub use transactions::*;
pub use types::{
    JsBatchInsert, JsBatchResult, JsDeleteNodeOptions, JsDeleteNodeResult, JsDeleteResult,
    JsDistanceMetric, JsEdge, JsEdgeResult, JsGraphOptions, JsGraphStats, JsHyperedge,
    JsHyperedgeQuery, JsHyperedgeResult, JsNode, JsNodeResult, JsQueryResult,
    JsTemporalGranularity, JsTemporalHyperedge,
};

/// Graph database for complex relationship queries
#[napi]
pub struct GraphDatabase {
    hypergraph: Arc<RwLock<CoreHypergraphIndex>>,
    causal_memory: Arc<RwLock<CoreCausalMemory>>,
    transaction_manager: Arc<RwLock<transactions::TransactionManager>>,
    /// Property graph database with Cypher support
    graph_db: Arc<RwLock<GraphDB>>,
    /// Persistent storage backend (optional)
    storage: Option<Arc<RwLock<GraphStorage>>>,
    /// Path to storage file (if persisted)
    storage_path: Option<String>,
}

/// Register a node into BOTH backing indexes so they stay consistent:
///   1. the hypergraph adjacency / vector index (used by `kHopNeighbors`,
///      `stats`, and `searchHyperedges`), and
///   2. the property graph + label index (used by the Cypher
///      `MATCH (n:Label) RETURN n` label scan).
///
/// This is the single source of truth shared by `create_node` and
/// `batch_insert`. Previously `batch_insert` only populated the hypergraph,
/// so bulk-loaded nodes were counted in `stats()` and traversable by
/// `kHopNeighbors` but invisible to the label-scoped Cypher scan.
fn register_node(
    hg: &mut CoreHypergraphIndex,
    gdb: &mut GraphDB,
    storage: Option<&Arc<RwLock<GraphStorage>>>,
    id: String,
    embedding: Vec<f32>,
    labels: Option<Vec<String>>,
    properties: Option<HashMap<String, String>>,
) -> Result<()> {
    // 1. Adjacency / vector index (kHop, stats, hyperedge search).
    hg.add_entity(id.clone(), embedding.clone());

    // 2. Property graph + label index (Cypher label scan).
    let mut builder = NodeBuilder::new().id(&id);
    if let Some(node_labels) = labels {
        for label in node_labels {
            builder = builder.label(&label);
        }
    }
    if let Some(props) = properties {
        for (key, value) in props {
            builder = builder.property(&key, value);
        }
    }
    // Persist the vector so hydrate_from_storage can rebuild the hypergraph index.
    builder = builder.property("__embedding", PropertyValue::FloatArray(embedding));
    let graph_node = builder.build();

    // Persist to storage if enabled (mirrors create_node behaviour).
    if let Some(storage_arc) = storage {
        let storage_guard = storage_arc.write().expect("Storage RwLock poisoned");
        storage_guard
            .insert_node(&graph_node)
            .map_err(|e| Error::from_reason(format!("Failed to persist node: {}", e)))?;
    }

    gdb.create_node(graph_node)
        .map_err(|e| Error::from_reason(format!("Failed to create node: {}", e)))?;

    Ok(())
}

#[napi]
impl GraphDatabase {
    /// Create a new graph database
    ///
    /// # Example
    /// ```javascript
    /// const db = new GraphDatabase({
    ///   distanceMetric: 'Cosine',
    ///   dimensions: 384
    /// });
    /// ```
    #[napi(constructor)]
    pub fn new(options: Option<JsGraphOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let metric = opts.distance_metric.unwrap_or(JsDistanceMetric::Cosine);
        let core_metric: DistanceMetric = metric.into();

        // Check if storage path is provided for persistence
        let (storage, storage_path) = if let Some(ref path) = opts.storage_path {
            let gs = GraphStorage::new(path)
                .map_err(|e| Error::from_reason(format!("Failed to open storage: {}", e)))?;
            (Some(Arc::new(RwLock::new(gs))), Some(path.clone()))
        } else {
            (None, None)
        };

        let db = Self {
            hypergraph: Arc::new(RwLock::new(CoreHypergraphIndex::new(core_metric))),
            causal_memory: Arc::new(RwLock::new(CoreCausalMemory::new(core_metric))),
            transaction_manager: Arc::new(RwLock::new(transactions::TransactionManager::new())),
            graph_db: Arc::new(RwLock::new(GraphDB::new())),
            storage,
            storage_path,
        };
        db.hydrate_from_storage()?;
        Ok(db)
    }

    /// Open an existing graph database from disk
    ///
    /// # Example
    /// ```javascript
    /// const db = GraphDatabase.open('./my-graph.db');
    /// ```
    #[napi(factory)]
    pub fn open(path: String) -> Result<Self> {
        let storage = GraphStorage::new(&path)
            .map_err(|e| Error::from_reason(format!("Failed to open storage: {}", e)))?;

        let metric = DistanceMetric::Cosine;

        let db = Self {
            hypergraph: Arc::new(RwLock::new(CoreHypergraphIndex::new(metric))),
            causal_memory: Arc::new(RwLock::new(CoreCausalMemory::new(metric))),
            transaction_manager: Arc::new(RwLock::new(transactions::TransactionManager::new())),
            graph_db: Arc::new(RwLock::new(GraphDB::new())),
            storage: Some(Arc::new(RwLock::new(storage))),
            storage_path: Some(path),
        };
        db.hydrate_from_storage()?;
        Ok(db)
    }

    /// Check if persistence is enabled
    ///
    /// # Example
    /// ```javascript
    /// if (db.isPersistent()) {
    ///   console.log('Data is being saved to:', db.getStoragePath());
    /// }
    /// ```
    #[napi]
    pub fn is_persistent(&self) -> bool {
        self.storage.is_some()
    }

    /// Get the storage path (if persisted)
    #[napi]
    pub fn get_storage_path(&self) -> Option<String> {
        self.storage_path.clone()
    }

    /// Replay persisted nodes and edges into the in-memory indexes.
    ///
    /// GraphStorage is write-through, but every index starts empty on
    /// construction, so without this replay a reopened database reports zero
    /// nodes. Multi-node hyperedges are not yet persisted.
    fn hydrate_from_storage(&self) -> Result<()> {
        let Some(storage_arc) = self.storage.clone() else {
            return Ok(());
        };
        let storage = storage_arc.read().expect("Storage RwLock poisoned");
        let mut hg = self.hypergraph.write().expect("RwLock poisoned");
        let gdb = self.graph_db.write().expect("RwLock poisoned");

        let node_ids = storage
            .all_node_ids()
            .map_err(|e| Error::from_reason(format!("hydrate nodes: {}", e)))?;
        for id in node_ids {
            let Some(node) = storage
                .get_node(&id)
                .map_err(|e| Error::from_reason(format!("hydrate node {}: {}", id, e)))?
            else {
                continue;
            };
            let embedding = prop_to_f32_vec(node.properties.get("__embedding"));
            hg.add_entity(node.id.clone(), embedding);
            gdb.create_node(node)
                .map_err(|e| Error::from_reason(format!("hydrate node insert: {}", e)))?;
        }

        let mut dangling: usize = 0;
        let edge_ids = storage
            .all_edge_ids()
            .map_err(|e| Error::from_reason(format!("hydrate edges: {}", e)))?;
        for id in edge_ids {
            let Some(edge) = storage
                .get_edge(&id)
                .map_err(|e| Error::from_reason(format!("hydrate edge {}: {}", id, e)))?
            else {
                continue;
            };
            let confidence = match edge.properties.get("__confidence") {
                Some(PropertyValue::FloatArray(v)) => v.first().copied().unwrap_or(1.0),
                _ => 1.0,
            };
            let embedding = prop_to_f32_vec(edge.properties.get("__embedding"));
            let mut core_edge = CoreHyperedge::new(
                vec![edge.from.clone(), edge.to.clone()],
                edge.edge_type.clone(),
                embedding,
                confidence,
            );
            core_edge.id = edge.id.clone();
            if hg.add_hyperedge(core_edge).is_err() {
                dangling += 1;
                continue;
            }
            if gdb.create_edge(edge).is_err() {
                dangling += 1;
            }
        }

        let hyperedge_ids = storage
            .all_hyperedge_ids()
            .map_err(|e| Error::from_reason(format!("hydrate hyperedges: {}", e)))?;
        for id in hyperedge_ids {
            let Some(he) = storage
                .get_hyperedge(&id)
                .map_err(|e| Error::from_reason(format!("hydrate hyperedge {}: {}", id, e)))?
            else {
                continue;
            };
            let embedding = prop_to_f32_vec(he.properties.get("__embedding"));
            let mut core_edge = CoreHyperedge::new(
                he.nodes.clone(),
                he.description.clone().unwrap_or_else(|| he.edge_type.clone()),
                embedding,
                he.confidence,
            );
            core_edge.id = he.id.clone();
            if hg.add_hyperedge(core_edge).is_err() {
                dangling += 1;
            }
        }

        if dangling > 0 {
            eprintln!(
                "[graph-node] hydrate: skipped {} dangling edge/hyperedge record(s)",
                dangling
            );
        }

        Ok(())
    }

    /// Create a node in the graph
    ///
    /// # Example
    /// ```javascript
    /// const nodeId = await db.createNode({
    ///   id: 'node1',
    ///   embedding: new Float32Array([1, 2, 3]),
    ///   properties: { name: 'Alice', age: 30 }
    /// });
    /// ```
    #[napi]
    pub async fn create_node(&self, node: JsNode) -> Result<String> {
        let hypergraph = self.hypergraph.clone();
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let id = node.id.clone();
        let embedding = node.embedding.to_vec();
        let properties = node.properties.clone();
        let labels = node.labels.clone();

        tokio::task::spawn_blocking(move || {
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            let mut gdb = graph_db.write().expect("RwLock poisoned");

            register_node(
                &mut hg,
                &mut gdb,
                storage.as_ref(),
                id.clone(),
                embedding,
                labels,
                properties,
            )?;

            Ok::<String, Error>(id)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Create an edge between two nodes
    ///
    /// # Example
    /// ```javascript
    /// const edgeId = await db.createEdge({
    ///   from: 'node1',
    ///   to: 'node2',
    ///   description: 'knows',
    ///   embedding: new Float32Array([0.5, 0.5, 0.5]),
    ///   confidence: 0.95
    /// });
    /// ```
    #[napi]
    pub async fn create_edge(&self, edge: JsEdge) -> Result<String> {
        let hypergraph = self.hypergraph.clone();
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let from = edge.from.clone();
        let to = edge.to.clone();
        let nodes = vec![edge.from.clone(), edge.to.clone()];
        let description = edge.description.clone();
        let embedding = edge.embedding.to_vec();
        let confidence = edge.confidence.unwrap_or(1.0) as f32;

        tokio::task::spawn_blocking(move || {
            let embedding_for_store = embedding.clone();
            let core_edge = CoreHyperedge::new(nodes, description.clone(), embedding, confidence);
            let edge_id = core_edge.id.clone();
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            hg.add_hyperedge(core_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create edge: {}", e)))?;
            drop(hg);

            // Write-through to the property graph and persistent storage so the
            // edge survives a restart (mirrors register_node for nodes).
            let mut props = std::collections::HashMap::new();
            props.insert(
                "__confidence".to_string(),
                PropertyValue::FloatArray(vec![confidence]),
            );
            props.insert(
                "__embedding".to_string(),
                PropertyValue::FloatArray(embedding_for_store),
            );
            let graph_edge = GraphEdge::new(edge_id.clone(), from, to, description, props);
            if let Some(storage_arc) = storage {
                let sg = storage_arc.write().expect("Storage RwLock poisoned");
                sg.insert_edge(&graph_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to persist edge: {}", e)))?;
            }
            let gdb = graph_db.read().expect("RwLock poisoned");
            gdb.create_edge(graph_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create edge: {}", e)))?;

            Ok(edge_id)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Create a hyperedge connecting multiple nodes
    ///
    /// # Example
    /// ```javascript
    /// const hyperedgeId = await db.createHyperedge({
    ///   nodes: ['node1', 'node2', 'node3'],
    ///   description: 'collaborated_on_project',
    ///   embedding: new Float32Array([0.3, 0.6, 0.9]),
    ///   confidence: 0.85,
    ///   metadata: { project: 'AI Research' }
    /// });
    /// ```
    #[napi]
    pub async fn create_hyperedge(&self, hyperedge: JsHyperedge) -> Result<String> {
        let hypergraph = self.hypergraph.clone();
        let storage = self.storage.clone();
        let nodes = hyperedge.nodes.clone();
        let description = hyperedge.description.clone();
        let embedding = hyperedge.embedding.to_vec();
        let confidence = hyperedge.confidence.unwrap_or(1.0) as f32;

        tokio::task::spawn_blocking(move || {
            let core_edge =
                CoreHyperedge::new(nodes.clone(), description.clone(), embedding.clone(), confidence);
            let edge_id = core_edge.id.clone();
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            hg.add_hyperedge(core_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create hyperedge: {}", e)))?;
            drop(hg);

            // Persist so hydrate_from_storage can replay multi-node hyperedges.
            if let Some(storage_arc) = storage {
                let mut props = std::collections::HashMap::new();
                props.insert(
                    "__confidence".to_string(),
                    PropertyValue::FloatArray(vec![confidence]),
                );
                props.insert("__embedding".to_string(), PropertyValue::FloatArray(embedding));
                let graph_hyperedge = GraphHyperedge {
                    id: edge_id.clone(),
                    nodes,
                    edge_type: "HYPEREDGE".to_string(),
                    description: Some(description),
                    properties: props,
                    confidence,
                };
                let sg = storage_arc.write().expect("Storage RwLock poisoned");
                sg.insert_hyperedge(&graph_hyperedge)
                    .map_err(|e| Error::from_reason(format!("Failed to persist hyperedge: {}", e)))?;
            }

            Ok(edge_id)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Query the graph using Cypher-like syntax
    ///
    /// # Example
    /// ```javascript
    /// const results = await db.query('MATCH (n) RETURN n LIMIT 10');
    /// ```
    #[napi]
    pub async fn query(&self, cypher: String) -> Result<JsQueryResult> {
        let graph_db = self.graph_db.clone();
        let hypergraph = self.hypergraph.clone();

        tokio::task::spawn_blocking(move || {
            // Parse the Cypher query
            let parsed = parse_cypher(&cypher)
                .map_err(|e| Error::from_reason(format!("Cypher parse error: {}", e)))?;

            let gdb = graph_db.read().expect("RwLock poisoned");
            let hg = hypergraph.read().expect("RwLock poisoned");

            let mut result_nodes: Vec<JsNodeResult> = Vec::new();
            let mut result_edges: Vec<JsEdgeResult> = Vec::new();

            // Execute each statement
            for statement in &parsed.statements {
                match statement {
                    Statement::Match(match_clause) => {
                        // Extract label from match patterns for query
                        for pattern in &match_clause.patterns {
                            if let ruvector_graph::cypher::ast::Pattern::Node(node_pattern) =
                                pattern
                            {
                                for label in &node_pattern.labels {
                                    let nodes = gdb.get_nodes_by_label(label);
                                    for node in nodes {
                                        result_nodes.push(JsNodeResult {
                                            id: node.id.clone(),
                                            labels: node
                                                .labels
                                                .iter()
                                                .map(|l| l.name.clone())
                                                .collect(),
                                            properties: node
                                                .properties
                                                .iter()
                                                .map(|(k, v)| (k.clone(), format!("{:?}", v)))
                                                .collect(),
                                        });
                                    }
                                }
                                // If no labels specified, return all nodes (simplified)
                                if node_pattern.labels.is_empty() && node_pattern.variable.is_some()
                                {
                                    // This would need iteration over all nodes - for now just stats
                                }
                            }
                        }
                    }
                    Statement::Create(create_clause) => {
                        // Handle CREATE - but we need mutable access, so skip in query
                    }
                    Statement::Return(_) => {
                        // RETURN is handled implicitly
                    }
                    _ => {}
                }
            }

            let stats = hg.stats();

            Ok::<JsQueryResult, Error>(JsQueryResult {
                nodes: result_nodes,
                edges: result_edges,
                stats: Some(JsGraphStats {
                    total_nodes: stats.total_entities as u32,
                    total_edges: stats.total_hyperedges as u32,
                    avg_degree: stats.avg_entity_degree as f64,
                }),
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Query the graph synchronously
    ///
    /// # Example
    /// ```javascript
    /// const results = db.querySync('MATCH (n) RETURN n LIMIT 10');
    /// ```
    #[napi]
    pub fn query_sync(&self, cypher: String) -> Result<JsQueryResult> {
        let hg = self.hypergraph.read().expect("RwLock poisoned");
        let stats = hg.stats();

        // Simplified query result for now
        Ok(JsQueryResult {
            nodes: vec![],
            edges: vec![],
            stats: Some(JsGraphStats {
                total_nodes: stats.total_entities as u32,
                total_edges: stats.total_hyperedges as u32,
                avg_degree: stats.avg_entity_degree as f64,
            }),
        })
    }

    /// Search for similar hyperedges
    ///
    /// # Example
    /// ```javascript
    /// const results = await db.searchHyperedges({
    ///   embedding: new Float32Array([0.5, 0.5, 0.5]),
    ///   k: 10
    /// });
    /// ```
    #[napi]
    pub async fn search_hyperedges(
        &self,
        query: JsHyperedgeQuery,
    ) -> Result<Vec<JsHyperedgeResult>> {
        let hypergraph = self.hypergraph.clone();
        let embedding = query.embedding.to_vec();
        let k = query.k as usize;

        tokio::task::spawn_blocking(move || {
            let hg = hypergraph.read().expect("RwLock poisoned");
            let results = hg.search_hyperedges(&embedding, k);

            Ok::<Vec<JsHyperedgeResult>, Error>(
                results
                    .into_iter()
                    .map(|(id, score)| JsHyperedgeResult {
                        id,
                        score: f64::from(score),
                    })
                    .collect(),
            )
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Get k-hop neighbors from a starting node
    ///
    /// # Example
    /// ```javascript
    /// const neighbors = await db.kHopNeighbors('node1', 2);
    /// ```
    #[napi]
    pub async fn k_hop_neighbors(&self, start_node: String, k: u32) -> Result<Vec<String>> {
        let hypergraph = self.hypergraph.clone();
        let hops = k as usize;

        tokio::task::spawn_blocking(move || {
            let hg = hypergraph.read().expect("RwLock poisoned");
            let neighbors = hg.k_hop_neighbors(start_node, hops);
            Ok::<Vec<String>, Error>(neighbors.into_iter().collect())
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Begin a new transaction
    ///
    /// # Example
    /// ```javascript
    /// const txId = await db.begin();
    /// ```
    #[napi]
    pub async fn begin(&self) -> Result<String> {
        let tm = self.transaction_manager.clone();

        tokio::task::spawn_blocking(move || {
            let mut manager = tm.write().expect("RwLock poisoned");
            Ok::<String, Error>(manager.begin())
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Commit a transaction
    ///
    /// # Example
    /// ```javascript
    /// await db.commit(txId);
    /// ```
    #[napi]
    pub async fn commit(&self, tx_id: String) -> Result<()> {
        let tm = self.transaction_manager.clone();

        tokio::task::spawn_blocking(move || {
            let mut manager = tm.write().expect("RwLock poisoned");
            manager
                .commit(&tx_id)
                .map_err(|e| Error::from_reason(format!("Failed to commit: {}", e)))
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Rollback a transaction
    ///
    /// # Example
    /// ```javascript
    /// await db.rollback(txId);
    /// ```
    #[napi]
    pub async fn rollback(&self, tx_id: String) -> Result<()> {
        let tm = self.transaction_manager.clone();

        tokio::task::spawn_blocking(move || {
            let mut manager = tm.write().expect("RwLock poisoned");
            manager
                .rollback(&tx_id)
                .map_err(|e| Error::from_reason(format!("Failed to rollback: {}", e)))
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Batch insert nodes and edges
    ///
    /// # Example
    /// ```javascript
    /// await db.batchInsert({
    ///   nodes: [{ id: 'n1', embedding: new Float32Array([1, 2]) }],
    ///   edges: [{ from: 'n1', to: 'n2', description: 'knows' }]
    /// });
    /// ```
    #[napi]
    pub async fn batch_insert(&self, batch: JsBatchInsert) -> Result<JsBatchResult> {
        let hypergraph = self.hypergraph.clone();
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let nodes = batch.nodes;
        let edges = batch.edges;

        tokio::task::spawn_blocking(move || {
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            let mut gdb = graph_db.write().expect("RwLock poisoned");
            let mut node_ids = Vec::new();
            let mut edge_ids = Vec::new();

            // Insert nodes into BOTH the hypergraph index and the property
            // graph + label index (so bulk-loaded nodes are also visible to
            // the Cypher `MATCH (n:Label)` scan, not just kHop/stats).
            for node in nodes {
                let id = node.id.clone();
                register_node(
                    &mut hg,
                    &mut gdb,
                    storage.as_ref(),
                    id.clone(),
                    node.embedding.to_vec(),
                    node.labels,
                    node.properties,
                )?;
                node_ids.push(id);
            }

            // Insert edges (write-through: hypergraph + property graph + storage,
            // mirroring create_edge so bulk-loaded edges survive a restart).
            for edge in edges {
                let nodes = vec![edge.from.clone(), edge.to.clone()];
                let embedding = edge.embedding.to_vec();
                let confidence = edge.confidence.unwrap_or(1.0) as f32;
                let core_edge =
                    CoreHyperedge::new(nodes, edge.description.clone(), embedding.clone(), confidence);
                let edge_id = core_edge.id.clone();
                hg.add_hyperedge(core_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to insert edge: {}", e)))?;

                let mut props = std::collections::HashMap::new();
                props.insert(
                    "__confidence".to_string(),
                    PropertyValue::FloatArray(vec![confidence]),
                );
                props.insert("__embedding".to_string(), PropertyValue::FloatArray(embedding));
                let graph_edge = GraphEdge::new(
                    edge_id.clone(),
                    edge.from.clone(),
                    edge.to.clone(),
                    edge.description,
                    props,
                );
                if let Some(ref storage_arc) = storage {
                    let sg = storage_arc.write().expect("Storage RwLock poisoned");
                    sg.insert_edge(&graph_edge)
                        .map_err(|e| Error::from_reason(format!("Failed to persist edge: {}", e)))?;
                }
                gdb.create_edge(graph_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to insert edge: {}", e)))?;

                edge_ids.push(edge_id);
            }

            Ok::<JsBatchResult, Error>(JsBatchResult { node_ids, edge_ids })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Delete a node and optionally cascade to its incident hyperedges
    ///
    /// # Example
    /// ```javascript
    /// const result = await db.deleteNode('node1', { cascade: true });
    /// console.log(`Deleted: ${result.deletedNode}, edges removed: ${result.deletedEdges}`);
    /// ```
    #[napi]
    pub async fn delete_node(
        &self,
        id: String,
        opts: Option<JsDeleteNodeOptions>,
    ) -> Result<JsDeleteNodeResult> {
        let hypergraph = self.hypergraph.clone();
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let cascade = opts.and_then(|o| o.cascade).unwrap_or(false);

        tokio::task::spawn_blocking(move || {
            let deleted_edges = {
                let mut hg = hypergraph.write().expect("RwLock poisoned");
                hg.remove_entity(&id, cascade) as u32
            };

            let deleted_node = {
                let gdb = graph_db.read().expect("RwLock poisoned");
                gdb.delete_node(&id)
                    .map_err(|e| Error::from_reason(format!("Failed to delete node: {}", e)))?
            };

            if deleted_node {
                if let Some(ref storage_arc) = storage {
                    let sg = storage_arc.write().expect("Storage RwLock poisoned");
                    sg.delete_node(&id)
                        .map_err(|e| Error::from_reason(format!("Storage delete failed: {}", e)))?;
                    // Cascade through persistent storage so a later hydrate
                    // never replays edges/hyperedges pointing at this node.
                    if let Ok(edge_ids) = sg.all_edge_ids() {
                        for eid in edge_ids {
                            if let Ok(Some(e)) = sg.get_edge(&eid) {
                                if e.from == id || e.to == id {
                                    let _ = sg.delete_edge(&eid);
                                    let gdb = graph_db.read().expect("RwLock poisoned");
                                    let _ = gdb.delete_edge(&eid);
                                }
                            }
                        }
                    }
                    if let Ok(he_ids) = sg.all_hyperedge_ids() {
                        for hid in he_ids {
                            if let Ok(Some(h)) = sg.get_hyperedge(&hid) {
                                if h.nodes.iter().any(|n| n == &id) {
                                    let _ = sg.delete_hyperedge(&hid);
                                }
                            }
                        }
                    }
                }
            }

            Ok::<JsDeleteNodeResult, Error>(JsDeleteNodeResult {
                deleted_node,
                deleted_edges,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Delete an edge by ID
    ///
    /// # Example
    /// ```javascript
    /// const result = await db.deleteEdge('edge-id');
    /// console.log(`Deleted: ${result.deleted}`);
    /// ```
    #[napi]
    pub async fn delete_edge(&self, id: String) -> Result<JsDeleteResult> {
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let hypergraph = self.hypergraph.clone();

        tokio::task::spawn_blocking(move || {
            {
                // Edge ids match their hypergraph twins since the write-through
                // patch, so keep the traversal index consistent too.
                let mut hg = hypergraph.write().expect("RwLock poisoned");
                hg.remove_hyperedge(&id);
            }
            let deleted = {
                let gdb = graph_db.read().expect("RwLock poisoned");
                gdb.delete_edge(&id)
                    .map_err(|e| Error::from_reason(format!("Failed to delete edge: {}", e)))?
            };

            if deleted {
                if let Some(ref storage_arc) = storage {
                    let sg = storage_arc.write().expect("Storage RwLock poisoned");
                    sg.delete_edge(&id)
                        .map_err(|e| Error::from_reason(format!("Storage delete failed: {}", e)))?;
                }
            }

            Ok::<JsDeleteResult, Error>(JsDeleteResult { deleted })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Delete a hyperedge by ID
    ///
    /// # Example
    /// ```javascript
    /// const result = await db.deleteHyperedge('hyperedge-id');
    /// console.log(`Deleted: ${result.deleted}`);
    /// ```
    #[napi]
    pub async fn delete_hyperedge(&self, id: String) -> Result<JsDeleteResult> {
        let hypergraph = self.hypergraph.clone();
        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();

        tokio::task::spawn_blocking(move || {
            {
                let mut hg = hypergraph.write().expect("RwLock poisoned");
                hg.remove_hyperedge(&id);
            }

            let deleted = {
                let gdb = graph_db.read().expect("RwLock poisoned");
                gdb.delete_hyperedge(&id)
                    .map_err(|e| Error::from_reason(format!("Failed to delete hyperedge: {}", e)))?
            };

            if deleted {
                if let Some(ref storage_arc) = storage {
                    let sg = storage_arc.write().expect("Storage RwLock poisoned");
                    sg.delete_hyperedge(&id)
                        .map_err(|e| Error::from_reason(format!("Storage delete failed: {}", e)))?;
                }
            }

            Ok::<JsDeleteResult, Error>(JsDeleteResult { deleted })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Subscribe to graph changes (returns a change stream)
    ///
    /// # Example
    /// ```javascript
    /// const unsubscribe = db.subscribe((change) => {
    ///   console.log('Graph changed:', change);
    /// });
    /// ```
    #[napi]
    pub fn subscribe(&self, callback: JsFunction) -> Result<()> {
        // Placeholder for event emitter pattern
        // In a real implementation, this would set up a change listener
        Ok(())
    }

    /// Get graph statistics
    ///
    /// # Example
    /// ```javascript
    /// const stats = await db.stats();
    /// console.log(`Nodes: ${stats.totalNodes}, Edges: ${stats.totalEdges}`);
    /// ```
    #[napi]
    pub async fn stats(&self) -> Result<JsGraphStats> {
        let hypergraph = self.hypergraph.clone();

        tokio::task::spawn_blocking(move || {
            let hg = hypergraph.read().expect("RwLock poisoned");
            let stats = hg.stats();

            Ok::<JsGraphStats, Error>(JsGraphStats {
                total_nodes: stats.total_entities as u32,
                total_edges: stats.total_hyperedges as u32,
                avg_degree: stats.avg_entity_degree as f64,
            })
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }
}

/// Get the version of the library
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Test function to verify bindings
#[napi]
pub fn hello() -> String {
    "Hello from RuVector Graph Node.js bindings!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the `batchInsert` ↔ label-index consistency bug.
    ///
    /// Both `create_node` and `batch_insert` funnel through `register_node`.
    /// This test drives `register_node` directly (the same code path the batch
    /// insert uses) and asserts that the resulting node is consistently visible
    /// through all three read surfaces:
    ///   1. `get_nodes_by_label` — the Cypher `MATCH (n:Label)` label scan,
    ///   2. `k_hop_neighbors`    — undirected adjacency traversal,
    ///   3. `stats`              — entity counts.
    ///
    /// Before the fix the batch path skipped (1), so this would report 0 nodes
    /// for the label scan while (2) and (3) saw the nodes.
    #[test]
    fn batch_inserted_nodes_are_visible_to_label_scan_khop_and_stats() {
        let mut hg = CoreHypergraphIndex::new(DistanceMetric::Cosine);
        let mut gdb = GraphDB::new();

        let n = 5usize;
        // Insert N labeled nodes via the shared registration path.
        for i in 0..n {
            register_node(
                &mut hg,
                &mut gdb,
                None,
                format!("p{i}"),
                vec![i as f32, i as f32, i as f32, i as f32],
                Some(vec!["Person".to_string()]),
                Some(HashMap::from([("name".to_string(), format!("name{i}"))])),
            )
            .expect("register_node should succeed");
        }

        // Chain edges p0->p1->...->p{n-1} so kHop has something to traverse.
        for i in 0..(n - 1) {
            let edge = CoreHyperedge::new(
                vec![format!("p{i}"), format!("p{}", i + 1)],
                "knows".to_string(),
                vec![0.0, 0.0, 0.0, 1.0],
                1.0,
            );
            hg.add_hyperedge(edge)
                .expect("add_hyperedge should succeed");
        }

        // 1. Label scan (the bug): every batch-inserted node must appear.
        let by_label = gdb.get_nodes_by_label("Person");
        assert_eq!(
            by_label.len(),
            n,
            "label-scoped MATCH must see all {n} batch-inserted nodes (regression: was 0)"
        );

        // 2. kHop adjacency: p0's 2-hop ball includes itself, p1 and p2.
        let neighbors = hg.k_hop_neighbors("p0".to_string(), 2);
        assert!(
            neighbors.contains("p0"),
            "kHop ball includes the start node"
        );
        assert!(
            neighbors.contains("p1"),
            "kHop ball includes 1-hop neighbor"
        );
        assert!(
            neighbors.contains("p2"),
            "kHop ball includes 2-hop neighbor"
        );

        // 3. stats entity count stays consistent with the above.
        let stats = hg.stats();
        assert_eq!(
            stats.total_entities, n,
            "stats() entity count must match the inserted node count"
        );
    }
}
