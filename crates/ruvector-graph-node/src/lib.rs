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
use ruvector_graph::node::NodeBuilder;
use ruvector_graph::storage::GraphStorage;
use ruvector_graph::types::PropertyValue;
use ruvector_graph::GraphDB;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

mod cypher_exec;
mod streaming;
mod transactions;
mod types;

/// Extract an f32 vector from current and legacy persisted property encodings.
fn prop_to_f32_vec(property: Option<&PropertyValue>) -> Vec<f32> {
    match property {
        Some(PropertyValue::FloatArray(values)) => values.clone(),
        Some(PropertyValue::Array(items)) | Some(PropertyValue::List(items)) => items
            .iter()
            .filter_map(|item| match item {
                PropertyValue::Float(value) => Some(*value as f32),
                PropertyValue::Integer(value) => Some(*value as f32),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

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
    /// Whether the durable store has been replayed into the in-memory indexes.
    /// Set on first use rather than in the constructor — see [`hydrate_once`].
    hydrated: Arc<Mutex<bool>>,
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

/// Replay persisted records into the in-memory property and hypergraph indexes,
/// exactly once per database handle.
///
/// This used to run synchronously inside the NAPI constructor, which blocked the
/// Node event loop for the whole replay — about 15 seconds on the 154k-node graph
/// in ruvnet/ruvector#826. Hydration is now deferred to the first operation that
/// actually needs the in-memory indexes; every async entry point calls this from
/// inside `spawn_blocking`, so the cost lands on the thread pool instead of the
/// main thread. Construction is O(1) again.
///
/// The `hydrated` flag is held across the replay, so concurrent first-callers
/// serialise here rather than each replaying the store into the same indexes.
fn hydrate_once(
    hydrated: &Arc<Mutex<bool>>,
    storage: Option<&Arc<RwLock<GraphStorage>>>,
    hypergraph: &Arc<RwLock<CoreHypergraphIndex>>,
    graph_db: &Arc<RwLock<GraphDB>>,
) -> Result<()> {
    let mut done = hydrated.lock().expect("hydration Mutex poisoned");
    if *done {
        return Ok(());
    }
    hydrate_from_storage(storage, hypergraph, graph_db)?;
    *done = true;
    Ok(())
}

fn hydrate_from_storage(
    storage: Option<&Arc<RwLock<GraphStorage>>>,
    hypergraph: &Arc<RwLock<CoreHypergraphIndex>>,
    graph_db: &Arc<RwLock<GraphDB>>,
) -> Result<()> {
    let Some(storage_arc) = storage else {
        return Ok(());
    };
    let storage = storage_arc.read().expect("Storage RwLock poisoned");
    let mut hg = hypergraph.write().expect("RwLock poisoned");
    let gdb = graph_db.write().expect("RwLock poisoned");

    for id in storage
        .all_node_ids()
        .map_err(|e| Error::from_reason(format!("hydrate nodes: {e}")))?
    {
        if let Some(node) = storage
            .get_node(&id)
            .map_err(|e| Error::from_reason(format!("hydrate node {id}: {e}")))?
        {
            let embedding = prop_to_f32_vec(node.properties.get("__embedding"));
            hg.add_entity(node.id.clone(), embedding);
            gdb.create_node(node)
                .map_err(|e| Error::from_reason(format!("hydrate node insert: {e}")))?;
        }
    }

    for id in storage
        .all_edge_ids()
        .map_err(|e| Error::from_reason(format!("hydrate edges: {e}")))?
    {
        if let Some(edge) = storage
            .get_edge(&id)
            .map_err(|e| Error::from_reason(format!("hydrate edge {id}: {e}")))?
        {
            let confidence = prop_to_f32_vec(edge.properties.get("__confidence"))
                .first()
                .copied()
                .unwrap_or(1.0);
            let embedding = prop_to_f32_vec(edge.properties.get("__embedding"));
            let mut core_edge = CoreHyperedge::new(
                vec![edge.from.clone(), edge.to.clone()],
                edge.edge_type.clone(),
                embedding,
                confidence,
            );
            core_edge.id = edge.id.clone();
            // A non-cascaded deletion deliberately leaves the durable edge,
            // but neither in-memory index accepts an edge with a missing node.
            if hg.add_hyperedge(core_edge).is_err() {
                continue;
            }
            gdb.create_edge(edge)
                .map_err(|e| Error::from_reason(format!("hydrate edge insert: {e}")))?;
        }
    }

    for id in storage
        .all_hyperedge_ids()
        .map_err(|e| Error::from_reason(format!("hydrate hyperedges: {e}")))?
    {
        if let Some(hyperedge) = storage
            .get_hyperedge(&id)
            .map_err(|e| Error::from_reason(format!("hydrate hyperedge {id}: {e}")))?
        {
            let embedding = prop_to_f32_vec(hyperedge.properties.get("__embedding"));
            let mut core_edge = CoreHyperedge::new(
                hyperedge.nodes,
                hyperedge
                    .description
                    .unwrap_or_else(|| hyperedge.edge_type.clone()),
                embedding,
                hyperedge.confidence,
            );
            core_edge.id = hyperedge.id;
            // A non-cascaded deletion can deliberately leave this dangling.
            let _ = hg.add_hyperedge(core_edge);
        }
    }

    Ok(())
}

/// Parse and execute a Cypher statement against the in-memory graph.
///
/// Shared by `query()` and `querySync()` so the two cannot drift — before
/// ruvnet/ruvector#879 they had entirely separate bodies, and `querySync()`
/// never even parsed its argument.
fn run_query(
    cypher: &str,
    graph_db: &Arc<RwLock<GraphDB>>,
    hypergraph: &Arc<RwLock<CoreHypergraphIndex>>,
) -> Result<JsQueryResult> {
    let parsed = parse_cypher(cypher)
        .map_err(|e| Error::from_reason(format!("Cypher parse error: {}", e)))?;

    let gdb = graph_db.read().expect("RwLock poisoned");
    let hg = hypergraph.read().expect("RwLock poisoned");

    let mut result_nodes: Vec<JsNodeResult> = Vec::new();
    let mut result_edges: Vec<JsEdgeResult> = Vec::new();
    let mut unsupported: Vec<String> = Vec::new();

    for statement in &parsed.statements {
        match statement {
            Statement::Match(match_clause) => {
                let outcome = cypher_exec::execute_match(&gdb, match_clause);
                unsupported.extend(outcome.unsupported);
                for node in outcome.nodes {
                    result_nodes.push(JsNodeResult {
                        id: node.id.clone(),
                        labels: node.labels.iter().map(|l| l.name.clone()).collect(),
                        properties: node
                            .properties
                            .iter()
                            .filter(|(k, _)| !is_internal_property(k))
                            .map(|(k, v)| (k.clone(), property_to_string(v)))
                            .collect(),
                    });
                }
                for edge in outcome.edges {
                    result_edges.push(JsEdgeResult {
                        id: edge.id.clone(),
                        from: edge.from.clone(),
                        to: edge.to.clone(),
                        edge_type: edge.edge_type.clone(),
                        properties: edge
                            .properties
                            .iter()
                            .filter(|(k, _)| !is_internal_property(k))
                            .map(|(k, v)| (k.clone(), property_to_string(v)))
                            .collect(),
                    });
                }
            }
            // Writes through `query()` were previously accepted and silently
            // discarded. Refuse them instead — the typed builders are the
            // supported path and a dropped write is worse than a loud error.
            Statement::Create(_) => unsupported.push(
                "CREATE via query() is not supported; use createNode()/createEdge()".to_string(),
            ),
            Statement::Return(_) => {}
            _ => {}
        }
    }

    if !unsupported.is_empty() {
        unsupported.sort();
        unsupported.dedup();
        return Err(Error::from_reason(format!(
            "Unsupported Cypher: {}",
            unsupported.join("; ")
        )));
    }

    let stats = hg.stats();
    Ok(JsQueryResult {
        nodes: result_nodes,
        edges: result_edges,
        stats: Some(JsGraphStats {
            total_nodes: stats.total_entities as u32,
            total_edges: stats.total_hyperedges as u32,
            avg_degree: stats.avg_entity_degree as f64,
        }),
    })
}

/// Properties this binding stores for its own use rather than the caller's.
///
/// Embeddings and edge confidences are persisted as ordinary properties under a
/// `__` prefix. Returning them would put a 384-element float dump in every
/// result row's property map — invisible until #879 made `query()` return rows
/// at all.
fn is_internal_property(key: &str) -> bool {
    key.starts_with("__")
}

/// Render a stored property for the string-valued JS result maps.
///
/// `format!("{:?}", v)` was used here, which surfaced `String("alice")` to
/// JavaScript instead of `alice`. Scalars now render as their plain value.
fn property_to_string(value: &PropertyValue) -> String {
    match value {
        PropertyValue::String(s) => s.clone(),
        PropertyValue::Integer(i) => i.to_string(),
        PropertyValue::Float(f) => f.to_string(),
        PropertyValue::Boolean(b) => b.to_string(),
        PropertyValue::Null => String::new(),
        other => format!("{:?}", other),
    }
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
            hydrated: Arc::new(Mutex::new(false)),
            storage,
            storage_path,
        };
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
            hydrated: Arc::new(Mutex::new(false)),
            storage: Some(Arc::new(RwLock::new(storage))),
            storage_path: Some(path),
        };
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            let core_edge =
                CoreHyperedge::new(nodes, description.clone(), embedding.clone(), confidence);
            let edge_id = core_edge.id.clone();
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            hg.add_hyperedge(core_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create edge: {}", e)))?;
            drop(hg);

            let properties = HashMap::from([
                (
                    "__confidence".to_string(),
                    PropertyValue::FloatArray(vec![confidence]),
                ),
                (
                    "__embedding".to_string(),
                    PropertyValue::FloatArray(embedding),
                ),
            ]);
            let graph_edge = GraphEdge::new(edge_id.clone(), from, to, description, properties);
            if let Some(storage_arc) = storage {
                storage_arc
                    .write()
                    .expect("Storage RwLock poisoned")
                    .insert_edge(&graph_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to persist edge: {e}")))?;
            }
            graph_db
                .read()
                .expect("RwLock poisoned")
                .create_edge(graph_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create edge: {e}")))?;
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

        let graph_db = self.graph_db.clone();
        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            let core_edge = CoreHyperedge::new(
                nodes.clone(),
                description.clone(),
                embedding.clone(),
                confidence,
            );
            let edge_id = core_edge.id.clone();
            let mut hg = hypergraph.write().expect("RwLock poisoned");
            hg.add_hyperedge(core_edge)
                .map_err(|e| Error::from_reason(format!("Failed to create hyperedge: {}", e)))?;
            drop(hg);

            if let Some(storage_arc) = storage {
                let graph_hyperedge = GraphHyperedge {
                    id: edge_id.clone(),
                    nodes,
                    edge_type: "HYPEREDGE".to_string(),
                    description: Some(description),
                    properties: HashMap::from([(
                        "__embedding".to_string(),
                        PropertyValue::FloatArray(embedding),
                    )]),
                    confidence,
                };
                storage_arc
                    .write()
                    .expect("Storage RwLock poisoned")
                    .insert_hyperedge(&graph_hyperedge)
                    .map_err(|e| Error::from_reason(format!("Failed to persist hyperedge: {e}")))?;
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
        let storage = self.storage.clone();
        let hydrated = self.hydrated.clone();

        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            run_query(&cypher, &graph_db, &hypergraph)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task failed: {}", e)))?
    }

    /// Query the graph synchronously
    ///
    /// Identical semantics to [`GraphDatabase::query`], but it runs on the
    /// calling thread. On a persisted database the first call also pays for
    /// hydration, so prefer the async `query()` on large graphs.
    ///
    /// # Example
    /// ```javascript
    /// const results = db.querySync('MATCH (n) RETURN n LIMIT 10');
    /// ```
    #[napi]
    pub fn query_sync(&self, cypher: String) -> Result<JsQueryResult> {
        hydrate_once(
            &self.hydrated,
            self.storage.as_ref(),
            &self.hypergraph,
            &self.graph_db,
        )?;
        run_query(&cypher, &self.graph_db, &self.hypergraph)
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

        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
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

        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
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

            // Insert edges into all three representations.
            for edge in edges {
                let nodes = vec![edge.from.clone(), edge.to.clone()];
                let embedding = edge.embedding.to_vec();
                let confidence = edge.confidence.unwrap_or(1.0) as f32;
                let core_edge = CoreHyperedge::new(
                    nodes,
                    edge.description.clone(),
                    embedding.clone(),
                    confidence,
                );
                let edge_id = core_edge.id.clone();
                hg.add_hyperedge(core_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to insert edge: {}", e)))?;
                let graph_edge = GraphEdge::new(
                    edge_id.clone(),
                    edge.from,
                    edge.to,
                    edge.description,
                    HashMap::from([
                        (
                            "__confidence".to_string(),
                            PropertyValue::FloatArray(vec![confidence]),
                        ),
                        (
                            "__embedding".to_string(),
                            PropertyValue::FloatArray(embedding),
                        ),
                    ]),
                );
                if let Some(storage_arc) = storage.as_ref() {
                    storage_arc
                        .write()
                        .expect("Storage RwLock poisoned")
                        .insert_edge(&graph_edge)
                        .map_err(|e| Error::from_reason(format!("Failed to persist edge: {e}")))?;
                }
                gdb.create_edge(graph_edge)
                    .map_err(|e| Error::from_reason(format!("Failed to insert edge: {e}")))?;
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            let graph_edge_ids = if cascade {
                let gdb = graph_db.read().expect("RwLock poisoned");
                let mut ids: Vec<String> = gdb
                    .get_outgoing_edges(&id)
                    .into_iter()
                    .chain(gdb.get_incoming_edges(&id))
                    .map(|edge| edge.id)
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                ids
            } else {
                Vec::new()
            };
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
                if cascade {
                    let gdb = graph_db.read().expect("RwLock poisoned");
                    for edge_id in &graph_edge_ids {
                        gdb.delete_edge(edge_id).map_err(|e| {
                            Error::from_reason(format!("Cascade graph edge delete failed: {e}"))
                        })?;
                    }
                }
                if let Some(ref storage_arc) = storage {
                    let sg = storage_arc.write().expect("Storage RwLock poisoned");
                    sg.delete_node(&id)
                        .map_err(|e| Error::from_reason(format!("Storage delete failed: {}", e)))?;

                    // Preserve incident records for `cascade: false`; deleting
                    // them would silently change semantics after reopening.
                    if cascade {
                        for edge_id in sg
                            .all_edge_ids()
                            .map_err(|e| Error::from_reason(format!("List edges failed: {e}")))?
                        {
                            let is_incident = sg
                                .get_edge(&edge_id)
                                .map_err(|e| Error::from_reason(format!("Read edge failed: {e}")))?
                                .is_some_and(|edge| edge.from == id || edge.to == id);
                            if is_incident {
                                sg.delete_edge(&edge_id).map_err(|e| {
                                    Error::from_reason(format!("Cascade edge delete failed: {e}"))
                                })?;
                            }
                        }
                        for hyperedge_id in sg.all_hyperedge_ids().map_err(|e| {
                            Error::from_reason(format!("List hyperedges failed: {e}"))
                        })? {
                            let is_incident = sg
                                .get_hyperedge(&hyperedge_id)
                                .map_err(|e| {
                                    Error::from_reason(format!("Read hyperedge failed: {e}"))
                                })?
                                .is_some_and(|hyperedge| {
                                    hyperedge.nodes.iter().any(|node| node == &id)
                                });
                            if is_incident {
                                sg.delete_hyperedge(&hyperedge_id).map_err(|e| {
                                    Error::from_reason(format!(
                                        "Cascade hyperedge delete failed: {e}"
                                    ))
                                })?;
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            let deleted_from_hypergraph = hypergraph
                .write()
                .expect("RwLock poisoned")
                .remove_hyperedge(&id);
            let deleted_from_graph = {
                let gdb = graph_db.read().expect("RwLock poisoned");
                gdb.delete_edge(&id)
                    .map_err(|e| Error::from_reason(format!("Failed to delete edge: {}", e)))?
            };

            let deleted_from_storage = if let Some(ref storage_arc) = storage {
                storage_arc
                    .write()
                    .expect("Storage RwLock poisoned")
                    .delete_edge(&id)
                    .map_err(|e| Error::from_reason(format!("Storage delete failed: {e}")))?
            } else {
                false
            };

            Ok::<JsDeleteResult, Error>(JsDeleteResult {
                deleted: deleted_from_hypergraph || deleted_from_graph || deleted_from_storage,
            })
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

        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
            let deleted_from_hypergraph = hypergraph
                .write()
                .expect("RwLock poisoned")
                .remove_hyperedge(&id);

            let deleted_from_graph = {
                let gdb = graph_db.read().expect("RwLock poisoned");
                gdb.delete_hyperedge(&id)
                    .map_err(|e| Error::from_reason(format!("Failed to delete hyperedge: {}", e)))?
            };

            let deleted_from_storage = if let Some(ref storage_arc) = storage {
                storage_arc
                    .write()
                    .expect("Storage RwLock poisoned")
                    .delete_hyperedge(&id)
                    .map_err(|e| Error::from_reason(format!("Storage delete failed: {e}")))?
            } else {
                false
            };

            Ok::<JsDeleteResult, Error>(JsDeleteResult {
                deleted: deleted_from_hypergraph || deleted_from_graph || deleted_from_storage,
            })
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

        let graph_db = self.graph_db.clone();
        let storage = self.storage.clone();
        let hydrated = self.hydrated.clone();
        tokio::task::spawn_blocking(move || {
            hydrate_once(&hydrated, storage.as_ref(), &hypergraph, &graph_db)?;
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

    fn temp_storage_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "ruvector-graph-node-{name}-{}.redb",
                uuid::Uuid::new_v4()
            ))
            .to_string_lossy()
            .into_owned()
    }

    async fn create_persistent_fixture(path: &str) -> (GraphDatabase, String, String) {
        let db = GraphDatabase::new(Some(JsGraphOptions {
            distance_metric: Some(JsDistanceMetric::Cosine),
            dimensions: Some(2),
            storage_path: Some(path.to_string()),
        }))
        .expect("create persistent database");
        for id in ["a", "b"] {
            db.create_node(JsNode {
                id: id.to_string(),
                embedding: Float32Array::new(vec![1.0, 0.0]),
                labels: Some(vec!["Test".to_string()]),
                properties: None,
            })
            .await
            .expect("persist node");
        }
        let edge_id = db
            .create_edge(JsEdge {
                from: "a".to_string(),
                to: "b".to_string(),
                description: "CONNECTED".to_string(),
                embedding: Float32Array::new(vec![0.5, 0.5]),
                confidence: Some(0.8),
                metadata: None,
            })
            .await
            .expect("persist edge");
        let hyperedge_id = db
            .create_hyperedge(JsHyperedge {
                nodes: vec!["a".to_string(), "b".to_string()],
                description: "GROUP".to_string(),
                embedding: Float32Array::new(vec![0.25, 0.75]),
                confidence: Some(0.9),
                metadata: None,
            })
            .await
            .expect("persist hyperedge");
        (db, edge_id, hyperedge_id)
    }

    /// Hydration is deferred to first use (ruvnet/ruvector#826), so the thing
    /// that can now go wrong is replaying the store more than once and double
    /// counting every record. Drive several entry points against one freshly
    /// opened handle and assert the counts never move.
    #[tokio::test]
    async fn deferred_hydration_replays_exactly_once() {
        let path = temp_storage_path("hydrate-once");
        let (db, _, _) = create_persistent_fixture(&path).await;
        drop(db);

        let reopened = GraphDatabase::open(path.clone()).expect("reopen persisted database");
        // First touch triggers the replay; every later one must be a no-op.
        assert_eq!(reopened.stats().await.expect("stats").total_nodes, 2);
        assert_eq!(reopened.stats().await.expect("stats").total_nodes, 2);
        let rows = reopened
            .query("MATCH (n) RETURN n".to_string())
            .await
            .expect("query");
        assert_eq!(rows.nodes.len(), 2, "one row per persisted node, not two");
        assert_eq!(
            reopened
                .k_hop_neighbors("a".to_string(), 1)
                .await
                .expect("khop")
                .len(),
            2
        );
        assert_eq!(reopened.stats().await.expect("stats").total_nodes, 2);
        drop(reopened);
        std::fs::remove_file(path).expect("remove test database");
    }

    /// The label-less `MATCH (n)` and `WHERE` shapes from #879, end to end
    /// through the NAPI surface rather than just the executor.
    #[tokio::test]
    async fn query_returns_rows_for_label_less_match_and_where() {
        let db = GraphDatabase::new(Some(JsGraphOptions {
            distance_metric: Some(JsDistanceMetric::Cosine),
            dimensions: Some(2),
            storage_path: None,
        }))
        .expect("create database");
        for id in ["n1", "n2"] {
            db.create_node(JsNode {
                id: id.to_string(),
                embedding: Float32Array::new(vec![1.0, 0.0]),
                labels: Some(vec!["Person".to_string()]),
                properties: None,
            })
            .await
            .expect("create node");
        }

        let all = db
            .query("MATCH (n) RETURN n".to_string())
            .await
            .expect("label-less match");
        assert_eq!(all.nodes.len(), 2);

        let one = db
            .query("MATCH (n) WHERE n.id = 'n1' RETURN n".to_string())
            .await
            .expect("point lookup");
        assert_eq!(one.nodes.len(), 1);
        assert_eq!(one.nodes[0].id, "n1");

        // querySync used to ignore its argument entirely and always return [].
        let sync = db
            .query_sync("MATCH (n) WHERE n.id = 'n2' RETURN n".to_string())
            .expect("sync point lookup");
        assert_eq!(sync.nodes.len(), 1);
        assert_eq!(sync.nodes[0].id, "n2");

        // A write through query() is refused rather than silently dropped.
        assert!(db
            .query("CREATE (n:Person {name: 'x'})".to_string())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn persisted_graph_hydrates_after_reopen() {
        let path = temp_storage_path("reopen");
        let (db, _, _) = create_persistent_fixture(&path).await;
        assert_eq!(db.stats().await.expect("stats").total_nodes, 2);
        assert_eq!(db.stats().await.expect("stats").total_edges, 2);
        drop(db);

        let reopened = GraphDatabase::open(path.clone()).expect("reopen persisted database");
        let stats = reopened.stats().await.expect("reopened stats");
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 2);
        assert_eq!(
            reopened
                .graph_db
                .read()
                .expect("graph lock")
                .get_nodes_by_label("Test")
                .len(),
            2
        );
        drop(reopened);
        std::fs::remove_file(path).expect("remove test database");
    }

    #[tokio::test]
    async fn non_cascade_retains_durable_relationships_but_cascade_removes_them() {
        let non_cascade_path = temp_storage_path("non-cascade");
        let (db, edge_id, hyperedge_id) = create_persistent_fixture(&non_cascade_path).await;
        let result = db
            .delete_node(
                "a".to_string(),
                Some(JsDeleteNodeOptions {
                    cascade: Some(false),
                }),
            )
            .await
            .expect("non-cascade delete");
        assert!(result.deleted_node);
        assert_eq!(result.deleted_edges, 0);
        {
            let storage = db
                .storage
                .as_ref()
                .expect("persistent storage")
                .read()
                .expect("storage lock");
            assert!(storage.get_edge(&edge_id).expect("read edge").is_some());
            assert!(storage
                .get_hyperedge(&hyperedge_id)
                .expect("read hyperedge")
                .is_some());
        }
        drop(db);
        // Dangling durable records must not make reopening fail.
        let reopened =
            GraphDatabase::open(non_cascade_path.clone()).expect("reopen non-cascaded graph");
        assert!(
            reopened
                .delete_edge(edge_id)
                .await
                .expect("delete retained edge")
                .deleted
        );
        assert!(
            reopened
                .delete_hyperedge(hyperedge_id)
                .await
                .expect("delete retained hyperedge")
                .deleted
        );
        drop(reopened);
        std::fs::remove_file(non_cascade_path).expect("remove non-cascade database");

        let cascade_path = temp_storage_path("cascade");
        let (db, edge_id, hyperedge_id) = create_persistent_fixture(&cascade_path).await;
        let result = db
            .delete_node(
                "a".to_string(),
                Some(JsDeleteNodeOptions {
                    cascade: Some(true),
                }),
            )
            .await
            .expect("cascade delete");
        assert!(result.deleted_node);
        assert_eq!(result.deleted_edges, 2);
        {
            let storage = db
                .storage
                .as_ref()
                .expect("persistent storage")
                .read()
                .expect("storage lock");
            assert!(storage.get_edge(&edge_id).expect("read edge").is_none());
            assert!(storage
                .get_hyperedge(&hyperedge_id)
                .expect("read hyperedge")
                .is_none());
        }
        drop(db);
        let reopened = GraphDatabase::open(cascade_path.clone()).expect("reopen cascaded graph");
        let stats = reopened.stats().await.expect("reopened stats");
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_edges, 0);
        drop(reopened);
        std::fs::remove_file(cascade_path).expect("remove cascade database");
    }

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
