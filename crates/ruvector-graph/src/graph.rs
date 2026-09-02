//! Graph database implementation with concurrent access and indexing

use crate::edge::Edge;
use crate::error::Result;
use crate::hyperedge::{Hyperedge, HyperedgeId};
use crate::index::{AdjacencyIndex, EdgeTypeIndex, HyperedgeNodeIndex, LabelIndex, PropertyIndex};
use crate::node::Node;
#[cfg(feature = "storage")]
use crate::storage::GraphStorage;
use crate::types::{EdgeId, NodeId, PropertyValue};
use dashmap::DashMap;
#[cfg(feature = "storage")]
use std::path::Path;
use std::sync::Arc;

/// High-performance graph database with concurrent access
pub struct GraphDB {
    /// In-memory node storage (DashMap for lock-free concurrent reads)
    nodes: Arc<DashMap<NodeId, Node>>,
    /// In-memory edge storage
    edges: Arc<DashMap<EdgeId, Edge>>,
    /// In-memory hyperedge storage
    hyperedges: Arc<DashMap<HyperedgeId, Hyperedge>>,
    /// Label index for fast label-based lookups
    label_index: LabelIndex,
    /// Property index for fast property-based lookups
    property_index: PropertyIndex,
    /// Edge type index
    edge_type_index: EdgeTypeIndex,
    /// Adjacency index for neighbor lookups
    adjacency_index: AdjacencyIndex,
    /// Hyperedge node index
    hyperedge_node_index: HyperedgeNodeIndex,
    /// Optional persistent storage
    #[cfg(feature = "storage")]
    storage: Option<GraphStorage>,
}

impl GraphDB {
    /// Create a new in-memory graph database
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
            edges: Arc::new(DashMap::new()),
            hyperedges: Arc::new(DashMap::new()),
            label_index: LabelIndex::new(),
            property_index: PropertyIndex::new(),
            edge_type_index: EdgeTypeIndex::new(),
            adjacency_index: AdjacencyIndex::new(),
            hyperedge_node_index: HyperedgeNodeIndex::new(),
            #[cfg(feature = "storage")]
            storage: None,
        }
    }

    /// Create a new graph database with persistent storage
    #[cfg(feature = "storage")]
    pub fn with_storage<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let storage = GraphStorage::new(path)?;

        let mut db = Self::new();
        db.storage = Some(storage);

        // Load existing data from storage
        db.load_from_storage()?;

        Ok(db)
    }

    /// Load all data from storage into memory
    #[cfg(feature = "storage")]
    fn load_from_storage(&mut self) -> anyhow::Result<()> {
        if let Some(storage) = &self.storage {
            // Load nodes
            for node_id in storage.all_node_ids()? {
                if let Some(node) = storage.get_node(&node_id)? {
                    self.nodes.insert(node_id.clone(), node.clone());
                    self.label_index.add_node(&node);
                    self.property_index.add_node(&node);
                }
            }

            // Load edges
            for edge_id in storage.all_edge_ids()? {
                if let Some(edge) = storage.get_edge(&edge_id)? {
                    self.edges.insert(edge_id.clone(), edge.clone());
                    self.edge_type_index.add_edge(&edge);
                    self.adjacency_index.add_edge(&edge);
                }
            }

            // Load hyperedges
            for hyperedge_id in storage.all_hyperedge_ids()? {
                if let Some(hyperedge) = storage.get_hyperedge(&hyperedge_id)? {
                    self.hyperedges
                        .insert(hyperedge_id.clone(), hyperedge.clone());
                    self.hyperedge_node_index.add_hyperedge(&hyperedge);
                }
            }
        }
        Ok(())
    }

    // Node operations

    /// Create a node
    pub fn create_node(&self, node: Node) -> Result<NodeId> {
        let id = node.id.clone();

        // Update indexes
        self.label_index.add_node(&node);
        self.property_index.add_node(&node);

        // Insert into memory
        self.nodes.insert(id.clone(), node.clone());

        // Persist to storage if available
        #[cfg(feature = "storage")]
        if let Some(storage) = &self.storage {
            storage.insert_node(&node)?;
        }

        Ok(id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: impl AsRef<str>) -> Option<Node> {
        self.nodes.get(id.as_ref()).map(|entry| entry.clone())
    }

    /// Borrow a node and apply `f` without cloning it.
    ///
    /// Hot-path accessor for scans that only need to read a node (e.g. vector
    /// scoring). Avoids the full `Node` + embedding clone that `get_node`
    /// incurs. Returns `None` if the node is absent.
    pub fn with_node<R>(&self, id: &str, f: impl FnOnce(&Node) -> R) -> Option<R> {
        self.nodes.get(id).map(|entry| f(entry.value()))
    }

    /// Node ids carrying `label`, straight from the label index (no node clones).
    pub fn node_ids_by_label(&self, label: &str) -> Vec<NodeId> {
        self.label_index.get_nodes_by_label(label)
    }

    /// Delete a node
    pub fn delete_node(&self, id: impl AsRef<str>) -> Result<bool> {
        if let Some((_, node)) = self.nodes.remove(id.as_ref()) {
            // Update indexes
            self.label_index.remove_node(&node);
            self.property_index.remove_node(&node);

            // Delete from storage if available
            #[cfg(feature = "storage")]
            if let Some(storage) = &self.storage {
                storage.delete_node(id.as_ref())?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Atomically update an existing node.
    ///
    /// Applies `f` to a clone of the current node, persists the new value, and
    /// refreshes the label and property indexes. Concurrent updates to the same
    /// node are serialized, so one update cannot silently overwrite another.
    /// The node ID is immutable and changing it returns a constraint error.
    ///
    /// Returns `Ok(false)` if the node was not found (no error), `Ok(true)` if
    /// updated successfully. This is the counterpart to `create_node` and
    /// enables SUPERSEDES-style versioning where a prior node is marked
    /// `deprecated` without deleting it.
    ///
    /// # Example
    ///
    /// ```ignore
    /// graph.update_node(&node_id, |n| {
    ///     n.set_property("status", PropertyValue::from("deprecated"));
    ///     n.set_property("deprecated_at", PropertyValue::from(now_iso8601));
    /// })?;
    /// ```
    ///
    /// The callback runs while the node's map shard is write-locked. It must
    /// not call back into this `GraphDB`, because doing so may deadlock.
    pub fn update_node<F>(&self, id: impl AsRef<str>, f: F) -> Result<bool>
    where
        F: FnOnce(&mut Node),
    {
        let id_ref = id.as_ref();
        let Some(mut entry) = self.nodes.get_mut(id_ref) else {
            return Ok(false);
        };
        let old_node = entry.value().clone();
        let mut new_node = old_node.clone();
        f(&mut new_node);

        if new_node.id != old_node.id {
            return Err(crate::error::GraphError::ConstraintViolation(
                "A node's ID cannot be changed by update_node".to_string(),
            ));
        }

        // Persist before changing memory so a storage error leaves the live
        // node and its indexes untouched.
        #[cfg(feature = "storage")]
        if let Some(storage) = &self.storage {
            storage.insert_node(&new_node)?;
        }

        self.label_index.remove_node(&old_node);
        self.property_index.remove_node(&old_node);
        *entry.value_mut() = new_node.clone();
        self.label_index.add_node(&new_node);
        self.property_index.add_node(&new_node);

        Ok(true)
    }

    /// Keyword (BM25) search over a node text property.
    ///
    /// Builds a transient `Bm25Index` from the `text_field` property of all
    /// nodes carrying `label`, and returns the top-`k` node IDs by BM25 score.
    /// This is the keyword arm of hybrid search — pair with vector ANN for
    /// reciprocal rank fusion.
    ///
    /// For large graphs, build the index once and reuse it; this method
    /// rebuilds on every call (suitable for small-to-medium graphs or
    /// one-shot queries). A cached variant can be added behind a feature
    /// flag if needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hits = graph.keyword_search("Memory", "content", "vector search", 10)?;
    /// ```
    pub fn keyword_search(
        &self,
        label: &str,
        text_field: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let docs: Vec<(NodeId, String)> = self
            .node_ids_by_label(label)
            .into_iter()
            .filter_map(|id| {
                self.with_node(&id, |node| {
                    node.get_property(text_field).and_then(|value| match value {
                        PropertyValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                })
                .flatten()
                .map(|text| (id, text))
            })
            .collect();

        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let index = crate::bm25::Bm25Index::build(docs, crate::bm25::Bm25Params::default());
        Ok(index.search(query, k))
    }

    /// Get nodes by label
    pub fn get_nodes_by_label(&self, label: &str) -> Vec<Node> {
        self.label_index
            .get_nodes_by_label(label)
            .into_iter()
            .filter_map(|id| self.get_node(&id))
            .collect()
    }

    /// Get every node in the graph.
    ///
    /// This is a full scan of the node map — it backs the label-less Cypher
    /// pattern `MATCH (n)`, which has no index to consult by definition.
    pub fn all_nodes(&self) -> Vec<Node> {
        self.nodes.iter().map(|e| e.value().clone()).collect()
    }

    /// Get every edge in the graph.
    ///
    /// Full scan, for the same reason as [`GraphDB::all_nodes`]: a relationship
    /// pattern with no type filter (`-[r]->`) cannot use the edge-type index.
    pub fn all_edges(&self) -> Vec<Edge> {
        self.edges.iter().map(|e| e.value().clone()).collect()
    }

    /// Get nodes by property
    pub fn get_nodes_by_property(&self, key: &str, value: &PropertyValue) -> Vec<Node> {
        self.property_index
            .get_nodes_by_property(key, value)
            .into_iter()
            .filter_map(|id| self.get_node(&id))
            .collect()
    }

    // Edge operations

    /// Create an edge
    pub fn create_edge(&self, edge: Edge) -> Result<EdgeId> {
        let id = edge.id.clone();

        // Verify nodes exist
        if !self.nodes.contains_key(&edge.from) || !self.nodes.contains_key(&edge.to) {
            return Err(crate::error::GraphError::NodeNotFound(
                "Source or target node not found".to_string(),
            ));
        }

        // Update indexes
        self.edge_type_index.add_edge(&edge);
        self.adjacency_index.add_edge(&edge);

        // Insert into memory
        self.edges.insert(id.clone(), edge.clone());

        // Persist to storage if available
        #[cfg(feature = "storage")]
        if let Some(storage) = &self.storage {
            storage.insert_edge(&edge)?;
        }

        Ok(id)
    }

    /// Get an edge by ID
    pub fn get_edge(&self, id: impl AsRef<str>) -> Option<Edge> {
        self.edges.get(id.as_ref()).map(|entry| entry.clone())
    }

    /// Delete an edge
    pub fn delete_edge(&self, id: impl AsRef<str>) -> Result<bool> {
        if let Some((_, edge)) = self.edges.remove(id.as_ref()) {
            // Update indexes
            self.edge_type_index.remove_edge(&edge);
            self.adjacency_index.remove_edge(&edge);

            // Delete from storage if available
            #[cfg(feature = "storage")]
            if let Some(storage) = &self.storage {
                storage.delete_edge(id.as_ref())?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete multiple edges (batch)
    pub fn delete_edges_batch(&self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let mut deleted = 0;
        let mut edges_to_update = Vec::with_capacity(ids.len());

        for id in ids {
            let key: &str = id.as_ref();
            if let Some((_, edge)) = self.edges.remove(key) {
                edges_to_update.push(edge);
                deleted += 1;
            }
        }

        for edge in &edges_to_update {
            self.edge_type_index.remove_edge(edge);
            self.adjacency_index.remove_edge(edge);
        }

        #[cfg(feature = "storage")]
        if let Some(storage) = &self.storage {
            let str_ids = ids.iter().map(|id| id.as_ref()).collect::<Vec<_>>();
            storage.delete_edges_batch(&str_ids)?;
        }

        Ok(deleted)
    }

    /// Get edges by type
    pub fn get_edges_by_type(&self, edge_type: &str) -> Vec<Edge> {
        self.edge_type_index
            .get_edges_by_type(edge_type)
            .into_iter()
            .filter_map(|id| self.get_edge(&id))
            .collect()
    }

    /// Get outgoing edges from a node
    pub fn get_outgoing_edges(&self, node_id: &NodeId) -> Vec<Edge> {
        self.adjacency_index
            .get_outgoing_edges(node_id)
            .into_iter()
            .filter_map(|id| self.get_edge(&id))
            .collect()
    }

    /// Get incoming edges to a node
    pub fn get_incoming_edges(&self, node_id: &NodeId) -> Vec<Edge> {
        self.adjacency_index
            .get_incoming_edges(node_id)
            .into_iter()
            .filter_map(|id| self.get_edge(&id))
            .collect()
    }

    /// Checks whether an edge exists from `from` → `to` with type `edge_type`.
    /// Returns true if found, false otherwise.
    ///
    /// Fast path: avoids cloning `Edge` by reading fields through the `DashMap`
    /// reference guard and short-circuits on first match.
    pub fn has_edge(&self, from: &NodeId, to: &NodeId, edge_type: &str) -> bool {
        self.adjacency_index
            .get_outgoing_edges(from)
            .into_iter()
            .any(|id| {
                self.edges
                    .get(&id)
                    .is_some_and(|e| e.to == *to && e.edge_type == edge_type)
            })
    }

    /// Get outgoing edges for multiple nodes in one call (O(k×avg_degree) vs O(E) for full scan).
    pub fn get_edges_for_nodes(&self, node_ids: &[NodeId]) -> Vec<Edge> {
        let mut result = Vec::with_capacity(node_ids.len() * 4);
        self.adjacency_index
            .for_each_outgoing_edge(node_ids, |edge_id| {
                if let Some(edge) = self.edges.get(edge_id.as_str()) {
                    result.push(edge.clone());
                }
            });

        result
    }

    // Hyperedge operations

    /// Create a hyperedge
    pub fn create_hyperedge(&self, hyperedge: Hyperedge) -> Result<HyperedgeId> {
        let id = hyperedge.id.clone();

        // Verify all nodes exist
        for node_id in &hyperedge.nodes {
            if !self.nodes.contains_key(node_id) {
                return Err(crate::error::GraphError::NodeNotFound(format!(
                    "Node {} not found",
                    node_id
                )));
            }
        }

        // Update index
        self.hyperedge_node_index.add_hyperedge(&hyperedge);

        // Insert into memory
        self.hyperedges.insert(id.clone(), hyperedge.clone());

        // Persist to storage if available
        #[cfg(feature = "storage")]
        if let Some(storage) = &self.storage {
            storage.insert_hyperedge(&hyperedge)?;
        }

        Ok(id)
    }

    /// Get a hyperedge by ID
    pub fn get_hyperedge(&self, id: &HyperedgeId) -> Option<Hyperedge> {
        self.hyperedges.get(id).map(|entry| entry.clone())
    }

    /// Get hyperedges containing a node
    pub fn get_hyperedges_by_node(&self, node_id: &NodeId) -> Vec<Hyperedge> {
        self.hyperedge_node_index
            .get_hyperedges_by_node(node_id)
            .into_iter()
            .filter_map(|id| self.get_hyperedge(&id))
            .collect()
    }

    /// Delete a hyperedge by ID
    pub fn delete_hyperedge(&self, id: &HyperedgeId) -> Result<bool> {
        if let Some((_, hyperedge)) = self.hyperedges.remove(id) {
            self.hyperedge_node_index.remove_hyperedge(&hyperedge);

            #[cfg(feature = "storage")]
            if let Some(storage) = &self.storage {
                storage.delete_hyperedge(id)?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all hyperedges that contain a given node
    pub fn delete_hyperedges_by_node(&self, node_id: &NodeId) -> Result<usize> {
        let ids: Vec<HyperedgeId> = self.hyperedge_node_index.get_hyperedges_by_node(node_id);
        let mut deleted = 0;
        for id in &ids {
            if self.delete_hyperedge(id)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    // Statistics

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of hyperedges
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }
}

impl Default for GraphDB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeBuilder;
    use crate::hyperedge::HyperedgeBuilder;
    use crate::node::NodeBuilder;
    use std::sync::{Arc, Barrier};

    #[test]
    fn test_graph_creation() {
        let db = GraphDB::new();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_node_operations() {
        let db = GraphDB::new();

        let node = NodeBuilder::new()
            .label("Person")
            .property("name", "Alice")
            .build();

        let id = db.create_node(node.clone()).unwrap();
        assert_eq!(db.node_count(), 1);

        let retrieved = db.get_node(&id);
        assert!(retrieved.is_some());

        let deleted = db.delete_node(&id).unwrap();
        assert!(deleted);
        assert_eq!(db.node_count(), 0);
    }

    #[test]
    fn test_edge_operations() {
        let db = GraphDB::new();

        let node1 = NodeBuilder::new().build();
        let node2 = NodeBuilder::new().build();

        let id1 = db.create_node(node1.clone()).unwrap();
        let id2 = db.create_node(node2.clone()).unwrap();

        let edge = EdgeBuilder::new(id1.clone(), id2.clone(), "KNOWS")
            .property("since", 2020i64)
            .build();

        let edge_id = db.create_edge(edge).unwrap();
        assert_eq!(db.edge_count(), 1);

        let retrieved = db.get_edge(&edge_id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_label_index() {
        let db = GraphDB::new();

        let node1 = NodeBuilder::new().label("Person").build();
        let node2 = NodeBuilder::new().label("Person").build();
        let node3 = NodeBuilder::new().label("Organization").build();

        db.create_node(node1).unwrap();
        db.create_node(node2).unwrap();
        db.create_node(node3).unwrap();

        let people = db.get_nodes_by_label("Person");
        assert_eq!(people.len(), 2);

        let orgs = db.get_nodes_by_label("Organization");
        assert_eq!(orgs.len(), 1);
    }

    #[test]
    fn test_hyperedge_operations() {
        let db = GraphDB::new();

        let node1 = NodeBuilder::new().build();
        let node2 = NodeBuilder::new().build();
        let node3 = NodeBuilder::new().build();

        let id1 = db.create_node(node1).unwrap();
        let id2 = db.create_node(node2).unwrap();
        let id3 = db.create_node(node3).unwrap();

        let hyperedge =
            HyperedgeBuilder::new(vec![id1.clone(), id2.clone(), id3.clone()], "MEETING")
                .description("Team meeting")
                .build();

        let hedge_id = db.create_hyperedge(hyperedge).unwrap();
        assert_eq!(db.hyperedge_count(), 1);

        let hedges = db.get_hyperedges_by_node(&id1);
        assert_eq!(hedges.len(), 1);
    }

    #[test]
    fn test_update_node() {
        let db = GraphDB::new();

        let node = NodeBuilder::new()
            .id("mem-001")
            .label("Memory")
            .property("content", "original content")
            .property("status", "active")
            .build();

        db.create_node(node).unwrap();

        // Update: mark as deprecated
        let updated = db
            .update_node("mem-001", |n| {
                n.set_property("status", PropertyValue::from("deprecated"));
                n.set_property("deprecated_at", PropertyValue::from("2026-07-12T12:00:00Z"));
            })
            .unwrap();
        assert!(updated);

        let retrieved = db.get_node("mem-001").unwrap();
        assert_eq!(
            retrieved.get_property("status").unwrap(),
            &PropertyValue::from("deprecated")
        );
        assert!(retrieved.get_property("deprecated_at").is_some());

        assert!(db
            .get_nodes_by_property("status", &PropertyValue::from("active"))
            .is_empty());
        assert_eq!(
            db.get_nodes_by_property("status", &PropertyValue::from("deprecated"))
                .len(),
            1
        );
    }

    #[test]
    fn test_update_node_refreshes_label_and_property_indexes() {
        let db = GraphDB::new();
        db.create_node(
            NodeBuilder::new()
                .id("indexed")
                .label("OldLabel")
                .property("state", "old")
                .property("removed", true)
                .build(),
        )
        .unwrap();

        db.update_node("indexed", |node| {
            node.remove_label("OldLabel");
            node.add_label("NewLabel");
            node.set_property("state", PropertyValue::from("new"));
            node.properties.remove("removed");
        })
        .unwrap();

        assert!(db.get_nodes_by_label("OldLabel").is_empty());
        assert_eq!(db.get_nodes_by_label("NewLabel").len(), 1);
        assert!(db
            .get_nodes_by_property("state", &PropertyValue::from("old"))
            .is_empty());
        assert_eq!(
            db.get_nodes_by_property("state", &PropertyValue::from("new"))
                .len(),
            1
        );
        assert!(db
            .get_nodes_by_property("removed", &PropertyValue::from(true))
            .is_empty());
    }

    #[test]
    fn test_update_node_rejects_id_changes_without_side_effects() {
        let db = GraphDB::new();
        db.create_node(NodeBuilder::new().id("original").label("Old").build())
            .unwrap();

        let error = db
            .update_node("original", |node| {
                node.id = "replacement".to_string();
                node.add_label("New");
            })
            .unwrap_err();

        assert!(matches!(
            error,
            crate::error::GraphError::ConstraintViolation(_)
        ));
        assert!(db.get_node("replacement").is_none());
        assert!(db.get_node("original").unwrap().has_label("Old"));
        assert!(db.get_nodes_by_label("New").is_empty());
    }

    #[test]
    fn test_concurrent_updates_do_not_lose_writes() {
        const THREADS: usize = 8;
        const UPDATES_PER_THREAD: usize = 100;

        let db = Arc::new(GraphDB::new());
        db.create_node(
            NodeBuilder::new()
                .id("counter")
                .property("value", 0_i64)
                .build(),
        )
        .unwrap();
        let barrier = Arc::new(Barrier::new(THREADS));
        let mut handles = Vec::new();

        for _ in 0..THREADS {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..UPDATES_PER_THREAD {
                    db.update_node("counter", |node| {
                        let value = match node.get_property("value") {
                            Some(PropertyValue::Integer(value)) => *value,
                            _ => panic!("counter property is missing"),
                        };
                        node.set_property("value", PropertyValue::Integer(value + 1));
                    })
                    .unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(
            db.get_node("counter")
                .unwrap()
                .get_property("value")
                .cloned(),
            Some(PropertyValue::Integer(
                (THREADS * UPDATES_PER_THREAD) as i64
            ))
        );
    }

    #[test]
    fn test_update_node_not_found() {
        let db = GraphDB::new();
        let result = db.update_node("nonexistent", |_| {}).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_keyword_search() {
        let db = GraphDB::new();

        let docs = vec![
            ("mem-1", "the quick brown fox jumps over the lazy dog"),
            ("mem-2", "machine learning models for vector search"),
            ("mem-3", "vector databases enable semantic search at scale"),
            ("mem-4", "a recipe for italian pasta with tomato sauce"),
        ];

        for (id, text) in docs {
            let node = NodeBuilder::new()
                .id(id)
                .label("Memory")
                .property("content", text)
                .build();
            db.create_node(node).unwrap();
        }

        let hits = db
            .keyword_search("Memory", "content", "vector search", 4)
            .unwrap();

        assert!(!hits.is_empty());
        // mem-2 and mem-3 both mention "vector" and "search"; pasta doc must not lead.
        assert!(hits[0].0 == "mem-2" || hits[0].0 == "mem-3");
        assert!(hits.iter().all(|(id, _)| id != "mem-4") || hits.last().unwrap().0 == "mem-4");
    }

    #[test]
    fn test_keyword_search_empty_label() {
        let db = GraphDB::new();
        let hits = db
            .keyword_search("Nonexistent", "content", "anything", 5)
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn test_keyword_search_ignores_other_labels_and_non_string_fields() {
        let db = GraphDB::new();
        for node in [
            NodeBuilder::new()
                .id("wanted")
                .label("Memory")
                .property("content", "unique needle")
                .build(),
            NodeBuilder::new()
                .id("wrong-label")
                .label("Other")
                .property("content", "unique needle")
                .build(),
            NodeBuilder::new()
                .id("wrong-type")
                .label("Memory")
                .property("content", 42_i64)
                .build(),
        ] {
            db.create_node(node).unwrap();
        }

        let hits = db
            .keyword_search("Memory", "content", "needle", 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "wanted");
        assert!(db
            .keyword_search("Memory", "content", "needle", 0)
            .unwrap()
            .is_empty());
    }

    #[cfg(feature = "storage")]
    #[test]
    fn test_update_node_persists() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("graph.redb");

        {
            let db = GraphDB::with_storage(&path).unwrap();
            db.create_node(
                NodeBuilder::new()
                    .id("persistent")
                    .label("Old")
                    .property("state", "old")
                    .build(),
            )
            .unwrap();
            db.update_node("persistent", |node| {
                node.remove_label("Old");
                node.add_label("New");
                node.set_property("state", PropertyValue::from("new"));
            })
            .unwrap();
        }

        let reopened = GraphDB::with_storage(&path).unwrap();
        let node = reopened.get_node("persistent").unwrap();
        assert!(node.has_label("New"));
        assert_eq!(
            node.get_property("state"),
            Some(&PropertyValue::from("new"))
        );
        assert_eq!(reopened.get_nodes_by_label("New").len(), 1);
        assert_eq!(
            reopened
                .get_nodes_by_property("state", &PropertyValue::from("new"))
                .len(),
            1
        );
    }
}
