//! Entity graph for semantic world-model objects.
//!
//! The [`EntityGraph`] stores typed nodes ([`Entity`]) and edges ([`Edge`]) that
//! represent objects, tracks, regions, and events in the world model, along with
//! their relationships (adjacency, containment, causality, etc.).

use std::collections::HashMap;

/// A world model entity (object, region, track, or event).
#[derive(Clone, Debug)]
pub struct Entity {
    /// Unique entity identifier.
    pub id: u64,
    /// Semantic type of this entity.
    pub entity_type: EntityType,
    /// Temporal extent [start_time, end_time].
    pub time_span: [f32; 2],
    /// Embedding vector for similarity search.
    pub embedding: Vec<f32>,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Privacy/access-control tags.
    pub privacy_tags: Vec<String>,
    /// Arbitrary key-value attributes.
    pub attributes: Vec<(String, AttributeValue)>,
    /// IDs of Gaussians associated with this entity.
    pub gaussian_ids: Vec<u32>,
}

/// Semantic type of an entity.
#[derive(Clone, Debug)]
pub enum EntityType {
    /// A physical object with a class label.
    Object {
        /// Semantic class (e.g. "car", "person", "tree").
        class: String,
    },
    /// A temporal track (sequence of observations of the same object).
    Track,
    /// A spatial region.
    Region,
    /// A discrete event.
    Event,
}

/// A dynamically-typed attribute value.
#[derive(Clone, Debug)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    Text(String),
    Bool(bool),
    Vec3([f32; 3]),
}

/// A typed, weighted edge in the entity graph.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Source entity ID.
    pub source: u64,
    /// Target entity ID.
    pub target: u64,
    /// Semantic type of the relationship.
    pub edge_type: EdgeType,
    /// Edge weight (interpretation depends on edge type).
    pub weight: f32,
    /// Optional temporal extent of the relationship.
    pub time_range: Option<[f32; 2]>,
}

/// Types of relationships between entities.
#[derive(Clone, Debug)]
pub enum EdgeType {
    /// Spatial adjacency.
    Adjacency,
    /// One entity contains the other.
    Containment,
    /// Temporal continuity (same object across time).
    Continuity,
    /// Causal relationship.
    Causality,
    /// Identity link (two observations of the same entity).
    SameIdentity,
}

/// In-memory entity graph with index for fast ID lookups.
pub struct EntityGraph {
    entities: Vec<Entity>,
    edges: Vec<Edge>,
    id_index: HashMap<u64, usize>,
}

impl EntityGraph {
    /// Create an empty entity graph.
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            edges: Vec::new(),
            id_index: HashMap::new(),
        }
    }

    /// Add an entity to the graph.
    ///
    /// If an entity with the same ID already exists, it is replaced.
    pub fn add_entity(&mut self, entity: Entity) {
        let id = entity.id;
        if let Some(&existing_idx) = self.id_index.get(&id) {
            self.entities[existing_idx] = entity;
        } else {
            let idx = self.entities.len();
            self.id_index.insert(id, idx);
            self.entities.push(entity);
        }
    }

    /// Look up an entity by ID.
    pub fn get_entity(&self, id: u64) -> Option<&Entity> {
        self.id_index.get(&id).map(|&idx| &self.entities[idx])
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Find all entities connected to the given entity by any edge (as source or target).
    ///
    /// Note: This is O(E) where E is the total number of edges, as it performs a
    /// linear scan over all edges. Consider an adjacency-list index for large graphs.
    pub fn neighbors(&self, id: u64) -> Vec<&Entity> {
        let mut neighbor_ids = Vec::new();

        // Linear scan of all edges -- O(E)
        for edge in &self.edges {
            if edge.source == id {
                neighbor_ids.push(edge.target);
            } else if edge.target == id {
                neighbor_ids.push(edge.source);
            }
        }

        // Deduplicate
        neighbor_ids.sort_unstable();
        neighbor_ids.dedup();

        neighbor_ids
            .iter()
            .filter_map(|&nid| self.get_entity(nid))
            .collect()
    }

    /// Query entities by type name.
    ///
    /// The `entity_type` string is matched against the variant name (case-insensitive)
    /// or, for `Object` types, the class label.
    ///
    /// Note: This is O(N) where N is the total number of entities, as it iterates
    /// over all entities. Consider a type index for frequent queries on large graphs.
    pub fn query_by_type(&self, entity_type: &str) -> Vec<&Entity> {
        let lower = entity_type.to_lowercase();
        self.entities
            .iter()
            .filter(|e| match &e.entity_type {
                EntityType::Object { class } => {
                    lower == "object" || class.to_lowercase() == lower
                }
                EntityType::Track => lower == "track",
                EntityType::Region => lower == "region",
                EntityType::Event => lower == "event",
            })
            .collect()
    }

    /// Query entities whose time span overlaps with [start, end].
    pub fn query_time_range(&self, start: f32, end: f32) -> Vec<&Entity> {
        self.entities
            .iter()
            .filter(|e| e.time_span[0] <= end && e.time_span[1] >= start)
            .collect()
    }

    /// Return the number of entities in the graph.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Return the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Search entities by cosine similarity against a query embedding.
    ///
    /// Returns entities whose embedding similarity exceeds `threshold`,
    /// sorted by descending similarity. Each result is `(entity_ref, score)`.
    ///
    /// Entities with empty embeddings or dimension mismatches are skipped.
    pub fn search_by_embedding(&self, query: &[f32], threshold: f32) -> Vec<(&Entity, f32)> {
        if query.is_empty() {
            return Vec::new();
        }
        let query_norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut results: Vec<(&Entity, f32)> = self
            .entities
            .iter()
            .filter_map(|e| {
                if e.embedding.len() != query.len() || e.embedding.is_empty() {
                    return None;
                }
                let sim = cosine_similarity(query, &e.embedding, query_norm);
                if sim >= threshold {
                    Some((e, sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find the top-k most similar entities to the query embedding.
    ///
    /// Returns up to `k` results sorted by descending similarity.
    pub fn top_k_by_embedding(&self, query: &[f32], k: usize) -> Vec<(&Entity, f32)> {
        let results = self.search_by_embedding(query, f32::NEG_INFINITY);
        results.into_iter().take(k).collect()
    }
}

/// Compute cosine similarity between two vectors.
///
/// `a_norm` is the precomputed L2 norm of `a`.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let b_norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if b_norm > 0.0 && a_norm > 0.0 {
        dot / (a_norm * b_norm)
    } else {
        0.0
    }
}

impl Default for EntityGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u64, class: &str, time: [f32; 2]) -> Entity {
        Entity {
            id,
            entity_type: EntityType::Object {
                class: class.to_string(),
            },
            time_span: time,
            embedding: vec![],
            confidence: 0.9,
            privacy_tags: vec![],
            attributes: vec![],
            gaussian_ids: vec![],
        }
    }

    fn make_track(id: u64, time: [f32; 2]) -> Entity {
        Entity {
            id,
            entity_type: EntityType::Track,
            time_span: time,
            embedding: vec![],
            confidence: 0.9,
            privacy_tags: vec![],
            attributes: vec![],
            gaussian_ids: vec![],
        }
    }

    #[test]
    fn test_add_and_get_entity() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity(1, "car", [0.0, 10.0]));
        let e = graph.get_entity(1).unwrap();
        assert_eq!(e.id, 1);
        assert_eq!(graph.entity_count(), 1);
    }

    #[test]
    fn test_replace_entity() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity(1, "car", [0.0, 10.0]));
        graph.add_entity(make_entity(1, "truck", [0.0, 20.0]));
        assert_eq!(graph.entity_count(), 1);
        let e = graph.get_entity(1).unwrap();
        match &e.entity_type {
            EntityType::Object { class } => assert_eq!(class, "truck"),
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn test_get_nonexistent() {
        let graph = EntityGraph::new();
        assert!(graph.get_entity(999).is_none());
    }

    #[test]
    fn test_neighbors() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity(1, "car", [0.0, 10.0]));
        graph.add_entity(make_entity(2, "person", [0.0, 10.0]));
        graph.add_entity(make_entity(3, "tree", [0.0, 10.0]));

        graph.add_edge(Edge {
            source: 1,
            target: 2,
            edge_type: EdgeType::Adjacency,
            weight: 1.0,
            time_range: None,
        });
        graph.add_edge(Edge {
            source: 3,
            target: 1,
            edge_type: EdgeType::Adjacency,
            weight: 0.5,
            time_range: None,
        });

        let neighbors = graph.neighbors(1);
        assert_eq!(neighbors.len(), 2);
        let ids: Vec<u64> = neighbors.iter().map(|e| e.id).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_query_by_type() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity(1, "car", [0.0, 10.0]));
        graph.add_entity(make_entity(2, "car", [0.0, 10.0]));
        graph.add_entity(make_entity(3, "person", [0.0, 10.0]));
        graph.add_entity(make_track(4, [0.0, 10.0]));

        let cars = graph.query_by_type("car");
        assert_eq!(cars.len(), 2);

        let tracks = graph.query_by_type("track");
        assert_eq!(tracks.len(), 1);

        let objects = graph.query_by_type("object");
        assert_eq!(objects.len(), 3);
    }

    #[test]
    fn test_query_time_range() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity(1, "a", [0.0, 5.0]));
        graph.add_entity(make_entity(2, "b", [3.0, 8.0]));
        graph.add_entity(make_entity(3, "c", [10.0, 15.0]));

        let result = graph.query_time_range(4.0, 6.0);
        assert_eq!(result.len(), 2); // entities 1 and 2
        let ids: Vec<u64> = result.iter().map(|e| e.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_edge_count() {
        let mut graph = EntityGraph::new();
        assert_eq!(graph.edge_count(), 0);
        graph.add_edge(Edge {
            source: 1,
            target: 2,
            edge_type: EdgeType::Causality,
            weight: 1.0,
            time_range: None,
        });
        assert_eq!(graph.edge_count(), 1);
    }

    // -- Embedding search tests --

    fn make_entity_with_embedding(id: u64, class: &str, embedding: Vec<f32>) -> Entity {
        Entity {
            id,
            entity_type: EntityType::Object {
                class: class.to_string(),
            },
            time_span: [0.0, 10.0],
            embedding,
            confidence: 0.9,
            privacy_tags: vec![],
            attributes: vec![],
            gaussian_ids: vec![],
        }
    }

    #[test]
    fn test_search_by_embedding_finds_similar() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity_with_embedding(1, "car", vec![1.0, 0.0, 0.0]));
        graph.add_entity(make_entity_with_embedding(2, "car", vec![0.9, 0.1, 0.0]));
        graph.add_entity(make_entity_with_embedding(3, "tree", vec![0.0, 0.0, 1.0]));

        let results = graph.search_by_embedding(&[1.0, 0.0, 0.0], 0.8);
        assert_eq!(results.len(), 2); // entities 1 and 2
        assert_eq!(results[0].0.id, 1); // exact match first
        assert!(results[0].1 > results[1].1); // sorted by descending similarity
    }

    #[test]
    fn test_search_by_embedding_threshold_filters() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity_with_embedding(1, "a", vec![1.0, 0.0, 0.0]));
        graph.add_entity(make_entity_with_embedding(2, "b", vec![0.0, 1.0, 0.0]));

        let results = graph.search_by_embedding(&[1.0, 0.0, 0.0], 0.99);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 1);
    }

    #[test]
    fn test_search_by_embedding_empty_query() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity_with_embedding(1, "a", vec![1.0, 0.0]));
        assert!(graph.search_by_embedding(&[], 0.0).is_empty());
    }

    #[test]
    fn test_search_by_embedding_dimension_mismatch() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity_with_embedding(1, "a", vec![1.0, 0.0]));
        // 3D query vs 2D embedding → skipped
        assert!(graph.search_by_embedding(&[1.0, 0.0, 0.0], 0.0).is_empty());
    }

    #[test]
    fn test_top_k_by_embedding() {
        let mut graph = EntityGraph::new();
        for i in 0..10 {
            let angle = i as f32 * 0.3;
            graph.add_entity(make_entity_with_embedding(
                i,
                "x",
                vec![angle.cos(), angle.sin(), 0.0],
            ));
        }
        let results = graph.top_k_by_embedding(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // First result should be the most similar
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_search_empty_embeddings_skipped() {
        let mut graph = EntityGraph::new();
        graph.add_entity(make_entity_with_embedding(1, "a", vec![])); // empty
        graph.add_entity(make_entity_with_embedding(2, "b", vec![1.0, 0.0]));
        let results = graph.search_by_embedding(&[1.0, 0.0], 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, 2);
    }
}
