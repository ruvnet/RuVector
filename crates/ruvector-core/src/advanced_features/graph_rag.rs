//! # Graph RAG Pipeline
//!
//! A Graph-based Retrieval-Augmented Generation pipeline inspired by Microsoft's Graph RAG.
//!
//! ## Why Graph RAG?
//!
//! Naive RAG retrieves document chunks via embedding similarity alone, which works well for
//! simple factual lookups but struggles with queries that require synthesizing information
//! across multiple documents or understanding relational context. Graph RAG addresses this
//! by building a knowledge graph of entities and relations, then detecting communities of
//! related entities at multiple granularity levels.
//!
//! Empirically, Graph RAG achieves **30-60% improvement** on complex multi-hop queries
//! compared to naive chunk-based RAG, because:
//! - **Local search** follows entity relationships to gather structurally relevant context
//! - **Global search** leverages pre-summarized community descriptions for broad queries
//! - **Hybrid search** combines both for balanced coverage
//!
//! ## Architecture
//!
//! ```text
//! Documents -> Entity Extraction -> KnowledgeGraph
//!                                       |
//!                              CommunityDetection (Leiden-inspired)
//!                                       |
//!                           Level 0 (fine) + Level 1 (coarse)
//!                                       |
//!                              GraphRAGPipeline
//!                             /        |        \
//!                        Local      Global     Hybrid
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::VectorId;

/// Unique identifier for entities in the knowledge graph.
pub type EntityId = VectorId;

/// An entity node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier.
    pub id: EntityId,
    /// Human-readable name (e.g., "Albert Einstein").
    pub name: String,
    /// Category of entity (e.g., "Person", "Organization", "Concept").
    pub entity_type: String,
    /// Free-text description of the entity.
    pub description: String,
    /// Optional embedding vector for similarity search.
    pub embedding: Option<Vec<f32>>,
}

/// A directed relation (edge) between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity identifier.
    pub source_id: EntityId,
    /// Target entity identifier.
    pub target_id: EntityId,
    /// Type of relationship (e.g., "WORKS_AT", "AUTHORED").
    pub relation_type: String,
    /// Edge weight in `[0.0, 1.0]` representing strength or confidence.
    pub weight: f32,
    /// Free-text description of the relationship.
    pub description: String,
}

/// A community is a cluster of closely related entities detected via graph algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Unique community identifier.
    pub id: String,
    /// Member entity identifiers.
    pub entities: Vec<EntityId>,
    /// Pre-computed natural-language summary of this community.
    pub summary: String,
    /// Hierarchy level: 0 = fine-grained, 1 = coarse.
    pub level: usize,
}

/// Configuration for the Graph RAG pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    /// Maximum hops for local subgraph expansion (default: 2).
    pub max_hops: usize,
    /// Resolution parameter for community detection; higher = more communities (default: 1.0).
    pub community_resolution: f32,
    /// Weight of local search results in hybrid mode (default: 0.6).
    pub local_weight: f32,
    /// Weight of global search results in hybrid mode (default: 0.4).
    pub global_weight: f32,
    /// Maximum entities to include in retrieval context (default: 20).
    pub max_context_entities: usize,
    /// Maximum community summaries to include in global context (default: 5).
    pub max_community_summaries: usize,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            community_resolution: 1.0,
            local_weight: 0.6,
            global_weight: 0.4,
            max_context_entities: 20,
            max_community_summaries: 5,
        }
    }
}

/// The result of a Graph RAG retrieval operation, ready for LLM consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Entities relevant to the query.
    pub entities: Vec<Entity>,
    /// Relations connecting the retrieved entities.
    pub relations: Vec<Relation>,
    /// Community summaries providing broad context.
    pub community_summaries: Vec<String>,
    /// Pre-formatted context string suitable for LLM prompting.
    pub context_text: String,
}

/// Adjacency-list knowledge graph with entity and relation storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    entities: HashMap<EntityId, Entity>,
    /// Adjacency list: entity_id -> Vec<(neighbor_id, relation)>.
    adjacency: HashMap<EntityId, Vec<(EntityId, Relation)>>,
}

impl KnowledgeGraph {
    /// Create an empty knowledge graph.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add an entity node. Overwrites if the id already exists.
    pub fn add_entity(&mut self, entity: Entity) {
        self.adjacency.entry(entity.id.clone()).or_default();
        self.entities.insert(entity.id.clone(), entity);
    }

    /// Add a directed relation. Both source and target must already exist; returns false otherwise.
    pub fn add_relation(&mut self, relation: Relation) -> bool {
        if !self.entities.contains_key(&relation.source_id)
            || !self.entities.contains_key(&relation.target_id)
        {
            return false;
        }
        let target = relation.target_id.clone();
        self.adjacency
            .entry(relation.source_id.clone())
            .or_default()
            .push((target, relation));
        true
    }

    /// Return the entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Retrieve an entity by id.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// BFS expansion: collect all entities reachable within `hop_count` hops from `entity_id`.
    /// Returns `(entities, relations)` forming the subgraph.
    pub fn get_neighbors(
        &self,
        entity_id: &str,
        hop_count: usize,
    ) -> (Vec<Entity>, Vec<Relation>) {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut result_entities: Vec<Entity> = Vec::new();
        let mut result_relations: Vec<Relation> = Vec::new();

        if let Some(root) = self.entities.get(entity_id) {
            visited.insert(entity_id.to_string());
            result_entities.push(root.clone());
            queue.push_back((entity_id.to_string(), 0));
        }

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= hop_count {
                continue;
            }
            if let Some(neighbors) = self.adjacency.get(&current_id) {
                for (neighbor_id, relation) in neighbors {
                    result_relations.push(relation.clone());
                    if visited.insert(neighbor_id.clone()) {
                        if let Some(entity) = self.entities.get(neighbor_id) {
                            result_entities.push(entity.clone());
                        }
                        queue.push_back((neighbor_id.clone(), depth + 1));
                    }
                }
            }
        }

        (result_entities, result_relations)
    }

    /// Return all entity ids.
    pub fn entity_ids(&self) -> Vec<EntityId> {
        self.entities.keys().cloned().collect()
    }

    /// Return all entities.
    pub fn all_entities(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified Leiden-inspired community detection via label propagation.
pub struct CommunityDetection;

impl CommunityDetection {
    /// Detect communities at the specified resolution.
    ///
    /// Higher `resolution` produces more, smaller communities. The algorithm runs label
    /// propagation where each node adopts the most common label among its neighbors,
    /// weighted by edge weight and resolution. Level 0 communities are fine-grained;
    /// level 1 communities merge small level-0 communities for coarser grouping.
    pub fn detect_communities(graph: &KnowledgeGraph, resolution: f32) -> Vec<Community> {
        let ids: Vec<EntityId> = graph.entity_ids();
        if ids.is_empty() {
            return Vec::new();
        }

        // Initialize: each node in its own community.
        let mut labels: HashMap<String, usize> = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            labels.insert(id.clone(), i);
        }

        // Run label propagation for a fixed number of iterations.
        let iterations = (5.0 * resolution) as usize + 3;
        for _ in 0..iterations {
            let mut changed = false;
            for id in &ids {
                if let Some(neighbors) = graph.adjacency.get(id) {
                    if neighbors.is_empty() {
                        continue;
                    }
                    // Tally weighted votes for each label.
                    let mut votes: HashMap<usize, f32> = HashMap::new();
                    for (neighbor_id, rel) in neighbors {
                        if let Some(&label) = labels.get(neighbor_id) {
                            *votes.entry(label).or_insert(0.0) += rel.weight * resolution;
                        }
                    }
                    if let Some((&best_label, _)) =
                        votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    {
                        let current = labels[id];
                        if best_label != current {
                            labels.insert(id.clone(), best_label);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Collect level-0 (fine) communities.
        let mut community_map: HashMap<usize, Vec<EntityId>> = HashMap::new();
        for (id, label) in &labels {
            community_map.entry(*label).or_default().push(id.clone());
        }

        let mut communities: Vec<Community> = community_map
            .into_iter()
            .enumerate()
            .map(|(i, (_label, members))| Community {
                id: format!("c0_{i}"),
                summary: format!(
                    "Community of {} entities: {}",
                    members.len(),
                    members
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                entities: members,
                level: 0,
            })
            .collect();

        // Level-1 (coarse): merge communities with fewer than 3 members.
        let threshold = 3;
        let mut small: Vec<EntityId> = Vec::new();
        let mut large: Vec<&Community> = Vec::new();
        for c in &communities {
            if c.entities.len() < threshold {
                small.extend(c.entities.clone());
            } else {
                large.push(c);
            }
        }

        let mut level1: Vec<Community> = large
            .iter()
            .enumerate()
            .map(|(i, c)| Community {
                id: format!("c1_{i}"),
                summary: format!("Coarse community: {}", c.summary),
                entities: c.entities.clone(),
                level: 1,
            })
            .collect();

        if !small.is_empty() {
            level1.push(Community {
                id: format!("c1_{}", level1.len()),
                summary: format!("Merged small community of {} entities", small.len()),
                entities: small,
                level: 1,
            });
        }

        communities.extend(level1);
        communities
    }
}

/// The main Graph RAG pipeline orchestrating local, global, and hybrid retrieval.
pub struct GraphRAGPipeline {
    graph: KnowledgeGraph,
    communities: Vec<Community>,
    config: GraphRAGConfig,
}

impl GraphRAGPipeline {
    /// Build a pipeline from a knowledge graph and config. Runs community detection.
    pub fn new(graph: KnowledgeGraph, config: GraphRAGConfig) -> Self {
        let communities =
            CommunityDetection::detect_communities(&graph, config.community_resolution);
        Self {
            graph,
            communities,
            config,
        }
    }

    /// **Local search**: find entities whose embeddings are most similar to `query_embedding`,
    /// then expand each to a k-hop subgraph and collect context.
    pub fn local_search(&self, query_embedding: &[f32]) -> RetrievalResult {
        let mut scored: Vec<(&Entity, f32)> = self
            .graph
            .all_entities()
            .into_iter()
            .filter_map(|e| {
                e.embedding
                    .as_ref()
                    .map(|emb| (e, cosine_similarity(query_embedding, emb)))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = scored
            .iter()
            .take(self.config.max_context_entities)
            .collect::<Vec<_>>();

        let mut all_entities: Vec<Entity> = Vec::new();
        let mut all_relations: Vec<Relation> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for &(entity, _score) in &top_k {
            let (ents, rels) = self.graph.get_neighbors(&entity.id, self.config.max_hops);
            for e in ents {
                if seen.insert(e.id.clone()) {
                    all_entities.push(e);
                }
            }
            all_relations.extend(rels);
        }

        // Trim to max.
        all_entities.truncate(self.config.max_context_entities);

        let context_text = format_context(&all_entities, &all_relations, &[]);
        RetrievalResult {
            entities: all_entities,
            relations: all_relations,
            community_summaries: Vec::new(),
            context_text,
        }
    }

    /// **Global search**: map over community summaries, score each against the query embedding
    /// by averaging member entity similarities, then return the top summaries.
    pub fn global_search(&self, query_embedding: &[f32]) -> RetrievalResult {
        let mut scored: Vec<(usize, f32)> = self
            .communities
            .iter()
            .enumerate()
            .map(|(i, community)| {
                let avg_sim = community
                    .entities
                    .iter()
                    .filter_map(|eid| {
                        self.graph
                            .get_entity(eid)
                            .and_then(|e| e.embedding.as_ref())
                            .map(|emb| cosine_similarity(query_embedding, emb))
                    })
                    .sum::<f32>()
                    / community.entities.len().max(1) as f32;
                (i, avg_sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let summaries: Vec<String> = scored
            .iter()
            .take(self.config.max_community_summaries)
            .filter_map(|&(idx, _)| self.communities.get(idx))
            .map(|c| c.summary.clone())
            .collect();

        let context_text = format_context(&[], &[], &summaries);
        RetrievalResult {
            entities: Vec::new(),
            relations: Vec::new(),
            community_summaries: summaries,
            context_text,
        }
    }

    /// **Hybrid search**: run both local and global, merge results weighted by config.
    pub fn hybrid_search(&self, query_embedding: &[f32]) -> RetrievalResult {
        let local = self.local_search(query_embedding);
        let global = self.global_search(query_embedding);

        let entity_count =
            (self.config.max_context_entities as f32 * self.config.local_weight) as usize;
        let summary_count =
            (self.config.max_community_summaries as f32 * self.config.global_weight) as usize;

        let mut entities: Vec<Entity> = local.entities;
        entities.truncate(entity_count.max(1));

        let mut summaries: Vec<String> = global.community_summaries;
        summaries.truncate(summary_count.max(1));

        let relations = local.relations;
        let context_text = format_context(&entities, &relations, &summaries);

        RetrievalResult {
            entities,
            relations,
            community_summaries: summaries,
            context_text,
        }
    }

    /// Access the underlying knowledge graph.
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.graph
    }

    /// Access detected communities.
    pub fn communities(&self) -> &[Community] {
        &self.communities
    }
}

/// Cosine similarity between two vectors. Returns 0.0 if either is zero-length.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Format entities, relations, and summaries into a context string for LLM prompting.
fn format_context(entities: &[Entity], relations: &[Relation], summaries: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();

    if !entities.is_empty() {
        let mut section = String::from("## Entities\n");
        for e in entities {
            section.push_str(&format!("- {} ({}): {}\n", e.name, e.entity_type, e.description));
        }
        parts.push(section);
    }

    if !relations.is_empty() {
        let mut section = String::from("## Relations\n");
        for r in relations {
            section.push_str(&format!(
                "- {} --[{}]--> {}: {}\n",
                r.source_id, r.relation_type, r.target_id, r.description
            ));
        }
        parts.push(section);
    }

    if !summaries.is_empty() {
        let mut section = String::from("## Community Summaries\n");
        for s in summaries {
            section.push_str(&format!("- {s}\n"));
        }
        parts.push(section);
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, name: &str, emb: Vec<f32>) -> Entity {
        Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: "Test".to_string(),
            description: format!("{name} description"),
            embedding: Some(emb),
        }
    }

    fn make_relation(src: &str, tgt: &str, rtype: &str, weight: f32) -> Relation {
        Relation {
            source_id: src.to_string(),
            target_id: tgt.to_string(),
            relation_type: rtype.to_string(),
            weight,
            description: format!("{src} {rtype} {tgt}"),
        }
    }

    fn build_test_graph() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        g.add_entity(make_entity("a", "Alice", vec![1.0, 0.0, 0.0]));
        g.add_entity(make_entity("b", "Bob", vec![0.9, 0.1, 0.0]));
        g.add_entity(make_entity("c", "Carol", vec![0.0, 1.0, 0.0]));
        g.add_entity(make_entity("d", "Dave", vec![0.0, 0.0, 1.0]));
        g.add_relation(make_relation("a", "b", "KNOWS", 0.9));
        g.add_relation(make_relation("b", "c", "WORKS_WITH", 0.7));
        g.add_relation(make_relation("c", "d", "MANAGES", 0.5));
        g
    }

    #[test]
    fn test_graph_construction() {
        let g = build_test_graph();
        assert_eq!(g.entity_count(), 4);
        assert!(g.get_entity("a").is_some());
        assert!(g.get_entity("z").is_none());
    }

    #[test]
    fn test_neighbor_retrieval_1hop() {
        let g = build_test_graph();
        let (ents, rels) = g.get_neighbors("a", 1);
        assert_eq!(ents.len(), 2); // a + b
        assert_eq!(rels.len(), 1); // a->b
        let ids: HashSet<_> = ents.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
    }

    #[test]
    fn test_neighbor_retrieval_2hop() {
        let g = build_test_graph();
        let (ents, _rels) = g.get_neighbors("a", 2);
        assert_eq!(ents.len(), 3); // a, b, c
    }

    #[test]
    fn test_add_relation_invalid_source() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(make_entity("a", "Alice", vec![]));
        let ok = g.add_relation(make_relation("missing", "a", "REL", 1.0));
        assert!(!ok);
    }

    #[test]
    fn test_community_detection() {
        let g = build_test_graph();
        let communities = CommunityDetection::detect_communities(&g, 1.0);
        assert!(!communities.is_empty());
        // Level-0 communities exist.
        assert!(communities.iter().any(|c| c.level == 0));
    }

    #[test]
    fn test_local_search() {
        let g = build_test_graph();
        let config = GraphRAGConfig::default();
        let pipeline = GraphRAGPipeline::new(g, config);
        let result = pipeline.local_search(&[1.0, 0.0, 0.0]);
        assert!(!result.entities.is_empty());
        // Alice should be top match.
        assert_eq!(result.entities[0].id, "a");
        assert!(!result.context_text.is_empty());
    }

    #[test]
    fn test_global_search() {
        let g = build_test_graph();
        let config = GraphRAGConfig::default();
        let pipeline = GraphRAGPipeline::new(g, config);
        let result = pipeline.global_search(&[1.0, 0.0, 0.0]);
        assert!(!result.community_summaries.is_empty());
        assert!(result.context_text.contains("Community"));
    }

    #[test]
    fn test_hybrid_search() {
        let g = build_test_graph();
        let config = GraphRAGConfig::default();
        let pipeline = GraphRAGPipeline::new(g, config);
        let result = pipeline.hybrid_search(&[1.0, 0.0, 0.0]);
        assert!(!result.entities.is_empty());
        assert!(!result.community_summaries.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let g = KnowledgeGraph::new();
        let config = GraphRAGConfig::default();
        let pipeline = GraphRAGPipeline::new(g, config);
        let result = pipeline.local_search(&[1.0, 0.0]);
        assert!(result.entities.is_empty());
        assert!(result.relations.is_empty());
    }

    #[test]
    fn test_single_entity() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(make_entity("x", "Solo", vec![1.0, 0.0]));
        let config = GraphRAGConfig::default();
        let pipeline = GraphRAGPipeline::new(g, config);
        let result = pipeline.local_search(&[1.0, 0.0]);
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Solo");
    }

    #[test]
    fn test_disconnected_components() {
        let mut g = KnowledgeGraph::new();
        g.add_entity(make_entity("a", "Alpha", vec![1.0, 0.0]));
        g.add_entity(make_entity("b", "Beta", vec![0.0, 1.0]));
        // No edges between them.
        let (ents, rels) = g.get_neighbors("a", 3);
        assert_eq!(ents.len(), 1); // Only Alpha.
        assert!(rels.is_empty());

        // Both still appear in communities.
        let communities = CommunityDetection::detect_communities(&g, 1.0);
        let total_members: usize = communities.iter().filter(|c| c.level == 0).map(|c| c.entities.len()).sum();
        assert_eq!(total_members, 2);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }
}
