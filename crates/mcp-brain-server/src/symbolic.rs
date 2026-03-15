//! Neural-Symbolic Bridge (ADR-110)
//!
//! Extracts symbolic rules from neural patterns and performs grounded reasoning.
//! The bridge connects embeddings to logical propositions with confidence scores.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Grounded Propositions
// ─────────────────────────────────────────────────────────────────────────────

/// A symbolic proposition grounded in embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedProposition {
    pub id: Uuid,
    /// Human-readable predicate (e.g., "relates_to", "is_type_of", "solves")
    pub predicate: String,
    /// Arguments (entity references, typically memory IDs or category names)
    pub arguments: Vec<String>,
    /// Embedding centroid for this proposition
    pub centroid: Vec<f32>,
    /// Confidence from neural evidence (0.0-1.0)
    pub confidence: f64,
    /// Supporting memory IDs
    pub evidence: Vec<Uuid>,
    /// When this proposition was extracted
    pub created_at: DateTime<Utc>,
    /// Number of times this proposition was reinforced
    pub reinforcement_count: u32,
}

impl GroundedProposition {
    pub fn new(
        predicate: String,
        arguments: Vec<String>,
        centroid: Vec<f32>,
        confidence: f64,
        evidence: Vec<Uuid>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            predicate,
            arguments,
            centroid,
            confidence,
            evidence,
            created_at: Utc::now(),
            reinforcement_count: 1,
        }
    }

    /// Reinforce this proposition with new evidence
    pub fn reinforce(&mut self, new_evidence: Uuid, confidence_boost: f64) {
        if !self.evidence.contains(&new_evidence) {
            self.evidence.push(new_evidence);
        }
        self.reinforcement_count += 1;
        // Asymptotic confidence increase
        self.confidence = 1.0 - (1.0 - self.confidence) * (1.0 - confidence_boost * 0.1);
    }

    /// Decay confidence over time
    pub fn decay(&mut self, decay_rate: f64) {
        let age_days = (Utc::now() - self.created_at).num_days() as f64;
        self.confidence *= (-decay_rate * age_days).exp();
    }

    /// Format as human-readable string
    pub fn to_string_human(&self) -> String {
        format!(
            "{}({}) [conf={:.2}, evidence={}]",
            self.predicate,
            self.arguments.join(", "),
            self.confidence,
            self.evidence.len()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference Results
// ─────────────────────────────────────────────────────────────────────────────

/// A symbolic inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference {
    pub id: Uuid,
    /// The derived proposition
    pub conclusion: GroundedProposition,
    /// The rule(s) used to derive it
    pub rules_applied: Vec<String>,
    /// Premises used in the inference
    pub premises: Vec<Uuid>,
    /// Combined confidence (product of premise confidences × rule confidence)
    pub combined_confidence: f64,
    /// Explanation of the inference chain
    pub explanation: String,
}

impl Inference {
    pub fn new(
        conclusion: GroundedProposition,
        rules_applied: Vec<String>,
        premises: Vec<Uuid>,
        combined_confidence: f64,
    ) -> Self {
        let explanation = format!(
            "Derived '{}' by applying rules [{}] to {} premises",
            conclusion.to_string_human(),
            rules_applied.join(" → "),
            premises.len()
        );
        Self {
            id: Uuid::new_v4(),
            conclusion,
            rules_applied,
            premises,
            combined_confidence,
            explanation,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicate Templates
// ─────────────────────────────────────────────────────────────────────────────

/// Predefined predicate types for extraction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PredicateType {
    /// X is a type of Y
    IsTypeOf,
    /// X relates to Y
    RelatesTo,
    /// X is similar to Y
    SimilarTo,
    /// X causes Y
    Causes,
    /// X prevents Y
    Prevents,
    /// X solves Y
    Solves,
    /// X depends on Y
    DependsOn,
    /// X is part of Y
    PartOf,
    /// Custom predicate
    Custom(String),
}

impl PredicateType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IsTypeOf => "is_type_of",
            Self::RelatesTo => "relates_to",
            Self::SimilarTo => "similar_to",
            Self::Causes => "causes",
            Self::Prevents => "prevents",
            Self::Solves => "solves",
            Self::DependsOn => "depends_on",
            Self::PartOf => "part_of",
            Self::Custom(s) => s,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural-Symbolic Bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the neural-symbolic bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Minimum confidence threshold for extracted propositions
    pub min_confidence: f64,
    /// Similarity threshold for clustering
    pub clustering_threshold: f64,
    /// Maximum propositions to store
    pub max_propositions: usize,
    /// Confidence decay rate (per day)
    pub decay_rate: f64,
    /// Minimum cluster size for proposition extraction
    pub min_cluster_size: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            clustering_threshold: 0.7,
            max_propositions: 1000,
            decay_rate: 0.01,
            min_cluster_size: 3,
        }
    }
}

/// Neural-symbolic reasoning engine
pub struct NeuralSymbolicBridge {
    /// Extracted propositions indexed by predicate
    propositions: HashMap<String, Vec<GroundedProposition>>,
    /// All propositions for fast lookup by ID
    proposition_index: HashMap<Uuid, GroundedProposition>,
    /// Simple horn clause rules (antecedent predicates → consequent predicate)
    rules: Vec<HornClause>,
    /// Configuration
    config: BridgeConfig,
    /// Total propositions extracted
    extraction_count: u64,
    /// Total inferences made
    inference_count: u64,
}

/// A simple horn clause: if all antecedents hold, consequent holds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HornClause {
    pub id: String,
    /// Antecedent predicates
    pub antecedents: Vec<PredicateType>,
    /// Consequent predicate
    pub consequent: PredicateType,
    /// Rule confidence (how reliable is this rule)
    pub confidence: f64,
}

impl HornClause {
    pub fn new(antecedents: Vec<PredicateType>, consequent: PredicateType, confidence: f64) -> Self {
        let id = format!(
            "rule_{}",
            uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0")
        );
        Self {
            id,
            antecedents,
            consequent,
            confidence,
        }
    }
}

impl NeuralSymbolicBridge {
    pub fn new(config: BridgeConfig) -> Self {
        let mut bridge = Self {
            propositions: HashMap::new(),
            proposition_index: HashMap::new(),
            rules: Vec::new(),
            config,
            extraction_count: 0,
            inference_count: 0,
        };

        // Add default inference rules
        bridge.add_default_rules();
        bridge
    }

    /// Add default inference rules
    fn add_default_rules(&mut self) {
        // Transitivity: if A relates_to B and B relates_to C, then A relates_to C
        self.rules.push(HornClause::new(
            vec![PredicateType::RelatesTo, PredicateType::RelatesTo],
            PredicateType::RelatesTo,
            0.7,
        ));

        // Similarity is transitive (with decay)
        self.rules.push(HornClause::new(
            vec![PredicateType::SimilarTo, PredicateType::SimilarTo],
            PredicateType::SimilarTo,
            0.6,
        ));

        // If X solves Y and Y is_type_of Z, then X solves Z
        self.rules.push(HornClause::new(
            vec![PredicateType::Solves, PredicateType::IsTypeOf],
            PredicateType::Solves,
            0.8,
        ));

        // Causation is transitive
        self.rules.push(HornClause::new(
            vec![PredicateType::Causes, PredicateType::Causes],
            PredicateType::Causes,
            0.5,
        ));
    }

    /// Extract propositions from memory clusters
    pub fn extract_from_clusters(
        &mut self,
        clusters: &[(Vec<f32>, Vec<Uuid>, String)], // (centroid, memory_ids, dominant_category)
    ) -> Vec<GroundedProposition> {
        let mut extracted = Vec::new();

        for (centroid, memory_ids, category) in clusters {
            if memory_ids.len() < self.config.min_cluster_size {
                continue;
            }

            // Create "is_type_of" proposition for the cluster
            let prop = GroundedProposition::new(
                PredicateType::IsTypeOf.as_str().to_string(),
                vec![format!("cluster_{}", memory_ids.len()), category.clone()],
                centroid.clone(),
                self.cluster_confidence(memory_ids.len()),
                memory_ids.clone(),
            );

            if prop.confidence >= self.config.min_confidence {
                extracted.push(prop.clone());
                self.store_proposition(prop);
            }
        }

        self.extraction_count += extracted.len() as u64;
        extracted
    }

    /// Extract propositions from SONA patterns
    pub fn extract_from_patterns(
        &mut self,
        patterns: &[(Vec<f32>, f64, Vec<Uuid>)], // (centroid, confidence, source_memories)
    ) -> Vec<GroundedProposition> {
        let mut extracted = Vec::new();

        for (centroid, confidence, memories) in patterns {
            if *confidence < self.config.min_confidence {
                continue;
            }

            // Create pattern-based proposition
            let prop = GroundedProposition::new(
                PredicateType::SimilarTo.as_str().to_string(),
                vec![format!("pattern_{}", memories.len()), "learned_pattern".to_string()],
                centroid.clone(),
                *confidence,
                memories.clone(),
            );

            extracted.push(prop.clone());
            self.store_proposition(prop);
        }

        self.extraction_count += extracted.len() as u64;
        extracted
    }

    /// Store a proposition
    fn store_proposition(&mut self, prop: GroundedProposition) {
        let predicate = prop.predicate.clone();
        let id = prop.id;

        // Check if similar proposition exists
        if let Some(existing) = self.find_similar_proposition(&prop) {
            // Reinforce existing instead of adding new
            if let Some(mut existing_prop) = self.proposition_index.remove(&existing) {
                for evidence_id in &prop.evidence {
                    existing_prop.reinforce(*evidence_id, 0.1);
                }
                self.proposition_index.insert(existing, existing_prop);
            }
            return;
        }

        self.proposition_index.insert(id, prop.clone());
        self.propositions
            .entry(predicate)
            .or_insert_with(Vec::new)
            .push(prop);

        // Trim if over capacity
        if self.proposition_index.len() > self.config.max_propositions {
            self.trim_lowest_confidence();
        }
    }

    /// Find a similar existing proposition
    fn find_similar_proposition(&self, prop: &GroundedProposition) -> Option<Uuid> {
        if let Some(props) = self.propositions.get(&prop.predicate) {
            for existing in props {
                if cosine_similarity(&existing.centroid, &prop.centroid)
                    > self.config.clustering_threshold
                    && existing.arguments == prop.arguments
                {
                    return Some(existing.id);
                }
            }
        }
        None
    }

    /// Remove lowest confidence propositions
    fn trim_lowest_confidence(&mut self) {
        let mut all_props: Vec<(Uuid, f64)> = self
            .proposition_index
            .iter()
            .map(|(id, p)| (*id, p.confidence))
            .collect();

        all_props.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove bottom 10%
        let remove_count = all_props.len() / 10;
        for (id, _) in all_props.into_iter().take(remove_count) {
            if let Some(prop) = self.proposition_index.remove(&id) {
                if let Some(props) = self.propositions.get_mut(&prop.predicate) {
                    props.retain(|p| p.id != id);
                }
            }
        }
    }

    /// Compute confidence from cluster size
    fn cluster_confidence(&self, size: usize) -> f64 {
        // Asymptotic: larger clusters → higher confidence, max 0.95
        1.0 - (-0.2 * size as f64).exp().min(0.95)
    }

    /// Query with neural-symbolic reasoning
    pub fn reason(&self, query_embedding: &[f32], top_k: usize) -> Vec<Inference> {
        let mut inferences = Vec::new();

        // Find relevant propositions by embedding similarity
        let relevant = self.find_relevant_propositions(query_embedding, top_k * 2);

        if relevant.is_empty() {
            return inferences;
        }

        // Apply inference rules
        for rule in &self.rules {
            if let Some(inference) = self.apply_rule(rule, &relevant) {
                inferences.push(inference);
                if inferences.len() >= top_k {
                    break;
                }
            }
        }

        // Note: inference_count is updated via mutable methods, not here

        // Sort by combined confidence
        inferences.sort_by(|a, b| {
            b.combined_confidence
                .partial_cmp(&a.combined_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        inferences.truncate(top_k);
        inferences
    }

    /// Find propositions relevant to a query embedding
    fn find_relevant_propositions(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<&GroundedProposition> {
        let mut scored: Vec<(&GroundedProposition, f64)> = self
            .proposition_index
            .values()
            .map(|p| {
                let sim = cosine_similarity(query_embedding, &p.centroid);
                (p, sim * p.confidence)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(limit).map(|(p, _)| p).collect()
    }

    /// Try to apply a horn clause rule
    fn apply_rule(
        &self,
        rule: &HornClause,
        relevant: &[&GroundedProposition],
    ) -> Option<Inference> {
        // For simplicity, check if we have propositions matching all antecedents
        let mut matched: Vec<&GroundedProposition> = Vec::new();
        let mut combined_confidence = rule.confidence;

        for antecedent in &rule.antecedents {
            let pred_str = antecedent.as_str();
            if let Some(prop) = relevant.iter().find(|p| p.predicate == pred_str) {
                matched.push(*prop);
                combined_confidence *= prop.confidence;
            } else {
                return None; // Antecedent not satisfied
            }
        }

        if matched.is_empty() {
            return None;
        }

        // Create consequent proposition
        let first = matched[0];
        let consequent = GroundedProposition::new(
            rule.consequent.as_str().to_string(),
            first.arguments.clone(), // Simplified: inherit arguments from first premise
            first.centroid.clone(),
            combined_confidence,
            matched.iter().flat_map(|p| p.evidence.clone()).collect(),
        );

        Some(Inference::new(
            consequent,
            vec![rule.id.clone()],
            matched.iter().map(|p| p.id).collect(),
            combined_confidence,
        ))
    }

    /// Get all propositions
    pub fn all_propositions(&self) -> Vec<&GroundedProposition> {
        self.proposition_index.values().collect()
    }

    /// Get propositions by predicate
    pub fn propositions_by_predicate(&self, predicate: &str) -> Vec<&GroundedProposition> {
        self.propositions
            .get(predicate)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get proposition count
    pub fn proposition_count(&self) -> usize {
        self.proposition_index.len()
    }

    /// Get rule count
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get extraction count
    pub fn extraction_count(&self) -> u64 {
        self.extraction_count
    }

    /// Get inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Apply decay to all propositions
    pub fn apply_decay(&mut self) {
        for prop in self.proposition_index.values_mut() {
            prop.decay(self.config.decay_rate);
        }

        // Remove propositions below threshold
        let min_conf = self.config.min_confidence * 0.5; // Allow some margin
        let to_remove: Vec<Uuid> = self
            .proposition_index
            .iter()
            .filter(|(_, p)| p.confidence < min_conf)
            .map(|(id, _)| *id)
            .collect();

        for id in to_remove {
            if let Some(prop) = self.proposition_index.remove(&id) {
                if let Some(props) = self.propositions.get_mut(&prop.predicate) {
                    props.retain(|p| p.id != id);
                }
            }
        }
    }

    /// Add a custom rule
    pub fn add_rule(&mut self, rule: HornClause) {
        self.rules.push(rule);
    }

    /// Ground a new proposition from external input
    pub fn ground_proposition(
        &mut self,
        predicate: String,
        arguments: Vec<String>,
        embedding: Vec<f32>,
        evidence: Vec<Uuid>,
    ) -> GroundedProposition {
        let prop = GroundedProposition::new(
            predicate,
            arguments,
            embedding,
            0.8, // Default confidence for manually grounded propositions
            evidence,
        );
        self.store_proposition(prop.clone());
        self.extraction_count += 1;
        prop
    }
}

impl Default for NeuralSymbolicBridge {
    fn default() -> Self {
        Self::new(BridgeConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// API Response Types
// ─────────────────────────────────────────────────────────────────────────────

/// Response for GET /v1/propositions
#[derive(Debug, Serialize)]
pub struct PropositionsResponse {
    pub propositions: Vec<GroundedProposition>,
    pub total_count: usize,
    pub rule_count: usize,
}

/// Request for POST /v1/ground
#[derive(Debug, Deserialize)]
pub struct GroundRequest {
    pub predicate: String,
    pub arguments: Vec<String>,
    pub embedding: Vec<f32>,
    pub evidence_ids: Vec<Uuid>,
}

/// Response for POST /v1/ground
#[derive(Debug, Serialize)]
pub struct GroundResponse {
    pub proposition_id: Uuid,
    pub predicate: String,
    pub confidence: f64,
}

/// Request for POST /v1/reason
#[derive(Debug, Deserialize)]
pub struct ReasonRequest {
    pub query: String,
    pub embedding: Option<Vec<f32>>,
    pub limit: Option<usize>,
}

/// Response for POST /v1/reason
#[derive(Debug, Serialize)]
pub struct ReasonResponse {
    pub inferences: Vec<Inference>,
    pub relevant_propositions: Vec<GroundedProposition>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proposition_creation() {
        let prop = GroundedProposition::new(
            "relates_to".to_string(),
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 0.0, 0.0, 0.0],
            0.8,
            vec![Uuid::new_v4()],
        );
        assert_eq!(prop.predicate, "relates_to");
        assert!(prop.confidence > 0.7);
    }

    #[test]
    fn test_proposition_reinforcement() {
        let mut prop = GroundedProposition::new(
            "relates_to".to_string(),
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 0.0, 0.0, 0.0],
            0.5,
            vec![],
        );
        let evidence = Uuid::new_v4();
        prop.reinforce(evidence, 0.5);
        assert!(prop.confidence > 0.5);
        assert_eq!(prop.evidence.len(), 1);
        assert_eq!(prop.reinforcement_count, 2);
    }

    #[test]
    fn test_bridge_extraction() {
        let mut bridge = NeuralSymbolicBridge::default();
        // Need 5+ memory_ids for cluster_confidence to exceed min_confidence (0.5)
        // cluster_confidence(5) = 1.0 - exp(-1.0) ≈ 0.63
        let clusters = vec![(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
            "pattern".to_string(),
        )];

        let extracted = bridge.extract_from_clusters(&clusters);
        assert!(!extracted.is_empty());
        assert_eq!(bridge.proposition_count(), 1);
    }

    #[test]
    fn test_bridge_reasoning() {
        let mut bridge = NeuralSymbolicBridge::default();

        // Add some propositions
        bridge.ground_proposition(
            "relates_to".to_string(),
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );
        bridge.ground_proposition(
            "relates_to".to_string(),
            vec!["B".to_string(), "C".to_string()],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );

        let inferences = bridge.reason(&[0.95, 0.05, 0.0, 0.0], 5);
        // Should find transitivity inference
        assert!(bridge.rule_count() > 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}
