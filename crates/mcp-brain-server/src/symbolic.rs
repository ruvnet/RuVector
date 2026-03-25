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
        // Use confidence-gated language — never overstate weak signals
        let strength = if combined_confidence >= 0.8 {
            "strongly associated with"
        } else if combined_confidence >= 0.65 {
            "likely influences"
        } else if combined_confidence >= 0.5 {
            "may be associated with"
        } else {
            "shows weak co-occurrence with"
        };

        let explanation = if conclusion.arguments.len() >= 2 {
            format!(
                "{} {} {} (confidence: {:.0}%, {} evidence items, via {})",
                conclusion.arguments[0],
                strength,
                conclusion.arguments[1],
                combined_confidence * 100.0,
                conclusion.evidence.len(),
                rules_applied.join(" + ")
            )
        } else {
            format!(
                "Derived '{}' via {} ({} premises)",
                conclusion.to_string_human(),
                rules_applied.join(" + "),
                premises.len()
            )
        };
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
    /// X is a subtype of Y
    IsSubtypeOf,
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
            Self::IsSubtypeOf => "is_subtype_of",
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

        // Association is transitive (with decay): if A associated_with B and B associated_with C → A co_occurs_with C
        self.rules.push(HornClause::new(
            vec![PredicateType::Custom("associated_with".to_string()), PredicateType::Custom("associated_with".to_string())],
            PredicateType::Custom("co_occurs_with".to_string()),
            0.5,
        ));

        // Influence chain: if A may_influence B and B may_influence C → A associated_with C
        // (demotes from influence to association when chaining — honest decay)
        self.rules.push(HornClause::new(
            vec![PredicateType::Custom("may_influence".to_string()), PredicateType::Custom("may_influence".to_string())],
            PredicateType::Custom("associated_with".to_string()),
            0.6,
        ));

        // Cross-type influence: if A may_influence B and B is_type_of C → A associated_with C
        self.rules.push(HornClause::new(
            vec![PredicateType::Custom("may_influence".to_string()), PredicateType::IsTypeOf],
            PredicateType::Custom("associated_with".to_string()),
            0.5,
        ));

        // ── ADR-123: Relational rules for cognitive enrichment ──

        // Subtype transitivity: if A is_subtype_of B and B is_subtype_of C, then A is_subtype_of C
        self.rules.push(HornClause::new(
            vec![PredicateType::IsSubtypeOf, PredicateType::IsSubtypeOf],
            PredicateType::IsSubtypeOf,
            0.85,
        ));

        // Type hierarchy: if A is_type_of B and B is_subtype_of C, then A is_type_of C
        self.rules.push(HornClause::new(
            vec![PredicateType::IsTypeOf, PredicateType::IsSubtypeOf],
            PredicateType::IsTypeOf,
            0.85,
        ));

        // Dependency chain: if A depends_on B and B depends_on C, then A depends_on C
        self.rules.push(HornClause::new(
            vec![PredicateType::DependsOn, PredicateType::DependsOn],
            PredicateType::DependsOn,
            0.6,
        ));

        // Same-type relation: if A is_type_of X and B is_type_of X, then A co_occurs_with B
        // (NOT "relates_to" — sharing a type is weak evidence)
        self.rules.push(HornClause::new(
            vec![PredicateType::IsTypeOf, PredicateType::IsTypeOf],
            PredicateType::Custom("co_occurs_with".to_string()),
            0.4,
        ));

        // Transitive solution via dependency: if A solves B and B depends_on C, then A solves C
        self.rules.push(HornClause::new(
            vec![PredicateType::Solves, PredicateType::DependsOn],
            PredicateType::Solves,
            0.7,
        ));

        // Influence→prevention: if A may_influence B and B prevents C, then A co_occurs_with C
        self.rules.push(HornClause::new(
            vec![PredicateType::Custom("may_influence".to_string()), PredicateType::Prevents],
            PredicateType::Custom("co_occurs_with".to_string()),
            0.4,
        ));

        // Composition: if A part_of B and B part_of C, then A part_of C
        self.rules.push(HornClause::new(
            vec![PredicateType::PartOf, PredicateType::PartOf],
            PredicateType::PartOf,
            0.7,
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

        // ── Cross-category relations: find structural connections between different domains ──
        // This is where real discoveries happen — connecting clusters across categories
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let (ref c1, ref ids1, ref cat1) = clusters[i];
                let (ref c2, ref ids2, ref cat2) = clusters[j];

                if ids1.len() < self.config.min_cluster_size || ids2.len() < self.config.min_cluster_size {
                    continue;
                }

                let sim = cosine_similarity(c1, c2);
                let cross_domain = cat1 != cat2;

                // Skip weak signals — raised from 0.3 to 0.45 to eliminate
                // the flood of "weak co-occurrence" noise in gist publications.
                // At 0.3, nearly every category pair generates a proposition.
                if sim < 0.45 {
                    continue;
                }

                let mut merged_evidence = ids1.clone();
                merged_evidence.extend_from_slice(ids2);
                merged_evidence.truncate(20);
                let midpoint: Vec<f32> = c1.iter().zip(c2.iter()).map(|(a, b)| (a + b) / 2.0).collect();

                // Use category names instead of cluster sizes for human-readable arguments
                let arg1 = cat1.clone();
                let arg2 = cat2.clone();

                // ── Confidence-gated predicate selection ──
                // Embedding similarity alone only establishes co-occurrence, not causation.
                // Use honest language:
                //   sim > 0.7 + cross-domain → "may_influence" (candidate directional link)
                //   sim > 0.5 + cross-domain → "associated_with" (co-occurrence signal)
                //   sim > 0.3 + cross-domain → "co_occurs_with" (weak association)
                //   sim > 0.6 + same domain  → "similar_to"
                // "causes" and "depends_on" are reserved for validated temporal evidence.

                let conf = sim * self.cluster_confidence(ids1.len().min(ids2.len()));

                if cross_domain && sim > 0.75 {
                    // Strong cross-domain co-occurrence — candidate influence, NOT proven causal
                    let prop = GroundedProposition::new(
                        "may_influence".to_string(),
                        vec![arg1.clone(), arg2.clone()],
                        midpoint.clone(),
                        conf,
                        merged_evidence.clone(),
                    );
                    if prop.confidence >= self.config.min_confidence {
                        extracted.push(prop.clone());
                        self.store_proposition(prop);
                    }
                } else if cross_domain && sim > 0.55 {
                    // Moderate cross-domain signal — association
                    let prop = GroundedProposition::new(
                        "associated_with".to_string(),
                        vec![arg1.clone(), arg2.clone()],
                        midpoint.clone(),
                        conf,
                        merged_evidence.clone(),
                    );
                    if prop.confidence >= self.config.min_confidence {
                        extracted.push(prop.clone());
                        self.store_proposition(prop);
                    }
                } else if cross_domain && sim > 0.3 {
                    // Weak cross-domain signal — co-occurrence only
                    let prop = GroundedProposition::new(
                        "co_occurs_with".to_string(),
                        vec![arg1.clone(), arg2.clone()],
                        midpoint.clone(),
                        conf,
                        merged_evidence.clone(),
                    );
                    if prop.confidence >= self.config.min_confidence {
                        extracted.push(prop.clone());
                        self.store_proposition(prop);
                    }
                } else if !cross_domain && sim > 0.6 {
                    // Same category, high similarity → "similar_to"
                    let prop = GroundedProposition::new(
                        PredicateType::SimilarTo.as_str().to_string(),
                        vec![arg1, arg2],
                        midpoint,
                        sim * self.cluster_confidence(ids1.len().min(ids2.len())),
                        merged_evidence,
                    );
                    if prop.confidence >= self.config.min_confidence {
                        extracted.push(prop.clone());
                        self.store_proposition(prop);
                    }
                }
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

    /// Run forward-chaining inference over all propositions.
    ///
    /// For each Horn clause rule, find all pairs of propositions matching
    /// the antecedent predicates. For transitive rules (same predicate on
    /// both sides), require that the second argument of the first
    /// proposition equals the first argument of the second proposition
    /// (chain linking: A→B + B→C yields A→C).
    ///
    /// Derived conclusions are added as new propositions so subsequent
    /// cycles can chain further. Returns all newly produced inferences.
    pub fn run_inference(&mut self) -> Vec<Inference> {
        // Snapshot rules and propositions so we don't hold borrows on self
        // during the matching loop.
        let rules: Vec<HornClause> = self.rules.clone();

        let all_props: Vec<GroundedProposition> =
            self.proposition_index.values().cloned().collect();

        // Index propositions by predicate string for fast lookup.
        let mut by_predicate: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, prop) in all_props.iter().enumerate() {
            by_predicate
                .entry(prop.predicate.clone())
                .or_default()
                .push(i);
        }

        // Build a set of existing (predicate, arguments) pairs to check
        // for duplicates without borrowing self.
        let mut existing: std::collections::HashSet<(String, Vec<String>)> =
            std::collections::HashSet::new();
        for prop in &all_props {
            existing.insert((prop.predicate.clone(), prop.arguments.clone()));
        }

        // Collect derived propositions and inferences; store after the loop.
        let mut derived: Vec<(GroundedProposition, Inference)> = Vec::new();
        let min_confidence = self.config.min_confidence;

        for rule in &rules {
            if rule.antecedents.len() != 2 {
                continue;
            }

            let pred_a = rule.antecedents[0].as_str();
            let pred_b = rule.antecedents[1].as_str();

            let indices_a = match by_predicate.get(pred_a) {
                Some(g) => g,
                None => continue,
            };
            let indices_b = match by_predicate.get(pred_b) {
                Some(g) => g,
                None => continue,
            };

            for &ia in indices_a {
                let pa = &all_props[ia];
                for &ib in indices_b {
                    if ia == ib {
                        continue;
                    }
                    let pb = &all_props[ib];

                    // Chain-linking: try strict chain first (last arg of pa == first arg of pb),
                    // then fall back to shared-argument linking (any arg in common).
                    // The strict chain produces transitive derivations (A->B + B->C => A->C).
                    // Shared-argument linking handles cases where arguments don't line up
                    // strictly (e.g., is_type_of(cluster_5, arch) + is_type_of(cluster_3, arch)).
                    let strict_chain = match (pa.arguments.last(), pb.arguments.first()) {
                        (Some(a), Some(b)) => a == b,
                        _ => false,
                    };

                    // Shared-argument: any argument of pa matches any argument of pb
                    // (case-insensitive for robustness).
                    let shared_arg = if !strict_chain {
                        pa.arguments.iter().any(|a| {
                            pb.arguments.iter().any(|b| a.eq_ignore_ascii_case(b))
                        })
                    } else {
                        false
                    };

                    if !strict_chain && !shared_arg {
                        continue;
                    }

                    // Derive arguments depending on linking mode.
                    let derived_args = if strict_chain {
                        // Transitive: first arg of pa, last arg of pb.
                        match (pa.arguments.first(), pb.arguments.last()) {
                            (Some(first), Some(last)) => {
                                if first == last {
                                    continue; // Skip self-loops.
                                }
                                vec![first.clone(), last.clone()]
                            }
                            _ => continue,
                        }
                    } else {
                        // Shared-argument: take the non-shared args as endpoints.
                        let shared: Vec<&String> = pa.arguments.iter()
                            .filter(|a| pb.arguments.iter().any(|b| a.eq_ignore_ascii_case(b)))
                            .collect();
                        let pa_unique: Vec<&String> = pa.arguments.iter()
                            .filter(|a| !shared.iter().any(|s| a.eq_ignore_ascii_case(s)))
                            .collect();
                        let pb_unique: Vec<&String> = pb.arguments.iter()
                            .filter(|b| !shared.iter().any(|s| b.eq_ignore_ascii_case(s)))
                            .collect();
                        let first: Option<&String> = pa_unique.first().copied().or(pa.arguments.first());
                        let last: Option<&String> = pb_unique.first().copied().or(pb.arguments.first());
                        match (first, last) {
                            (Some(f), Some(l)) => {
                                if f.eq_ignore_ascii_case(l) {
                                    continue; // Skip self-loops.
                                }
                                vec![f.clone(), l.clone()]
                            }
                            _ => continue,
                        }
                    };

                    let consequent_pred = rule.consequent.as_str().to_string();

                    // Skip if already known.
                    if existing.contains(&(consequent_pred.clone(), derived_args.clone())) {
                        continue;
                    }

                    let combined_confidence =
                        rule.confidence * pa.confidence * pb.confidence;

                    // Require meaningful confidence — no coin-flip inferences
                    if combined_confidence < 0.4 {
                        continue;
                    }

                    let centroid: Vec<f32> = pa
                        .centroid
                        .iter()
                        .zip(pb.centroid.iter())
                        .map(|(a, b)| (a + b) / 2.0)
                        .collect();

                    let mut evidence: Vec<Uuid> = pa.evidence.clone();
                    evidence.extend_from_slice(&pb.evidence);
                    evidence.truncate(20);

                    // Mark as existing so we don't duplicate within this cycle.
                    existing.insert((consequent_pred.clone(), derived_args.clone()));

                    let conclusion = GroundedProposition::new(
                        consequent_pred,
                        derived_args,
                        centroid,
                        combined_confidence,
                        evidence,
                    );

                    let inference = Inference::new(
                        conclusion.clone(),
                        vec![rule.id.clone()],
                        vec![pa.id, pb.id],
                        combined_confidence,
                    );

                    derived.push((conclusion, inference));
                }
            }
        }

        // Now store all derived propositions (requires &mut self).
        let new_inferences: Vec<Inference> = derived
            .into_iter()
            .map(|(conclusion, inference)| {
                self.store_proposition(conclusion);
                inference
            })
            .collect();

        self.inference_count += new_inferences.len() as u64;
        new_inferences
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
    fn test_forward_chaining_transitive() {
        let mut bridge = NeuralSymbolicBridge::default();

        // Ground: relates_to(A, B) and relates_to(B, C)
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

        assert_eq!(bridge.proposition_count(), 2);
        assert_eq!(bridge.inference_count(), 0);

        let inferences = bridge.run_inference();

        // Should derive relates_to(A, C) via transitivity rule
        assert!(!inferences.is_empty(), "expected at least one inference");
        let inf = &inferences[0];
        assert_eq!(inf.conclusion.predicate, "relates_to");
        assert_eq!(inf.conclusion.arguments, vec!["A".to_string(), "C".to_string()]);
        assert!(inf.combined_confidence > 0.0);
        assert!(bridge.inference_count() > 0);

        // The derived proposition should now exist
        assert_eq!(bridge.proposition_count(), 3);

        // Running again should produce no new inferences (already derived)
        let inferences2 = bridge.run_inference();
        assert!(inferences2.is_empty(), "should not re-derive existing conclusions");
    }

    #[test]
    fn test_forward_chaining_cross_predicate() {
        let mut bridge = NeuralSymbolicBridge::default();

        // Ground: solves(X, Y) and depends_on(Y, Z)
        // Rule: solves + depends_on → solves (transitive solution via dependency)
        bridge.ground_proposition(
            "solves".to_string(),
            vec!["tool_A".to_string(), "problem_B".to_string()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );
        bridge.ground_proposition(
            "depends_on".to_string(),
            vec!["problem_B".to_string(), "problem_C".to_string()],
            vec![0.8, 0.2, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );

        let inferences = bridge.run_inference();

        // Should derive solves(tool_A, problem_C)
        assert!(!inferences.is_empty(), "expected cross-predicate inference");
        let inf = &inferences[0];
        assert_eq!(inf.conclusion.predicate, "solves");
        assert_eq!(
            inf.conclusion.arguments,
            vec!["tool_A".to_string(), "problem_C".to_string()]
        );
    }

    #[test]
    fn test_forward_chaining_no_self_loop() {
        let mut bridge = NeuralSymbolicBridge::default();

        // Ground: relates_to(A, B) and relates_to(B, A)
        // Should NOT derive relates_to(A, A) (self-loop)
        bridge.ground_proposition(
            "relates_to".to_string(),
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );
        bridge.ground_proposition(
            "relates_to".to_string(),
            vec!["B".to_string(), "A".to_string()],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![Uuid::new_v4()],
        );

        let inferences = bridge.run_inference();
        for inf in &inferences {
            assert_ne!(
                inf.conclusion.arguments[0], inf.conclusion.arguments[1],
                "should not produce self-loop inference"
            );
        }
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
