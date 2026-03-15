//! Internal Voice System (ADR-110)
//!
//! Provides continuous self-narration, working memory, and goal-directed deliberation.
//! The internal voice bridges neural patterns and symbolic reasoning with transparent
//! meta-cognitive processing.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Voice Token Types
// ─────────────────────────────────────────────────────────────────────────────

/// Types of internal thoughts (reasoning transparency)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThoughtType {
    /// "I notice that..." - observational thoughts from perception
    Observation,
    /// "I wonder if..." - inquiry-driven thoughts
    Question,
    /// "Perhaps..." - hypothesis formation
    Hypothesis,
    /// "Therefore..." - logical conclusions
    Conclusion,
    /// "I should..." - goal-directed intentions
    Goal,
    /// "Looking back..." - retrospective analysis
    Reflection,
    /// "I'm not sure..." - epistemic uncertainty
    Uncertainty,
    /// "But on the other hand..." - conflicting evidence
    Conflict,
    /// "I remember..." - memory retrieval
    Recall,
    /// "This is similar to..." - pattern recognition
    Pattern,
}

/// Source of a thought (provenance tracking)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ThoughtSource {
    /// From memory retrieval
    Perception { memory_id: Uuid },
    /// From symbolic inference
    Reasoning { rule_id: String },
    /// From Strange Loop meta-cognition
    MetaCognition,
    /// From goal-directed planner
    GoalDirected { goal: String },
    /// From pattern matching in SONA
    PatternMatch { pattern_id: String },
    /// From external input (user query)
    External,
}

/// A single internal monologue token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceToken {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub thought_type: ThoughtType,
    pub content: String,
    /// Attention weight (0.0-1.0) - decays over time
    pub attention_weight: f64,
    pub source: ThoughtSource,
    /// Optional embedding for semantic search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl VoiceToken {
    pub fn new(thought_type: ThoughtType, content: String, source: ThoughtSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            thought_type,
            content,
            attention_weight: 1.0,
            source,
            embedding: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Apply attention decay based on age
    pub fn apply_decay(&mut self, decay_rate: f64) {
        let age_secs = (Utc::now() - self.timestamp).num_seconds() as f64;
        self.attention_weight *= (-decay_rate * age_secs).exp();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Working Memory
// ─────────────────────────────────────────────────────────────────────────────

/// Content source for working memory items
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentSource {
    /// From memory retrieval
    Perception,
    /// From reasoning/inference
    Reasoning,
    /// From learning/training
    Learning,
    /// From user input
    External,
}

/// Working memory item with activation decay (Miller's Law: 7±2 items)
#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryItem {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub activation: f64,
    pub last_accessed: DateTime<Utc>,
    pub source: ContentSource,
}

impl WorkingMemoryItem {
    pub fn new(content: String, embedding: Vec<f32>, source: ContentSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            activation: 1.0,
            last_accessed: Utc::now(),
            source,
        }
    }

    /// Apply activation decay based on time since last access
    pub fn apply_decay(&mut self, decay_rate: f64) {
        let age_secs = (Utc::now() - self.last_accessed).num_seconds() as f64;
        self.activation *= (-decay_rate * age_secs).exp();
    }

    /// Boost activation when item is accessed
    pub fn boost(&mut self, amount: f64) {
        self.activation = (self.activation + amount).min(1.0);
        self.last_accessed = Utc::now();
    }
}

/// Working memory buffer with capacity management and attention
pub struct WorkingMemory {
    items: Vec<WorkingMemoryItem>,
    /// Capacity (default: 7, range: 5-9 per Miller's Law)
    capacity: usize,
    /// Decay rate (per second)
    decay_rate: f64,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::new(),
            capacity: capacity.clamp(5, 9),
            decay_rate: 0.01, // ~1% decay per second
        }
    }

    /// Add item with automatic capacity management
    pub fn add(&mut self, content: String, embedding: Vec<f32>, source: ContentSource) {
        // Apply decay to existing items
        self.apply_decay();

        // If at capacity, remove lowest activation item
        if self.items.len() >= self.capacity {
            self.evict_lowest();
        }

        self.items.push(WorkingMemoryItem::new(content, embedding, source));
    }

    /// Retrieve items similar to query embedding
    pub fn retrieve(&mut self, query: &[f32], limit: usize) -> Vec<&WorkingMemoryItem> {
        self.apply_decay();

        // Compute similarity scores
        let mut scored: Vec<(usize, f64)> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let sim = cosine_similarity(query, &item.embedding);
                (i, sim * item.activation) // Weight by activation
            })
            .collect();

        // Sort by combined score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Boost retrieved items
        for (idx, _) in scored.iter().take(limit) {
            self.items[*idx].boost(0.2);
        }

        scored
            .into_iter()
            .take(limit)
            .map(|(i, _)| &self.items[i])
            .collect()
    }

    /// Apply decay to all items
    fn apply_decay(&mut self) {
        for item in &mut self.items {
            item.apply_decay(self.decay_rate);
        }
    }

    /// Evict item with lowest activation
    fn evict_lowest(&mut self) {
        if let Some((min_idx, _)) = self
            .items
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.activation
                    .partial_cmp(&b.activation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            self.items.remove(min_idx);
        }
    }

    /// Get current utilization (0.0-1.0)
    pub fn utilization(&self) -> f64 {
        self.items.len() as f64 / self.capacity as f64
    }

    /// Get all items (for serialization)
    pub fn items(&self) -> &[WorkingMemoryItem] {
        &self.items
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new(7)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Goal Stack
// ─────────────────────────────────────────────────────────────────────────────

/// A goal frame for deliberation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalFrame {
    pub id: Uuid,
    pub description: String,
    pub priority: f64,
    pub created_at: DateTime<Utc>,
    pub subgoals: Vec<GoalFrame>,
    pub status: GoalStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    Active,
    Completed,
    Failed,
    Suspended,
}

impl GoalFrame {
    pub fn new(description: String, priority: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            priority,
            created_at: Utc::now(),
            subgoals: Vec::new(),
            status: GoalStatus::Active,
        }
    }

    pub fn add_subgoal(&mut self, subgoal: GoalFrame) {
        self.subgoals.push(subgoal);
    }

    pub fn is_active(&self) -> bool {
        self.status == GoalStatus::Active
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal Voice Engine
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the internal voice system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Working memory capacity (default: 7)
    pub working_memory_size: usize,
    /// Voice buffer capacity (max thoughts to retain)
    pub voice_buffer_size: usize,
    /// Verbosity level (0.0 = silent, 1.0 = verbose)
    pub verbosity: f64,
    /// Enable meta-cognitive reflection
    pub enable_reflection: bool,
    /// Maximum deliberation depth
    pub max_deliberation_depth: usize,
    /// Thought decay rate (per second)
    pub thought_decay_rate: f64,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 7,
            voice_buffer_size: 50,
            verbosity: 0.5,
            enable_reflection: true,
            max_deliberation_depth: 3,
            thought_decay_rate: 0.005,
        }
    }
}

/// Internal voice engine for self-narration and deliberation
pub struct InternalVoice {
    /// Voice buffer (recent thoughts)
    thoughts: VecDeque<VoiceToken>,
    /// Working memory buffer
    working_memory: WorkingMemory,
    /// Current goal stack
    goals: Vec<GoalFrame>,
    /// Configuration
    config: VoiceConfig,
    /// Total thoughts generated
    thought_count: u64,
}

impl InternalVoice {
    pub fn new(config: VoiceConfig) -> Self {
        Self {
            thoughts: VecDeque::new(),
            working_memory: WorkingMemory::new(config.working_memory_size),
            goals: Vec::new(),
            config,
            thought_count: 0,
        }
    }

    /// Push a new goal frame
    pub fn set_goal(&mut self, description: String, priority: f64) -> Uuid {
        let goal = GoalFrame::new(description.clone(), priority);
        let goal_id = goal.id;
        self.goals.push(goal);
        self.emit(
            ThoughtType::Goal,
            format!("I should {}", description),
            ThoughtSource::GoalDirected {
                goal: description,
            },
        );
        goal_id
    }

    /// Complete the current goal
    pub fn complete_goal(&mut self) -> Option<GoalFrame> {
        if let Some(mut goal) = self.goals.pop() {
            goal.status = GoalStatus::Completed;
            self.emit(
                ThoughtType::Conclusion,
                format!("Goal completed: {}", goal.description),
                ThoughtSource::MetaCognition,
            );
            Some(goal)
        } else {
            None
        }
    }

    /// Get the current active goal
    pub fn current_goal(&self) -> Option<&GoalFrame> {
        self.goals.last().filter(|g| g.is_active())
    }

    /// Emit an observation thought
    pub fn observe(&mut self, content: String, memory_id: Uuid) -> Uuid {
        self.emit(
            ThoughtType::Observation,
            format!("I notice that {}", content),
            ThoughtSource::Perception { memory_id },
        )
    }

    /// Emit a question thought
    pub fn question(&mut self, content: String) -> Uuid {
        self.emit(
            ThoughtType::Question,
            format!("I wonder {}", content),
            ThoughtSource::MetaCognition,
        )
    }

    /// Emit a hypothesis thought
    pub fn hypothesize(&mut self, content: String) -> Uuid {
        self.emit(
            ThoughtType::Hypothesis,
            format!("Perhaps {}", content),
            ThoughtSource::MetaCognition,
        )
    }

    /// Emit a conclusion thought
    pub fn conclude(&mut self, content: String, rule_id: String) -> Uuid {
        self.emit(
            ThoughtType::Conclusion,
            format!("Therefore, {}", content),
            ThoughtSource::Reasoning { rule_id },
        )
    }

    /// Emit an uncertainty thought
    pub fn express_uncertainty(&mut self, content: String) -> Uuid {
        self.emit(
            ThoughtType::Uncertainty,
            format!("I'm not sure about {}", content),
            ThoughtSource::MetaCognition,
        )
    }

    /// Emit a conflict thought
    pub fn note_conflict(&mut self, content: String) -> Uuid {
        self.emit(
            ThoughtType::Conflict,
            format!("But on the other hand, {}", content),
            ThoughtSource::MetaCognition,
        )
    }

    /// Emit a pattern recognition thought
    pub fn recognize_pattern(&mut self, content: String, pattern_id: String) -> Uuid {
        self.emit(
            ThoughtType::Pattern,
            format!("This is similar to {}", content),
            ThoughtSource::PatternMatch { pattern_id },
        )
    }

    /// Emit a reflection thought
    pub fn reflect(&mut self, content: String) -> Uuid {
        if self.config.enable_reflection {
            self.emit(
                ThoughtType::Reflection,
                format!("Looking back, {}", content),
                ThoughtSource::MetaCognition,
            )
        } else {
            Uuid::nil()
        }
    }

    /// Reflect on a learning result
    pub fn reflect_on_learning(&mut self, sona_result: &str) -> Vec<VoiceToken> {
        if !self.config.enable_reflection {
            return Vec::new();
        }

        let mut reflections = Vec::new();

        // Emit a reflection about the learning
        let _thought_id = self.emit(
            ThoughtType::Reflection,
            format!("Learning cycle completed: {}", sona_result),
            ThoughtSource::MetaCognition,
        );

        // Clone recent thoughts for return
        for thought in self.thoughts.iter().rev().take(5) {
            reflections.push(thought.clone());
        }

        reflections
    }

    /// Core emit function
    fn emit(&mut self, thought_type: ThoughtType, content: String, source: ThoughtSource) -> Uuid {
        let token = VoiceToken::new(thought_type, content, source);
        let id = token.id;

        self.thoughts.push_back(token);
        self.thought_count += 1;

        // Trim to buffer size
        while self.thoughts.len() > self.config.voice_buffer_size {
            self.thoughts.pop_front();
        }

        id
    }

    /// Add to working memory
    pub fn remember(&mut self, content: String, embedding: Vec<f32>, source: ContentSource) {
        self.working_memory.add(content, embedding, source);
    }

    /// Retrieve from working memory
    pub fn recall(&mut self, query: &[f32], limit: usize) -> Vec<&WorkingMemoryItem> {
        self.working_memory.retrieve(query, limit)
    }

    /// Get recent thoughts
    pub fn recent_thoughts(&self, limit: usize) -> Vec<&VoiceToken> {
        self.thoughts.iter().rev().take(limit).collect()
    }

    /// Get thoughts by type
    pub fn thoughts_by_type(&self, thought_type: ThoughtType) -> Vec<&VoiceToken> {
        self.thoughts
            .iter()
            .filter(|t| t.thought_type == thought_type)
            .collect()
    }

    /// Get working memory utilization
    pub fn working_memory_utilization(&self) -> f64 {
        self.working_memory.utilization()
    }

    /// Get total thought count
    pub fn thought_count(&self) -> u64 {
        self.thought_count
    }

    /// Get goal stack depth
    pub fn goal_depth(&self) -> usize {
        self.goals.len()
    }

    /// Get all active goals
    pub fn active_goals(&self) -> Vec<&GoalFrame> {
        self.goals.iter().filter(|g| g.is_active()).collect()
    }

    /// Get working memory items
    pub fn working_memory_items(&self) -> &[WorkingMemoryItem] {
        self.working_memory.items()
    }

    /// Clear working memory
    pub fn clear_working_memory(&mut self) {
        self.working_memory.clear();
    }

    /// Apply decay to all thoughts
    pub fn apply_decay(&mut self) {
        for thought in &mut self.thoughts {
            thought.apply_decay(self.config.thought_decay_rate);
        }
    }
}

impl Default for InternalVoice {
    fn default() -> Self {
        Self::new(VoiceConfig::default())
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

/// Response for GET /v1/voice/working
#[derive(Debug, Serialize)]
pub struct WorkingMemoryResponse {
    pub items: Vec<WorkingMemoryItemSummary>,
    pub utilization: f64,
    pub capacity: usize,
}

#[derive(Debug, Serialize)]
pub struct WorkingMemoryItemSummary {
    pub id: Uuid,
    pub content: String,
    pub activation: f64,
    pub source: ContentSource,
    pub last_accessed: DateTime<Utc>,
}

/// Response for GET /v1/voice/history
#[derive(Debug, Serialize)]
pub struct VoiceHistoryResponse {
    pub thoughts: Vec<VoiceToken>,
    pub total_count: u64,
    pub goal_depth: usize,
}

/// Request for POST /v1/voice/goal
#[derive(Debug, Deserialize)]
pub struct SetGoalRequest {
    pub description: String,
    pub priority: Option<f64>,
}

/// Response for POST /v1/voice/goal
#[derive(Debug, Serialize)]
pub struct SetGoalResponse {
    pub goal_id: Uuid,
    pub description: String,
    pub priority: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_token_creation() {
        let token = VoiceToken::new(
            ThoughtType::Observation,
            "test observation".to_string(),
            ThoughtSource::External,
        );
        assert_eq!(token.thought_type, ThoughtType::Observation);
        assert!(token.attention_weight > 0.9);
    }

    #[test]
    fn test_working_memory_capacity() {
        // Note: capacity is clamped to 5-9 per Miller's Law (7±2)
        let mut wm = WorkingMemory::new(5);
        for i in 0..10 {
            wm.add(
                format!("item {}", i),
                vec![i as f32; 4],
                ContentSource::External,
            );
        }
        // Should only keep 5 items (Miller's Law minimum)
        assert!(wm.items.len() <= 5);
    }

    #[test]
    fn test_working_memory_retrieval() {
        let mut wm = WorkingMemory::new(5);
        wm.add("hello world".to_string(), vec![1.0, 0.0, 0.0, 0.0], ContentSource::External);
        wm.add("goodbye world".to_string(), vec![0.0, 1.0, 0.0, 0.0], ContentSource::External);

        let results = wm.retrieve(&[0.9, 0.1, 0.0, 0.0], 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_internal_voice_emit() {
        let mut voice = InternalVoice::default();
        let id = voice.observe("something interesting".to_string(), Uuid::new_v4());
        assert!(!id.is_nil());
        assert_eq!(voice.thought_count(), 1);
    }

    #[test]
    fn test_goal_management() {
        let mut voice = InternalVoice::default();
        let goal_id = voice.set_goal("understand the codebase".to_string(), 1.0);
        assert!(!goal_id.is_nil());
        assert_eq!(voice.goal_depth(), 1);

        let completed = voice.complete_goal();
        assert!(completed.is_some());
        assert_eq!(voice.goal_depth(), 0);
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
