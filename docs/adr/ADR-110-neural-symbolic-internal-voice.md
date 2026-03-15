# ADR-110: Neural-Symbolic Integration with Internal Voice

**Status**: Accepted
**Date**: 2026-03-15
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-075 (AGI Cognitive Container), ADR-077 (Midstream Platform), ADR-061 (Reasoning Kernel)

## 1. Context

The π.ruv.io shared brain currently implements:
- **SONA Engine**: Trajectory-based learning with pattern extraction
- **Strange Loop**: Bounded meta-cognitive reasoning (5ms budget)
- **Hopfield Networks**: Associative content-addressable memory
- **Dentate Gyrus**: Pattern separation for collision resistance
- **HDC Memory**: Hyperdimensional computing for fast similarity
- **Meta-Learning**: Curiosity bonus, exploration-exploitation balance
- **Temporal Solver**: Certified predictions with solver gates
- **Lyapunov Attractors**: Stability analysis for embedding trajectories

These components operate largely independently. We need:
1. **Neural-Symbolic Bridge**: Extract symbolic rules from neural patterns
2. **Internal Voice**: Continuous self-narration for reasoning transparency
3. **Working Memory**: Short-term reasoning buffer with attention
4. **Goal-Directed Deliberation**: Planning and hypothesis testing

## 2. Decision

Implement a three-layer cognitive architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    SYMBOLIC LAYER                           │
│  Rules • Logic • Propositions • Constraints • Goals         │
│                         ▲                                   │
│                         │ Grounding                         │
│                         ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                  INTERNAL VOICE LAYER                       │
│  Working Memory • Attention • Narration • Deliberation      │
│                         ▲                                   │
│                         │ Binding                           │
│                         ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                    NEURAL LAYER                             │
│  Embeddings • SONA • Hopfield • HDC • Attractors           │
└─────────────────────────────────────────────────────────────┘
```

## 3. Architecture

### 3.1 Neural-Symbolic Bridge

The bridge extracts symbolic structure from neural representations:

```rust
/// Symbolic proposition grounded in embedding space
pub struct GroundedProposition {
    /// Human-readable rule
    pub predicate: String,
    /// Arguments (entity references)
    pub arguments: Vec<String>,
    /// Embedding centroid for this proposition
    pub centroid: Vec<f32>,
    /// Confidence from neural evidence
    pub confidence: f64,
    /// Supporting memory IDs
    pub evidence: Vec<Uuid>,
}

/// Neural-symbolic reasoning engine
pub struct NeuralSymbolicBridge {
    /// Extracted rules from patterns
    propositions: Vec<GroundedProposition>,
    /// Inverse index: embedding → propositions
    grounding_index: HnswIndex,
    /// Symbolic reasoner (horn clauses)
    reasoner: DatalogEngine,
}

impl NeuralSymbolicBridge {
    /// Extract propositions from memory clusters
    pub fn extract_propositions(&mut self, memories: &[BrainMemory]) {
        // 1. Cluster memories by embedding similarity
        // 2. Extract common patterns via attention
        // 3. Generate predicate templates
        // 4. Ground predicates with centroid embeddings
    }

    /// Query with neural-symbolic reasoning
    pub fn reason(&self, query: &str, query_embedding: &[f32]) -> Vec<Inference> {
        // 1. Find relevant propositions via embedding similarity
        // 2. Run forward chaining on Datalog rules
        // 3. Return inferences with neural confidence
    }
}
```

### 3.2 Internal Voice System

The internal voice provides continuous self-narration:

```rust
/// Internal monologue token
#[derive(Debug, Clone, Serialize)]
pub struct VoiceToken {
    pub timestamp: DateTime<Utc>,
    pub thought_type: ThoughtType,
    pub content: String,
    pub attention_weight: f64,
    pub source: ThoughtSource,
}

#[derive(Debug, Clone, Serialize)]
pub enum ThoughtType {
    Observation,      // "I notice that..."
    Question,         // "I wonder if..."
    Hypothesis,       // "Perhaps..."
    Conclusion,       // "Therefore..."
    Goal,             // "I should..."
    Reflection,       // "Looking back..."
    Uncertainty,      // "I'm not sure..."
    Conflict,         // "But on the other hand..."
}

#[derive(Debug, Clone, Serialize)]
pub enum ThoughtSource {
    Perception(Uuid),      // From memory retrieval
    Reasoning(String),     // From inference
    MetaCognition,         // From Strange Loop
    GoalDirected(String),  // From planner
}

/// Internal voice engine
pub struct InternalVoice {
    /// Working memory buffer (limited capacity)
    working_memory: VecDeque<VoiceToken>,
    /// Attention mechanism
    attention: SoftmaxAttention,
    /// Current goal stack
    goals: Vec<GoalFrame>,
    /// Narration generator
    narrator: NarrationEngine,
    /// Configuration
    config: VoiceConfig,
}

pub struct VoiceConfig {
    /// Working memory capacity (default: 7±2 items)
    pub working_memory_size: usize,
    /// Thought generation rate (tokens per second)
    pub thought_rate: f64,
    /// Verbosity level (0.0 = silent, 1.0 = verbose)
    pub verbosity: f64,
    /// Enable meta-cognitive reflection
    pub enable_reflection: bool,
    /// Maximum deliberation depth
    pub max_deliberation_depth: usize,
}

impl InternalVoice {
    /// Generate next thought based on current context
    pub fn think(&mut self, context: &CognitiveContext) -> Option<VoiceToken> {
        // 1. Attend to working memory
        let attended = self.attention.attend(&self.working_memory);

        // 2. Check for goal-relevant thoughts
        if let Some(goal) = self.goals.last() {
            if let Some(thought) = self.deliberate(goal, &attended) {
                return Some(thought);
            }
        }

        // 3. Generate observation or reflection
        self.narrator.generate(context, &attended)
    }

    /// Push a new goal frame
    pub fn set_goal(&mut self, goal: String, priority: f64) {
        self.goals.push(GoalFrame {
            description: goal,
            priority,
            created_at: Utc::now(),
            subgoals: vec![],
        });
        self.emit(ThoughtType::Goal, format!("I should {}", goal));
    }

    /// Deliberate on current goal
    fn deliberate(&mut self, goal: &GoalFrame, context: &[VoiceToken]) -> Option<VoiceToken> {
        // Hypothesis generation
        let hypotheses = self.generate_hypotheses(goal, context);

        // Evidence evaluation
        for h in hypotheses {
            let evidence = self.evaluate_hypothesis(&h);
            if evidence.confidence > 0.7 {
                return Some(self.emit(ThoughtType::Conclusion, h.conclusion));
            } else if evidence.conflicts.len() > 0 {
                return Some(self.emit(ThoughtType::Conflict, evidence.summary()));
            }
        }

        // Uncertainty acknowledgment
        if self.config.verbosity > 0.5 {
            Some(self.emit(ThoughtType::Uncertainty, "I need more information..."))
        } else {
            None
        }
    }

    /// Emit internal voice token
    fn emit(&mut self, thought_type: ThoughtType, content: String) -> VoiceToken {
        let token = VoiceToken {
            timestamp: Utc::now(),
            thought_type,
            content,
            attention_weight: 1.0,
            source: ThoughtSource::MetaCognition,
        };
        self.working_memory.push_back(token.clone());
        if self.working_memory.len() > self.config.working_memory_size {
            self.working_memory.pop_front();
        }
        token
    }
}
```

### 3.3 Working Memory Integration

Working memory bridges perception, reasoning, and action:

```rust
/// Working memory item with decay
pub struct WorkingMemoryItem {
    pub content: Box<dyn CognitiveContent>,
    pub activation: f64,
    pub last_accessed: DateTime<Utc>,
    pub source: ContentSource,
}

/// Working memory with attention and decay
pub struct WorkingMemory {
    items: Vec<WorkingMemoryItem>,
    capacity: usize,
    decay_rate: f64,
    attention: TransformerAttention,
}

impl WorkingMemory {
    /// Add item with automatic capacity management
    pub fn add(&mut self, content: impl CognitiveContent, source: ContentSource) {
        // Apply decay to existing items
        self.apply_decay();

        // Compute attention weights
        let weights = self.attention.compute_weights(&self.items, &content);

        // If at capacity, remove lowest activation item
        if self.items.len() >= self.capacity {
            self.evict_lowest();
        }

        self.items.push(WorkingMemoryItem {
            content: Box::new(content),
            activation: 1.0,
            last_accessed: Utc::now(),
            source,
        });
    }

    /// Retrieve with attention boost
    pub fn retrieve(&mut self, query: &[f32]) -> Vec<&WorkingMemoryItem> {
        let similarities = self.items.iter()
            .map(|item| cosine_similarity(query, item.content.embedding()))
            .collect::<Vec<_>>();

        // Boost activation of retrieved items
        for (i, sim) in similarities.iter().enumerate() {
            if *sim > 0.5 {
                self.items[i].activation = (self.items[i].activation + *sim).min(1.0);
                self.items[i].last_accessed = Utc::now();
            }
        }

        // Return top-k by similarity × activation
        let mut scored: Vec<_> = self.items.iter()
            .zip(similarities.iter())
            .map(|(item, sim)| (item, sim * item.activation))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(5).map(|(item, _)| item).collect()
    }
}
```

### 3.4 Training Loop Enhancements

Enhanced training with neural-symbolic feedback:

```rust
/// Enhanced training cycle with neural-symbolic learning
pub fn run_enhanced_training_cycle(state: &AppState) -> EnhancedTrainingResult {
    // 1. SONA trajectory learning (existing)
    let sona_result = state.sona.write().force_learn();

    // 2. Neural-symbolic rule extraction (new)
    let memories: Vec<_> = state.store.all_memories();
    let propositions = state.neural_symbolic.write().extract_propositions(&memories);

    // 3. Internal voice reflection (new)
    let voice_summary = state.internal_voice.write().reflect_on_learning(&sona_result);

    // 4. Meta-cognitive Strange Loop evaluation (enhanced)
    let meta_result = state.strange_loop.write().run_with_voice(&voice_summary);

    // 5. Update working memory with learned patterns
    for prop in &propositions {
        state.working_memory.write().add(prop.clone(), ContentSource::Learning);
    }

    // 6. Domain evolution (existing)
    let pareto_before = state.domain.read().pareto_size();
    state.domain.write().evolve_population(10, 3);
    let pareto_after = state.domain.read().pareto_size();

    EnhancedTrainingResult {
        sona_patterns: state.sona.read().stats().patterns_stored,
        propositions_extracted: propositions.len(),
        voice_thoughts: voice_summary.len(),
        meta_convergence: meta_result.convergence_score,
        working_memory_load: state.working_memory.read().utilization(),
        pareto_before,
        pareto_after,
    }
}
```

### 3.5 API Extensions

New endpoints for the cognitive layer:

```
# Neural-Symbolic
POST /v1/reason           # Run neural-symbolic inference
GET  /v1/propositions     # List extracted propositions
POST /v1/ground           # Ground a new proposition

# Internal Voice
GET  /v1/voice/stream     # SSE stream of internal thoughts
GET  /v1/voice/working    # Current working memory contents
POST /v1/voice/goal       # Set a deliberation goal
GET  /v1/voice/history    # Recent thought history

# Enhanced Status
GET  /v1/cognitive/status # Full cognitive system status
```

## 4. Implementation Plan

| Phase | Component | Effort | Priority |
|-------|-----------|--------|----------|
| 1 | Working Memory module | 4 hrs | High |
| 2 | Internal Voice core | 6 hrs | High |
| 3 | Neural-Symbolic Bridge | 8 hrs | High |
| 4 | Training loop integration | 4 hrs | Medium |
| 5 | API endpoints | 4 hrs | Medium |
| 6 | SSE streaming for voice | 2 hrs | Low |
| 7 | Deliberation planner | 6 hrs | Low |

**Total: ~34 hours**

## 5. Integration with Existing Systems

### 5.1 SONA Integration

```rust
impl SonaEngine {
    /// Extract symbolic propositions from learned patterns
    pub fn to_propositions(&self) -> Vec<GroundedProposition> {
        self.patterns.iter()
            .filter(|p| p.confidence > 0.7)
            .map(|p| GroundedProposition {
                predicate: self.pattern_to_predicate(p),
                arguments: self.extract_entities(p),
                centroid: p.centroid.clone(),
                confidence: p.confidence,
                evidence: p.source_memories.clone(),
            })
            .collect()
    }
}
```

### 5.2 Strange Loop Integration

```rust
impl StrangeLoop {
    /// Run with internal voice context
    pub fn run_with_voice(&mut self, voice: &InternalVoice) -> LoopResult {
        // Include voice's working memory in context
        let mut ctx = self.context.clone();
        for token in voice.working_memory.iter() {
            ctx.insert(token.content.clone(), token.attention_weight);
        }
        self.run(&mut ctx)
    }
}
```

### 5.3 Hopfield Integration

```rust
impl CognitiveEngine {
    /// Store with symbolic grounding
    pub fn store_grounded(&mut self, id: &str, embedding: &[f32], proposition: &GroundedProposition) {
        self.store_pattern(id, embedding);
        self.grounding_map.insert(id.to_string(), proposition.clone());
    }

    /// Recall with symbolic interpretation
    pub fn recall_symbolic(&self, query: &[f32]) -> Option<(Vec<f32>, GroundedProposition)> {
        let recalled = self.recall(query)?;
        let prop = self.grounding_map.get(&self.nearest_id(&recalled))?;
        Some((recalled, prop.clone()))
    }
}
```

## 6. Monitoring and Observability

### 6.1 Cognitive Metrics

```rust
pub struct CognitiveMetrics {
    // Neural layer
    pub hopfield_patterns: usize,
    pub sona_trajectories: usize,
    pub embedding_drift: f64,

    // Internal voice layer
    pub working_memory_utilization: f64,
    pub thoughts_per_minute: f64,
    pub goal_completion_rate: f64,
    pub deliberation_depth_avg: f64,

    // Symbolic layer
    pub propositions_count: usize,
    pub inference_success_rate: f64,
    pub rule_coverage: f64,
}
```

### 6.2 Voice Telemetry

Stream internal voice to logs for debugging:
```rust
if config.enable_voice_telemetry {
    tracing::info!(
        thought_type = ?token.thought_type,
        content = %token.content,
        attention = token.attention_weight,
        "internal_voice"
    );
}
```

## 7. Consequences

### Positive
- **Explainability**: Internal voice provides reasoning transparency
- **Transfer**: Symbolic rules generalize across contexts
- **Debugging**: Working memory visible for troubleshooting
- **Composition**: Rules can be combined for complex inferences

### Negative
- **Complexity**: Three-layer architecture adds cognitive load
- **Latency**: Symbolic reasoning adds overhead (~10-50ms)
- **Storage**: Working memory and propositions consume RAM

### Neutral
- Training frequency may need adjustment for rule extraction
- Voice verbosity is configurable per deployment

## 8. Future Work

1. **Natural Language Generation**: Generate human-readable explanations from voice
2. **Causal Reasoning**: Add causal inference to symbolic layer
3. **Emotional Valence**: Add affect to internal voice
4. **Multi-Agent Dialogue**: Allow multiple brains to converse via voice
5. **Dream State**: Consolidate memories during low-activity periods

## 9. References

- [Neuro-Symbolic AI: The 3rd Wave](https://arxiv.org/abs/2012.05876)
- [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory)
- [Inner Speech and Metacognition](https://doi.org/10.1016/j.tics.2018.12.001)
- [ACT-R Cognitive Architecture](http://act-r.psy.cmu.edu/)
- [Soar Cognitive Architecture](https://soar.eecs.umich.edu/)
