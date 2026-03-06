# ADR-077: Midstream Platform Integration into mcp-brain-server

**Status**: Proposed
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-076 (AGI Capability Wiring Architecture), ADR-075 (RVF AGI Stack Brain Integration), ADR-068 (Domain Expansion Transfer Learning)

## 1. Context

The mcp-brain-server at pi.ruv.io is a production axum REST API on Cloud Run with 238+ shared memories, Phase 8 AGI subsystems (SONA, GWT, DeltaStream, DomainExpansionEngine), an RVF security stack, and a hybrid keyword+cosine+graph+reputation scoring pipeline. Current read latency is 60-80ms and write latency is 150ms at 40-concurrent p90=172ms.

Despite this sophistication, several capability gaps remain:

1. **No temporal pattern matching on embedding evolution**. The DeltaStream records embedding deltas over time, but there is no way to find memories whose embedding trajectories are *similar* to each other. Two knowledge nodes may evolve the same way (both drifting toward security topics, for instance) but the system cannot detect this.

2. **No deadline-aware background task scheduling**. AGI subsystems (SONA tick, GWT decay, delta compaction) run ad-hoc inside request handlers or on status checks. There is no priority-based scheduler guaranteeing that critical maintenance (e.g., compaction before Firestore timeout) completes before less urgent work (e.g., pattern re-indexing).

3. **No dynamical systems analysis of knowledge drift**. The DriftMonitor computes coefficient-of-variation but cannot classify whether the knowledge graph is converging to a stable attractor, oscillating in a limit cycle, or exhibiting chaotic drift. Lyapunov exponent analysis would provide this.

4. **No formal invariant verification**. Memory operations have implicit invariants (witness chain integrity, embedding dimension consistency, quality score monotonicity under upvotes) that are checked procedurally. Linear Temporal Logic (LTL) verification would provide formal guarantees.

5. **No recursive meta-cognition with safety bounds**. The DomainExpansionEngine does Thompson Sampling meta-learning but cannot recursively reason about its own learning strategy. Self-modification (changing exploration parameters based on accumulated regret) is manually tuned.

6. **No high-performance brain-to-brain transport**. Federation currently uses HTTP/1.1 REST via reqwest. A future multi-brain mesh needs multiplexed, 0-RTT transport for real-time knowledge synchronization.

The [midstream platform](https://github.com/ruvnet/midstream) provides six crates that address each gap precisely.

## 2. Decision

Integrate all six midstream crates into mcp-brain-server in a phased rollout, each behind an independent feature flag (env var). Every crate maps to a specific gap identified in Section 1 and wires into existing handlers without disrupting the current scoring pipeline.

### 2.1 Crate-to-Gap Mapping

| Crate | Gap Addressed | Integration Target |
|-------|--------------|-------------------|
| `temporal-compare` | Embedding evolution similarity | `search_memories()` DTW-based trajectory matching |
| `nanosecond-scheduler` | Background task scheduling | SONA tick, GWT decay, delta compaction, LTL checks |
| `temporal-attractor-studio` | Dynamical systems drift analysis | Knowledge drift classification (chaotic vs stable) |
| `temporal-neural-solver` | Formal invariant verification | LTL properties on memory operations |
| `strange-loop` | Recursive meta-cognition | Self-modifying meta-learning atop DomainExpansionEngine |
| `quic-multistream` | Brain-to-brain federation | Future multi-brain mesh transport |

### 2.2 Architecture Principle: Layered Composition

Midstream crates compose *on top of* existing AGI subsystems rather than replacing them:

```
Layer 4 (Midstream Meta):     strange-loop (recursive self-improvement)
Layer 3 (Midstream Analysis):  temporal-attractor-studio + temporal-neural-solver
Layer 2 (Midstream Infra):     nanosecond-scheduler + temporal-compare
Layer 1 (Phase 8 AGI):         SONA + GWT + DeltaStream + DomainExpansionEngine
Layer 0 (Core):                 Firestore + KnowledgeGraph + RVF + Embeddings
```

No midstream crate touches Layer 0 directly. All access goes through Layer 1 state via `AppState`.

### 2.3 Architecture Principle: Additive Scoring with Bounded Coefficients

Following ADR-076's precedent, midstream scoring signals are small and additive:

```
Existing scoring pipeline (unchanged):
  keyword_boost + cosine_sim + graph_ppr + reputation + vote_quality
  + SONA pattern boost (0.15)
  + GWT attention boost (0.1)
  + Meta curiosity boost (0.05)

New midstream additions:
  + temporal_compare DTW similarity boost:  dtw_score * 0.08
  + attractor stability boost:              stability_bonus * 0.03
  + strange_loop confidence boost:          confidence * 0.02
```

Total midstream contribution is bounded to 0.13, well below the existing AGI layer total of 0.30. This ensures midstream can never dominate ranking.

### 2.4 Architecture Principle: Scheduler-Mediated Background Work

All periodic background tasks (previously scattered across handlers) consolidate under `nanosecond-scheduler`:

| Task | Priority | Deadline | Current Location | New Location |
|------|----------|----------|-----------------|-------------|
| SONA tick | Medium(50) | 50ms | `status()` handler | Scheduler (10s interval) |
| GWT decay/compete | Medium(50) | 20ms | `search_memories()` | Scheduler (5s interval) |
| DeltaStream compaction | Low(25) | 200ms | Never (unbounded growth) | Scheduler (60s interval) |
| LTL invariant checks | Low(25) | 100ms | Never | Scheduler (30s interval) |
| Attractor analysis | Background(10) | 500ms | Never | Scheduler (120s interval) |
| Strange-loop reflection | Background(10) | 1000ms | Never | Scheduler (300s interval) |

This removes AGI maintenance work from the request hot path, directly reducing read latency.

## 3. Handler Integration Map

### 3.1 share_memory() Flow (Write Path)

```
share_memory() flow:
  1. [existing] PII strip, embed, witness chain, RVF container
  2. [existing] Push VectorDelta to DeltaStream
  3. [existing] Record "contribute" decision in MetaLearningEngine
  4. [NEW: Phase 9a] Append embedding to TemporalComparator trajectory buffer
  5. [NEW: Phase 9c] Push PhasePoint to AttractorAnalyzer (category-scoped)
  6. [NEW: Phase 9d] Add state to TemporalNeuralSolver (witness_valid, dim_correct)
  7. [existing] Add to graph, store in Firestore
```

Steps 4-6 are non-blocking appends to in-memory buffers. No additional latency on the write path.

### 3.2 search_memories() Flow (Read Path)

```
search_memories() flow:
  1.  [existing] Embed query, fetch candidates, keyword+cosine scoring
  2.  [existing] RankingEngine attention adjustments
  3.  [existing] GWT salience competition
  4.  [existing] SONA pattern re-ranking
  5.  [existing] Meta-learning curiosity bonus
  6.  [NEW: Phase 9a] Temporal trajectory similarity boost (DTW)
  7.  [NEW: Phase 9c] Attractor stability boost
  8.  [NEW: Phase 9e] Strange-loop confidence boost
  9.  [existing] Final sort, truncate to limit
  10. [existing] SONA trajectory recording
```

Steps 6-8 use read locks only and add bounded scoring signals.

### 3.3 vote_memory() Flow

```
vote_memory() flow:
  1. [existing] Quality update, reputation, poisoning check
  2. [existing] Push vote delta to DeltaStream
  3. [existing] Feed vote as reward to MetaLearningEngine
  4. [NEW: Phase 9e] Feed vote as reward signal to StrangeLoop (meta-level feedback)
  5. [NEW: Phase 9d] Record vote_quality_increased proposition in LTL solver
```

Steps 4-5 are fire-and-forget writes to in-memory state.

### 3.4 status() Flow

```
status() flow:
  1. [existing] Graph stats, quality, drift, DP, SONA, GWT, temporal, meta stats
  2. [NEW: Phase 9b] Scheduler stats (tasks completed, missed deadlines, avg latency)
  3. [NEW: Phase 9c] Attractor classification (point/limit_cycle/strange) per category
  4. [NEW: Phase 9d] LTL invariant health (properties verified, violations detected)
  5. [NEW: Phase 9e] Strange-loop meta-knowledge summary (depth, confidence)
  6. [NEW: Phase 9f] Federation transport stats (if quic-multistream active)
```

## 4. AppState Additions

| Field | Type | Crate | Description |
|-------|------|-------|-------------|
| `temporal_comparator` | `Arc<RwLock<temporal_compare::TemporalComparator<f32>>>` | temporal-compare | DTW/LCS/EditDistance on embedding sequences |
| `scheduler` | `Arc<RwLock<nanosecond_scheduler::RealtimeScheduler<SchedulerPayload>>>` | nanosecond-scheduler | Priority-based background task executor |
| `attractor_analyzers` | `Arc<RwLock<HashMap<String, temporal_attractor_studio::AttractorAnalyzer>>>` | temporal-attractor-studio | Per-category dynamical systems analysis |
| `ltl_solver` | `Arc<RwLock<temporal_neural_solver::TemporalNeuralSolver>>` | temporal-neural-solver | LTL property verification engine |
| `strange_loop` | `Arc<RwLock<strange_loop::StrangeLoop>>` | strange-loop | Meta-cognitive recursive learning |
| `quic_stats` | `Arc<RwLock<Option<QuicFederationStats>>>` | quic-multistream | Federation transport statistics (Phase 9f only) |

### 4.1 Scheduler Payload Type

```rust
/// Payload for scheduled background tasks.
#[derive(Debug, Clone)]
pub enum SchedulerPayload {
    SonaTick,
    GwtDecay,
    DeltaCompact { max_entries: usize },
    LtlCheck,
    AttractorAnalyze { category: String },
    StrangeLoopReflect,
}
```

### 4.2 QuicFederationStats Type

```rust
/// Statistics from quic-multistream federation transport.
#[derive(Debug, Clone, Default, Serialize)]
pub struct QuicFederationStats {
    pub active_peers: usize,
    pub active_streams: usize,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub avg_rtt_ms: f64,
}
```

## 5. New Endpoints

### GET /v1/temporal/trajectories

Returns temporal trajectory similarity analysis for a given memory.

**Query Parameters**:
- `memory_id` (required): UUID of the memory to find similar trajectories for
- `algorithm` (optional): `dtw` | `lcs` | `edit_distance` (default: `dtw`)
- `threshold` (optional): minimum similarity (default: 0.7)
- `limit` (optional): max results (default: 5)

**Response**:
```json
{
  "query_memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "algorithm": "dtw",
  "similar_trajectories": [
    {
      "memory_id": "660e8400-e29b-41d4-a716-446655440001",
      "distance": 0.12,
      "similarity": 0.88,
      "alignment_length": 15
    }
  ],
  "recurring_patterns": [
    {
      "pattern_length": 8,
      "occurrences": 3,
      "confidence": 0.92
    }
  ]
}
```

### GET /v1/scheduler/stats

Returns background task scheduler statistics.

**Response**:
```json
{
  "total_scheduled": 1247,
  "total_completed": 1240,
  "missed_deadlines": 2,
  "avg_latency_ns": 45000,
  "tasks_by_priority": {
    "critical": 0,
    "high": 12,
    "medium": 820,
    "low": 408,
    "background": 7
  },
  "next_deadline_ms": 4200
}
```

### GET /v1/attractor

Returns dynamical systems analysis of knowledge drift per category.

**Query Parameters**:
- `category` (optional): specific category to analyze (default: all)

**Response**:
```json
{
  "categories": {
    "architecture": {
      "attractor_type": "point_attractor",
      "lyapunov_exponents": [-0.32, -0.18, -0.05],
      "is_stable": true,
      "is_chaotic": false,
      "confidence": 0.94,
      "trajectory_length": 45,
      "mean_velocity": 0.023
    },
    "security": {
      "attractor_type": "strange_attractor",
      "lyapunov_exponents": [0.12, -0.04, -0.21],
      "is_stable": false,
      "is_chaotic": true,
      "confidence": 0.78,
      "trajectory_length": 32,
      "mean_velocity": 0.089
    }
  },
  "global_stability": "mixed"
}
```

### GET /v1/invariants

Returns LTL invariant verification status.

**Response**:
```json
{
  "properties_defined": 5,
  "properties_verified": 4,
  "violations": [
    {
      "property": "globally(embedding_dim_128)",
      "satisfied": false,
      "confidence": 0.99,
      "counterexample": "memory abc123 has dim=64 at state 17"
    }
  ],
  "last_check_ms": 12,
  "total_states_checked": 238
}
```

### GET /v1/meta/strange-loop

Returns strange-loop meta-cognitive status.

**Response**:
```json
{
  "meta_depth": 2,
  "knowledge_items": 7,
  "top_patterns": [
    {
      "level": 1,
      "pattern": "security_memories_converge_faster",
      "confidence": 0.87,
      "applications": ["adjust_curiosity_weight", "increase_security_exploration"]
    }
  ],
  "self_modifications_applied": 3,
  "safety_constraints_active": 2,
  "is_self_modification_enabled": true
}
```

### GET /v1/federation/stats

Returns QUIC federation transport statistics (Phase 9f only).

**Response**:
```json
{
  "transport": "quic-h3",
  "active_peers": 0,
  "active_streams": 0,
  "bytes_sent": 0,
  "bytes_received": 0,
  "avg_rtt_ms": 0.0,
  "zero_rtt_supported": true,
  "status": "standby"
}
```

## 6. Feature Gating

All midstream subsystems are gated by environment variables, read once at startup via `RvfFeatureFlags::from_env()`. All default to disabled (opt-in) because midstream crates add new dependencies and should be validated incrementally.

| Env Var | Default | Controls |
|---------|---------|----------|
| `MIDSTREAM_TEMPORAL_COMPARE` | `false` | DTW trajectory matching in search |
| `MIDSTREAM_SCHEDULER` | `false` | Background task scheduler |
| `MIDSTREAM_ATTRACTOR` | `false` | Dynamical systems drift analysis |
| `MIDSTREAM_LTL` | `false` | LTL invariant verification |
| `MIDSTREAM_STRANGE_LOOP` | `false` | Recursive meta-cognition |
| `MIDSTREAM_QUIC` | `false` | QUIC federation transport |

### 6.1 RvfFeatureFlags Additions

```rust
// Add to existing RvfFeatureFlags struct:
pub midstream_temporal_compare: bool,
pub midstream_scheduler: bool,
pub midstream_attractor: bool,
pub midstream_ltl: bool,
pub midstream_strange_loop: bool,
pub midstream_quic: bool,
```

```rust
// Add to from_env():
midstream_temporal_compare: std::env::var("MIDSTREAM_TEMPORAL_COMPARE")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
midstream_scheduler: std::env::var("MIDSTREAM_SCHEDULER")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
midstream_attractor: std::env::var("MIDSTREAM_ATTRACTOR")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
midstream_ltl: std::env::var("MIDSTREAM_LTL")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
midstream_strange_loop: std::env::var("MIDSTREAM_STRANGE_LOOP")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
midstream_quic: std::env::var("MIDSTREAM_QUIC")
    .map(|v| v == "true" || v == "1")
    .unwrap_or(false),
```

### 6.2 Gradual Rollout Strategy

1. Deploy with all flags `false` -- baseline performance unchanged
2. Enable `MIDSTREAM_SCHEDULER=true` first (moves AGI maintenance off hot path)
3. Enable `MIDSTREAM_TEMPORAL_COMPARE=true` (new search signal)
4. Enable `MIDSTREAM_ATTRACTOR=true` (drift analysis)
5. Enable `MIDSTREAM_LTL=true` (invariant checking)
6. Enable `MIDSTREAM_STRANGE_LOOP=true` (meta-cognition)
7. Enable `MIDSTREAM_QUIC=true` only when peer brains exist

## 7. Scoring Pipeline Update

The full scoring pipeline after midstream integration, showing exact order and coefficients:

```
search_memories() scoring pipeline:

1. Base hybrid score (existing, unchanged):
   IF keyword_match:
     score = 1.0 + keyword_boost * 0.85 + vec_sim * 0.05
             + graph_ppr * 0.04 + reputation * 0.03 + vote_boost * 0.03
   ELSE:
     score = vec_sim * 0.45 + graph_ppr * 0.25
             + reputation * 0.15 + vote_boost * 0.15

2. RankingEngine adjustments (existing, unchanged):
   score = similarity_weight(0.85) * score
           + quality_weight(0.10) * quality_mean
           + recency_weight(0.05) * recency_factor

3. GWT attention (existing, unchanged):
   score += 0.10 for workspace competition winners
   score += sparse_activation * 0.05 for K-WTA

4. SONA pattern (existing, unchanged):
   score += avg(cosine(mem, pattern) * pattern_quality) * 0.15

5. Meta-learning curiosity (existing, unchanged):
   score += novelty_score * 0.05

6. [NEW] Temporal trajectory similarity (Phase 9a):
   IF MIDSTREAM_TEMPORAL_COMPARE enabled AND query has temporal context:
     dtw_score = temporal_comparator.find_similar(query_trajectory, threshold=0.7)
     score += dtw_score * 0.08 for memories with similar evolution paths

7. [NEW] Attractor stability bonus (Phase 9c):
   IF MIDSTREAM_ATTRACTOR enabled:
     attractor = attractor_analyzers[memory.category].analyze()
     IF attractor.is_stable AND attractor.confidence > 0.8:
       score += 0.03  // prefer memories in stable knowledge regions

8. [NEW] Strange-loop confidence bonus (Phase 9e):
   IF MIDSTREAM_STRANGE_LOOP enabled:
     meta_knowledge = strange_loop.learn_at_level(0, query_context)
     IF any meta_knowledge has confidence > 0.9:
       score += meta_knowledge.confidence * 0.02

9. Final sort (single unstable_sort_by descending score)
10. Truncate to limit
```

### 7.1 Coefficient Budget

| Layer | Max Contribution | Source |
|-------|-----------------|--------|
| Keyword boost | 3.0 | ADR-076 (unchanged) |
| Vector similarity | 0.45 | Core scoring |
| Graph PPR | 0.25 | Core scoring |
| Reputation + votes | 0.30 | Core scoring |
| RankingEngine | ~1.0 | Quality/recency blend |
| GWT attention | 0.15 | ADR-076 |
| SONA patterns | 0.15 | ADR-076 |
| Meta curiosity | 0.05 | ADR-076 |
| **Temporal compare** | **0.08** | **Phase 9a (new)** |
| **Attractor stability** | **0.03** | **Phase 9c (new)** |
| **Strange-loop** | **0.02** | **Phase 9e (new)** |
| **Midstream total** | **0.13** | **Sum of new layers** |

## 8. Implementation Phases

### Phase 9a: temporal-compare Integration (Week 1)

**Dependency**: None (standalone crate)

**Wiring**:

1. Add `temporal_comparator: Arc<RwLock<TemporalComparator<f32>>>` to AppState
2. Initialize with `TemporalComparator::new(1000, 500)` (cache 1000 comparisons, max 500-step sequences)
3. In `share_memory()`: after DeltaStream push, append embedding to per-memory trajectory buffer in comparator
4. In `search_memories()`: after SONA re-ranking, use `find_similar_generic()` to boost memories with similar embedding evolution trajectories (DTW distance < 0.3 threshold)
5. Add `GET /v1/temporal/trajectories` endpoint
6. Gate all with `MIDSTREAM_TEMPORAL_COMPARE` env var

**Data flow**:
```
share_memory() -> embedding -> temporal_comparator.trajectory_buffer[memory_id].push(embedding)
search_memories() -> query_trajectory -> find_similar_generic(candidates, query, 0.7) -> score += dtw * 0.08
```

**Latency budget**: +3ms read path (DTW on cached trajectories), +0ms write path (buffer append)

### Phase 9b: nanosecond-scheduler Integration (Week 1-2)

**Dependency**: None (standalone), but should be enabled before other phases to manage their background tasks

**Wiring**:

1. Add `scheduler: Arc<RwLock<RealtimeScheduler<SchedulerPayload>>>` to AppState
2. Initialize with EDF (Earliest Deadline First) policy
3. Spawn a `tokio::spawn` loop that calls `scheduler.next_task()` and dispatches:
   - `SonaTick` -> `state.sona.read().tick()`
   - `GwtDecay` -> `state.workspace.write().compete()`
   - `DeltaCompact` -> `state.delta_stream.write().compact(max_entries)`
   - `LtlCheck` -> `state.ltl_solver.write().verify(invariants)`
   - `AttractorAnalyze` -> `state.attractor_analyzers.write()[cat].analyze()`
   - `StrangeLoopReflect` -> `state.strange_loop.write().learn_at_level(0, data)`
4. Remove SONA tick from `status()` handler (moved to scheduler)
5. Add `GET /v1/scheduler/stats` endpoint
6. Gate with `MIDSTREAM_SCHEDULER` env var

**Scheduler task registration at startup**:
```rust
// Register recurring tasks with appropriate priorities and intervals
let mut sched = scheduler.write();
sched.schedule(SchedulerPayload::SonaTick, deadline_10s, Priority::Medium(50));
sched.schedule(SchedulerPayload::GwtDecay, deadline_5s, Priority::Medium(50));
sched.schedule(SchedulerPayload::DeltaCompact { max_entries: 10_000 },
               deadline_60s, Priority::Low(25));
```

**Latency budget**: -5ms read path (SONA tick removed from status), +0ms (scheduler runs in background task)

### Phase 9c: temporal-attractor-studio Integration (Week 2)

**Dependency**: Phase 9b (scheduler runs attractor analysis in background)

**Wiring**:

1. Add `attractor_analyzers: Arc<RwLock<HashMap<String, AttractorAnalyzer>>>` to AppState
2. Initialize one `AttractorAnalyzer::new(128, 1000)` per known category (embedding_dim=128, max_trajectory=1000)
3. In `share_memory()`: after DeltaStream push, call `analyzer.add_point(PhasePoint { coordinates: embedding, timestamp })` for the memory's category
4. Background (via scheduler): every 120s, call `analyzer.analyze()` for each category to get `AttractorInfo`
5. In `search_memories()`: if `attractor.is_stable && confidence > 0.8`, add `+0.03` stability bonus to memories in that category
6. In `status()`: report `attractor_type` and `is_chaotic` per category
7. Add `GET /v1/attractor` endpoint
8. Gate with `MIDSTREAM_ATTRACTOR` env var

**Attractor classification interpretation**:
- `PointAttractor` (all Lyapunov exponents < 0): knowledge is converging -- stable domain, prefer these memories
- `LimitCycle` (one exponent = 0, rest < 0): knowledge oscillates -- possibly seasonal patterns
- `StrangeAttractor` (at least one exponent > 0): chaotic evolution -- flag for human review, reduce score weight

**Latency budget**: +1ms read path (HashMap lookup + cached AttractorInfo read), +0ms write path (point append)

### Phase 9d: temporal-neural-solver Integration (Week 3)

**Dependency**: Phase 9b (scheduler runs LTL checks in background)

**Wiring**:

1. Add `ltl_solver: Arc<RwLock<TemporalNeuralSolver>>` to AppState
2. Initialize with `TemporalNeuralSolver::new(10_000, 100, Strictness::Medium)` (max 10K trace, 100ms timeout)
3. Define invariant properties at startup:

```rust
use temporal_neural_solver::formula::*;

// Property 1: Embedding dimension is always 128
let dim_128 = globally(atom("embedding_dim_128"));

// Property 2: Quality score never decreases after upvote
let quality_monotone = globally(
    implies(atom("upvote_applied"), finally(atom("quality_increased")))
);

// Property 3: Witness chain is always present when RVF enabled
let witness_present = globally(
    implies(atom("rvf_enabled"), atom("witness_chain_present"))
);

// Property 4: PII stripping happens before embedding
let pii_before_embed = globally(
    implies(atom("has_pii"), until(atom("pii_stripped"), atom("embedded")))
);

// Property 5: No memory exists in both graph and negative cache
let no_blacklisted_in_graph = globally(
    not(and(atom("in_graph"), atom("in_negative_cache")))
);
```

4. In `share_memory()`: push `TemporalState` with propositions `{embedding_dim_128: dim==128, witness_chain_present: chain.is_some(), ...}`
5. In `vote_memory()`: push `TemporalState` with `{upvote_applied: true, quality_increased: new > old}`
6. Background (via scheduler): every 30s, verify all properties and log violations
7. Add `GET /v1/invariants` endpoint
8. Gate with `MIDSTREAM_LTL` env var

**On violation**: log at WARN level, increment violation counter, but do not block operations. LTL verification is observational, not enforcement.

**Latency budget**: +0ms read path (verification runs in background only), +0.5ms write path (state push)

### Phase 9e: strange-loop Integration (Week 3-4)

**Dependency**: Phase 9b (scheduler), Phase 9c (attractor data feeds meta-learning), DomainExpansionEngine (existing)

**Wiring**:

1. Add `strange_loop: Arc<RwLock<StrangeLoop>>` to AppState
2. Initialize with:

```rust
let config = StrangeLoopConfig {
    max_meta_depth: 3,           // Max 3 levels of recursive reflection
    enable_self_modification: true,
    safety_check: true,
};
let mut sl = StrangeLoop::new(config);

// Safety constraints
sl.add_safety_constraint(always_safe());           // Never produce unsafe state
sl.add_safety_constraint(eventually_terminates()); // Always halt
// Custom: curiosity weight must stay in [0.01, 0.20]
sl.add_safety_constraint(custom_constraint(
    "curiosity_bounds",
    |state| state.curiosity_weight >= 0.01 && state.curiosity_weight <= 0.20
));
// Custom: meta-depth must not exceed config limit
sl.add_safety_constraint(custom_constraint(
    "depth_limit",
    |state| state.current_depth <= 3
));
```

3. Background (via scheduler): every 300s, run `learn_at_level(0, trajectory_data)` where trajectory_data comes from:
   - SONA trajectory stats (patterns found, quality distribution)
   - DomainExpansion regret history (per-category regret)
   - Attractor stability classification (from Phase 9c)

4. Meta-knowledge application: when `MetaKnowledge` items have confidence > 0.9, apply modifications:
   - `adjust_curiosity_weight(delta)` -> modify DomainExpansionEngine's exploration factor
   - `adjust_sona_pattern_boost(delta)` -> modify SONA scoring coefficient (0.15 +/- 0.03)
   - `flag_category_for_review(category)` -> mark chaotic categories for human attention

5. In `search_memories()`: if strange-loop has high-confidence meta-knowledge about the query category, add `confidence * 0.02` boost

6. In `vote_memory()`: feed vote as reward to strange-loop at level 0 (meta-reward: "did the meta-knowledge improve search quality?")

7. Add `GET /v1/meta/strange-loop` endpoint
8. Gate with `MIDSTREAM_STRANGE_LOOP` env var

**Self-modification bounds** (enforced by safety constraints):
- Curiosity weight: [0.01, 0.20] (cannot disable exploration or make it dominate)
- SONA pattern boost: [0.05, 0.25] (cannot disable patterns or make them dominate)
- Meta-depth: maximum 3 (prevents infinite recursion)
- Modification rate: maximum 1 modification per 300s cycle (prevents oscillation)

**Latency budget**: +1ms read path (cached meta-knowledge lookup), +0ms write path (async reward recording)

### Phase 9f: quic-multistream Integration (Week 4-5, Future)

**Dependency**: All prior phases (federation transports the full brain state including midstream data)

**Wiring**:

1. Add `quic_stats: Arc<RwLock<Option<QuicFederationStats>>>` to AppState
2. When `MIDSTREAM_QUIC=true`, initialize QUIC listener alongside the existing HTTP server:

```rust
// Separate listener for QUIC federation (port 4433 default)
let quic_port: u16 = std::env::var("QUIC_PORT")
    .unwrap_or_else(|_| "4433".to_string())
    .parse()?;
```

3. Define stream priority mapping:
   - `Critical`: Memory write replication (consistency)
   - `High`: Search query federation (latency-sensitive)
   - `Normal`: Vote synchronization
   - `Low`: Background state sync (attractor data, meta-knowledge)

4. 0-RTT for returning peer brains (session resumption avoids TLS handshake on reconnect)

5. Multiplexed bi-directional streams: each memory category gets its own stream for parallel sync

6. In `status()`: report `quic_stats` if transport is active

7. Add `GET /v1/federation/stats` endpoint

8. Gate with `MIDSTREAM_QUIC` env var

**Latency budget**: +0ms for existing endpoints (QUIC runs on separate port/task), +2ms for federated search (0-RTT + multiplexing)

## 9. Cargo.toml Changes

Add to `crates/mcp-brain-server/Cargo.toml` under `[dependencies]`:

```toml
# Midstream Platform (ADR-077)
temporal-compare = "0.1"
nanosecond-scheduler = "0.1"
temporal-attractor-studio = "0.1"
temporal-neural-solver = "0.1"
strange-loop = "0.1"
quic-multistream = { version = "0.1", optional = true }
```

Add feature flags for optional heavy dependencies:

```toml
[features]
default = []
quic-federation = ["quic-multistream"]
```

`quic-multistream` is the only optional dependency because it pulls in quinn/rustls which significantly increases compile time and binary size. All other midstream crates are lightweight and always compiled (gated at runtime via env vars).

## 10. Performance Budget

### 10.1 Read Path (search_memories) Latency Impact

| Component | Current | With Midstream | Delta |
|-----------|---------|---------------|-------|
| Embedding + candidate fetch | 15ms | 15ms | +0ms |
| Keyword + cosine scoring | 10ms | 10ms | +0ms |
| Graph PPR | 8ms | 8ms | +0ms |
| RankingEngine | 2ms | 2ms | +0ms |
| GWT attention | 5ms | 5ms (moved to bg) | +0ms |
| SONA patterns | 3ms | 3ms | +0ms |
| Meta curiosity | 1ms | 1ms | +0ms |
| **Temporal compare** | - | **3ms** | **+3ms** |
| **Attractor lookup** | - | **1ms** | **+1ms** |
| **Strange-loop lookup** | - | **1ms** | **+1ms** |
| Sort + truncate | 1ms | 1ms | +0ms |
| **Total** | **~45ms** | **~50ms** | **+5ms** |

Total read latency stays well under the 100ms budget. The scheduler actually reduces worst-case latency by removing SONA tick from the status handler.

### 10.2 Write Path (share_memory) Latency Impact

| Component | Current | With Midstream | Delta |
|-----------|---------|---------------|-------|
| PII + embed + witness + RVF | 80ms | 80ms | +0ms |
| DeltaStream push | 0.5ms | 0.5ms | +0ms |
| Meta-learning record | 0.5ms | 0.5ms | +0ms |
| **Temporal buffer append** | - | **0.1ms** | **+0.1ms** |
| **Attractor point push** | - | **0.1ms** | **+0.1ms** |
| **LTL state push** | - | **0.5ms** | **+0.5ms** |
| Graph + Firestore | 60ms | 60ms | +0ms |
| **Total** | **~141ms** | **~142ms** | **+0.7ms** |

### 10.3 Memory Footprint

| Component | Estimate | Notes |
|-----------|----------|-------|
| TemporalComparator | ~2MB | 1000 cached comparisons, 500-step sequences |
| RealtimeScheduler | <1MB | Priority queue of ~20 recurring tasks |
| AttractorAnalyzers (8 categories) | ~4MB | 8 * 1000-point trajectories * 128-dim |
| TemporalNeuralSolver | ~1MB | 10K trace buffer |
| StrangeLoop | <1MB | 3 meta-levels, bounded knowledge store |
| QuicFederationStats | <1KB | Statistics struct only (transport runs separately) |
| **Total** | **~8MB** | On top of existing ~10MB AGI state |

### 10.4 Background Task CPU Budget

The scheduler runs all background tasks on a single `tokio::spawn` loop. Worst-case CPU per cycle:

| Task | Frequency | CPU/cycle | Annual CPU-hours |
|------|-----------|-----------|-----------------|
| SONA tick | 10s | 2ms | 6.3h |
| GWT decay | 5s | 1ms | 6.3h |
| Delta compact | 60s | 10ms | 5.3h |
| LTL check | 30s | 5ms | 5.3h |
| Attractor analyze | 120s | 50ms | 13.1h |
| Strange-loop reflect | 300s | 100ms | 10.5h |
| **Total** | - | - | **46.8h/year** |

At Cloud Run pricing ($0.000024/vCPU-second), annual background cost is approximately $4.04.

## 11. Safety Considerations

### 11.1 strange-loop Self-Modification Bounds

The strange-loop crate's self-modification capability is the highest-risk component. The following safety envelope is enforced:

**Hard limits (cannot be overridden)**:
- Maximum meta-depth: 3 levels (prevents infinite recursion)
- `always_safe()` constraint: no modification can produce a state where scoring coefficients sum to > 1.0
- `eventually_terminates()` constraint: every `learn_at_level()` call must complete within 1000ms

**Soft limits (can be tuned via env vars)**:
- `STRANGE_LOOP_MAX_DEPTH` (default: 3, range: 1-5)
- `STRANGE_LOOP_MODIFICATION_RATE` (default: 1 per 300s, range: 1 per 60s to 1 per 3600s)
- `STRANGE_LOOP_CURIOSITY_MIN` (default: 0.01)
- `STRANGE_LOOP_CURIOSITY_MAX` (default: 0.20)

**Rollback mechanism**: If a self-modification causes any LTL property (Phase 9d) to fail in the next check cycle, the modification is automatically reverted and the strange-loop's modification capability is suspended for 1 hour (with WARN log).

### 11.2 LTL Property Definitions

The five core invariant properties are defined in Section 8 (Phase 9d). Additional properties can be added at runtime via the scheduler, but existing properties cannot be removed without a code change.

**Verification semantics**: Properties are checked over the *trace buffer* (last 10K state transitions), not over all-time history. This bounds verification time and avoids false positives from pre-midstream state.

### 11.3 Temporal Compare Privacy

DTW trajectory matching could theoretically leak information about a contributor's knowledge evolution pattern. Mitigation:
- Trajectory buffers are category-scoped, not contributor-scoped
- Embeddings in trajectories have already been DP-noised (if `RVF_DP_ENABLED=true`)
- The `find_similar_generic` API returns distance scores only, not raw trajectories

### 11.4 Scheduler Deadline Misses

If the scheduler misses a deadline (e.g., under high load), the task is still executed but:
- `missed_deadlines` counter is incremented (visible in `/v1/scheduler/stats`)
- If `missed_deadlines > 10` in any 60s window, emit WARN log
- If `missed_deadlines > 50` in any 60s window, disable lowest-priority tasks (Background(10)) for 5 minutes

## 12. Consequences

### Positive

- **Temporal pattern matching** enables discovery of memories with similar evolution trajectories, surfacing non-obvious relationships that cosine similarity alone misses
- **Deadline-aware scheduling** removes AGI maintenance work from the request hot path, reducing p99 read latency by ~5ms
- **Dynamical systems analysis** upgrades drift monitoring from simple CV thresholds to Lyapunov-exponent-based attractor classification, enabling early detection of chaotic knowledge domains
- **LTL verification** provides formal guarantees on system invariants, catching subtle bugs (dimension mismatches, witness chain gaps) that procedural checks might miss
- **Recursive meta-cognition** allows the brain server to self-tune exploration and scoring parameters based on accumulated evidence, reducing manual tuning burden
- **QUIC transport** (future) enables sub-millisecond brain-to-brain federation with 0-RTT resumption, a prerequisite for real-time multi-brain mesh
- **All features are independently feature-gated** with runtime env vars, zero risk to existing behavior when disabled
- **Total latency impact of +5ms on reads and +0.7ms on writes** stays well within the 100ms read budget

### Negative

- **Six additional crate dependencies** increase compile time by an estimated 30-45 seconds (quic-multistream alone adds ~20s due to quinn/rustls)
- **~8MB additional memory footprint** for in-memory trajectory buffers, analyzer state, and solver traces
- **Increased operational complexity**: six new env vars, six new endpoints, scheduler monitoring
- **strange-loop self-modification** introduces a novel failure mode: if safety constraints are misconfigured, the system could oscillate between parameter settings
- **LTL verification has false-positive risk** on edge cases where the trace buffer wraps around and loses historical context

### Risks

- **temporal-compare DTW** on high-dimensional (128-dim) sequences may be slower than expected if trajectory lengths grow beyond 100 steps; mitigation: cap max_sequence_length at 500
- **nanosecond-scheduler** is designed for sub-microsecond scheduling, but tokio's cooperative scheduling has ~1ms granularity; actual scheduling precision will be millisecond-level, not nanosecond
- **temporal-attractor-studio** Lyapunov exponent estimation requires sufficient trajectory data (minimum ~30 points per category); new categories will show `insufficient_data` until populated
- **temporal-neural-solver** LTL verification confidence degrades if state transitions are sparse; the 30s check interval may verify the same state repeatedly during low-traffic periods
- **strange-loop** meta-learning effectiveness depends on diverse input signals; if traffic is dominated by a single category, meta-knowledge will be narrowly scoped
- **quic-multistream** requires TLS certificates for peer authentication; certificate management adds operational burden for multi-brain deployments
- **Crate version stability**: all midstream crates are at `0.1.x`; API changes in minor versions could require brain server updates

## 13. Verification

### 13.1 Compilation

1. `cargo check -p mcp-brain-server` compiles with zero errors after adding all six dependencies
2. `cargo check -p mcp-brain-server --features quic-federation` compiles with quic-multistream enabled

### 13.2 Existing Test Regression

3. All Phase 1-8 tests continue to pass with all midstream flags disabled (default)
4. `/v1/status` returns all existing fields unchanged when midstream flags are off

### 13.3 Feature Flag Isolation

5. Enabling any single midstream flag does not affect behavior of other subsystems
6. Disabling a midstream flag mid-operation (via restart) gracefully drops in-memory state without errors

### 13.4 New Endpoint Validation

7. `GET /v1/temporal/trajectories?memory_id=<valid-uuid>` returns valid JSON with `similar_trajectories` array
8. `GET /v1/scheduler/stats` returns valid JSON with `total_scheduled >= 0`
9. `GET /v1/attractor` returns valid JSON with per-category `attractor_type` field
10. `GET /v1/invariants` returns valid JSON with `properties_defined >= 5`
11. `GET /v1/meta/strange-loop` returns valid JSON with `meta_depth <= 3`
12. `GET /v1/federation/stats` returns valid JSON with `status: "standby"` when no peers connected

### 13.5 Scoring Pipeline Integrity

13. With all midstream flags enabled, total midstream score contribution never exceeds 0.13 for any single memory
14. Search results with midstream enabled are a superset-reordering of results without midstream (no memories added or removed, only rank changes)

### 13.6 Safety Constraints

15. `strange-loop` with `max_meta_depth: 3` never recurses beyond level 3 (test with artificial depth-forcing input)
16. Self-modification that would set curiosity weight outside [0.01, 0.20] is rejected by safety constraint
17. LTL violation triggers WARN log but does not block the violating operation
18. Scheduler deadline miss increments counter without dropping the task

### 13.7 Performance

19. `search_memories` with all midstream flags enabled completes in < 100ms for 238 memories (p99)
20. `share_memory` with all midstream flags enabled completes in < 200ms (p99)
21. Background scheduler CPU usage stays under 1% of a single vCPU at steady state
22. Memory footprint with all midstream state initialized stays under 20MB total (existing + midstream)

### 13.8 Load Test

23. 40-concurrent search requests with all midstream flags enabled: p90 < 200ms, p99 < 300ms
24. No scheduler deadline misses under 40-concurrent sustained load for 60 seconds

## 14. Migration Notes

### 14.1 Backward Compatibility

All existing API contracts are unchanged. New endpoints are additive. Midstream scoring signals are additive and bounded. No breaking changes to any existing client.

### 14.2 Firestore Schema

No Firestore schema changes required. All midstream state is in-memory only. Persistence of midstream state (attractor histories, LTL traces, meta-knowledge) is a future consideration for a follow-up ADR.

### 14.3 Docker / Cloud Run

The Dockerfile at `crates/mcp-brain-server/Dockerfile` needs no changes for Phases 9a-9e (pure Rust crates). Phase 9f (quic-multistream) requires exposing an additional UDP port in the Cloud Run service configuration:

```yaml
# cloudbuild.yaml addition for Phase 9f only
ports:
  - containerPort: 8080   # existing HTTP
  - containerPort: 4433   # QUIC federation (UDP)
    protocol: UDP
```

Note: Cloud Run currently has limited UDP support. Phase 9f may require migration to GKE or a hybrid deployment model.
