# ADR-076: AGI Capability Wiring Architecture

**Status**: Implemented
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-075 (RVF AGI Stack Brain Integration), ADR-068 (Domain Expansion Transfer Learning), ADR-074 (RuvLLM Neural Embeddings)

## 1. Context

The mcp-brain-server at pi.ruv.io had four AGI subsystem crates available in the workspace but minimally integrated:

- **SONA** (`sona`): 3-tier hierarchical learning engine with pattern detection and trajectory tracking
- **Global Workspace Theory** (`ruvector-nervous-system`): Salience-based attention competition inspired by cognitive GWT
- **Temporal Delta Tracking** (`ruvector-delta-core`): Time-series delta streams for tracking embedding evolution
- **Meta-Learning Exploration** (`ruvector-domain-expansion`): Thompson Sampling meta-learning with curiosity, regret, and plateau detection

Each crate had comprehensive unit tests but no integration with the live brain server. The `DomainExpansionEngine` was in `AppState` but only touched by the transfer endpoint. The other three subsystems were not in AppState at all.

## 2. Decision

Wire all four AGI subsystems into the brain server's core handlers (`share_memory`, `search_memories`, `vote_memory`, `status`) with independent feature flags for gradual rollout. Each subsystem adds a distinct cognitive capability without disrupting existing search/ranking behavior.

### 2.1 Architecture Principle: Additive Scoring Layers

Each AGI subsystem contributes a small additive score adjustment to the existing hybrid ranking pipeline:

```
Base score: keyword_boost + cosine_similarity + graph_ppr + reputation + vote_quality
  + SONA pattern boost:    cosine(mem, pattern) * quality * 0.15
  + GWT attention boost:   +0.1 for workspace competition winners
  + Meta curiosity boost:  novelty_score * 0.05
```

The small coefficients (0.05-0.15) ensure no single subsystem can dominate ranking, while still providing measurable signal from each capability.

### 2.2 Architecture Principle: Read-Lock Scoring, Write-Lock Learning

All scoring operations (search) use read locks on AGI state. Learning operations (share, vote) use write locks. This minimizes contention:

| Operation | SONA | GWT | DeltaStream | MetaLearning |
|-----------|------|-----|-------------|--------------|
| search (score) | read | write (compete) | - | read |
| share (learn) | - | - | write | write |
| vote (learn) | - | - | write | write |
| status (report) | read | read | read | read |

GWT is the exception: `compete()` mutates workspace state during search. This is intentional — attention competition is inherently stateful.

### 2.3 Architecture Principle: Feature-Gated with Defaults On

All four subsystems default to enabled. This is safe because:

1. Score contributions are small (0.05-0.15) and additive
2. Each subsystem starts with no learned state (cold start = no effect)
3. Feature flags allow instant disable without redeployment via env vars
4. Subsystems learn passively from existing traffic — no active exploration that could degrade quality

### 2.4 Handler Integration Map

```
share_memory() flow:
  1. [existing] PII strip, embed, witness chain, RVF container
  2. [Phase 8] Push VectorDelta to DeltaStream (temporal)
  3. [Phase 8] Record "contribute" decision in MetaLearningEngine
  4. [existing] Add to graph, store in Firestore

search_memories() flow:
  1. [existing] Embed query, fetch candidates, keyword+cosine scoring
  2. [existing] RankingEngine attention adjustments
  3. [Phase 8] GWT salience competition (broadcast → compete → boost winners)
  4. [Phase 8] SONA pattern re-ranking (centroid similarity × quality)
  5. [Phase 8] Meta-learning curiosity bonus (novelty_score × 0.05)
  6. [existing] Truncate to limit
  7. [Phase 8] SONA trajectory recording (search→result for online learning)

vote_memory() flow:
  1. [existing] Quality update, reputation, poisoning check
  2. [Phase 8] Push vote delta to DeltaStream (temporal)
  3. [Phase 8] Feed vote as reward signal to MetaLearningEngine
  4. [existing] Record contribution
```

## 3. New Endpoints

### GET /v1/sona/stats

Returns SONA learning engine statistics:
```json
{
  "patterns_stored": 12,
  "trajectories_buffered": 45,
  "background_ticks": 3
}
```

### GET /v1/temporal

Returns temporal delta tracking statistics:
```json
{
  "total_deltas": 237,
  "recent_hour_deltas": 14,
  "knowledge_velocity": 14.0,
  "trend": "growing"
}
```

### GET /v1/explore

Returns meta-learning exploration diagnostics:
```json
{
  "most_curious_category": "security",
  "most_curious_novelty": 0.92,
  "regret_summary": {
    "total_regret": 0.0,
    "average_regret": 0.0,
    "mean_growth_rate": 1.0,
    "converged_buckets": 0,
    "bucket_count": 0,
    "total_observations": 0
  },
  "plateau_status": "learning",
  "is_learning": false,
  "is_diverse": false,
  "is_exploring": false,
  "curiosity_total_visits": 0,
  "pareto_size": 0
}
```

## 4. AppState Additions

| Field | Type | Subsystem |
|-------|------|-----------|
| `sona` | `Arc<RwLock<SonaEngine>>` | SONA 3-tier learning |
| `workspace` | `Arc<RwLock<GlobalWorkspace>>` | GWT attention |
| `delta_stream` | `Arc<RwLock<DeltaStream<VectorDelta>>>` | Temporal tracking |
| `domain_engine` | `Arc<RwLock<DomainExpansionEngine>>` | Meta-learning (pre-existing) |

## 5. Consequences

### Positive

- Brain server now has four distinct AGI learning capabilities operating in production
- Search ranking benefits from multi-signal fusion: patterns, attention, curiosity, keywords
- Knowledge evolution is tracked over time, enabling trend detection and velocity monitoring
- Meta-learning provides self-diagnostic capabilities (regret, plateau, Pareto optimization)
- All capabilities are feature-gated for safe gradual rollout
- Cold-start behavior is neutral (no learned state = no effect on ranking)

### Negative

- Four additional read/write locks in the search path increase contention potential
- GWT workspace mutation during search is a sequential bottleneck
- Each subsystem adds ~1-5ms to search latency (total ~5-15ms)
- Memory footprint increases by ~2-8MB for AGI state (patterns, workspace, delta stream)

### Risks

- SONA pattern learning may create feedback loops (popular patterns get more popular)
- GWT K-WTA competition with small candidate sets may not produce meaningful selection
- Meta-learning curiosity bonus may be too small (0.05) to noticeably affect ranking
- Temporal delta stream grows unbounded without periodic compaction — needs future cleanup

## 6. Verification

1. `cargo check` from `crates/mcp-brain-server/` compiles with zero errors
2. All Phase 1-7 tests continue to pass
3. `/v1/status` returns new fields: `sona_patterns`, `gwt_workspace_load`, `knowledge_velocity`, `meta_avg_regret`, `meta_plateau_status`
4. `/v1/explore`, `/v1/temporal`, `/v1/sona/stats` return valid JSON
5. Feature flags disable each subsystem independently without affecting others
