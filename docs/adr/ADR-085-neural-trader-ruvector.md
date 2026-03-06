# ADR-085: RuVector Neural Trader — Dynamic Market Graphs, MinCut Coherence Gating, and Proof-Gated Mutation

## Status

Proposed

## Date

2026-03-06

## Deciders

ruv

## Related

- ADR-016 RuVector integration patterns
- ADR-030 RVF computational container
- ADR-040 WASM programmable sensing
- ADR-041 curated module registry
- ADR-042 Security RVF AIDefence TEE
- ADR-047 Proof-gated mutation protocol
- `examples/neural-trader/` existing example scaffold
- Cognitive MinCut Engine
- Mincut Gated Transformer
- ruvector-postgres architecture
- Cognitum Gate coherence layer

## Context

Most trading systems still split the problem the wrong way.

They keep market data in one system, features in another, models in another, audit logs somewhere else, and risk logic in handwritten code wrapped around the outside. That creates latency, drift, and a complete mess when you try to explain why a model made a decision or why a learning update happened.

A neural trader built on RuVector should treat the market as a living graph, not a table of candles. The limit order book, executions, cancellations, venue changes, and cross-asset interactions form a dynamic relational structure. That structure is exactly where short-horizon edge exists.

The design goal is a single substrate where:

1. Raw market events become typed graph state
2. Vector embeddings represent evolving microstructure state
3. GNN and temporal attention operate directly on that state
4. Dynamic mincut acts as a first-class coherence and fragility signal
5. Every state mutation and policy action is proof-gated and attestable
6. Online learning remains bounded, replayable, and auditable

This ADR defines the RuVector-native implementation for Neural Trader as a coherence-first trading substrate for prediction, risk control, and bounded execution research.

## Decision

We will implement Neural Trader as a RuVector-native market intelligence stack with six layers:

1. Ingest and normalization layer
2. Dynamic heterogeneous market graph in RuVector
3. Vector and graph learning layer using temporal GNN and attention
4. Two-stage memory selection and replay layer
5. MinCut-based coherence gate for write, retrieve, learn, and act
6. Policy and actuation layer with proof-gated mutation and witness logs

The system will use Postgres as the relational source of record, with `ruvector-postgres` as the embedded vector engine and the RuVector graph substrate for dynamic structural reasoning.

**No model output may directly mutate live strategy state, place orders, or update memory without passing coherence, risk, and policy gates.**

## Why This Decision

This approach matches the actual shape of markets.

A limit order book is not just a time series. It is a dynamic graph with queue locality, price adjacency, event causality, hidden liquidity hints, and regime-dependent cross-symbol coupling. A graph-plus-vector substrate captures that directly.

RuVector also gives us something most trading systems do not have:

1. Dynamic mincut as a real-time structural integrity signal
2. Unified vector-plus-graph storage
3. Replayable witness logs
4. Proof-gated state mutation
5. Local-first deployment paths from server to WASM to edge nodes

The result is a trading research platform that optimizes for **bounded intelligence** rather than blind prediction.

## Scope

### In scope

1. Market data representation
2. RuVector schema
3. Embedding and learning design
4. Memory selection
5. Coherence gating
6. Actuation policy
7. Verification and auditability
8. Deployment topology

### Out of scope

1. Broker-specific adapters in detail
2. Exchange colocation engineering
3. Final production capital allocation policy
4. Regulatory filing requirements by jurisdiction

## Assumptions

1. Primary use case is short-horizon market making, execution assistance, or micro-alpha research.
2. Input streams include order book updates, trades, cancels, modifies, venue metadata, and optional cross-asset feeds.
3. Latency budget is sub-second for research serving, with optional lower-latency kernels for action gating.
4. Hidden liquidity cannot be observed directly, so proxies are inferred from event patterns.
5. Online learning must remain bounded and reversible.
6. Correctness is treated as adversarially stressed rather than guaranteed.

---

## Architecture

### 1. Ingest and Normalization

**Input streams:**

1. L2 or L3 order book deltas
2. Trades and fills
3. Order lifecycle events (new, modify, cancel, expire)
4. Venue state and session markers
5. Symbol metadata
6. Optional news, macro, or derived volatility streams

**Normalization output:**

1. Canonical event envelopes
2. Sequence-aligned timestamps
3. Symbol and venue partition keys
4. Side, price, size, aggressor, queue, and microstructure features
5. Compact hashes for traceability

**Canonical event envelope:**

```rust
pub struct MarketEvent {
    pub event_id: [u8; 16],
    pub ts_exchange_ns: u64,
    pub ts_ingest_ns: u64,
    pub venue_id: u16,
    pub symbol_id: u32,
    pub event_type: EventType,
    pub side: Option<Side>,
    pub price_fp: i64,
    pub qty_fp: i64,
    pub order_id_hash: Option<[u8; 16]>,
    pub participant_id_hash: Option<[u8; 16]>,
    pub flags: u32,
    pub seq: u64,
}
```

### 2. RuVector Graph Model

The order book becomes a typed heterogeneous dynamic graph.

**Node kinds:**

| # | Kind | Description |
|---|------|-------------|
| 1 | Symbol | Tradable instrument |
| 2 | Venue | Exchange or dark pool |
| 3 | PriceLevel | Individual price level in the book |
| 4 | Order | Resting or aggressing order proxy |
| 5 | Trade | Matched execution |
| 6 | Event | Raw market event |
| 7 | Participant | Anonymized participant proxy |
| 8 | TimeBucket | Discretized time window |
| 9 | Regime | Market regime classification |
| 10 | StrategyState | Current strategy context |

**Edge kinds:**

| # | Edge | From → To |
|---|------|-----------|
| 1 | `AT_LEVEL` | Order → PriceLevel |
| 2 | `NEXT_TICK` | PriceLevel ↔ PriceLevel |
| 3 | `GENERATED` | Event → Order or Trade |
| 4 | `MATCHED` | Aggressor ↔ Resting order proxy |
| 5 | `MODIFIED_FROM` | Order → Order (prior version) |
| 6 | `CANCELED_BY` | Event → Order |
| 7 | `BELONGS_TO_SYMBOL` | * → Symbol |
| 8 | `ON_VENUE` | * → Venue |
| 9 | `IN_WINDOW` | * → TimeBucket |
| 10 | `CORRELATED_WITH` | Symbol ↔ Symbol |
| 11 | `IN_REGIME` | TimeBucket → Regime |
| 12 | `AFFECTS_STATE` | * → StrategyState |

**Core properties — PriceLevel:**

- Visible depth
- Estimated hidden depth
- Queue length
- Local imbalance
- Refill rate
- Depletion rate
- Spread distance
- Local realized volatility

**Core properties — Order:**

- Side
- Limit price
- Current queue estimate
- Age
- Modify count
- Cancel hazard score
- Fill hazard score

**Core properties — Trade:**

- Aggressor side
- Size
- Slippage to mid
- Post-trade impact window

**Core properties — Edge:**

- Event time delta
- Transition count
- Influence score
- Coherence contribution
- Venue confidence

### 3. Vector Representation

Each important subgraph window is embedded into RuVector.

**Embedding families:**

1. Book state embedding
2. Queue state embedding
3. Event stream embedding
4. Cross-symbol regime embedding
5. Strategy context embedding
6. Risk context embedding

**Recommended representation split:**

1. Dense float embeddings for state similarity
2. Compressed low-bit serving vectors for fast retrieval
3. Graph neighborhood fingerprints for structural similarity
4. Contrastive delta embeddings for regime shift detection

**Example keyspaces in ruvector-postgres:**

```sql
-- Event log: range-partitioned by ts_exchange_ns for bounded retention
CREATE TABLE nt_event_log (
    event_id       BYTEA NOT NULL,
    ts_exchange_ns BIGINT NOT NULL,
    ts_ingest_ns   BIGINT NOT NULL,
    venue_id       INT NOT NULL,
    symbol_id      INT NOT NULL,
    event_type     INT NOT NULL,
    payload        JSONB NOT NULL,
    witness_hash   BYTEA,
    PRIMARY KEY (ts_exchange_ns, event_id)
) PARTITION BY RANGE (ts_exchange_ns);

CREATE INDEX idx_event_log_symbol_ts
    ON nt_event_log (symbol_id, ts_exchange_ns);
CREATE INDEX idx_event_log_venue_ts
    ON nt_event_log (venue_id, ts_exchange_ns);

-- Embeddings: composite index for time-range similarity queries
CREATE TABLE nt_embeddings (
    embedding_id   BIGSERIAL PRIMARY KEY,
    symbol_id      INT NOT NULL,
    venue_id       INT NOT NULL,
    ts_ns          BIGINT NOT NULL,
    embedding_type TEXT NOT NULL,
    dim            INT NOT NULL,
    metadata       JSONB NOT NULL,
    embedding      vector(256)
);

CREATE INDEX idx_embeddings_symbol_ts
    ON nt_embeddings (symbol_id, ts_ns DESC);
CREATE INDEX idx_embeddings_type_ts
    ON nt_embeddings (embedding_type, ts_ns DESC);
CREATE INDEX idx_embeddings_vec_hnsw
    ON nt_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Replay segments: partitioned by start_ts_ns for retention management
CREATE TABLE nt_segments (
    segment_id   BIGSERIAL NOT NULL,
    symbol_id    INT NOT NULL,
    start_ts_ns  BIGINT NOT NULL,
    end_ts_ns    BIGINT NOT NULL,
    segment_kind TEXT NOT NULL,
    rvf_blob     BYTEA,
    signature    BYTEA,
    witness_hash BYTEA,
    metadata     JSONB,
    PRIMARY KEY (start_ts_ns, segment_id)
) PARTITION BY RANGE (start_ts_ns);

CREATE INDEX idx_segments_symbol_ts
    ON nt_segments (symbol_id, start_ts_ns DESC);
CREATE INDEX idx_segments_kind
    ON nt_segments (segment_kind, start_ts_ns DESC);
```

### 4. Learning Layer

We will use a temporal graph learning stack.

**Model family:**

1. Typed message passing over dynamic graph neighborhoods
2. Temporal attention over recent event windows
3. Optional sequence head for action or risk outputs
4. Auxiliary contrastive loss for regime separation
5. Coherence regularization using mincut and boundary stability

**Primary prediction heads:**

1. Next-window mid-price move probability
2. Fill probability for candidate placements
3. Cancel probability for resting liquidity
4. Slippage risk
5. Local volatility jump risk
6. Regime transition probability

**Control heads:**

1. Place or do-not-place
2. Modify or hold
3. Size scaling factor
4. Venue selection
5. Learning write admission score

**Loss design — total loss:**

```
L = L_pred + λ₁·L_fill + λ₂·L_risk + λ₃·L_contrast + λ₄·L_coherence + λ₅·L_budget
```

Where:

- `L_pred` — predicts short-horizon outcome
- `L_fill` — estimates execution quality
- `L_risk` — penalizes unstable high-drawdown actions
- `L_contrast` — separates regimes and recurrent motifs
- `L_coherence` — penalizes representation drift across stable partitions
- `L_budget` — penalizes actions that exceed risk or actuation budgets

### 5. Memory Design

Memory must be selective, bounded, and useful.

#### Stage A: Streaming Sketch

Keep cheap summaries for recent heavy hitters.

**Structures:**

1. Count-Min sketch for repeated motifs
2. Top-K for impactful levels, venues, regimes
3. Rolling range sketches for volatility and imbalance bands
4. Delta histograms for event transitions

**Purpose:**

1. Detect recurring market motifs
2. Prioritize candidate memory writes
3. Reduce storage pressure
4. Preserve streaming summaries even when raw fragments age out

#### Stage B: Uncertainty-Guided Reservoir

Store high-value replay fragments when one or more conditions hold:

1. High model uncertainty
2. Large realized PnL impact
3. Regime transition
4. Structural anomaly
5. Rare queue pattern
6. High disagreement between model heads

Each stored fragment becomes an RVF or signed segment containing:

1. Compact subgraph
2. Embeddings
3. Labels and realized outcomes
4. Coherence statistics
5. Lineage metadata
6. Witness hash and signature

### 6. Coherence Gate

Dynamic mincut is the central gate.

We compute a compact induced subgraph linking:

1. Incoming market events
2. Local price levels
3. Relevant prior memories
4. Current strategy state
5. Risk nodes

From this graph we derive:

1. Canonical mincut partition
2. Cut value
3. Boundary node identities
4. Cut drift over time
5. Embedding drift by partition
6. CUSUM alarms over cut metrics

**Gate uses:**

1. Memory write admission
2. Memory retrieval confidence
3. Online learner update permission
4. Action permission
5. Early rollback trigger
6. Anomaly escalation

**Gate policy — allow only when ALL are true:**

1. Cut value above floor for current regime
2. Boundary identity stable across last N windows
3. No sustained CUSUM breach
4. Risk budgets available
5. Policy allows actuation
6. Model confidence exceeds threshold conditioned on coherence

**Gate result type:**

```rust
pub struct CoherenceDecision {
    pub allow_retrieve: bool,
    pub allow_write: bool,
    pub allow_learn: bool,
    pub allow_act: bool,
    pub mincut_value: u64,
    pub partition_hash: [u8; 16],
    pub drift_score: f32,
    pub cusum_score: f32,
    pub reasons: Vec<String>,
}
```

---

## Proof-Gated Mutation

No state mutation occurs without a proof token.

This includes:

1. Memory writes
2. Model promotion
3. Policy threshold changes
4. Live order intents
5. Strategy state transitions

**Mutation protocol:**

1. Compute features and local graph
2. Compute coherence decision
3. Evaluate policy kernel
4. Mint verified token if allowed
5. Apply mutation
6. Append witness receipt

**Receipt fields:**

1. Timestamp
2. Model ID
3. Input segment hash
4. Coherence witness hash
5. Policy hash
6. Action intent
7. Verified token ID
8. Resulting state hash

---

## Serving Flow

### Research or Paper Trading Path

1. Ingest market events
2. Update graph and embeddings
3. Retrieve similar memory fragments
4. Compute model outputs
5. Run coherence gate
6. Run policy and budget checks
7. Emit action recommendation
8. Store replay artifacts if admitted

### Live Bounded Execution Path

1. Ingest event burst
2. Update local graph cache
3. Score candidate actions
4. Compute mincut coherence
5. Check exposure and slippage budgets
6. Require proof token
7. Publish broker intent
8. Record signed receipt

---

## Policy Kernel

The policy kernel is explicit and auditable.

**Inputs:**

1. Coherence decision
2. Model outputs
3. Position state
4. Exposure limits
5. Venue constraints
6. Liquidity conditions
7. Market halts or macro blocks

**Rules:**

1. Never place if coherence is unstable
2. Never upsize in regime uncertainty spike
3. Never write memory during adversarial drift burst unless explicitly quarantined
4. Never learn online when realized slippage exceeds bound and cut drift is rising
5. Always throttle actuation when order rate or cancel rate limits approach venue thresholds

---

## Data Retention and Lineage

### Three Tiers

**Hot tier:**

- Recent event graph state
- Recent embeddings
- Recent witness chain
- Active memory reservoir

**Warm tier:**

- Signed replay segments
- Compressed embeddings
- Model evaluation sets
- Daily partition statistics

**Cold tier:**

- Long-horizon archives
- Training corpora
- Promoted model lineage
- Audit snapshots

**Lineage requirements:**

1. Every model maps to training fragments
2. Every live action maps to model and policy version
3. Every mutation maps to a verified token and witness chain
4. Every rollback maps to explicit trigger and prior state hash

---

## RuVector Implementation Details

### Collections

Recommended logical collections:

1. `nt_market_graph`
2. `nt_embeddings_hot`
3. `nt_embeddings_archive`
4. `nt_memory_segments`
5. `nt_policy_receipts`
6. `nt_model_registry`
7. `nt_regime_index`

### Indexing

1. HNSW or RuVector ANN for embedding retrieval
2. Graph neighborhood cache for local subgraph extraction
3. Time-partitioned relational tables in Postgres
4. Quantized serving vectors for low-latency retrieval
5. Optional hyperbolic geometry for regime and hierarchy embeddings

### Retrieval Strategy

Hybrid retrieval score:

```
S = α·similarity + β·structural_overlap + γ·regime_match + δ·coherence_bonus
```

Where:

- `similarity` — vector distance
- `structural_overlap` — graph neighborhood match
- `regime_match` — volatility and spread regime comparison
- `coherence_bonus` — reward for fragments from stable partitions

Weights are constrained: `α + β + γ + δ = 1`. Defaults: `α=0.4, β=0.25, γ=0.2, δ=0.15`. Tuned per regime via walk-forward validation.

---

## Rust Module Layout

```
crates/
  neural-trader-core/         # Event schema, types, ingest
  neural-trader-graph/         # Dynamic heterogeneous market graph
  neural-trader-features/      # Feature extraction and embedding
  neural-trader-memory/        # Two-stage memory selection
  neural-trader-coherence/     # MinCut coherence gate
  neural-trader-policy/        # Policy kernel and risk budgets
  neural-trader-execution/     # Broker adapters, order intent
  neural-trader-replay/        # RVF replay segments, witness logs
  neural-trader-rvf/           # RVF serialization bindings
  neural-trader-server/        # gRPC/HTTP serving layer
```

### Core Traits

```rust
pub trait EventIngestor {
    fn ingest(&mut self, event: MarketEvent) -> anyhow::Result<()>;
}

pub trait GraphUpdater {
    fn apply_event(&mut self, event: &MarketEvent) -> anyhow::Result<GraphDelta>;
}

pub trait Embedder {
    fn embed_state(&self, ctx: &StateWindow) -> anyhow::Result<Vec<f32>>;
}

pub trait MemoryStore {
    fn retrieve(&self, query: &MemoryQuery) -> anyhow::Result<Vec<MemorySegment>>;
    fn maybe_write(
        &mut self,
        seg: MemorySegment,
        gate: &CoherenceDecision,
    ) -> anyhow::Result<bool>;
}

pub trait CoherenceGate {
    fn evaluate(&self, ctx: &GateContext) -> anyhow::Result<CoherenceDecision>;
}

pub trait PolicyKernel {
    fn decide(&self, input: &PolicyInput) -> anyhow::Result<ActionDecision>;
}

pub trait WitnessLogger {
    fn append_receipt(&mut self, receipt: WitnessReceipt) -> anyhow::Result<()>;
}
```

---

## Training Plan

### Offline Phase

1. Ingest historical L2 or L3 streams
2. Build dynamic graph windows
3. Create replay segments
4. Train temporal GNN and retrieval heads
5. Calibrate confidence
6. Validate on walk-forward splits
7. Measure coherence-aware versus non-coherence baselines

### Online Bounded Adaptation

**Allowed:**

1. Calibration updates
2. Retrieval weighting
3. Memory admission thresholds
4. Narrow regime adaptation

**Forbidden without manual promotion:**

1. Major architecture changes
2. Policy kernel changes
3. Risk budget changes
4. Output head rewiring

---

## Evaluation

### Core Metrics

**Prediction:**

1. Fill probability calibration
2. Short-horizon direction AUC
3. Slippage error
4. Realized adverse selection

**Trading:**

1. PnL
2. Sharpe or information ratio
3. Max drawdown
4. Inventory risk
5. Cancel-to-fill quality
6. Venue quality

**Coherence:**

1. Average mincut by regime
2. Partition stability
3. Drift detection precision
4. False-positive gate rate
5. Rollback trigger quality

**Systems:**

1. p50 / p95 / p99 latency
2. Retrieval latency
3. Write amplification
4. Storage growth
5. Witness overhead

---

## Acceptance Criteria

### Phase 1 — Research

1. Replayable end-to-end pipeline
2. Deterministic witness logs
3. Measurable improvement from graph-plus-coherence over price-only baseline
4. Bounded online updates with rollback

### Phase 2 — Paper Trading

1. Stable gate behavior under live feed noise
2. No uncontrolled action bursts
3. No unverified mutations
4. Explainable receipts for every recommendation

### Phase 3 — Live Small Capital

1. Strict exposure limits enforced
2. Slippage within approved band
3. Rollback tested in production shadow mode
4. Daily audit completeness at 100%

---

## Safety and Governance

### Mandatory Controls

1. Notional exposure caps
2. Per-symbol limits
3. Sector or cross-asset correlation caps
4. Order rate and cancel rate caps
5. Slippage budget
6. Venue health checks
7. Market halt awareness
8. Human override and kill switch

### Governance Requirements

1. All policy changes versioned
2. All model promotions signed
3. All live mutations proof-gated
4. All replay sets immutable after seal
5. All exceptions logged with witness chain

---

## Failure Modes

### 1. Regime Shift Masquerading as Edge

**Symptom:** Model confidence rises while execution deteriorates.

**Fix:** Increase weight of coherence gate, reduce online learning scope, quarantine new memory writes.

### 2. Retrieval Poisoning

**Symptom:** Bad fragments dominate replay or inference retrieval.

**Fix:** Signed segment lineage, structural overlap thresholding, memory deprecation, reservoir diversity constraints.

### 3. Feedback Loop with Market Impact

**Symptom:** Strategy reacts to its own footprint.

**Fix:** Actuation throttles, self-impact features, venue split, delayed reinforcement of impacted samples.

### 4. Overfitting to Stable Partitions

**Symptom:** System ignores true novelty.

**Fix:** Maintain novelty quota in memory reservoir, adversarial validation, regime-balanced evaluation.

### 5. Latency Creep

**Symptom:** Graph growth degrades serving time.

**Fix:** Compact local subgraphs, quantized embeddings, hot-path kernels, bounded neighborhood extraction.

---

## Alternatives Considered

### Alternative A: Pure Time-Series Transformer

Over candles and book tensors.

**Rejected:** Ignores explicit queue topology, event causality, and structural integrity.

### Alternative B: Traditional Feature Engineering + Boosted Trees

**Rejected:** Works in narrow slices, but memory, structure, and drift handling remain bolted on rather than native.

### Alternative C: End-to-End RL Trader

**Rejected:** Action-space instability, reward hacking risk, and poor auditability for early deployment.

---

## Consequences

### Positive

1. Unified substrate for data, memory, learning, and governance
2. Explicit structural reasoning over market microstructure
3. Bounded and auditable online learning
4. First-class drift and fragility detection
5. Reproducible replays and mutation receipts

### Negative

1. More complex graph engineering
2. Higher initial systems effort than plain tensor pipelines
3. Policy design must be disciplined
4. Coherence thresholds require calibration by regime

---

## Implementation Plan

### Phase 1 — Foundation

1. Define canonical market event schema
2. Implement RuVector graph projection
3. Implement hot embedding pipeline
4. Implement replay segment writer
5. Implement mincut gate service
6. Implement witness receipts

### Phase 2 — Learning

1. Train baseline GNN plus temporal attention
2. Add retrieval-augmented prediction
3. Add uncertainty scoring
4. Add reservoir memory writer
5. Compare against price-only baseline

### Phase 3 — Bounded Action

1. Implement policy kernel
2. Implement paper trading adapter
3. Add risk budgets and throttles
4. Test rollback
5. Certify live shadow mode

### Phase 4 — Live Research

1. Small capital deployment
2. Conservative venue set
3. Daily audit review
4. Promote only signed models
5. Continuous regime monitoring

---

## Minimal Example Configuration

```yaml
neural_trader:
  symbol_universe:
    - ES
    - NQ
    - CL

  ingest:
    venue_clock_tolerance_ns: 500000
    reorder_buffer_events: 2048

  graph:
    max_local_levels_per_side: 32
    max_orders_per_window: 5000
    neighborhood_hops: 2

  embeddings:
    dim: 256
    quantized_dim: 256
    similarity_metric: cosine

  memory:
    stage_a:
      count_min_width: 4096
      count_min_depth: 4
      topk: 256
    stage_b:
      reservoir_size: 50000
      min_uncertainty: 0.18
      min_realized_impact_bp: 1.5

  coherence:
    mincut_floor_by_regime:
      calm: 12
      normal: 9
      volatile: 6
    cusum_threshold: 4.5
    boundary_stability_windows: 8

  policy:
    max_notional_usd: 250000
    max_symbol_notional_usd: 50000
    max_order_rate_per_sec: 10
    max_cancel_rate_per_sec: 15
    max_slippage_bp: 2.0
    require_verified_token: true

  learning:
    online_mode: bounded
    allow_calibration_updates: true
    allow_memory_write: true
    allow_weight_updates: false

  retention:
    hot_window_hours: 4
    warm_retention_days: 30
    cold_archive_days: 365
    partition_interval_ns: 3600000000000  # 1 hour per partition
    vacuum_schedule_cron: "0 */6 * * *"
```

---

## Decision Summary

Neural Trader will be built as a RuVector-native dynamic market graph system where vectors, graphs, temporal learning, and dynamic mincut work together as one bounded intelligence loop.

The core principle is simple:

> **Do not trust prediction alone. Trust prediction only when the surrounding market structure is coherent enough to justify learning, remembering, or acting.**

That gives us a trader that is not just neural, but **structurally self-aware**.

### Implementation Priority

Best immediate path is three crates first:

1. **`neural-trader-core`** — ingest, canonical types, event schema
2. **`neural-trader-coherence`** — mincut gating, coherence decisions
3. **`neural-trader-replay`** — witnessable segments, RVF integration

That gets ingest, witnessable segments, and mincut gating working before the full model stack is finalized.

**Stretch option:** Adding a Mincut Gated Transformer head for early exit and sparse compute during regime instability.

**Frontier option:** Deploying the coherence gate as a tiny deterministic kernel on Cognitum-style edge nodes or WASM workers so action permission stays cheap, bounded, and independently attestable.

**Benchmark test:** On replay, the coherence-gated model should beat a tensor-only baseline on slippage-adjusted PnL while reducing unstable memory writes and false actuation during regime shifts.
