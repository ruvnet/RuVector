# ADR-272: Adaptive ef-Search Control for HNSW

**Status:** Proposed  
**Date:** 2026-07-05  
**Deciders:** ruvnet/ruvector nightly research  
**Supersedes:** None  
**Related:** ADR-254 (coherence-HNSW), ADR-268 (capability-gated ANN), ADR-264 (LSM-ANN)

---

## Context

HNSW approximate nearest-neighbour search requires a beam width parameter `ef_search`. This scalar controls the recall-latency tradeoff: higher ef yields better recall but takes longer. In every production deployment — Qdrant, Milvus, pgvector, Weaviate — this parameter is set statically by the operator.

This is inadequate for agentic workloads:

1. **Agent contexts vary**. A voice assistant answering a user question needs sub-100µs retrieval. A background memory consolidation job tolerates 2ms. Same index, different SLAs.
2. **Load varies**. Under low load, unused latency budget could improve recall. Under high load, ef should shrink to maintain throughput.
3. **ruFlo workflows have declared budgets**. A ruFlo node can declare `latency_budget_us`; today there is no mechanism to enforce that declaration at the HNSW level.
4. **Edge devices differ**. A Raspberry Pi 4 and a Jetson Orin have different throughput limits; the optimal ef differs by hardware without a static way to discover it.

---

## Decision

Introduce `ruvector-adaptive-ef`, a standalone Rust crate that:

1. Defines a `SearchPolicy` trait with `recommend_ef(budget_us) -> u32` and `observe(latency_us, recall, ef_used)`.
2. Ships four implementations: `FixedPolicy` (baseline), `EwmaGreedy` (exponential moving average hill-climb), `BanditPolicy` (ε-greedy multi-armed bandit over discrete ef levels), and `PidController` (proportional-integral-derivative control on latency error).
3. Provides a benchmark binary that measures all four policies against a deterministic single-layer k-NN graph simulator.

The `SearchPolicy` trait is the API surface that production HNSW integration will use. The simulator is a test-bed; it is not shipped to production.

---

## Consequences

### Positive

- Agent code can declare a latency budget once; the policy adapts ef transparently.
- Bandit policy discovers per-hardware optimal ef without offline benchmarking.
- PID controller provides theoretically grounded latency SLA enforcement.
- Zero external dependencies; WASM-safe (FixedPolicy and BanditPolicy).
- 14.5 percentage-point recall improvement over conservatively-set Fixed(ef=64) within the same latency budget.

### Negative

- Two additional call sites per query (recommend_ef + observe) add ~100ns overhead.
- Policy state is mutable; must be per-thread or wrapped in Mutex for concurrent callers.
- Recall estimation in production requires either ground truth (expensive) or a shadow-search heuristic (approximate).

### Neutral

- This ADR does not change the HNSW graph structure, storage format, or recall behaviour. It only controls which ef value is passed to existing search code.

---

## Alternatives Considered

### A. Static ef with operator tuning guide
The status quo. Rejected because it does not adapt to context-varying agent workloads and requires manual intervention per deployment.

### B. Reinforcement learning offline policy
Train an RL policy on historical query logs. Rejected for nightly because it requires training infrastructure not present in RuVector today. Remains a research direction (see Open Questions).

### C. Percentile-based reactive controller
Instead of EWMA mean, target p95 latency. Not implemented in this nightly but noted as a natural next step for high-variance latency distributions.

### D. Cost model from index metadata
Predict latency from index size, ef, and hardware specs without runtime measurement. Rejected because it requires calibration per hardware and does not adapt to runtime load.

---

## Implementation Plan

**Phase 1 (this nightly): Proof of concept**
- `crates/ruvector-adaptive-ef/` with trait, four policies, HNSW simulator, benchmark binary.
- All unit tests and benchmark pass.

**Phase 2 (next sprint): Integration**
- Integrate `SearchPolicy` into `ruvector-core` HNSW search path.
- Add `ef_policy: Box<dyn SearchPolicy>` field to `HnswIndex` configuration.
- Default: `FixedPolicy(ef_search)` preserves backward compatibility.

**Phase 3 (production hardening):**
- Per-tenant policy isolation.
- Recall estimator (shadow search or duplicate-query heuristic).
- WASM build target for Cognitum Seed.
- MCP tool: `ruvector_set_search_budget(latency_us, recall_floor)`.
- ruFlo `SearchBudgetNode` wrapper.

---

## Benchmark Evidence

Measured on: Intel Xeon @ 2.10GHz, Ubuntu 24.04.4 LTS, Rust 1.94.1  
Dataset: N=3,000, dim=64, M=16 neighbours, K=10, 500 queries, budget=400µs

| Policy | Mean(µs) | p95(µs) | QPS | Recall@10 | FinalEf | Converged |
|--------|----------|---------|-----|-----------|---------|-----------|
| Fixed(ef=64) | 70.0 | 85 | 14,278 | 0.850 | 64 | NO |
| EwmaGreedy | 254.5 | 282 | 3,929 | 0.995 | 512 | YES |
| Bandit | 166.9 | 194 | 5,991 | 0.966 | 256 | YES |
| PID | 254.3 | 283 | 3,932 | 0.994 | 512 | YES |

Acceptance: all adaptive policies recall ≥ 0.70 ✓, tail latency ≤ 130% budget ✓.

The Fixed policy uses 17.5% of the available latency budget and accepts 15 pp lower recall. Bandit is the Pareto-efficient choice at this dataset scale.

---

## Failure Modes

1. **High-variance latency** (σ > 100µs): EWMA and PID may oscillate. Mitigation: use Bandit (arm averages smooth variance) or a percentile-based controller.
2. **Budget < noise floor** (~10–50µs on loaded servers): All policies converge to EF_MIN. Mitigation: use Fixed(EF_MIN) explicitly.
3. **Concurrent callers**: Policy state is mutable. Mitigation: one policy per thread, or wrap in `std::sync::Mutex`.
4. **Distribution shift**: New document batch changes optimal ef. Mitigation: periodic policy reset (clear arm counts, reset EWMA).

---

## Security Considerations

- ef value reveals budget; in multi-tenant systems, don't expose policy state across tenants.
- Rate-limit ef-down steps to prevent adversarial latency inflation attacks.
- In proof-gated deployments, log ef used per query in the witness record.

---

## Migration Path

Existing `ruvector-core` callers pass `ef_search` as a `u32`. Migration:

```rust
// Before
let results = index.search(&query, k, ef_search);

// After (backward compatible default)
let mut policy = FixedPolicy::new(ef_search);
let ef = policy.recommend_ef(u64::MAX); // no budget → same as before
let results = index.search(&query, k, ef);
policy.observe(elapsed_us, 1.0, ef);
```

Existing behaviour is preserved by default. Opt-in to adaptive policies by changing `FixedPolicy` to `BanditPolicy` or `PidController`.

---

## Open Questions

1. Should `SearchPolicy` carry a `recall_floor: f32` that prevents ef from dropping below the floor even if the budget demands it?
2. Should the bandit use UCB1 instead of ε-greedy for stronger theoretical guarantees?
3. How should policy state be serialized for checkpoint/restore in long-running ruFlo workflows?
4. Is a per-query recall estimator (shadow search) worth the 2× latency overhead on a subset of queries?
5. Should `PidController` target p95 instead of mean latency for SLA compliance?
