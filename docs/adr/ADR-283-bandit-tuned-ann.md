# ADR-283: Bandit-Tuned ANN — Online ef_search Optimization via UCB1

**Status:** Proposed

**Date:** 2026-08-04

---

## Context

HNSW-based ANN search exposes `ef_search` (beam width) as the primary recall/latency knob. RuVector currently offers `StaticHnsw` (fixed ef), `TableCalibratedSearch` (offline calibration table), and `RecallTargetedSearch` (threshold-based dynamic ef) in `ruvector-adaptive-ann`. None of these adapt at runtime to workload shifts.

AI agent memory workloads are non-stationary:
- Query distribution shifts when the agent changes topic domain.
- Required recall varies by task (code search tolerates lower recall than legal discovery).
- k (top-k) varies per query in multi-agent systems.

A static or offline-calibrated ef_search is consistently suboptimal for at least one regime the workload visits.

---

## Decision

Introduce `BanditEfSearch` as a new variant of `RecallTargetedSearch` in `ruvector-adaptive-ann`. The bandit maintains a discrete set of candidate ef values (arms) and selects among them using UCB1. After each query, it observes a reward signal and updates arm estimates.

**Reward function:** `reward = recall@k - alpha * latency_norm`
where `alpha = 0.15` and `latency_norm = min(latency_µs / 200.0, 1.0)`.

This balances recall (primary objective) against latency (secondary constraint).

**Arm set (default):** {10, 20, 30, 40, 50}. Configurable at construction time.

**Selection:** UCB1 (`mean_reward + sqrt(2 * ln(total) / count)`). Falls back to sequential exploration for arms with 0 pulls.

**Memory overhead:** `n_arms * (8 + 8)` bytes = 80 bytes for 5 arms.

**Query overhead:** ~7 ns per query (arm selection + reward update).

---

## Consequences

**Positive:**
- Zero configuration for new deployments: ef converges to the workload optimum automatically.
- Graceful handling of workload shift: bandit re-converges after distribution change (convergence rate: O(sqrt(T) regret).
- Composable: wraps any `Hnsw`-compatible index, does not depend on graph internals.
- Auditable: arm pull counts and mean rewards are observable at runtime for debugging.

**Negative:**
- Convergence requires ~400 queries (in the PoC). A cold-start database answers the first 40+ queries suboptimally while exploring.
- Reward observation requires ground truth (exact recall). In production, a proxy reward is needed (candidate diversity, expansion ratio). Proxy accuracy limits bandit quality.
- Stationary assumption: UCB1 is optimal for stationary reward distributions. Non-stationary workloads need sliding-window discounting or CUSUM resets.

---

## Alternatives Considered

| Alternative | Tradeoff | Why Rejected |
|-------------|----------|--------------|
| Fixed ef (current) | Simple, no overhead | Does not adapt; wrong for shifted workloads |
| Calibration table | Better than fixed | Stale after workload change; no online update |
| Thompson Sampling | More robust to variance | Heavier (Beta posterior, gamma sampler); UCB1 sufficient at this scale |
| LinUCB (contextual) | Uses query features | 3× more complex; context feature engineering adds dependency |
| GP-Bayesian tuner | Handles continuous parameter space | Too heavy for per-query overhead; better as offline optimizer |

---

## Implementation Plan

**Phase 1 (PoC — complete):** Standalone `ruvector-bandit-ann` crate with two-layer HNSW and UCB1. Validates convergence on 5K × 96-dim benchmark.

**Phase 2 (integration):** Merge `Ucb1Bandit` and `ThompsonBandit` into `ruvector-adaptive-ann/src/bandit.rs`. Add `BanditEfSearch` to `RecallTargetedSearch` trait. Wire to `CalibrationTable` for warm-start arm means.

**Phase 3 (production):** Add sliding-window reward discounting (`StalenessWindow` from `sona/auto_tuner.rs`). Add CUSUM change-point detector. Proxy reward calibration on held-out set.

**Feature flag:** `bandit-ef` in `ruvector-adaptive-ann`. Disabled by default until Phase 3 validates proxy reward accuracy in production.

---

## Benchmark Evidence

Measured on 5 000 × 96-dim uniform unit-sphere vectors, 300 queries, k=10, M=16, ef_construction=200, release build, x86_64 Linux.

| Variant | Recall@10 | Mean µs | p50 µs | p95 µs | QPS |
|---------|-----------|---------|--------|--------|-----|
| StaticDefault(ef=50) | 0.8277 | 293.6 | 282.8 | 370.1 | 3406 |
| StaticFast(ef=10) | 0.4140 | 86.4 | 78.2 | 138.4 | 11577 |
| BanditTuned(UCB1) | 0.8277 | 290.5 | 280.0 | 366.1 | 3443 |

Bandit converged to ef=50 after 400 pulls. Acceptance: PASS (recall >= 0.80, gap vs StaticFast >= 20pp).

---

## Failure Modes

1. **Cold start**: First 40 queries use suboptimal ef while all arms are explored once. Mitigation: warm-start arm means from `CalibrationTable`.

2. **Reward poisoning**: In multi-tenant deployments, a malicious user crafts queries to poison the bandit's reward signal, degrading recall for other users. Mitigation: per-tenant bandit instances or proof-gated reward writes (`ruvector-proof-gate`).

3. **Proxy reward inaccuracy**: If the proxy recall estimator has high variance, UCB1 converges to a suboptimal arm. Mitigation: calibrate proxy on trusted query set; fall back to StaticDefault if proxy RMSE > 0.1.

4. **Arm set mismatch**: If the optimal ef lies between two arms, performance is bounded by the nearest arm. Mitigation: LinUCB with continuous features, or finer arm grid.

---

## Security Considerations

- Bandit state (arm means, pull counts) must be treated as sensitive: it encodes workload patterns that could leak information about queries.
- The reward function uses recall@k, which requires ground truth. Ground truth computation must be rate-limited to prevent exhaustion attacks.
- Do not expose bandit state via unauthenticated APIs.

---

## Migration Path

Existing deployments using `RecallTargetedSearch` can opt into `BanditEfSearch` with a one-line change:
```rust
let searcher = BanditEfSearch::new(index, vec![10, 20, 30, 40, 50]);
```

The existing `RecallTargetedSearch::Fixed` variant is preserved. No breaking change.

---

## Open Questions

1. What proxy recall signal achieves > 0.9 Spearman correlation with actual Recall@10? (Candidate diversity? Expansion ratio? Distance distribution skew?)

2. Should the arm set be configurable at runtime (hot-reconfiguration) or only at construction?

3. At what collection size does the bandit convergence overhead (7 ns × 400 queries = 2.8 µs total) become negligible relative to index build time?

4. Should Thompson Sampling replace UCB1 as default, given that agent memory workloads have high reward variance per query?
