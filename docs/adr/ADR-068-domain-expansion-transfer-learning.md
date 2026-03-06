# ADR-068: Domain Expansion Transfer Learning

**Status**: Accepted, Implemented
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-057 (Federated RVF Transfer Learning), ADR-060 (Shared Brain Capabilities), ADR-061 (Reasoning Kernel Architecture)

## 1. Context

The Shared Brain (ADR-060) enables cumulative learning within a single domain. But real intelligence growth appears when knowledge from one domain accelerates learning in a different domain. A debugging heuristic from Rust should inform debugging in TypeScript. A planning strategy from infrastructure deployment should transfer to release management.

The `ruvector-domain-expansion` crate (`crates/ruvector-domain-expansion/`) implements cross-domain transfer learning using Meta Thompson Sampling with compact prior transfer. The core insight: true generalization is measured by whether Domain 2 converges faster than Domain 1 did, given priors extracted from Domain 1. If cost curves compress with each new domain, general problem-solving capability is increasing.

## 2. Decision

Implement a two-layer architecture: a policy learning layer (Meta Thompson Sampling with Beta priors) that chooses strategies across context buckets, and an operator layer (deterministic domain kernels) that generates tasks, evaluates solutions, and produces embeddings. Transfer happens through compact priors — not raw trajectories. Verification requires dual conditions: improved target AND not regressed source.

## 3. Architecture

### 3.1 Two-Layer Design

**Policy Learning Layer** (`MetaThompsonEngine`):
- Maintains per-domain, per-bucket, per-arm Beta distribution parameters
- Selects strategy arms via Thompson Sampling (sample from posterior, pick highest)
- Records outcomes as Bayesian posterior updates
- Extracts compact `TransferPrior` summaries for cross-domain shipping
- Seeds new domains with dampened priors from source domains

**Operator Layer** (Domain implementations):
- `RustSynthesisDomain`: Generates Rust function synthesis tasks, evaluates correctness
- `PlanningDomain`: Generates multi-step planning tasks with dependencies and resources
- `ToolOrchestrationDomain`: Generates multi-tool coordination tasks
- Each domain implements `generate_tasks()`, `evaluate()`, `embed()`, `reference_solution()`

### 3.2 Meta Thompson Sampling

Standard Thompson Sampling maintains a Beta(alpha, beta) distribution per arm and samples to select. Meta Thompson Sampling extends this across domains:

1. **Train on Domain 1**: Record outcomes, accumulate per-bucket posteriors
2. **Extract TransferPrior**: Filter to buckets with sufficient evidence (alpha + beta > 12), package as a compact summary
3. **Initialize Domain 2**: Seed with dampened priors from Domain 1 using sqrt-scaling
4. **Train on Domain 2**: The transferred priors give Domain 2 a head start
5. **Measure acceleration**: Compare convergence cycles with vs without transfer

### 3.3 Dampened Sqrt-Scaling Priors

When transferring priors from source to target, raw posteriors would over-commit the target to source-domain strategies. Dampening uses sqrt-scaling to reduce confidence while preserving the mean:

```rust
let dampened = BetaParams {
    alpha: 1.0 + (params.alpha - 1.0).sqrt(),
    beta: 1.0 + (params.beta - 1.0).sqrt(),
};
```

Example: A source prior of Beta(81, 21) (mean=0.79, very confident) becomes Beta(1 + sqrt(80), 1 + sqrt(20)) = Beta(9.94, 5.47) (mean=0.64, much less confident). The target retains the directional signal (strategy A is probably better than strategy B) but has enough uncertainty to adapt if the source signal does not transfer.

Cost EMA priors are transferred with pessimistic scaling (1.5x) to avoid under-budgeting in an unfamiliar domain.

### 3.4 TransferPrior and PolicyKernel RVF Segments

Transfer artifacts are serialized as RVF segments for the Shared Brain:

| Segment | Type Code | Content |
|---------|-----------|---------|
| `TransferPrior` | `0x30` | Per-bucket, per-arm Beta parameters + cost EMA priors + training cycle count + witness hash |
| `PolicyKernel` | `0x31` | Policy knobs (skip mode, prepass flag, speculation threshold) + holdout scores + fitness |
| `CostCurve` | `0x32` | Ordered data points (cycle, accuracy, cost_per_solve, robustness, violations) + convergence thresholds |

The `rvf_bridge` module (enabled with `feature = "rvf"`) handles serialization via wire-format wrappers that convert `HashMap<ContextBucket, _>` to `Vec<(K, V)>` for JSON-safe encoding. SHAKE-256 witness chains cover the entire serialization for integrity.

### 3.5 CostCurve Acceleration Tracking

The `AccelerationScoreboard` tracks convergence speed across domains:

```rust
pub struct CostCurvePoint {
    pub cycle: u64,
    pub accuracy: f32,
    pub cost_per_solve: f32,
    pub robustness: f32,
    pub policy_violations: u32,
    pub timestamp: f64,
}
```

Convergence thresholds define the acceptance test:
- Target accuracy: 0.95
- Target cost per solve: 0.01
- Target robustness: 0.90
- Max policy violations: 0

A domain has converged when all four thresholds are simultaneously met. The acceleration factor is `baseline_cycles / transfer_cycles` — the ratio of how many cycles the target took without transfer vs with transfer. An acceleration factor > 1.0 means transfer helped.

### 3.6 TransferVerification

The verification protocol enforces the generalization rule:

```rust
impl TransferVerification {
    pub fn verify(
        source: DomainId,
        target: DomainId,
        source_before: f32,
        source_after: f32,   // must not regress beyond 0.01 tolerance
        target_before: f32,
        target_after: f32,   // must improve
        baseline_cycles: u64,
        transfer_cycles: u64,
    ) -> Self {
        let improved_target = target_after > target_before;
        let regressed_source = source_after < source_before - 0.01;
        let promotable = improved_target && !regressed_source;
        let acceleration_factor = baseline_cycles as f32 / transfer_cycles as f32;
        // ...
    }
}
```

A transfer delta is promotable only when:
1. `improved_target`: The target domain's score increased after applying the transfer
2. `NOT regressed_source`: The source domain's score did not decrease beyond a 0.01 tolerance

Both conditions must hold. This prevents transfers that help the target at the expense of corrupting the source.

### 3.7 Holdout Evaluation Protocol

Transfers are validated on held-out tasks, not training tasks:

1. `generate_holdouts(tasks_per_domain, difficulty)` creates holdout task sets per domain
2. `evaluate_population()` runs all policy kernels against holdout tasks
3. Holdout scores are recorded per kernel per domain
4. `evolve_population()` selects top performers, mutates, records Pareto front

The holdout set is never used for training. This prevents overfitting to the evaluation metric.

## 4. Implementation

### 4.1 DomainExpansionEngine

The central orchestrator:

```rust
pub struct DomainExpansionEngine {
    domains: HashMap<DomainId, Box<dyn Domain>>,
    pub thompson: MetaThompsonEngine,
    pub population: PopulationSearch,
    pub scoreboard: AccelerationScoreboard,
    pub meta: MetaLearningEngine,
    holdouts: HashMap<DomainId, Vec<Task>>,
    counterexamples: HashMap<DomainId, Vec<(Task, Solution, Evaluation)>>,
}
```

Initialized with three domains: `rust_synthesis`, `structured_planning`, `tool_orchestration`. Four strategy arms: `greedy`, `exploratory`, `conservative`, `speculative`. Three difficulty tiers: `easy`, `medium`, `hard`.

### 4.2 Transfer Flow

```rust
// 1. Record outcomes in source domain
engine.evaluate_and_record(&source, task, solution, bucket, arm);

// 2. Initiate transfer (extracts dampened priors)
engine.initiate_transfer(&source, &target);

// 3. Train on target domain (uses transferred priors)
engine.evaluate_and_record(&target, task, solution, bucket, arm);

// 4. Verify the transfer
let verification = engine.verify_transfer(
    &source, &target,
    source_before, source_after,
    target_before, target_after,
    baseline_cycles, transfer_cycles,
);
assert!(verification.promotable);
```

### 4.3 Population-Based Policy Search

A population of 8 `PolicyKernel` variants runs in parallel. Each kernel tunes strategy knobs:

- Skip mode (whether to skip low-confidence contexts)
- Prepass flag (whether to run a fast prepass before full evaluation)
- Speculation threshold (when to trigger dual-path execution)

Evolution: evaluate all kernels on holdouts, record in Pareto front (accuracy vs cost vs robustness), keep top performers, mutate, increment generation.

### 4.4 Meta-Learning Engine

Five composable improvements layered on top of Thompson Sampling:

| Component | Purpose |
|-----------|---------|
| `RegretTracker` | Measures cumulative regret vs oracle policy |
| `DecayingBeta` | Time-decays Beta parameters for non-stationary environments |
| `PlateauDetector` | Detects when cost curve flattens, suggests action (increase difficulty, transfer, restart) |
| `ParetoFront` | Maintains non-dominated set of (accuracy, cost, robustness) |
| `CuriosityBonus` | UCB-style exploration bonus for under-visited bucket/arm combinations |

### 4.5 Integration with Shared Brain

The `DomainExpansionEngine` is instantiated in the `mcp-brain-server`'s `AppState`:

```rust
let domain_engine = Arc::new(parking_lot::RwLock::new(
    ruvector_domain_expansion::DomainExpansionEngine::new(),
));
```

The `POST /v1/transfer` REST endpoint and the `brain_transfer` MCP tool invoke `engine.initiate_transfer()` and return the `TransferVerification` result.

### 4.6 Counterexample Tracking

Solutions scoring below 0.3 are stored as counterexamples per domain. These serve two purposes:
1. Negative examples for future strategy selection (avoid strategies that produced poor results in similar contexts)
2. Diagnostic data for understanding domain boundaries where transfer fails

## 5. Consequences

### Positive

- **Measurable generalization**: The acceleration scoreboard provides a quantitative answer to "is the system getting smarter across domains?"
- **Safe transfer**: Dampened priors prevent over-commitment to source-domain strategies. Dual verification prevents source regression.
- **Compact transfer artifacts**: TransferPriors are small (per-bucket Beta parameters, not raw trajectories). They serialize to a few KB as RVF segments.
- **Composable meta-learning**: Regret tracking, plateau detection, Pareto optimization, and curiosity bonuses layer independently and can be enabled/disabled per deployment.

### Negative

- **Three domains only**: The current implementation has three hard-coded domains (Rust synthesis, planning, tool orchestration). Adding new domains requires implementing the `Domain` trait and registering with the engine.
- **No online transfer**: Transfer is initiated manually via `initiate_transfer()`. Automatic transfer triggering (e.g., when a domain's cost curve plateaus) is deferred.
- **Population size fixed**: The population search uses 8 kernels. Tuning population size requires code changes.

### Neutral

- The 0.01 tolerance on source regression allows minor noise in evaluation scores without blocking transfers. This is a practical trade-off — evaluation noise from holdout sampling can cause small score fluctuations that are not true regressions.
- Counterexamples grow unboundedly per domain. A pruning strategy (keep top-N by recency or informativeness) is deferred.
