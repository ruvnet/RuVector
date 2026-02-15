# Chaos Resilience Domain Shard

**Domain**: chaos-resilience
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Hypothesis-Driven Experiments**: All chaos experiments MUST have a defined hypothesis with measurable steady-state indicators; exploratory chaos is prohibited in production.

2. **Blast Radius Control**: Chaos experiments MUST define and enforce blast radius limits; unconstrained fault injection is a constitutional violation.

3. **Rollback Plan Required**: Every chaos experiment MUST have an automated rollback plan with defined triggers; experiments without rollback are rejected.

4. **Production Gating**: Chaos experiments in production MUST pass staging validation first; direct-to-production experiments require explicit approval.

5. **Load Test Baseline**: Performance load tests MUST establish baselines before comparative analysis; baseline-less load tests produce invalid conclusions.

6. **Circuit Breaker Verification**: Resilience testing MUST verify circuit breaker behavior under fault conditions; untested circuit breakers are deployment blockers.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Experiment Success Rate | 0.8 | 0.95 | < 0.6 |
| Recovery Time (MTTR) | < 5 min | < 2 min | > 15 min |
| Blast Radius Control | 100% | 100% | < 90% |
| Load Test Reliability | 0.9 | 0.98 | < 0.8 |
| Circuit Breaker Coverage | 0.8 | 0.95 | < 0.6 |
| Hypothesis Validation | 0.7 | 0.85 | < 0.5 |

---

## Invariants

```
INVARIANT hypothesis_required:
  FOR ALL experiment IN chaos_experiments:
    experiment.hypothesis IS NOT NULL AND
    experiment.steady_state_definition IS NOT NULL AND
    experiment.steady_state_definition.probes.length >= 1
```

```
INVARIANT blast_radius_control:
  FOR ALL experiment IN chaos_experiments:
    experiment.blast_radius IS NOT NULL AND
    experiment.blast_radius.affected_services.length <= experiment.blast_radius.max_services AND
    experiment.blast_radius.affected_users_percent <= experiment.blast_radius.max_users_percent
```

```
INVARIANT rollback_plan_required:
  FOR ALL experiment IN chaos_experiments:
    experiment.rollback_plan IS NOT NULL AND
    experiment.rollback_plan.triggers.length >= 1 AND
    experiment.rollback_plan.automated = true
```

```
INVARIANT staging_before_production:
  FOR ALL prod_experiment IN production_experiments:
    EXISTS staging_run WHERE
      staging_run.experiment_type = prod_experiment.type AND
      staging_run.passed = true AND
      staging_run.timestamp < prod_experiment.timestamp
```

```
INVARIANT load_test_baseline:
  FOR ALL load_test IN comparative_load_tests:
    load_test.baseline_established = true AND
    load_test.baseline_timestamp IS NOT NULL
```

---

## Patterns

**Domain Source**: `v3/src/domains/chaos-resilience/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Chaos Engineer Service | `services/chaos-engineer.ts` | Fault injection orchestration |
| Load Tester Service | `services/load-tester.ts` | Performance load testing |
| Performance Profiler Service | `services/performance-profiler.ts` | Bottleneck detection |
| Chaos Resilience Coordinator | `coordinator.ts` | Workflow orchestration |

**Key Interfaces**: `interfaces/index.ts` defines `ChaosExperiment`, `LoadTest`, `BlastRadius`, `RollbackPlan`, and related types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-chaos-engineer` | Full experiment design and execution |
| **Secondary** | `qe-performance-tester` | Load testing, profiling |
| **Secondary** | `qe-resilience-assessor` | Recovery testing |
| **Support** | `qe-code-analyst` | Service dependency mapping |
| **Approval** | `sre-team` (human) | Production chaos approval |

**Forbidden Actions**: No agent may execute production chaos experiments without staging validation and explicit approval.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Blast radius exceeded | CRITICAL | Immediate rollback, escalate to SRE team |
| MTTR > 15 min | CRITICAL | Halt experiment, escalate to Queen Coordinator |
| Rollback failure | CRITICAL | Manual intervention required, escalate immediately |
| Production chaos without staging | CRITICAL | Block execution, escalate |
| Hypothesis invalidated | HIGH | Escalate for experiment redesign |
| Circuit breaker failure | HIGH | Escalate, flag service for remediation |
| Load test baseline missing | MEDIUM | Block comparative analysis |
| Experiment success rate < 0.6 | HIGH | Pause experiments, review design |

---

## Memory Namespace

- **Namespace**: `qe-patterns/chaos-resilience`
- **Retention**: 90 days (experiment results retained longer)
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `code-intelligence` | Input | Service dependency graphs |
| `quality-assessment` | Output | Report resilience scores |
| `test-execution` | Bidirectional | Execute resilience tests |
| `defect-intelligence` | Output | Report failure patterns |
| `learning-optimization` | Bidirectional | Share resilience patterns |

---

## Fault Types

| Fault Type | Description | Blast Radius Risk |
|------------|-------------|-------------------|
| network-latency | Add latency to network calls | Medium |
| network-partition | Isolate services | High |
| cpu-stress | Exhaust CPU resources | Medium |
| memory-pressure | Consume memory | Medium |
| disk-io | Slow disk operations | Low |
| process-kill | Terminate processes | High |
| dns-failure | Break DNS resolution | High |
| clock-skew | Advance/retard system clock | Low |

---

## Load Test Types

| Type | Description | Use Case |
|------|-------------|----------|
| smoke | Basic functionality under load | Pre-deployment validation |
| load | Normal expected load | Baseline establishment |
| stress | Beyond normal load | Find breaking points |
| spike | Sudden traffic increase | Autoscaling validation |
| soak | Extended duration load | Memory leak detection |

---

## Rollback Trigger Schema

```typescript
interface RollbackTrigger {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
  threshold: number;
  duration: string;
  action: 'rollback' | 'pause' | 'alert';
  automatic: boolean;
}
```

---

## Blast Radius Definition

```typescript
interface BlastRadius {
  maxServices: number;
  maxUsersPercent: number;
  excludedServices: string[];
  geographicScope: 'single-region' | 'multi-region' | 'global';
  timeWindow: {
    start: string;
    end: string;
    timezone: string;
  };
  trafficPercentage: number;
}
```

---

## GameDay Protocol

1. **Planning**: Define hypothesis, metrics, rollback criteria
2. **Communication**: Notify all stakeholders 24h before
3. **Staging Run**: Execute in staging environment
4. **Review**: Analyze staging results
5. **Production Run**: Execute with blast radius controls
6. **Monitoring**: Real-time observation dashboard
7. **Rollback**: If triggers activated, automatic rollback
8. **Post-mortem**: Document findings, update resilience patterns

---

*This shard is enforced by @claude-flow/guidance governance system.*
