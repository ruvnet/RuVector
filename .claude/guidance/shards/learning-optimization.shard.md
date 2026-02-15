# Learning Optimization Domain Shard

**Domain**: learning-optimization
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Pattern Validation Before Sharing**: Learned patterns MUST be validated (minimum success rate 0.7) before cross-domain sharing; unvalidated patterns pollute other domains.

2. **Experience Source Verification**: Experiences used for learning MUST have verified outcomes; learning from incomplete or assumed results is prohibited.

3. **Knowledge Synthesis Traceability**: Synthesized knowledge MUST maintain provenance links to source experiences; orphan knowledge is not actionable.

4. **Strategy A/B Testing**: New strategies MUST be A/B tested against baselines before full adoption; untested strategy changes are high-risk.

5. **Model Export Integrity**: Exported learning models MUST include version, training data summary, and validation metrics; metadata-free exports are rejected.

6. **Cross-Domain Conflict Resolution**: When patterns from different domains conflict, the domain with higher confidence and more recent validation wins.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Pattern Success Rate | 0.7 | 0.85 | < 0.5 |
| Experience Verification Rate | 0.9 | 0.98 | < 0.8 |
| Knowledge Provenance | 1.0 | 1.0 | < 0.9 |
| Strategy A/B Confidence | 0.95 | 0.99 | < 0.9 |
| Cross-Domain Sharing Rate | 0.3 | 0.5 | N/A |
| Learning Velocity | > 0 | > 0.1/week | < 0 |

---

## Invariants

```
INVARIANT pattern_validation_before_sharing:
  FOR ALL shared_pattern IN cross_domain_patterns:
    shared_pattern.success_rate >= 0.7 AND
    shared_pattern.validation_count >= 3 AND
    shared_pattern.last_validated IS NOT NULL
```

```
INVARIANT experience_verification:
  FOR ALL experience IN learning_experiences:
    experience.outcome_verified = true AND
    experience.verification_method IN ['test_execution', 'human_review', 'automated_check']
```

```
INVARIANT knowledge_provenance:
  FOR ALL knowledge IN synthesized_knowledge:
    knowledge.source_experiences.length >= 1 AND
    FOR ALL source IN knowledge.source_experiences:
      source.experience_id IS NOT NULL
```

```
INVARIANT strategy_ab_testing:
  FOR ALL new_strategy IN adopted_strategies:
    EXISTS ab_test WHERE
      ab_test.strategy_id = new_strategy.id AND
      ab_test.completed = true AND
      ab_test.confidence >= 0.95 AND
      ab_test.new_strategy_wins = true
```

```
INVARIANT model_export_integrity:
  FOR ALL export IN model_exports:
    export.version IS NOT NULL AND
    export.training_data_summary IS NOT NULL AND
    export.validation_metrics IS NOT NULL AND
    export.created_at IS NOT NULL
```

---

## Patterns

**Domain Source**: `v3/src/domains/learning-optimization/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Learning Coordinator Service | `services/learning-coordinator.ts` | Cross-domain orchestration |
| Transfer Specialist Service | `services/transfer-specialist.ts` | Pattern transfer between domains |
| Metrics Optimizer Service | `services/metrics-optimizer.ts` | Strategy optimization |
| Production Intel Service | `services/production-intel.ts` | Production feedback integration |
| Learning Optimization Coordinator | `coordinator.ts` | Workflow orchestration |

**Key Interfaces**: `interfaces/index.ts` defines `LearnedPattern`, `Experience`, `Knowledge`, `Strategy`, and related types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-learning-coordinator` | Full learning orchestration |
| **Secondary** | `qe-pattern-learner` | Pattern extraction, validation |
| **Secondary** | `qe-transfer-specialist` | Cross-domain pattern transfer |
| **Secondary** | `qe-metrics-optimizer` | Strategy optimization |
| **Support** | All domain agents | Report experiences |
| **Readonly** | `qe-quality-gate` | Query learning status |

**Forbidden Actions**: No agent may share unvalidated patterns across domains.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Learning velocity < 0 (regression) | CRITICAL | Escalate to Queen Coordinator, investigate |
| Pattern success rate < 0.5 | HIGH | Quarantine pattern, escalate |
| Unverified experience used | HIGH | Rollback learning, escalate |
| Knowledge provenance broken | HIGH | Quarantine knowledge, reconstruct |
| A/B test confidence < 0.9 | MEDIUM | Extend test duration |
| Cross-domain conflict unresolved | MEDIUM | Escalate to Queen Coordinator |
| Model export without metadata | MEDIUM | Block export, request metadata |
| Experience verification rate < 0.8 | MEDIUM | Audit verification pipeline |

---

## Memory Namespace

- **Namespace**: `qe-patterns/learning-optimization`
- **Retention**: 180 days (patterns), 90 days (experiences)
- **Contradiction Check**: Enabled (critical for learning consistency)

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| All domains | Input | Receive experiences and outcomes |
| All domains | Output | Share validated patterns |
| `quality-assessment` | Output | Report learning metrics |
| `defect-intelligence` | Bidirectional | Share defect patterns |
| `coverage-analysis` | Bidirectional | Share coverage patterns |

---

## Pattern Types

| Type | Description | Sharing Scope |
|------|-------------|---------------|
| test-pattern | Effective test strategies | test-generation, test-execution |
| coverage-pattern | Gap detection strategies | coverage-analysis |
| defect-pattern | Defect prediction indicators | defect-intelligence |
| security-pattern | Security testing strategies | security-compliance |
| performance-pattern | Load test configurations | chaos-resilience |
| meta-pattern | Cross-domain optimizations | All domains |

---

## Experience Recording Schema

```typescript
interface Experience {
  id: string;
  domain: string;
  action: string;
  state: StateSnapshot;
  result: ExperienceResult;
  outcome: {
    verified: boolean;
    verificationMethod: 'test_execution' | 'human_review' | 'automated_check';
    success: boolean;
    metrics: Record<string, number>;
  };
  timestamp: Date;
  agentId: string;
  sessionId: string;
  resourceUsage: ResourceUsage;
}
```

---

## Strategy A/B Testing Protocol

```typescript
interface ABTestConfig {
  strategyId: string;
  baselineStrategyId: string;
  trafficSplit: number; // 0.5 = 50/50
  minSampleSize: number;
  targetMetric: string;
  confidenceLevel: number; // 0.95 default
  maxDuration: string; // "7d" default
  earlyStoppingEnabled: boolean;
}
```

---

## Cross-Domain Conflict Resolution

| Priority | Factor | Weight |
|----------|--------|--------|
| 1 | Confidence score | 0.4 |
| 2 | Recency of validation | 0.3 |
| 3 | Sample size | 0.2 |
| 4 | Domain authority | 0.1 |

When patterns conflict:
1. Calculate weighted score for each pattern
2. Pattern with highest score wins
3. Losing pattern marked as superseded
4. Supersession recorded in memory with explanation

---

## Learning Cycle Report Schema

```typescript
interface LearningCycleReport {
  cycleId: string;
  period: {
    start: Date;
    end: Date;
  };
  experiencesRecorded: number;
  experiencesVerified: number;
  patternsLearned: number;
  patternsShared: number;
  strategiesOptimized: number;
  abTestsCompleted: number;
  learningVelocity: number;
  domainBreakdown: Record<string, {
    experiences: number;
    patterns: number;
    improvements: number;
  }>;
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
