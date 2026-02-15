# Quality Assessment Domain Shard

**Domain**: quality-assessment
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Coherence-Gated Decisions**: All quality gate pass/fail decisions MUST pass through the CoherenceGateService (ADR-030) to ensure consistency with project quality standards.

2. **Multi-Metric Assessment**: Quality scores MUST consider multiple metrics (coverage, complexity, defect density, security posture), not single-metric decisions.

3. **Deployment Advisory Accuracy**: Deployment recommendations MUST be based on historical deployment success correlation, with tracked accuracy metrics.

4. **No Override Without Audit**: Quality gate overrides MUST be logged with justification, approver identity, and timestamp in the audit trail.

5. **Threshold Transparency**: All quality thresholds MUST be documented and visible to development teams; hidden thresholds are prohibited.

6. **Trend-Aware Decisions**: Quality gate decisions MUST consider trend direction (improving vs. degrading) in addition to absolute values.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Overall Quality Score | 0.7 | 0.85 | < 0.5 |
| Confidence | 0.7 | 0.9 | < 0.5 |
| Coverage | 0.6 | 0.8 | < 0.4 |
| Complexity Index | < 15 | < 10 | > 25 |
| Defect Density | < 0.1 | < 0.05 | > 0.2 |
| Security Score | 0.8 | 0.95 | < 0.6 |

---

## Invariants

```
INVARIANT coherence_gated_decisions:
  FOR ALL gate_decision IN quality_gate_decisions:
    EXISTS coherence_check WHERE
      coherence_check.decision_id = gate_decision.id AND
      coherence_check.passed = true
```

```
INVARIANT multi_metric_assessment:
  FOR ALL quality_report IN reports:
    quality_report.metrics.length >= 3 AND
    quality_report.metrics INCLUDES ['coverage', 'complexity', 'security']
```

```
INVARIANT override_audit_trail:
  FOR ALL override IN quality_gate_overrides:
    override.justification IS NOT NULL AND
    override.approver_id IS NOT NULL AND
    override.timestamp IS NOT NULL AND
    override.audit_logged = true
```

```
INVARIANT threshold_visibility:
  FOR ALL threshold IN active_thresholds:
    threshold.documented = true AND
    threshold.visibility = 'public'
```

---

## Patterns

**Domain Source**: `v3/src/domains/quality-assessment/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Quality Gate Service | `services/quality-gate.ts` | Core gate evaluation |
| Quality Analyzer Service | `services/quality-analyzer.ts` | Multi-metric analysis |
| Deployment Advisor Service | `services/deployment-advisor.ts` | Deploy recommendations |
| Coherence Gate Service | `services/coherence-gate.ts` | ADR-030 coherence validation |
| Coherence Module | `coherence/index.ts` | Coherence algorithms |

**ADR Reference**: ADR-030 defines coherence-gated quality gates.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-quality-gate` | Full gate evaluation, override approval |
| **Secondary** | `qe-deployment-advisor` | Deployment recommendations |
| **Support** | `qe-coverage-specialist` | Provide coverage metrics |
| **Support** | `qe-security-scanner` | Provide security metrics |
| **Readonly** | All other agents | Query quality status |

**Forbidden Actions**: No agent may bypass quality gates without logged override.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Quality score < 0.5 | CRITICAL | Block deployment, escalate to Queen Coordinator |
| Security score < 0.6 | CRITICAL | Block deployment, escalate to security team |
| Coherence check failure | CRITICAL | Halt decision, request clarification |
| Override without audit | CRITICAL | Revert override, escalate for investigation |
| Trend showing 3+ consecutive declines | HIGH | Escalate to Queen Coordinator |
| Deployment accuracy < 0.7 | HIGH | Recalibrate advisor model |
| Complexity > 25 | MEDIUM | Flag for refactoring, notify coordinator |
| Hidden threshold detected | HIGH | Escalate, require documentation |

---

## Memory Namespace

- **Namespace**: `qe-patterns/quality-assessment`
- **Retention**: 90 days (longer for deployment decisions)
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `coverage-analysis` | Input | Receive coverage metrics |
| `defect-intelligence` | Input | Receive defect density |
| `security-compliance` | Input | Receive security posture |
| `code-intelligence` | Input | Receive complexity metrics |
| `test-execution` | Input | Receive test results |
| `learning-optimization` | Output | Share quality patterns |

---

## Gate Evaluation Schema

```typescript
interface GateEvaluation {
  id: string;
  timestamp: Date;
  metrics: {
    coverage: number;
    complexity: number;
    defectDensity: number;
    securityScore: number;
    testPassRate: number;
  };
  thresholds: GateThresholds;
  coherenceCheck: CoherenceResult;
  decision: 'pass' | 'fail' | 'warn';
  confidence: number;
  trend: 'improving' | 'stable' | 'degrading';
  recommendations: Recommendation[];
  override?: GateOverride;
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
