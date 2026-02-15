# AQE Fleet v3 Constitution

**Version**: 1.0.0
**Date**: 2026-02-03
**Authority**: Architecture Team
**ADR Reference**: [ADR-058](../../v3/implementation/adrs/ADR-058-guidance-governance-integration.md)

---

## Purpose

This Constitution defines unbreakable invariants for the Agentic QE Fleet v3. These rules are mechanically enforced by the @claude-flow/guidance governance system and cannot be overridden by any agent, task, or user request.

---

## Section I: Unbreakable Invariants

### Invariant 1: Test Execution Integrity

```
INVARIANT test_execution_integrity:
  FOR ALL task IN tasks:
    IF task.claims_tests_passed THEN
      EXISTS execution_proof WHERE
        execution_proof.task_id = task.id AND
        execution_proof.all_tests_executed = true AND
        execution_proof.timestamp IS NOT NULL
```

**Enforcement**: ContinueGate blocks claims of test success without execution proof.

### Invariant 2: Security Scan Requirement

```
INVARIANT security_scan_required:
  FOR ALL change IN code_changes:
    IF change.affects_auth_code OR change.affects_security_code THEN
      EXISTS security_scan WHERE
        security_scan.change_id = change.id AND
        security_scan.status = 'complete' AND
        security_scan.critical_vulnerabilities = 0
```

**Enforcement**: Quality gates block deployment without security scan.

### Invariant 3: Backup Before Delete

```
INVARIANT backup_before_delete:
  FOR ALL operation IN operations:
    IF operation.type = 'delete' AND
       operation.target IN ['memory.db', 'coverage/*', '*.db'] THEN
      EXISTS backup WHERE
        backup.source = operation.target AND
        backup.timestamp < operation.timestamp AND
        backup.verified = true
```

**Enforcement**: MemoryWriteGate blocks destructive operations without backup.

### Invariant 4: Loop Detection

```
INVARIANT loop_detection:
  FOR ALL agent IN active_agents:
    agent.consecutive_identical_actions < MAX_IDENTICAL_ACTIONS (default: 3) AND
    agent.rework_ratio < MAX_REWORK_RATIO (default: 0.5)
```

**Enforcement**: ContinueGate throttles or blocks agents exceeding limits.

### Invariant 5: Budget Enforcement

```
INVARIANT budget_enforcement:
  FOR ALL session IN sessions:
    session.total_cost <= session.budget_limit AND
    session.token_usage <= session.token_limit
```

**Enforcement**: BudgetMeter blocks operations exceeding budget.

### Invariant 6: Memory Consistency

```
INVARIANT memory_consistency:
  FOR ALL pattern IN patterns:
    NOT EXISTS conflicting_pattern WHERE
      conflicting_pattern.domain = pattern.domain AND
      conflicting_pattern.contradicts(pattern) AND
      conflicting_pattern.supersession_marker IS NULL
```

**Enforcement**: MemoryWriteGate blocks contradictory patterns.

### Invariant 7: Verification Before Claim

```
INVARIANT verification_before_claim:
  FOR ALL claim IN success_claims:
    EXISTS verification WHERE
      verification.claim_id = claim.id AND
      verification.method IN ['test_execution', 'manual_review', 'automated_check'] AND
      verification.result = 'passed'
```

**Enforcement**: All success claims require proof of verification.

---

## Section II: Agent Governance

### Agent Loop Protection

All agents are subject to ContinueGate enforcement:

- **Max consecutive retries**: 3 (configurable via `GOVERNANCE_MAX_RETRIES`)
- **Rework ratio threshold**: 0.5 (configurable via `GOVERNANCE_REWORK_THRESHOLD`)
- **Idle timeout**: 5 minutes (configurable via `GOVERNANCE_IDLE_TIMEOUT`)

### Memory Write Protection

All memory writes are subject to MemoryWriteGate enforcement:

- **Contradiction detection**: Enabled
- **Temporal decay**: Patterns older than 30 days with <3 uses may be archived
- **Authority scope**: Domain-specific (agents can only write to their domain namespace)

### Trust-Based Routing

Agent selection is influenced by trust scores:

- **Performance history**: 50% weight
- **Task similarity**: 30% weight
- **Capability match**: 20% weight
- **Minimum trust score for critical tasks**: 0.7

---

## Section III: Domain Rules

Domain-specific rules are loaded from shards located at:
`.claude/guidance/shards/{domain}.shard.md`

### Available Domains

1. test-generation
2. test-execution
3. coverage-analysis
4. quality-assessment
5. defect-intelligence
6. requirements-validation
7. code-intelligence
8. security-compliance
9. contract-testing
10. visual-accessibility
11. chaos-resilience
12. learning-optimization

---

## Section IV: Escalation Procedures

### When Invariant Violation Detected

1. **Log violation** with full context to audit trail
2. **Block operation** that would violate invariant
3. **Notify coordinator** (Queen Coordinator for critical violations)
4. **Create remediation task** if automated fix possible
5. **Escalate to human** if no automated remediation

### When Agent Exceeds Limits

1. **Throttle agent** (reduce priority, increase delays)
2. **Log pattern** for future learning
3. **Consider agent demotion** if repeated violations
4. **Route future tasks** to alternative agents

---

## Section V: Amendment Process

This Constitution may only be amended through:

1. **ADR proposal** with architectural review
2. **Consensus from Architecture Team**
3. **Implementation verification** that new rules are enforceable
4. **Gradual rollout** with feature flags

No agent, task, or emergency can bypass these invariants without formal amendment.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Architecture Team | Initial Constitution for ADR-058 |

---

*This Constitution is enforced by @claude-flow/guidance governance system.*
