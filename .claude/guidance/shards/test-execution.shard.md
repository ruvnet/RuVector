# Test Execution Domain Shard

**Domain**: test-execution
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Execution Proof Required**: Every test execution claim MUST be backed by actual execution proof (timestamps, exit codes, output logs). No synthetic or assumed results.

2. **Parallel Execution Safety**: Parallel test execution MUST use proper isolation (separate database instances, unique temp directories, non-conflicting ports).

3. **Flaky Test Quarantine**: Tests identified as flaky (>10% failure variance across 5+ runs) MUST be quarantined and reported to the FlakyDetectorService.

4. **Retry Policy Compliance**: Retry attempts MUST follow the configured RetryPolicy, with exponential backoff and maximum retry limits enforced.

5. **E2E Test Integrity**: End-to-end tests MUST capture screenshots on failure and preserve browser state for debugging.

6. **Prioritization Respect**: Test execution order MUST respect priorities assigned by TestPrioritizerService based on risk and change impact.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Pass Rate | 0.8 | 0.95 | < 0.6 |
| Execution Confidence | 0.7 | 0.9 | < 0.5 |
| Flaky Rate | < 0.1 | < 0.05 | > 0.2 |
| Retry Success Rate | 0.5 | 0.8 | < 0.3 |
| E2E Reliability | 0.75 | 0.9 | < 0.5 |

---

## Invariants

```
INVARIANT execution_proof_required:
  FOR ALL claim IN execution_claims:
    IF claim.status = 'passed' OR claim.status = 'failed' THEN
      EXISTS proof WHERE
        proof.claim_id = claim.id AND
        proof.timestamp IS NOT NULL AND
        proof.exit_code IS NOT NULL AND
        proof.output_captured = true
```

```
INVARIANT parallel_isolation:
  FOR ALL parallel_run IN parallel_executions:
    FOR ALL test_a, test_b IN parallel_run.concurrent_tests:
      test_a.resource_namespace != test_b.resource_namespace
```

```
INVARIANT flaky_quarantine:
  FOR ALL test IN test_suite:
    IF test.failure_variance > 0.1 AND test.run_count >= 5 THEN
      test.quarantined = true
```

```
INVARIANT retry_limit_enforcement:
  FOR ALL retry_session IN retries:
    retry_session.attempts <= retry_session.policy.max_retries
```

---

## Patterns

**Domain Source**: `v3/src/domains/test-execution/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Test Executor Service | `services/test-executor.ts` | Core test execution engine |
| Flaky Detector Service | `services/flaky-detector.ts` | Flaky test identification |
| Retry Handler Service | `services/retry-handler.ts` | Intelligent retry logic |
| Test Prioritizer Service | `services/test-prioritizer.ts` | Risk-based prioritization |
| E2E Test Runner Service | `services/e2e-runner.ts` | Browser-based E2E execution |

**E2E Step Types**: `types/index.ts` defines step types (NavigateStep, ClickStep, TypeStep, WaitStep, AssertStep, etc.).

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-parallel-executor` | Full execution, parallel orchestration |
| **Secondary** | `qe-flaky-hunter` | Flaky detection, quarantine decisions |
| **Support** | `qe-test-architect` | Test selection guidance |
| **Readonly** | `qe-quality-gate` | Results validation, no execution |

**Forbidden Agents**: Agents MUST NOT execute tests without execution proof capability. No blind pass/fail claims.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Pass rate < 0.6 | CRITICAL | Escalate to Queen Coordinator, halt pipeline |
| Flaky rate > 0.2 | CRITICAL | Escalate, initiate flaky test hunt |
| Execution proof missing | CRITICAL | Block claim, escalate for investigation |
| 3+ retries exhausted | HIGH | Escalate to Queen Coordinator |
| Parallel isolation breach | HIGH | Halt parallel execution, escalate |
| E2E screenshot capture failure | MEDIUM | Log warning, attempt recovery |
| Prioritization conflict | LOW | Log, use default ordering |

---

## Memory Namespace

- **Namespace**: `qe-patterns/test-execution`
- **Retention**: 30 days with minimum 3 uses
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-generation` | Input | Receive generated tests |
| `coverage-analysis` | Output | Report coverage data |
| `quality-assessment` | Output | Report execution results |
| `defect-intelligence` | Output | Report failures for RCA |
| `learning-optimization` | Bidirectional | Share execution patterns |

---

## Execution Proof Schema

```typescript
interface ExecutionProof {
  taskId: string;
  testId: string;
  status: 'passed' | 'failed' | 'skipped' | 'error';
  timestamp: Date;
  duration: number;
  exitCode: number;
  outputLog: string;
  errorLog?: string;
  screenshots?: string[]; // For E2E
  retryCount: number;
  isolationNamespace: string;
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
