# Test Generation Domain Shard

**Domain**: test-generation
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Pattern-Driven Generation**: All test generation MUST leverage the PatternMatcherService to apply learned patterns before generating new tests from scratch.

2. **Coherence Gate Requirement**: Generated tests MUST pass the TestGenerationCoherenceGate validation, ensuring they align with source requirements and do not contradict existing test specifications.

3. **TDD Workflow Compliance**: When TDD mode is requested, tests MUST be generated BEFORE implementation code, with proper red-green-refactor cycle tracking.

4. **Coverage Intent Verification**: Generated tests MUST target the coverage gaps identified by the coverage-analysis domain, not duplicate existing coverage.

5. **No Mock Abuse**: Mock objects MUST be limited to external dependencies; internal service interactions SHOULD use real implementations when feasible.

6. **Assertion Quality**: Each generated test MUST contain meaningful assertions (minimum 1 assertion per test case), not just execution checks.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Quality Score | 0.7 | 0.85 | < 0.5 |
| Confidence | 0.6 | 0.8 | < 0.4 |
| Pattern Match Rate | 0.3 | 0.6 | N/A |
| Coherence Score | 0.7 | 0.9 | < 0.5 |
| Assertion Density | 1.0 | 2.5 | < 1.0 |

---

## Invariants

```
INVARIANT test_generation_coherence:
  FOR ALL generated_test IN test_output:
    generated_test.coherence_score >= 0.7 AND
    generated_test.assertions.length >= 1 AND
    NOT EXISTS duplicate IN existing_tests WHERE
      duplicate.coverage_target = generated_test.coverage_target AND
      duplicate.assertion_set EQUALS generated_test.assertion_set
```

```
INVARIANT tdd_workflow_integrity:
  FOR ALL tdd_session IN tdd_sessions:
    IF tdd_session.mode = 'strict' THEN
      tdd_session.test_created_before_implementation = true
```

```
INVARIANT pattern_application_before_generation:
  FOR ALL generation_request IN requests:
    EXISTS pattern_search_attempt WHERE
      pattern_search_attempt.request_id = generation_request.id AND
      pattern_search_attempt.completed = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/test-generation/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Test Generator Service | `services/test-generator.ts` | Core AI-powered test generation |
| Pattern Matcher Service | `services/pattern-matcher.ts` | Learned pattern application |
| Coherence Gate Service | `services/coherence-gate-service.ts` | Requirement alignment validation |
| Test Generation Coordinator | `coordinator.ts` | Workflow orchestration |

**Related Interfaces**: `interfaces/index.ts` defines `GenerateTestsRequest`, `TDDRequest`, `PropertyTestRequest`, and response types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-test-architect` | Full test generation, pattern learning |
| **Secondary** | `qe-tdd-specialist` | TDD workflow, red-green-refactor |
| **Support** | `qe-coverage-specialist` | Gap identification for targeting |
| **Readonly** | `qe-quality-gate` | Validation, no generation |

**Forbidden Agents**: Agents without `test-generation` domain capability MUST NOT invoke test generation services directly.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Quality score < 0.5 | CRITICAL | Escalate to Queen Coordinator, halt generation |
| Coherence score < 0.5 | CRITICAL | Escalate to Queen Coordinator, request requirement clarification |
| Pattern match rate > 0.9 | WARNING | Log potential over-fitting, review patterns |
| 3+ consecutive generation failures | HIGH | Escalate to Queen Coordinator, switch agent |
| TDD sequence violation | MEDIUM | Log violation, notify coordinator |
| Duplicate test detection | LOW | Auto-deduplicate, log pattern |

---

## Memory Namespace

- **Namespace**: `qe-patterns/test-generation`
- **Retention**: 30 days with minimum 3 uses
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `coverage-analysis` | Input | Receive coverage gaps for targeting |
| `requirements-validation` | Input | Receive BDD scenarios for test generation |
| `test-execution` | Output | Provide generated tests for execution |
| `learning-optimization` | Bidirectional | Share and receive patterns |
| `quality-assessment` | Output | Submit tests for quality gate evaluation |

---

*This shard is enforced by @claude-flow/guidance governance system.*
