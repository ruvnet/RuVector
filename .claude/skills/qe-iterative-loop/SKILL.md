---
name: "QE Iterative Loop"
description: "Quality Engineering iteration loops for autonomous test improvement, coverage achievement, and quality gate compliance. Use when tests need to pass, coverage targets must be met, quality gates require compliance, or flaky tests need stabilization. Integrates with AQE v3 fleet agents for coordinated quality iteration."
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# QE Iterative Loop

## Overview

QE Iterative Loop is a specialized adaptation of the Ralph Wiggum technique for **Quality Engineering workflows**. It enables autonomous, self-correcting quality cycles where AI agents iterate until quality objectives are achieved - tests pass, coverage targets met, quality gates satisfied, or flaky tests stabilized.

## Why QE Benefits from Iteration

Quality Engineering has **objective, measurable success criteria**:
- Tests either pass or fail (exit code 0 vs non-zero)
- Coverage is quantifiable (78.5% vs 80% target)
- Quality gates have binary outcomes (pass/fail)
- Contract validation has clear schemas

This makes QE ideal for iterative loops - we know exactly when we're done.

## Prerequisites

- AQE v3 fleet initialized
- Test framework configured (Jest, Vitest, Pytest, etc.)
- Coverage tooling (c8, istanbul, coverage.py)
- Quality gate definitions

---

## Quick Start

### Pattern 1: Test Fix Iteration

```bash
# Task: Fix all failing tests
/qe-loop "Run npm test and fix all failing tests.
Success: npm test exits with code 0
Output <promise>TESTS_GREEN</promise> when all tests pass."
```

### Pattern 2: Coverage Target Iteration

```bash
# Task: Achieve 80% coverage
/qe-loop "Increase test coverage to 80%.
Success: Coverage report shows >= 80%
Output <promise>COVERAGE_MET</promise> when target achieved."
```

### Pattern 3: Quality Gate Iteration

```bash
# Task: Pass all quality gates
/qe-loop "Pass all quality gates for deployment.
Gates:
- Unit tests: pass
- Integration tests: pass
- Coverage: >= 80%
- No critical vulnerabilities
- Performance < 200ms P95
Output <promise>QUALITY_GATES_PASSED</promise> when all pass."
```

---

## QE Iteration Patterns

### Pattern 1: Test-Fix Iteration Loop

**Goal**: All tests pass

```markdown
## QE Test-Fix Loop

### Success Criteria
- `npm test` (or test command) returns exit code 0
- No skipped tests (unless explicitly allowed)
- No pending tests

### Iteration Steps
1. Run full test suite
2. Parse output for failures
3. Analyze first failure:
   - Identify failing test file
   - Understand assertion that failed
   - Check if production code or test is wrong
4. Fix the issue
5. Re-run failed test file only (faster feedback)
6. If file passes, run full suite
7. If all pass -> output <promise>TESTS_GREEN</promise>
8. If failures remain -> continue to next failure

### Safety
- Max iterations: 30
- After 10 iterations: report remaining failures
- Stop if same test fails 5 times (possible design issue)
```

### Pattern 2: Coverage Improvement Loop

**Goal**: Achieve coverage target

```markdown
## QE Coverage Loop

### Success Criteria
- Line coverage >= {target}%
- Branch coverage >= {target - 5}% (typically lower target)
- No critical paths uncovered

### Iteration Steps
1. Run tests with coverage: `npm test -- --coverage`
2. Parse coverage report
3. If target met -> output <promise>COVERAGE_MET</promise>
4. Identify uncovered files, sorted by:
   - Critical business logic (highest priority)
   - Lines uncovered (most impact)
   - Complexity (McCabe score)
5. Generate test for highest-impact uncovered code
6. Run tests to verify new test passes
7. Check coverage improvement
8. Continue until target met

### Intelligence Integration
- Store successful test patterns in memory
- Learn from coverage achievements
- Predict best coverage strategies

### Commands
```bash
# Check coverage status (via AQE MCP)
mcp__agentic-qe__memory_retrieve --key "coverage-status" --namespace "coverage"

# Store coverage achievement pattern (via AQE MCP)
mcp__agentic-qe__memory_store \
  --key "coverage-pattern-auth" \
  --value '{"approach": "mock external deps", "improvement": "12%"}' \
  --namespace "coverage-patterns"
```
```

### Pattern 3: Quality Gate Compliance Loop

**Goal**: Pass all quality gates

```markdown
## QE Quality Gate Loop

### Gate Definitions
| Gate | Criteria | Priority |
|------|----------|----------|
| unit-tests | All pass | P0 |
| integration-tests | All pass | P0 |
| coverage | >= 80% | P1 |
| lint | No errors | P1 |
| typecheck | No errors | P1 |
| security | No critical/high CVEs | P0 |
| performance | P95 < 200ms | P2 |

### Iteration Strategy
1. Run all gate checks
2. Identify failing gates (sorted by priority)
3. Fix highest-priority failing gate
4. Re-run that gate to verify
5. When gate passes, move to next failing gate
6. When all pass -> output <promise>QUALITY_GATES_PASSED</promise>

### Gate Check Commands
```bash
# Check all gates
npm test && npm run lint && npm run typecheck && npm run coverage && npm audit

# Individual gate checks
npm test                        # unit-tests
npm run test:integration        # integration-tests
npm run coverage               # coverage
npm run lint                   # lint
npx tsc --noEmit               # typecheck
npm audit --audit-level=high   # security
npm run benchmark              # performance
```

### Integration with AQE v3
```bash
# Submit quality gate assessment task
mcp__agentic-qe__quality_assess --runGate true

# Task orchestration for gate compliance
mcp__agentic-qe__task_orchestrate --task "Pass all quality gates" --strategy adaptive
```
```

### Pattern 4: Flaky Test Stabilization Loop

**Goal**: Eliminate test flakiness

```markdown
## QE Flaky Test Loop

### Flakiness Detection
1. Run test suite N times (e.g., 5 runs)
2. Identify tests that pass/fail inconsistently
3. Calculate flakiness score: (inconsistent runs / total runs)

### Iteration Steps
1. Run: `for i in {1..5}; do npm test; done`
2. Aggregate results per test
3. Identify flaky tests (passed some, failed some)
4. For each flaky test:
   - Analyze failure modes
   - Common causes:
     - Timing issues (add retries/waits)
     - Shared state (isolate test data)
     - Network calls (mock external services)
     - Random data (use deterministic seeds)
   - Apply appropriate fix
   - Re-run 5 times to verify stability
5. When all tests stable -> output <promise>TESTS_STABLE</promise>

### AQE v3 Flaky Detection
```bash
# Use qe-flaky-hunter agent
Task("Hunt flaky tests", "Detect and stabilize flaky tests", "qe-flaky-hunter")

# Or submit flaky detection task
mcp__agentic-qe__task_submit --type "flaky-detection" --priority "p1"
```
```

### Pattern 5: Contract Validation Loop

**Goal**: API contracts aligned

```markdown
## QE Contract Loop

### Success Criteria
- Provider implements all consumer contracts
- No breaking changes detected
- Schema validation passes

### Iteration Steps
1. Run contract tests: `npm run test:contracts`
2. Parse contract violations
3. For each violation:
   - Determine if provider or consumer needs update
   - Update appropriate side
   - Re-run contract tests
4. When all contracts valid -> output <promise>CONTRACTS_VALID</promise>

### AQE v3 Integration
```bash
# Validate contracts
mcp__agentic-qe__contract_validate --contractPath "./contracts"

# Or use specialized agent
Task("Validate API contracts", "Check consumer-provider alignment", "qe-contract-validator")
```
```

---

## AQE v3 Fleet Integration

### Spawning QE Iteration Agents

```bash
# Initialize AQE fleet for QE iteration
mcp__agentic-qe__fleet_init --topology "hierarchical" --maxAgents 8

# Spawn specialized QE iterators using Task tool
Task("Fix failing tests", "Iterate until all tests pass", "qe-tdd-green", {run_in_background: true})
Task("Improve coverage", "Iterate until 80% coverage", "qe-coverage-analyzer", {run_in_background: true})
Task("Fix security issues", "Iterate until security scan passes", "qe-security-scanner", {run_in_background: true})
Task("Stabilize flaky tests", "Iterate until tests stable", "qe-flaky-hunter", {run_in_background: true})
```

### Memory-Enhanced QE Iteration

```bash
# Store iteration patterns for learning (via AQE MCP)
mcp__agentic-qe__memory_store \
  --key "qe-iteration-test-fix" \
  --value '{"approach": "mock external deps", "success_rate": 0.85}' \
  --namespace "qe-patterns"

# Search for relevant QE patterns (via AQE MCP)
mcp__agentic-qe__memory_query \
  --pattern "test-fix-*" \
  --namespace "qe-patterns"

# Record successful iteration completion (via AQE task tracking)
mcp__agentic-qe__task_status --taskId "test-fix-iteration"
```

### QE-Specific Agent Routing

| QE Task | Recommended Agent | Iteration Goal |
|---------|-------------------|----------------|
| Test fixes | `qe-tdd-green` | All tests pass |
| Coverage gaps | `qe-coverage-analyzer` | Target coverage met |
| Quality gates | `qe-quality-gate` | All gates pass |
| Flaky tests | `qe-flaky-hunter` | Tests stable |
| Contract validation | `qe-contract-validator` | Contracts aligned |
| Security fixes | `qe-security-scanner` | No vulnerabilities |
| Performance | `qe-performance-validator` | Benchmarks pass |

---

## Completion Promises for QE

### Standard QE Promises

```markdown
# Test-related
<promise>TESTS_GREEN</promise>       # All tests pass
<promise>TESTS_STABLE</promise>      # Flaky tests fixed
<promise>TDD_COMPLETE</promise>      # TDD cycle done

# Coverage-related
<promise>COVERAGE_MET</promise>      # Target coverage achieved
<promise>GAPS_FILLED</promise>       # Coverage gaps addressed

# Quality gates
<promise>QUALITY_GATES_PASSED</promise>  # All gates pass
<promise>DEPLOYMENT_READY</promise>      # Ready for deploy

# Contract/API
<promise>CONTRACTS_VALID</promise>   # Contracts aligned
<promise>API_COMPLIANT</promise>     # API matches spec

# Security
<promise>SECURITY_CLEARED</promise>  # No vulnerabilities
<promise>COMPLIANCE_MET</promise>    # Compliance requirements met

# Performance
<promise>PERF_TARGET_MET</promise>   # Benchmarks satisfied
```

---

## Example: Full QE Iteration Workflow

```markdown
## Complete QE Iteration Task

### Objective
Achieve deployment readiness through iterative quality improvement

### Phase 1: Test Health (Priority)
1. Run `npm test`
2. Fix failing tests iteratively
3. Success: <promise>TESTS_GREEN</promise>

### Phase 2: Coverage (After Phase 1)
1. Run `npm test -- --coverage`
2. Write tests for uncovered critical paths
3. Success: Coverage >= 80% -> <promise>COVERAGE_MET</promise>

### Phase 3: Quality Gates (After Phase 2)
1. Run lint: `npm run lint`
2. Run typecheck: `npx tsc --noEmit`
3. Fix any violations
4. Success: <promise>LINT_PASS</promise> + <promise>TYPES_PASS</promise>

### Phase 4: Security (Parallel with Phase 3)
1. Run `npm audit`
2. Fix critical/high vulnerabilities
3. Success: <promise>SECURITY_CLEARED</promise>

### Phase 5: Integration
1. Run `npm run test:integration`
2. Fix any integration failures
3. Success: <promise>INTEGRATION_PASS</promise>

### Final Gate
When ALL phases complete -> <promise>DEPLOYMENT_READY</promise>

### Safety Limits
- Max iterations per phase: 15
- Total max iterations: 50
- Stuck detection: 5 iterations without progress triggers escalation
```

---

## Troubleshooting

### Issue: Tests Keep Failing Same Assertion

**Cause**: Likely a design issue, not implementation bug

**Solution**:
1. Stop iteration after 5 attempts on same test
2. Analyze if test expectation is correct
3. Review if production behavior is as designed
4. Escalate to human review if unclear

### Issue: Coverage Plateau

**Cause**: Remaining uncovered code is complex/conditional

**Solution**:
1. Identify uncovered branches (not just lines)
2. Generate edge case tests
3. Consider if uncovered code is dead code
4. Accept lower target for genuinely untestable code

### Issue: Flaky Tests Won't Stabilize

**Cause**: Deep timing or state issues

**Solution**:
1. Add explicit waits/retries
2. Mock time-dependent behavior
3. Isolate test environment
4. Consider marking as skip with explanation

---

## Related Skills

- [iterative-loop](../iterative-loop/) - General iteration technique
- [qe-test-generation](../qe-test-generation/) - AI-powered test creation
- [qe-coverage-analysis](../qe-coverage-analysis/) - Coverage gap detection
- [qe-quality-assessment](../qe-quality-assessment/) - Quality gate management
- [qe-chaos-resilience](../qe-chaos-resilience/) - Resilience iteration testing

## Resources

- [AQE v3 Documentation](../../v3/README.md) - Full v3 reference
- [Ralph Wiggum Technique](https://ghuntley.com/ralph/) - Original methodology
- [Agentic QE MCP Tools](https://github.com/proffesor-for-testing/agentic-qe) - MCP tool reference

---

**Origin**: Adapted from Ralph Wiggum plugin (anthropics/claude-code)
**Specialized for**: Agentic QE v3 Fleet with 60 QE agents
**Domains**: test-generation, test-execution, coverage-analysis, quality-assessment
