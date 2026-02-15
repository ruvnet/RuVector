---
name: "QE Test Execution"
description: "Parallel test execution orchestration with intelligent scheduling, retry logic, and comprehensive result aggregation."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-test-execution.yaml

---

# QE Test Execution

## Purpose

Guide the use of v3's test execution capabilities including parallel orchestration, smart test selection, flaky test handling, and distributed execution across multiple environments.

## Activation

- When running test suites
- When optimizing test execution time
- When handling flaky tests
- When setting up CI/CD test pipelines
- When executing tests across environments

## Quick Start

```bash
# Run all tests with parallelization
aqe test run --parallel --workers 4

# Run affected tests only
aqe test run --affected --since HEAD~1

# Run with retry for flaky tests
aqe test run --retry 3 --retry-delay 1000

# Run specific test types
aqe test run --type unit,integration --exclude e2e
```

## Agent Workflow

```typescript
// Orchestrate test execution
Task("Execute test suite", `
  Run the full test suite with:
  - 4 parallel workers
  - Retry flaky tests up to 3 times
  - Generate JUnit report
  - Fail fast on critical tests
  Report results and any failures.
`, "qe-test-executor")

// Smart test selection
Task("Run affected tests", `
  Analyze changes in PR #123 and:
  - Identify affected test files
  - Run only relevant tests
  - Include integration tests for changed modules
  - Report coverage delta
`, "qe-test-selector")
```

## Execution Strategies

### 1. Parallel Execution

```typescript
await testExecutor.runParallel({
  suites: ['unit', 'integration'],
  workers: 4,
  distribution: 'by-file',  // or 'by-test', 'by-duration'
  isolation: 'process',
  sharding: {
    enabled: true,
    total: 4,
    index: process.env.SHARD_INDEX
  }
});
```

### 2. Smart Test Selection

```typescript
await testExecutor.runAffected({
  changes: gitChanges,
  selection: {
    direct: true,      // Tests for changed files
    transitive: true,  // Tests for dependents
    integration: true  // Integration tests touching changed code
  },
  fallback: 'full-suite'  // If analysis fails
});
```

### 3. Flaky Test Handling

```typescript
await testExecutor.handleFlaky({
  detection: {
    enabled: true,
    threshold: 0.1,  // 10% flake rate
    window: 100      // Last 100 runs
  },
  strategy: {
    retry: 3,
    quarantine: true,
    notify: ['#flaky-tests']
  }
});
```

## Execution Configuration

```yaml
execution:
  parallel:
    workers: auto  # CPU cores - 1
    timeout: 30000
    bail: false

  retry:
    count: 2
    delay: 1000
    only_failed: true

  reporting:
    formats: [junit, json, html]
    include_timing: true
    include_logs: true

  environments:
    - name: node-18
      image: node:18-alpine
    - name: node-20
      image: node:20-alpine
```

## CI/CD Integration

```yaml
# GitHub Actions example
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      shard: [1, 2, 3, 4]
  steps:
    - uses: actions/checkout@v4
    - name: Run tests
      run: |
        aqe test run \
          --shard ${{ matrix.shard }}/4 \
          --parallel \
          --report junit
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: test-results-${{ matrix.shard }}
        path: reports/
```

## Result Aggregation

```typescript
interface ExecutionResults {
  summary: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
    flaky: number;
    duration: number;
  };
  shards: ShardResult[];
  failures: TestFailure[];
  flakyTests: FlakyTest[];
  coverage: CoverageReport;
  timing: TimingAnalysis;
}
```

## Coordination

**Primary Agents**: qe-test-executor, qe-test-selector, qe-flaky-detector
**Coordinator**: qe-test-execution-coordinator
**Related Skills**: qe-test-generation, qe-coverage-analysis
