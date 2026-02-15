---
name: "QE Coverage Analysis"
description: "O(log n) sublinear coverage gap detection with risk-weighted analysis and intelligent test prioritization."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-coverage-analysis.yaml

---

# QE Coverage Analysis

## Purpose

Guide the use of v3's advanced coverage analysis capabilities including sublinear gap detection algorithms, risk-weighted coverage scoring, and intelligent test prioritization based on code criticality.

## Activation

- When analyzing test coverage
- When identifying coverage gaps
- When prioritizing testing effort
- When setting coverage targets
- When assessing code risk

## Quick Start

```bash
# Analyze coverage with gap detection
aqe coverage analyze --source src/ --tests tests/

# Find high-risk uncovered code
aqe coverage gaps --risk-weighted --threshold 80

# Generate coverage report
aqe coverage report --format html --output coverage-report/

# Compare coverage between branches
aqe coverage diff --base main --head feature-branch
```

## Agent Workflow

```typescript
// Comprehensive coverage analysis
Task("Analyze coverage gaps", `
  Perform O(log n) coverage analysis on src/:
  - Calculate statement, branch, function coverage
  - Identify uncovered critical paths
  - Risk-weight gaps by code complexity and change frequency
  - Recommend tests to write for maximum coverage impact
`, "qe-coverage-specialist")

// Risk-based prioritization
Task("Prioritize coverage effort", `
  Analyze coverage gaps and prioritize by:
  - Business criticality (payment, auth, data)
  - Code complexity (cyclomatic > 10)
  - Recent bug history
  - Change frequency
  Output prioritized list of files needing tests.
`, "qe-coverage-analyzer")
```

## Analysis Strategies

### 1. Sublinear Gap Detection

```typescript
await coverageAnalyzer.detectGaps({
  algorithm: 'sublinear',  // O(log n) complexity
  source: 'src/**/*.ts',
  metrics: ['statement', 'branch', 'function'],
  sampling: {
    enabled: true,
    confidence: 0.95,
    maxSamples: 1000
  }
});
```

### 2. Risk-Weighted Coverage

```typescript
await coverageAnalyzer.riskWeightedAnalysis({
  coverage: coverageReport,
  riskFactors: {
    complexity: { weight: 0.3, threshold: 10 },
    changeFrequency: { weight: 0.25, window: '90d' },
    bugHistory: { weight: 0.25, window: '180d' },
    criticality: { weight: 0.2, tags: ['payment', 'auth'] }
  },
  output: {
    riskScore: true,
    prioritizedGaps: true
  }
});
```

### 3. Differential Coverage

```typescript
await coverageAnalyzer.diffCoverage({
  base: 'main',
  head: 'feature-branch',
  requirements: {
    newCode: 80,           // New code must have 80% coverage
    modifiedCode: 'maintain',  // Don't decrease existing
    deletedCode: 'ignore'
  }
});
```

## Coverage Thresholds

```yaml
thresholds:
  global:
    statements: 80
    branches: 75
    functions: 85
    lines: 80

  per_file:
    min_statements: 70
    critical_paths: 90

  new_code:
    statements: 85
    branches: 80

  exceptions:
    - path: "src/migrations/**"
      reason: "Database migrations"
    - path: "src/generated/**"
      reason: "Auto-generated code"
```

## Coverage Report

```typescript
interface CoverageAnalysis {
  summary: {
    statements: { covered: number; total: number; percentage: number };
    branches: { covered: number; total: number; percentage: number };
    functions: { covered: number; total: number; percentage: number };
  };
  gaps: {
    file: string;
    uncoveredLines: number[];
    uncoveredBranches: BranchInfo[];
    riskScore: number;
    suggestedTests: string[];
  }[];
  trends: {
    period: string;
    coverageChange: number;
    newGaps: number;
    closedGaps: number;
  };
  recommendations: {
    priority: 'critical' | 'high' | 'medium' | 'low';
    file: string;
    action: string;
    expectedImpact: number;
  }[];
}
```

## Quality Gates

```yaml
quality_gates:
  coverage:
    block_merge:
      - new_code_coverage < 80
      - coverage_regression > 5
      - critical_path_uncovered

    warn:
      - overall_coverage < 75
      - branch_coverage < 70

    metrics:
      - track_trends: true
      - alert_on_decline: 3  # consecutive PRs
```

## Coordination

**Primary Agents**: qe-coverage-specialist, qe-coverage-analyzer, qe-gap-detector
**Coordinator**: qe-coverage-coordinator
**Related Skills**: qe-test-generation, qe-quality-assessment
