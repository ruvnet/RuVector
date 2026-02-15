---
name: "QE Defect Intelligence"
description: "AI-powered defect prediction, pattern learning, and root cause analysis for proactive quality management."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-defect-intelligence.yaml

---

# QE Defect Intelligence

## Purpose

Guide the use of v3's defect intelligence capabilities including ML-based defect prediction, pattern recognition from historical data, and automated root cause analysis.

## Activation

- When predicting defect-prone code
- When analyzing failure patterns
- When performing root cause analysis
- When learning from past defects
- When prioritizing testing based on risk

## Quick Start

```bash
# Predict defects in changed code
aqe defect predict --changes HEAD~5..HEAD

# Analyze failure patterns
aqe defect patterns --period 90d --min-occurrences 3

# Root cause analysis
aqe defect rca --failure "test/auth.test.ts:45"

# Learn from resolved defects
aqe defect learn --source jira --status resolved
```

## Agent Workflow

```typescript
// Defect prediction
Task("Predict defect-prone code", `
  Analyze PR #456 changes and predict defect likelihood:
  - Historical defect correlation
  - Code complexity factors
  - Author experience with module
  - Test coverage gaps
  Flag high-risk changes requiring extra review.
`, "qe-defect-predictor")

// Root cause analysis
Task("Analyze test failure", `
  Investigate recurring failure in AuthService tests:
  - Collect failure history (last 30 days)
  - Identify common patterns
  - Trace to potential root causes
  - Suggest fixes using 5-whys analysis
`, "qe-root-cause-analyzer")
```

## Prediction Models

### 1. Change-Based Prediction

```typescript
await defectPredictor.predictFromChanges({
  changes: prChanges,
  factors: {
    codeChurn: { weight: 0.2 },
    complexity: { weight: 0.25 },
    authorExperience: { weight: 0.15 },
    fileHistory: { weight: 0.2 },
    testCoverage: { weight: 0.2 }
  },
  threshold: {
    high: 0.7,
    medium: 0.4,
    low: 0.2
  }
});
```

### 2. Pattern Learning

```typescript
await patternLearner.learnPatterns({
  source: {
    defects: 'jira:project=MYAPP&type=bug',
    commits: 'git:last-6-months',
    tests: 'test-results:last-1000-runs'
  },
  patterns: [
    'code-smell-to-defect',
    'change-coupling',
    'test-gap-correlation',
    'complexity-defect-density'
  ],
  output: {
    rules: true,
    visualizations: true,
    recommendations: true
  }
});
```

### 3. Root Cause Analysis

```typescript
await rootCauseAnalyzer.analyze({
  failure: testFailure,
  methods: [
    'five-whys',
    'fishbone-diagram',
    'fault-tree',
    'change-impact'
  ],
  context: {
    recentChanges: true,
    environmentDiff: true,
    dependencyChanges: true,
    similarFailures: true
  }
});
```

## Defect Prediction Report

```typescript
interface DefectPrediction {
  file: string;
  riskScore: number;  // 0-1
  riskLevel: 'critical' | 'high' | 'medium' | 'low';
  factors: {
    name: string;
    contribution: number;
    details: string;
  }[];
  historicalDefects: {
    count: number;
    recent: Defect[];
    patterns: string[];
  };
  recommendations: {
    action: string;
    priority: string;
    expectedRiskReduction: number;
  }[];
}
```

## Pattern Categories

| Pattern | Detection | Prevention |
|---------|-----------|------------|
| Null pointer | Static analysis | Null checks, Optional |
| Race condition | Concurrency analysis | Locks, atomic ops |
| Memory leak | Heap analysis | Resource cleanup |
| Off-by-one | Boundary analysis | Loop invariants |
| Injection | Taint analysis | Input validation |

## Root Cause Templates

```yaml
root_cause_analysis:
  five_whys:
    max_depth: 5
    prompt_template: "Why did {effect} happen?"

  fishbone:
    categories:
      - people
      - process
      - tools
      - environment
      - materials
      - measurement

  fault_tree:
    top_event: "Test Failure"
    gate_types: [AND, OR, NOT]
    basic_events: true
```

## Integration with Issue Tracking

```typescript
await defectIntelligence.syncWithTracker({
  source: 'jira',
  project: 'MYAPP',
  sync: {
    defectData: 'bidirectional',
    predictions: 'create-tasks',
    patterns: 'update-labels'
  },
  automation: {
    flagHighRisk: true,
    suggestAssignee: true,
    linkRelated: true
  }
});
```

## Coordination

**Primary Agents**: qe-defect-predictor, qe-pattern-learner, qe-root-cause-analyzer
**Coordinator**: qe-defect-intelligence-coordinator
**Related Skills**: qe-coverage-analysis, qe-quality-assessment
