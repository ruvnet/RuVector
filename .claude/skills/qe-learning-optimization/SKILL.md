---
name: "QE Learning Optimization"
description: "Transfer learning, metrics optimization, and continuous improvement for AI-powered QE agents."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-learning-optimization.yaml

---

# QE Learning Optimization

## Purpose

Guide the use of v3's learning optimization capabilities including transfer learning between agents, hyperparameter tuning, A/B testing, and continuous performance improvement.

## Activation

- When optimizing agent performance
- When transferring knowledge between agents
- When tuning learning parameters
- When running A/B tests
- When analyzing learning metrics

## Quick Start

```bash
# Transfer knowledge between agents
aqe learn transfer --from jest-generator --to vitest-generator

# Tune hyperparameters
aqe learn tune --agent defect-predictor --metric accuracy

# Run A/B test
aqe learn ab-test --hypothesis "new-algorithm" --duration 7d

# View learning metrics
aqe learn metrics --agent test-generator --period 30d
```

## Agent Workflow

```typescript
// Transfer learning
Task("Transfer test patterns", `
  Transfer learned patterns from Jest test generator to Vitest:
  - Map framework-specific syntax
  - Adapt assertion styles
  - Preserve test structure patterns
  - Validate transfer accuracy
`, "qe-transfer-specialist")

// Metrics optimization
Task("Optimize prediction accuracy", `
  Tune defect-predictor agent:
  - Analyze current performance metrics
  - Run Bayesian hyperparameter search
  - Validate improvements on holdout set
  - Deploy if accuracy improves >5%
`, "qe-metrics-optimizer")
```

## Learning Operations

### 1. Transfer Learning

```typescript
await transferSpecialist.transfer({
  source: {
    agent: 'qe-jest-generator',
    knowledge: ['patterns', 'heuristics', 'optimizations']
  },
  target: {
    agent: 'qe-vitest-generator',
    adaptations: ['framework-syntax', 'api-differences']
  },
  strategy: 'fine-tuning',
  validation: {
    testSet: 'validation-samples',
    minAccuracy: 0.9
  }
});
```

### 2. Hyperparameter Tuning

```typescript
await metricsOptimizer.tune({
  agent: 'defect-predictor',
  parameters: {
    learningRate: { min: 0.001, max: 0.1, type: 'log' },
    batchSize: { values: [16, 32, 64, 128] },
    patternThreshold: { min: 0.5, max: 0.95 }
  },
  optimization: {
    method: 'bayesian',
    objective: 'accuracy',
    trials: 50,
    parallelism: 4
  }
});
```

### 3. A/B Testing

```typescript
await metricsOptimizer.abTest({
  hypothesis: 'ML pattern matching improves test quality',
  variants: {
    control: { algorithm: 'rule-based' },
    treatment: { algorithm: 'ml-enhanced' }
  },
  metrics: ['test-quality-score', 'generation-time'],
  traffic: {
    split: 50,
    minSampleSize: 1000
  },
  duration: '7d',
  significance: 0.05
});
```

### 4. Feedback Loop

```typescript
await metricsOptimizer.feedbackLoop({
  agent: 'test-generator',
  feedback: {
    sources: ['user-corrections', 'test-results', 'code-reviews'],
    aggregation: 'weighted',
    frequency: 'real-time'
  },
  learning: {
    strategy: 'incremental',
    validationSplit: 0.2,
    earlyStoppingPatience: 5
  }
});
```

## Learning Metrics Dashboard

```typescript
interface LearningDashboard {
  agent: string;
  period: DateRange;
  performance: {
    current: MetricValues;
    trend: 'improving' | 'stable' | 'declining';
    percentile: number;
  };
  learning: {
    samplesProcessed: number;
    patternsLearned: number;
    improvementRate: number;
  };
  experiments: {
    active: Experiment[];
    completed: ExperimentResult[];
  };
  recommendations: {
    action: string;
    expectedImpact: number;
    confidence: number;
  }[];
}
```

## Cross-Framework Transfer

```yaml
transfer_mappings:
  jest_to_vitest:
    syntax:
      "describe": "describe"
      "it": "it"
      "expect": "expect"
      "jest.mock": "vi.mock"
      "jest.fn": "vi.fn"
    patterns:
      - mock-module
      - async-testing
      - snapshot-testing

  mocha_to_jest:
    syntax:
      "describe": "describe"
      "it": "it"
      "chai.expect": "expect"
      "sinon.stub": "jest.fn"
    adaptations:
      - assertion-style
      - hook-naming
```

## Continuous Improvement

```typescript
await learningOptimizer.continuousImprovement({
  agents: ['test-generator', 'coverage-analyzer', 'defect-predictor'],
  schedule: {
    metricCollection: 'hourly',
    tuning: 'weekly',
    majorUpdates: 'monthly'
  },
  thresholds: {
    degradationAlert: 5,  // percent
    improvementTarget: 2,  // percent per week
  },
  automation: {
    autoTune: true,
    autoRollback: true,
    requireApproval: ['major-changes']
  }
});
```

## Pattern Learning

```typescript
await patternLearner.learn({
  sources: {
    codeExamples: 'examples/**/*.ts',
    testExamples: 'tests/**/*.test.ts',
    userFeedback: 'feedback/*.json'
  },
  extraction: {
    syntacticPatterns: true,
    semanticPatterns: true,
    contextualPatterns: true
  },
  storage: {
    vectorDB: 'agentdb',
    versioning: true
  }
});
```

## Coordination

**Primary Agents**: qe-transfer-specialist, qe-metrics-optimizer, qe-pattern-learner
**Coordinator**: qe-learning-coordinator
**Related Skills**: qe-test-generation, qe-defect-intelligence
