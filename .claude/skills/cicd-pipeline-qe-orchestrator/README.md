# CI/CD Pipeline QE Orchestrator

**Comprehensive quality engineering orchestration across all CI/CD pipeline phases.**

## What is This?

The CI/CD Pipeline QE Orchestrator is an advanced Claude Code skill that provides intelligent, phase-based quality engineering across your entire software delivery pipeline. It intelligently selects from 41 QE skills and coordinates 19 specialized QE agents to ensure holistic quality coverage from commit to production.

## Quick Start

### 1. Invoke the Skill

```javascript
// In Claude Code
Skill("cicd-pipeline-qe-orchestrator")
```

### 2. Analyze Your Pipeline

```javascript
Task("Pipeline Analysis",
     "Analyze our CI/CD pipeline and recommend quality strategy for all phases",
     "qe-fleet-commander")
```

### 3. Get Phase-Specific Recommendations

The orchestrator will provide:
- Skill selections per pipeline phase
- Agent coordination patterns
- Quality gate configurations
- Timing budgets
- Rollback strategies

## What Makes This Different?

### Traditional QE Approach
```
┌─────────────┐
│ Run Tests   │  Manual selection
│ Manual QA   │  Ad-hoc coverage
│ Deploy      │  Hope for the best
└─────────────┘
```

### Orchestrated QE Approach
```
┌──────────────────────────────────────────────┐
│ Commit Phase (Shift-Left)                    │
│  ├─ Skills: shift-left-testing, tdd          │
│  ├─ Agents: qe-test-generator (parallel)     │
│  └─ Gates: 80% coverage, 0 critical issues   │
├──────────────────────────────────────────────┤
│ Build Phase                                  │
│  ├─ Skills: test-automation, mutation        │
│  ├─ Agents: qe-test-executor (batched)       │
│  └─ Gates: 90% coverage, mutation > 70%      │
├──────────────────────────────────────────────┤
│ Integration Phase                            │
│  ├─ Skills: api-testing, performance         │
│  ├─ Agents: qe-performance-tester (parallel) │
│  └─ Gates: p95 < 200ms, 0 security critical  │
├──────────────────────────────────────────────┤
│ Staging Phase                                │
│  ├─ Skills: chaos-engineering, visual        │
│  ├─ Agents: qe-chaos-engineer (sequential)   │
│  └─ Gates: resilience validated, 0 visual    │
├──────────────────────────────────────────────┤
│ Production Phase (Shift-Right)               │
│  ├─ Skills: shift-right-testing              │
│  ├─ Agents: qe-production-intelligence       │
│  └─ Gates: error rate < 0.1%, monitors pass  │
└──────────────────────────────────────────────┘
```

## Key Features

### 🎯 Intelligent Skill Selection

Automatically selects optimal skills based on:
- Pipeline phase (commit, build, test, staging, production)
- Application type (API, web, mobile, backend)
- Risk level (critical, high, medium, low)
- Deployment frequency (continuous, daily, weekly, monthly)

### 🤖 Agent Coordination

Orchestrates 19 specialized QE agents:
- Parallel execution for speed
- Sequential execution for dependencies
- Memory-based coordination via `aqe/*` namespace
- Smart batching to avoid OOM issues

### 📊 Quality Gates

Configurable gates per phase:
- Coverage thresholds
- Performance SLAs
- Security vulnerability limits
- Deployment readiness scores

### 🔄 Adaptive Strategies

Adapts testing approach based on:
- Code change risk analysis
- Historical failure patterns
- Resource constraints
- Time budgets

## Use Cases

### Microservices Pipeline
```javascript
// See resources/workflows/microservice-pipeline.md
// Fast feedback, contract testing, chaos engineering
```

### Monolith Pipeline
```javascript
// See resources/workflows/monolith-pipeline.md
// Smart test selection, comprehensive regression, DB migrations
```

### Mobile App Pipeline
```javascript
// See resources/workflows/mobile-pipeline.md
// Device testing, accessibility, localization, staged rollout
```

## Integration with AQE Fleet

### All 41 Skills Available

The orchestrator can invoke any of the 41 QE skills:

**Phase 1 Skills (18)**:
- Core Testing: agentic-quality-engineering, context-driven-testing, holistic-testing-pact
- Methodologies: tdd-london-chicago, xp-practices, risk-based-testing, test-automation-strategy
- Techniques: api-testing-patterns, exploratory-testing-advanced, performance-testing, security-testing
- Code Quality: code-review-quality, refactoring-patterns, quality-metrics
- Communication: bug-reporting-excellence, technical-writing, consultancy-practices

**Phase 2 Skills (16)**:
- Methodologies: regression-testing, shift-left-testing, shift-right-testing, test-design-techniques, mutation-testing, test-data-management
- Specialized: accessibility-testing, mobile-testing, database-testing, contract-testing, chaos-engineering-resilience, compatibility-testing, localization-testing, compliance-testing, visual-testing-advanced
- Infrastructure: test-environment-management, test-reporting-analytics

**Phase 3 Skills (4)**:
- Strategic: six-thinking-hats, brutal-honesty-review, sherlock-review, cicd-pipeline-qe-orchestrator

**Phase 4 Skills (3)**:
- Advanced: testability-scoring, verification-quality, pair-programming

### All 19 Agents Available

The orchestrator coordinates all 19 QE agents:

**Core Testing (5)**: test-generator, test-executor, coverage-analyzer, quality-gate, quality-analyzer
**Performance & Security (2)**: performance-tester, security-scanner
**Strategic Planning (3)**: requirements-validator, production-intelligence, fleet-commander
**Deployment (1)**: deployment-readiness
**Advanced Testing (4)**: regression-risk-analyzer, test-data-architect, api-contract-validator, flaky-test-hunter
**Specialized (2)**: visual-tester, chaos-engineer

## Example Workflows

### Example 1: New Feature Development

```javascript
// Commit phase
Task("TDD Tests", "Generate tests for new UserService.createUser()", "qe-test-generator")
Skill("tdd-london-chicago")

// Build phase
Task("Run Tests", "Execute full suite with coverage", "qe-test-executor")
Task("Coverage Check", "Analyze and report gaps", "qe-coverage-analyzer")

// Integration phase
Task("API Tests", "Validate new API contracts", "qe-api-contract-validator")
Task("Performance", "Load test new endpoint", "qe-performance-tester")

// Production
Task("Monitor", "Track new feature adoption", "qe-production-intelligence")
```

### Example 2: Hotfix Deployment

```javascript
// Commit phase (minimal)
Task("Risk Analysis", "Identify affected code", "qe-regression-risk-analyzer")
Task("Targeted Tests", "Run minimal test suite", "qe-test-executor")

// Build phase (fast)
Task("Smoke Tests", "Critical path validation", "qe-test-executor")

// Production (monitored)
Task("Canary Deploy", "Monitor hotfix rollout", "qe-production-intelligence")
```

### Example 3: Comprehensive Release

```javascript
// Full pipeline orchestration
Skill("cicd-pipeline-qe-orchestrator")

// The skill will guide you through:
// 1. Commit phase: shift-left testing, TDD
// 2. Build phase: full regression, mutation testing
// 3. Integration: API, performance, security testing
// 4. Staging: chaos testing, visual regression
// 5. Production: staged rollout, monitoring
```

## Configuration

### Pipeline Phase Mapping

Customize phase names to match your pipeline:

```javascript
const pipelinePhases = {
  commit: "pre-merge",
  build: "ci-build",
  integration: "test-env",
  staging: "uat",
  production: "prod"
};
```

### Quality Gate Thresholds

Adjust thresholds per phase:

```json
{
  "commit": { "coverage": 80, "max_violations": 0 },
  "build": { "coverage": 90, "mutation_score": 70 },
  "integration": { "p95_ms": 200, "security_critical": 0 },
  "staging": { "readiness_score": 85 },
  "production": { "error_rate": 0.001 }
}
```

## Best Practices

### 1. Start with Quick Start
```javascript
// Get recommendations first
Skill("cicd-pipeline-qe-orchestrator")
```

### 2. Adapt to Your Context
```javascript
// Use context-driven testing skill
Skill("context-driven-testing")
```

### 3. Measure and Improve
```javascript
// Track quality metrics
Task("Quality Trends", "Analyze quality improvements", "qe-quality-analyzer")
```

### 4. Learn from Production
```javascript
// Convert incidents to tests
Task("Production Intelligence", "Turn incidents into test scenarios", "qe-production-intelligence")
```

## Troubleshooting

See main SKILL.md [Troubleshooting section](SKILL.md#troubleshooting) for:
- Too many tests running (OOM)
- Pipeline takes too long
- Quality gates failing

## Resources

- [Main Skill Documentation](SKILL.md)
- [Microservice Pipeline Workflow](resources/workflows/microservice-pipeline.md)
- [Monolith Pipeline Workflow](resources/workflows/monolith-pipeline.md)
- [Mobile Pipeline Workflow](resources/workflows/mobile-pipeline.md)
- [All 61 QE Skills Reference](https://github.com/proffesor-for-testing/agentic-qe/blob/main/docs/reference/skills.md)
- [All 50 QE Agents Reference](https://github.com/proffesor-for-testing/agentic-qe/blob/main/docs/reference/agents.md)

## Contributing

To add new pipeline workflows:
1. Create workflow in `resources/workflows/[name]-pipeline.md`
2. Follow existing template structure
3. Include phase-by-phase breakdown
4. Add skill and agent selections
5. Define quality gates
6. Provide complete code examples

## License

Part of the Agentic QE Fleet - MIT License

---

**Created**: 2025-11-13
**Version**: 1.0.0
**Integrations**: 61 QE Skills, 50 QE Agents, All CI/CD platforms
