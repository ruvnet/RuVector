---
name: cicd-pipeline-qe-orchestrator
description: "Orchestrate quality engineering across CI/CD pipeline phases. Use when designing test strategies, planning quality gates, or implementing shift-left/shift-right testing."
category: infrastructure
priority: critical
tokenEstimate: 1300
agents: [qe-fleet-commander, qe-test-generator, qe-test-executor, qe-coverage-analyzer, qe-quality-gate, qe-deployment-readiness]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: [agentic-quality-engineering, shift-left-testing]
quick_reference_card: true
tags: [cicd, pipeline, orchestration, quality-gates, shift-left, shift-right, fleet]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/cicd-pipeline-qe-orchestrator.yaml

---

# CI/CD Pipeline QE Orchestrator

<default_to_action>
When orchestrating quality across CI/CD pipeline:
1. ANALYZE pipeline phases: commit, build, test, staging, production
2. SELECT optimal skills and agents for each phase
3. CONFIGURE quality gates with measurable thresholds
4. EXECUTE with parallel agent coordination
5. ADAPT strategy based on risk, complexity, and environment

**Quick Phase Selection:**
- Commit (Shift-Left) → TDD, code review, unit tests
- Build → Coverage analysis, mutation testing, flaky detection
- Integration → API contracts, performance, security
- Staging → Chaos testing, visual regression, accessibility
- Production (Shift-Right) → Synthetic monitoring, RUM, compliance

**Critical Success Factors:**
- Quality gates block bad deployments
- Agents coordinate through memory namespaces
- Adapt strategy based on risk level
</default_to_action>

## Quick Reference Card

### When to Use
- Designing pipeline test strategies
- Implementing quality gates
- Coordinating multiple QE agents
- Shift-left and shift-right testing

### Phase-Agent Matrix
| Phase | Primary Agents | Key Skills |
|-------|---------------|------------|
| Commit | qe-test-generator, qe-requirements-validator | tdd-london-chicago, shift-left |
| Build | qe-test-executor, qe-coverage-analyzer, qe-flaky-test-hunter | test-automation, mutation-testing |
| Test | qe-api-contract-validator, qe-performance-tester, qe-security-scanner | api-testing, performance, security |
| Staging | qe-chaos-engineer, qe-visual-tester, qe-deployment-readiness | chaos-engineering, accessibility |
| Prod | qe-production-intelligence, qe-quality-analyzer | shift-right, compliance |

### Quality Gate Thresholds
| Phase | Metric | Threshold | Blocking |
|-------|--------|-----------|----------|
| Commit | Unit coverage | > 80% | Yes |
| Build | All tests pass | 100% | Yes |
| Build | Mutation score | > 70% | No |
| Test | API contracts | No breaking changes | Yes |
| Test | p95 response | < 200ms | Yes |
| Test | Security critical | 0 | Yes |
| Staging | Deployment readiness | > 85% | Yes |

### Fleet Configuration
- **Topology**: hierarchical
- **Max Agents**: 10 per phase
- **Coordination**: aqe/* memory namespace

---

## Pipeline Phases

### Phase 1: Commit (Shift-Left)

**Goal:** Catch defects early, ensure testability

**Agents:**
- `qe-test-generator` - Generate unit tests
- `qe-requirements-validator` - BDD scenarios, INVEST criteria

**Skills:** `shift-left-testing`, `tdd-london-chicago`, `code-review-quality`

```javascript
// Parallel execution
Task("Generate Tests", "Create unit tests for new methods", "qe-test-generator")
Task("Validate Requirements", "Check BDD scenarios", "qe-requirements-validator")
```

**Gates:** Unit coverage > 80%, Static analysis clean, Code review approved

---

### Phase 2: Build

**Goal:** Validate integration, ensure coverage

**Agents:**
- `qe-test-executor` - Run test suites
- `qe-coverage-analyzer` - Coverage gaps (O(log n))
- `qe-flaky-test-hunter` - Detect/stabilize flaky tests
- `qe-regression-risk-analyzer` - Minimal regression suite

**Skills:** `test-automation-strategy`, `mutation-testing`, `regression-testing`

```javascript
Task("Execute Tests", "Run full suite, store in aqe/test-results/*", "qe-test-executor")
Task("Coverage Analysis", "Identify gaps", "qe-coverage-analyzer")
Task("Flaky Detection", "Analyze test history", "qe-flaky-test-hunter")
```

**Gates:** All tests pass, Coverage > 90% critical paths, No new flaky tests

---

### Phase 3: Integration/Test

**Goal:** Validate contracts, performance, security

**Agents:**
- `qe-api-contract-validator` - Breaking changes detection
- `qe-performance-tester` - Load test critical paths
- `qe-security-scanner` - SAST/DAST scans
- `qe-test-data-architect` - Realistic test data (10k+/sec)

**Skills:** `api-testing-patterns`, `performance-testing`, `security-testing`

```javascript
// Parallel testing
Task("API Contracts", "Validate for breaking changes", "qe-api-contract-validator")
Task("Performance", "Load test 1000 users", "qe-performance-tester")
Task("Security", "SAST/DAST scan", "qe-security-scanner")
```

**Gates:** No breaking API changes, p95 < 200ms, No critical vulnerabilities

---

### Phase 4: Staging

**Goal:** Validate production-like environment, resilience

**Agents:**
- `qe-chaos-engineer` - Fault injection
- `qe-visual-tester` - Visual regression
- `qe-deployment-readiness` - Risk assessment

**Skills:** `chaos-engineering-resilience`, `accessibility-testing`, `visual-testing`

```javascript
Task("Chaos Testing", "Controlled failure injection", "qe-chaos-engineer")
Task("Visual Testing", "Visual regression", "qe-visual-tester")
Task("Deployment Check", "Risk assessment", "qe-deployment-readiness")
```

**Gates:** Chaos tests pass, No visual regressions, Readiness > 85%

---

### Phase 5: Production (Shift-Right)

**Goal:** Monitor real usage, validate compliance

**Agents:**
- `qe-production-intelligence` - Incident → test scenarios
- `qe-quality-analyzer` - Quality metrics and trends

**Skills:** `shift-right-testing`, `compliance-testing`

```javascript
Task("Production Intelligence", "Convert incidents to tests", "qe-production-intelligence")
Task("Quality Analysis", "Production metrics", "qe-quality-analyzer")
```

**Gates:** Synthetic monitors pass, Error rate < 0.1%, Compliance validated

---

## Complete Pipeline Example

```javascript
// Phase 1: Commit
Task("TDD Generation", "Generate tests for new features", "qe-test-generator")
Task("Requirements", "Validate BDD scenarios", "qe-requirements-validator")

// Phase 2: Build
Task("Execute Tests", "Full suite with coverage", "qe-test-executor")
Task("Coverage", "Analyze gaps", "qe-coverage-analyzer")
Task("Flaky Hunt", "Stabilize flaky tests", "qe-flaky-test-hunter")

// Phase 3: Integration
Task("API Contracts", "Check breaking changes", "qe-api-contract-validator")
Task("Performance", "1000 user load test", "qe-performance-tester")
Task("Security", "SAST/DAST scans", "qe-security-scanner")

// Phase 4: Staging
Task("Chaos", "Fault injection testing", "qe-chaos-engineer")
Task("Visual", "Visual regression", "qe-visual-tester")
Task("Readiness", "Deployment assessment", "qe-deployment-readiness")

// Phase 5: Production
Task("Intelligence", "Convert incidents", "qe-production-intelligence")
Task("Quality Gate", "Final validation", "qe-quality-gate")
```

---

## Adaptive Strategy

### By Risk Level
| Risk | Strategy | Agents |
|------|----------|--------|
| Critical | All phases, manual gates | Full fleet |
| High | Automated gates, comprehensive | 10+ agents |
| Medium | Smart selection, risk-based | 5-8 agents |
| Low | Minimal regression, fast | 2-3 agents |

### By Application Type
| Type | Focus Skills | Primary Agents |
|------|-------------|----------------|
| API | api-testing, contract, performance | api-contract-validator, performance-tester |
| Web UI | visual-testing, accessibility | visual-tester, accessibility |
| Mobile | mobile-testing, compatibility | performance-tester, visual-tester |
| Backend | database-testing, security | security-scanner, performance-tester |

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/pipeline/
├── phase-results/*         - Results from each phase
├── quality-gates/*         - Gate validation results
├── orchestration-plan/*    - Selected skills and agents
├── test-plan/generated     - Test plans
├── coverage/gaps           - Coverage analysis
├── security/findings       - Security results
└── performance/results     - Performance data
```

### Fleet Orchestration
```typescript
Task("Fleet Orchestration",
     "Coordinate 10 agents across phases: commit (2), build (3), test (3), staging (2)",
     "qe-fleet-commander")
```

### Blackboard Events
| Event | Trigger | Subscribers |
|-------|---------|-------------|
| `phase:commit:complete` | Commit phase done | build agents |
| `coverage:gap:detected` | Gap found | test-generator |
| `security:finding:critical` | Vulnerability | quality-gate |
| `quality:gate:evaluated` | Gate decision | fleet-commander |

---

## Quality Gate Configuration

```json
{
  "commit": {
    "gates": [
      { "metric": "unit_coverage", "threshold": 80, "blocking": true },
      { "metric": "static_analysis_critical", "max": 0, "blocking": true }
    ]
  },
  "build": {
    "gates": [
      { "metric": "all_tests_passed", "threshold": 100, "blocking": true },
      { "metric": "mutation_score", "threshold": 70, "blocking": false }
    ]
  },
  "integration": {
    "gates": [
      { "metric": "api_breaking_changes", "max": 0, "blocking": true },
      { "metric": "performance_p95_ms", "threshold": 200, "blocking": true },
      { "metric": "security_critical", "max": 0, "blocking": true }
    ]
  }
}
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM during tests | Running all tests in parallel | Use batched execution |
| Pipeline too slow | Comprehensive testing every commit | Smart test selection |
| Gates always failing | Thresholds too strict | Analyze trends, adjust |

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - Fleet coordination
- [shift-left-testing](../shift-left-testing/) - Early defect detection
- [shift-right-testing](../shift-right-testing/) - Production monitoring
- [chaos-engineering-resilience](../chaos-engineering-resilience/) - Resilience testing

---

## Remember

The CI/CD Pipeline QE Orchestrator provides:
- **Phase-based strategy** with optimal skill/agent selection
- **Quality gates** that block bad deployments
- **Adaptive strategy** based on risk and context
- **Full fleet coordination** through memory namespaces

**With Agents:** Use `qe-fleet-commander` for multi-agent orchestration. Coordinate through `aqe/*` memory namespace. Batch operations for efficiency.
