# Microservice CI/CD Pipeline Workflow

Complete quality orchestration workflow for microservice architectures.

## Pipeline Configuration

**Service Type**: Microservice (REST API)
**Tech Stack**: Node.js, Express, PostgreSQL
**Deployment**: Kubernetes, Docker
**Frequency**: Multiple deploys per day

---

## Phase 1: Commit (Pre-Merge)

### Quality Strategy
- Fast feedback (<5 minutes)
- Unit tests + static analysis
- API contract validation

### Agent Orchestration

```javascript
// Parallel execution for fast feedback
Task("Unit Tests", "Generate and run unit tests for changed services", "qe-test-generator")
Task("Contract Validation", "Validate service contracts haven't broken", "qe-api-contract-validator")
Task("Static Analysis", "Run static analysis and linting", "code-review-swarm")
```

### Quality Gates
- [ ] Unit coverage > 80% on changed files
- [ ] No API contract breaks
- [ ] No critical static analysis violations
- [ ] Execution time < 5 minutes

---

## Phase 2: Build (Post-Merge)

### Quality Strategy
- Integration testing between services
- Contract testing with consumers
- Test data generation

### Agent Orchestration

```javascript
// Sequential with memory coordination
Task("Integration Tests", "Run integration tests and store results", "qe-test-executor")
Task("Coverage Analysis", "Analyze coverage from integration tests", "qe-coverage-analyzer")
Task("Contract Testing", "Run Pact consumer-driven contract tests", "qe-api-contract-validator")
Task("Data Generation", "Generate realistic test data for downstream services", "qe-test-data-architect")
```

### Quality Gates
- [ ] All integration tests passing
- [ ] Coverage > 85% including integration
- [ ] All consumer contracts validated
- [ ] Test data generated successfully

---

## Phase 3: Integration Environment

### Quality Strategy
- Performance testing under load
- Security scanning (SAST/DAST)
- Chaos testing for resilience

### Agent Orchestration

```javascript
// Parallel execution for comprehensive validation
Task("Load Testing", "Test with 1000 concurrent requests", "qe-performance-tester")
Task("Security Scan", "Run SAST/DAST security checks", "qe-security-scanner")
Task("Chaos Testing", "Inject failures to test resilience", "qe-chaos-engineer")
```

### Quality Gates
- [ ] p95 latency < 100ms
- [ ] p99 latency < 500ms
- [ ] No critical security vulnerabilities
- [ ] Service recovers from injected failures

---

## Phase 4: Staging

### Quality Strategy
- End-to-end testing across services
- Visual regression for admin UI (if applicable)
- Deployment readiness assessment

### Agent Orchestration

```javascript
// Staging validation
Task("E2E Tests", "Run end-to-end user journeys", "qe-test-executor")
Task("Visual Testing", "Check admin UI for regressions", "qe-visual-tester")
Task("Deployment Check", "Assess production readiness", "qe-deployment-readiness")
```

### Quality Gates
- [ ] All E2E scenarios passing
- [ ] No visual regressions
- [ ] Deployment readiness score > 90%
- [ ] Rollback plan validated

---

## Phase 5: Production

### Quality Strategy
- Canary deployment with synthetic monitoring
- Production intelligence collection
- Real-time quality metrics

### Agent Orchestration

```javascript
// Production monitoring
Task("Production Intelligence", "Monitor canary and collect metrics", "qe-production-intelligence")
Task("Quality Analysis", "Track production quality metrics", "qe-quality-analyzer")
```

### Quality Gates (Canary)
- [ ] Error rate < 0.1%
- [ ] Latency within 5% of baseline
- [ ] No increase in 5xx errors
- [ ] Synthetic monitors passing

---

## Skill Integration

### Skills Used by Phase

**Commit Phase**:
- `shift-left-testing`
- `tdd-london-chicago`
- `api-testing-patterns`

**Build Phase**:
- `contract-testing`
- `test-automation-strategy`
- `test-data-management`

**Integration Phase**:
- `performance-testing`
- `security-testing`
- `chaos-engineering-resilience`

**Staging Phase**:
- `shift-right-testing`
- `visual-testing-advanced`

**Production Phase**:
- `shift-right-testing`
- `quality-metrics`

---

## Complete Pipeline Code

```javascript
// Full microservice pipeline orchestration

// PHASE 1: Commit
const commitAgents = [
  Task("Unit Tests", "Generate tests for UserService", "qe-test-generator"),
  Task("Contract Validation", "Check API contracts", "qe-api-contract-validator"),
  Task("Code Review", "Automated code review", "code-review-swarm")
];

// Wait for commit gates to pass...

// PHASE 2: Build
const buildAgents = [
  Task("Integration Tests", "Run service integration tests", "qe-test-executor"),
  Task("Coverage Check", "Analyze integration coverage", "qe-coverage-analyzer"),
  Task("Pact Tests", "Validate consumer contracts", "qe-api-contract-validator")
];

// Wait for build gates to pass...

// PHASE 3: Integration
const integrationAgents = [
  Task("Load Test", "1000 concurrent users", "qe-performance-tester"),
  Task("Security Scan", "SAST/DAST checks", "qe-security-scanner"),
  Task("Chaos Test", "Inject network failures", "qe-chaos-engineer")
];

// Wait for integration gates to pass...

// PHASE 4: Staging
const stagingAgents = [
  Task("E2E Tests", "User journey validation", "qe-test-executor"),
  Task("Visual Tests", "UI regression check", "qe-visual-tester"),
  Task("Ready Check", "Deployment assessment", "qe-deployment-readiness")
];

// Wait for staging gates to pass...

// PHASE 5: Production (Canary)
const productionAgents = [
  Task("Monitor Canary", "Collect production metrics", "qe-production-intelligence"),
  Task("Quality Metrics", "Track quality trends", "qe-quality-analyzer")
];
```

---

## Timing Budget

| Phase | Target Time | Max Time |
|-------|-------------|----------|
| Commit | 3 min | 5 min |
| Build | 8 min | 15 min |
| Integration | 15 min | 30 min |
| Staging | 10 min | 20 min |
| Production (Canary) | 30 min | 60 min |
| **Total** | **66 min** | **130 min** |

---

## Rollback Strategy

If any phase fails:

1. **Commit/Build**: Block merge, developer fixes
2. **Integration**: Auto-rollback, investigate
3. **Staging**: Block deployment, team review
4. **Production**: Immediate rollback, incident created

---

**Use Cases**: REST APIs, GraphQL services, Event-driven services
**Team Size**: 2-5 developers per service
**Deploy Frequency**: 5-20 deploys/day
