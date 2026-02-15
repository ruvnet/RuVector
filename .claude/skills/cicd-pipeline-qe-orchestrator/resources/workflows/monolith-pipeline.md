# Monolith CI/CD Pipeline Workflow

Complete quality orchestration workflow for monolithic applications.

## Pipeline Configuration

**Service Type**: Monolithic Web Application
**Tech Stack**: Full-stack (Frontend + Backend)
**Deployment**: Traditional servers, VM-based
**Frequency**: Weekly releases

---

## Phase 1: Commit (Pre-Merge)

### Quality Strategy
- Fast feedback on changed modules
- Component-level testing
- Risk-based test selection

### Agent Orchestration

```javascript
// Smart test selection for large codebase
Task("Risk Analysis", "Identify affected modules and tests", "qe-regression-risk-analyzer")
Task("Targeted Tests", "Run minimal test suite based on changes", "qe-test-executor")
Task("Flaky Detection", "Check for flaky tests in suite", "qe-flaky-test-hunter")
```

### Quality Gates
- [ ] Affected modules tested (100% coverage)
- [ ] No flaky tests in selection
- [ ] No breaking changes to public APIs
- [ ] Execution time < 10 minutes

---

## Phase 2: Build (Full Suite)

### Quality Strategy
- Comprehensive regression testing
- Full coverage analysis
- Mutation testing for critical paths

### Agent Orchestration

```javascript
// Full regression suite (batched execution)
Task("Full Regression", "Run complete test suite in batches", "qe-test-executor")
Task("Coverage Analysis", "Analyze coverage across all modules", "qe-coverage-analyzer")
Task("Mutation Testing", "Validate test quality on critical code", "qe-test-executor")
```

### Quality Gates
- [ ] All 1000+ tests passing
- [ ] Coverage > 85% overall
- [ ] Coverage > 95% on critical paths
- [ ] Mutation score > 75%

---

## Phase 3: Integration / Test Environment

### Quality Strategy
- Database migration testing
- Performance baseline validation
- Security compliance scanning
- Accessibility validation

### Agent Orchestration

```javascript
// Comprehensive validation
Task("DB Migration", "Validate database schema changes", "qe-test-executor")
Task("Performance Baseline", "Establish performance benchmarks", "qe-performance-tester")
Task("Security Compliance", "OWASP Top 10 + compliance checks", "qe-security-scanner")
Task("Accessibility", "WCAG 2.2 validation", "qe-visual-tester")
```

### Quality Gates
- [ ] Database migrations successful
- [ ] Performance within 10% of baseline
- [ ] No critical/high security vulnerabilities
- [ ] WCAG AA compliance achieved

---

## Phase 4: UAT / Staging

### Quality Strategy
- Exploratory testing sessions
- Visual regression across browsers
- Mobile compatibility
- User acceptance criteria validation

### Agent Orchestration

```javascript
// User acceptance and compatibility
Task("Visual Regression", "Test across Chrome, Firefox, Safari, Edge", "qe-visual-tester")
Task("Mobile Testing", "iOS and Android compatibility", "qe-visual-tester")
Task("Deployment Readiness", "Pre-production assessment", "qe-deployment-readiness")
```

### Skills for Manual Testing
```javascript
// Invoke skills for exploratory testing guidance
Skill("exploratory-testing-advanced")  // SBTM session planning
Skill("context-driven-testing")        // Adapt testing to context
```

### Quality Gates
- [ ] Visual tests passing (4 browsers)
- [ ] Mobile responsive (iOS/Android)
- [ ] UAT sign-off received
- [ ] Deployment checklist complete

---

## Phase 5: Production

### Quality Strategy
- Blue-green deployment
- Smoke tests in production
- Compliance validation
- Production monitoring

### Agent Orchestration

```javascript
// Production validation
Task("Smoke Tests", "Run critical path smoke tests", "qe-test-executor")
Task("Compliance Check", "GDPR, HIPAA, SOC2 validation", "qe-quality-gate")
Task("Production Intelligence", "Monitor first hour metrics", "qe-production-intelligence")
```

### Quality Gates
- [ ] All smoke tests passing
- [ ] Compliance checks passed
- [ ] Error rate < 0.5%
- [ ] No user-reported critical bugs (first 24h)

---

## Skill Integration

### Skills Used by Phase

**Commit Phase**:
- `shift-left-testing`
- `risk-based-testing`
- `regression-testing`

**Build Phase**:
- `test-automation-strategy`
- `mutation-testing`
- `refactoring-patterns`

**Integration Phase**:
- `database-testing`
- `performance-testing`
- `security-testing`
- `accessibility-testing`

**UAT/Staging Phase**:
- `exploratory-testing-advanced`
- `context-driven-testing`
- `compatibility-testing`
- `visual-testing-advanced`

**Production Phase**:
- `shift-right-testing`
- `compliance-testing`
- `quality-metrics`

---

## Complete Pipeline Code

```javascript
// Full monolith pipeline orchestration

// PHASE 1: Commit (Smart Selection)
Task("Risk Analysis", "Select minimal test suite for changes", "qe-regression-risk-analyzer")
Task("Targeted Testing", "Run selected tests", "qe-test-executor")

// PHASE 2: Build (Full Suite)
Task("Full Regression", "Run all 1000+ tests in batches", "qe-test-executor")
Task("Coverage Analysis", "Analyze full coverage", "qe-coverage-analyzer")
Task("Mutation Testing", "Test quality validation", "qe-test-executor")

// PHASE 3: Integration
Task("DB Migrations", "Validate schema changes", "qe-test-executor")
Task("Performance Test", "Baseline performance validation", "qe-performance-tester")
Task("Security Scan", "OWASP + compliance scanning", "qe-security-scanner")
Task("Accessibility", "WCAG 2.2 validation", "qe-visual-tester")

// PHASE 4: UAT/Staging
Task("Visual Regression", "Cross-browser testing", "qe-visual-tester")
Task("Mobile Testing", "iOS/Android compatibility", "qe-visual-tester")
Task("Readiness Check", "Deployment assessment", "qe-deployment-readiness")

// Manual testing with skill guidance
Skill("exploratory-testing-advanced")

// PHASE 5: Production
Task("Smoke Tests", "Critical path validation", "qe-test-executor")
Task("Compliance", "Regulatory validation", "qe-quality-gate")
Task("Monitoring", "Production intelligence", "qe-production-intelligence")
```

---

## Timing Budget

| Phase | Target Time | Max Time |
|-------|-------------|----------|
| Commit | 8 min | 15 min |
| Build | 45 min | 90 min |
| Integration | 60 min | 120 min |
| UAT/Staging | 4 hours (+ manual) | 8 hours |
| Production Deploy | 30 min | 60 min |
| **Total** | **~6 hours** | **~12 hours** |

---

## Weekly Release Schedule

**Monday**: Freeze features, start testing
**Tuesday-Wednesday**: Integration + UAT
**Thursday**: Staging validation
**Friday**: Production deployment (morning)
**Weekend**: Monitor production

---

## Database Migration Strategy

### Pre-Deploy Validation

```javascript
// Validate migrations before deployment
Task("Schema Validation", "Check migrations are reversible", "qe-test-executor")
Task("Data Integrity", "Validate data after migration", "qe-test-executor")
```

### Post-Deploy Verification

```javascript
// Verify migrations in production
Task("Production Validation", "Check schema and data integrity", "qe-production-intelligence")
```

---

## Rollback Strategy

**Build Failure**: Fix and rebuild
**Integration Failure**: Investigate, may delay release
**UAT Failure**: Block release, fix critical issues
**Production Issues**: Blue-green rollback, investigate

---

**Use Cases**: E-commerce, CMS, Enterprise web apps
**Team Size**: 10-50 developers
**Deploy Frequency**: Weekly releases
**Database**: PostgreSQL, MySQL with migrations
