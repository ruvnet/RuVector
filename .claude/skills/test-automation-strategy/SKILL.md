---
name: test-automation-strategy
description: "Design and implement effective test automation with proper pyramid, patterns, and CI/CD integration. Use when building automation frameworks or improving test efficiency."
category: testing-methodologies
priority: high
tokenEstimate: 1000
agents: [qe-test-generator, qe-test-executor, qe-coverage-analyzer, qe-flaky-test-hunter, qe-regression-risk-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [automation, test-pyramid, page-object, first-principles, ci-cd, flaky-tests]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/test-automation-strategy.yaml

---

# Test Automation Strategy

<default_to_action>
When designing or improving test automation:
1. FOLLOW test pyramid: 70% unit, 20% integration, 10% E2E
2. APPLY F.I.R.S.T. principles: Fast, Isolated, Repeatable, Self-validating, Timely
3. USE patterns: Page Object Model, Builder pattern, Factory pattern
4. INTEGRATE in CI/CD: Every commit runs tests, fail fast, clear feedback
5. MANAGE flaky tests: Quarantine, fix, or delete - never ignore

**Quick Anti-Pattern Detection:**
- Ice cream cone (many E2E, few unit) → Invert to pyramid
- Slow tests (> 10 min suite) → Parallelize, mock external deps
- Flaky tests → Fix timing, isolate data, or quarantine
- Test duplication → Share fixtures, use page objects
- Brittle selectors → Use data-testid, semantic locators

**Critical Success Factors:**
- Fast feedback is the goal (< 10 min full suite)
- Automation supports testing, doesn't replace judgment
- Invest in test infrastructure like production code
</default_to_action>

## Quick Reference Card

### When to Use
- Building new automation framework
- Improving existing test efficiency
- Reducing flaky test burden
- Optimizing CI/CD pipeline speed

### Test Pyramid
| Layer | % | Speed | Isolation | Examples |
|-------|---|-------|-----------|----------|
| **Unit** | 70% | < 1ms | Complete | Pure functions, logic |
| **Integration** | 20% | < 1s | Partial | API, database |
| **E2E** | 10% | < 30s | None | User journeys |

### F.I.R.S.T. Principles
| Principle | Meaning | How |
|-----------|---------|-----|
| **F**ast | Quick execution | Mock external deps |
| **I**solated | No shared state | Fresh fixtures per test |
| **R**epeatable | Same result every time | No random data |
| **S**elf-validating | Clear pass/fail | Assert, don't print |
| **T**imely | Written with code | TDD, not after |

### Anti-Patterns
| Problem | Symptom | Fix |
|---------|---------|-----|
| Ice cream cone | 80% E2E, 10% unit | Invert pyramid |
| Slow suite | 30+ min CI | Parallelize, prune |
| Flaky tests | Random failures | Quarantine, fix timing |
| Coupled tests | Order-dependent | Isolate data |
| Brittle selectors | Break on CSS change | Use data-testid |

---

## Page Object Model

```javascript
// pages/LoginPage.js
class LoginPage {
  constructor(page) {
    this.page = page;
    this.emailInput = '[data-testid="email"]';
    this.passwordInput = '[data-testid="password"]';
    this.submitButton = '[data-testid="submit"]';
    this.errorMessage = '[data-testid="error"]';
  }

  async login(email, password) {
    await this.page.fill(this.emailInput, email);
    await this.page.fill(this.passwordInput, password);
    await this.page.click(this.submitButton);
  }

  async getError() {
    return this.page.textContent(this.errorMessage);
  }
}

// Test uses page object
test('shows error for invalid credentials', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.login('bad@email.com', 'wrong');
  expect(await loginPage.getError()).toBe('Invalid credentials');
});
```

---

## CI/CD Integration

```yaml
name: Test Pipeline
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:unit -- --coverage
        timeout-minutes: 5
      - uses: codecov/codecov-action@v3

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
    steps:
      - run: npm run test:integration
        timeout-minutes: 10

  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - run: npx playwright test
        timeout-minutes: 15
```

---

## Flaky Test Management

```typescript
// Quarantine flaky tests
describe.skip('Quarantined - INC-123', () => {
  test('flaky test awaiting fix', () => { /* ... */ });
});

// Agent-assisted stabilization
await Task("Fix Flaky Tests", {
  tests: quarantinedTests,
  analysis: ['timing-issues', 'data-isolation', 'race-conditions'],
  strategies: ['add-waits', 'isolate-fixtures', 'mock-externals']
}, "qe-flaky-test-hunter");
```

---

## Agent-Assisted Automation

```typescript
// Generate tests following pyramid
await Task("Generate Test Suite", {
  sourceCode: 'src/',
  pyramid: { unit: 70, integration: 20, e2e: 10 },
  patterns: ['page-object', 'builder', 'factory'],
  framework: 'jest'
}, "qe-test-generator");

// Optimize test execution
await Task("Optimize Suite", {
  algorithm: 'johnson-lindenstrauss',
  targetReduction: 0.3,
  maintainCoverage: 0.95
}, "qe-regression-risk-analyzer");

// Analyze flaky patterns
await Task("Flaky Analysis", {
  testHistory: 'last-30-days',
  detectPatterns: ['timing', 'data', 'environment'],
  recommend: 'stabilization-strategy'
}, "qe-flaky-test-hunter");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/automation/
├── test-pyramid/*        - Coverage by layer
├── page-objects/*        - Shared page objects
├── flaky-registry/*      - Quarantined tests
└── execution-metrics/*   - Suite performance data
```

### Fleet Coordination
```typescript
const automationFleet = await FleetManager.coordinate({
  strategy: 'test-automation',
  agents: [
    'qe-test-generator',         // Generate pyramid-compliant tests
    'qe-test-executor',          // Parallel execution
    'qe-coverage-analyzer',      // Coverage gaps
    'qe-flaky-test-hunter',      // Flaky detection
    'qe-regression-risk-analyzer' // Smart selection
  ],
  topology: 'hierarchical'
});
```

---

## Related Skills
- [tdd-london-chicago](../tdd-london-chicago/) - TDD for unit tests
- [api-testing-patterns](../api-testing-patterns/) - Integration patterns
- [cicd-pipeline-qe-orchestrator](../cicd-pipeline-qe-orchestrator/) - Pipeline integration
- [shift-left-testing](../shift-left-testing/) - Early automation

---

## Remember

**Pyramid: 70% unit, 20% integration, 10% E2E.** F.I.R.S.T. principles for every test. Page Object Model for E2E. Parallelize for speed. Quarantine flaky tests - never ignore them. Treat test code like production code.

**With Agents:** Agents generate pyramid-compliant tests, detect flaky patterns, optimize execution time, and maintain test infrastructure. Use agents to scale automation quality.
