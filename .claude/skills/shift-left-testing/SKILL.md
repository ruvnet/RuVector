---
name: shift-left-testing
description: "Move testing activities earlier in the development lifecycle to catch defects when they're cheapest to fix. Use when implementing TDD, CI/CD, or early quality practices."
category: testing-methodologies
priority: high
tokenEstimate: 900
agents: [qe-test-generator, qe-requirements-validator, qe-regression-risk-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [shift-left, early-testing, tdd, bdd, ci-cd, prevention]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/shift-left-testing.yaml

---

# Shift-Left Testing

<default_to_action>
When implementing early testing practices:
1. VALIDATE requirements before coding (testability, BDD scenarios)
2. WRITE tests before implementation (TDD red-green-refactor)
3. AUTOMATE in CI pipeline (every commit triggers tests)
4. FOLLOW test pyramid: Many unit (70%), some integration (20%), few E2E (10%)
5. FIX defects immediately - never let them accumulate

**Quick Shift-Left Levels:**
- Level 1: Unit tests with each PR (developer responsibility)
- Level 2: TDD practice (tests before code)
- Level 3: BDD/Example mapping in refinement (requirements testing)
- Level 4: Risk analysis in design (architecture testing)

**Critical Success Factors:**
- Defects found in requirements cost 1x; in production cost 100x
- Every commit must run automated tests
- Quality is built in, not tested in
</default_to_action>

## Quick Reference Card

### When to Use
- Reducing cost of defects
- Implementing CI/CD pipelines
- Starting TDD practice
- Improving requirements quality

### Cost of Defects by Phase
| Phase | Relative Cost | Example |
|-------|--------------|---------|
| Requirements | 1x | Fix doc: 30 min |
| Design | 5x | Redesign: few hours |
| Development | 10x | Code fix: 1 day |
| Testing | 20x | Fix + retest: 2 days |
| Production | 100x | Hotfix + rollback + investigation |

### Test Pyramid
```
       /\        E2E (10%) - Critical user journeys
      /  \
     /    \      Integration (20%) - Component interactions
    /      \
   /________\    Unit (70%) - Fast, isolated, comprehensive
```

### Shift-Left Levels
| Level | Practice | When |
|-------|----------|------|
| 1 | Unit tests in PR | Before merge |
| 2 | TDD | Before implementation |
| 3 | BDD/Example Mapping | During refinement |
| 4 | Risk Analysis | During design |

---

## Level 1: Tests in Every PR

```yaml
# CI pipeline - tests run on every commit
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:unit
      - run: npm run test:integration
      - run: npm run lint

  quality-gate:
    needs: test
    steps:
      - name: Coverage Check
        run: npx coverage-check --min 80
      - name: No New Warnings
        run: npm run lint -- --max-warnings 0
```

---

## Level 2: TDD Practice

```javascript
// Red: Write failing test first
test('calculates discount for orders over $100', () => {
  const order = new Order([{ price: 150 }]);
  expect(order.discount).toBe(15); // 10% off
});

// Green: Minimal implementation
class Order {
  get discount() {
    return this.total > 100 ? this.total * 0.1 : 0;
  }
}

// Refactor: Improve while keeping green
```

---

## Level 3: BDD in Refinement

```gherkin
# Example mapping before coding
Feature: Loyalty Discount
  Scenario: Gold member gets 15% discount
    Given a customer with "gold" membership
    When they checkout with $200 in cart
    Then discount applied is $30
    And order total is $170

  Scenario: New customer gets no discount
    Given a customer with no membership
    When they checkout with $200 in cart
    Then no discount is applied
```

---

## Level 4: Risk Analysis in Design

```typescript
// During architecture review
await Task("Risk Analysis", {
  phase: 'design',
  component: 'payment-service',
  questions: [
    'What happens when payment gateway times out?',
    'How do we handle duplicate submissions?',
    'What if inventory changed during checkout?'
  ]
}, "qe-requirements-validator");

// Output: Testability requirements, failure mode tests
```

---

## Agent-Assisted Shift-Left

```typescript
// Validate requirements testability
await Task("Requirements Validation", {
  requirements: userStories,
  check: ['INVEST-criteria', 'testability', 'ambiguity'],
  generateBDD: true
}, "qe-requirements-validator");

// Generate tests from requirements
await Task("Generate Tests", {
  source: 'requirements',
  types: ['unit', 'integration', 'e2e'],
  coverage: 'comprehensive'
}, "qe-test-generator");

// Smart test selection for changes
await Task("Select Regression Tests", {
  changedFiles: prFiles,
  algorithm: 'risk-based',
  targetReduction: 0.7  // 70% time savings
}, "qe-regression-risk-analyzer");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/shift-left/
├── requirements/*       - Validated requirements
├── generated-tests/*    - Auto-generated tests
├── coverage-targets/*   - Coverage goals by component
└── pipeline-results/*   - CI/CD test history
```

### Fleet Coordination
```typescript
const shiftLeftFleet = await FleetManager.coordinate({
  strategy: 'shift-left',
  agents: [
    'qe-requirements-validator',     // Level 3-4
    'qe-test-generator',             // Level 2
    'qe-regression-risk-analyzer'    // Smart selection
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [tdd-london-chicago](../tdd-london-chicago/) - TDD practices
- [holistic-testing-pact](../holistic-testing-pact/) - Proactive testing
- [cicd-pipeline-qe-orchestrator](../cicd-pipeline-qe-orchestrator/) - Pipeline integration
- [shift-right-testing](../shift-right-testing/) - Production feedback

---

## Remember

**Earlier = Cheaper.** Requirements defects cost 1x; production defects cost 100x. Test pyramid: 70% unit, 20% integration, 10% E2E. Every commit runs tests. TDD builds quality in.

**With Agents:** Agents validate requirements testability, generate tests from specs, and select optimal regression suites. Use agents to implement shift-left practices consistently.
