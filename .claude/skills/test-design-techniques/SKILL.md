---
name: test-design-techniques
description: "Systematic test design with boundary value analysis, equivalence partitioning, decision tables, state transition testing, and combinatorial testing. Use when designing comprehensive test cases, reducing redundant tests, or ensuring systematic coverage."
category: specialized-testing
priority: high
tokenEstimate: 900
agents: [qe-test-generator, qe-coverage-analyzer, qe-quality-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [test-design, bva, equivalence-partitioning, decision-tables, pairwise, state-transition]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/test-design-techniques.yaml

---

# Test Design Techniques

<default_to_action>
When designing test cases systematically:
1. APPLY Boundary Value Analysis (test at min, max, edges)
2. USE Equivalence Partitioning (one test per partition)
3. CREATE Decision Tables (for complex business rules)
4. MODEL State Transitions (for stateful behavior)
5. REDUCE with Pairwise Testing (for combinations)

**Quick Design Selection:**
- Numeric ranges → BVA + EP
- Multiple conditions → Decision Tables
- Workflows → State Transition
- Many parameters → Pairwise Testing

**Critical Success Factors:**
- Systematic design finds more bugs with fewer tests
- Random testing is inefficient
- 40+ years of research backs these techniques
</default_to_action>

## Quick Reference Card

### When to Use
- Designing new test suites
- Optimizing existing tests
- Complex business rules
- Reducing test redundancy

### Technique Selection Guide
| Scenario | Technique |
|----------|-----------|
| **Numeric input ranges** | BVA + EP |
| **Multiple conditions** | Decision Tables |
| **Stateful workflows** | State Transition |
| **Many parameter combinations** | Pairwise |
| **All combinations critical** | Full Factorial |

---

## Boundary Value Analysis (BVA)

**Principle:** Bugs cluster at boundaries.

**Test at boundaries:**
- Minimum valid value
- Just below minimum (invalid)
- Just above minimum (valid)
- Maximum valid value
- Just above maximum (invalid)

```javascript
// Age field: 18-120 valid
const boundaryTests = [
  { input: 17, expected: 'invalid' },  // Below min
  { input: 18, expected: 'valid' },    // Min boundary
  { input: 19, expected: 'valid' },    // Above min
  { input: 119, expected: 'valid' },   // Below max
  { input: 120, expected: 'valid' },   // Max boundary
  { input: 121, expected: 'invalid' }  // Above max
];
```

---

## Equivalence Partitioning (EP)

**Principle:** One test per equivalent class.

```javascript
// Discount rules:
// 1-10:  No discount
// 11-100: 10% discount
// 101+:   20% discount

const partitionTests = [
  { quantity: -1, expected: 'invalid' },  // Invalid partition
  { quantity: 5, expected: 0 },           // Partition 1: 1-10
  { quantity: 50, expected: 0.10 },       // Partition 2: 11-100
  { quantity: 200, expected: 0.20 }       // Partition 3: 101+
];

// 4 tests cover all behavior (vs 200+ if testing every value)
```

---

## Decision Tables

**Use for:** Complex business rules with multiple conditions.

```
Loan Approval Rules:
┌──────────────┬───────┬───────┬───────┬───────┬───────┐
│ Conditions   │ R1    │ R2    │ R3    │ R4    │ R5    │
├──────────────┼───────┼───────┼───────┼───────┼───────┤
│ Age ≥ 18     │ Yes   │ Yes   │ Yes   │ No    │ Yes   │
│ Credit ≥ 700 │ Yes   │ Yes   │ No    │ Yes   │ No    │
│ Income ≥ 50k │ Yes   │ No    │ Yes   │ Yes   │ Yes   │
├──────────────┼───────┼───────┼───────┼───────┼───────┤
│ Result       │Approve│Approve│Reject │Reject │Reject │
└──────────────┴───────┴───────┴───────┴───────┴───────┘

// 5 tests cover all decision combinations
```

---

## State Transition Testing

**Model state changes:**

```
States: Logged Out → Logged In → Premium → Suspended

Valid Transitions:
- Login: Logged Out → Logged In
- Upgrade: Logged In → Premium
- Payment Fail: Premium → Suspended
- Logout: Any → Logged Out

Invalid Transitions to Test:
- Logged Out → Premium (should reject)
- Suspended → Premium (should reject)
```

```javascript
test('cannot upgrade without login', async () => {
  const result = await user.upgrade(); // While logged out
  expect(result.error).toBe('Login required');
});
```

---

## Pairwise (Combinatorial) Testing

**Problem:** All combinations explode exponentially.

```javascript
// Parameters:
// Browser: Chrome, Firefox, Safari (3)
// OS: Windows, Mac, Linux (3)
// Screen: Desktop, Tablet, Mobile (3)

// All combinations: 3 × 3 × 3 = 27 tests
// Pairwise: 9 tests cover all pairs

const pairwiseTests = [
  { browser: 'Chrome', os: 'Windows', screen: 'Desktop' },
  { browser: 'Chrome', os: 'Mac', screen: 'Tablet' },
  { browser: 'Chrome', os: 'Linux', screen: 'Mobile' },
  { browser: 'Firefox', os: 'Windows', screen: 'Tablet' },
  { browser: 'Firefox', os: 'Mac', screen: 'Mobile' },
  { browser: 'Firefox', os: 'Linux', screen: 'Desktop' },
  { browser: 'Safari', os: 'Windows', screen: 'Mobile' },
  { browser: 'Safari', os: 'Mac', screen: 'Desktop' },
  { browser: 'Safari', os: 'Linux', screen: 'Tablet' }
];
// Each pair appears at least once
```

---

## Agent-Driven Test Design

```typescript
// Auto-generate BVA tests
await Task("Generate BVA Tests", {
  field: 'age',
  dataType: 'integer',
  constraints: { min: 18, max: 120 }
}, "qe-test-generator");
// Returns: 6 boundary test cases

// Auto-generate pairwise tests
await Task("Generate Pairwise Tests", {
  parameters: {
    browser: ['Chrome', 'Firefox', 'Safari'],
    os: ['Windows', 'Mac', 'Linux'],
    screen: ['Desktop', 'Tablet', 'Mobile']
  }
}, "qe-test-generator");
// Returns: 9-12 tests (vs 27 full combination)
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/test-design/
├── bva-analysis/*       - Boundary value tests
├── partitions/*         - Equivalence partitions
├── decision-tables/*    - Decision table tests
└── pairwise/*           - Combinatorial reduction
```

### Fleet Coordination
```typescript
const designFleet = await FleetManager.coordinate({
  strategy: 'systematic-test-design',
  agents: [
    'qe-test-generator',    // Apply design techniques
    'qe-coverage-analyzer', // Analyze coverage
    'qe-quality-analyzer'   // Assess test quality
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - Agent-driven testing
- [risk-based-testing](../risk-based-testing/) - Prioritize by risk
- [mutation-testing](../mutation-testing/) - Validate test effectiveness

---

## Remember

**Systematic design > Random testing.** 40+ years of research shows these techniques find more bugs with fewer tests than ad-hoc approaches.

**Combine techniques for comprehensive coverage.** BVA for boundaries, EP for partitions, decision tables for rules, pairwise for combinations.

**With Agents:** `qe-test-generator` applies these techniques automatically, generating optimal test suites with maximum coverage and minimum redundancy. Agents identify boundaries, partitions, and combinations from code analysis.
