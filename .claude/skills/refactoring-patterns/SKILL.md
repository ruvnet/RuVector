---
name: refactoring-patterns
description: "Apply safe refactoring patterns to improve code structure without changing behavior. Use when cleaning up code, reducing technical debt, or improving maintainability."
category: development-practices
priority: medium
tokenEstimate: 1000
agents: [qe-code-reviewer, qe-quality-analyzer, qe-test-refactorer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [refactoring, code-quality, technical-debt, maintainability, clean-code]
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# Refactoring Patterns

<default_to_action>
When refactoring:
1. ENSURE tests pass (never refactor without tests)
2. MAKE small change (one refactoring at a time)
3. RUN tests (must stay green)
4. COMMIT (save progress)
5. REPEAT

**Safe Refactoring Cycle:**
```bash
npm test               # Green ✅
# Make ONE small change
npm test               # Still green ✅
git commit -m "refactor: extract calculateTotal"
# Repeat
```

**Code Smells → Refactoring:**
| Smell | Refactoring |
|-------|-------------|
| Long method (>20 lines) | Extract Method |
| Large class | Extract Class |
| Long parameter list (>3) | Introduce Parameter Object |
| Duplicated code | Extract Method/Class |
| Complex conditional | Decompose Conditional |
| Magic numbers | Named Constants |
| Nested loops | Replace Loop with Pipeline |

**NEVER REFACTOR:**
- Without tests (write tests first)
- When deadline is tomorrow
- Code you don't understand
- Code that works and won't be touched
</default_to_action>

## Quick Reference Card

### Common Refactorings

| Pattern | Before | After |
|---------|--------|-------|
| **Extract Method** | 50-line function | 5 small functions |
| **Extract Class** | Class doing 5 things | 5 single-purpose classes |
| **Parameter Object** | `fn(a,b,c,d,e,f)` | `fn(options)` |
| **Replace Conditional** | `if (type === 'a') {...}` | Polymorphism |
| **Pipeline** | Nested loops | `.filter().map().reduce()` |

### The Rule of Three

1. First time → Just do it
2. Second time → Wince and duplicate
3. Third time → **Refactor**

---

## Key Patterns

### Extract Method
```javascript
// Before: Long method
function processOrder(order) {
  // 50 lines of validation, calculation, saving, emailing...
}

// After: Clear responsibilities
function processOrder(order) {
  validateOrder(order);
  const pricing = calculatePricing(order);
  const saved = saveOrder(order, pricing);
  sendConfirmationEmail(saved);
  return saved;
}
```

### Replace Loop with Pipeline
```javascript
// Before
let results = [];
for (let item of items) {
  if (item.inStock) {
    results.push(item.name.toUpperCase());
  }
}

// After
const results = items
  .filter(item => item.inStock)
  .map(item => item.name.toUpperCase());
```

### Decompose Conditional
```javascript
// Before
if (order.total > 1000 && customer.isPremium && allInStock(order)) {
  return 'FREE_SHIPPING';
}

// After
function isEligibleForFreeShipping(order, customer) {
  return isLargeOrder(order) &&
         isPremiumCustomer(customer) &&
         allInStock(order);
}
```

---

## Refactoring Anti-Patterns

| ❌ Anti-Pattern | Problem | ✅ Better |
|-----------------|---------|-----------|
| Without tests | No safety net | Write tests first |
| Big bang | Rewrite everything | Small incremental steps |
| For perfection | Endless tweaking | Good enough, move on |
| Premature abstraction | Pattern not clear yet | Wait for Rule of Three |
| During feature work | Mixed changes | Separate commits |

---

## Agent Integration

```typescript
// Detect code smells
const smells = await Task("Detect Code Smells", {
  source: 'src/services/',
  patterns: ['long-method', 'large-class', 'duplicate-code']
}, "qe-quality-analyzer");

// Safe refactoring with test verification
await Task("Verify Refactoring", {
  beforeCommit: 'abc123',
  afterCommit: 'def456',
  expectSameBehavior: true
}, "qe-test-executor");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/refactoring/
├── smells/*          - Detected code smells
├── suggestions/*     - Refactoring recommendations
├── verifications/*   - Behavior preservation checks
└── history/*         - Refactoring log
```

### Fleet Coordination
```typescript
const refactoringFleet = await FleetManager.coordinate({
  strategy: 'refactoring',
  agents: [
    'qe-quality-analyzer',   // Identify targets
    'qe-test-generator',     // Add safety tests
    'qe-test-executor',      // Verify behavior
    'qe-test-refactorer'     // TDD refactor phase
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [tdd-london-chicago](../tdd-london-chicago/) - TDD refactor phase
- [code-review-quality](../code-review-quality/) - Review refactored code
- [xp-practices](../xp-practices/) - Collective ownership

---

## Remember

**Refactoring is NOT:**
- Adding features
- Fixing bugs
- Performance optimization
- Rewriting from scratch

**Refactoring IS:**
- Improving structure
- Making code clearer
- Reducing complexity
- Removing duplication
- **Without changing behavior**

**Always have tests. Always take small steps. Always keep tests green.**
