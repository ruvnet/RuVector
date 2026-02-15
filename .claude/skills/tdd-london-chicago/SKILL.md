---
name: tdd-london-chicago
description: "Apply London (mock-based) and Chicago (state-based) TDD schools. Use when practicing test-driven development or choosing testing style for your context."
category: development-practices
priority: high
tokenEstimate: 1100
agents: [qe-test-generator, qe-test-implementer, qe-test-refactorer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [tdd, testing, london-school, chicago-school, red-green-refactor, mocks]
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# Test-Driven Development: London & Chicago Schools

<default_to_action>
When implementing TDD or choosing testing style:
1. IDENTIFY code type: domain logic → Chicago, external deps → London
2. WRITE failing test first (Red phase)
3. IMPLEMENT minimal code to pass (Green phase)
4. REFACTOR while keeping tests green (Refactor phase)
5. REPEAT cycle for next functionality

**Quick Style Selection:**
- Pure functions/calculations → Chicago (real objects, state verification)
- Controllers/services with deps → London (mocks, interaction verification)
- Value objects → Chicago (test final state)
- API integrations → London (mock external services)
- Mix both in practice (London for controllers, Chicago for domain)

**Critical Success Factors:**
- Tests drive design, not just verify it
- Make tests fail first to ensure they test something
- Write minimal code - no features beyond what's tested
</default_to_action>

## Quick Reference Card

### When to Use
- Starting new feature with test-first approach
- Refactoring legacy code with test coverage
- Teaching TDD practices to team
- Choosing between mocking vs real objects

### TDD Cycle
| Phase | Action | Discipline |
|-------|--------|------------|
| **Red** | Write failing test | Verify it fails, check message is clear |
| **Green** | Minimal code to pass | No extra features, don't refactor |
| **Refactor** | Improve structure | Keep tests passing, no new functionality |

### School Comparison
| Aspect | Chicago (Classicist) | London (Mockist) |
|--------|---------------------|------------------|
| Collaborators | Real objects | Mocks/stubs |
| Verification | State (assert outcomes) | Interaction (assert calls) |
| Isolation | Lower (integrated) | Higher (unit only) |
| Refactoring | Easier | Harder (mocks break) |
| Design feedback | Emerges from use | Explicit from start |

### Agent Coordination
- `qe-test-generator`: Generate tests in both schools
- `qe-test-implementer`: Implement minimal code (Green)
- `qe-test-refactorer`: Safe refactoring (Refactor)

---

## Chicago School (State-Based)

**Philosophy:** Test observable behavior through public API. Keep tests close to consumer usage.

```javascript
// State verification - test final outcome
describe('Order', () => {
  it('calculates total with tax', () => {
    const order = new Order();
    order.addItem(new Product('Widget', 10.00), 2);
    order.addItem(new Product('Gadget', 15.00), 1);

    expect(order.totalWithTax(0.10)).toBe(38.50);
  });
});
```

**When Chicago Shines:**
- Domain logic with clear state
- Algorithms and calculations
- Value objects (`Money`, `Email`)
- Simple collaborations
- Learning new domain

---

## London School (Mock-Based)

**Philosophy:** Test each unit in isolation. Focus on how objects collaborate.

```javascript
// Interaction verification - test method calls
describe('Order', () => {
  it('delegates tax calculation', () => {
    const taxCalculator = {
      calculateTax: jest.fn().mockReturnValue(3.50)
    };
    const order = new Order(taxCalculator);
    order.addItem({ price: 10 }, 2);

    order.totalWithTax();

    expect(taxCalculator.calculateTax).toHaveBeenCalledWith(20.00);
  });
});
```

**When London Shines:**
- External integrations (DB, APIs)
- Command patterns with side effects
- Complex workflows
- Slow operations (network, I/O)

---

## Mixed Approach (Recommended)

```javascript
// London for controller (external deps)
describe('OrderController', () => {
  it('creates order and sends confirmation', async () => {
    const orderService = { create: jest.fn().mockResolvedValue({ id: 123 }) };
    const emailService = { send: jest.fn() };

    const controller = new OrderController(orderService, emailService);
    await controller.placeOrder(orderData);

    expect(orderService.create).toHaveBeenCalledWith(orderData);
    expect(emailService.send).toHaveBeenCalled();
  });
});

// Chicago for domain logic
describe('OrderService', () => {
  it('applies discount when threshold met', () => {
    const service = new OrderService();
    const order = service.create({ items: [...], total: 150 });

    expect(order.discount).toBe(15); // 10% off > $100
  });
});
```

---

## Common Pitfalls

### ❌ Over-Mocking (London)
```javascript
// BAD - mocking everything
const product = { getName: jest.fn(), getPrice: jest.fn() };
```
**Better:** Only mock external dependencies.

### ❌ Mocking Internals
```javascript
// BAD - testing private methods
expect(order._calculateSubtotal).toHaveBeenCalled();
```
**Better:** Test public behavior only.

### ❌ Test Pain = Design Pain
- Need many mocks? → Too many dependencies
- Hard to set up? → Constructor does too much
- Can't test without database? → Coupling issue

---

## Agent-Assisted TDD

```typescript
// Agent generates tests in both schools
await Task("Generate Tests", {
  style: 'chicago',      // or 'london'
  target: 'src/domain/Order.ts',
  focus: 'state-verification'  // or 'collaboration-patterns'
}, "qe-test-generator");

// Agent-human ping-pong TDD
// Human writes test concept
const testIdea = "Order applies 10% discount when total > $100";

// Agent generates formal failing test (Red)
await Task("Create Failing Test", testIdea, "qe-test-generator");

// Human writes minimal code (Green)

// Agent suggests refactorings
await Task("Suggest Refactorings", { preserveTests: true }, "qe-test-refactorer");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/tdd/
├── test-plan/*        - TDD session plans
├── red-phase/*        - Failing tests generated
├── green-phase/*      - Implementation code
└── refactor-phase/*   - Refactoring suggestions
```

### Fleet Coordination
```typescript
const tddFleet = await FleetManager.coordinate({
  workflow: 'red-green-refactor',
  agents: {
    testGenerator: 'qe-test-generator',
    testExecutor: 'qe-test-executor',
    qualityAnalyzer: 'qe-quality-analyzer'
  },
  mode: 'sequential'
});
```

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - TDD with agent coordination
- [refactoring-patterns](../refactoring-patterns/) - Refactor phase techniques
- [api-testing-patterns](../api-testing-patterns/) - London school for API testing

---

## Remember

**Chicago:** Test state, use real objects, refactor freely
**London:** Test interactions, mock dependencies, design interfaces first
**Both:** Write the test first, make it pass, refactor

Neither is "right." Choose based on context. Mix as needed. Goal: well-designed, tested code.

**With Agents:** Agents excel at generating tests, validating green phase, and suggesting refactorings. Use agents to maintain TDD discipline while humans focus on design decisions.
