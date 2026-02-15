---
name: api-testing-patterns
description: "Comprehensive API testing patterns including contract testing, REST/GraphQL testing, and integration testing. Use when testing APIs or designing API test strategies."
category: testing-methodologies
priority: high
tokenEstimate: 1200
agents: [qe-api-contract-validator, qe-test-generator, qe-performance-tester, qe-security-scanner]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [api, rest, graphql, contract-testing, pact, integration, microservices]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/api-testing-patterns.yaml
---

# API Testing Patterns

<default_to_action>
When testing APIs or designing API test strategy:
1. IDENTIFY testing level: contract, integration, or component
2. TEST the contract, not implementation (consumer perspective)
3. VALIDATE auth, input, errors, idempotency, concurrency
4. AUTOMATE in CI/CD with schema validation
5. MONITOR production APIs for contract drift

**Quick Pattern Selection:**
- Microservices → Consumer-driven contracts (Pact)
- REST APIs → CRUD + pagination + filtering tests
- GraphQL → Query validation + complexity limits
- External deps → Mock with component testing
- Performance → Load test critical endpoints

**Critical Success Factors:**
- APIs are contracts - test from consumer perspective
- Always test error scenarios, not just happy paths
- Version your API tests to prevent breaking changes
</default_to_action>

## Quick Reference Card

### When to Use
- Testing REST or GraphQL APIs
- Validating microservice contracts
- Designing API test strategies
- Preventing breaking API changes

### Testing Levels
| Level | Purpose | Dependencies | Speed |
|-------|---------|--------------|-------|
| Contract | Provider-consumer agreement | None | Fast |
| Component | API in isolation | Mocked | Fast |
| Integration | Real dependencies | Database, services | Slower |

### Critical Test Scenarios
| Scenario | Must Test | Example |
|----------|----------|---------|
| Auth | 401/403 handling | Expired token, wrong user |
| Input | 400 validation | Missing fields, wrong types |
| Errors | 500 graceful handling | DB down, timeout |
| Idempotency | Duplicate prevention | Same idempotency key |
| Concurrency | Race conditions | Parallel checkout |

### Tools
- **Contract**: Pact, Spring Cloud Contract
- **REST**: Supertest, REST-assured, Playwright
- **Load**: k6, Artillery, JMeter

### Agent Coordination
- `qe-api-contract-validator`: Validate contracts, detect breaking changes
- `qe-test-generator`: Generate tests from OpenAPI spec
- `qe-performance-tester`: Load test endpoints
- `qe-security-scanner`: API security testing

---

## Contract Testing

**Pattern: Consumer-Driven Contracts**
```javascript
// Consumer defines expectations
const contract = {
  request: { method: 'POST', path: '/orders', body: { productId: 'abc', quantity: 2 } },
  response: { status: 201, body: { orderId: 'string', total: 'number' } }
};

// Provider must fulfill
test('order API meets contract', async () => {
  const response = await api.post('/orders', { productId: 'abc', quantity: 2 });

  expect(response.status).toBe(201);
  expect(response.body).toMatchSchema({
    orderId: expect.any(String),
    total: expect.any(Number)
  });
});
```

**When:** Microservices, distributed systems, third-party integrations

---

## Critical Test Patterns

### Authentication & Authorization
```javascript
describe('Auth', () => {
  it('rejects without token', async () => {
    expect((await api.get('/orders')).status).toBe(401);
  });

  it('rejects expired token', async () => {
    const expired = generateExpiredToken();
    expect((await api.get('/orders', { headers: { Authorization: `Bearer ${expired}` } })).status).toBe(401);
  });

  it('blocks cross-user access', async () => {
    const userAToken = generateToken({ userId: 'A' });
    expect((await api.get('/orders/user-B-order', { headers: { Authorization: `Bearer ${userAToken}` } })).status).toBe(403);
  });
});
```

### Input Validation
```javascript
describe('Validation', () => {
  it('validates required fields', async () => {
    const response = await api.post('/orders', { quantity: 2 }); // Missing productId
    expect(response.status).toBe(400);
    expect(response.body.errors).toContain('productId is required');
  });

  it('validates types', async () => {
    expect((await api.post('/orders', { productId: 'abc', quantity: 'two' })).status).toBe(400);
  });

  it('validates ranges', async () => {
    expect((await api.post('/orders', { productId: 'abc', quantity: -5 })).status).toBe(400);
  });
});
```

### Idempotency
```javascript
it('prevents duplicates with idempotency key', async () => {
  const key = 'unique-123';
  const data = { productId: 'abc', quantity: 2 };

  const r1 = await api.post('/orders', data, { headers: { 'Idempotency-Key': key } });
  const r2 = await api.post('/orders', data, { headers: { 'Idempotency-Key': key } });

  expect(r1.body.orderId).toBe(r2.body.orderId); // Same order
});
```

### Concurrency
```javascript
it('handles race condition on inventory', async () => {
  const promises = Array(10).fill().map(() =>
    api.post('/orders', { productId: 'abc', quantity: 1 })
  );
  const responses = await Promise.all(promises);
  const successful = responses.filter(r => r.status === 201);

  const inventory = await db.inventory.findById('abc');
  expect(inventory.quantity).toBe(initialQuantity - successful.length);
});
```

---

## REST CRUD Pattern
```javascript
describe('Product CRUD', () => {
  let productId;

  it('CREATE', async () => {
    const r = await api.post('/products', { name: 'Widget', price: 10 });
    expect(r.status).toBe(201);
    productId = r.body.id;
  });

  it('READ', async () => {
    const r = await api.get(`/products/${productId}`);
    expect(r.body.name).toBe('Widget');
  });

  it('UPDATE', async () => {
    const r = await api.put(`/products/${productId}`, { price: 12 });
    expect(r.body.price).toBe(12);
  });

  it('DELETE', async () => {
    expect((await api.delete(`/products/${productId}`)).status).toBe(204);
    expect((await api.get(`/products/${productId}`)).status).toBe(404);
  });
});
```

---

## Best Practices

### ✅ Do This
- Test from consumer perspective
- Use schema validation (not exact values)
- Test error scenarios extensively
- Version API tests
- Automate in CI/CD

### ❌ Avoid This
- Testing implementation, not contract
- Ignoring HTTP semantics (status codes)
- No negative testing
- Asserting on field order or extra fields
- Slow tests (mock external services)

---

## Agent-Assisted API Testing

```typescript
// Validate contracts
await Task("Contract Validation", {
  spec: 'openapi.yaml',
  endpoint: '/orders',
  checkBreakingChanges: true
}, "qe-api-contract-validator");

// Generate tests from spec
await Task("Generate API Tests", {
  spec: 'openapi.yaml',
  coverage: 'comprehensive',
  include: ['happy-paths', 'input-validation', 'auth-scenarios', 'error-handling']
}, "qe-test-generator");

// Load test
await Task("API Load Test", {
  endpoint: '/orders',
  rps: 1000,
  duration: '5min'
}, "qe-performance-tester");

// Security scan
await Task("API Security Scan", {
  spec: 'openapi.yaml',
  checks: ['sql-injection', 'xss', 'broken-auth', 'rate-limiting']
}, "qe-security-scanner");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/api-testing/
├── contracts/*        - API contract definitions
├── generated-tests/*  - Generated test suites
├── validation/*       - Contract validation results
└── performance/*      - Load test results
```

### Fleet Coordination
```typescript
const apiFleet = await FleetManager.coordinate({
  strategy: 'contract-testing',
  agents: ['qe-api-contract-validator', 'qe-test-generator', 'qe-test-executor'],
  topology: 'mesh'
});

await apiFleet.execute({
  services: [
    { name: 'orders-api', consumers: ['checkout-ui', 'admin-api'] },
    { name: 'payment-api', consumers: ['orders-api'] }
  ]
});
```

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - API testing with agents
- [tdd-london-chicago](../tdd-london-chicago/) - London school for API testing
- [performance-testing](../performance-testing/) - API load testing
- [security-testing](../security-testing/) - API security validation
- [contract-testing](../contract-testing/) - Consumer-driven contracts deep dive

---

## Remember

API testing = verifying contracts and behavior, not implementation. Focus on what matters to consumers: correct responses, proper error handling, acceptable performance.

**With Agents:** Agents automate contract validation, generate comprehensive test suites from specs, and monitor production APIs for drift. Use agents to maintain API quality at scale.
