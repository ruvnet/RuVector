---
name: contract-testing
description: "Consumer-driven contract testing for microservices using Pact, schema validation, API versioning, and backward compatibility testing. Use when testing API contracts or coordinating distributed teams."
category: testing-methodologies
priority: high
tokenEstimate: 900
agents: [qe-api-contract-validator, qe-test-generator, qe-security-scanner]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [contract, pact, consumer-driven, api, microservices, schema-validation]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/contract-testing.yaml
---

# Contract Testing

<default_to_action>
When testing API contracts or microservices:
1. DEFINE consumer expectations (what consumers actually need)
2. VERIFY provider fulfills contracts (Pact verification)
3. DETECT breaking changes before deployment (CI/CD integration)
4. VERSION APIs semantically (breaking = major bump)
5. MAINTAIN backward compatibility for supported versions

**Quick Contract Testing Steps:**
- Consumer: Define expected request/response pairs
- Provider: Verify against all consumer contracts
- CI/CD: Block deploys that break contracts
- Versioning: Document supported versions and deprecation

**Critical Success Factors:**
- Consumers own the contract (they define what they need)
- Provider must pass all consumer contracts before deploy
- Breaking changes require coordination, not surprise
</default_to_action>

## Quick Reference Card

### When to Use
- Microservices communication
- Third-party API integrations
- Distributed team coordination
- Preventing breaking changes

### Consumer-Driven Contract Flow
```
Consumer → Defines Expectations → Contract
                    ↓
Provider → Verifies Contract → Pass/Fail
                    ↓
CI/CD → Blocks Breaking Changes
```

### Breaking vs Non-Breaking Changes
| Change Type | Breaking? | Semver |
|-------------|-----------|--------|
| Remove field | ✅ Yes | Major |
| Rename field | ✅ Yes | Major |
| Change type | ✅ Yes | Major |
| Add optional field | ❌ No | Minor |
| Add new endpoint | ❌ No | Minor |
| Bug fix | ❌ No | Patch |

### Tools
| Tool | Best For |
|------|----------|
| **Pact** | Consumer-driven contracts |
| **OpenAPI/Swagger** | API-first design |
| **JSON Schema** | Schema validation |
| **GraphQL** | Schema-first contracts |

---

## Consumer Contract (Pact)

```javascript
// Consumer defines what it needs
const { Pact } = require('@pact-foundation/pact');

describe('Order API Consumer', () => {
  const provider = new Pact({
    consumer: 'CheckoutUI',
    provider: 'OrderService'
  });

  beforeAll(() => provider.setup());
  afterAll(() => provider.finalize());

  it('creates an order', async () => {
    await provider.addInteraction({
      state: 'products exist',
      uponReceiving: 'a create order request',
      withRequest: {
        method: 'POST',
        path: '/orders',
        body: { productId: 'abc', quantity: 2 }
      },
      willRespondWith: {
        status: 201,
        body: {
          orderId: like('order-123'),  // Any string matching pattern
          total: like(19.99)           // Any number
        }
      }
    });

    const response = await orderClient.create({ productId: 'abc', quantity: 2 });
    expect(response.orderId).toBeDefined();
  });
});
```

---

## Provider Verification

```javascript
// Provider verifies it fulfills all consumer contracts
const { Verifier } = require('@pact-foundation/pact');

describe('Order Service Provider', () => {
  it('fulfills all consumer contracts', async () => {
    await new Verifier({
      provider: 'OrderService',
      providerBaseUrl: 'http://localhost:3000',
      pactUrls: ['./pacts/checkoutui-orderservice.json'],
      stateHandlers: {
        'products exist': async () => {
          await db.products.create({ id: 'abc', price: 9.99 });
        }
      }
    }).verifyProvider();
  });
});
```

---

## Breaking Change Detection

```typescript
// Agent detects breaking changes
await Task("Contract Validation", {
  currentContract: 'openapi-v2.yaml',
  previousContract: 'openapi-v1.yaml',
  detectBreaking: true,
  calculateSemver: true,
  generateMigrationGuide: true
}, "qe-api-contract-validator");

// Output:
// Breaking changes found: 2
// - Removed field: order.discount
// - Type change: order.total (number → string)
// Recommended version: 3.0.0 (major bump)
```

---

## CI/CD Integration

```yaml
name: Contract Tests
on: [push]

jobs:
  consumer-tests:
    steps:
      - run: npm run test:contract
      - name: Publish Pacts
        run: npx pact-broker publish ./pacts --broker-base-url $PACT_BROKER

  provider-verification:
    needs: consumer-tests
    steps:
      - name: Verify Provider
        run: npm run verify:contracts
      - name: Can I Deploy?
        run: npx pact-broker can-i-deploy --pacticipant OrderService --version $VERSION
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/contract-testing/
├── contracts/*           - Current contracts
├── breaking-changes/*    - Detected breaking changes
├── versioning/*          - Version compatibility matrix
└── verification-results/* - Provider verification history
```

### Fleet Coordination
```typescript
const contractFleet = await FleetManager.coordinate({
  strategy: 'contract-testing',
  agents: [
    'qe-api-contract-validator',  // Validation, breaking detection
    'qe-test-generator',          // Generate contract tests
    'qe-security-scanner'         // API security
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [api-testing-patterns](../api-testing-patterns/) - API testing strategies
- [shift-left-testing](../shift-left-testing/) - Early contract validation
- [cicd-pipeline-qe-orchestrator](../cicd-pipeline-qe-orchestrator/) - Pipeline integration

---

## Remember

**Consumers own the contract.** They define what they need; providers must fulfill it. Breaking changes require major version bumps and coordination. CI/CD blocks deploys that break contracts. Use Pact for consumer-driven, OpenAPI for API-first.

**With Agents:** Agents validate contracts, detect breaking changes with semver recommendations, and generate migration guides. Use agents to maintain contract compliance at scale.
