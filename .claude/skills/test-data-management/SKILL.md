---
name: test-data-management
description: "Strategic test data generation, management, and privacy compliance. Use when creating test data, handling PII, ensuring GDPR/CCPA compliance, or scaling data generation for realistic testing scenarios."
category: specialized-testing
priority: high
tokenEstimate: 1000
agents: [qe-test-data-architect, qe-test-executor, qe-security-scanner]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [test-data, faker, synthetic, gdpr, pii, anonymization, factories]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/test-data-management.yaml

---

# Test Data Management

<default_to_action>
When creating or managing test data:
1. NEVER use production PII directly
2. GENERATE synthetic data with faker libraries
3. ANONYMIZE production data if used (mask, hash)
4. ISOLATE test data (transactions, per-test cleanup)
5. SCALE with batch generation (10k+ records/sec)

**Quick Data Strategy:**
- Unit tests: Minimal data (just enough)
- Integration: Realistic data (full complexity)
- Performance: Volume data (10k+ records)

**Critical Success Factors:**
- 40% of test failures from inadequate data
- GDPR fines up to €20M for PII violations
- Never store production PII in test environments
</default_to_action>

## Quick Reference Card

### When to Use
- Creating test datasets
- Handling sensitive data
- Performance testing with volume
- GDPR/CCPA compliance

### Data Strategies
| Type | When | Size |
|------|------|------|
| **Minimal** | Unit tests | 1-10 records |
| **Realistic** | Integration | 100-1000 records |
| **Volume** | Performance | 10k+ records |
| **Edge cases** | Boundary testing | Targeted |

### Privacy Techniques
| Technique | Use Case |
|-----------|----------|
| **Synthetic** | Generate fake data (preferred) |
| **Masking** | j***@example.com |
| **Hashing** | Irreversible pseudonymization |
| **Tokenization** | Reversible with key |

---

## Synthetic Data Generation

```javascript
import { faker } from '@faker-js/faker';

// Seed for reproducibility
faker.seed(123);

function generateUser() {
  return {
    id: faker.string.uuid(),
    email: faker.internet.email(),
    firstName: faker.person.firstName(),
    lastName: faker.person.lastName(),
    phone: faker.phone.number(),
    address: {
      street: faker.location.streetAddress(),
      city: faker.location.city(),
      zip: faker.location.zipCode()
    },
    createdAt: faker.date.past()
  };
}

// Generate 1000 users
const users = Array.from({ length: 1000 }, generateUser);
```

---

## Test Data Builder Pattern

```typescript
class UserBuilder {
  private user: Partial<User> = {};

  asAdmin() {
    this.user.role = 'admin';
    this.user.permissions = ['read', 'write', 'delete'];
    return this;
  }

  asCustomer() {
    this.user.role = 'customer';
    this.user.permissions = ['read'];
    return this;
  }

  withEmail(email: string) {
    this.user.email = email;
    return this;
  }

  build(): User {
    return {
      id: this.user.id ?? faker.string.uuid(),
      email: this.user.email ?? faker.internet.email(),
      role: this.user.role ?? 'customer',
      ...this.user
    } as User;
  }
}

// Usage
const admin = new UserBuilder().asAdmin().withEmail('admin@test.com').build();
const customer = new UserBuilder().asCustomer().build();
```

---

## Data Anonymization

```javascript
// Masking
function maskEmail(email) {
  const [user, domain] = email.split('@');
  return `${user[0]}***@${domain}`;
}
// john@example.com → j***@example.com

function maskCreditCard(cc) {
  return `****-****-****-${cc.slice(-4)}`;
}
// 4242424242424242 → ****-****-****-4242

// Anonymize production data
const anonymizedUsers = prodUsers.map(user => ({
  id: user.id, // Keep ID for relationships
  email: `user-${user.id}@example.com`, // Fake email
  firstName: faker.person.firstName(), // Generated
  phone: null, // Remove PII
  createdAt: user.createdAt // Keep non-PII
}));
```

---

## Database Transaction Isolation

```javascript
// Best practice: use transactions for cleanup
beforeEach(async () => {
  await db.beginTransaction();
});

afterEach(async () => {
  await db.rollbackTransaction(); // Auto cleanup!
});

test('user registration', async () => {
  const user = await userService.register({
    email: 'test@example.com'
  });
  expect(user.id).toBeDefined();
  // Automatic rollback after test - no cleanup needed
});
```

---

## Volume Data Generation

```javascript
// Generate 10,000 users efficiently
async function generateLargeDataset(count = 10000) {
  const batchSize = 1000;
  const batches = Math.ceil(count / batchSize);

  for (let i = 0; i < batches; i++) {
    const users = Array.from({ length: batchSize }, (_, index) => ({
      id: i * batchSize + index,
      email: `user${i * batchSize + index}@example.com`,
      firstName: faker.person.firstName()
    }));

    await db.users.insertMany(users); // Batch insert
    console.log(`Batch ${i + 1}/${batches}`);
  }
}
```

---

## Agent-Driven Data Generation

```typescript
// High-speed generation with constraints
await Task("Generate Test Data", {
  schema: 'ecommerce',
  count: { users: 10000, products: 500, orders: 5000 },
  preserveReferentialIntegrity: true,
  constraints: {
    age: { min: 18, max: 90 },
    roles: ['customer', 'admin']
  }
}, "qe-test-data-architect");

// GDPR-compliant anonymization
await Task("Anonymize Production Data", {
  source: 'production-snapshot',
  piiFields: ['email', 'phone', 'ssn'],
  method: 'pseudonymization',
  retainStructure: true
}, "qe-test-data-architect");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/test-data-management/
├── schemas/*            - Data schemas
├── generators/*         - Generator configs
├── anonymization/*      - PII handling rules
└── fixtures/*           - Reusable fixtures
```

### Fleet Coordination
```typescript
const dataFleet = await FleetManager.coordinate({
  strategy: 'test-data-generation',
  agents: [
    'qe-test-data-architect',  // Generate data
    'qe-test-executor',        // Execute with data
    'qe-security-scanner'      // Validate no PII exposure
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [database-testing](../database-testing/) - Schema and integrity testing
- [compliance-testing](../compliance-testing/) - GDPR/CCPA compliance
- [performance-testing](../performance-testing/) - Volume data for perf tests

---

## Remember

**Test data is infrastructure, not an afterthought.** 40% of test failures are caused by inadequate test data. Poor data = poor tests.

**Never use production PII directly.** GDPR fines up to €20M or 4% of revenue. Always use synthetic data or properly anonymized production snapshots.

**With Agents:** `qe-test-data-architect` generates 10k+ records/sec with realistic patterns, relationships, and constraints. Agents ensure GDPR/CCPA compliance automatically and eliminate test data bottlenecks.
