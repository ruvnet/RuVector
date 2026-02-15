---
name: database-testing
description: "Database schema validation, data integrity testing, migration testing, transaction isolation, and query performance. Use when testing data persistence, ensuring referential integrity, or validating database migrations."
category: specialized-testing
priority: high
tokenEstimate: 900
agents: [qe-test-data-architect, qe-test-executor, qe-performance-tester]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [database, schema, migration, transactions, integrity, sql, performance]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/database-testing.yaml
---

# Database Testing

<default_to_action>
When testing database operations:
1. VALIDATE schema (tables, columns, constraints exist as expected)
2. TEST data integrity (unique, foreign key, check constraints)
3. VERIFY migrations (forward works, rollback works, data preserved)
4. CHECK transaction isolation (ACID properties, concurrent access)
5. MEASURE query performance (indexes used, execution time)

**Quick DB Testing Checklist:**
- Schema matches specification
- Unique constraints prevent duplicates
- Foreign keys prevent orphaned records
- Migrations are reversible
- Transactions roll back on error

**Critical Success Factors:**
- Database bugs cause data loss/corruption (catastrophic)
- Test migrations in staging before production
- Transaction tests catch concurrency bugs
</default_to_action>

## Quick Reference Card

### When to Use
- New table/schema creation
- Migration development
- Data integrity validation
- Query performance optimization

### Database Test Types
| Type | Focus | When |
|------|-------|------|
| **Schema** | Structure correct | Table creation |
| **Integrity** | Constraints work | Data operations |
| **Migration** | Up/down work | Schema changes |
| **Transaction** | ACID properties | Concurrent access |
| **Performance** | Query speed | Optimization |

### ACID Properties
| Property | Test Focus |
|----------|------------|
| **Atomicity** | All or nothing (rollback on error) |
| **Consistency** | Constraints always valid |
| **Isolation** | Concurrent transactions don't interfere |
| **Durability** | Committed data persists |

---

## Schema Testing

```javascript
test('users table has correct schema', async () => {
  const schema = await db.raw(`
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'users'
  `);

  expect(schema).toContainEqual({
    column_name: 'id',
    data_type: 'integer',
    is_nullable: 'NO'
  });

  expect(schema).toContainEqual({
    column_name: 'email',
    data_type: 'character varying',
    is_nullable: 'NO'
  });
});
```

---

## Data Integrity Testing

```javascript
test('email must be unique', async () => {
  await db.users.create({ email: 'test@example.com' });

  await expect(
    db.users.create({ email: 'test@example.com' })
  ).rejects.toThrow('unique constraint violation');
});

test('foreign key prevents orphaned records', async () => {
  const user = await db.users.create({ email: 'test@example.com' });
  await db.orders.create({ userId: user.id, total: 100 });

  await expect(
    db.users.delete({ id: user.id })
  ).rejects.toThrow('foreign key constraint');
});
```

---

## Migration Testing

```javascript
test('migration is reversible', async () => {
  await migrate('add-users-table');

  // Table exists after migration
  const tables = await db.raw(`SELECT table_name FROM information_schema.tables`);
  expect(tables.map(t => t.table_name)).toContain('users');

  await rollback('add-users-table');

  // Table gone after rollback
  const tablesAfter = await db.raw(`SELECT table_name FROM information_schema.tables`);
  expect(tablesAfter.map(t => t.table_name)).not.toContain('users');
});

test('migration preserves existing data', async () => {
  await db.users.create({ email: 'test@example.com' });

  await migrate('add-age-column');

  const user = await db.users.findOne({ email: 'test@example.com' });
  expect(user).toBeDefined();
  expect(user.age).toBeNull(); // New column, null default
});
```

---

## Transaction Testing

```javascript
test('transaction rolls back on error', async () => {
  const initialCount = await db.users.count();

  try {
    await db.transaction(async (trx) => {
      await trx('users').insert({ email: 'user1@example.com' });
      await trx('users').insert({ email: 'user2@example.com' });
      throw new Error('Rollback test');
    });
  } catch (error) { /* Expected */ }

  expect(await db.users.count()).toBe(initialCount);
});

test('concurrent transactions isolated', async () => {
  const user = await db.users.create({ balance: 100 });

  // Two concurrent withdrawals (race condition test)
  await Promise.all([
    db.transaction(async (trx) => {
      const current = await trx('users').where({ id: user.id }).first();
      await trx('users').update({ balance: current.balance - 50 });
    }),
    db.transaction(async (trx) => {
      const current = await trx('users').where({ id: user.id }).first();
      await trx('users').update({ balance: current.balance - 50 });
    })
  ]);

  const final = await db.users.findOne({ id: user.id });
  expect(final.balance).toBe(0); // Proper isolation
});
```

---

## Agent-Driven Database Testing

```typescript
// Generate test data with integrity
await Task("Generate Test Data", {
  schema: 'ecommerce',
  tables: ['users', 'products', 'orders'],
  count: { users: 1000, products: 500, orders: 5000 },
  preserveReferentialIntegrity: true
}, "qe-test-data-architect");

// Test migration safety
await Task("Migration Test", {
  migration: 'add-payment-status-column',
  tests: ['forward', 'rollback', 'data-preservation'],
  environment: 'staging'
}, "qe-test-executor");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/database-testing/
├── schema-snapshots/*   - Current schema state
├── migrations/*         - Migration test results
├── integrity/*          - Constraint validation
└── performance/*        - Query benchmarks
```

### Fleet Coordination
```typescript
const dbFleet = await FleetManager.coordinate({
  strategy: 'database-testing',
  agents: [
    'qe-test-data-architect',  // Generate test data
    'qe-test-executor',        // Run DB tests
    'qe-performance-tester'    // Query performance
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [test-data-management](../test-data-management/) - Generate test data
- [performance-testing](../performance-testing/) - Query performance
- [compliance-testing](../compliance-testing/) - Data protection

---

## Remember

**Database bugs are catastrophic.** Data loss is unrecoverable. Corruption spreads silently. Performance issues compound.

**Test migrations before production:** Forward works, rollback works, data preserved, performance acceptable. Never deploy untested migrations.

**With Agents:** `qe-test-data-architect` generates realistic test data with referential integrity. `qe-test-executor` validates migrations automatically in CI/CD.
