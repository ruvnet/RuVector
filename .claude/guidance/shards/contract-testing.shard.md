# Contract Testing Domain Shard

**Domain**: contract-testing
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Consumer-First Contracts**: Consumer-driven contract tests MUST be written by consuming teams and verified by provider teams; provider-only contracts are insufficient.

2. **Breaking Change Detection**: All API changes MUST be analyzed for breaking changes before deployment; undetected breaking changes are deployment blockers.

3. **Schema Validation Required**: All contract endpoints MUST pass schema validation (OpenAPI, JSON Schema, GraphQL SDL) before verification.

4. **Pact Broker Integration**: Contract verification results MUST be published to a contract broker (Pact or equivalent) for cross-team visibility.

5. **Version Compatibility Matrix**: Provider services MUST maintain compatibility with all active consumer versions; premature deprecation without migration is prohibited.

6. **Mock Accuracy Requirement**: Mock responses MUST match actual provider behavior; divergent mocks invalidate contract tests.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Contract Coverage | 0.8 | 0.95 | < 0.6 |
| Verification Pass Rate | 0.95 | 1.0 | < 0.85 |
| Schema Compliance | 1.0 | 1.0 | < 0.9 |
| Breaking Change Detection | 0.95 | 1.0 | < 0.8 |
| Consumer Coverage | 0.8 | 0.95 | < 0.6 |

---

## Invariants

```
INVARIANT consumer_driven_contracts:
  FOR ALL contract IN contracts:
    contract.consumer_defined = true AND
    contract.provider_verified = true
```

```
INVARIANT breaking_change_detection:
  FOR ALL api_change IN api_changes:
    EXISTS compatibility_check WHERE
      compatibility_check.change_id = api_change.id AND
      compatibility_check.completed = true AND
      IF compatibility_check.breaking_changes.length > 0 THEN
        api_change.deployment_blocked = true
```

```
INVARIANT schema_validation:
  FOR ALL endpoint IN contract_endpoints:
    EXISTS schema_check WHERE
      schema_check.endpoint_id = endpoint.id AND
      schema_check.valid = true
```

```
INVARIANT broker_publication:
  FOR ALL verification IN verifications:
    verification.published_to_broker = true AND
    verification.broker_url IS NOT NULL
```

```
INVARIANT mock_accuracy:
  FOR ALL mock IN contract_mocks:
    mock.validated_against_provider = true AND
    mock.divergence_detected = false
```

---

## Patterns

**Domain Source**: `v3/src/domains/contract-testing/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Contract Validator Service | `services/contract-validator.ts` | Contract verification |
| API Compatibility Service | `services/api-compatibility.ts` | Breaking change detection |
| Schema Validator Service | `services/schema-validator.ts` | Schema compliance |
| Contract Testing Coordinator | `coordinator.ts` | Workflow orchestration |

**Key Interfaces**: `interfaces/index.ts` defines `ApiContract`, `VerificationResult`, `BreakingChange`, and related types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-contract-validator` | Full verification, compatibility analysis |
| **Secondary** | `qe-schema-validator` | Schema validation, OpenAPI/GraphQL |
| **Support** | `qe-api-analyst` | Breaking change detection |
| **Readonly** | `qe-test-architect` | Query contracts for test planning |
| **Publisher** | All verification agents | Publish results to broker |

**Forbidden Actions**: No agent may deploy API changes with unresolved breaking changes.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Breaking change undetected (post-deploy) | CRITICAL | Rollback, escalate to Queen Coordinator |
| Schema validation failure | CRITICAL | Block deployment |
| Consumer contract verification < 0.85 | HIGH | Escalate, halt provider deployment |
| Mock-provider divergence detected | HIGH | Invalidate contract tests, regenerate mocks |
| Broker publication failure | HIGH | Retry, escalate if persistent |
| Consumer coverage < 0.6 | MEDIUM | Request additional consumer contracts |
| Deprecation without migration guide | MEDIUM | Block deprecation, request migration plan |

---

## Memory Namespace

- **Namespace**: `qe-patterns/contract-testing`
- **Retention**: 90 days
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-generation` | Output | Generate contract test stubs |
| `test-execution` | Output | Execute contract verifications |
| `security-compliance` | Input | Validate API security contracts |
| `quality-assessment` | Output | Report contract coverage |
| `learning-optimization` | Bidirectional | Share contract patterns |

---

## Breaking Change Categories

| Category | Severity | Example |
|----------|----------|---------|
| Removed Endpoint | CRITICAL | DELETE /api/v1/users/:id removed |
| Changed Response Schema | HIGH | Field type changed string -> number |
| New Required Field | HIGH | Request body requires new field |
| Removed Response Field | MEDIUM | Response no longer includes field |
| New Optional Field | LOW | Response includes new optional field |
| Documentation Only | INFO | Description or example updated |

---

## Contract Verification Flow

```
1. Consumer defines contract (Pact, OpenAPI, GraphQL)
2. Contract published to broker
3. Provider CI triggers verification
4. Schema validation runs
5. Breaking change detection runs
6. Mock accuracy validation
7. Results published to broker
8. If pass: deployment allowed
   If fail: deployment blocked, consumers notified
```

---

## Supported Contract Formats

| Format | Use Case | Validation Service |
|--------|----------|-------------------|
| Pact | Consumer-driven contracts | ContractValidatorService |
| OpenAPI 3.x | REST API specifications | SchemaValidatorService |
| JSON Schema | Data structure validation | SchemaValidatorService |
| GraphQL SDL | GraphQL API contracts | SchemaValidatorService |
| AsyncAPI | Event-driven contracts | ContractValidatorService |

---

## Migration Guide Requirements

```typescript
interface MigrationGuide {
  deprecatedEndpoint: string;
  deprecationDate: Date;
  removalDate: Date;
  replacementEndpoint: string;
  migrationSteps: MigrationStep[];
  codeExamples: {
    before: string;
    after: string;
    language: string;
  }[];
  breakingChanges: BreakingChange[];
  affectedConsumers: string[];
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
