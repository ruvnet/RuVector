---
name: "QE Contract Testing"
description: "Consumer-driven contract testing for APIs including REST, GraphQL, and event-driven systems with schema validation."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-contract-testing.yaml

---

# QE Contract Testing

## Purpose

Guide the use of v3's contract testing capabilities including consumer-driven contracts, schema validation, backward compatibility checking, and API versioning verification.

## Activation

- When testing API contracts
- When validating schema changes
- When checking backward compatibility
- When testing GraphQL APIs
- When verifying event schemas

## Quick Start

```bash
# Generate contract from API
aqe contract generate --api openapi.yaml --output contracts/

# Verify provider against contracts
aqe contract verify --provider http://localhost:3000 --contracts contracts/

# Check breaking changes
aqe contract breaking --old api-v1.yaml --new api-v2.yaml

# Test GraphQL schema
aqe contract graphql --schema schema.graphql --operations queries/
```

## Agent Workflow

```typescript
// Contract generation
Task("Generate API contracts", `
  Analyze the REST API and generate consumer contracts:
  - Parse OpenAPI specification
  - Identify critical endpoints
  - Generate Pact contracts
  - Include example requests/responses
  Output to contracts/ directory.
`, "qe-api-contract")

// Breaking change detection
Task("Check API compatibility", `
  Compare API v2.0 against v1.0:
  - Detect removed endpoints
  - Check parameter changes
  - Verify response schema changes
  - Identify deprecations
  Report breaking vs non-breaking changes.
`, "qe-api-compatibility")
```

## Contract Testing Types

### 1. Consumer-Driven Contracts (Pact)

```typescript
await contractTester.consumerDriven({
  consumer: 'web-app',
  provider: 'user-service',
  contracts: 'contracts/web-app-user-service.json',
  verification: {
    providerBaseUrl: 'http://localhost:3000',
    providerStates: providerStateHandlers,
    publishResults: true
  }
});
```

### 2. Schema Validation

```typescript
await contractTester.validateSchema({
  type: 'openapi',
  schema: 'api/openapi.yaml',
  requests: actualRequests,
  validation: {
    requestBody: true,
    responseBody: true,
    headers: true,
    statusCodes: true
  }
});
```

### 3. GraphQL Contract Testing

```typescript
await graphqlTester.testContracts({
  schema: 'schema.graphql',
  operations: 'queries/**/*.graphql',
  validation: {
    queryValidity: true,
    responseShapes: true,
    nullability: true,
    deprecations: true
  }
});
```

### 4. Event Contract Testing

```typescript
await contractTester.eventContracts({
  schema: 'events/schemas/',
  events: {
    'user.created': {
      schema: 'UserCreatedEvent.json',
      examples: ['examples/user-created.json']
    },
    'order.completed': {
      schema: 'OrderCompletedEvent.json',
      examples: ['examples/order-completed.json']
    }
  },
  compatibility: 'backward'
});
```

## Breaking Change Detection

```yaml
breaking_changes:
  always_breaking:
    - endpoint_removed
    - required_param_added
    - response_field_removed
    - type_changed

  potentially_breaking:
    - optional_param_removed
    - response_field_added
    - enum_value_removed

  non_breaking:
    - endpoint_added
    - optional_param_added
    - response_field_made_optional
    - documentation_changed
```

## Contract Report

```typescript
interface ContractReport {
  summary: {
    contracts: number;
    passed: number;
    failed: number;
    warnings: number;
  };
  consumers: {
    name: string;
    contracts: ContractResult[];
    compatibility: 'compatible' | 'breaking' | 'unknown';
  }[];
  breakingChanges: {
    type: string;
    location: string;
    description: string;
    impact: 'high' | 'medium' | 'low';
    migration: string;
  }[];
  deprecations: {
    item: string;
    deprecatedIn: string;
    removeIn: string;
    replacement: string;
  }[];
}
```

## CI/CD Integration

```yaml
contract_verification:
  consumer_side:
    - generate_contracts
    - publish_to_broker
    - can_i_deploy_check

  provider_side:
    - fetch_contracts_from_broker
    - verify_against_provider
    - publish_results

  pre_release:
    - check_breaking_changes
    - verify_all_consumers
    - update_compatibility_matrix
```

## Pact Broker Integration

```typescript
await contractTester.withBroker({
  brokerUrl: 'https://pact-broker.example.com',
  auth: { token: process.env.PACT_TOKEN },
  operations: {
    publish: true,
    canIDeploy: true,
    webhooks: true
  }
});
```

## Coordination

**Primary Agents**: qe-api-contract, qe-api-compatibility, qe-graphql-tester
**Coordinator**: qe-contract-coordinator
**Related Skills**: qe-test-generation, qe-security-compliance
