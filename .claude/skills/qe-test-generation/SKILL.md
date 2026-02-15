---
name: "QE-Test-Generation"
description: "AI-powered test generation using pattern recognition, code analysis, and intelligent test synthesis for comprehensive test coverage."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-test-generation.yaml

---

# QE Test Generation

## Purpose

Guide the use of v3's AI-powered test generation capabilities including pattern-based test synthesis, multi-framework support, and intelligent test case derivation from code analysis.

## Activation

- When generating tests for new code
- When improving test coverage
- When migrating tests between frameworks
- When applying TDD patterns
- When generating edge case tests

## Quick Start

```bash
# Generate unit tests for a file
aqe test generate --file src/services/UserService.ts --framework jest

# Generate tests with coverage target
aqe test generate --scope src/api/ --coverage 90 --type unit

# Generate integration tests
aqe test generate --file src/controllers/AuthController.ts --type integration

# Generate from patterns
aqe test generate --pattern repository --target src/repositories/
```

## Agent Workflow

```typescript
// Spawn test generation agents
Task("Generate unit tests", `
  Analyze src/services/PaymentService.ts and generate comprehensive Jest tests.
  Include:
  - Happy path tests for all public methods
  - Edge cases and boundary conditions
  - Error handling scenarios
  - Mock external dependencies
  Output to tests/unit/services/PaymentService.test.ts
`, "qe-test-generator")

// Pattern-based generation
Task("Apply test patterns", `
  Scan src/repositories/ and apply repository test pattern:
  - CRUD operation tests
  - Query builder tests
  - Transaction tests
  - Connection error handling
`, "qe-pattern-matcher")
```

## Test Generation Strategies

### 1. Code Analysis Based

```typescript
await testGenerator.analyzeAndGenerate({
  source: 'src/services/OrderService.ts',
  analysis: {
    methods: true,
    branches: true,
    dependencies: true,
    errorPaths: true
  },
  output: {
    framework: 'jest',
    style: 'describe-it',
    assertions: 'expect'
  }
});
```

### 2. Pattern-Based Generation

```typescript
await testGenerator.applyPattern({
  pattern: 'service-layer',
  targets: ['src/services/*.ts'],
  customizations: {
    mockStrategy: 'jest.mock',
    asyncHandling: 'async-await',
    errorAssertion: 'toThrow'
  }
});
```

### 3. Coverage-Driven Generation

```typescript
await testGenerator.fillCoverageGaps({
  coverageReport: 'coverage/lcov.info',
  targetCoverage: 90,
  prioritize: ['uncovered-branches', 'error-paths'],
  maxTests: 50
});
```

## Framework Support

| Framework | Unit | Integration | E2E | Mocking |
|-----------|------|-------------|-----|---------|
| Jest | ✅ | ✅ | ⚠️ | jest.mock |
| Vitest | ✅ | ✅ | ⚠️ | vi.mock |
| Mocha | ✅ | ✅ | ❌ | sinon |
| Pytest | ✅ | ✅ | ❌ | pytest-mock |
| JUnit | ✅ | ✅ | ❌ | Mockito |

## Test Quality Checks

```yaml
quality_checks:
  assertions:
    minimum_per_test: 1
    meaningful: true

  isolation:
    no_shared_state: true
    proper_setup_teardown: true

  naming:
    descriptive: true
    follows_convention: true

  coverage:
    branches: 80
    statements: 85
```

## Coordination

**Primary Agents**: qe-test-generator, qe-pattern-matcher, qe-test-architect
**Coordinator**: qe-test-generation-coordinator
**Related Skills**: qe-coverage-analysis, qe-test-execution
