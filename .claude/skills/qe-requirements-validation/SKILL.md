---
name: "QE Requirements Validation"
description: "Requirements traceability, acceptance criteria validation, and BDD scenario management for complete requirements coverage."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-requirements-validation.yaml

---

# QE Requirements Validation

## Purpose

Guide the use of v3's requirements validation capabilities including acceptance criteria parsing, requirements traceability, BDD scenario generation, and coverage gap identification.

## Activation

- When validating requirements
- When tracing requirements to tests
- When generating BDD scenarios
- When assessing requirements coverage
- When reviewing acceptance criteria

## Quick Start

```bash
# Parse acceptance criteria
aqe requirements parse --source jira --project MYAPP

# Build traceability matrix
aqe requirements trace --requirements reqs/ --tests tests/

# Generate BDD scenarios
aqe requirements bdd --story US-123 --output features/

# Check requirements coverage
aqe requirements coverage --sprint current
```

## Agent Workflow

```typescript
// Requirements validation
Task("Validate acceptance criteria", `
  Review acceptance criteria for sprint stories:
  - Check SMART criteria (Specific, Measurable, Achievable, Relevant, Testable)
  - Identify ambiguous requirements
  - Flag missing edge cases
  - Suggest improvements
`, "qe-acceptance-criteria")

// Traceability matrix
Task("Build traceability", `
  Create requirements traceability matrix:
  - Map user stories to test cases
  - Identify untested requirements
  - Find orphan tests (no linked requirement)
  - Calculate coverage metrics
`, "qe-traceability-builder")
```

## Requirements Operations

### 1. Acceptance Criteria Validation

```typescript
await acceptanceCriteria.validate({
  source: {
    type: 'jira',
    project: 'MYAPP',
    stories: 'sprint=current'
  },
  validation: {
    specific: true,
    measurable: true,
    achievable: true,
    relevant: true,
    testable: true
  },
  output: {
    score: true,
    issues: true,
    suggestions: true
  }
});
```

### 2. Traceability Matrix

```typescript
await traceabilityBuilder.build({
  requirements: {
    source: 'jira',
    types: ['story', 'task', 'bug']
  },
  artifacts: {
    tests: 'tests/**/*.test.ts',
    code: 'src/**/*.ts',
    documentation: 'docs/**/*.md'
  },
  output: {
    matrix: true,
    coverage: true,
    gaps: true,
    orphans: true
  }
});
```

### 3. BDD Scenario Generation

```typescript
await bddGenerator.generate({
  requirements: userStory,
  format: 'gherkin',
  scenarios: {
    happyPath: true,
    edgeCases: true,
    errorCases: true,
    dataVariations: true
  },
  output: {
    featureFile: true,
    stepDefinitions: 'skeleton'
  }
});
```

### 4. Coverage Analysis

```typescript
await requirementsCoverage.analyze({
  scope: 'sprint-23',
  metrics: {
    requirementsCovered: true,
    testCasesCoverage: true,
    automationCoverage: true,
    riskAssessment: true
  },
  report: {
    summary: true,
    details: true,
    recommendations: true
  }
});
```

## Traceability Matrix

```typescript
interface TraceabilityMatrix {
  requirements: {
    id: string;
    title: string;
    type: string;
    priority: string;
    status: string;
    linkedTests: string[];
    linkedCode: string[];
    coverage: 'full' | 'partial' | 'none';
  }[];
  tests: {
    id: string;
    name: string;
    type: 'unit' | 'integration' | 'e2e';
    linkedRequirements: string[];
    automated: boolean;
  }[];
  coverage: {
    requirementsCovered: number;
    requirementsPartial: number;
    requirementsUncovered: number;
    orphanTests: number;
  };
  gaps: {
    requirement: string;
    missingCoverage: string[];
    risk: 'high' | 'medium' | 'low';
  }[];
}
```

## BDD Integration

```gherkin
# Generated feature file
Feature: User Registration
  As a new user
  I want to create an account
  So that I can access the platform

  @happy-path
  Scenario: Successful registration with valid details
    Given I am on the registration page
    When I enter valid email "user@example.com"
    And I enter valid password "SecurePass123!"
    And I click the register button
    Then I should see a success message
    And I should receive a confirmation email

  @edge-case
  Scenario: Registration with existing email
    Given a user exists with email "existing@example.com"
    When I try to register with email "existing@example.com"
    Then I should see an error "Email already registered"
```

## Requirements Quality

```yaml
quality_checks:
  acceptance_criteria:
    has_given_when_then: preferred
    is_testable: required
    is_measurable: required
    no_ambiguity: required

  user_story:
    follows_template: "As a <role>, I want <feature>, so that <benefit>"
    has_acceptance_criteria: required
    estimated: preferred

  completeness:
    edge_cases_identified: required
    error_scenarios_covered: required
    non_functional_considered: preferred
```

## Sprint Integration

```typescript
await requirementsValidator.sprintReview({
  sprint: 'current',
  checks: {
    storiesComplete: true,
    criteriaValidated: true,
    testsLinked: true,
    coverageAdequate: true
  },
  gates: {
    minCoverage: 80,
    maxUntested: 2,
    requireDemo: true
  }
});
```

## Coordination

**Primary Agents**: qe-acceptance-criteria, qe-traceability-builder, qe-bdd-specialist
**Coordinator**: qe-requirements-coordinator
**Related Skills**: qe-test-generation, qe-quality-assessment
