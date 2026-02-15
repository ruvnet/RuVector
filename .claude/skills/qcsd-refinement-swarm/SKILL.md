---
name: qcsd-refinement-swarm
description: "QCSD Refinement phase swarm for Sprint Refinement sessions using SFDIPOT product factors, BDD scenario generation, and requirements validation."
category: qcsd-phases
priority: critical
version: 1.0.0
tokenEstimate: 3200
# DDD Domain Mapping (from QCSD-AGENTIC-QE-MAPPING-FRAMEWORK.md)
domains:
  primary:
    - domain: requirements-validation
      agents: [qe-product-factors-assessor, qe-bdd-generator, qe-requirements-validator]
  conditional:
    - domain: contract-testing
      agents: [qe-contract-validator]
    - domain: code-intelligence
      agents: [qe-impact-analyzer, qe-dependency-mapper]
    - domain: enterprise-integration
      agents: [qe-middleware-validator, qe-odata-contract-tester, qe-sod-analyzer]
  transformation:
    - domain: test-generation
      agents: [qe-test-idea-rewriter]
# Agent Inventory
agents:
  core: [qe-product-factors-assessor, qe-bdd-generator, qe-requirements-validator]
  conditional: [qe-contract-validator, qe-impact-analyzer, qe-dependency-mapper, qe-middleware-validator, qe-odata-contract-tester, qe-sod-analyzer]
  transformation: [qe-test-idea-rewriter]
  total: 10
  sub_agents: 0
skills: [context-driven-testing, testability-scoring, risk-based-testing]
# Execution Models (Task Tool is PRIMARY)
execution:
  primary: task-tool
  alternatives: [mcp-tools, cli]
swarm_pattern: true
parallel_batches: 3
last_updated: 2026-02-02
enforcement_level: strict
tags: [qcsd, refinement, sfdipot, bdd, gherkin, requirements, swarm, parallel, ddd]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qcsd-refinement-swarm.yaml

---

# QCSD Refinement Swarm v1.0

Shift-left quality engineering swarm for Sprint Refinement sessions.

---

## Overview

The Refinement Swarm takes user stories that passed Ideation and prepares them
for Sprint commitment. Where the Ideation Swarm asks "Should we build this?"
using HTSM quality criteria, the Refinement Swarm asks "How should we test
this?" using SFDIPOT product factors, BDD scenarios, and INVEST validation.

### Key Differentiators from Ideation Swarm

| Dimension | Ideation Swarm | Refinement Swarm |
|-----------|---------------|------------------|
| Framework | HTSM v6.3 (10 categories) | SFDIPOT (7 factors, 37 subcategories) |
| Agents | 9 (3 core + 6 conditional) | 10 (3 core + 6 conditional + 1 transformation) |
| Core Output | Quality Criteria HTML report | BDD Gherkin scenarios |
| Decision | GO / CONDITIONAL / NO-GO | READY / CONDITIONAL / NOT-READY |
| Flags | HAS_UI, HAS_SECURITY, HAS_UX, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION | HAS_API, HAS_REFACTORING, HAS_DEPENDENCIES, HAS_SECURITY, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION |
| Phase | PI Planning / Sprint Planning | Sprint Refinement |
| Final Step | Report generation | Test idea rewriter transformation |

---

## ENFORCEMENT RULES - READ FIRST

**These rules are NON-NEGOTIABLE. Violation means skill execution failure.**

| Rule | Enforcement |
|------|-------------|
| **E1** | You MUST spawn ALL THREE core agents (qe-product-factors-assessor, qe-bdd-generator, qe-requirements-validator) in Phase 2. No exceptions. |
| **E2** | You MUST put all parallel Task calls in a SINGLE message. |
| **E3** | You MUST STOP and WAIT after each batch. No proceeding early. |
| **E4** | You MUST spawn conditional agents if flags are TRUE. No skipping. |
| **E5** | You MUST apply READY/CONDITIONAL/NOT-READY logic exactly as specified in Phase 5. |
| **E6** | You MUST generate the full report structure. No abbreviated versions. |
| **E7** | Each agent MUST read its reference files before analysis. |
| **E8** | You MUST apply qe-test-idea-rewriter transformation on ALL test ideas in Phase 8. Always. |
| **E9** | You MUST execute Phase 7 learning persistence. Store refinement findings to memory BEFORE Phase 8. No skipping. |

**PROHIBITED BEHAVIORS:**
- Summarizing instead of spawning agents
- Skipping agents "for brevity"
- Proceeding before background tasks complete
- Providing your own analysis instead of spawning specialists
- Omitting report sections
- Using placeholder text like "[details here]"
- Skipping the test idea rewriter transformation
- Skipping learning persistence (Phase 7) or treating it as optional
- Generating BDD scenarios yourself instead of using qe-bdd-generator

---

## PHASE 1: Analyze Story Content (Flag Detection)

**MANDATORY: You must complete this analysis before Phase 2.**

Scan the story content and SET these flags. Do not skip any flag.

### Flag Detection (Check ALL SEVEN)

```
HAS_API = FALSE
  Set TRUE if story contains ANY of: API, endpoint, REST, GraphQL, contract,
  integration, webhook, microservice, service-to-service, HTTP, gRPC,
  request, response, payload, schema, OpenAPI, Swagger, consumer, provider

HAS_REFACTORING = FALSE
  Set TRUE if story contains ANY of: refactor, rearchitect, rewrite,
  migrate, modernize, tech debt, legacy, restructure, decouple,
  consolidate, simplify, extract, decompose, re-engineer

HAS_DEPENDENCIES = FALSE
  Set TRUE if story contains ANY of: dependency, coupling, module,
  package, library, import, shared, cross-team, upstream, downstream,
  third-party, external, vendor, SDK, plugin, middleware

HAS_SECURITY = FALSE
  Set TRUE if story contains ANY of: auth, security, credential, token,
  encrypt, PII, compliance, password, login, session, OAuth, JWT,
  permission, role, access control, RBAC, sensitive, private

HAS_MIDDLEWARE = FALSE
  Set TRUE if story contains ANY of: middleware, ESB, message broker, MQ,
  Kafka, RabbitMQ, integration bus, API gateway, message queue, pub/sub

HAS_SAP_INTEGRATION = FALSE
  Set TRUE if story contains ANY of: SAP, RFC, BAPI, IDoc, OData,
  S/4HANA, EWM, ECC, ABAP, CDS view, Fiori

HAS_AUTHORIZATION = FALSE
  Set TRUE if story contains ANY of: SoD, segregation of duties,
  role conflict, authorization object, T-code, user role,
  access control matrix, GRC
```

### Validation Checkpoint

Before proceeding to Phase 2, confirm:

```
+-- I have read the entire story content
+-- I have evaluated ALL SEVEN flags
+-- I have recorded which flags are TRUE
+-- I understand which conditional agents will be needed
```

**DO NOT proceed to Phase 2 until all checkboxes are confirmed.**

### MANDATORY: Output Flag Detection Results

You MUST output flag detection results before proceeding:

```
+-------------------------------------------------------------+
|                    FLAG DETECTION RESULTS                    |
+-------------------------------------------------------------+
|                                                             |
|  HAS_API:          [TRUE/FALSE]                             |
|  Evidence:         [what triggered it - specific patterns]  |
|                                                             |
|  HAS_REFACTORING:  [TRUE/FALSE]                             |
|  Evidence:         [what triggered it - specific patterns]  |
|                                                             |
|  HAS_DEPENDENCIES: [TRUE/FALSE]                             |
|  Evidence:         [what triggered it - specific patterns]  |
|                                                             |
|  HAS_SECURITY:     [TRUE/FALSE]                             |
|  Evidence:         [what triggered it - specific patterns]  |
|                                                             |
|  HAS_MIDDLEWARE:       [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific]       |
|                                                             |
|  HAS_SAP_INTEGRATION: [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific]       |
|                                                             |
|  HAS_AUTHORIZATION:   [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific]       |
|                                                             |
|  EXPECTED AGENTS:                                           |
|  - Core: 3 (always)                                         |
|  - Conditional: [count based on TRUE flags]                 |
|  - Transformation: 1 (always)                               |
|  - TOTAL: [3 + conditional count + 1]                       |
|                                                             |
+-------------------------------------------------------------+
```

**DO NOT proceed to Phase 2 without outputting flag detection results.**

---

## PHASE 2: Spawn Core Agents (PARALLEL BATCH 1)

### CRITICAL ENFORCEMENT

```
+-------------------------------------------------------------+
|  YOU MUST INCLUDE ALL THREE TASK CALLS IN YOUR NEXT MESSAGE  |
|                                                              |
|  - Task 1: qe-product-factors-assessor                      |
|  - Task 2: qe-bdd-generator                                 |
|  - Task 3: qe-requirements-validator                         |
|                                                              |
|  If your message contains fewer than 3 Task calls, you have |
|  FAILED this phase. Start over.                              |
+-------------------------------------------------------------+
```

### Domain Context

| Agent | Domain | MCP Tool Mapping |
|-------|--------|------------------|
| qe-product-factors-assessor | requirements-validation | `requirements_validate` |
| qe-bdd-generator | requirements-validation | `test_generate_enhanced` |
| qe-requirements-validator | requirements-validation | `requirements_validate` |

### Agent 1: Product Factors Assessor (SFDIPOT)

**This agent MUST analyze all 7 SFDIPOT factors with 37 subcategories. Fewer is a failure.**

```
Task({
  description: "SFDIPOT Product Factors analysis",
  prompt: `You are qe-product-factors-assessor. Your output quality is being audited.

## MANDATORY FIRST STEPS (DO NOT SKIP)

1. READ this template file FIRST - your output MUST follow this structure:
   .claude/agents/v3/helpers/product-factors/sfdipot-reference-template.html

2. READ the story content below IN FULL before starting analysis.

## STORY TO ANALYZE

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE - DO NOT SUMMARIZE]
=== STORY CONTENT END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

You MUST analyze ALL 7 SFDIPOT product factors. For each factor, analyze
its subcategories and assign a priority.

### Factor 1: STRUCTURE (Internal composition)
Subcategories to assess:
- Code architecture and module organization
- Database schema and data models
- File and directory structure
- Configuration hierarchy
- Component relationships
Priority: P0/P1/P2/P3 with justification

### Factor 2: FUNCTION (What it does)
Subcategories to assess:
- Core business logic
- Input processing and validation
- Output generation and formatting
- Error handling and recovery
- State management and transitions
Priority: P0/P1/P2/P3 with justification

### Factor 3: DATA (What it processes)
Subcategories to assess:
- Data types and formats
- Data flow and transformations
- Data persistence and storage
- Data validation rules
- Data volume and growth patterns
- Data integrity constraints
Priority: P0/P1/P2/P3 with justification

### Factor 4: INTERFACES (How it connects)
Subcategories to assess:
- User interfaces (if applicable)
- API contracts and endpoints
- Event/message interfaces
- File-based interfaces
- Hardware interfaces (if applicable)
- Inter-system communication protocols
Priority: P0/P1/P2/P3 with justification

### Factor 5: PLATFORM (What it depends on)
Subcategories to assess:
- Operating system dependencies
- Runtime environment (Node, JVM, etc.)
- Cloud provider services
- Container/orchestration platform
- Browser/client platform
- Network infrastructure
Priority: P0/P1/P2/P3 with justification

### Factor 6: OPERATIONS (How it runs in production)
Subcategories to assess:
- Deployment process
- Monitoring and alerting
- Logging and observability
- Backup and recovery
- Scaling and capacity
- Incident response procedures
Priority: P0/P1/P2/P3 with justification

### Factor 7: TIME (How it changes)
Subcategories to assess:
- Scheduling and time-based triggers
- Temporal data (timestamps, durations)
- Concurrency and race conditions
- Timeout and retry behavior
- Historical data and versioning
- Time zone handling
Priority: P0/P1/P2/P3 with justification

## SCORING REQUIREMENTS

For each of the 7 factors, provide:

| Field | Requirement |
|-------|-------------|
| Factor Name | One of the 7 SFDIPOT factors |
| Priority | P0, P1, P2, or P3 with justification |
| Subcategories Assessed | Count of subcategories analyzed |
| Coverage Depth | Shallow / Adequate / Deep |
| Key Findings | Top 3 findings for this factor |
| Test Focus Areas | Specific areas requiring test coverage |

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/02-sfdipot-analysis.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I read the SFDIPOT reference template?
+-- Did I analyze all 7 factors?
+-- Did I assess at least 37 subcategories total?
+-- Does every factor have a priority with justification?
+-- Did I identify test focus areas for each factor?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-product-factors-assessor",
  run_in_background: true
})
```

### Agent 2: BDD Scenario Generator

**This agent MUST generate at least 5 Gherkin scenarios. Fewer is a failure.**

```
Task({
  description: "BDD Gherkin scenario generation",
  prompt: `You are qe-bdd-generator. Your output quality is being audited.

## METHODOLOGY

Apply Behaviour-Driven Development (BDD) principles to generate
comprehensive Gherkin scenarios from the story and its acceptance criteria.

## STORY TO ANALYZE

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE - DO NOT SUMMARIZE]
=== STORY CONTENT END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Feature File Header

Write a proper Gherkin feature header:

Feature: [Feature name derived from story]
  As a [role from story]
  I want [capability from story]
  So that [business value from story]

### 2. Scenario Categories (ALL MANDATORY)

You MUST generate scenarios in ALL of these categories:

#### Category A: Happy Path Scenarios (minimum 2)
- Primary success flow
- Variant success flows

#### Category B: Error/Negative Path Scenarios (minimum 2)
- Invalid input handling
- System error handling
- Boundary violations

#### Category C: Boundary/Edge Case Scenarios (minimum 1)
- Minimum and maximum values
- Empty states
- Null/undefined handling

#### Category D: Security Scenarios (if HAS_SECURITY flag is TRUE)
- Authentication failures
- Authorization violations
- Data protection scenarios

### 3. Gherkin Format Requirements

EVERY scenario MUST follow this structure:

  Scenario: [Descriptive name]
    Given [precondition with specific data]
    And [additional precondition if needed]
    When [action with specific input]
    And [additional action if needed]
    Then [expected outcome with specific assertion]
    And [additional assertion if needed]

RULES:
- Use concrete data, not abstract descriptions
- Include data tables where 3+ similar checks apply
- Use Scenario Outline + Examples for parameterized cases
- Every Then step must be verifiable/assertable
- No vague steps like "the system works correctly"

### 4. Scenario Outline with Examples (minimum 1)

At least one scenario MUST use Scenario Outline:

  Scenario Outline: [Parameterized scenario name]
    Given [precondition with <parameter>]
    When [action with <input>]
    Then [expected <outcome>]

    Examples:
      | parameter | input | outcome |
      | value1    | val1  | result1 |
      | value2    | val2  | result2 |
      | value3    | val3  | result3 |

### 5. Traceability Matrix

Map each scenario back to acceptance criteria:

| Scenario | Acceptance Criterion | Category |
|----------|---------------------|----------|
| Scenario 1 | AC-1 | Happy Path |
| Scenario 2 | AC-1, AC-3 | Error Path |
| ... | ... | ... |

### 6. Summary Metrics

| Metric | Value |
|--------|-------|
| Total Scenarios | __ |
| Happy Path | __ |
| Error Path | __ |
| Boundary | __ |
| Security | __ |
| Scenario Outlines | __ |
| Total Examples | __ |
| AC Coverage | __% |

**MINIMUM: 5 total scenarios. Target: 10+ scenarios.**

## OUTPUT FORMAT

Save your complete analysis in Markdown (with embedded Gherkin) to:
${OUTPUT_FOLDER}/03-bdd-scenarios.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I generate at least 5 scenarios?
+-- Did I cover Happy Path, Error Path, and Boundary categories?
+-- Does every scenario have concrete data (not abstract)?
+-- Did I include at least one Scenario Outline with Examples?
+-- Did I map scenarios back to acceptance criteria?
+-- Did I calculate AC coverage percentage?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-bdd-generator",
  run_in_background: true
})
```

### Agent 3: Requirements Validator (INVEST)

**This agent MUST provide an INVEST completeness score 0-100. No ranges.**

```
Task({
  description: "INVEST requirements validation and testability scoring",
  prompt: `You are qe-requirements-validator. Your output quality is being audited.

## METHODOLOGY

Apply INVEST validation and testability-scoring principles.

## STORY AND ACCEPTANCE CRITERIA TO VALIDATE

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY + ACCEPTANCE CRITERIA HERE - DO NOT SUMMARIZE]
=== STORY CONTENT END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. INVEST Validation (MANDATORY - ALL SIX CRITERIA)

Score each INVEST criterion 0-100:

| Criterion | Score | Evidence | Issues |
|-----------|-------|----------|--------|
| **I**ndependent | X/100 | Can this story be developed and tested independently of other stories? | [issues found] |
| **N**egotiable | X/100 | Is the story expressed as intent (not implementation detail)? | [issues found] |
| **V**aluable | X/100 | Does it deliver clear value to users or the business? | [issues found] |
| **E**stimable | X/100 | Can the team estimate effort with confidence? | [issues found] |
| **S**mall | X/100 | Can it be completed in a single sprint? | [issues found] |
| **T**estable | X/100 | Are there clear, verifiable acceptance criteria? | [issues found] |

**INVEST COMPLETENESS SCORE: XX/100** (average of all six criteria)

### 2. Acceptance Criteria Assessment

For EACH acceptance criterion:

| AC ID | Text | Testable? | Issues | Suggested Rewrite |
|-------|------|-----------|--------|-------------------|
| AC-1 | [original text] | Yes/No | [issues] | [improved version if needed] |
| AC-2 | ... | ... | ... | ... |

For non-testable ACs, explain WHY and provide a testable rewrite.

### 3. Gap Analysis (MANDATORY)

Identify ALL gaps in the story:

#### Missing Requirements
- [Gap 1: Description of what is missing]
- [Gap 2: ...]

#### Ambiguous Language
- [Ambiguity 1: Quote the ambiguous text and explain]
- [Ambiguity 2: ...]

#### Missing Edge Cases
- [Edge Case 1: Scenario not covered by current ACs]
- [Edge Case 2: ...]

#### Missing Non-Functional Requirements
- Performance: [specified? missing?]
- Security: [specified? missing?]
- Accessibility: [specified? missing?]
- Error handling: [specified? missing?]

**CRITICAL GAPS COUNT: __** (gaps that block sprint readiness)

**MINIMUM: Identify at least 3 gaps or explicitly state "No gaps found
after thorough analysis of all requirement dimensions"**

### 4. Definition of Ready (DoR) Checklist

| DoR Item | Status | Notes |
|----------|--------|-------|
| Story has clear title and description | Pass/Fail | ... |
| Acceptance criteria are defined | Pass/Fail | ... |
| ACs are testable | Pass/Fail | ... |
| Dependencies are identified | Pass/Fail | ... |
| Story is estimated | Pass/Fail | ... |
| Story fits in a sprint | Pass/Fail | ... |
| UX designs available (if needed) | Pass/Fail/N/A | ... |
| API contracts defined (if needed) | Pass/Fail/N/A | ... |
| Test data requirements known | Pass/Fail | ... |
| No blockers identified | Pass/Fail | ... |

**DoR PASS RATE: X/10 (or X/N if N/A items excluded)**

### 5. Recommendations

Specific, actionable recommendations to make this story sprint-ready:

| Priority | Recommendation | Owner |
|----------|---------------|-------|
| P0 - Blocker | [must fix before sprint] | [who] |
| P1 - Important | [should fix before sprint] | [who] |
| P2 - Nice to have | [can refine during sprint] | [who] |

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/04-requirements-validation.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I score all 6 INVEST criteria?
+-- Did I calculate a single INVEST completeness score (not a range)?
+-- Did I assess every AC for testability?
+-- Did I provide rewrites for non-testable ACs?
+-- Did I identify gaps (or explicitly confirm none)?
+-- Did I count critical gaps?
+-- Did I complete the DoR checklist?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-requirements-validator",
  run_in_background: true
})
```

### Alternative: MCP Tools Execution

If using MCP instead of Task tool:

```javascript
// Option 1: Orchestrate via Fleet
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["requirements-validation", "test-generation"],
  maxAgents: 7,
  lazyLoading: true
})

// Submit tasks to specific domains
mcp__agentic-qe__task_submit({
  type: "product-factors-assessment",
  priority: "p0",
  payload: {
    storyContent: storyContent,
    framework: "sfdipot",
    factorCount: 7
  }
})

mcp__agentic-qe__task_submit({
  type: "bdd-scenario-generation",
  priority: "p0",
  payload: {
    storyContent: storyContent,
    format: "gherkin",
    minScenarios: 5
  }
})

mcp__agentic-qe__task_submit({
  type: "requirements-validation",
  priority: "p0",
  payload: {
    storyContent: storyContent,
    validateInvest: true,
    assessDoR: true
  }
})

// Check task status
mcp__agentic-qe__task_list({ status: "pending" })
```

### Alternative: CLI Execution

If using CLI instead of Task tool:

```bash
# Initialize swarm for refinement
npx @claude-flow/cli@latest swarm init \
  --topology hierarchical \
  --max-agents 7 \
  --strategy specialized

# Pre-task hook for routing
npx @claude-flow/cli@latest hooks pre-task \
  --description "QCSD Refinement: SFDIPOT, BDD Scenarios, INVEST Validation"

# Spawn agents (run in separate terminals or background)
npx @claude-flow/cli@latest agent spawn \
  --type qe-product-factors-assessor \
  --task "SFDIPOT 7-factor analysis for story" &

npx @claude-flow/cli@latest agent spawn \
  --type qe-bdd-generator \
  --task "Gherkin BDD scenario generation" &

npx @claude-flow/cli@latest agent spawn \
  --type qe-requirements-validator \
  --task "INVEST validation and DoR assessment" &

# Wait for completion
wait

# Check swarm status
npx @claude-flow/cli@latest swarm status
```

### Post-Spawn Confirmation

After sending all three Task calls, you MUST tell the user:

```
I've launched 3 core agents in parallel:

  qe-product-factors-assessor [Domain: requirements-validation]
   - Analyzing all 7 SFDIPOT factors (37 subcategories)
   - Assigning P0-P3 priorities per factor
   - Identifying test focus areas

  qe-bdd-generator [Domain: requirements-validation]
   - Generating Gherkin scenarios (happy, error, boundary, security)
   - Creating Scenario Outlines with Examples tables
   - Mapping scenarios to acceptance criteria

  qe-requirements-validator [Domain: requirements-validation]
   - Scoring INVEST criteria (6 dimensions)
   - Assessing AC testability with rewrites
   - Running Definition of Ready checklist

  WAITING for all agents to complete before proceeding...
```

**DO NOT proceed to Phase 3 until you have sent this confirmation.**

---

## PHASE 3: Wait for Batch 1 Completion

### ENFORCEMENT: NO EARLY PROCEEDING

```
+-------------------------------------------------------------+
|  YOU MUST WAIT FOR ALL THREE BACKGROUND TASKS TO COMPLETE    |
|                                                              |
|  DO NOT summarize what agents "would" find                   |
|  DO NOT proceed to Phase 4 early                             |
|  DO NOT provide your own analysis as substitute              |
|                                                              |
|  WAIT for actual agent results                               |
|  ONLY proceed when all three have returned                   |
+-------------------------------------------------------------+
```

### Results Extraction Checklist

When results return, extract and record:

```
From qe-product-factors-assessor:
[ ] sfdipotCoverage = __/7 factors analyzed
[ ] p0Count = __ P0 priority factors
[ ] subcategoriesAssessed = __/37 subcategories
[ ] testFocusAreas = __ areas identified

From qe-bdd-generator:
[ ] bddScenarioCount = __ scenarios generated
[ ] happyPathCount = __ happy path scenarios
[ ] errorPathCount = __ error path scenarios
[ ] boundaryCount = __ boundary scenarios
[ ] acCoverage = __% acceptance criteria covered

From qe-requirements-validator:
[ ] investCompleteness = __% (average of 6 INVEST criteria)
[ ] criticalGaps = __ critical gaps identified
[ ] dorPassRate = __/10 Definition of Ready items passing
[ ] untestableACs = __ acceptance criteria needing rewrite
```

### Metrics Summary Box

Output extracted metrics:

```
+-------------------------------------------------------------+
|                    BATCH 1 RESULTS SUMMARY                   |
+-------------------------------------------------------------+
|                                                              |
|  SFDIPOT Coverage:       __/7 factors                        |
|  P0 Factors:             __                                  |
|  Subcategories Assessed: __/37                               |
|                                                              |
|  BDD Scenarios:          __ total                            |
|  - Happy Path:           __                                  |
|  - Error Path:           __                                  |
|  - Boundary:             __                                  |
|  AC Coverage:            __%                                 |
|                                                              |
|  INVEST Completeness:    __%                                 |
|  Critical Gaps:          __                                  |
|  DoR Pass Rate:          __/10                               |
|  Untestable ACs:         __                                  |
|                                                              |
+-------------------------------------------------------------+
```

**DO NOT proceed to Phase 4 until ALL fields are filled.**

---

## PHASE 4: Spawn Conditional Agents (PARALLEL BATCH 2)

### ENFORCEMENT: NO SKIPPING CONDITIONAL AGENTS

```
+-------------------------------------------------------------+
|  IF A FLAG IS TRUE, YOU MUST SPAWN THAT AGENT                |
|                                                              |
|  HAS_API = TRUE              -> MUST spawn qe-contract-validator    |
|  HAS_REFACTORING = TRUE      -> MUST spawn qe-impact-analyzer      |
|  HAS_DEPENDENCIES = TRUE     -> MUST spawn qe-dependency-mapper     |
|  HAS_MIDDLEWARE = TRUE        -> MUST spawn qe-middleware-validator  |
|  HAS_SAP_INTEGRATION = TRUE  -> MUST spawn qe-odata-contract-tester|
|  HAS_AUTHORIZATION = TRUE    -> MUST spawn qe-sod-analyzer          |
|                                                              |
|  Skipping a flagged agent is a FAILURE of this skill.        |
+-------------------------------------------------------------+
```

### Conditional Domain Mapping

| Flag | Agent | Domain | MCP Tool |
|------|-------|--------|----------|
| HAS_API | qe-contract-validator | contract-testing | `contract_test` |
| HAS_REFACTORING | qe-impact-analyzer | code-intelligence | `coverage_analyze_sublinear` |
| HAS_DEPENDENCIES | qe-dependency-mapper | code-intelligence | `defect_predict` |
| HAS_MIDDLEWARE | qe-middleware-validator | enterprise-integration | `task_orchestrate` |
| HAS_SAP_INTEGRATION | qe-odata-contract-tester | enterprise-integration | `task_orchestrate` |
| HAS_AUTHORIZATION | qe-sod-analyzer | enterprise-integration | `task_orchestrate` |

### Decision Tree

```
IF HAS_API == FALSE AND HAS_REFACTORING == FALSE AND HAS_DEPENDENCIES == FALSE AND HAS_MIDDLEWARE == FALSE AND HAS_SAP_INTEGRATION == FALSE AND HAS_AUTHORIZATION == FALSE:
    -> Skip to Phase 5 (no conditional agents needed)
    -> State: "No conditional agents needed based on story analysis"

ELSE:
    -> Spawn ALL applicable agents in ONE message
    -> Count how many you're spawning: __
```

### IF HAS_API: Contract Validator (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Consumer-driven contract validation",
  prompt: `You are qe-contract-validator. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. API Contract Inventory

List EVERY API interaction mentioned or implied:

| Endpoint | Method | Consumer | Provider | Contract Defined? |
|----------|--------|----------|----------|--------------------|
| ... | GET/POST/etc | [who calls] | [who serves] | Yes/No |

### 2. Consumer-Driven Contract Assessment

For each API contract:

| Contract | Schema Defined? | Breaking Change Risk | Backward Compatible? |
|----------|-----------------|---------------------|---------------------|
| ... | Yes/No/Partial | High/Medium/Low | Yes/No/Unknown |

### 3. Breaking Change Detection

Analyze the story for potential breaking changes:

| Change | Type | Impact | Consumers Affected |
|--------|------|--------|-------------------|
| ... | Schema/Behavior/Removal | High/Medium/Low | [list] |

### 4. Contract Testing Recommendations

| Test Type | Tool | Priority | Scenarios |
|-----------|------|----------|-----------|
| Consumer contract | Pact/MSW | P0/P1/P2 | [scenarios] |
| Provider verification | Pact/Supertest | P0/P1/P2 | [scenarios] |
| Schema validation | Ajv/Zod | P0/P1/P2 | [scenarios] |

### 5. Contract Risk Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Schema completeness | X/10 | ... |
| Backward compatibility | X/10 | ... |
| Consumer coverage | X/10 | ... |
| Versioning strategy | X/10 | ... |

**CONTRACT RISK SCORE: X/40**

**MINIMUM: Identify at least 3 contract-related findings**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/05-contract-validation.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-contract-validator",
  run_in_background: true
})
```

### IF HAS_REFACTORING: Impact Analyzer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Refactoring blast radius and impact analysis",
  prompt: `You are qe-impact-analyzer. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Blast Radius Assessment

Identify all components affected by the refactoring:

| Component | Direct/Indirect | Change Type | Risk Level |
|-----------|----------------|-------------|------------|
| ... | Direct | Modified/Renamed/Moved/Deleted | High/Medium/Low |

### 2. Affected Test Inventory

| Test File/Suite | Status | Action Required |
|-----------------|--------|-----------------|
| ... | Will Break / May Break / Safe | Update / Rewrite / None |

### 3. Dependency Chain Analysis

Map the dependency chain for affected components:

```
[Changed Component]
  +-- [Direct Dependent 1]
  |     +-- [Transitive Dependent 1a]
  |     +-- [Transitive Dependent 1b]
  +-- [Direct Dependent 2]
        +-- [Transitive Dependent 2a]
```

### 4. Regression Risk Matrix

| Area | Change Probability | Test Coverage | Regression Risk |
|------|-------------------|---------------|-----------------|
| ... | High/Medium/Low | Good/Partial/None | Critical/High/Medium/Low |

### 5. Migration Strategy

| Phase | Action | Rollback Plan | Verification |
|-------|--------|---------------|-------------|
| 1 | ... | ... | [how to verify] |
| 2 | ... | ... | [how to verify] |

### 6. Impact Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Blast radius | X/10 | Number of affected components |
| Test impact | X/10 | Tests requiring updates |
| Rollback complexity | X/10 | Difficulty of rollback |
| Data migration risk | X/10 | Data transformation needed |

**IMPACT SCORE: X/40**

**MINIMUM: Map at least 5 affected components or explain why fewer exist**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/06-impact-analysis.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-impact-analyzer",
  run_in_background: true
})
```

### IF HAS_DEPENDENCIES: Dependency Mapper (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Dependency coupling analysis and circular detection",
  prompt: `You are qe-dependency-mapper. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Dependency Inventory

List ALL dependencies (direct and transitive) mentioned or implied:

| Dependency | Type | Version | Status | Risk |
|------------|------|---------|--------|------|
| ... | Direct/Transitive | X.Y.Z | Current/Outdated/Deprecated | High/Medium/Low |

### 2. Coupling Metrics

Calculate Robert C. Martin's package coupling metrics:

| Module | Ca (Afferent) | Ce (Efferent) | I (Instability) | Category |
|--------|---------------|---------------|-----------------|----------|
| ... | [incoming deps] | [outgoing deps] | Ce/(Ca+Ce) | Stable/Flexible/Unstable |

Where:
- Ca = number of classes outside that depend on this
- Ce = number of classes inside that depend on outside
- I = Ce / (Ca + Ce), range [0,1]: 0=maximally stable, 1=maximally unstable

### 3. Circular Dependency Detection

| Cycle | Components Involved | Severity | Resolution |
|-------|--------------------|----------|------------|
| Cycle 1 | A -> B -> C -> A | Critical/Warning | [how to break] |

If no cycles: "No circular dependencies detected"

### 4. Cross-Team Dependencies

| Dependency | Owning Team | Communication Channel | SLA | Risk |
|------------|-------------|----------------------|-----|------|
| ... | [team name] | [Slack/Jira/etc] | [response time] | High/Medium/Low |

### 5. Dependency Health Dashboard

| Health Indicator | Status | Details |
|-----------------|--------|---------|
| Outdated dependencies | X of Y | [list] |
| Known vulnerabilities | X CVEs | [list] |
| License compliance | Pass/Fail | [issues] |
| Bundle size impact | +X KB | [breakdown] |

### 6. Coupling Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Afferent coupling | X/10 | Fan-in complexity |
| Efferent coupling | X/10 | Fan-out complexity |
| Circular dependencies | X/10 | Cycle severity |
| Cross-team coupling | X/10 | External team risk |

**COUPLING SCORE: X/40**

**MINIMUM: Map at least 5 dependencies or explain why fewer exist**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/07-dependency-map.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-dependency-mapper",
  run_in_background: true
})
```

### IF HAS_MIDDLEWARE: Middleware Validator (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Middleware routing, transformation, and ESB flow validation",
  prompt: `You are qe-middleware-validator. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Middleware Topology Inventory

Map ALL middleware components mentioned or implied:

| Component | Type | Protocol | Direction | SLA |
|-----------|------|----------|-----------|-----|
| ... | ESB/MQ/Gateway/Broker | AMQP/MQTT/HTTP/gRPC | Inbound/Outbound/Bidirectional | [latency/throughput] |

### 2. Message Flow Analysis

For each message flow:

| Flow | Source | Middleware | Target | Transform? | Error Handling |
|------|--------|-----------|--------|------------|----------------|
| ... | [producer] | [broker/ESB] | [consumer] | Yes/No | DLQ/Retry/Drop |

### 3. Transformation Validation

| Transform | Input Format | Output Format | Validation Rules | Risk |
|-----------|-------------|---------------|-----------------|------|
| ... | JSON/XML/CSV | JSON/XML/CSV | [schema rules] | High/Medium/Low |

### 4. Middleware Testing Recommendations

| Test Type | Tool | Priority | Scenarios |
|-----------|------|----------|-----------|
| Message routing | WireMock/Testcontainers | P0/P1/P2 | [scenarios] |
| Transform accuracy | Schema validators | P0/P1/P2 | [scenarios] |
| Dead letter queue | DLQ monitors | P0/P1/P2 | [scenarios] |
| Throughput | Load generators | P0/P1/P2 | [scenarios] |

### 5. Middleware Risk Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Routing complexity | X/10 | ... |
| Transform accuracy | X/10 | ... |
| Error handling coverage | X/10 | ... |
| Throughput risk | X/10 | ... |

**MIDDLEWARE RISK SCORE: X/40**

**MINIMUM: Identify at least 3 middleware-related findings**

## OUTPUT FORMAT

Save to: ${"$"}{OUTPUT_FOLDER}/10-middleware-validation.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-middleware-validator",
  run_in_background: true
})
```

### IF HAS_SAP_INTEGRATION: OData Contract Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "OData contract and SAP service validation",
  prompt: `You are qe-odata-contract-tester. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. SAP Service Inventory

Map ALL SAP services and OData endpoints mentioned or implied:

| Service | Type | Protocol | Entity Set | Operations |
|---------|------|----------|-----------|------------|
| ... | OData V2/V4/RFC/BAPI | HTTP/RFC | [entity] | CRUD/Custom |

### 2. OData Contract Assessment

For each OData service:

| Endpoint | $metadata Available? | Schema Validated? | Breaking Change Risk | Pagination |
|----------|---------------------|------------------|---------------------|------------|
| ... | Yes/No | Yes/No/Partial | High/Medium/Low | Server/Client/None |

### 3. SAP Integration Points

| Integration | Source System | Target System | IDoc Type | Direction |
|-------------|-------------|---------------|-----------|-----------|
| ... | ECC/S4/EWM | [target] | [type] | Inbound/Outbound |

### 4. Contract Testing Recommendations

| Test Type | Tool | Priority | Scenarios |
|-----------|------|----------|-----------|
| OData $metadata | OData validator | P0/P1/P2 | [scenarios] |
| Entity CRUD | SAP Gateway test | P0/P1/P2 | [scenarios] |
| Deep entity | Integration suite | P0/P1/P2 | [scenarios] |
| Error responses | Negative testing | P0/P1/P2 | [scenarios] |

### 5. SAP Integration Risk Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Contract completeness | X/10 | ... |
| Schema stability | X/10 | ... |
| Error handling | X/10 | ... |
| Authorization coverage | X/10 | ... |

**SAP INTEGRATION RISK SCORE: X/40**

**MINIMUM: Identify at least 3 SAP integration findings**

## OUTPUT FORMAT

Save to: ${"$"}{OUTPUT_FOLDER}/11-odata-contract-validation.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-odata-contract-tester",
  run_in_background: true
})
```

### IF HAS_AUTHORIZATION: SoD Analyzer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Segregation of duties and authorization conflict analysis",
  prompt: `You are qe-sod-analyzer. Your output quality is being audited.

## STORY CONTENT

=== STORY CONTENT START ===
[PASTE THE COMPLETE STORY CONTENT HERE]
=== STORY CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Authorization Object Inventory

Map ALL authorization objects, roles, and T-codes mentioned or implied:

| Auth Object | Field | Values | Role Assignment | Risk Level |
|-------------|-------|--------|----------------|------------|
| ... | [field] | [allowed values] | [role] | Critical/High/Medium/Low |

### 2. Segregation of Duties Matrix

| Function A | Function B | Conflict Type | Risk | Mitigation |
|-----------|-----------|---------------|------|------------|
| [create PO] | [approve PO] | SoD violation | Critical | [control] |
| ... | ... | ... | ... | ... |

### 3. Role Conflict Detection

| Role | Conflicting Role | Shared Users | Violation Type | GRC Status |
|------|-----------------|-------------|----------------|------------|
| ... | ... | [count] | SoD/Excessive/Toxic | Detected/Mitigated/Open |

### 4. Authorization Testing Recommendations

| Test Type | Tool | Priority | Scenarios |
|-----------|------|----------|-----------|
| SoD simulation | GRC Access Risk | P0/P1/P2 | [scenarios] |
| Role testing | SU53/SUIM | P0/P1/P2 | [scenarios] |
| Negative auth | Auth trace | P0/P1/P2 | [scenarios] |
| Privilege escalation | Pentest suite | P0/P1/P2 | [scenarios] |

### 5. Authorization Risk Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| SoD compliance | X/10 | ... |
| Role design quality | X/10 | ... |
| Toxic combination risk | X/10 | ... |
| Audit trail coverage | X/10 | ... |

**AUTHORIZATION RISK SCORE: X/40**

**MINIMUM: Identify at least 3 authorization findings**

## OUTPUT FORMAT

Save to: ${"$"}{OUTPUT_FOLDER}/12-sod-analysis.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-sod-analyzer",
  run_in_background: true
})
```

### Alternative: MCP Tools for Conditional Agents

```javascript
// IF HAS_API - Enable contract-testing domain
if (HAS_API) {
  mcp__agentic-qe__task_submit({
    type: "contract-validation",
    priority: "p0",
    payload: {
      storyContent: storyContent,
      framework: "consumer-driven",
      detectBreaking: true
    }
  })
}

// IF HAS_REFACTORING - Enable code-intelligence domain
if (HAS_REFACTORING) {
  mcp__agentic-qe__coverage_analyze_sublinear({
    target: "src/",
    detectGaps: true,
    blastRadius: true
  })
}

// IF HAS_DEPENDENCIES - Enable code-intelligence domain
if (HAS_DEPENDENCIES) {
  mcp__agentic-qe__defect_predict({
    target: "src/",
    analyzeCoupling: true,
    detectCircular: true
  })
}

// IF HAS_MIDDLEWARE - Enable enterprise-integration domain
if (HAS_MIDDLEWARE) {
  mcp__agentic_qe__task_orchestrate({
    task: "middleware-routing-validation",
    strategy: "parallel",
    payload: {
      storyContent: storyContent,
      validateRouting: true,
      validateTransforms: true
    }
  })
}

// IF HAS_SAP_INTEGRATION - Enable enterprise-integration domain
if (HAS_SAP_INTEGRATION) {
  mcp__agentic_qe__task_orchestrate({
    task: "odata-contract-validation",
    strategy: "parallel",
    payload: {
      storyContent: storyContent,
      validateMetadata: true,
      validateSchema: true
    }
  })
}

// IF HAS_AUTHORIZATION - Enable enterprise-integration domain
if (HAS_AUTHORIZATION) {
  mcp__agentic_qe__task_orchestrate({
    task: "sod-authorization-analysis",
    strategy: "parallel",
    payload: {
      storyContent: storyContent,
      detectSoD: true,
      detectToxicCombos: true
    }
  })
}
```

### Alternative: CLI for Conditional Agents

```bash
# IF HAS_API
if [ "$HAS_API" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-contract-validator \
    --task "Consumer-driven contract validation" &
fi

# IF HAS_REFACTORING
if [ "$HAS_REFACTORING" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-impact-analyzer \
    --task "Refactoring blast radius analysis" &
fi

# IF HAS_DEPENDENCIES
if [ "$HAS_DEPENDENCIES" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-dependency-mapper \
    --task "Coupling metrics and circular detection" &
fi

# IF HAS_MIDDLEWARE
if [ "$HAS_MIDDLEWARE" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-middleware-validator \
    --task "Middleware routing and ESB flow validation" &
fi

# IF HAS_SAP_INTEGRATION
if [ "$HAS_SAP_INTEGRATION" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-odata-contract-tester \
    --task "OData contract and SAP service validation" &
fi

# IF HAS_AUTHORIZATION
if [ "$HAS_AUTHORIZATION" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-sod-analyzer \
    --task "Segregation of duties analysis" &
fi

# Wait for conditional agents
wait
```

### Agent Count Validation

**Before proceeding, verify agent count:**

```
+-------------------------------------------------------------+
|                   AGENT COUNT VALIDATION                     |
+-------------------------------------------------------------+
|                                                              |
|  CORE AGENTS (ALWAYS 3):                                     |
|    [ ] qe-product-factors-assessor - SPAWNED? [Y/N]         |
|    [ ] qe-bdd-generator - SPAWNED? [Y/N]                    |
|    [ ] qe-requirements-validator - SPAWNED? [Y/N]            |
|                                                              |
|  CONDITIONAL AGENTS (based on flags):                        |
|    [ ] qe-contract-validator - SPAWNED? [Y/N] (HAS_API)     |
|    [ ] qe-impact-analyzer - SPAWNED? [Y/N] (HAS_REFACTORING)|
|    [ ] qe-dependency-mapper - SPAWNED? [Y/N] (HAS_DEPS)     |
|    [ ] qe-middleware-validator - SPAWNED? [Y/N] (HAS_MIDDLEWARE)    |
|    [ ] qe-odata-contract-tester - SPAWNED? [Y/N] (HAS_SAP_INTEG)  |
|    [ ] qe-sod-analyzer - SPAWNED? [Y/N] (HAS_AUTHORIZATION)       |
|                                                              |
|  VALIDATION:                                                 |
|    Expected agents: [3 + count of TRUE flags]                |
|    Actual spawned:  [count]                                  |
|    Status:          [PASS/FAIL]                              |
|                                                              |
|  If ACTUAL < EXPECTED, you have FAILED. Spawn missing        |
|  agents before proceeding.                                   |
|                                                              |
+-------------------------------------------------------------+
```

**DO NOT proceed if validation FAILS.**

### Post-Spawn Confirmation (If Applicable)

```
I've launched [N] conditional agent(s) in parallel:

[IF HAS_API]             qe-contract-validator [Domain: contract-testing]
                         - Consumer-driven contracts, breaking changes
[IF HAS_REFACTORING]     qe-impact-analyzer [Domain: code-intelligence]
                         - Blast radius, affected tests, regression risk
[IF HAS_DEPENDENCIES]    qe-dependency-mapper [Domain: code-intelligence]
                         - Coupling metrics Ca/Ce/I, circular detection
[IF HAS_MIDDLEWARE]      qe-middleware-validator [Domain: enterprise-integration]
                         - Middleware routing, transformation, ESB flow validation
[IF HAS_SAP_INTEGRATION] qe-odata-contract-tester [Domain: enterprise-integration]
                         - OData contract, SAP service, $metadata validation
[IF HAS_AUTHORIZATION]   qe-sod-analyzer [Domain: enterprise-integration]
                         - Segregation of duties, role conflict, authorization analysis

  WAITING for conditional agents to complete...
```

---

## PHASE 5: Synthesize Results & Determine Recommendation

### ENFORCEMENT: EXACT DECISION LOGIC

**You MUST apply this logic EXACTLY. No interpretation.**

```
STEP 1: Check NOT-READY conditions (ANY triggers NOT-READY)
-----------------------------------------------------------
IF sfdipotCoverage < 5          -> NOT-READY ("Insufficient factor coverage")
IF bddScenarioCount < 3         -> NOT-READY ("Too few BDD scenarios")
IF investCompleteness < 50       -> NOT-READY ("Requirements incomplete")
IF criticalGaps > 3              -> NOT-READY ("Too many critical gaps")

STEP 2: Check READY conditions (ALL required for READY)
-----------------------------------------------------------
IF sfdipotCoverage >= 7
   AND bddScenarioCount >= 10
   AND investCompleteness >= 90
   AND criticalGaps == 0         -> READY

STEP 3: Default
-----------------------------------------------------------
ELSE                             -> CONDITIONAL
```

### Decision Recording

```
METRICS:
- sfdipotCoverage = __/7
- bddScenarioCount = __
- investCompleteness = __%
- criticalGaps = __

NOT-READY CHECK:
- sfdipotCoverage < 5? __ (YES/NO)
- bddScenarioCount < 3? __ (YES/NO)
- investCompleteness < 50? __ (YES/NO)
- criticalGaps > 3? __ (YES/NO)

READY CHECK (only if no NOT-READY triggered):
- sfdipotCoverage >= 7? __ (YES/NO)
- bddScenarioCount >= 10? __ (YES/NO)
- investCompleteness >= 90? __ (YES/NO)
- criticalGaps == 0? __ (YES/NO)

FINAL RECOMMENDATION: [READY / CONDITIONAL / NOT-READY]
REASON: ___
```

### Conditional Recommendations

If recommendation is CONDITIONAL, provide specific blockers:

| Blocker | Current Value | Required Value | Owner | Action |
|---------|--------------|----------------|-------|--------|
| ... | ... | ... | [who] | [what to do] |

If recommendation is NOT-READY, provide mandatory remediation:

| Remediation | Priority | Effort | Deadline |
|-------------|----------|--------|----------|
| ... | P0 | [hours/days] | [before next refinement] |

---

## PHASE 6: Generate Refinement Report

### ENFORCEMENT: COMPLETE REPORT STRUCTURE

**ALL sections below are MANDATORY. No abbreviations.**

```markdown
# QCSD Refinement Report: [Story Name]

**Generated**: [Date/Time]
**Recommendation**: [READY / CONDITIONAL / NOT-READY]
**Agents Executed**: [List all agents that ran]
**Parallel Batches**: [2 or 3 depending on conditional agents]

---

## Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| SFDIPOT Coverage | X/7 | >=7 | PASS/WARN/FAIL |
| BDD Scenarios | X | >=10 | PASS/WARN/FAIL |
| INVEST Completeness | X% | >=90% | PASS/WARN/FAIL |
| Critical Gaps | X | 0 | PASS/WARN/FAIL |

**Recommendation Rationale**: [1-2 sentences explaining why READY/CONDITIONAL/NOT-READY]

---

## SFDIPOT Product Factors Analysis

[EMBED or LINK the full report from qe-product-factors-assessor]

### Factor Priority Summary

| Factor | Priority | Key Finding | Test Focus |
|--------|----------|-------------|------------|
| Structure | P0/P1/P2/P3 | [top finding] | [areas] |
| Function | P0/P1/P2/P3 | [top finding] | [areas] |
| Data | P0/P1/P2/P3 | [top finding] | [areas] |
| Interfaces | P0/P1/P2/P3 | [top finding] | [areas] |
| Platform | P0/P1/P2/P3 | [top finding] | [areas] |
| Operations | P0/P1/P2/P3 | [top finding] | [areas] |
| Time | P0/P1/P2/P3 | [top finding] | [areas] |

### Cross-Cutting Concerns
[List any concerns that span multiple factors]

---

## BDD Scenarios

[EMBED the full Gherkin from qe-bdd-generator]

### Scenario Coverage Summary

| Category | Count | AC Coverage |
|----------|-------|-------------|
| Happy Path | X | [ACs covered] |
| Error Path | X | [ACs covered] |
| Boundary | X | [ACs covered] |
| Security | X | [ACs covered] |
| **Total** | **X** | **X%** |

---

## Requirements Validation

### INVEST Score: X/100

| Criterion | Score | Status |
|-----------|-------|--------|
| Independent | X/100 | PASS/WARN/FAIL |
| Negotiable | X/100 | PASS/WARN/FAIL |
| Valuable | X/100 | PASS/WARN/FAIL |
| Estimable | X/100 | PASS/WARN/FAIL |
| Small | X/100 | PASS/WARN/FAIL |
| Testable | X/100 | PASS/WARN/FAIL |

### AC Assessment

| AC | Testable? | Issues | Rewrite Needed? |
|----|-----------|--------|-----------------|
[All ACs evaluated]

### Gaps Identified
1. [Gap 1]
2. [Gap 2]
[All gaps from qe-requirements-validator]

### Definition of Ready: X/10 passing

---

## Conditional Analysis

[INCLUDE ONLY IF APPLICABLE - based on which conditional agents ran]

### Contract Validation (IF HAS_API)
[Full output from qe-contract-validator]

### Impact Analysis (IF HAS_REFACTORING)
[Full output from qe-impact-analyzer]

### Dependency Map (IF HAS_DEPENDENCIES)
[Full output from qe-dependency-mapper]

---

## Recommended Actions

### Before Sprint Commitment
- [ ] [Action based on findings]
- [ ] [Action based on findings]

### Sprint Planning Notes
- [ ] [Testing approach recommendation]
- [ ] [Dependency coordination needed]

### During Sprint
- [ ] [Action based on findings]

---

## Appendix: Agent Outputs

[Link to or embed full outputs from each agent]

---

*Generated by QCSD Refinement Swarm v1.0*
*Execution Model: Task Tool Parallel Swarm*
```

Write the executive summary report to:
`${OUTPUT_FOLDER}/01-executive-summary.md`

### Report Validation Checklist

Before presenting report:

```
+-- Executive Summary table is complete with all 4 metrics
+-- Recommendation matches decision logic output
+-- SFDIPOT section includes all 7 factor priorities
+-- BDD section includes scenario coverage summary
+-- INVEST score shows all 6 criteria
+-- All gaps are listed
+-- Conditional sections included for all spawned agents
+-- Recommended actions are specific (not generic)
+-- Report saved to output folder
```

**DO NOT present an incomplete report.**

---

## PHASE 7: Store Learnings & Persist State

### ENFORCEMENT: ALWAYS RUN THIS PHASE

```
+-------------------------------------------------------------+
|  LEARNING PERSISTENCE MUST ALWAYS EXECUTE                    |
|                                                              |
|  This is NOT optional. It runs on EVERY refinement.          |
|  It stores findings for cross-phase feedback loops,          |
|  historical decision analysis, and pattern learning.         |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
|  DO NOT treat this as "nice to have".                        |
|  Enforcement Rule E9 applies.                                |
+-------------------------------------------------------------+
```

### Purpose

Store refinement findings for:
- Cross-phase feedback loops (Development -> next Refinement cycle)
- Historical analysis of READY/CONDITIONAL/NOT-READY decisions
- SFDIPOT pattern learning across stories
- BDD scenario quality improvement over time

### Auto-Execution Steps (ALL THREE are MANDATORY)

**Step 1: Store refinement findings to memory**

You MUST execute this MCP call with actual values from the refinement analysis:

```javascript
mcp__agentic-qe__memory_store({
  key: `qcsd-refinement-${storyId}-${Date.now()}`,
  namespace: "qcsd-refinement",
  value: {
    storyId: storyId,
    storyName: storyName,
    recommendation: recommendation,  // READY, CONDITIONAL, NOT-READY
    metrics: {
      sfdipotCoverage: sfdipotCoverage,
      bddScenarioCount: bddScenarioCount,
      investCompleteness: investCompleteness,
      criticalGaps: criticalGaps
    },
    sfdipotPriorities: {
      structure: "P0/P1/P2/P3",
      function: "P0/P1/P2/P3",
      data: "P0/P1/P2/P3",
      interfaces: "P0/P1/P2/P3",
      platform: "P0/P1/P2/P3",
      operations: "P0/P1/P2/P3",
      time: "P0/P1/P2/P3"
    },
    flags: {
      HAS_API: HAS_API,
      HAS_REFACTORING: HAS_REFACTORING,
      HAS_DEPENDENCIES: HAS_DEPENDENCIES,
      HAS_SECURITY: HAS_SECURITY,
      HAS_MIDDLEWARE: HAS_MIDDLEWARE,
      HAS_SAP_INTEGRATION: HAS_SAP_INTEGRATION,
      HAS_AUTHORIZATION: HAS_AUTHORIZATION
    },
    agentsInvoked: agentList,
    timestamp: new Date().toISOString()
  }
})
```

**Step 2: Share learnings with learning coordinator**

You MUST execute this MCP call to propagate patterns cross-domain:

```javascript
mcp__agentic-qe__memory_share({
  sourceAgentId: "qcsd-refinement-swarm",
  targetAgentIds: ["qe-learning-coordinator", "qe-pattern-learner"],
  knowledgeDomain: "refinement-patterns"
})
```

**Step 3: Save learning persistence record to output folder**

You MUST use the Write tool to save a JSON record of the persisted learnings:

```
Save to: ${OUTPUT_FOLDER}/09-learning-persistence.json

Contents:
{
  "phase": "QCSD-Refinement",
  "storyId": "[story ID]",
  "storyName": "[story name]",
  "recommendation": "[READY/CONDITIONAL/NOT-READY]",
  "memoryKey": "qcsd-refinement-[storyId]-[timestamp]",
  "namespace": "qcsd-refinement",
  "metrics": {
    "sfdipotCoverage": [0-7],
    "bddScenarioCount": [N],
    "investCompleteness": [0-100],
    "criticalGaps": [N],
    "testIdeaQuality": [0-100]
  },
  "sfdipotPriorities": {
    "structure": "P0/P1/P2/P3",
    "function": "P0/P1/P2/P3",
    "data": "P0/P1/P2/P3",
    "interfaces": "P0/P1/P2/P3",
    "platform": "P0/P1/P2/P3",
    "operations": "P0/P1/P2/P3",
    "time": "P0/P1/P2/P3"
  },
  "flags": {
    "HAS_API": true/false,
    "HAS_REFACTORING": true/false,
    "HAS_DEPENDENCIES": true/false,
    "HAS_SECURITY": true/false,
    "HAS_MIDDLEWARE": true/false,
    "HAS_SAP_INTEGRATION": true/false,
    "HAS_AUTHORIZATION": true/false
  },
  "agentsInvoked": ["list", "of", "agents"],
  "crossPhaseSignals": {
    "toDevelopment": "BDD scenarios as test specification",
    "toVerification": "INVEST gaps as verification focus"
  },
  "persistedAt": "[ISO timestamp]"
}
```

### Fallback: CLI Memory Commands

If MCP memory_store tool is unavailable, use CLI instead (STILL MANDATORY):

```bash
npx @claude-flow/cli@latest memory store \
  --key "qcsd-refinement-${STORY_ID}-$(date +%s)" \
  --value '{"recommendation":"[VALUE]","investCompleteness":[N],"sfdipotCoverage":[N],"bddScenarioCount":[N],"criticalGaps":[N]}' \
  --namespace qcsd-refinement

npx @claude-flow/cli@latest hooks post-task \
  --task-id "qcsd-refinement-${STORY_ID}" \
  --success true
```

### Validation Before Proceeding to Phase 8

```
+-- Did I execute mcp__agentic-qe__memory_store with actual values? (not placeholders)
+-- Did I execute mcp__agentic-qe__memory_share to propagate learnings?
+-- Did I save 09-learning-persistence.json to the output folder?
+-- Does the JSON contain the correct recommendation from Phase 5?
+-- Does the JSON contain actual SFDIPOT priorities from Phase 2?
+-- Does the JSON contain actual flag values from Phase 1?
```

**If ANY validation check fails, DO NOT proceed to Phase 8.**

### Cross-Phase Signal Consumption

The Refinement Swarm both consumes and produces signals for other QCSD phases:

```
CONSUMES (from other phases):
+-- Loop 2 (Tactical): SFDIPOT weights from Production Telemetry
|   - Production incidents inform which factors to weight higher
|   - Telemetry data adjusts P0-P3 priority thresholds
|
+-- Loop 4 (Quality-Criteria): Untestable patterns from Development
    - Patterns the Development phase flagged as hard to test
    - Feeds into INVEST testability scoring

PRODUCES (for other phases):
+-- To Development Phase: BDD scenarios as test specification
|   - Gherkin scenarios become automated test templates
|   - SFDIPOT priorities guide test depth allocation
|
+-- To Verification Phase: INVEST gaps as verification focus
    - Critical gaps become verification checkpoints
    - DoR failures become sprint exit criteria
```

---

## PHASE 8: Apply Test Idea Rewriter (Transformation)

### ENFORCEMENT: ALWAYS RUN THIS PHASE

```
+-------------------------------------------------------------+
|  THE TEST IDEA REWRITER MUST ALWAYS RUN                      |
|                                                              |
|  This is NOT conditional. It runs on EVERY refinement.       |
|  It transforms passive test ideas into active, actionable    |
|  test specifications using strong action verbs.              |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
+-------------------------------------------------------------+
```

### Transformation Rules

The qe-test-idea-rewriter transforms ALL test ideas from the refinement
analysis using these rules:

| Pattern | Transform To | Example |
|---------|-------------|---------|
| "Verify that X" | "Confirm X by [specific action]" | "Verify login works" -> "Confirm authentication succeeds by submitting valid credentials and observing session token" |
| "Check if X" | "Validate X by [specific action]" | "Check if error shows" -> "Validate error message displays by submitting empty form" |
| "Test X" | "Exercise X by [specific action]" | "Test pagination" -> "Exercise pagination by navigating through 5+ pages and verifying item counts" |
| "Ensure X" | "Demonstrate X by [specific action]" | "Ensure data saves" -> "Demonstrate data persistence by creating record and retrieving it" |
| Passive voice | Active voice | "Data is validated" -> "Validate data against schema constraints" |
| Vague assertion | Concrete assertion | "Works correctly" -> "Returns HTTP 200 with expected JSON payload" |

### Agent Spawn

```
Task({
  description: "Transform test ideas with action verbs",
  prompt: `You are qe-test-idea-rewriter. Your output quality is being audited.

## PURPOSE

Transform ALL test ideas generated during this refinement session from
passive/vague descriptions into active, specific, actionable test
specifications.

## INPUT: RAW TEST IDEAS

Collect test ideas from ALL previous agent outputs:

### From SFDIPOT Analysis (02-sfdipot-analysis.md):
[List all test focus areas identified per factor]

### From BDD Scenarios (03-bdd-scenarios.md):
[List all scenario names as test ideas]

### From Requirements Validation (04-requirements-validation.md):
[List all gap-driven test ideas and recommendations]

### From Conditional Agents (if applicable):
[List all test recommendations from contract/impact/dependency analysis]

## TRANSFORMATION RULES (APPLY TO EVERY TEST IDEA)

1. Replace passive verbs with active verbs:
   - "Verify" -> "Confirm ... by [action]"
   - "Check" -> "Validate ... by [action]"
   - "Test" -> "Exercise ... by [action]"
   - "Ensure" -> "Demonstrate ... by [action]"
   - "Should" -> "[Action verb] ... to [outcome]"

2. Add concrete actions:
   - Every test idea must specify WHAT action triggers the test
   - Every test idea must specify HOW to observe the result
   - No vague phrases: "works correctly", "functions properly", "behaves as expected"

3. Add expected outcomes:
   - Every test idea must have a specific, verifiable assertion
   - Use concrete values where possible (HTTP codes, counts, strings)

## REQUIRED OUTPUT

### Rewritten Test Ideas Table

| # | Original Test Idea | Rewritten Test Idea | Source Agent | Priority |
|---|-------------------|---------------------|-------------|----------|
| 1 | [original] | [rewritten with action verb + concrete assertion] | [agent] | P0/P1/P2 |
| 2 | ... | ... | ... | ... |

### Transformation Metrics

| Metric | Value |
|--------|-------|
| Total test ideas processed | __ |
| Ideas requiring rewrite | __ |
| Ideas already actionable | __ |
| New ideas added (from gaps) | __ |
| P0 test ideas | __ |
| P1 test ideas | __ |
| P2 test ideas | __ |

### Test Idea Quality Score

| Dimension | Before | After |
|-----------|--------|-------|
| Action verb usage | __% | 100% |
| Concrete assertions | __% | __% |
| Specific test data | __% | __% |
| Observable outcomes | __% | __% |

**OVERALL TEST IDEA QUALITY: __/100**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/08-rewritten-test-ideas.md
Use the Write tool to save BEFORE completing.

## VALIDATION BEFORE SUBMITTING

+-- Did I process ALL test ideas from ALL agent outputs?
+-- Does every rewritten idea use an active verb?
+-- Does every rewritten idea have a concrete assertion?
+-- Did I calculate transformation metrics?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-test-idea-rewriter",
  run_in_background: true
})
```

### Wait for Transformation Completion

```
+-------------------------------------------------------------+
|  WAIT for qe-test-idea-rewriter to complete before           |
|  proceeding to Phase 9.                                      |
|                                                              |
|  The rewritten test ideas are the PRIMARY deliverable of     |
|  the Refinement Swarm - they feed directly into the          |
|  Development phase as test specifications.                   |
+-------------------------------------------------------------+
```

---

## PHASE 9: Final Output

**At the very end of swarm execution, ALWAYS output this completion summary:**

```
+---------------------------------------------------------------------+
|                  QCSD REFINEMENT SWARM COMPLETE                      |
+---------------------------------------------------------------------+
|                                                                      |
|  Story Analyzed: [Story Name/ID]                                     |
|  Reports Generated: [count]                                          |
|  Output Folder: ${OUTPUT_FOLDER}                                     |
|                                                                      |
|  REFINEMENT SCORES:                                                  |
|  +-- SFDIPOT Coverage:     __/7 factors                              |
|  +-- BDD Scenarios:        __ generated                              |
|  +-- INVEST Completeness:  __%                                       |
|  +-- Critical Gaps:        __                                        |
|  +-- Test Idea Quality:    __/100                                    |
|                                                                      |
|  RECOMMENDATION: [READY / CONDITIONAL / NOT-READY]                   |
|  REASON: [1-2 sentence rationale]                                    |
|                                                                      |
|  DELIVERABLES:                                                       |
|  +-- 01-executive-summary.md                                         |
|  +-- 02-sfdipot-analysis.md                                          |
|  +-- 03-bdd-scenarios.md                                             |
|  +-- 04-requirements-validation.md                                   |
|  [IF HAS_API]                                                        |
|  +-- 05-contract-validation.md                                       |
|  [IF HAS_REFACTORING]                                                |
|  +-- 06-impact-analysis.md                                           |
|  [IF HAS_DEPENDENCIES]                                               |
|  +-- 07-dependency-map.md                                            |
|  +-- 08-rewritten-test-ideas.md                                      |
|  +-- 09-learning-persistence.json                                    |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is NOT-READY, ALSO output this prominent action box:**

```
+---------------------------------------------------------------------+
|  ACTION REQUIRED: STORY IS NOT READY FOR SPRINT                      |
+---------------------------------------------------------------------+
|                                                                      |
|  The following blockers MUST be resolved before next refinement:     |
|                                                                      |
|  1. [Blocker 1 with specific remediation]                            |
|  2. [Blocker 2 with specific remediation]                            |
|  3. [Blocker 3 with specific remediation]                            |
|                                                                      |
|  NEXT STEPS:                                                         |
|  - Address all P0 blockers listed above                              |
|  - Re-run /qcsd-refinement-swarm after fixes                        |
|  - Target: INVEST completeness >= 90%, 0 critical gaps               |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is CONDITIONAL, output this guidance box:**

```
+---------------------------------------------------------------------+
|  CONDITIONAL: STORY NEEDS MINOR ADJUSTMENTS                          |
+---------------------------------------------------------------------+
|                                                                      |
|  The story can enter the sprint WITH these conditions:               |
|                                                                      |
|  1. [Condition 1 - must be addressed in first 2 days]                |
|  2. [Condition 2 - must be addressed before testing]                 |
|                                                                      |
|  RISK ACCEPTANCE:                                                    |
|  - Team acknowledges remaining gaps                                  |
|  - Sprint scope may need adjustment if conditions not met            |
|                                                                      |
+---------------------------------------------------------------------+
```

**DO NOT end the swarm without displaying the completion summary.**

---

## Report Filename Mapping

| Agent | Report Filename | Phase |
|-------|----------------|-------|
| qe-product-factors-assessor | `02-sfdipot-analysis.md` | Batch 1 |
| qe-bdd-generator | `03-bdd-scenarios.md` | Batch 1 |
| qe-requirements-validator | `04-requirements-validation.md` | Batch 1 |
| qe-contract-validator | `05-contract-validation.md` | Batch 2 (conditional) |
| qe-impact-analyzer | `06-impact-analysis.md` | Batch 2 (conditional) |
| qe-dependency-mapper | `07-dependency-map.md` | Batch 2 (conditional) |
| qe-test-idea-rewriter | `08-rewritten-test-ideas.md` | Batch 3 (transformation) |
| Learning Persistence | `09-learning-persistence.json` | Phase 7 (auto-execute) |
| Synthesis | `01-executive-summary.md` | Phase 6 |

---

## DDD Domain Integration

This swarm operates across **1 primary domain**, **3 conditional domains**,
and **1 transformation domain**:

```
+-----------------------------------------------------------------------------+
|                    QCSD REFINEMENT - DOMAIN MAP                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|  PRIMARY DOMAIN (Always Active)                                              |
|  +-----------------------------------------------------------------------+  |
|  |                    requirements-validation                            |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-product-factors-assessor (SFDIPOT 7 factors)                   |  |
|  |  - qe-bdd-generator (Gherkin scenarios)                              |  |
|  |  - qe-requirements-validator (INVEST + DoR)                          |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  CONDITIONAL DOMAINS (Based on Story Content)                                |
|  +-----------------------+  +-----------------------+                        |
|  |  contract-testing     |  |  code-intelligence    |                        |
|  |  -------------------  |  |  -------------------  |                        |
|  |  qe-contract-         |  |  qe-impact-analyzer   |                        |
|  |    validator           |  |  qe-dependency-mapper |                        |
|  |  [IF HAS_API]         |  |  [IF HAS_REFACTORING  |                        |
|  |                       |  |   or HAS_DEPENDENCIES]|                        |
|  +-----------------------+  +-----------------------+                        |
|                                                                              |
|  +------------------------------------------+                                |
|  |  enterprise-integration                   |                                |
|  |  ---------------------------------------- |                                |
|  |  qe-middleware-validator                  |                                |
|  |    [IF HAS_MIDDLEWARE]                    |                                |
|  |  qe-odata-contract-tester                |                                |
|  |    [IF HAS_SAP_INTEGRATION]              |                                |
|  |  qe-sod-analyzer                         |                                |
|  |    [IF HAS_AUTHORIZATION]                |                                |
|  +------------------------------------------+                                |
|                                                                              |
|  TRANSFORMATION DOMAIN (Always Active)                                       |
|  +-----------------------------------------------------------------------+  |
|  |                      test-generation                                  |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-test-idea-rewriter (action verb transformation)                |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## Execution Model Options

This skill supports **3 execution models**. Choose based on your environment:

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **Task Tool** (PRIMARY) | Claude Code sessions | Full agent capabilities, parallel execution | Requires Claude Code |
| **MCP Tools** | MCP server available | Fleet coordination, memory persistence | Requires MCP setup |
| **CLI** | Terminal/scripts | Works anywhere, scriptable | Sequential only |

### Quick Start by Model

**Option A: Task Tool (RECOMMENDED)**
```
Just follow the skill phases below - uses Task() calls with run_in_background: true
```

**Option B: MCP Tools**
```javascript
// Initialize fleet for Refinement domains
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["requirements-validation", "contract-testing", "code-intelligence", "test-generation"],
  maxAgents: 7
})

// Orchestrate refinement task
mcp__agentic-qe__task_orchestrate({
  task: "qcsd-refinement-analysis",
  strategy: "parallel"
})
```

**Option C: CLI**
```bash
# Initialize coordination
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 7

# Route task
npx @claude-flow/cli@latest hooks pre-task --description "QCSD Refinement for [Story]"

# Execute agents
npx @claude-flow/cli@latest agent spawn --type qe-product-factors-assessor
npx @claude-flow/cli@latest agent spawn --type qe-bdd-generator
npx @claude-flow/cli@latest agent spawn --type qe-requirements-validator
```

---

## Quick Reference

### Enforcement Summary

| Phase | Must Do | Failure Condition |
|-------|---------|-------------------|
| 1 | Check ALL 7 flags | Missing flag evaluation |
| 2 | Spawn ALL 3 core agents in ONE message | Fewer than 3 Task calls |
| 3 | WAIT for completion | Proceeding before results |
| 4 | Spawn ALL flagged conditional agents | Skipping a TRUE flag |
| 5 | Apply EXACT decision logic | Wrong recommendation |
| 6 | Generate COMPLETE report | Missing sections |
| 7 | ALWAYS store learnings + save 09-learning-persistence.json | Pattern loss, missing audit trail |
| 8 | ALWAYS run test idea rewriter | Skipping transformation |
| 9 | Output completion summary | Missing final output |

### Quality Gate Thresholds

| Metric | READY | CONDITIONAL | NOT-READY |
|--------|-------|-------------|-----------|
| SFDIPOT Coverage | >=7/7 | 5-6/7 | <5/7 |
| BDD Scenarios | >=10 | 3-9 | <3 |
| INVEST Completeness | >=90% | 50-89% | <50% |
| Critical Gaps | 0 | 1-3 | >3 |

### Domain-to-Agent Mapping

| Domain | Agent | Phase | Batch |
|--------|-------|-------|-------|
| requirements-validation | qe-product-factors-assessor | Core | 1 |
| requirements-validation | qe-bdd-generator | Core | 1 |
| requirements-validation | qe-requirements-validator | Core | 1 |
| contract-testing | qe-contract-validator | Conditional (HAS_API) | 2 |
| code-intelligence | qe-impact-analyzer | Conditional (HAS_REFACTORING) | 2 |
| code-intelligence | qe-dependency-mapper | Conditional (HAS_DEPENDENCIES) | 2 |
| enterprise-integration | qe-middleware-validator | Conditional (HAS_MIDDLEWARE) | 2 |
| enterprise-integration | qe-odata-contract-tester | Conditional (HAS_SAP_INTEGRATION) | 2 |
| enterprise-integration | qe-sod-analyzer | Conditional (HAS_AUTHORIZATION) | 2 |
| test-generation | qe-test-idea-rewriter | Transformation (ALWAYS) | 3 |

### Execution Model Quick Reference

| Model | Initialization | Agent Spawn | Memory Store |
|-------|---------------|-------------|--------------|
| **Task Tool** | N/A | `Task({ subagent_type, run_in_background: true })` | N/A (use MCP) |
| **MCP Tools** | `fleet_init({})` | `task_submit({})` | `memory_store({})` |
| **CLI** | `swarm init` | `agent spawn` | `memory store` |

### MCP Tools Quick Reference

```javascript
// Initialization
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["requirements-validation", "contract-testing", "code-intelligence", "test-generation"],
  maxAgents: 7
})

// Task submission
mcp__agentic-qe__task_submit({ type: "...", priority: "p0", payload: {...} })
mcp__agentic-qe__task_orchestrate({ task: "...", strategy: "parallel" })

// Status
mcp__agentic-qe__fleet_status({ verbose: true })
mcp__agentic-qe__task_list({ status: "pending" })

// Memory
mcp__agentic-qe__memory_store({ key: "...", value: {...}, namespace: "qcsd-refinement" })
mcp__agentic-qe__memory_query({ pattern: "qcsd-refinement-*", namespace: "qcsd-refinement" })
mcp__agentic-qe__memory_share({
  sourceAgentId: "qcsd-refinement-swarm",
  targetAgentIds: ["qe-learning-coordinator"],
  knowledgeDomain: "refinement-patterns"
})
```

### CLI Quick Reference

```bash
# Initialization
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 7

# Agent operations
npx @claude-flow/cli@latest agent spawn --type [agent-type] --task "[description]"
npx @claude-flow/cli@latest hooks pre-task --description "[task]"
npx @claude-flow/cli@latest hooks post-task --task-id "[id]" --success true

# Status
npx @claude-flow/cli@latest swarm status

# Memory
npx @claude-flow/cli@latest memory store --key "[key]" --value "[json]" --namespace qcsd-refinement
npx @claude-flow/cli@latest memory search --query "[query]" --namespace qcsd-refinement
npx @claude-flow/cli@latest memory list --namespace qcsd-refinement
```

---

## Swarm Topology

```
                 QCSD REFINEMENT SWARM v1.0
                          |
          BATCH 1 (Core - Parallel)
          +-----------+---+-----------+
          |           |               |
    +-----v-----+ +---v---+ +--------v--------+
    | Product   | |  BDD  | | Requirements    |
    | Factors   | | Gen   | | Validator       |
    | (SFDIPOT) | |(Ghrkn)| | (INVEST+DoR)   |
    |-----------| |-------| |-----------------|
    | req-valid | |req-val| | req-valid       |
    +-----+-----+ +---+---+ +--------+--------+
          |            |              |
          +------------+--------------+
                       |
                [METRICS GATE]
                       |
          BATCH 2 (Conditional - Parallel)
          +-----------+---+-----------+
          |           |               |
    +-----v-----+ +---v--------+ +---v----------+
    | Contract  | | Impact     | | Dependency   |
    | Validator | | Analyzer   | | Mapper       |
    | [IF API]  | | [IF REFAC] | | [IF DEPS]    |
    |-----------| |------------| |--------------|
    | contract  | | code-intel | | code-intel   |
    +-----------+ +------------+ +--------------+
                       |
                [SYNTHESIS]
                       |
          PHASE 7 (Learning Persistence - Always)
                       |
               +-------v-------+
               | memory_store  |
               | memory_share  |
               | 09-learning-  |
               | persistence   |
               | (ALWAYS RUNS) |
               +-------+-------+
                       |
          BATCH 3 (Transformation - Always)
                       |
               +-------v-------+
               | Test Idea     |
               | Rewriter      |
               | (ALWAYS RUNS) |
               |---------------|
               | test-gen      |
               +-------+-------+
                       |
              [FINAL REPORT]
```

---

## Inventory Summary

| Resource Type | Count | Primary | Conditional | Transformation |
|---------------|:-----:|:-------:|:-----------:|:--------------:|
| **Agents** | 10 | 3 | 6 | 1 |
| **Sub-agents** | 0 | - | - | - |
| **Skills** | 3 | 3 | - | - |
| **Domains** | 5 | 1 | 3 | 1 |
| **Parallel Batches** | 3 | 1 | 1 | 1 |

**Skills Used:**
1. `context-driven-testing` - Context-appropriate test strategy
2. `testability-scoring` - 10 testability principles
3. `risk-based-testing` - Risk prioritization matrix

**Frameworks Applied:**
1. SFDIPOT - 7 product factors, 37 subcategories
2. BDD/Gherkin - Scenario specification language
3. INVEST - Story quality validation (6 criteria)
4. DoR - Definition of Ready checklist (10 items)
5. Robert C. Martin coupling metrics (Ca/Ce/I)

---

## Key Principle

**Stories enter sprints ready to test, not ready to argue about.**

This swarm provides:
1. **What product factors matter?** -> SFDIPOT Analysis (7 factors, 37 subcategories)
2. **How should we test it?** -> BDD Scenarios (Gherkin specifications)
3. **Are requirements complete?** -> INVEST Validation (6 criteria + DoR)
4. **Are contracts safe?** -> Contract Validation (if APIs involved)
5. **What is the blast radius?** -> Impact Analysis (if refactoring)
6. **What are the coupling risks?** -> Dependency Mapping (if dependencies)
7. **Are test ideas actionable?** -> Test Idea Rewriter (always)
8. **Is it sprint-ready?** -> READY/CONDITIONAL/NOT-READY decision
9. **What did we learn?** -> Memory persistence for future cycles
