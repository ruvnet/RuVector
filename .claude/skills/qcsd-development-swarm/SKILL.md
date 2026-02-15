---
name: qcsd-development-swarm
description: "QCSD Development phase swarm for in-sprint code quality assurance using TDD adherence, code complexity analysis, coverage gap detection, and defect prediction. Consumes Refinement outputs (BDD scenarios, SFDIPOT priorities) and produces signals for Verification."
category: qcsd-phases
priority: critical
version: 1.0.0
tokenEstimate: 3400
# DDD Domain Mapping (from QCSD-AGENTIC-QE-MAPPING-FRAMEWORK.md)
domains:
  primary:
    - domain: test-generation
      agents: [qe-tdd-specialist]
    - domain: coverage-analysis
      agents: [qe-coverage-specialist]
    - domain: code-intelligence
      agents: [qe-code-complexity]
  conditional:
    - domain: security-compliance
      agents: [qe-security-scanner]
    - domain: chaos-resilience
      agents: [qe-performance-tester]
    - domain: test-generation
      agents: [qe-mutation-tester]
    - domain: enterprise-integration
      agents: [qe-message-broker-tester, qe-sap-idoc-tester, qe-sod-analyzer]
  analysis:
    - domain: defect-intelligence
      agents: [qe-defect-predictor]
# Agent Inventory
agents:
  core: [qe-tdd-specialist, qe-code-complexity, qe-coverage-specialist]
  conditional: [qe-security-scanner, qe-performance-tester, qe-mutation-tester, qe-message-broker-tester, qe-sap-idoc-tester, qe-sod-analyzer]
  analysis: [qe-defect-predictor]
  total: 10
  sub_agents: 0
skills: [tdd-london-chicago, mutation-testing, performance-testing, security-testing]
# Execution Models (Task Tool is PRIMARY)
execution:
  primary: task-tool
  alternatives: [mcp-tools, cli]
swarm_pattern: true
parallel_batches: 3
last_updated: 2026-02-03
enforcement_level: strict
tags: [qcsd, development, tdd, complexity, coverage, security, performance, mutation, defect-prediction, swarm, parallel, ddd]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qcsd-development-swarm.yaml

---

# QCSD Development Swarm v1.0

Shift-left quality engineering swarm for in-sprint code quality assurance.

---

## Overview

The Development Swarm takes refined stories (that passed Refinement) and validates
code quality during sprint execution. Where the Ideation Swarm asks "Should we
build this?" and the Refinement Swarm asks "How should we test this?", the
Development Swarm asks "Is the code quality sufficient to ship?"

### QCSD Phase Positioning

| Phase | Swarm | Question | Decision | When |
|-------|-------|----------|----------|------|
| Ideation | qcsd-ideation-swarm | Should we build this? | GO / CONDITIONAL / NO-GO | PI/Sprint Planning |
| Refinement | qcsd-refinement-swarm | How should we test this? | READY / CONDITIONAL / NOT-READY | Sprint Refinement |
| **Development** | **qcsd-development-swarm** | **Is the code quality sufficient?** | **SHIP / CONDITIONAL / HOLD** | **During Sprint** |
| Verification | qcsd-cicd-swarm | Is this change safe to release? | RELEASE / REMEDIATE / BLOCK | Pre-Release / CI-CD |

### Key Differentiators from Refinement Swarm

| Dimension | Refinement Swarm | Development Swarm |
|-----------|------------------|-------------------|
| Framework | SFDIPOT (7 factors) | TDD + Complexity + Coverage |
| Agents | 10 (3 core + 6 conditional + 1 transformation) | 10 (3 core + 6 conditional + 1 analysis) |
| Core Output | BDD Gherkin scenarios | Code quality assessment |
| Decision | READY / CONDITIONAL / NOT-READY | SHIP / CONDITIONAL / HOLD |
| Flags | HAS_API, HAS_REFACTORING, HAS_DEPENDENCIES, HAS_SECURITY, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION | HAS_SECURITY_CODE, HAS_PERFORMANCE_CODE, HAS_CRITICAL_CODE, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION |
| Phase | Sprint Refinement | During Sprint Development |
| Input | User stories + acceptance criteria | Source code + test files |
| Final Step | Test idea rewriter transformation | Defect prediction analysis |

---

### Parameters

- `SOURCE_PATH`: Source code directory to analyze (required, e.g., `src/auth/`)
- `TEST_PATH`: Test directory for coverage analysis (optional, e.g., `tests/auth/`)
- `OUTPUT_FOLDER`: Where to save reports (default: `${PROJECT_ROOT}/Agentic QCSD/development/`)

---

## ENFORCEMENT RULES - READ FIRST

**These rules are NON-NEGOTIABLE. Violation means skill execution failure.**

| Rule | Enforcement |
|------|-------------|
| **E1** | You MUST spawn ALL THREE core agents (qe-tdd-specialist, qe-code-complexity, qe-coverage-specialist) in Phase 2. No exceptions. |
| **E2** | You MUST put all parallel Task calls in a SINGLE message. |
| **E3** | You MUST STOP and WAIT after each batch. No proceeding early. |
| **E4** | You MUST spawn conditional agents if flags are TRUE. No skipping. |
| **E5** | You MUST apply SHIP/CONDITIONAL/HOLD logic exactly as specified in Phase 5. |
| **E6** | You MUST generate the full report structure. No abbreviated versions. |
| **E7** | Each agent MUST read its reference files before analysis. |
| **E8** | You MUST apply qe-defect-predictor analysis on ALL code changes in Phase 8. Always. |
| **E9** | You MUST execute Phase 7 learning persistence. Store development findings to memory BEFORE Phase 8. No skipping. |

**PROHIBITED BEHAVIORS:**
- Summarizing instead of spawning agents
- Skipping agents "for brevity"
- Proceeding before background tasks complete
- Providing your own analysis instead of spawning specialists
- Omitting report sections
- Using placeholder text like "[details here]"
- Skipping the defect prediction analysis
- Skipping learning persistence (Phase 7) or treating it as optional
- Generating code analysis yourself instead of using specialist agents

---

## PHASE 1: Analyze Code Context (Flag Detection)

**MANDATORY: You must complete this analysis before Phase 2.**

Scan the source code, test files, and story context to SET these flags. Do not skip any flag.

### Flag Detection (Check ALL SIX)

```
HAS_SECURITY_CODE = FALSE
  Set TRUE if code contains ANY of: auth, authentication, authorization,
  password, credential, token, JWT, OAuth, session, encrypt, decrypt,
  hash, salt, PII, GDPR, HIPAA, secret, private key, certificate,
  RBAC, ACL, sanitize, XSS, CSRF, injection, eval(

HAS_PERFORMANCE_CODE = FALSE
  Set TRUE if code contains ANY of: loop, while, for, recursion,
  query, SELECT, JOIN, aggregate, cache, memoize, concurrent,
  parallel, async, await, Promise.all, batch, stream, buffer,
  pagination, lazy load, debounce, throttle, worker, thread,
  O(n^2), O(n log n), large dataset, bulk operation

HAS_CRITICAL_CODE = FALSE
  Set TRUE if code contains ANY of: payment, billing, invoice,
  transaction, charge, refund, subscription, medical, health,
  patient, diagnosis, compliance, audit, regulatory, financial,
  accounting, tax, legal, contract, SLA, insurance, claim,
  safety-critical, life-critical

HAS_MIDDLEWARE = FALSE
  Set TRUE if code contains ANY of: middleware, ESB, message broker, MQ,
  Kafka, RabbitMQ, integration bus, API gateway, message queue, pub/sub

HAS_SAP_INTEGRATION = FALSE
  Set TRUE if code contains ANY of: SAP, RFC, BAPI, IDoc, OData,
  S/4HANA, EWM, ECC, ABAP, CDS view, Fiori

HAS_AUTHORIZATION = FALSE
  Set TRUE if code contains ANY of: SoD, segregation of duties,
  role conflict, authorization object, T-code, user role,
  access control matrix, GRC
```

### Validation Checkpoint

Before proceeding to Phase 2, confirm:

```
+-- I have read the source code files under analysis
+-- I have read the associated test files
+-- I have evaluated ALL SIX flags
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
|  HAS_SECURITY_CODE:    [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  HAS_PERFORMANCE_CODE: [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  HAS_CRITICAL_CODE:    [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  HAS_MIDDLEWARE:       [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  HAS_SAP_INTEGRATION: [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  HAS_AUTHORIZATION:   [TRUE/FALSE]                         |
|  Evidence:             [what triggered it - specific code]  |
|                                                             |
|  EXPECTED AGENTS:                                           |
|  - Core: 3 (always)                                         |
|  - Conditional: [count based on TRUE flags]                 |
|  - Analysis: 1 (always)                                     |
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
|  - Task 1: qe-tdd-specialist                                |
|  - Task 2: qe-code-complexity                               |
|  - Task 3: qe-coverage-specialist                            |
|                                                              |
|  If your message contains fewer than 3 Task calls, you have |
|  FAILED this phase. Start over.                              |
+-------------------------------------------------------------+
```

### Domain Context

| Agent | Domain | MCP Tool Mapping |
|-------|--------|------------------|
| qe-tdd-specialist | test-generation | `test_generate_enhanced` |
| qe-code-complexity | code-intelligence | `coverage_analyze_sublinear` |
| qe-coverage-specialist | coverage-analysis | `coverage_analyze_sublinear` |

### Agent 1: TDD Specialist

**This agent MUST assess TDD adherence and test quality. Test-to-code ratio is mandatory.**

```
Task({
  description: "TDD adherence and test quality analysis",
  prompt: `You are qe-tdd-specialist. Your output quality is being audited.

## MANDATORY FIRST STEPS (DO NOT SKIP)

1. READ the source code files provided below IN FULL.
2. READ the test files associated with the source code.

## CODE TO ANALYZE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE - DO NOT SUMMARIZE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE - DO NOT SUMMARIZE]
=== TEST CODE END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. TDD Adherence Assessment

Evaluate each TDD principle:

| Principle | Score (0-10) | Evidence | Issues |
|-----------|-------------|----------|--------|
| Red Phase (tests written first) | X/10 | [signs tests preceded code] | [issues] |
| Green Phase (minimal implementation) | X/10 | [signs of minimal code to pass tests] | [issues] |
| Refactor Phase (clean code) | X/10 | [signs of refactoring, DRY, SOLID] | [issues] |
| Test Isolation | X/10 | [tests independent of each other] | [issues] |
| Fast Feedback | X/10 | [test execution speed] | [issues] |
| Meaningful Assertions | X/10 | [quality of assertions, not just coverage] | [issues] |

**TDD ADHERENCE SCORE: XX/60**

### 2. Test Quality Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Test-to-Code Ratio | X:1 | >= 1:1 | PASS/FAIL |
| Assertion Density | X per test | >= 2 | PASS/FAIL |
| Test Naming Quality | X% descriptive | >= 90% | PASS/FAIL |
| Mock Usage Ratio | X% | <= 50% | PASS/FAIL |
| Test Isolation Score | X/10 | >= 7 | PASS/FAIL |
| Edge Case Coverage | X% | >= 60% | PASS/FAIL |

### 3. Test Patterns Detected

| Pattern | Count | Quality | Recommendation |
|---------|-------|---------|----------------|
| Arrange-Act-Assert | X | Good/Poor | ... |
| Given-When-Then | X | Good/Poor | ... |
| Test Doubles (mocks) | X | Good/Poor | ... |
| Integration Tests | X | Good/Poor | ... |
| Property-Based Tests | X | Good/Poor | ... |

### 4. Missing Test Categories

| Category | Status | Priority | Suggested Tests |
|----------|--------|----------|-----------------|
| Happy path | Covered/Missing | P0/P1/P2 | [specific tests] |
| Error handling | Covered/Missing | P0/P1/P2 | [specific tests] |
| Boundary values | Covered/Missing | P0/P1/P2 | [specific tests] |
| Null/undefined | Covered/Missing | P0/P1/P2 | [specific tests] |
| Concurrency | Covered/Missing/N-A | P0/P1/P2 | [specific tests] |

### 5. Recommendations

| Priority | Recommendation | Impact |
|----------|---------------|--------|
| P0 | [must fix before merge] | [what improves] |
| P1 | [should fix this sprint] | [what improves] |
| P2 | [nice to have] | [what improves] |

**MINIMUM: Assess all 6 TDD principles and identify at least 3 missing test categories.**

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/02-tdd-adherence.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I read all source code and test files?
+-- Did I score all 6 TDD principles?
+-- Did I calculate test-to-code ratio?
+-- Did I identify missing test categories?
+-- Did I provide specific recommendations?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-tdd-specialist",
  run_in_background: true
})
```

### Agent 2: Code Complexity Analyzer

**This agent MUST calculate cyclomatic and cognitive complexity. Hotspots are mandatory.**

```
Task({
  description: "Cyclomatic and cognitive complexity analysis",
  prompt: `You are qe-code-complexity. Your output quality is being audited.

## CODE TO ANALYZE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE - DO NOT SUMMARIZE]
=== SOURCE CODE END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Function-Level Complexity

For EACH function/method in the codebase:

| Function | File:Line | Cyclomatic | Cognitive | LOC | Risk |
|----------|-----------|------------|-----------|-----|------|
| functionName | file.ts:42 | X | X | X | High/Medium/Low |

**Thresholds:**
- Cyclomatic: <=10 (Good), 11-20 (Warning), >20 (Critical)
- Cognitive: <=15 (Good), 16-25 (Warning), >25 (Critical)
- LOC: <=50 (Good), 51-100 (Warning), >100 (Critical)

### 2. Hotspot Detection

Identify the top 5 complexity hotspots:

| Rank | Function | Cyclomatic | Cognitive | Change Frequency | Priority |
|------|----------|------------|-----------|------------------|----------|
| 1 | ... | X | X | High/Medium/Low | P0 |
| 2 | ... | X | X | High/Medium/Low | P0/P1 |
| ... | ... | X | X | ... | ... |

### 3. Code Smell Detection

| Smell | Location | Severity | Refactoring Suggestion |
|-------|----------|----------|----------------------|
| Long Method | file:line | Critical/Major/Minor | Extract method: [specific suggestion] |
| Deep Nesting | file:line | Critical/Major/Minor | Guard clauses, early returns |
| God Class | file:line | Critical/Major/Minor | Split into: [specific classes] |
| Feature Envy | file:line | Critical/Major/Minor | Move to: [target class] |
| Duplicate Code | file:line | Critical/Major/Minor | Extract: [common abstraction] |

**MINIMUM: Identify at least 3 code smells or explicitly state "No significant code smells found after thorough analysis".**

### 4. Maintainability Index

| Component | MI Score (0-100) | Rating | Notes |
|-----------|-----------------|--------|-------|
| Overall | X | A/B/C/D/F | ... |
| Module 1 | X | A/B/C/D/F | ... |
| Module 2 | X | A/B/C/D/F | ... |

**Rating Scale:**
- A (85-100): Highly maintainable
- B (65-84): Moderately maintainable
- C (40-64): Difficult to maintain
- D (20-39): Very difficult to maintain
- F (0-19): Unmaintainable

### 5. Complexity Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Avg Cyclomatic Complexity | X | <= 10 | PASS/WARN/FAIL |
| Max Cyclomatic Complexity | X | <= 20 | PASS/WARN/FAIL |
| Avg Cognitive Complexity | X | <= 15 | PASS/WARN/FAIL |
| Max Cognitive Complexity | X | <= 25 | PASS/WARN/FAIL |
| Functions > 50 LOC | X% | <= 10% | PASS/WARN/FAIL |
| Nesting Depth > 4 | X | 0 | PASS/WARN/FAIL |

**COMPLEXITY SCORE: X/100** (inverse: lower complexity = higher score)

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/03-code-complexity.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I analyze every function/method?
+-- Did I calculate both cyclomatic and cognitive complexity?
+-- Did I identify top 5 hotspots?
+-- Did I detect code smells?
+-- Did I calculate maintainability index?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-code-complexity",
  run_in_background: true
})
```

### Agent 3: Coverage Specialist

**This agent MUST detect coverage gaps using O(log n) sublinear analysis. Gap count is mandatory.**

```
Task({
  description: "O(log n) sublinear coverage gap detection",
  prompt: `You are qe-coverage-specialist. Your output quality is being audited.

## CODE TO ANALYZE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE - DO NOT SUMMARIZE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE - DO NOT SUMMARIZE]
=== TEST CODE END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Coverage Overview

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Line Coverage | X% | >= 80% | PASS/WARN/FAIL |
| Branch Coverage | X% | >= 70% | PASS/WARN/FAIL |
| Function Coverage | X% | >= 90% | PASS/WARN/FAIL |
| Statement Coverage | X% | >= 80% | PASS/WARN/FAIL |

### 2. Coverage Gap Analysis (O(log n) Sublinear)

For EACH detected gap:

| Gap ID | File:Line | Type | Risk Score | Description |
|--------|-----------|------|------------|-------------|
| G001 | file.ts:42-56 | Uncovered Branch | High/Medium/Low | [what is not tested] |
| G002 | file.ts:78 | Missing Error Path | High/Medium/Low | [what error is not handled] |
| G003 | ... | ... | ... | ... |

**Gap Types:**
- Uncovered Branch: if/else path not tested
- Missing Error Path: catch/error handler not tested
- Untested Function: entire function without tests
- Partial Coverage: function tested but branches missed
- Dead Code: unreachable code detected

### 3. Risk-Weighted Gap Prioritization

| Priority | Gap IDs | Risk Rationale | Suggested Test |
|----------|---------|---------------|----------------|
| P0 - Critical | G001, G003 | [why critical] | [specific test to write] |
| P1 - High | G002, G005 | [why high] | [specific test to write] |
| P2 - Medium | G004 | [why medium] | [specific test to write] |

### 4. Module-Level Coverage Heatmap

| Module | Line% | Branch% | Function% | Risk Level |
|--------|-------|---------|-----------|------------|
| Module 1 | X% | X% | X% | High/Medium/Low |
| Module 2 | X% | X% | X% | High/Medium/Low |
| ... | ... | ... | ... | ... |

### 5. Coverage Trend Assessment

Based on code structure analysis:

| Dimension | Assessment | Recommendation |
|-----------|-----------|----------------|
| Test-to-Code Alignment | Good/Partial/Poor | [specific action] |
| Edge Case Coverage | Good/Partial/Poor | [specific action] |
| Integration Coverage | Good/Partial/Poor | [specific action] |
| Error Path Coverage | Good/Partial/Poor | [specific action] |

**COVERAGE GAPS TOTAL: X**
**CRITICAL GAPS (P0): X**

**MINIMUM: Identify at least 3 coverage gaps or explicitly state "No coverage gaps found after thorough sublinear analysis".**

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/04-coverage-analysis.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I calculate all 4 coverage metrics?
+-- Did I detect coverage gaps with specific file:line references?
+-- Did I risk-weight and prioritize gaps?
+-- Did I provide specific test suggestions for each gap?
+-- Did I generate the module-level heatmap?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-coverage-specialist",
  run_in_background: true
})
```

### Post-Spawn Confirmation

After sending all three Task calls, you MUST tell the user:

```
I've launched 3 core agents in parallel:

  qe-tdd-specialist [Domain: test-generation]
   - Assessing TDD red-green-refactor adherence
   - Calculating test-to-code ratio and assertion density
   - Identifying missing test categories

  qe-code-complexity [Domain: code-intelligence]
   - Computing cyclomatic and cognitive complexity per function
   - Detecting hotspots and code smells
   - Calculating maintainability index

  qe-coverage-specialist [Domain: coverage-analysis]
   - Running O(log n) sublinear coverage gap detection
   - Risk-weighting and prioritizing gaps
   - Generating module-level coverage heatmap

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
From qe-tdd-specialist:
[ ] tddAdherence = __/60 TDD score
[ ] testToCodeRatio = __:1
[ ] assertionDensity = __ per test
[ ] missingCategories = __ test categories missing

From qe-code-complexity:
[ ] avgCyclomatic = __ average cyclomatic complexity
[ ] maxCyclomatic = __ maximum cyclomatic complexity
[ ] complexityScore = __/100 (inverse complexity)
[ ] hotspots = __ hotspots identified
[ ] codeSmells = __ code smells detected

From qe-coverage-specialist:
[ ] lineCoverage = __% line coverage
[ ] branchCoverage = __% branch coverage
[ ] coverageGaps = __ total gaps
[ ] criticalGaps = __ P0 gaps
```

### Metrics Summary Box

Output extracted metrics:

```
+-------------------------------------------------------------+
|                    BATCH 1 RESULTS SUMMARY                   |
+-------------------------------------------------------------+
|                                                              |
|  TDD Adherence:          __/60                               |
|  Test-to-Code Ratio:     __:1                                |
|  Assertion Density:      __ per test                         |
|  Missing Test Categories: __                                 |
|                                                              |
|  Avg Cyclomatic:         __                                  |
|  Max Cyclomatic:         __                                  |
|  Complexity Score:       __/100                               |
|  Hotspots:               __                                  |
|  Code Smells:            __                                  |
|                                                              |
|  Line Coverage:          __%                                 |
|  Branch Coverage:        __%                                 |
|  Coverage Gaps:          __ total                            |
|  Critical Gaps (P0):     __                                  |
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
|  HAS_SECURITY_CODE = TRUE    -> MUST spawn qe-security-scanner     |
|  HAS_PERFORMANCE_CODE = TRUE -> MUST spawn qe-performance-tester  |
|  HAS_CRITICAL_CODE = TRUE    -> MUST spawn qe-mutation-tester     |
|  HAS_MIDDLEWARE = TRUE        -> MUST spawn qe-message-broker-tester|
|  HAS_SAP_INTEGRATION = TRUE  -> MUST spawn qe-sap-idoc-tester     |
|  HAS_AUTHORIZATION = TRUE    -> MUST spawn qe-sod-analyzer         |
|                                                              |
|  Skipping a flagged agent is a FAILURE of this skill.        |
+-------------------------------------------------------------+
```

### Conditional Domain Mapping

| Flag | Agent | Domain | MCP Tool |
|------|-------|--------|----------|
| HAS_SECURITY_CODE | qe-security-scanner | security-compliance | `security_scan_comprehensive` |
| HAS_PERFORMANCE_CODE | qe-performance-tester | chaos-resilience | `performance_benchmark` |
| HAS_CRITICAL_CODE | qe-mutation-tester | test-generation | `test_generate_enhanced` |
| HAS_MIDDLEWARE | qe-message-broker-tester | enterprise-integration | `task_orchestrate` |
| HAS_SAP_INTEGRATION | qe-sap-idoc-tester | enterprise-integration | `task_orchestrate` |
| HAS_AUTHORIZATION | qe-sod-analyzer | enterprise-integration | `task_orchestrate` |

### Decision Tree

```
IF HAS_SECURITY_CODE == FALSE AND HAS_PERFORMANCE_CODE == FALSE AND HAS_CRITICAL_CODE == FALSE AND HAS_MIDDLEWARE == FALSE AND HAS_SAP_INTEGRATION == FALSE AND HAS_AUTHORIZATION == FALSE:
    -> Skip to Phase 5 (no conditional agents needed)
    -> State: "No conditional agents needed based on code analysis"

ELSE:
    -> Spawn ALL applicable agents in ONE message
    -> Count how many you're spawning: __
```

### IF HAS_SECURITY_CODE: Security Scanner (MANDATORY WHEN FLAGGED)

```
Task({
  description: "SAST security scanning of source code",
  prompt: `You are qe-security-scanner. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. OWASP Top 10 Assessment

For each applicable OWASP category:

| OWASP ID | Category | Applicable? | Findings | Severity |
|----------|----------|-------------|----------|----------|
| A01:2021 | Broken Access Control | Yes/No | [findings] | Critical/High/Medium/Low |
| A02:2021 | Cryptographic Failures | Yes/No | [findings] | Critical/High/Medium/Low |
| A03:2021 | Injection | Yes/No | [findings] | Critical/High/Medium/Low |
| A04:2021 | Insecure Design | Yes/No | [findings] | Critical/High/Medium/Low |
| A05:2021 | Security Misconfiguration | Yes/No | [findings] | Critical/High/Medium/Low |
| A06:2021 | Vulnerable Components | Yes/No | [findings] | Critical/High/Medium/Low |
| A07:2021 | Auth Failures | Yes/No | [findings] | Critical/High/Medium/Low |
| A08:2021 | Software Integrity Failures | Yes/No | [findings] | Critical/High/Medium/Low |
| A09:2021 | Logging Failures | Yes/No | [findings] | Critical/High/Medium/Low |
| A10:2021 | SSRF | Yes/No | [findings] | Critical/High/Medium/Low |

### 2. Vulnerability Inventory

| Vuln ID | File:Line | Type | Severity | CVSS | Remediation |
|---------|-----------|------|----------|------|-------------|
| V001 | ... | SQL Injection/XSS/etc | Critical/High/Medium/Low | X.X | [specific fix] |

### 3. Secrets Detection

| Finding | File:Line | Type | Status |
|---------|-----------|------|--------|
| ... | ... | API Key/Password/Token/Cert | Exposed/Safe |

### 4. Dependency Vulnerability Check

| Dependency | Version | Known CVEs | Severity | Fix Version |
|------------|---------|------------|----------|-------------|
| ... | X.Y.Z | CVE-XXXX-XXXXX | Critical/High | X.Y.Z+ |

### 5. Security Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Input validation | X/10 | ... |
| Authentication | X/10 | ... |
| Authorization | X/10 | ... |
| Data protection | X/10 | ... |
| Error handling | X/10 | ... |

**SECURITY SCORE: X/50**

**MINIMUM: Assess all 10 OWASP categories and identify at least 3 security-related findings.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/05-security-scan.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-security-scanner",
  run_in_background: true
})
```

### IF HAS_PERFORMANCE_CODE: Performance Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Performance profiling and bottleneck detection",
  prompt: `You are qe-performance-tester. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Algorithm Complexity Analysis

For each significant function:

| Function | File:Line | Time Complexity | Space Complexity | Concern |
|----------|-----------|-----------------|------------------|---------|
| ... | ... | O(n)/O(n^2)/etc | O(1)/O(n)/etc | [if problematic] |

### 2. Performance Bottleneck Detection

| Bottleneck | File:Line | Type | Impact | Recommendation |
|------------|-----------|------|--------|----------------|
| ... | ... | CPU/Memory/I-O/Network | High/Medium/Low | [specific fix] |

**Bottleneck Types:**
- CPU Bound: nested loops, complex calculations, regex backtracking
- Memory Bound: large allocations, memory leaks, unbounded caches
- I/O Bound: synchronous file ops, blocking network calls, N+1 queries
- Network Bound: excessive API calls, large payloads, no batching

### 3. Database Query Analysis (if applicable)

| Query | Location | Estimated Cost | Index Used? | Optimization |
|-------|----------|---------------|-------------|-------------|
| ... | file:line | High/Medium/Low | Yes/No | [specific fix] |

### 4. Resource Usage Estimation

| Resource | Current Pattern | Risk | Recommendation |
|----------|----------------|------|----------------|
| Memory allocation | [pattern] | High/Medium/Low | [action] |
| Connection pooling | [pattern] | High/Medium/Low | [action] |
| Cache strategy | [pattern] | High/Medium/Low | [action] |
| Concurrency model | [pattern] | High/Medium/Low | [action] |

### 5. Performance Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Algorithm efficiency | X/10 | ... |
| Memory management | X/10 | ... |
| I/O optimization | X/10 | ... |
| Concurrency safety | X/10 | ... |

**PERFORMANCE SCORE: X/40**

**MINIMUM: Analyze at least 5 functions for complexity or explain why fewer exist.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/06-performance-profile.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-performance-tester",
  run_in_background: true
})
```

### IF HAS_CRITICAL_CODE: Mutation Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Mutation testing for test suite effectiveness",
  prompt: `You are qe-mutation-tester. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE]
=== TEST CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Mutation Analysis

Apply these mutation operators to critical code paths:

| Operator | Mutations Applied | Killed | Survived | Kill Rate |
|----------|------------------|--------|----------|-----------|
| Arithmetic (+,-,*,/) | X | X | X | X% |
| Relational (<,>,==,!=) | X | X | X | X% |
| Logical (&&, or, !) | X | X | X | X% |
| Conditional (if/else) | X | X | X | X% |
| Return Value | X | X | X | X% |
| Null/Undefined | X | X | X | X% |
| **TOTAL** | **X** | **X** | **X** | **X%** |

### 2. Surviving Mutants (Test Suite Weaknesses)

For each surviving mutant:

| Mutant ID | File:Line | Original | Mutation | Why Survived | Missing Test |
|-----------|-----------|----------|----------|-------------|-------------|
| M001 | file:42 | `a > b` | `a >= b` | No boundary test | Test for a == b case |
| M002 | ... | ... | ... | ... | ... |

### 3. Critical Path Mutation Score

| Critical Path | Mutations | Killed | Score | Status |
|---------------|-----------|--------|-------|--------|
| Payment flow | X | X | X% | PASS/FAIL (>= 95%) |
| Auth flow | X | X | X% | PASS/FAIL (>= 95%) |
| Data validation | X | X | X% | PASS/FAIL (>= 90%) |
| Error handling | X | X | X% | PASS/FAIL (>= 85%) |

### 4. Test Suite Effectiveness

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Mutation Score | X% | >= 80% | PASS/FAIL |
| Critical Path Score | X% | >= 95% | PASS/FAIL |
| Equivalent Mutants | X | (informational) | - |
| Test Strength Index | X/10 | >= 7 | PASS/FAIL |

### 5. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical path mutation escapes] | [what risk] | [effort] |
| P1 | [high-value missing tests] | [what risk] | [effort] |

**MUTATION SCORE: X%**

**MINIMUM: Apply at least 20 mutations across at least 3 mutation operators.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/07-mutation-analysis.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-mutation-tester",
  run_in_background: true
})
```

### IF HAS_MIDDLEWARE: Message Broker Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Message broker and middleware testing for integration reliability",
  prompt: `You are qe-message-broker-tester. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE]
=== TEST CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Message Broker Inventory

Identify all middleware/message broker components in the source code:

| Component | Type | Protocol | Direction | Status |
|-----------|------|----------|-----------|--------|
| [name] | Queue/Topic/Exchange | AMQP/Kafka/JMS/MQ | Producer/Consumer/Both | Active/Passive |

### 2. Message Flow Analysis

For each message flow:

| Flow ID | Producer | Broker | Consumer | Payload Schema | Ordering | Idempotency |
|---------|----------|--------|----------|----------------|----------|-------------|
| MF-001 | [source] | [broker] | [target] | [schema ref] | Guaranteed/Best-effort | Yes/No |

### 3. Error Handling & Retry Assessment

| Pattern | Implemented | Correct | Issue |
|---------|-------------|---------|-------|
| Dead Letter Queue (DLQ) | Yes/No | Yes/No | [issue] |
| Retry with backoff | Yes/No | Yes/No | [issue] |
| Circuit breaker | Yes/No | Yes/No | [issue] |
| Poison message handling | Yes/No | Yes/No | [issue] |
| Duplicate detection | Yes/No | Yes/No | [issue] |

### 4. Pub/Sub Verification

| Topic/Exchange | Publishers | Subscribers | Fan-out | Filtering | Test Coverage |
|----------------|------------|-------------|---------|-----------|---------------|
| [name] | [count] | [count] | [type] | [rules] | [%] |

### 5. Message Contract Validation

| Contract | Schema Validation | Versioning | Backward Compatible | Breaking Changes |
|----------|-------------------|------------|---------------------|------------------|
| [name] | Yes/No | [strategy] | Yes/No | [list] |

### 6. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical middleware issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**MIDDLEWARE HEALTH SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/10-middleware-testing.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-message-broker-tester",
  run_in_background: true
})
```

### IF HAS_SAP_INTEGRATION: SAP IDoc Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "SAP IDoc processing and BAPI testing for data flow validation",
  prompt: `You are qe-sap-idoc-tester. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE]
=== TEST CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. SAP Integration Inventory

Identify all SAP integration points in the source code:

| Integration Point | Type | Direction | SAP System | Protocol | Status |
|-------------------|------|-----------|------------|----------|--------|
| [name] | RFC/BAPI/IDoc/OData/CDS | Inbound/Outbound | S/4HANA/ECC/EWM | [protocol] | Active/Passive |

### 2. IDoc Processing Analysis

For each IDoc type:

| IDoc Type | Message Type | Direction | Segments | Partner Profile | Error Handling | Test Coverage |
|-----------|-------------|-----------|----------|-----------------|----------------|---------------|
| [type] | [msg type] | In/Out | [count] | [profile] | [strategy] | [%] |

### 3. BAPI/RFC Call Assessment

| BAPI/RFC | Parameters | Commit Handling | Error Codes | Rollback | Idempotency |
|----------|------------|-----------------|-------------|----------|-------------|
| [name] | In:[n] Out:[n] | BAPI_TRANSACTION_COMMIT? | [handled codes] | Yes/No | Yes/No |

### 4. Data Flow Validation

| Flow | Source | Mapping | Target | Transformation | Validation Rules | Gaps |
|------|--------|---------|--------|----------------|------------------|------|
| [name] | [field map] | [logic] | [SAP field] | [rules] | [validation] | [gaps] |

### 5. SAP-Specific Test Coverage

| Test Type | Coverage | Critical Gaps |
|-----------|----------|---------------|
| IDoc parsing/generation | [%] | [gaps] |
| BAPI call/response | [%] | [gaps] |
| RFC error handling | [%] | [gaps] |
| Data mapping accuracy | [%] | [gaps] |
| Transaction integrity | [%] | [gaps] |

### 6. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical SAP integration issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**SAP INTEGRATION HEALTH SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/11-sap-idoc-testing.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-sap-idoc-tester",
  run_in_background: true
})
```

### IF HAS_AUTHORIZATION: SoD Analyzer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Segregation of duties and authorization conflict analysis",
  prompt: `You are qe-sod-analyzer. Your output quality is being audited.

## SOURCE CODE

=== SOURCE CODE START ===
[PASTE THE COMPLETE SOURCE CODE HERE]
=== SOURCE CODE END ===

=== TEST CODE START ===
[PASTE THE COMPLETE TEST CODE HERE]
=== TEST CODE END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Authorization Model Inventory

Identify all authorization constructs in the source code:

| Component | Type | Mechanism | Granularity | Status |
|-----------|------|-----------|-------------|--------|
| [name] | Role/Permission/Policy | RBAC/ABAC/ACL | Resource/Action/Field | Active/Passive |

### 2. Segregation of Duties Matrix

| Function A | Function B | Conflict Type | Risk Level | Mitigating Control | Test Coverage |
|------------|------------|---------------|------------|-------------------|---------------|
| [create order] | [approve order] | SoD violation | Critical/High/Med | [control] | [%] |

### 3. Role Conflict Detection

| Role | Permissions | Conflicts With | Conflict Type | Remediation |
|------|-------------|---------------|---------------|-------------|
| [role] | [permissions list] | [conflicting role] | [SoD/privilege escalation] | [action] |

### 4. Authorization Test Coverage

| Test Type | Coverage | Critical Gaps |
|-----------|----------|---------------|
| Positive access tests | [%] | [gaps] |
| Negative access tests | [%] | [gaps] |
| Cross-role access | [%] | [gaps] |
| Privilege escalation | [%] | [gaps] |
| SoD enforcement | [%] | [gaps] |

### 5. Access Control Code Quality

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Authorization checks per endpoint | [ratio] | >= 1.0 | PASS/FAIL |
| Hardcoded roles/permissions | [count] | 0 | PASS/FAIL |
| Missing access control | [count] | 0 | PASS/FAIL |
| Default-allow patterns | [count] | 0 | PASS/FAIL |

### 6. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical authorization issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**AUTHORIZATION HEALTH SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/12-sod-analysis.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-sod-analyzer",
  run_in_background: true
})
```

### Agent Count Validation

**Before proceeding, verify agent count:**

```
+-------------------------------------------------------------+
|                   AGENT COUNT VALIDATION                     |
+-------------------------------------------------------------+
|                                                              |
|  CORE AGENTS (ALWAYS 3):                                     |
|    [ ] qe-tdd-specialist - SPAWNED? [Y/N]                   |
|    [ ] qe-code-complexity - SPAWNED? [Y/N]                  |
|    [ ] qe-coverage-specialist - SPAWNED? [Y/N]              |
|                                                              |
|  CONDITIONAL AGENTS (based on flags):                        |
|    [ ] qe-security-scanner - SPAWNED? [Y/N] (HAS_SECURITY)  |
|    [ ] qe-performance-tester - SPAWNED? [Y/N] (HAS_PERF)    |
|    [ ] qe-mutation-tester - SPAWNED? [Y/N] (HAS_CRITICAL)   |
|    [ ] qe-message-broker-tester - SPAWNED? [Y/N] (HAS_MIDDLEWARE)  |
|    [ ] qe-sap-idoc-tester - SPAWNED? [Y/N] (HAS_SAP_INTEG)        |
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

[IF HAS_SECURITY_CODE]    qe-security-scanner [Domain: security-compliance]
                          - OWASP Top 10, SAST scanning, secrets detection
[IF HAS_PERFORMANCE_CODE] qe-performance-tester [Domain: chaos-resilience]
                          - Algorithm complexity, bottleneck detection, resource analysis
[IF HAS_CRITICAL_CODE]    qe-mutation-tester [Domain: test-generation]
                          - Mutation testing, surviving mutant analysis, test effectiveness
[IF HAS_MIDDLEWARE]       qe-message-broker-tester [Domain: enterprise-integration]
                          - Message broker testing, queue validation, pub/sub verification
[IF HAS_SAP_INTEGRATION] qe-sap-idoc-tester [Domain: enterprise-integration]
                          - IDoc processing, BAPI testing, SAP data flow validation
[IF HAS_AUTHORIZATION]    qe-sod-analyzer [Domain: enterprise-integration]
                          - Segregation of duties, role conflict, authorization analysis

  WAITING for conditional agents to complete...
```

---

## PHASE 5: Synthesize Results & Determine Recommendation

### ENFORCEMENT: EXACT DECISION LOGIC

**You MUST apply this logic EXACTLY. No interpretation.**

```
STEP 1: Derive composite metrics
-----------------------------------------------------------
testCoverage = (lineCoverage + branchCoverage) / 2
complexityScore = avgCyclomatic  (use the average, not max)
coverageGaps = criticalGaps  (P0 gaps only)
criticalDefects = (criticalVulnerabilities from security scan, if ran)
                  + (criticalGaps from coverage)
                  + (missingCategories where priority == P0)

STEP 2: Check HOLD conditions (ANY triggers HOLD)
-----------------------------------------------------------
IF testCoverage < 50          -> HOLD ("Test coverage critically low")
IF complexityScore > 30       -> HOLD ("Code complexity dangerously high")
IF coverageGaps > 5           -> HOLD ("Too many critical coverage gaps")
IF criticalDefects > 2        -> HOLD ("Too many critical defects")

STEP 3: Check SHIP conditions (ALL required for SHIP)
-----------------------------------------------------------
IF testCoverage >= 80
   AND complexityScore <= 15
   AND coverageGaps == 0
   AND criticalDefects == 0   -> SHIP

STEP 4: Default
-----------------------------------------------------------
ELSE                          -> CONDITIONAL
```

### Decision Recording

```
METRICS:
- testCoverage = __% (average of line + branch)
- complexityScore = __ (average cyclomatic)
- coverageGaps = __ (P0 critical gaps only)
- criticalDefects = __

HOLD CHECK:
- testCoverage < 50? __ (YES/NO)
- complexityScore > 30? __ (YES/NO)
- coverageGaps > 5? __ (YES/NO)
- criticalDefects > 2? __ (YES/NO)

SHIP CHECK (only if no HOLD triggered):
- testCoverage >= 80? __ (YES/NO)
- complexityScore <= 15? __ (YES/NO)
- coverageGaps == 0? __ (YES/NO)
- criticalDefects == 0? __ (YES/NO)

FINAL RECOMMENDATION: [SHIP / CONDITIONAL / HOLD]
REASON: ___
```

### Conditional Recommendations

If recommendation is CONDITIONAL, provide specific blockers:

| Blocker | Current Value | Required Value | Owner | Action |
|---------|--------------|----------------|-------|--------|
| ... | ... | ... | [who] | [what to do] |

If recommendation is HOLD, provide mandatory remediation:

| Remediation | Priority | Effort | Deadline |
|-------------|----------|--------|----------|
| ... | P0 | [scope] | [before merge] |

---

## PHASE 6: Generate Development Report

### ENFORCEMENT: COMPLETE REPORT STRUCTURE

**ALL sections below are MANDATORY. No abbreviations.**

```markdown
# QCSD Development Report: [Feature/Module Name]

**Generated**: [Date/Time]
**Recommendation**: [SHIP / CONDITIONAL / HOLD]
**Agents Executed**: [List all agents that ran]
**Parallel Batches**: [2 or 3 depending on conditional agents]

---

## Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Test Coverage | X% | >= 80% | PASS/WARN/FAIL |
| Avg Cyclomatic Complexity | X | <= 15 | PASS/WARN/FAIL |
| Coverage Gaps (P0) | X | 0 | PASS/WARN/FAIL |
| Critical Defects | X | 0 | PASS/WARN/FAIL |

**Recommendation Rationale**: [1-2 sentences explaining why SHIP/CONDITIONAL/HOLD]

---

## TDD Adherence Analysis

[EMBED or LINK the full report from qe-tdd-specialist]

### TDD Score Summary

| Principle | Score | Status |
|-----------|-------|--------|
[All 6 TDD principles from qe-tdd-specialist]

### Test Quality Metrics
[Key metrics from agent output]

---

## Code Complexity Analysis

[EMBED or LINK the full report from qe-code-complexity]

### Hotspot Summary

| Rank | Function | Cyclomatic | Cognitive | Risk |
|------|----------|------------|-----------|------|
[Top 5 hotspots from qe-code-complexity]

### Code Smells Detected
[List from agent output]

---

## Coverage Analysis

[EMBED or LINK the full report from qe-coverage-specialist]

### Coverage Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Line | X% | PASS/WARN/FAIL |
| Branch | X% | PASS/WARN/FAIL |
| Function | X% | PASS/WARN/FAIL |

### Coverage Gaps
[All gaps from qe-coverage-specialist]

---

## Conditional Analysis

[INCLUDE ONLY IF APPLICABLE - based on which conditional agents ran]

### Security Scan (IF HAS_SECURITY_CODE)
[Full output from qe-security-scanner]

### Performance Profile (IF HAS_PERFORMANCE_CODE)
[Full output from qe-performance-tester]

### Mutation Analysis (IF HAS_CRITICAL_CODE)
[Full output from qe-mutation-tester]

### Middleware Testing (IF HAS_MIDDLEWARE)
[Full output from qe-message-broker-tester]

### SAP IDoc Testing (IF HAS_SAP_INTEGRATION)
[Full output from qe-sap-idoc-tester]

### SoD Analysis (IF HAS_AUTHORIZATION)
[Full output from qe-sod-analyzer]

---

## Recommended Actions

### Before Merge (P0 - Blockers)
- [ ] [Action based on findings]

### Before Sprint End (P1 - Important)
- [ ] [Action based on findings]

### Tech Debt Backlog (P2 - Improvement)
- [ ] [Action based on findings]

---

## Appendix: Agent Outputs

[Link to or embed full outputs from each agent]

---

*Generated by QCSD Development Swarm v1.0*
*Execution Model: Task Tool Parallel Swarm*
```

Write the executive summary report to:
`${OUTPUT_FOLDER}/01-executive-summary.md`

### Report Validation Checklist

Before presenting report:

```
+-- Executive Summary table is complete with all 4 metrics
+-- Recommendation matches decision logic output
+-- TDD section includes all 6 principle scores
+-- Complexity section includes top 5 hotspots
+-- Coverage section includes all gap details
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
|  This is NOT optional. It runs on EVERY development scan.    |
|  It stores findings for cross-phase feedback loops,          |
|  historical code quality tracking, and pattern learning.     |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
|  DO NOT treat this as "nice to have".                        |
|  Enforcement Rule E9 applies.                                |
+-------------------------------------------------------------+
```

### Purpose

Store development findings for:
- Cross-phase feedback loops (Development -> Verification -> next Ideation)
- Historical code quality tracking across sprints
- Complexity trend analysis over time
- Pattern learning for defect prediction improvement

### Auto-Execution Steps (ALL THREE are MANDATORY)

**Step 1: Store development findings to memory**

You MUST execute this MCP call with actual values from the development analysis:

```javascript
mcp__agentic-qe__memory_store({
  key: `qcsd-development-${featureId}-${Date.now()}`,
  namespace: "qcsd-development",
  value: {
    featureId: featureId,
    featureName: featureName,
    recommendation: recommendation,  // SHIP, CONDITIONAL, HOLD
    metrics: {
      testCoverage: testCoverage,
      avgCyclomatic: avgCyclomatic,
      complexityScore: complexityScore,
      coverageGaps: coverageGaps,
      criticalDefects: criticalDefects,
      tddAdherence: tddAdherence,
      mutationScore: mutationScore  // if applicable
    },
    flags: {
      HAS_SECURITY_CODE: HAS_SECURITY_CODE,
      HAS_PERFORMANCE_CODE: HAS_PERFORMANCE_CODE,
      HAS_CRITICAL_CODE: HAS_CRITICAL_CODE,
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
  sourceAgentId: "qcsd-development-swarm",
  targetAgentIds: ["qe-learning-coordinator", "qe-pattern-learner"],
  knowledgeDomain: "development-patterns"
})
```

**Step 3: Save learning persistence record to output folder**

You MUST use the Write tool to save a JSON record of the persisted learnings:

```
Save to: ${OUTPUT_FOLDER}/09-learning-persistence.json

Contents:
{
  "phase": "QCSD-Development",
  "featureId": "[feature ID]",
  "featureName": "[feature name]",
  "recommendation": "[SHIP/CONDITIONAL/HOLD]",
  "memoryKey": "qcsd-development-[featureId]-[timestamp]",
  "namespace": "qcsd-development",
  "metrics": {
    "testCoverage": [0-100],
    "avgCyclomatic": [N],
    "complexityScore": [0-100],
    "coverageGaps": [N],
    "criticalDefects": [N],
    "tddAdherence": [0-60],
    "mutationScore": [0-100 or null]
  },
  "flags": {
    "HAS_SECURITY_CODE": true/false,
    "HAS_PERFORMANCE_CODE": true/false,
    "HAS_CRITICAL_CODE": true/false,
    "HAS_MIDDLEWARE": true/false,
    "HAS_SAP_INTEGRATION": true/false,
    "HAS_AUTHORIZATION": true/false
  },
  "agentsInvoked": ["list", "of", "agents"],
  "crossPhaseSignals": {
    "toVerification": "Quality metrics as verification baseline",
    "toIdeation": "Complexity patterns for future risk assessment"
  },
  "persistedAt": "[ISO timestamp]"
}
```

### Fallback: CLI Memory Commands

If MCP memory_store tool is unavailable, use CLI instead (STILL MANDATORY):

```bash
npx @claude-flow/cli@latest memory store \
  --key "qcsd-development-${FEATURE_ID}-$(date +%s)" \
  --value '{"recommendation":"[VALUE]","testCoverage":[N],"avgCyclomatic":[N],"coverageGaps":[N]}' \
  --namespace qcsd-development

npx @claude-flow/cli@latest hooks post-task \
  --task-id "qcsd-development-${FEATURE_ID}" \
  --success true
```

### Validation Before Proceeding to Phase 8

```
+-- Did I execute mcp__agentic-qe__memory_store with actual values? (not placeholders)
+-- Did I execute mcp__agentic-qe__memory_share to propagate learnings?
+-- Did I save 09-learning-persistence.json to the output folder?
+-- Does the JSON contain the correct recommendation from Phase 5?
+-- Does the JSON contain actual metrics from Phases 2-4?
+-- Does the JSON contain actual flag values from Phase 1?
```

**If ANY validation check fails, DO NOT proceed to Phase 8.**

### Cross-Phase Signal Consumption

The Development Swarm both consumes and produces signals for other QCSD phases:

```
CONSUMES (from other phases):
+-- Loop 2 (Tactical): BDD scenarios from Refinement as test specs
|   - Gherkin scenarios become verification checklists
|   - SFDIPOT priorities guide test depth allocation
|
+-- Loop 4 (Quality-Criteria): INVEST gaps from Refinement
    - Untestable acceptance criteria flagged during development
    - DoR failures influence what to test more thoroughly

PRODUCES (for other phases):
+-- To Verification Phase: Quality metrics as verification baseline
|   - Coverage targets and complexity thresholds
|   - Known gaps that need verification attention
|
+-- To next Ideation Cycle: Complexity patterns for future risk assessment
    - Which SFDIPOT factors had highest code complexity
    - Defect patterns that should weight future HTSM analysis
```

---

## PHASE 8: Apply Defect Predictor (Analysis)

### ENFORCEMENT: ALWAYS RUN THIS PHASE

```
+-------------------------------------------------------------+
|  THE DEFECT PREDICTOR MUST ALWAYS RUN                        |
|                                                              |
|  This is NOT conditional. It runs on EVERY development scan. |
|  It uses code metrics, change history, and complexity data   |
|  to predict defect likelihood per module.                    |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
+-------------------------------------------------------------+
```

### Agent Spawn

```
Task({
  description: "ML-powered defect prediction analysis",
  prompt: `You are qe-defect-predictor. Your output quality is being audited.

## PURPOSE

Predict defect likelihood for each code module using metrics from the
development analysis. This is the final quality signal before the
SHIP/CONDITIONAL/HOLD recommendation is delivered.

## INPUT: CODE METRICS FROM PREVIOUS AGENTS

### From TDD Specialist (02-tdd-adherence.md):
[Summarize: TDD score, test-to-code ratio, missing categories]

### From Code Complexity (03-code-complexity.md):
[Summarize: cyclomatic/cognitive scores, hotspots, code smells]

### From Coverage Specialist (04-coverage-analysis.md):
[Summarize: coverage percentages, gaps, risk-weighted priorities]

### From Conditional Agents (if applicable):
[Summarize: security findings, performance bottlenecks, mutation scores]

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Defect Prediction per Module

| Module | Defect Probability | Confidence | Risk Factors | Evidence |
|--------|-------------------|------------|-------------|----------|
| module1.ts | X% (High/Medium/Low) | X% | [what contributes] | [metrics used] |
| module2.ts | X% (High/Medium/Low) | X% | [what contributes] | [metrics used] |

**Prediction Model Inputs:**
- Cyclomatic complexity > 15 = +20% defect probability
- Coverage < 60% = +25% defect probability
- No tests = +40% defect probability
- High change frequency + low coverage = +30% defect probability
- Deep nesting > 4 = +15% defect probability
- Code smells present = +10% per smell
- Security vulnerability present = +35% defect probability

### 2. Defect Hotspot Map

| Rank | File:Line | Defect Probability | Contributing Factors | Suggested Action |
|------|-----------|-------------------|---------------------|-----------------|
| 1 | ... | X% | [factors] | [specific remediation] |
| 2 | ... | X% | [factors] | [specific remediation] |
| 3 | ... | X% | [factors] | [specific remediation] |

### 3. Historical Pattern Matching

| Pattern | Detected? | Historical Defect Rate | Recommendation |
|---------|-----------|----------------------|----------------|
| High complexity + low coverage | Yes/No | X% defect rate | [action] |
| Untested error paths | Yes/No | X% defect rate | [action] |
| Missing boundary tests | Yes/No | X% defect rate | [action] |
| Security-sensitive without SAST | Yes/No | X% defect rate | [action] |
| New code without integration tests | Yes/No | X% defect rate | [action] |

### 4. Prediction Summary

| Metric | Value |
|--------|-------|
| Total modules analyzed | X |
| High-risk modules (>50% probability) | X |
| Medium-risk modules (25-50%) | X |
| Low-risk modules (<25%) | X |
| Average defect probability | X% |
| Prediction confidence | X% |

**DEFECT PREDICTION SCORE: X/100** (inverse: lower probability = higher score)

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/08-defect-prediction.md
Use the Write tool to save BEFORE completing.

## VALIDATION BEFORE SUBMITTING

+-- Did I analyze ALL modules from the codebase?
+-- Did I use metrics from ALL previous agent outputs?
+-- Did I rank defect hotspots by probability?
+-- Did I check historical patterns?
+-- Did I provide specific remediation actions?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-defect-predictor",
  run_in_background: true
})
```

### Wait for Analysis Completion

```
+-------------------------------------------------------------+
|  WAIT for qe-defect-predictor to complete before             |
|  proceeding to Phase 9.                                      |
|                                                              |
|  The defect prediction is the FINAL quality signal of        |
|  the Development Swarm - it synthesizes all metrics into     |
|  actionable defect risk predictions.                         |
+-------------------------------------------------------------+
```

---

## PHASE 9: Final Output

**At the very end of swarm execution, ALWAYS output this completion summary:**

```
+---------------------------------------------------------------------+
|                  QCSD DEVELOPMENT SWARM COMPLETE                      |
+---------------------------------------------------------------------+
|                                                                      |
|  Code Analyzed: [Feature/Module Name]                                 |
|  Reports Generated: [count]                                           |
|  Output Folder: ${OUTPUT_FOLDER}                                     |
|                                                                      |
|  DEVELOPMENT SCORES:                                                  |
|  +-- Test Coverage:         __%                                       |
|  +-- TDD Adherence:         __/60                                     |
|  +-- Complexity Score:      __/100                                    |
|  +-- Coverage Gaps (P0):    __                                        |
|  +-- Defect Probability:    __%                                       |
|  [IF HAS_SECURITY_CODE]                                               |
|  +-- Security Score:        __/50                                     |
|  [IF HAS_PERFORMANCE_CODE]                                            |
|  +-- Performance Score:     __/40                                     |
|  [IF HAS_CRITICAL_CODE]                                               |
|  +-- Mutation Score:        __%                                       |
|  [IF HAS_MIDDLEWARE]                                                  |
|  +-- Middleware Health:     __/50                                     |
|  [IF HAS_SAP_INTEGRATION]                                            |
|  +-- SAP Integration:      __/50                                     |
|  [IF HAS_AUTHORIZATION]                                              |
|  +-- Authorization Health:  __/50                                     |
|                                                                      |
|  RECOMMENDATION: [SHIP / CONDITIONAL / HOLD]                          |
|  REASON: [1-2 sentence rationale]                                     |
|                                                                      |
|  DELIVERABLES:                                                        |
|  +-- 01-executive-summary.md                                          |
|  +-- 02-tdd-adherence.md                                              |
|  +-- 03-code-complexity.md                                            |
|  +-- 04-coverage-analysis.md                                          |
|  [IF HAS_SECURITY_CODE]                                               |
|  +-- 05-security-scan.md                                              |
|  [IF HAS_PERFORMANCE_CODE]                                            |
|  +-- 06-performance-profile.md                                        |
|  [IF HAS_CRITICAL_CODE]                                               |
|  +-- 07-mutation-analysis.md                                          |
|  [IF HAS_MIDDLEWARE]                                                  |
|  +-- 10-middleware-testing.md                                         |
|  [IF HAS_SAP_INTEGRATION]                                            |
|  +-- 11-sap-idoc-testing.md                                          |
|  [IF HAS_AUTHORIZATION]                                              |
|  +-- 12-sod-analysis.md                                              |
|  +-- 08-defect-prediction.md                                          |
|  +-- 09-learning-persistence.json                                     |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is HOLD, ALSO output this prominent action box:**

```
+---------------------------------------------------------------------+
|  ACTION REQUIRED: CODE IS NOT READY TO SHIP                           |
+---------------------------------------------------------------------+
|                                                                      |
|  The following blockers MUST be resolved before merge:                |
|                                                                      |
|  1. [Blocker 1 with specific remediation]                             |
|  2. [Blocker 2 with specific remediation]                             |
|  3. [Blocker 3 with specific remediation]                             |
|                                                                      |
|  NEXT STEPS:                                                          |
|  - Address all P0 blockers listed above                               |
|  - Re-run /qcsd-development-swarm after fixes                        |
|  - Target: coverage >= 80%, complexity <= 15, 0 critical gaps         |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is CONDITIONAL, output this guidance box:**

```
+---------------------------------------------------------------------+
|  CONDITIONAL: CODE NEEDS MINOR IMPROVEMENTS BEFORE MERGE              |
+---------------------------------------------------------------------+
|                                                                      |
|  The code can be merged WITH these conditions:                        |
|                                                                      |
|  1. [Condition 1 - must be addressed before merge]                    |
|  2. [Condition 2 - must be addressed in follow-up PR]                 |
|                                                                      |
|  RISK ACCEPTANCE:                                                     |
|  - Tech lead acknowledges remaining quality gaps                      |
|  - Follow-up issues created for deferred improvements                 |
|                                                                      |
+---------------------------------------------------------------------+
```

**DO NOT end the swarm without displaying the completion summary.**

---

## Report Filename Mapping

| Agent | Report Filename | Phase |
|-------|----------------|-------|
| qe-tdd-specialist | `02-tdd-adherence.md` | Batch 1 |
| qe-code-complexity | `03-code-complexity.md` | Batch 1 |
| qe-coverage-specialist | `04-coverage-analysis.md` | Batch 1 |
| qe-security-scanner | `05-security-scan.md` | Batch 2 (conditional) |
| qe-performance-tester | `06-performance-profile.md` | Batch 2 (conditional) |
| qe-mutation-tester | `07-mutation-analysis.md` | Batch 2 (conditional) |
| qe-message-broker-tester | `10-middleware-testing.md` | Batch 2 (conditional) |
| qe-sap-idoc-tester | `11-sap-idoc-testing.md` | Batch 2 (conditional) |
| qe-sod-analyzer | `12-sod-analysis.md` | Batch 2 (conditional) |
| qe-defect-predictor | `08-defect-prediction.md` | Batch 3 (analysis) |
| Learning Persistence | `09-learning-persistence.json` | Phase 7 (auto-execute) |
| Synthesis | `01-executive-summary.md` | Phase 6 |

---

## DDD Domain Integration

This swarm operates across **3 primary domains**, **4 conditional domains**,
and **1 analysis domain**:

```
+-----------------------------------------------------------------------------+
|                    QCSD DEVELOPMENT - DOMAIN MAP                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|  PRIMARY DOMAINS (Always Active)                                             |
|  +-------------------------------+  +-------------------------------+       |
|  |       test-generation         |  |      coverage-analysis        |       |
|  |  ---------------------------  |  |  ---------------------------  |       |
|  |  - qe-tdd-specialist          |  |  - qe-coverage-specialist    |       |
|  |    (TDD adherence, test       |  |    (O(log n) gap detection,  |       |
|  |     quality metrics)          |  |     risk-weighted coverage)  |       |
|  +-------------------------------+  +-------------------------------+       |
|                                                                              |
|  +-------------------------------+                                          |
|  |      code-intelligence        |                                          |
|  |  ---------------------------  |                                          |
|  |  - qe-code-complexity         |                                          |
|  |    (cyclomatic/cognitive,     |                                          |
|  |     hotspots, code smells)    |                                          |
|  +-------------------------------+                                          |
|                                                                              |
|  CONDITIONAL DOMAINS (Based on Code Content)                                 |
|  +-----------------------+  +-----------------------+  +------------------+ |
|  | security-compliance   |  |  chaos-resilience     |  | test-generation  | |
|  | ───────────────────── |  |  ──────────────────── |  | ──────────────── | |
|  | qe-security-scanner   |  |  qe-performance-      |  | qe-mutation-     | |
|  | [IF HAS_SECURITY_CODE]|  |  tester               |  | tester           | |
|  |                       |  |  [IF HAS_PERF_CODE]   |  | [IF HAS_CRITICAL]| |
|  +-----------------------+  +-----------------------+  +------------------+ |
|                                                                              |
|  +-----------------------------------------------------------------------+  |
|  |                  enterprise-integration                                |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-message-broker-tester [IF HAS_MIDDLEWARE]                      |  |
|  |  - qe-sap-idoc-tester [IF HAS_SAP_INTEGRATION]                      |  |
|  |  - qe-sod-analyzer [IF HAS_AUTHORIZATION]                           |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  ANALYSIS DOMAIN (Always Active)                                             |
|  +-----------------------------------------------------------------------+  |
|  |                    defect-intelligence                                 |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-defect-predictor (ML-powered defect probability, hotspot map)  |  |
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
Just follow the skill phases above - uses Task() calls with run_in_background: true
```

**Option B: MCP Tools**
```javascript
// Initialize fleet for Development domains
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["test-generation", "coverage-analysis", "code-intelligence", "security-compliance", "chaos-resilience", "defect-intelligence"],
  maxAgents: 7
})

// Orchestrate development task
mcp__agentic-qe__task_orchestrate({
  task: "qcsd-development-analysis",
  strategy: "parallel"
})
```

**Option C: CLI**
```bash
# Initialize coordination
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 7

# Route task
npx @claude-flow/cli@latest hooks pre-task --description "QCSD Development for [Feature]"

# Execute agents
npx @claude-flow/cli@latest agent spawn --type qe-tdd-specialist
npx @claude-flow/cli@latest agent spawn --type qe-code-complexity
npx @claude-flow/cli@latest agent spawn --type qe-coverage-specialist
```

---

## Quick Reference

### Enforcement Summary

| Phase | Must Do | Failure Condition |
|-------|---------|-------------------|
| 1 | Check ALL 6 flags | Missing flag evaluation |
| 2 | Spawn ALL 3 core agents in ONE message | Fewer than 3 Task calls |
| 3 | WAIT for completion | Proceeding before results |
| 4 | Spawn ALL flagged conditional agents | Skipping a TRUE flag |
| 5 | Apply EXACT decision logic | Wrong recommendation |
| 6 | Generate COMPLETE report | Missing sections |
| 7 | ALWAYS store learnings + save 09-learning-persistence.json | Pattern loss, missing audit trail |
| 8 | ALWAYS run defect predictor | Skipping analysis |
| 9 | Output completion summary | Missing final output |

### Quality Gate Thresholds

| Metric | SHIP | CONDITIONAL | HOLD |
|--------|------|-------------|------|
| Test Coverage | >= 80% | 50-79% | < 50% |
| Avg Cyclomatic | <= 15 | 16-30 | > 30 |
| Coverage Gaps (P0) | 0 | 1-5 | > 5 |
| Critical Defects | 0 | 1-2 | > 2 |

### Domain-to-Agent Mapping

| Domain | Agent | Phase | Batch |
|--------|-------|-------|-------|
| test-generation | qe-tdd-specialist | Core | 1 |
| code-intelligence | qe-code-complexity | Core | 1 |
| coverage-analysis | qe-coverage-specialist | Core | 1 |
| security-compliance | qe-security-scanner | Conditional (HAS_SECURITY_CODE) | 2 |
| chaos-resilience | qe-performance-tester | Conditional (HAS_PERFORMANCE_CODE) | 2 |
| test-generation | qe-mutation-tester | Conditional (HAS_CRITICAL_CODE) | 2 |
| enterprise-integration | qe-message-broker-tester | Conditional (HAS_MIDDLEWARE) | 2 |
| enterprise-integration | qe-sap-idoc-tester | Conditional (HAS_SAP_INTEGRATION) | 2 |
| enterprise-integration | qe-sod-analyzer | Conditional (HAS_AUTHORIZATION) | 2 |
| defect-intelligence | qe-defect-predictor | Analysis (ALWAYS) | 3 |

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
  enabledDomains: ["test-generation", "coverage-analysis", "code-intelligence", "security-compliance", "chaos-resilience", "defect-intelligence"],
  maxAgents: 7
})

// Task submission
mcp__agentic-qe__task_submit({ type: "...", priority: "p0", payload: {...} })
mcp__agentic-qe__task_orchestrate({ task: "...", strategy: "parallel" })

// Status
mcp__agentic-qe__fleet_status({ verbose: true })
mcp__agentic-qe__task_list({ status: "pending" })

// Memory
mcp__agentic-qe__memory_store({ key: "...", value: {...}, namespace: "qcsd-development" })
mcp__agentic-qe__memory_query({ pattern: "qcsd-development-*", namespace: "qcsd-development" })
mcp__agentic-qe__memory_share({
  sourceAgentId: "qcsd-development-swarm",
  targetAgentIds: ["qe-learning-coordinator"],
  knowledgeDomain: "development-patterns"
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
npx @claude-flow/cli@latest memory store --key "[key]" --value "[json]" --namespace qcsd-development
npx @claude-flow/cli@latest memory search --query "[query]" --namespace qcsd-development
npx @claude-flow/cli@latest memory list --namespace qcsd-development
```

---

## Swarm Topology

```
                 QCSD DEVELOPMENT SWARM v1.0
                          |
          BATCH 1 (Core - Parallel)
          +-----------+---+-----------+
          |           |               |
    +-----v-----+ +---v--------+ +---v-----------+
    |   TDD     | |   Code     | |  Coverage     |
    | Specialist| | Complexity | |  Specialist   |
    | (R-G-R)   | | (Cyc/Cog)  | |  (O(log n))   |
    |-----------| |------------| |---------------|
    | test-gen  | | code-intel | | coverage-anly |
    +-----+-----+ +-----+------+ +------+--------+
          |              |               |
          +--------------+---------------+
                         |
                  [METRICS GATE]
                         |
          BATCH 2 (Conditional - Parallel)
          +-----------+---+-----------+
          |           |               |
    +-----v-----+ +---v--------+ +---v----------+
    | Security  | | Performance| | Mutation     |
    | Scanner   | | Tester     | | Tester       |
    | [IF SEC]  | | [IF PERF]  | | [IF CRIT]    |
    |-----------| |------------| |--------------|
    | sec-compl | | chaos-res  | | test-gen     |
    +-----------+ +------------+ +--------------+
          +-------------+---+-------------+
          |             |                 |
    +-----v------+ +---v--------+ +------v-------+
    | Msg Broker | | SAP IDoc   | | SoD          |
    | Tester     | | Tester     | | Analyzer     |
    | [IF MIDW]  | | [IF SAP]   | | [IF AUTH]    |
    |------------| |------------| |--------------|
    | ent-integ  | | ent-integ  | | ent-integ    |
    +------------+ +------------+ +--------------+
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
          BATCH 3 (Analysis - Always)
                         |
                 +-------v-------+
                 | Defect        |
                 | Predictor     |
                 | (ALWAYS RUNS) |
                 |---------------|
                 | defect-intel  |
                 +-------+-------+
                         |
                [FINAL REPORT]
```

---

## Inventory Summary

| Resource Type | Count | Primary | Conditional | Analysis |
|---------------|:-----:|:-------:|:-----------:|:--------:|
| **Agents** | 10 | 3 | 6 | 1 |
| **Sub-agents** | 0 | - | - | - |
| **Skills** | 4 | 4 | - | - |
| **Domains** | 8 | 3 | 4 | 1 |
| **Parallel Batches** | 3 | 1 | 1 | 1 |

**Skills Used:**
1. `tdd-london-chicago` - TDD methodology guidance
2. `mutation-testing` - Mutation testing patterns
3. `performance-testing` - Performance analysis framework
4. `security-testing` - OWASP scanning patterns

**Frameworks Applied:**
1. TDD Red-Green-Refactor - Test-first development assessment
2. Cyclomatic/Cognitive Complexity - Code complexity measurement
3. Sublinear Coverage Analysis - O(log n) gap detection
4. OWASP Top 10 - Security vulnerability assessment
5. Mutation Testing - Test suite effectiveness validation
6. ML Defect Prediction - Pattern-based defect forecasting

---

## Key Principle

**Code ships when quality is proven, not when deadlines arrive.**

This swarm provides:
1. **Is TDD practiced?** -> TDD Adherence Assessment (6 principles)
2. **Is code maintainable?** -> Complexity Analysis (cyclomatic + cognitive)
3. **Are there test gaps?** -> Coverage Gap Detection (O(log n) sublinear)
4. **Is it secure?** -> SAST Security Scan (if security code present)
5. **Is it performant?** -> Performance Profiling (if performance code present)
6. **Are tests effective?** -> Mutation Analysis (if critical code present)
7. **Where will bugs appear?** -> Defect Prediction (always)
8. **Should we ship?** -> SHIP/CONDITIONAL/HOLD decision
9. **What did we learn?** -> Memory persistence for future cycles
