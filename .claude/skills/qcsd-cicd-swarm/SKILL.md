---
name: qcsd-cicd-swarm
description: "QCSD Verification phase swarm for CI/CD pipeline quality gates using regression analysis, flaky test detection, quality gate enforcement, and deployment readiness assessment. Consumes Development outputs (SHIP/CONDITIONAL/HOLD decisions, quality metrics) and produces signals for Production monitoring."
category: qcsd-phases
priority: critical
version: 1.0.0
tokenEstimate: 3500
# DDD Domain Mapping (from QCSD-AGENTIC-QE-MAPPING-FRAMEWORK.md)
domains:
  primary:
    - domain: quality-assessment
      agents: [qe-quality-gate]
    - domain: test-execution
      agents: [qe-regression-analyzer]
    - domain: test-execution
      agents: [qe-flaky-hunter]
  conditional:
    - domain: security-compliance
      agents: [qe-security-scanner]
    - domain: chaos-resilience
      agents: [qe-chaos-engineer]
    - domain: coverage-analysis
      agents: [qe-coverage-specialist]
    - domain: enterprise-integration
      agents: [qe-middleware-validator, qe-soap-tester, qe-sod-analyzer]
  analysis:
    - domain: quality-assessment
      agents: [qe-deployment-advisor]
# Agent Inventory
agents:
  core: [qe-quality-gate, qe-regression-analyzer, qe-flaky-hunter]
  conditional: [qe-security-scanner, qe-chaos-engineer, qe-coverage-specialist, qe-middleware-validator, qe-soap-tester, qe-sod-analyzer]
  analysis: [qe-deployment-advisor]
  total: 10
  sub_agents: 0
skills: [shift-left-testing, shift-right-testing, regression-testing, security-testing]
# Execution Models (Task Tool is PRIMARY)
execution:
  primary: task-tool
  alternatives: [mcp-tools, cli]
swarm_pattern: true
parallel_batches: 3
last_updated: 2026-02-03
enforcement_level: strict
tags: [qcsd, verification, cicd, pipeline, quality-gate, regression, flaky, security, chaos, coverage, deployment, swarm, parallel, ddd]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qcsd-cicd-swarm.yaml

---

# QCSD CI/CD Swarm v1.0

Shift-left quality engineering swarm for CI/CD pipeline verification and release readiness.

---

## Overview

The CI/CD Swarm takes code that passed Development quality checks and validates it is
safe to release through the CI/CD pipeline. Where the Development Swarm asks "Is the
code quality sufficient to ship?", the CI/CD Swarm asks "Is this change safe to release?"

This swarm operates at the pipeline level, analyzing test results, regression risk,
flaky test impact, security pipeline status, and infrastructure changes to render
a RELEASE / REMEDIATE / BLOCK decision.

### QCSD Phase Positioning

| Phase | Swarm | Question | Decision | When |
|-------|-------|----------|----------|------|
| Ideation | qcsd-ideation-swarm | Should we build this? | GO / CONDITIONAL / NO-GO | PI/Sprint Planning |
| Refinement | qcsd-refinement-swarm | How should we test this? | READY / CONDITIONAL / NOT-READY | Sprint Refinement |
| Development | qcsd-development-swarm | Is the code quality sufficient? | SHIP / CONDITIONAL / HOLD | During Sprint |
| **Verification** | **qcsd-cicd-swarm** | **Is this change safe to release?** | **RELEASE / REMEDIATE / BLOCK** | **Pre-Release / CI-CD** |

### Key Differentiators from Development Swarm

| Dimension | Development Swarm | CI/CD Swarm |
|-----------|-------------------|-------------|
| Framework | TDD + Complexity + Coverage | Quality Gates + Regression + Stability |
| Agents | 10 (3 core + 6 conditional + 1 analysis) | 10 (3 core + 6 conditional + 1 analysis) |
| Core Output | Code quality assessment | Release readiness assessment |
| Decision | SHIP / CONDITIONAL / HOLD | RELEASE / REMEDIATE / BLOCK |
| Flags | HAS_SECURITY_CODE, HAS_PERFORMANCE_CODE, HAS_CRITICAL_CODE, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION | HAS_SECURITY_PIPELINE, HAS_PERFORMANCE_PIPELINE, HAS_INFRA_CHANGE, HAS_MIDDLEWARE, HAS_SAP_INTEGRATION, HAS_AUTHORIZATION |
| Phase | During Sprint Development | Pre-Release / CI-CD Pipeline |
| Input | Source code + test files | Pipeline artifacts + test results + build output |
| Final Step | Defect prediction analysis | Deployment readiness advisory |

---

### Parameters

- `PIPELINE_ARTIFACTS`: Path to CI/CD artifacts, test results, and build output (required, e.g., `ci/artifacts/`)
- `BASELINE_REF`: Git ref for baseline comparison (optional, default: `main`)
- `OUTPUT_FOLDER`: Where to save reports (default: `${PROJECT_ROOT}/Agentic QCSD/cicd/`)
- `DEPLOY_TARGET`: Target deployment environment (optional, e.g., `staging`, `production`)

---

## ENFORCEMENT RULES - READ FIRST

**These rules are NON-NEGOTIABLE. Violation means skill execution failure.**

| Rule | Enforcement |
|------|-------------|
| **E1** | You MUST spawn ALL THREE core agents (qe-quality-gate, qe-regression-analyzer, qe-flaky-hunter) in Phase 2. No exceptions. |
| **E2** | You MUST put all parallel Task calls in a SINGLE message. |
| **E3** | You MUST STOP and WAIT after each batch. No proceeding early. |
| **E4** | You MUST spawn conditional agents if flags are TRUE. No skipping. |
| **E5** | You MUST apply RELEASE/REMEDIATE/BLOCK logic exactly as specified in Phase 5. |
| **E6** | You MUST generate the full report structure. No abbreviated versions. |
| **E7** | Each agent MUST read its reference files before analysis. |
| **E8** | You MUST apply qe-deployment-advisor analysis on ALL pipeline data in Phase 8. Always. |
| **E9** | You MUST execute Phase 7 learning persistence. Store verification findings to memory BEFORE Phase 8. No skipping. |

**PROHIBITED BEHAVIORS:**
- Summarizing instead of spawning agents
- Skipping agents "for brevity"
- Proceeding before background tasks complete
- Providing your own analysis instead of spawning specialists
- Omitting report sections
- Using placeholder text like "[details here]"
- Skipping the deployment readiness analysis
- Skipping learning persistence (Phase 7) or treating it as optional
- Generating pipeline analysis yourself instead of using specialist agents

---

## PHASE 1: Analyze Pipeline Context (Flag Detection)

**MANDATORY: You must complete this analysis before Phase 2.**

Scan the pipeline artifacts, CI/CD configuration, test results, and change diff to SET these flags. Do not skip any flag.

### Flag Detection (Check ALL SIX)

```
HAS_SECURITY_PIPELINE = FALSE
  Set TRUE if pipeline contains ANY of: security scan results, SAST output,
  DAST output, dependency audit results, CVE reports, container scan,
  secrets detection output, compliance check results, SBOM generation,
  penetration test results, security gate failures, auth-related changes,
  certificate changes, encryption changes, API key rotation

HAS_PERFORMANCE_PIPELINE = FALSE
  Set TRUE if pipeline contains ANY of: load test results, performance
  benchmark output, latency metrics, throughput data, stress test results,
  memory profiling output, CPU profiling data, response time baselines,
  scalability test results, database query performance, cache hit ratios,
  CDN performance data, API response times, SLA compliance metrics

HAS_INFRA_CHANGE = FALSE
  Set TRUE if changes include ANY of: Dockerfile, docker-compose,
  kubernetes manifests, terraform files, CloudFormation templates,
  CI/CD pipeline config (.github/workflows, .gitlab-ci, Jenkinsfile),
  infrastructure as code, helm charts, ansible playbooks,
  environment variables, nginx config, database migrations,
  service mesh config, load balancer config, DNS changes

HAS_MIDDLEWARE = FALSE
  Set TRUE if pipeline artifacts or changes reference ANY of:
  middleware, ESB, message broker, MQ, Kafka, RabbitMQ,
  integration bus, API gateway, message queue, pub/sub,
  event bus, service bus, ActiveMQ, NATS, Redis Streams

HAS_SAP_INTEGRATION = FALSE
  Set TRUE if pipeline artifacts or changes reference ANY of:
  SAP, RFC, BAPI, IDoc, OData, S/4HANA, EWM, ECC, ABAP,
  CDS view, Fiori, SAP Cloud Integration, SAP PI/PO,
  SAP Gateway, SOAP service, SAP connector

HAS_AUTHORIZATION = FALSE
  Set TRUE if pipeline artifacts or changes reference ANY of:
  SoD, segregation of duties, role conflict, authorization object,
  T-code, user role, access control matrix, GRC, RBAC policy,
  permission matrix, privilege escalation, role assignment
```

### Validation Checkpoint

Before proceeding to Phase 2, confirm:

```
+-- I have read the pipeline artifacts and test results
+-- I have read the CI/CD configuration files
+-- I have reviewed the change diff against baseline
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
|  HAS_SECURITY_PIPELINE:    [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
|                                                             |
|  HAS_PERFORMANCE_PIPELINE: [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
|                                                             |
|  HAS_INFRA_CHANGE:         [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
|                                                             |
|  HAS_MIDDLEWARE:            [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
|                                                             |
|  HAS_SAP_INTEGRATION:      [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
|                                                             |
|  HAS_AUTHORIZATION:        [TRUE/FALSE]                     |
|  Evidence:                 [what triggered it - specific]   |
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
|  - Task 1: qe-quality-gate                                  |
|  - Task 2: qe-regression-analyzer                           |
|  - Task 3: qe-flaky-hunter                                  |
|                                                              |
|  If your message contains fewer than 3 Task calls, you have |
|  FAILED this phase. Start over.                              |
+-------------------------------------------------------------+
```

### Domain Context

| Agent | Domain | MCP Tool Mapping |
|-------|--------|------------------|
| qe-quality-gate | quality-assessment | `quality_assess` |
| qe-regression-analyzer | test-execution | `test_execute_parallel` |
| qe-flaky-hunter | test-execution | `test_execute_parallel` |

### Agent 1: Quality Gate Evaluator

**This agent MUST evaluate quality gate thresholds and enforce pass/fail criteria.**

```
Task({
  description: "Quality gate threshold evaluation",
  prompt: `You are qe-quality-gate. Your output quality is being audited.

## MANDATORY FIRST STEPS (DO NOT SKIP)

1. READ the pipeline test results and build artifacts provided below IN FULL.
2. READ the quality gate configuration if available.
3. READ any previous QCSD Development phase signals if available.

## PIPELINE DATA TO ANALYZE

=== TEST RESULTS START ===
[PASTE THE COMPLETE TEST RESULTS HERE - DO NOT SUMMARIZE]
=== TEST RESULTS END ===

=== BUILD ARTIFACTS START ===
[PASTE BUILD OUTPUT / COVERAGE REPORTS HERE - DO NOT SUMMARIZE]
=== BUILD ARTIFACTS END ===

=== DEVELOPMENT PHASE SIGNALS (if available) START ===
[PASTE any Development phase SHIP/CONDITIONAL/HOLD signals]
=== DEVELOPMENT PHASE SIGNALS END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Quality Gate Assessment

Evaluate each quality dimension against thresholds:

| Gate | Metric | Value | Threshold | Status |
|------|--------|-------|-----------|--------|
| Test Pass Rate | X/Y passed | X% | >= 100% | PASS/FAIL |
| Code Coverage | Line coverage | X% | >= 80% | PASS/WARN/FAIL |
| Branch Coverage | Branch coverage | X% | >= 70% | PASS/WARN/FAIL |
| Build Success | Build status | Pass/Fail | Pass | PASS/FAIL |
| Lint Errors | Error count | X | 0 | PASS/WARN/FAIL |
| Type Check | Type errors | X | 0 | PASS/FAIL |
| Bundle Size | Size delta | +X KB | <= +50 KB | PASS/WARN/FAIL |
| Test Duration | Total time | Xs | <= baseline + 10% | PASS/WARN/FAIL |

**QUALITY GATE STATUS: PASSED / FAILED (X/Y gates passed)**

### 2. Test Results Analysis

| Category | Total | Passed | Failed | Skipped | Pass Rate |
|----------|-------|--------|--------|---------|-----------|
| Unit Tests | X | X | X | X | X% |
| Integration Tests | X | X | X | X | X% |
| E2E Tests | X | X | X | X | X% |
| Contract Tests | X | X | X | X | X% |
| **Total** | **X** | **X** | **X** | **X** | **X%** |

### 3. Failed Test Analysis

For each failed test:

| Test Name | Suite | Failure Reason | Severity | Flaky? |
|-----------|-------|---------------|----------|--------|
| test_name | suite | [error message] | Critical/High/Medium | Yes/No |

### 4. Coverage Delta Analysis

| Module | Before | After | Delta | Status |
|--------|--------|-------|-------|--------|
| Module 1 | X% | X% | +/-X% | Improved/Declined/Stable |
| Module 2 | X% | X% | +/-X% | Improved/Declined/Stable |
| **Overall** | **X%** | **X%** | **+/-X%** | **Improved/Declined/Stable** |

### 5. Quality Gate Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Test completeness | X/10 | ... |
| Coverage adequacy | X/10 | ... |
| Build health | X/10 | ... |
| Pipeline stability | X/10 | ... |
| Threshold compliance | X/10 | ... |

**QUALITY GATE SCORE: X/50**

**MINIMUM: Evaluate all 8 quality gates and provide test results breakdown by category.**

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/02-quality-gate.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I read all test results and build artifacts?
+-- Did I evaluate all 8 quality gates?
+-- Did I analyze test results by category?
+-- Did I analyze coverage delta?
+-- Did I identify all failed tests?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-quality-gate",
  run_in_background: true
})
```

### Agent 2: Regression Analyzer

**This agent MUST analyze regression risk and test selection effectiveness.**

```
Task({
  description: "Regression risk analysis and test selection",
  prompt: `You are qe-regression-analyzer. Your output quality is being audited.

## PIPELINE DATA TO ANALYZE

=== CHANGE DIFF START ===
[PASTE THE COMPLETE DIFF/CHANGESET HERE - DO NOT SUMMARIZE]
=== CHANGE DIFF END ===

=== TEST RESULTS START ===
[PASTE THE COMPLETE TEST RESULTS HERE - DO NOT SUMMARIZE]
=== TEST RESULTS END ===

=== HISTORICAL TEST DATA (if available) START ===
[PASTE any historical test run data]
=== HISTORICAL TEST DATA END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Change Impact Analysis

| File Changed | Lines Changed | Modules Affected | Risk Score |
|-------------|---------------|-------------------|------------|
| file.ts | +X / -Y | [dependent modules] | High/Medium/Low |

**Impact Radius:**
- Direct changes: X files
- Directly affected modules: X
- Transitively affected modules: X
- Total blast radius: X files

### 2. Regression Risk Assessment

| Risk Factor | Score (0-10) | Evidence | Mitigation |
|-------------|-------------|----------|------------|
| Code churn | X/10 | [X files, Y lines changed] | [action] |
| Dependency depth | X/10 | [X transitive deps affected] | [action] |
| Historical failure rate | X/10 | [X% failure rate in affected area] | [action] |
| Test coverage of changes | X/10 | [X% of changed code tested] | [action] |
| Complexity of changes | X/10 | [cyclomatic complexity delta] | [action] |

**OVERALL REGRESSION RISK: X/50 (High/Medium/Low)**

### 3. Test Selection Effectiveness

| Selection Criteria | Tests Selected | Tests Relevant | Precision |
|-------------------|----------------|----------------|-----------|
| Changed file mapping | X | X | X% |
| Dependency analysis | X | X | X% |
| Historical correlation | X | X | X% |
| Risk-based selection | X | X | X% |

### 4. Missing Test Coverage for Changes

| Changed Code | Coverage Status | Risk | Suggested Test |
|-------------|----------------|------|----------------|
| file:line-range | Covered/Uncovered | High/Medium/Low | [specific test] |

### 5. Regression Prediction

| Module | Regression Probability | Confidence | Key Risk |
|--------|----------------------|------------|----------|
| Module 1 | X% | High/Medium/Low | [factor] |
| Module 2 | X% | High/Medium/Low | [factor] |

**REGRESSION RISK SCORE: X/100** (inverse: lower risk = higher score)

**MINIMUM: Analyze all changed files and identify at least 3 regression risk factors.**

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/03-regression-analysis.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I analyze all changed files?
+-- Did I calculate blast radius?
+-- Did I score all 5 regression risk factors?
+-- Did I evaluate test selection effectiveness?
+-- Did I identify missing test coverage?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-regression-analyzer",
  run_in_background: true
})
```

### Agent 3: Flaky Test Hunter

**This agent MUST detect flaky tests and assess pipeline stability. Flaky count is mandatory.**

```
Task({
  description: "Flaky test detection and pipeline stability assessment",
  prompt: `You are qe-flaky-hunter. Your output quality is being audited.

## PIPELINE DATA TO ANALYZE

=== TEST RESULTS START ===
[PASTE THE COMPLETE TEST RESULTS HERE - DO NOT SUMMARIZE]
=== TEST RESULTS END ===

=== TEST HISTORY (if available) START ===
[PASTE historical test results from previous runs]
=== TEST HISTORY END ===

=== CI/CD LOGS START ===
[PASTE relevant CI/CD pipeline logs]
=== CI/CD LOGS END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Flaky Test Detection

For EACH suspected flaky test:

| Test Name | Suite | Flakiness Score | Evidence | Root Cause |
|-----------|-------|----------------|----------|------------|
| test_name | suite | X/10 | [why suspected flaky] | [timing/ordering/state/env] |

**Flakiness Indicators:**
- Test passes on retry but fails initially
- Test fails inconsistently across runs
- Test depends on execution order
- Test has timing-sensitive assertions
- Test depends on external state
- Test uses non-deterministic data
- Test has race conditions

### 2. Pipeline Stability Assessment

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall pass rate (last 10 runs) | X% | >= 95% | PASS/WARN/FAIL |
| Flaky test rate | X% | <= 2% | PASS/WARN/FAIL |
| Average retry count | X | <= 0.5 | PASS/WARN/FAIL |
| Pipeline timeout rate | X% | <= 1% | PASS/WARN/FAIL |
| Deterministic pass rate | X% | >= 98% | PASS/WARN/FAIL |

### 3. Flaky Test Root Cause Analysis

| Root Cause Category | Count | Tests Affected | Remediation |
|--------------------|-------|----------------|-------------|
| Timing/Race Conditions | X | [test list] | [specific fix] |
| External Dependencies | X | [test list] | [mock/stub] |
| Shared State | X | [test list] | [isolation] |
| Environment Sensitivity | X | [test list] | [env fix] |
| Data Dependencies | X | [test list] | [data setup] |
| Ordering Dependencies | X | [test list] | [independence] |

### 4. Test Stability Trends

| Time Period | Pass Rate | Flaky Rate | Trend |
|-------------|-----------|------------|-------|
| Current run | X% | X% | - |
| Last 5 runs | X% | X% | Improving/Declining/Stable |
| Last 10 runs | X% | X% | Improving/Declining/Stable |
| Last 30 runs | X% | X% | Improving/Declining/Stable |

### 5. Flaky Test Impact Assessment

| Impact | Count | Description |
|--------|-------|-------------|
| False failures blocking releases | X | [which tests cause false blocks] |
| Developer confidence erosion | High/Medium/Low | [evidence] |
| CI/CD resource waste | X% extra runs | [retry cost] |
| Mean time to resolution | X hours | [average time to investigate] |

**FLAKY TESTS TOTAL: X**
**CRITICAL FLAKY (blocking releases): X**

**STABILITY SCORE: X/100** (higher = more stable)

**MINIMUM: Identify at least 3 flaky test indicators or explicitly state "No flaky tests detected after thorough analysis".**

## OUTPUT FORMAT

Save your complete analysis in Markdown to:
${OUTPUT_FOLDER}/04-flaky-test-analysis.md

Use the Write tool to save BEFORE completing.
Report MUST be complete - no placeholders.

## VALIDATION BEFORE SUBMITTING

+-- Did I analyze all test results for flakiness indicators?
+-- Did I check historical test data for patterns?
+-- Did I categorize root causes?
+-- Did I assess pipeline stability with all 5 metrics?
+-- Did I calculate stability trends?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-flaky-hunter",
  run_in_background: true
})
```

### Post-Spawn Confirmation

After sending all three Task calls, you MUST tell the user:

```
I've launched 3 core agents in parallel:

  qe-quality-gate [Domain: quality-assessment]
   - Evaluating quality gate thresholds (8 dimensions)
   - Analyzing test results by category (unit, integration, e2e, contract)
   - Calculating coverage delta against baseline

  qe-regression-analyzer [Domain: test-execution]
   - Computing change impact blast radius
   - Scoring regression risk across 5 factors
   - Evaluating test selection effectiveness

  qe-flaky-hunter [Domain: test-execution]
   - Detecting flaky tests with root cause analysis
   - Assessing pipeline stability (5 metrics)
   - Calculating stability trends

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
From qe-quality-gate:
[ ] qualityGateStatus = PASSED/FAILED (X/Y gates passed)
[ ] testPassRate = __% overall pass rate
[ ] coverageDelta = +/-__% coverage change
[ ] failedTests = __ count of failed tests
[ ] qualityGateScore = __/50

From qe-regression-analyzer:
[ ] regressionRisk = __/50 risk score
[ ] blastRadius = __ files in blast radius
[ ] changedFiles = __ files changed
[ ] missingCoverage = __ uncovered changes
[ ] regressionRiskLevel = High/Medium/Low

From qe-flaky-hunter:
[ ] flakyTests = __ total flaky tests detected
[ ] criticalFlaky = __ blocking releases
[ ] pipelineStability = __% pass rate
[ ] stabilityScore = __/100
[ ] flakyRate = __% flaky rate
```

### Metrics Summary Box

Output extracted metrics:

```
+-------------------------------------------------------------+
|                    BATCH 1 RESULTS SUMMARY                   |
+-------------------------------------------------------------+
|                                                              |
|  Quality Gate:             PASSED/FAILED (X/Y gates)         |
|  Test Pass Rate:           __%                               |
|  Coverage Delta:           +/-__%                            |
|  Failed Tests:             __                                |
|  Quality Gate Score:       __/50                              |
|                                                              |
|  Regression Risk:          __/50 (High/Med/Low)              |
|  Blast Radius:             __ files                          |
|  Changed Files:            __                                |
|  Missing Coverage:         __ uncovered changes              |
|                                                              |
|  Flaky Tests:              __                                |
|  Critical Flaky:           __                                |
|  Pipeline Stability:       __%                               |
|  Stability Score:          __/100                            |
|  Flaky Rate:               __%                               |
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
|  HAS_SECURITY_PIPELINE = TRUE    -> MUST spawn qe-security-scanner   |
|  HAS_PERFORMANCE_PIPELINE = TRUE -> MUST spawn qe-chaos-engineer     |
|  HAS_INFRA_CHANGE = TRUE         -> MUST spawn qe-coverage-specialist|
|  HAS_MIDDLEWARE = TRUE            -> MUST spawn qe-middleware-validator|
|  HAS_SAP_INTEGRATION = TRUE      -> MUST spawn qe-soap-tester        |
|  HAS_AUTHORIZATION = TRUE        -> MUST spawn qe-sod-analyzer       |
|                                                              |
|  Skipping a flagged agent is a FAILURE of this skill.        |
+-------------------------------------------------------------+
```

### Conditional Domain Mapping

| Flag | Agent | Domain | MCP Tool |
|------|-------|--------|----------|
| HAS_SECURITY_PIPELINE | qe-security-scanner | security-compliance | `security_scan_comprehensive` |
| HAS_PERFORMANCE_PIPELINE | qe-chaos-engineer | chaos-resilience | `performance_benchmark` |
| HAS_INFRA_CHANGE | qe-coverage-specialist | coverage-analysis | `coverage_analyze_sublinear` |
| HAS_MIDDLEWARE | qe-middleware-validator | enterprise-integration | `task_orchestrate` |
| HAS_SAP_INTEGRATION | qe-soap-tester | enterprise-integration | `task_orchestrate` |
| HAS_AUTHORIZATION | qe-sod-analyzer | enterprise-integration | `task_orchestrate` |

### Decision Tree

```
IF HAS_SECURITY_PIPELINE == FALSE AND HAS_PERFORMANCE_PIPELINE == FALSE AND HAS_INFRA_CHANGE == FALSE AND HAS_MIDDLEWARE == FALSE AND HAS_SAP_INTEGRATION == FALSE AND HAS_AUTHORIZATION == FALSE:
    -> Skip to Phase 5 (no conditional agents needed)
    -> State: "No conditional agents needed based on pipeline analysis"

ELSE:
    -> Spawn ALL applicable agents in ONE message
    -> Count how many you're spawning: __
```

### IF HAS_SECURITY_PIPELINE: Security Scanner (MANDATORY WHEN FLAGGED)

```
Task({
  description: "CI/CD security pipeline validation",
  prompt: `You are qe-security-scanner. Your output quality is being audited.

## PURPOSE

Validate security gate results in the CI/CD pipeline. Analyze SAST/DAST outputs,
dependency audit results, container scan findings, and secrets detection results.

## PIPELINE SECURITY DATA

=== SECURITY SCAN RESULTS START ===
[PASTE SAST/DAST/dependency audit results]
=== SECURITY SCAN RESULTS END ===

=== DEPENDENCY AUDIT START ===
[PASTE npm audit / pip audit / snyk results]
=== DEPENDENCY AUDIT END ===

=== CONTAINER SCAN (if applicable) START ===
[PASTE container image scan results]
=== CONTAINER SCAN END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Security Gate Status

| Security Gate | Status | Findings | Severity |
|--------------|--------|----------|----------|
| SAST Scan | Pass/Fail | X findings | Critical/High/Medium/Low |
| DAST Scan | Pass/Fail/N-A | X findings | Critical/High/Medium/Low |
| Dependency Audit | Pass/Fail | X vulnerabilities | Critical/High/Medium/Low |
| Container Scan | Pass/Fail/N-A | X findings | Critical/High/Medium/Low |
| Secrets Detection | Pass/Fail | X findings | Critical/High/Medium/Low |
| License Compliance | Pass/Fail | X violations | Critical/High/Medium/Low |

### 2. Vulnerability Inventory (from pipeline scans)

| Vuln ID | Source | Type | Severity | CVSS | Remediation | Status |
|---------|--------|------|----------|------|-------------|--------|
| V001 | SAST/DAST/Dep | [type] | Critical/High | X.X | [fix] | New/Known/Accepted |

### 3. Dependency Risk Assessment

| Dependency | Version | Known CVEs | Risk | Upgrade Path |
|------------|---------|------------|------|-------------|
| pkg-name | X.Y.Z | CVE-XXXX-XXXXX | Critical/High | X.Y.Z+ |

### 4. Security Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| SAST compliance | X/10 | ... |
| Dependency health | X/10 | ... |
| Container security | X/10 | ... |
| Secrets hygiene | X/10 | ... |
| License compliance | X/10 | ... |

**SECURITY PIPELINE SCORE: X/50**

**MINIMUM: Evaluate all 6 security gates and inventory all vulnerabilities found.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/05-security-pipeline.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-security-scanner",
  run_in_background: true
})
```

### IF HAS_PERFORMANCE_PIPELINE: Chaos Engineer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Performance pipeline validation and resilience assessment",
  prompt: `You are qe-chaos-engineer. Your output quality is being audited.

## PURPOSE

Validate performance test results in the CI/CD pipeline. Analyze load test output,
latency baselines, throughput metrics, and resilience test results.

## PERFORMANCE PIPELINE DATA

=== PERFORMANCE TEST RESULTS START ===
[PASTE load test / benchmark results]
=== PERFORMANCE TEST RESULTS END ===

=== BASELINE METRICS (if available) START ===
[PASTE previous performance baselines]
=== BASELINE METRICS END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Performance Gate Assessment

| Gate | Metric | Value | Baseline | Delta | Status |
|------|--------|-------|----------|-------|--------|
| Response Time (p50) | Xs | Xs | +/-X% | PASS/WARN/FAIL |
| Response Time (p95) | Xs | Xs | +/-X% | PASS/WARN/FAIL |
| Response Time (p99) | Xs | Xs | +/-X% | PASS/WARN/FAIL |
| Throughput (RPS) | X | X | +/-X% | PASS/WARN/FAIL |
| Error Rate | X% | X% | +/-X% | PASS/WARN/FAIL |
| Memory Usage | X MB | X MB | +/-X% | PASS/WARN/FAIL |

### 2. Performance Regression Detection

| Endpoint/Feature | Before | After | Regression? | Severity |
|-----------------|--------|-------|-------------|----------|
| GET /api/... | Xms | Xms | Yes/No | Critical/High/Medium |

### 3. Resource Consumption Analysis

| Resource | Peak Usage | Limit | Utilization | Risk |
|----------|-----------|-------|-------------|------|
| CPU | X% | X% | X% | High/Medium/Low |
| Memory | X MB | X MB | X% | High/Medium/Low |
| Disk I/O | X MB/s | X MB/s | X% | High/Medium/Low |
| Network | X MB/s | X MB/s | X% | High/Medium/Low |

### 4. Resilience Assessment

| Scenario | Result | Recovery Time | Status |
|----------|--------|--------------|--------|
| High load (2x baseline) | Pass/Fail | Xs | PASS/FAIL |
| Dependency timeout | Pass/Fail | Xs | PASS/FAIL |
| Memory pressure | Pass/Fail | Xs | PASS/FAIL |
| Connection pool exhaustion | Pass/Fail/N-A | Xs | PASS/FAIL/N-A |

### 5. Performance Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Latency compliance | X/10 | ... |
| Throughput stability | X/10 | ... |
| Resource efficiency | X/10 | ... |
| Resilience | X/10 | ... |

**PERFORMANCE PIPELINE SCORE: X/40**

**MINIMUM: Evaluate all 6 performance gates and detect any regressions.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/06-performance-pipeline.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-chaos-engineer",
  run_in_background: true
})
```

### IF HAS_INFRA_CHANGE: Coverage Specialist (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Infrastructure change coverage and impact analysis",
  prompt: `You are qe-coverage-specialist. Your output quality is being audited.

## PURPOSE

Analyze coverage impact of infrastructure changes. Validate that infrastructure
modifications are properly tested, configuration changes are covered, and
deployment artifacts are verified.

## INFRASTRUCTURE CHANGE DATA

=== INFRA CHANGES START ===
[PASTE infrastructure file diffs - Dockerfiles, CI configs, k8s manifests, etc.]
=== INFRA CHANGES END ===

=== TEST RESULTS FOR INFRA START ===
[PASTE any infrastructure test results - smoke tests, config validation, etc.]
=== TEST RESULTS END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Infrastructure Change Inventory

| File | Change Type | Risk | Test Coverage |
|------|-----------|------|---------------|
| Dockerfile | Modified/New | High/Medium/Low | Tested/Untested |
| .github/workflows/... | Modified/New | High/Medium/Low | Tested/Untested |
| k8s/deployment.yaml | Modified/New | High/Medium/Low | Tested/Untested |
| terraform/... | Modified/New | High/Medium/Low | Tested/Untested |

### 2. Configuration Drift Analysis

| Config Parameter | Previous | New | Impact | Verified |
|-----------------|----------|-----|--------|----------|
| env_var | old_value | new_value | High/Medium/Low | Yes/No |
| resource_limit | old | new | High/Medium/Low | Yes/No |

### 3. Deployment Artifact Verification

| Artifact | Status | Size Delta | Integrity Check |
|----------|--------|-----------|-----------------|
| Container image | Built/Failed | +/-X MB | Pass/Fail |
| Bundle | Built/Failed | +/-X KB | Pass/Fail |
| Migration script | Valid/Invalid | N/A | Pass/Fail |

### 4. Infrastructure Test Coverage

| Test Type | Count | Status | Coverage |
|-----------|-------|--------|----------|
| Smoke tests | X | Pass/Fail | X% of changes |
| Config validation | X | Pass/Fail | X% of changes |
| Integration tests | X | Pass/Fail | X% of changes |
| Deployment dry-run | X | Pass/Fail | X% of changes |

### 5. Infrastructure Risk Score

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Change scope | X/10 | ... |
| Test coverage | X/10 | ... |
| Rollback capability | X/10 | ... |
| Configuration safety | X/10 | ... |

**INFRASTRUCTURE COVERAGE SCORE: X/40**

**MINIMUM: Inventory all infrastructure changes and assess test coverage for each.**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/07-infrastructure-coverage.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-coverage-specialist",
  run_in_background: true
})
```

### IF HAS_MIDDLEWARE: Middleware Validator (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Middleware and message broker pipeline verification",
  prompt: `You are qe-middleware-validator. Your output quality is being audited.

## PURPOSE

Validate middleware and message broker components in the CI/CD pipeline.
Analyze message flow test results, broker health checks, queue metrics,
and integration test outcomes for middleware components.

## PIPELINE ARTIFACTS

=== MIDDLEWARE TEST RESULTS START ===
[PASTE middleware integration test results, broker health checks, queue metrics]
=== MIDDLEWARE TEST RESULTS END ===

=== MIDDLEWARE CONFIG CHANGES START ===
[PASTE middleware configuration diffs - broker configs, queue definitions, routing rules]
=== MIDDLEWARE CONFIG CHANGES END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Middleware Component Inventory

| Component | Type | Protocol | Pipeline Status | Test Coverage |
|-----------|------|----------|-----------------|---------------|
| [name] | Queue/Topic/Exchange/Gateway | AMQP/Kafka/JMS/HTTP | Healthy/Degraded/Failed | [%] |

### 2. Message Flow Test Results

| Flow | Producer | Consumer | Test Status | Latency | Throughput | Errors |
|------|----------|----------|-------------|---------|------------|--------|
| [name] | [source] | [target] | Pass/Fail | [ms] | [msg/s] | [count] |

### 3. Broker Health Assessment

| Metric | Current | Threshold | Status |
|--------|---------|-----------|--------|
| Queue depth | [value] | [max] | PASS/FAIL |
| Consumer lag | [value] | [max] | PASS/FAIL |
| Dead letter count | [value] | 0 | PASS/FAIL |
| Connection pool | [value] | [max] | PASS/FAIL |
| Memory usage | [value] | [max] | PASS/FAIL |

### 4. Configuration Change Impact

| Config Change | Risk | Backward Compatible | Rollback Plan | Verified |
|---------------|------|---------------------|---------------|----------|
| [change] | High/Medium/Low | Yes/No | [plan] | Yes/No |

### 5. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical middleware pipeline issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**MIDDLEWARE PIPELINE SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/10-middleware-pipeline.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-middleware-validator",
  run_in_background: true
})
```

### IF HAS_SAP_INTEGRATION: SOAP Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "SAP SOAP service and integration pipeline verification",
  prompt: `You are qe-soap-tester. Your output quality is being audited.

## PURPOSE

Validate SAP SOAP services and integration points in the CI/CD pipeline.
Analyze WSDL contracts, SOAP message validation results, SAP connector
test outcomes, and integration gateway health.

## PIPELINE ARTIFACTS

=== SAP INTEGRATION TEST RESULTS START ===
[PASTE SAP integration test results, SOAP service tests, connector health checks]
=== SAP INTEGRATION TEST RESULTS END ===

=== SAP CONFIG CHANGES START ===
[PASTE SAP configuration diffs - WSDL changes, connector configs, endpoint changes]
=== SAP CONFIG CHANGES END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. SAP Service Inventory

| Service | Type | Endpoint | WSDL Version | Pipeline Status | Test Coverage |
|---------|------|----------|-------------|-----------------|---------------|
| [name] | SOAP/OData/RFC | [url] | [version] | Healthy/Failed | [%] |

### 2. SOAP Contract Validation

| Contract | WSDL Change | Backward Compatible | Breaking Changes | Schema Valid |
|----------|-------------|---------------------|------------------|-------------|
| [service] | Yes/No | Yes/No | [list] | Yes/No |

### 3. Integration Gateway Health

| Gateway | Status | Response Time | Error Rate | Throughput | Connection Pool |
|---------|--------|---------------|------------|------------|-----------------|
| [name] | Up/Down | [ms] | [%] | [req/s] | [used/max] |

### 4. SAP Connector Test Results

| Connector | Test Type | Status | Messages Processed | Errors | Data Integrity |
|-----------|-----------|--------|-------------------|--------|----------------|
| [name] | End-to-end/Unit/Contract | Pass/Fail | [count] | [count] | Pass/Fail |

### 5. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical SAP pipeline issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**SAP PIPELINE SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/11-sap-pipeline.md
Use the Write tool to save BEFORE completing.`,
  subagent_type: "qe-soap-tester",
  run_in_background: true
})
```

### IF HAS_AUTHORIZATION: SoD Analyzer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Segregation of duties and authorization verification in pipeline",
  prompt: `You are qe-sod-analyzer. Your output quality is being audited.

## PURPOSE

Validate segregation of duties and authorization controls in the CI/CD pipeline.
Analyze role-based access changes, permission matrix diffs, SoD policy
compliance, and access control test results.

## PIPELINE ARTIFACTS

=== AUTHORIZATION TEST RESULTS START ===
[PASTE authorization test results, access control tests, SoD compliance checks]
=== AUTHORIZATION TEST RESULTS END ===

=== AUTHORIZATION CONFIG CHANGES START ===
[PASTE authorization configuration diffs - role definitions, permission matrices, policies]
=== AUTHORIZATION CONFIG CHANGES END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Authorization Change Inventory

| Change | Type | Scope | Risk | Compliance Impact |
|--------|------|-------|------|-------------------|
| [change] | Role/Permission/Policy | [scope] | Critical/High/Medium/Low | [impact] |

### 2. SoD Compliance Verification

| Policy | Status | Violations Found | New Violations | Remediated |
|--------|--------|------------------|----------------|------------|
| [policy] | Compliant/Non-compliant | [count] | [count] | [count] |

### 3. Role Change Impact Analysis

| Role | Changes | Users Affected | Privilege Direction | SoD Check | Approved |
|------|---------|---------------|---------------------|-----------|----------|
| [role] | [changes] | [count] | Escalation/Reduction/Neutral | Pass/Fail | Yes/No |

### 4. Access Control Test Results

| Test Type | Tests Run | Passed | Failed | Coverage |
|-----------|-----------|--------|--------|----------|
| Positive access | [count] | [count] | [count] | [%] |
| Negative access | [count] | [count] | [count] | [%] |
| Cross-role | [count] | [count] | [count] | [%] |
| Privilege escalation | [count] | [count] | [count] | [%] |

### 5. Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | [critical authorization pipeline issues] | [what risk] | [effort] |
| P1 | [important improvements] | [what risk] | [effort] |

**AUTHORIZATION PIPELINE SCORE: X/50**

## OUTPUT FORMAT

Save to: $\{OUTPUT_FOLDER}/12-sod-pipeline.md
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
|    [ ] qe-quality-gate - SPAWNED? [Y/N]                     |
|    [ ] qe-regression-analyzer - SPAWNED? [Y/N]              |
|    [ ] qe-flaky-hunter - SPAWNED? [Y/N]                     |
|                                                              |
|  CONDITIONAL AGENTS (based on flags):                        |
|    [ ] qe-security-scanner - SPAWNED? [Y/N] (HAS_SEC_PIPE)  |
|    [ ] qe-chaos-engineer - SPAWNED? [Y/N] (HAS_PERF_PIPE)   |
|    [ ] qe-coverage-specialist - SPAWNED? [Y/N] (HAS_INFRA)  |
|    [ ] qe-middleware-validator - SPAWNED? [Y/N] (HAS_MIDDLEWARE)  |
|    [ ] qe-soap-tester - SPAWNED? [Y/N] (HAS_SAP_INTEG)          |
|    [ ] qe-sod-analyzer - SPAWNED? [Y/N] (HAS_AUTHORIZATION)     |
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

[IF HAS_SECURITY_PIPELINE]    qe-security-scanner [Domain: security-compliance]
                              - SAST/DAST validation, dependency audit, container scan
[IF HAS_PERFORMANCE_PIPELINE] qe-chaos-engineer [Domain: chaos-resilience]
                              - Performance regression detection, resource analysis, resilience
[IF HAS_INFRA_CHANGE]         qe-coverage-specialist [Domain: coverage-analysis]
                              - Infrastructure change coverage, config drift, artifact verification
[IF HAS_MIDDLEWARE]           qe-middleware-validator [Domain: enterprise-integration]
                              - Middleware pipeline verification, broker health, message flow testing
[IF HAS_SAP_INTEGRATION]     qe-soap-tester [Domain: enterprise-integration]
                              - SAP SOAP service validation, WSDL contracts, integration gateway health
[IF HAS_AUTHORIZATION]        qe-sod-analyzer [Domain: enterprise-integration]
                              - SoD compliance verification, role change impact, access control testing

  WAITING for conditional agents to complete...
```

---

## PHASE 5: Synthesize Results & Determine Recommendation

### ENFORCEMENT: EXACT DECISION LOGIC

**You MUST apply this logic EXACTLY. No interpretation.**

```
STEP 1: Derive composite metrics
-----------------------------------------------------------
qualityGatesPassed = (qualityGateStatus == "PASSED")
testPassRate = testPassRate from qe-quality-gate
regressionRisk = regressionRisk from qe-regression-analyzer (0-50 scale)
flakyRate = flakyRate from qe-flaky-hunter
criticalFlaky = criticalFlaky from qe-flaky-hunter
failedTests = failedTests from qe-quality-gate
securityFindings = (critical + high severity findings from security, if ran)
performanceRegressions = (count of regressions from chaos engineer, if ran)

STEP 2: Check BLOCK conditions (ANY triggers BLOCK)
-----------------------------------------------------------
IF qualityGatesPassed == FALSE        -> BLOCK ("Quality gates failed")
IF testPassRate < 95                  -> BLOCK ("Test pass rate critically low")
IF failedTests > 0 AND critical       -> BLOCK ("Critical test failures")
IF regressionRisk > 40               -> BLOCK ("Regression risk too high")
IF securityFindings > 0 (critical)    -> BLOCK ("Critical security vulnerabilities")

STEP 3: Check RELEASE conditions (ALL required for RELEASE)
-----------------------------------------------------------
IF qualityGatesPassed == TRUE
   AND testPassRate >= 99
   AND regressionRisk <= 15
   AND flakyRate <= 2
   AND criticalFlaky == 0
   AND failedTests == 0
   AND securityFindings == 0          -> RELEASE

STEP 4: Default
-----------------------------------------------------------
ELSE                                  -> REMEDIATE
```

### Decision Recording

```
METRICS:
- qualityGatesPassed = TRUE/FALSE
- testPassRate = __%
- regressionRisk = __/50
- flakyRate = __%
- criticalFlaky = __
- failedTests = __
- securityFindings = __ (if applicable)
- performanceRegressions = __ (if applicable)

BLOCK CHECK:
- qualityGatesPassed == FALSE? __ (YES/NO)
- testPassRate < 95? __ (YES/NO)
- critical test failures? __ (YES/NO)
- regressionRisk > 40? __ (YES/NO)
- critical security findings? __ (YES/NO)

RELEASE CHECK (only if no BLOCK triggered):
- qualityGatesPassed == TRUE? __ (YES/NO)
- testPassRate >= 99? __ (YES/NO)
- regressionRisk <= 15? __ (YES/NO)
- flakyRate <= 2? __ (YES/NO)
- criticalFlaky == 0? __ (YES/NO)
- failedTests == 0? __ (YES/NO)
- securityFindings == 0? __ (YES/NO)

FINAL RECOMMENDATION: [RELEASE / REMEDIATE / BLOCK]
REASON: ___
```

### Remediate Recommendations

If recommendation is REMEDIATE, provide specific remediation steps:

| Issue | Current Value | Required Value | Owner | Action |
|-------|--------------|----------------|-------|--------|
| ... | ... | ... | [who] | [what to do] |

If recommendation is BLOCK, provide mandatory fixes:

| Fix | Priority | Effort | Must Complete Before |
|-----|----------|--------|---------------------|
| ... | P0 | [scope] | [release can proceed] |

---

## PHASE 6: Generate Verification Report

### ENFORCEMENT: COMPLETE REPORT STRUCTURE

**ALL sections below are MANDATORY. No abbreviations.**

```markdown
# QCSD CI/CD Verification Report: [Feature/Release Name]

**Generated**: [Date/Time]
**Recommendation**: [RELEASE / REMEDIATE / BLOCK]
**Agents Executed**: [List all agents that ran]
**Parallel Batches**: [2 or 3 depending on conditional agents]
**Baseline Ref**: [BASELINE_REF value]
**Deploy Target**: [DEPLOY_TARGET value or "Not specified"]

---

## Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Quality Gates | X/Y passed | All pass | PASS/FAIL |
| Test Pass Rate | X% | >= 99% | PASS/WARN/FAIL |
| Regression Risk | X/50 | <= 15 | PASS/WARN/FAIL |
| Flaky Rate | X% | <= 2% | PASS/WARN/FAIL |
| Pipeline Stability | X/100 | >= 90 | PASS/WARN/FAIL |

**Recommendation Rationale**: [1-2 sentences explaining why RELEASE/REMEDIATE/BLOCK]

---

## Quality Gate Analysis

[EMBED or LINK the full report from qe-quality-gate]

### Gate Summary

| Gate | Status |
|------|--------|
[All 8 quality gates from qe-quality-gate]

### Failed Test Details
[Key findings from agent output]

---

## Regression Analysis

[EMBED or LINK the full report from qe-regression-analyzer]

### Impact Summary

| Dimension | Value | Risk |
|-----------|-------|------|
[Key metrics from agent output]

### Missing Coverage
[Gaps identified in changed code]

---

## Flaky Test Analysis

[EMBED or LINK the full report from qe-flaky-hunter]

### Stability Summary

| Metric | Value | Status |
|--------|-------|--------|
[Key metrics from agent output]

### Flaky Tests Identified
[List from agent output]

---

## Conditional Analysis

[INCLUDE ONLY IF APPLICABLE - based on which conditional agents ran]

### Security Pipeline (IF HAS_SECURITY_PIPELINE)
[Full output from qe-security-scanner]

### Performance Pipeline (IF HAS_PERFORMANCE_PIPELINE)
[Full output from qe-chaos-engineer]

### Infrastructure Coverage (IF HAS_INFRA_CHANGE)
[Full output from qe-coverage-specialist]

### Middleware Pipeline (IF HAS_MIDDLEWARE)
[Full output from qe-middleware-validator]

### SAP Pipeline (IF HAS_SAP_INTEGRATION)
[Full output from qe-soap-tester]

### Authorization Pipeline (IF HAS_AUTHORIZATION)
[Full output from qe-sod-analyzer]

---

## Recommended Actions

### Before Release (P0 - Blockers)
- [ ] [Action based on findings]

### Before Next Sprint (P1 - Important)
- [ ] [Action based on findings]

### Tech Debt Backlog (P2 - Improvement)
- [ ] [Action based on findings]

---

## Appendix: Agent Outputs

[Link to or embed full outputs from each agent]

---

*Generated by QCSD CI/CD Swarm v1.0*
*Execution Model: Task Tool Parallel Swarm*
```

Write the executive summary report to:
`${OUTPUT_FOLDER}/01-executive-summary.md`

### Report Validation Checklist

Before presenting report:

```
+-- Executive Summary table is complete with all 5 metrics
+-- Recommendation matches decision logic output
+-- Quality Gate section includes all 8 gate statuses
+-- Regression section includes blast radius and risk score
+-- Flaky Test section includes stability score
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
|  This is NOT optional. It runs on EVERY verification scan.   |
|  It stores findings for cross-phase feedback loops,          |
|  historical pipeline quality tracking, and pattern learning. |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
|  DO NOT treat this as "nice to have".                        |
|  Enforcement Rule E9 applies.                                |
+-------------------------------------------------------------+
```

### Purpose

Store verification findings for:
- Cross-phase feedback loops (Verification -> next Ideation cycle)
- Historical pipeline stability tracking across releases
- Regression risk trend analysis over time
- Pattern learning for flaky test prediction improvement

### Auto-Execution Steps (ALL THREE are MANDATORY)

**Step 1: Store verification findings to memory**

You MUST execute this MCP call with actual values from the verification analysis:

```javascript
mcp__agentic-qe__memory_store({
  key: `qcsd-cicd-${releaseId}-${Date.now()}`,
  namespace: "qcsd-cicd",
  value: {
    releaseId: releaseId,
    releaseName: releaseName,
    recommendation: recommendation,  // RELEASE, REMEDIATE, BLOCK
    metrics: {
      qualityGatesPassed: qualityGatesPassed,
      testPassRate: testPassRate,
      regressionRisk: regressionRisk,
      flakyRate: flakyRate,
      criticalFlaky: criticalFlaky,
      failedTests: failedTests,
      pipelineStability: stabilityScore,
      securityFindings: securityFindings,  // if applicable
      performanceRegressions: performanceRegressions  // if applicable
    },
    flags: {
      HAS_SECURITY_PIPELINE: HAS_SECURITY_PIPELINE,
      HAS_PERFORMANCE_PIPELINE: HAS_PERFORMANCE_PIPELINE,
      HAS_INFRA_CHANGE: HAS_INFRA_CHANGE,
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
  sourceAgentId: "qcsd-cicd-swarm",
  targetAgentIds: ["qe-learning-coordinator", "qe-pattern-learner"],
  knowledgeDomain: "cicd-verification-patterns"
})
```

**Step 3: Save learning persistence record to output folder**

You MUST use the Write tool to save a JSON record of the persisted learnings:

```
Save to: ${OUTPUT_FOLDER}/09-learning-persistence.json

Contents:
{
  "phase": "QCSD-Verification",
  "releaseId": "[release ID]",
  "releaseName": "[release name]",
  "recommendation": "[RELEASE/REMEDIATE/BLOCK]",
  "memoryKey": "qcsd-cicd-[releaseId]-[timestamp]",
  "namespace": "qcsd-cicd",
  "metrics": {
    "qualityGatesPassed": true/false,
    "testPassRate": [0-100],
    "regressionRisk": [0-50],
    "flakyRate": [0-100],
    "criticalFlaky": [N],
    "failedTests": [N],
    "pipelineStability": [0-100],
    "securityFindings": [N or null],
    "performanceRegressions": [N or null]
  },
  "flags": {
    "HAS_SECURITY_PIPELINE": true/false,
    "HAS_PERFORMANCE_PIPELINE": true/false,
    "HAS_INFRA_CHANGE": true/false,
    "HAS_MIDDLEWARE": true/false,
    "HAS_SAP_INTEGRATION": true/false,
    "HAS_AUTHORIZATION": true/false
  },
  "agentsInvoked": ["list", "of", "agents"],
  "crossPhaseSignals": {
    "toProduction": "Release readiness metrics as production monitoring baseline",
    "toIdeation": "Pipeline patterns for future risk assessment"
  },
  "persistedAt": "[ISO timestamp]"
}
```

### Fallback: CLI Memory Commands

If MCP memory_store tool is unavailable, use CLI instead (STILL MANDATORY):

```bash
npx @claude-flow/cli@latest memory store \
  --key "qcsd-cicd-${RELEASE_ID}-$(date +%s)" \
  --value '{"recommendation":"[VALUE]","testPassRate":[N],"regressionRisk":[N],"flakyRate":[N]}' \
  --namespace qcsd-cicd

npx @claude-flow/cli@latest hooks post-task \
  --task-id "qcsd-cicd-${RELEASE_ID}" \
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

The CI/CD Swarm both consumes and produces signals for other QCSD phases:

```
CONSUMES (from other phases):
+-- Loop 3 (Development): SHIP/CONDITIONAL/HOLD decisions
|   - Code quality metrics guide verification depth
|   - HOLD decisions trigger enhanced scrutiny
|   - Coverage data informs regression risk assessment
|
+-- Loop 5 (Pipeline History): Previous verification results
    - Historical flaky test patterns
    - Regression risk baselines
    - Performance benchmarks

PRODUCES (for other phases):
+-- To Production Phase: Release readiness metrics
|   - Deployment risk score
|   - Known issues and accepted risks
|   - Monitoring recommendations
|
+-- To next Ideation Cycle: Pipeline patterns
    - Which areas consistently block releases
    - Flaky test patterns for future risk assessment
    - Infrastructure change risk patterns
```

---

## PHASE 8: Apply Deployment Advisor (Analysis)

### ENFORCEMENT: ALWAYS RUN THIS PHASE

```
+-------------------------------------------------------------+
|  THE DEPLOYMENT ADVISOR MUST ALWAYS RUN                       |
|                                                              |
|  This is NOT conditional. It runs on EVERY verification scan.|
|  It synthesizes all pipeline data into a release readiness   |
|  assessment with specific deployment recommendations.        |
|                                                              |
|  DO NOT skip this phase for any reason.                      |
+-------------------------------------------------------------+
```

### Agent Spawn

```
Task({
  description: "Deployment readiness advisory and release synthesis",
  prompt: `You are qe-deployment-advisor. Your output quality is being audited.

## PURPOSE

Synthesize all verification analysis into a deployment readiness assessment.
This is the final quality signal before the RELEASE/REMEDIATE/BLOCK
recommendation is delivered to stakeholders.

## INPUT: PIPELINE METRICS FROM PREVIOUS AGENTS

### From Quality Gate (02-quality-gate.md):
[Summarize: gate status, test pass rate, coverage delta, failed tests]

### From Regression Analyzer (03-regression-analysis.md):
[Summarize: regression risk, blast radius, missing coverage]

### From Flaky Hunter (04-flaky-test-analysis.md):
[Summarize: flaky count, stability score, pipeline stability]

### From Conditional Agents (if applicable):
[Summarize: security findings, performance regressions, infra coverage]

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Deployment Readiness Matrix

| Dimension | Score (0-10) | Status | Notes |
|-----------|-------------|--------|-------|
| Test Confidence | X/10 | Ready/Conditional/Not Ready | [evidence] |
| Regression Safety | X/10 | Ready/Conditional/Not Ready | [evidence] |
| Pipeline Stability | X/10 | Ready/Conditional/Not Ready | [evidence] |
| Security Posture | X/10 | Ready/Conditional/Not Ready | [evidence] |
| Performance Impact | X/10 | Ready/Conditional/Not Ready | [evidence] |
| Infrastructure Readiness | X/10 | Ready/Conditional/Not Ready | [evidence] |

**DEPLOYMENT READINESS SCORE: X/60**

### 2. Risk Register

| Risk ID | Description | Probability | Impact | Mitigation | Status |
|---------|------------|-------------|--------|------------|--------|
| R001 | [risk description] | High/Med/Low | High/Med/Low | [mitigation] | Open/Mitigated |
| R002 | ... | ... | ... | ... | ... |

### 3. Deployment Recommendation

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| Deploy Strategy | Blue-Green/Canary/Rolling/Direct | [why] |
| Rollback Plan | Automated/Manual/N-A | [how] |
| Monitoring Focus | [specific metrics to watch] | [why] |
| Feature Flags | Required/Recommended/Not Needed | [which features] |
| Canary Percentage | X% (if canary) | [risk-based] |

### 4. Go/No-Go Checklist

| Criteria | Status | Notes |
|----------|--------|-------|
| All quality gates pass | Pass/Fail | ... |
| No critical test failures | Pass/Fail | ... |
| Regression risk acceptable | Pass/Fail | ... |
| No critical security findings | Pass/Fail | ... |
| Performance within baselines | Pass/Fail | ... |
| Rollback plan documented | Pass/Fail | ... |
| Monitoring configured | Pass/Fail | ... |

### 5. Post-Deployment Monitoring Plan

| Metric | Baseline | Alert Threshold | Check Frequency |
|--------|----------|----------------|----------------|
| Error rate | X% | > X% | Every X min |
| Response time (p95) | Xms | > Xms | Every X min |
| CPU utilization | X% | > X% | Every X min |
| Memory usage | X MB | > X MB | Every X min |
| Active users | X | < X (drop) | Every X min |

**DEPLOYMENT READINESS ASSESSMENT: READY / CONDITIONAL / NOT READY**

## OUTPUT FORMAT

Save to: ${OUTPUT_FOLDER}/08-deployment-advisory.md
Use the Write tool to save BEFORE completing.

## VALIDATION BEFORE SUBMITTING

+-- Did I synthesize findings from ALL previous agents?
+-- Did I score all 6 readiness dimensions?
+-- Did I create a risk register?
+-- Did I provide deployment strategy recommendation?
+-- Did I complete the go/no-go checklist?
+-- Did I define post-deployment monitoring plan?
+-- Did I save the report to the correct output path?`,
  subagent_type: "qe-deployment-advisor",
  run_in_background: true
})
```

### Wait for Analysis Completion

```
+-------------------------------------------------------------+
|  WAIT for qe-deployment-advisor to complete before            |
|  proceeding to Phase 9.                                      |
|                                                              |
|  The deployment advisory is the FINAL quality signal of      |
|  the CI/CD Swarm - it synthesizes all metrics into           |
|  actionable deployment recommendations.                      |
+-------------------------------------------------------------+
```

---

## PHASE 9: Final Output

**At the very end of swarm execution, ALWAYS output this completion summary:**

```
+---------------------------------------------------------------------+
|                  QCSD CI/CD SWARM COMPLETE                            |
+---------------------------------------------------------------------+
|                                                                      |
|  Pipeline Verified: [Feature/Release Name]                            |
|  Reports Generated: [count]                                           |
|  Output Folder: ${OUTPUT_FOLDER}                                     |
|                                                                      |
|  VERIFICATION SCORES:                                                 |
|  +-- Quality Gate:            PASSED/FAILED (X/Y gates)              |
|  +-- Test Pass Rate:          __%                                     |
|  +-- Regression Risk:         __/50                                   |
|  +-- Flaky Rate:              __%                                     |
|  +-- Pipeline Stability:      __/100                                  |
|  +-- Deployment Readiness:    __/60                                   |
|  [IF HAS_SECURITY_PIPELINE]                                           |
|  +-- Security Score:          __/50                                   |
|  [IF HAS_PERFORMANCE_PIPELINE]                                        |
|  +-- Performance Score:       __/40                                   |
|  [IF HAS_INFRA_CHANGE]                                                |
|  +-- Infrastructure Score:    __/40                                   |
|  [IF HAS_MIDDLEWARE]                                                  |
|  +-- Middleware Pipeline:     __/50                                   |
|  [IF HAS_SAP_INTEGRATION]                                            |
|  +-- SAP Pipeline:            __/50                                   |
|  [IF HAS_AUTHORIZATION]                                              |
|  +-- Authorization Pipeline:  __/50                                   |
|                                                                      |
|  RECOMMENDATION: [RELEASE / REMEDIATE / BLOCK]                        |
|  REASON: [1-2 sentence rationale]                                     |
|                                                                      |
|  DELIVERABLES:                                                        |
|  +-- 01-executive-summary.md                                          |
|  +-- 02-quality-gate.md                                               |
|  +-- 03-regression-analysis.md                                        |
|  +-- 04-flaky-test-analysis.md                                        |
|  [IF HAS_SECURITY_PIPELINE]                                           |
|  +-- 05-security-pipeline.md                                          |
|  [IF HAS_PERFORMANCE_PIPELINE]                                        |
|  +-- 06-performance-pipeline.md                                       |
|  [IF HAS_INFRA_CHANGE]                                                |
|  +-- 07-infrastructure-coverage.md                                    |
|  [IF HAS_MIDDLEWARE]                                                  |
|  +-- 10-middleware-pipeline.md                                        |
|  [IF HAS_SAP_INTEGRATION]                                            |
|  +-- 11-sap-pipeline.md                                              |
|  [IF HAS_AUTHORIZATION]                                              |
|  +-- 12-sod-pipeline.md                                              |
|  +-- 08-deployment-advisory.md                                        |
|  +-- 09-learning-persistence.json                                     |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is BLOCK, ALSO output this prominent action box:**

```
+---------------------------------------------------------------------+
|  ACTION REQUIRED: PIPELINE BLOCKED - DO NOT RELEASE                   |
+---------------------------------------------------------------------+
|                                                                      |
|  The following blockers MUST be resolved before release:              |
|                                                                      |
|  1. [Blocker 1 with specific remediation]                             |
|  2. [Blocker 2 with specific remediation]                             |
|  3. [Blocker 3 with specific remediation]                             |
|                                                                      |
|  NEXT STEPS:                                                          |
|  - Address all P0 blockers listed above                               |
|  - Re-run CI/CD pipeline after fixes                                  |
|  - Re-run /qcsd-cicd-swarm after pipeline passes                     |
|  - Target: 100% test pass, risk <= 15, 0 critical findings           |
|                                                                      |
+---------------------------------------------------------------------+
```

**IF recommendation is REMEDIATE, output this guidance box:**

```
+---------------------------------------------------------------------+
|  REMEDIATE: PIPELINE NEEDS ATTENTION BEFORE RELEASE                   |
+---------------------------------------------------------------------+
|                                                                      |
|  The pipeline can proceed WITH these remediations:                    |
|                                                                      |
|  1. [Remediation 1 - must be addressed before release]                |
|  2. [Remediation 2 - must be addressed in follow-up]                  |
|                                                                      |
|  DEPLOYMENT STRATEGY:                                                 |
|  - Use canary/blue-green deployment for risk mitigation               |
|  - Monitor [specific metrics] post-deployment                        |
|  - Automated rollback if [conditions]                                 |
|                                                                      |
|  RISK ACCEPTANCE:                                                     |
|  - Release owner acknowledges remaining risks                        |
|  - Follow-up issues created for deferred remediations                 |
|                                                                      |
+---------------------------------------------------------------------+
```

**DO NOT end the swarm without displaying the completion summary.**

---

## Report Filename Mapping

| Agent | Report Filename | Phase |
|-------|----------------|-------|
| qe-quality-gate | `02-quality-gate.md` | Batch 1 |
| qe-regression-analyzer | `03-regression-analysis.md` | Batch 1 |
| qe-flaky-hunter | `04-flaky-test-analysis.md` | Batch 1 |
| qe-security-scanner | `05-security-pipeline.md` | Batch 2 (conditional) |
| qe-chaos-engineer | `06-performance-pipeline.md` | Batch 2 (conditional) |
| qe-coverage-specialist | `07-infrastructure-coverage.md` | Batch 2 (conditional) |
| qe-middleware-validator | `10-middleware-pipeline.md` | Batch 2 (conditional) |
| qe-soap-tester | `11-sap-pipeline.md` | Batch 2 (conditional) |
| qe-sod-analyzer | `12-sod-pipeline.md` | Batch 2 (conditional) |
| qe-deployment-advisor | `08-deployment-advisory.md` | Batch 3 (analysis) |
| Learning Persistence | `09-learning-persistence.json` | Phase 7 (auto-execute) |
| Synthesis | `01-executive-summary.md` | Phase 6 |

---

## DDD Domain Integration

This swarm operates across **2 primary domains**, **4 conditional domains**,
and **1 analysis domain**:

```
+-----------------------------------------------------------------------------+
|                   QCSD CI/CD VERIFICATION - DOMAIN MAP                       |
+-----------------------------------------------------------------------------+
|                                                                              |
|  PRIMARY DOMAINS (Always Active)                                             |
|  +-------------------------------+  +-------------------------------+       |
|  |     quality-assessment        |  |       test-execution          |       |
|  |  ---------------------------  |  |  ---------------------------  |       |
|  |  - qe-quality-gate            |  |  - qe-regression-analyzer    |       |
|  |    (threshold enforcement,    |  |    (change impact, blast      |       |
|  |     pass/fail evaluation)     |  |     radius, regression risk)  |       |
|  +-------------------------------+  |                               |       |
|                                     |  - qe-flaky-hunter            |       |
|                                     |    (flaky detection, pipeline  |       |
|                                     |     stability assessment)      |       |
|                                     +-------------------------------+       |
|                                                                              |
|  CONDITIONAL DOMAINS (Based on Pipeline Content)                             |
|  +-----------------------+  +-----------------------+  +------------------+ |
|  | security-compliance   |  |  chaos-resilience     |  | coverage-analysis| |
|  | ───────────────────── |  |  ──────────────────── |  | ──────────────── | |
|  | qe-security-scanner   |  |  qe-chaos-engineer    |  | qe-coverage-     | |
|  | [IF HAS_SEC_PIPELINE] |  |  [IF HAS_PERF_PIPE]   |  | specialist       | |
|  |                       |  |                       |  | [IF HAS_INFRA]   | |
|  +-----------------------+  +-----------------------+  +------------------+ |
|                                                                              |
|  +-----------------------------------------------------------------------+  |
|  |                  enterprise-integration                                |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-middleware-validator [IF HAS_MIDDLEWARE]                        |  |
|  |  - qe-soap-tester [IF HAS_SAP_INTEGRATION]                          |  |
|  |  - qe-sod-analyzer [IF HAS_AUTHORIZATION]                           |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  ANALYSIS DOMAIN (Always Active)                                             |
|  +-----------------------------------------------------------------------+  |
|  |                    quality-assessment                                   |  |
|  |  -----------------------------------------------------------------   |  |
|  |  - qe-deployment-advisor (readiness matrix, risk register, go/no-go) |  |
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
// Initialize fleet for Verification domains
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["quality-assessment", "test-execution", "security-compliance", "chaos-resilience", "coverage-analysis"],
  maxAgents: 7
})

// Orchestrate verification task
mcp__agentic-qe__task_orchestrate({
  task: "qcsd-cicd-verification",
  strategy: "parallel"
})
```

**Option C: CLI**
```bash
# Initialize coordination
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 7

# Route task
npx @claude-flow/cli@latest hooks pre-task --description "QCSD Verification for [Release]"

# Execute agents
npx @claude-flow/cli@latest agent spawn --type qe-quality-gate
npx @claude-flow/cli@latest agent spawn --type qe-regression-analyzer
npx @claude-flow/cli@latest agent spawn --type qe-flaky-hunter
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
| 8 | ALWAYS run deployment advisor | Skipping analysis |
| 9 | Output completion summary | Missing final output |

### Quality Gate Thresholds

| Metric | RELEASE | REMEDIATE | BLOCK |
|--------|---------|-----------|-------|
| Test Pass Rate | >= 99% | 95-98% | < 95% |
| Regression Risk | <= 15 | 16-40 | > 40 |
| Flaky Rate | <= 2% | 3-10% | > 10% |
| Critical Flaky | 0 | 1-3 | > 3 |
| Security Findings (Critical) | 0 | 0 | > 0 |

### Domain-to-Agent Mapping

| Domain | Agent | Phase | Batch |
|--------|-------|-------|-------|
| quality-assessment | qe-quality-gate | Core | 1 |
| test-execution | qe-regression-analyzer | Core | 1 |
| test-execution | qe-flaky-hunter | Core | 1 |
| security-compliance | qe-security-scanner | Conditional (HAS_SECURITY_PIPELINE) | 2 |
| chaos-resilience | qe-chaos-engineer | Conditional (HAS_PERFORMANCE_PIPELINE) | 2 |
| coverage-analysis | qe-coverage-specialist | Conditional (HAS_INFRA_CHANGE) | 2 |
| enterprise-integration | qe-middleware-validator | Conditional (HAS_MIDDLEWARE) | 2 |
| enterprise-integration | qe-soap-tester | Conditional (HAS_SAP_INTEGRATION) | 2 |
| enterprise-integration | qe-sod-analyzer | Conditional (HAS_AUTHORIZATION) | 2 |
| quality-assessment | qe-deployment-advisor | Analysis (ALWAYS) | 3 |

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
  enabledDomains: ["quality-assessment", "test-execution", "security-compliance", "chaos-resilience", "coverage-analysis"],
  maxAgents: 7
})

// Task submission
mcp__agentic-qe__task_submit({ type: "...", priority: "p0", payload: {...} })
mcp__agentic-qe__task_orchestrate({ task: "...", strategy: "parallel" })

// Status
mcp__agentic-qe__fleet_status({ verbose: true })
mcp__agentic-qe__task_list({ status: "pending" })

// Memory
mcp__agentic-qe__memory_store({ key: "...", value: {...}, namespace: "qcsd-cicd" })
mcp__agentic-qe__memory_query({ pattern: "qcsd-cicd-*", namespace: "qcsd-cicd" })
mcp__agentic-qe__memory_share({
  sourceAgentId: "qcsd-cicd-swarm",
  targetAgentIds: ["qe-learning-coordinator"],
  knowledgeDomain: "cicd-verification-patterns"
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
npx @claude-flow/cli@latest memory store --key "[key]" --value "[json]" --namespace qcsd-cicd
npx @claude-flow/cli@latest memory search --query "[query]" --namespace qcsd-cicd
npx @claude-flow/cli@latest memory list --namespace qcsd-cicd
```

---

## Swarm Topology

```
                 QCSD CI/CD SWARM v1.0
                          |
          BATCH 1 (Core - Parallel)
          +-----------+---+-----------+
          |           |               |
    +-----v-----+ +---v--------+ +---v-----------+
    |  Quality  | | Regression | |  Flaky        |
    |  Gate     | | Analyzer   | |  Hunter       |
    | (Threshd) | | (Risk/Imp) | |  (Stability)  |
    |-----------| |------------| |---------------|
    | qual-asmt | | test-exec  | | test-exec     |
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
    | Security  | | Chaos      | | Coverage     |
    | Scanner   | | Engineer   | | Specialist   |
    | [IF SEC]  | | [IF PERF]  | | [IF INFRA]   |
    |-----------| |------------| |--------------|
    | sec-compl | | chaos-res  | | cov-analy    |
    +-----------+ +------------+ +--------------+
          +-------------+---+-------------+
          |             |                 |
    +-----v------+ +---v--------+ +------v-------+
    | Middleware | | SOAP       | | SoD          |
    | Validator  | | Tester     | | Analyzer     |
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
                 | Deployment    |
                 | Advisor       |
                 | (ALWAYS RUNS) |
                 |---------------|
                 | qual-asmt     |
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
| **Domains** | 7 | 2 | 4 | 1 |
| **Parallel Batches** | 3 | 1 | 1 | 1 |

**Skills Used:**
1. `shift-left-testing` - Pre-merge test strategy
2. `shift-right-testing` - Post-deploy monitoring patterns
3. `regression-testing` - Regression detection framework
4. `security-testing` - OWASP scanning patterns

**Frameworks Applied:**
1. Quality Gate Enforcement - Threshold-based pass/fail evaluation
2. Regression Risk Analysis - Change impact and blast radius calculation
3. Flaky Test Detection - Pattern-based stability assessment
4. SAST/DAST Pipeline Validation - Security gate verification
5. Performance Regression Detection - Baseline comparison analysis
6. Deployment Readiness Matrix - Multi-dimensional release assessment

---

## Key Principle

**Releases ship when pipelines are green, not when deadlines arrive.**

This swarm provides:
1. **Are quality gates passing?** -> Quality Gate Evaluation (8 dimensions)
2. **Will changes break existing features?** -> Regression Risk Analysis (5 factors)
3. **Is the pipeline stable?** -> Flaky Test Detection (6 root cause categories)
4. **Is the security pipeline clean?** -> Security Gate Validation (if security changes)
5. **Are performance baselines met?** -> Performance Regression Detection (if perf changes)
6. **Is infrastructure properly tested?** -> Infrastructure Coverage (if infra changes)
7. **Is it safe to deploy?** -> Deployment Readiness Advisory (always)
8. **Should we release?** -> RELEASE/REMEDIATE/BLOCK decision
9. **What did we learn?** -> Memory persistence for future cycles
