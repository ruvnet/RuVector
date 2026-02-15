# Requirements Validation Domain Shard

**Domain**: requirements-validation
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Pre-Development Validation**: Requirements MUST be validated BEFORE development begins; post-hoc validation is insufficient for shift-left quality.

2. **Testability Scoring Mandatory**: All requirements MUST receive a testability score from TestabilityScorerService; untestable requirements MUST be flagged for rewriting.

3. **BDD Scenario Completeness**: Generated BDD scenarios MUST cover happy path, edge cases, and error handling; single-scenario requirements are incomplete.

4. **Ambiguity Detection**: Requirements with ambiguous terms MUST be flagged via AmbiguityReport before acceptance; vague language is a quality gate blocker.

5. **QCSD/HTSM Alignment**: Quality criteria analysis MUST use HTSM categories with never-omit rules enforced for security, performance, and accessibility.

6. **Dependency Graph Accuracy**: Requirement dependency graphs MUST be validated against actual code dependencies when implementation exists.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Testability Score | 0.6 | 0.8 | < 0.4 |
| Confidence | 0.7 | 0.85 | < 0.5 |
| BDD Coverage | 0.7 | 0.9 | < 0.5 |
| Ambiguity Rate | < 0.2 | < 0.1 | > 0.3 |
| HTSM Completeness | 0.8 | 0.95 | < 0.6 |

---

## Invariants

```
INVARIANT pre_development_validation:
  FOR ALL requirement IN requirements:
    IF requirement.status = 'ready_for_dev' THEN
      requirement.validation_completed = true AND
      requirement.validation_timestamp < requirement.dev_start_timestamp
```

```
INVARIANT testability_score_required:
  FOR ALL requirement IN requirements:
    requirement.testability_score IS NOT NULL AND
    requirement.testability_score >= 0.0 AND
    requirement.testability_score <= 1.0
```

```
INVARIANT bdd_scenario_completeness:
  FOR ALL requirement IN validated_requirements:
    requirement.bdd_scenarios.length >= 2 AND
    EXISTS happy_path IN requirement.bdd_scenarios AND
    EXISTS error_case IN requirement.bdd_scenarios
```

```
INVARIANT ambiguity_detection:
  FOR ALL requirement IN requirements:
    requirement.ambiguity_report IS NOT NULL AND
    IF requirement.ambiguity_report.ambiguous_terms.length > 0 THEN
      requirement.status != 'approved'
```

```
INVARIANT htsm_never_omit:
  FOR ALL qcsd_analysis IN quality_criteria_analyses:
    qcsd_analysis.categories INCLUDES ['Security', 'Performance', 'Accessibility']
```

---

## Patterns

**Domain Source**: `v3/src/domains/requirements-validation/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Requirements Validator Service | `services/requirements-validator.ts` | Core validation |
| BDD Scenario Writer Service | `services/bdd-scenario-writer.ts` | Gherkin generation |
| Testability Scorer Service | `services/testability-scorer.ts` | Testability assessment |
| Test Idea Transformer Service | `services/test-idea-transformer.ts` | Test idea conversion |
| Quality Criteria Service | `services/quality-criteria/` | HTSM/QCSD analysis |
| QCSD Ideation Plugin | `qcsd-ideation-plugin.ts` | Ideation swarm integration |
| Product Factors Assessment | `services/product-factors-assessment/` | Comprehensive analysis |

**HTSM Categories**: Defined in `services/quality-criteria/index.ts` with NEVER_OMIT_CATEGORIES enforcement.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-requirements-analyst` | Full validation, testability scoring |
| **Secondary** | `qe-bdd-specialist` | BDD scenario generation |
| **Secondary** | `qe-quality-criteria-recommender` | HTSM/QCSD analysis |
| **Support** | `qe-risk-assessor` | Risk factor identification |
| **Support** | `qe-security-scanner` | Security criteria analysis |
| **Support** | `qe-accessibility-auditor` | A11y criteria analysis |
| **Readonly** | `qe-test-architect` | Query requirements for test planning |

**Forbidden Actions**: No agent may mark requirements as "ready for development" without completed validation.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Testability score < 0.4 | CRITICAL | Block approval, escalate to Queen Coordinator |
| Ambiguity rate > 0.3 | CRITICAL | Block approval, request requirement rewrite |
| HTSM never-omit category missing | CRITICAL | Block approval, add missing categories |
| BDD scenarios < 2 | HIGH | Request additional scenarios |
| Dependency graph conflict | HIGH | Escalate for resolution |
| Post-hoc validation attempted | MEDIUM | Log violation, recommend shift-left |
| Validation timestamp missing | MEDIUM | Block status change |

---

## Memory Namespace

- **Namespace**: `qe-patterns/requirements-validation`
- **Retention**: 90 days
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-generation` | Output | Provide BDD scenarios for test generation |
| `security-compliance` | Bidirectional | Security requirements validation |
| `visual-accessibility` | Bidirectional | Accessibility requirements validation |
| `quality-assessment` | Output | Report validation status |
| `learning-optimization` | Bidirectional | Share validation patterns |

---

## Testability Factor Weights

```typescript
interface FactorWeights {
  specificity: 0.25;      // Clear, measurable acceptance criteria
  atomicity: 0.20;        // Single responsibility
  traceability: 0.15;     // Links to tests and code
  independence: 0.15;     // Can be tested in isolation
  automatable: 0.15;      // Suitable for automation
  unambiguous: 0.10;      // No vague language
}
```

---

## HTSM Categories (Never-Omit Rules)

| Category | Never-Omit | Description |
|----------|-----------|-------------|
| Security | YES | Authentication, authorization, data protection |
| Performance | YES | Response time, throughput, scalability |
| Accessibility | YES | WCAG compliance, assistive technology |
| Usability | NO | User experience, learnability |
| Reliability | NO | Error handling, recovery |
| Maintainability | NO | Code quality, documentation |
| Portability | NO | Cross-platform, compatibility |

---

*This shard is enforced by @claude-flow/guidance governance system.*
