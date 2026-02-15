# Skill Validation MCP Integration Specification

**Version**: 1.0.0
**Created**: 2026-02-02
**Status**: Active

## Overview

All skill validation components MUST use AQE MCP tool calls for shared learning. This ensures validation patterns, outcomes, and insights are stored in the ReasoningBank and shared across the QE agent fleet.

## Required MCP Tool Calls

### 1. Pattern Storage

Skills MUST store successful validation patterns for future reference:

```typescript
// After successful validation
mcp__agentic-qe__memory_store({
  key: `skill-validation-${skillName}-${timestamp}`,
  value: {
    skillName: string,
    trustTier: number,
    validationResult: ValidationResult,
    model: string,
    passRate: number,
    patterns: LearnedPattern[]
  },
  namespace: "skill-validation"
})
```

### 2. Pattern Query

Before validation, query for existing patterns:

```typescript
// Query learned patterns
const existingPatterns = await mcp__agentic-qe__memory_query({
  pattern: `skill-validation-${skillName}-*`,
  namespace: "skill-validation",
  limit: 10
})

// Use patterns to inform validation expectations
```

### 3. Outcome Tracking

Track all validation outcomes for the learning feedback loop:

```typescript
// Record validation outcome
mcp__agentic-qe__test_outcome_track({
  testId: `skill-${skillName}-${evalId}`,
  generatedBy: agentId,
  patternId: usedPatternId,
  passed: boolean,
  coverage: {
    lines: number,
    branches: number,
    functions: number
  },
  executionTime: number,
  flaky: false
})
```

### 4. Cross-Agent Learning Share

Share validation insights with the learning coordinator:

```typescript
// Share learning with fleet
mcp__agentic-qe__memory_share({
  sourceAgentId: currentAgentId,
  targetAgentIds: ["qe-learning-coordinator", "qe-queen-coordinator"],
  knowledgeDomain: "skill-validation",
  data: {
    skillName: string,
    insights: ValidationInsight[],
    recommendations: string[]
  }
})
```

### 5. Quality Gate Integration

Update skill quality scores via quality assessment:

```typescript
// After validation completes
mcp__agentic-qe__quality_assess({
  target: `skill:${skillName}`,
  metrics: {
    passRate: number,
    schemaCompliance: boolean,
    validatorPassed: boolean,
    evalSuiteScore: number
  },
  updateQualityScore: true
})
```

## Memory Namespace Structure

```
aqe/skill-validation/
├── patterns/
│   ├── security-testing/*       - Security validation patterns
│   ├── accessibility-testing/*  - A11y validation patterns
│   └── {skill-name}/*          - Per-skill patterns
├── outcomes/
│   ├── by-skill/               - Outcomes grouped by skill
│   ├── by-model/               - Outcomes grouped by model
│   └── by-date/                - Outcomes grouped by date
├── insights/
│   ├── cross-model/            - Cross-model behavior differences
│   ├── regressions/            - Detected regressions
│   └── improvements/           - Improvement recommendations
└── confidence/
    └── {skill-name}/           - Confidence scores per skill
```

## Eval Runner Integration

The `scripts/run-skill-eval.ts` evaluation runner MUST:

1. **Before running evals**: Query ReasoningBank for learned patterns
2. **During evals**: Track each test case outcome
3. **After evals**: Store patterns and share learning
4. **On regression**: Alert via quality gate

```typescript
// Evaluation runner pseudocode
async function runSkillEval(skill: string, model: string) {
  // 1. Query existing patterns
  const patterns = await mcp__agentic-qe__memory_query({
    pattern: `skill-validation-${skill}-*`,
    namespace: "skill-validation"
  });

  // 2. Run evaluation test cases
  const results = await runTestCases(skill, model, patterns);

  // 3. Track outcomes
  for (const result of results) {
    await mcp__agentic-qe__test_outcome_track({
      testId: result.id,
      passed: result.passed,
      // ...
    });
  }

  // 4. Store new patterns
  await mcp__agentic-qe__memory_store({
    key: `skill-validation-${skill}-${Date.now()}`,
    value: { results, patterns: extractPatterns(results) },
    namespace: "skill-validation"
  });

  // 5. Share learning
  await mcp__agentic-qe__memory_share({
    sourceAgentId: "eval-runner",
    targetAgentIds: ["qe-learning-coordinator"],
    knowledgeDomain: "skill-validation",
    data: summarizeResults(results)
  });

  // 6. Update quality gate
  await mcp__agentic-qe__quality_assess({
    target: `skill:${skill}`,
    metrics: calculateMetrics(results),
    updateQualityScore: true
  });

  return results;
}
```

## Validator Script Integration

Bash validators should call the MCP tools via the CLI wrapper:

```bash
# In validate-skill.cjs after validation
store_validation_result() {
  local skill="$1"
  local result="$2"

  npx aqe memory store \
    --key "skill-validation-${skill}-$(date +%s)" \
    --value "$result" \
    --namespace skill-validation
}

track_outcome() {
  local test_id="$1"
  local passed="$2"

  npx aqe feedback track \
    --test-id "$test_id" \
    --passed "$passed"
}
```

## CI Pipeline Integration

GitHub Actions workflow MUST use MCP tools:

```yaml
- name: Query Baseline Patterns
  run: |
    npx aqe memory query \
      --pattern "skill-validation-${{ matrix.skill }}-*" \
      --namespace skill-validation \
      --limit 5 \
      --output baseline-patterns.json

- name: Run Validation
  run: |
    npx ts-node scripts/run-skill-eval.ts \
      --skill "${{ matrix.skill }}" \
      --model "${{ matrix.model }}" \
      --use-mcp-learning

- name: Share Results with Fleet
  run: |
    npx aqe memory share \
      --source "ci-validator" \
      --targets "qe-learning-coordinator" \
      --domain "skill-validation" \
      --data-file validation-results.json
```

## Success Criteria

- [ ] All validators call `memory_store` after validation
- [ ] Eval runner queries patterns before running
- [ ] Outcomes tracked via `test_outcome_track`
- [ ] Learning shared with coordinator
- [ ] Quality gate updated with validation metrics
- [ ] CI pipeline uses MCP tools for learning

## References

- [ADR-056: Deterministic Skill Validation System](../v3/implementation/adrs/ADR-056-skill-validation-system.md)
- [ADR-021: QE ReasoningBank](../v3/implementation/adrs/v3-adrs.md#adr-021)
- [ADR-023: Quality Feedback Loop](../v3/implementation/adrs/v3-adrs.md#adr-023)
- [AQE MCP Tools Reference](../reference/aqe-fleet.md)
