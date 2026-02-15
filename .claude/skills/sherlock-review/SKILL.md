---
name: sherlock-review
description: "Evidence-based investigative code review using deductive reasoning to determine what actually happened versus what was claimed. Use when verifying implementation claims, investigating bugs, validating fixes, or conducting root cause analysis. Elementary approach to finding truth through systematic observation."
category: quality-review
priority: high
tokenEstimate: 1100
agents: [qe-code-reviewer, qe-security-auditor, qe-performance-validator]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [investigation, evidence-based, code-review, root-cause, deduction]
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# Sherlock Review

<default_to_action>
When investigating code claims:
1. OBSERVE: Gather all evidence (code, tests, history, behavior)
2. DEDUCE: What does evidence actually show vs. what was claimed?
3. ELIMINATE: Rule out what cannot be true
4. CONCLUDE: Does evidence support the claim?
5. DOCUMENT: Findings with proof, not assumptions

**The 3-Step Investigation:**
```bash
# 1. OBSERVE: Gather evidence
git diff <commit>
npm test -- --coverage

# 2. DEDUCE: Compare claim vs reality
# Does code match description?
# Do tests prove the fix/feature?

# 3. CONCLUDE: Verdict with evidence
# SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED
```

**Holmesian Principles:**
- "Data! Data! Data!" - Collect before concluding
- "Eliminate the impossible" - What cannot be true?
- "You see, but do not observe" - Run code, don't just read
- Trust only reproducible evidence
</default_to_action>

## Quick Reference Card

### Evidence Collection Checklist

| Category | What to Check | How |
|----------|---------------|-----|
| **Claim** | PR description, commit messages | Read thoroughly |
| **Code** | Actual file changes | `git diff` |
| **Tests** | Coverage, assertions | Run independently |
| **Behavior** | Runtime output | Execute locally |
| **Timeline** | When things happened | `git log`, `git blame` |

### Verdict Levels

| Verdict | Meaning |
|---------|---------|
| ✓ **TRUE** | Evidence fully supports claim |
| ⚠ **PARTIALLY TRUE** | Claim accurate but incomplete |
| ✗ **FALSE** | Evidence contradicts claim |
| ? **NONSENSICAL** | Claim doesn't apply to context |

---

## Investigation Template

```markdown
## Sherlock Investigation: [Claim]

### The Claim
"[What PR/commit claims to do]"

### Evidence Examined
- Code changes: [files, lines]
- Tests added: [count, coverage]
- Behavior observed: [what actually happens]

### Deductive Analysis

**Claim**: [specific assertion]
**Evidence**: [what you found]
**Deduction**: [logical conclusion]
**Verdict**: ✓/⚠/✗

### Findings
- What works: [with evidence]
- What doesn't: [with evidence]
- What's missing: [gaps in implementation/testing]

### Recommendations
1. [Action based on findings]
```

---

## Investigation Scenarios

### Scenario 1: "This Fixed the Bug"

**Steps:**
1. Reproduce bug on commit before fix
2. Verify bug is gone on commit with fix
3. Check if fix addresses root cause or symptom
4. Test edge cases not in original report

**Red Flags:**
- Fix that just removes error logging
- Works only for specific test case
- Workarounds instead of root cause fix
- No regression test added

### Scenario 2: "Improved Performance by 50%"

**Steps:**
1. Run benchmark on baseline commit
2. Run same benchmark on optimized commit
3. Compare in identical conditions
4. Verify measurement methodology

**Red Flags:**
- Tested only on toy data
- Different comparison conditions
- Trade-offs not mentioned

### Scenario 3: "Handles All Edge Cases"

**Steps:**
1. List all edge cases in code path
2. Check each has test coverage
3. Test boundary conditions
4. Verify error handling paths

**Red Flags:**
- `catch {}` swallowing errors
- Generic error messages
- No logging of critical errors

---

## Example Investigation

```markdown
## Case: PR #123 "Fix race condition in async handler"

### Claims Examined:
1. "Eliminates race condition"
2. "Adds mutex locking"
3. "100% thread safe"

### Evidence:
- File: src/handlers/async-handler.js
- Changes: Added `async/await`, removed callbacks
- Tests: 2 new tests for async flow
- Coverage: 85% (was 75%)

### Analysis:

**Claim 1: "Eliminates race condition"**
Evidence: Added `await` to sequential operations. No actual mutex.
Deduction: Race avoided by removing concurrency, not synchronization.
Verdict: ⚠ PARTIALLY TRUE (solved differently than claimed)

**Claim 2: "Adds mutex locking"**
Evidence: No mutex library, no lock variables, no sync primitives.
Verdict: ✗ FALSE

**Claim 3: "100% thread safe"**
Evidence: JavaScript is single-threaded. No worker threads used.
Verdict: ? NONSENSICAL (meaningless in this context)

### Conclusion:
Fix works but not for reasons claimed. Race condition avoided by
making operations sequential, not by adding synchronization.

### Recommendations:
1. Update PR description to accurately reflect solution
2. Add test for concurrent request handling
3. Remove incorrect technical claims
```

---

## Agent Integration

```typescript
// Evidence-based code review
await Task("Sherlock Review", {
  prNumber: 123,
  claims: [
    "Fixes memory leak",
    "Improves performance 30%"
  ],
  verifyReproduction: true,
  testEdgeCases: true
}, "qe-code-reviewer");

// Bug fix verification
await Task("Verify Fix", {
  bugCommit: 'abc123',
  fixCommit: 'def456',
  reproductionSteps: steps,
  testBoundaryConditions: true
}, "qe-code-reviewer");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/sherlock/
├── investigations/*   - Investigation reports
├── evidence/*         - Collected evidence
├── verdicts/*         - Claim verdicts
└── patterns/*         - Common deception patterns
```

### Fleet Coordination
```typescript
const investigationFleet = await FleetManager.coordinate({
  strategy: 'evidence-investigation',
  agents: [
    'qe-code-reviewer',        // Code analysis
    'qe-security-auditor',     // Security claim verification
    'qe-performance-validator' // Performance claim verification
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [brutal-honesty-review](../brutal-honesty-review/) - Direct technical criticism
- [context-driven-testing](../context-driven-testing/) - Adapt to context
- [bug-reporting-excellence](../bug-reporting-excellence/) - Document findings

---

## Remember

**"It is a capital mistake to theorize before one has data."** Trust only reproducible evidence. Don't trust commit messages, documentation, or "works on my machine."

**The Sherlock Standard:** Every claim must be verified empirically. What does the evidence actually show?
