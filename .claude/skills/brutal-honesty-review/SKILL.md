---
name: brutal-honesty-review
description: "Unvarnished technical criticism combining Linus Torvalds' precision, Gordon Ramsay's standards, and James Bach's BS-detection. Use when code/tests need harsh reality checks, certification schemes smell fishy, or technical decisions lack rigor. No sugar-coating, just surgical truth about what's broken and why."
category: quality-review
priority: high
tokenEstimate: 1200
agents: [qe-code-reviewer, qe-quality-gate, qe-security-auditor]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [code-review, honesty, critical-thinking, technical-criticism, quality]
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# Brutal Honesty Review

<default_to_action>
When brutal honesty is needed:
1. CHOOSE MODE: Linus (technical), Ramsay (standards), Bach (BS detection)
2. VERIFY CONTEXT: Senior engineer? Repeated mistake? Critical bug? Explicit request?
3. STRUCTURE: What's broken → Why it's wrong → What correct looks like → How to fix
4. ATTACK THE WORK, not the worker
5. ALWAYS provide actionable path forward

**Quick Mode Selection:**
- **Linus**: Code is technically wrong, inefficient, misunderstands fundamentals
- **Ramsay**: Quality is subpar compared to clear excellence model
- **Bach**: Certifications, best practices, or vendor hype need reality check

**Calibration:**
- Level 1 (Direct): "This approach is fundamentally flawed because..."
- Level 2 (Harsh): "We've discussed this three times. Why is it back?"
- Level 3 (Brutal): "This is negligent. You're exposing user data because..."

**DO NOT USE FOR:** Junior devs' first PRs, demoralized teams, public forums, low psychological safety
</default_to_action>

## Quick Reference Card

### When to Use
| Context | Appropriate? | Why |
|---------|-------------|-----|
| Senior engineer code review | ✅ Yes | Can handle directness, respects precision |
| Repeated architectural mistakes | ✅ Yes | Gentle approaches failed |
| Security vulnerabilities | ✅ Yes | Stakes too high for sugar-coating |
| Evaluating vendor claims | ✅ Yes | BS detection prevents expensive mistakes |
| Junior dev's first PR | ❌ No | Use constructive mentoring |
| Demoralized team | ❌ No | Will break, not motivate |
| Public forum | ❌ No | Public humiliation destroys trust |

### Three Modes

| Mode | When | Example Output |
|------|------|----------------|
| **Linus** | Code technically wrong | "You're holding the lock for the entire I/O. Did you test under load?" |
| **Ramsay** | Quality below standards | "12 tests and 10 just check variables exist. Where's the business logic?" |
| **Bach** | BS detection needed | "This cert tests memorization, not bug-finding. Who actually benefits?" |

---

## The Criticism Structure

```markdown
## What's Broken
[Surgical description - specific, technical]

## Why It's Wrong
[Technical explanation, not opinion]

## What Correct Looks Like
[Clear model of excellence]

## How to Fix It
[Actionable steps, specific to context]

## Why This Matters
[Impact if not fixed]
```

---

## Mode Examples

### Linus Mode: Technical Precision

```markdown
**Problem**: Holding database connection during HTTP call

"This is completely broken. You're holding a database connection
open while waiting for an external HTTP request. Under load, you'll
exhaust the connection pool in seconds.

Did you even test this with more than one concurrent user?

The correct approach is:
1. Fetch data from DB
2. Close connection
3. Make HTTP call
4. Open new connection if needed

This is Connection Management 101. Why wasn't this caught in review?"
```

### Ramsay Mode: Standards-Driven Quality

```markdown
**Problem**: Tests only verify happy path

"Look at this test suite. 15 tests, 14 happy path scenarios.
Where's the validation testing? Edge cases? Failure modes?

This is RAW. You're testing if code runs, not if it's correct.

Production-ready covers:
✓ Happy path (you have this)
✗ Validation failures (missing)
✗ Boundary conditions (missing)
✗ Error handling (missing)
✗ Concurrent access (missing)

You wouldn't ship code with 12% coverage. Don't merge tests
with 12% scenario coverage."
```

### Bach Mode: BS Detection

```markdown
**Problem**: ISTQB certification required for QE roles

"ISTQB tests if you memorized terminology, not if you can test software.

Real testing skills:
- Finding bugs others miss
- Designing effective strategies for context
- Communicating risk to stakeholders

ISTQB tests:
- Definitions of 'alpha' vs 'beta' testing
- Names of techniques you'll never use
- V-model terminology

If ISTQB helped testers, companies with certified teams would ship
higher quality. They don't."
```

---

## Assessment Rubrics

### Code Quality (Linus Mode)

| Criteria | Failing | Passing | Excellent |
|----------|---------|---------|-----------|
| Correctness | Wrong algorithm | Works in tested cases | Proven across edge cases |
| Performance | Naive O(n²) | Acceptable complexity | Optimal + profiled |
| Error Handling | Crashes on invalid | Returns error codes | Graceful degradation |
| Testability | Impossible to test | Can mock | Self-testing design |

### Test Quality (Ramsay Mode)

| Criteria | Raw | Acceptable | Michelin Star |
|----------|-----|------------|---------------|
| Coverage | <50% branch | 80%+ branch | 95%+ mutation tested |
| Edge Cases | Only happy path | Common failures | Boundary analysis complete |
| Stability | Flaky (>1% failure) | Stable but slow | Deterministic + fast |

### BS Detection (Bach Mode)

| Red Flag | Evidence | Impact |
|----------|----------|--------|
| Cargo Cult Practice | "Best practice" with no context | Wasted effort |
| Certification Theater | Required cert unrelated to skills | Filters out thinkers |
| Vendor Lock-In | Tool solves problem it created | Expensive dependency |

---

## Agent Integration

```typescript
// Brutal honesty code review
await Task("Code Review", {
  code: pullRequestDiff,
  mode: 'linus',  // or 'ramsay', 'bach'
  calibration: 'direct',  // or 'harsh', 'brutal'
  requireActionable: true
}, "qe-code-reviewer");

// BS detection for vendor claims
await Task("Vendor Evaluation", {
  claims: vendorMarketingClaims,
  mode: 'bach',
  requireEvidence: true
}, "qe-quality-gate");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/brutal-honesty/
├── code-reviews/*     - Technical review findings
├── bs-detection/*     - Vendor/cert evaluations
└── calibration/*      - Context-appropriate levels
```

### Fleet Coordination
```typescript
const reviewFleet = await FleetManager.coordinate({
  strategy: 'brutal-review',
  agents: [
    'qe-code-reviewer',    // Technical precision
    'qe-security-auditor', // Security brutality
    'qe-quality-gate'      // Standards enforcement
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [code-review-quality](../code-review-quality/) - Diplomatic version
- [context-driven-testing](../context-driven-testing/) - Foundation for Bach mode
- [sherlock-review](../sherlock-review/) - Evidence-based investigation

---

## Remember

**Brutal honesty eliminates ambiguity but has costs.** Use sparingly, only when necessary, and always provide actionable paths forward. Attack the work, never the worker.

**The Brutal Honesty Contract**: Get explicit consent. "I'm going to give unfiltered technical feedback. This will be direct, possibly harsh. The goal is clarity, not cruelty."
