# Brutal Honesty Review Skill

## Overview

A QE skill that delivers unvarnished technical criticism combining three legendary personas:
- **Linus Torvalds**: Surgical technical precision
- **Gordon Ramsay**: Standards-driven quality assessment
- **James Bach**: BS detection in testing practices

## Purpose

Unlike diplomatic reviews, this skill eliminates ambiguity about technical standards. It **dissects why something is wrong, explains the correct approach, and has zero patience for repeated mistakes or sloppy thinking**.

## When to Use

✅ **Appropriate**:
- Senior engineers wanting unfiltered feedback
- Repeated architectural mistakes
- Critical bugs requiring immediate attention
- Evaluating vendor claims/certifications
- Teams explicitly requesting no-BS feedback

❌ **Inappropriate**:
- Junior developers' first contributions
- Demoralized teams
- Public forums (avoid humiliation)
- When psychological safety is low

## Three Modes

### 1. Linus Mode (Technical Precision)
Focus on code correctness, performance, concurrency, and architecture.

**Example**:
> "This is completely broken. You're holding the lock during I/O, which means every thread serializes. Did you even test under load? The correct approach is..."

### 2. Ramsay Mode (Standards-Driven Quality)
Compare reality against clear excellence model using concrete metrics.

**Example**:
> "Look at this! 12 tests and 10 are just checking if variables exist. Where's the business logic coverage? This is RAW. Don't merge it."

### 3. Bach Mode (BS Detection)
Question certifications, best practices, and vendor hype.

**Example**:
> "This certification teaches scripts, not thinking. Does it help testers find bugs faster? No. It helps the cert body make money."

## Quick Start

### Using the Skill

Invoke the skill when you need brutal honesty:

```markdown
Use brutal-honesty-review skill in Linus mode to review this code:
[paste code]
```

### Using Assessment Scripts

```bash
# Assess code quality (Linus Mode)
./scripts/assess-code.sh src/myfile.js

# Assess test quality (Ramsay Mode)
./scripts/assess-tests.sh tests/
```

## Files Included

```
brutal-honesty-review/
├── SKILL.md                          # Main skill instructions
├── README.md                         # This file
├── resources/
│   ├── review-template.md            # Template for structured reviews
│   └── assessment-rubrics.md         # Scoring rubrics for all modes
└── scripts/
    ├── assess-code.sh                # Automated code quality check
    └── assess-tests.sh               # Automated test quality check
```

## Assessment Rubrics

### Code Quality (Linus Mode)

| Criteria | Failing | Passing | Excellent |
|----------|---------|---------|-----------|
| Correctness | Wrong logic | Works in tested cases | Proven across edge cases |
| Performance | O(n²) where O(n) exists | Acceptable | Optimal + profiled |
| Error Handling | Crashes | Returns errors | Graceful degradation |
| Concurrency | Race conditions | Thread-safe | Lock-free/proven |
| Testability | Can't unit test | Mockable | Self-testing |
| Maintainability | "Clever" code | Clear | Self-documenting |

**Threshold**: Minimum "Passing" on ALL to merge.

### Test Quality (Ramsay Mode)

| Criteria | Raw | Acceptable | Michelin Star |
|----------|-----|------------|---------------|
| Coverage | <50% | 80%+ | 95%+ + mutation |
| Edge Cases | Happy path only | Common failures | Boundary analysis |
| Clarity | Unclear names | Clear names | Self-documenting |
| Speed | Minutes | <10s | <1s, parallel |
| Stability | Flaky (>1%) | Stable | Deterministic |
| Isolation | Dependent | Independent | Pure, stateless |

**Threshold**: Minimum "Acceptable" on ALL to merge.

### BS Detection (Bach Mode)

Red flags:
- Cargo cult practices (no context)
- Certification theater (filters thinkers)
- Vendor lock-in (solves own problem)
- False automation (still needs humans)
- Checkbox quality (compliance, not outcomes)
- Hype cycle (10x claims without proof)

**Green flag test**: "Does this help testers/developers do better work in THIS context?"

## Examples

### Example 1: Linus Mode - Concurrency Bug

**Problem**: Holding DB connection during HTTP call

**Brutal Honesty**:
> "This is completely broken. You're holding a database connection open while waiting for an HTTP request. Under load, you'll exhaust the connection pool in seconds. Did you test with >1 user? The correct approach: 1) Fetch data, 2) Close connection, 3) Make HTTP call."

### Example 2: Ramsay Mode - Weak Tests

**Problem**: 15 tests, all happy path

**Brutal Honesty**:
> "Look at this test suite. 15 tests, 14 are happy path. Where's validation? Where are edge cases? This is RAW. You're testing if code runs, not if it's correct. You have 35% coverage. Production needs 80%+. Don't merge this."

### Example 3: Bach Mode - Certification Theater

**Problem**: Required ISTQB certification

**Brutal Honesty**:
> "ISTQB tests if you memorized terminology, not if you can test software. Real skills: finding bugs, designing strategies, communicating risk. ISTQB tests: definitions, model names, checkbox thinking. If it helped, ISTQB-certified teams would ship higher quality. They don't."

## Calibration Guide

### Harshness Levels

**Level 1 - Direct**: For experienced engineers
> "This approach is flawed because..."

**Level 2 - Harsh**: For repeated mistakes
> "We've discussed this pattern three times. Why is it back?"

**Level 3 - Brutal**: For critical issues or willful ignorance
> "This is negligent. You're exposing user data because..."

### Context Matters

Before being brutal, verify:
1. **Audience maturity**: Can they handle directness?
2. **Relationship capital**: Have you earned harsh feedback rights?
3. **Actionability**: Can they actually fix this?
4. **Intent**: Helping or venting?

## Philosophy

### Why Brutal Honesty Works

1. **Eliminates Ambiguity**: No confusion about technical standards
2. **Scales Teaching**: Public technical breakdown teaches entire team
3. **Cuts Through BS**: Exposes cargo cult practices and vendor hype

### The Costs

1. **Relationship Damage**: Harsh criticism without trust destroys collaboration
2. **Chills Participation**: Fear stops newcomers from contributing
3. **Burnout**: Constant harshness is exhausting

### The Contract

Before using brutal honesty, establish explicit contract:

> "I'm going to give unfiltered technical feedback. This will be direct, possibly harsh. Goal is clarity, not cruelty. I'll explain: 1) What's wrong, 2) Why, 3) What correct looks like, 4) How to fix. Want diplomatic instead?"

**Get explicit consent first.**

## Integration with Agentic QE Fleet

This skill extends the QE skill set:

- **Complements**: `code-review-quality`, `context-driven-testing`
- **Contrasts with**: Diplomatic, constructive review approaches
- **Use when**: Technical precision and BS-cutting are priorities
- **Avoid when**: Building confidence or psychological safety

## Related Skills

- `code-review-quality` - Diplomatic code review
- `context-driven-testing` - Foundation for BS detection
- `tdd-london-chicago` - Systematic quality approach
- `exploratory-testing-advanced` - Critical thinking in testing

## License

MIT - Part of Agentic QE Fleet

## Contributing

Found this too harsh? Good. Found this too soft? You're Linus Torvalds.

Submit issues if the skill produces ineffective criticism (not actionable, personal attacks, wrong context).

---

**Remember**: Brutal honesty is a tool, not a personality. Use it when it helps, not when it harms.
