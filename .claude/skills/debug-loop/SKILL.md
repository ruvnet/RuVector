---
name: debug-loop
description: Hypothesis-driven autonomous debugging with real command validation
trust_tier: 3
domain: debugging
---

# Debug Loop

Autonomous hypothesis-driven debugging against real data. No guessing, no simulating.

## Arguments

- `<symptom>` — Description of the bug or unexpected behavior. If omitted, prompt the user.

## Phases

### Phase 1 — Reproduce
Run the exact command that shows the bug. Capture and display the REAL output. Confirm the bug is visible.

If the bug cannot be reproduced, stop and explain what was tried.

### Phase 2 — Hypothesize and Test (up to 5 iterations)
For each iteration:
1. State a specific hypothesis (e.g., "the query targets v2 tables but data is in v3 tables")
2. Run a REAL command to test it (e.g., `sqlite3 [db path] '.tables'` then `SELECT COUNT(*) FROM [table]`)
3. Record whether the hypothesis was confirmed or rejected
4. If rejected, form the next hypothesis based on what you learned

**Do NOT make code changes until you have a confirmed root cause.**

Important checks:
- Always check both v2 and v3 SQLite tables when data issues are suspected
- Check dependency versions (e.g., sqlite3 vs better-sqlite3)
- Check for hardcoded values that may have been missed

### Phase 3 — Fix
Make the minimal targeted fix. Explain:
- What the root cause was
- What you're changing and why
- What the blast radius is (which other code paths are affected)

Before applying, grep for ALL instances of the problematic pattern across the entire codebase.

### Phase 4 — Verify
Run the SAME reproduction command from Phase 1. The output must now show correct values. If it doesn't, go back to Phase 2.

Show before/after output comparison.

### Phase 5 — Regression
```bash
npm test
```
Run the full test suite. If tests fail, fix them before committing.

## Rules

- NEVER guess or simulate output — always run real commands
- NEVER make code changes before confirming root cause
- Always check for the pattern across the entire codebase, not just one file
- If blocked after 5 hypotheses, stop and ask the user for guidance
