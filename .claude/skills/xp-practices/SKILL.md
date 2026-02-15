---
name: xp-practices
description: "Apply XP practices including pair programming, ensemble programming, continuous integration, and sustainable pace. Use when implementing agile development practices, improving team collaboration, or adopting technical excellence practices."
category: methodology
priority: medium
tokenEstimate: 1000
agents: [qe-quality-analyzer, qe-test-executor, qe-code-reviewer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [xp, agile, pair-programming, tdd, continuous-integration, collaboration]
trust_tier: 0
validation:

---

# Extreme Programming (XP) Practices

<default_to_action>
When applying XP practices:
1. START with practices that give immediate value
2. BUILD supporting practices gradually
3. ADAPT to your context
4. MEASURE results

**Core XP Practices (Prioritized):**
| Practice | Start Here | Why First |
|----------|------------|-----------|
| TDD | ✅ Yes | Foundation for everything |
| Continuous Integration | ✅ Yes | Fast feedback |
| Pair Programming | ✅ Yes | Knowledge sharing |
| Collective Ownership | After CI+TDD | Needs safety net |
| Small Releases | After CI | Infrastructure dependent |

**Pairing Quick Start:**
```
Driver-Navigator (Classic):
- Driver: Writes code
- Navigator: Reviews, thinks ahead
- Rotate every 20-30 min

Ping-Pong (with TDD):
A: Write failing test
B: Make test pass + refactor
B: Write next failing test
A: Make test pass + refactor
```
</default_to_action>

## Quick Reference Card

### The Five XP Values

| Value | Meaning | Practice |
|-------|---------|----------|
| **Communication** | Everyone knows what everyone does | Daily standups, pairing |
| **Simplicity** | Simplest thing that works | YAGNI, small design |
| **Feedback** | Get feedback early and often | TDD, CI, short iterations |
| **Courage** | Tell truth, adapt | Refactor, admit mistakes |
| **Respect** | Everyone contributes value | Sustainable pace, pairing |

### Core Practices

| Practice | Description | Benefit |
|----------|-------------|---------|
| **Pair Programming** | Two devs, one workstation | Quality + knowledge sharing |
| **TDD** | Red-Green-Refactor | Confidence + design |
| **CI** | Integrate multiple times/day | Fast feedback |
| **Collective Ownership** | Anyone can change anything | No bottlenecks |
| **Sustainable Pace** | 40-hour weeks | Long-term productivity |
| **Small Releases** | Ship frequently | Risk reduction |

---

## Pair Programming

### When to Pair

| Context | Pair? | Why |
|---------|-------|-----|
| Complex/risky code | ✅ Always | Needs multiple perspectives |
| New technology | ✅ Always | Learning accelerator |
| Onboarding | ✅ Always | Knowledge transfer |
| Critical bugs | ✅ Always | Two heads better |
| Simple tasks | ❌ Skip | Not worth overhead |
| Research spikes | ❌ Skip | Pair to discuss findings |

### Pairing Dos and Don'ts

**Do:**
- ✅ Switch roles every 20-30 min
- ✅ Take breaks together
- ✅ Think out loud
- ✅ Ask questions
- ✅ Keep sessions 2-4 hours max

**Don't:**
- ❌ Grab keyboard without asking
- ❌ Check phone while pairing
- ❌ Dominate conversation
- ❌ Pair all day (exhausting)

---

## Ensemble (Mob) Programming

**Setup:** 3+ developers, one screen, rotating driver

```
[Screen]
   ↓
[Driver] ← Directions from navigators
   ↑
[Navigator 1] [Navigator 2] [Navigator 3]
```

**Rotation:** Driver switches every 5-10 min

**Best for:**
- Complex problem solving
- Architectural decisions
- Learning new frameworks
- Resolving blockers

---

## Continuous Integration

**CI Workflow:**
```
1. Pull latest from main
2. Make small change (<2 hrs work)
3. Run tests locally (all pass)
4. Commit and push
5. CI runs tests automatically
6. If fail → fix immediately
```

**Best Practices:**
- Commit frequently (small changes)
- Keep build fast (<10 min)
- Fix broken builds immediately
- Never commit to broken build

---

## Four Rules of Simple Design

(In priority order)
1. **Passes all tests** - Works correctly
2. **Reveals intention** - Clear, expressive code
3. **No duplication** - DRY principle
4. **Fewest elements** - No speculative code

---

## Agent Integration

```typescript
// Agent-human pair testing
const charter = "Test payment edge cases";
const tests = await Task("Generate Tests", { charter }, "qe-test-generator");
const reviewed = await human.review(tests);
await Task("Implement", { tests: reviewed }, "qe-test-generator");

// Continuous integration with agents
await Task("Risk Analysis", { prDiff }, "qe-regression-risk-analyzer");
await Task("Generate Tests", { changes: prDiff }, "qe-test-generator");
await Task("Execute Tests", { scope: 'affected' }, "qe-test-executor");

// Sustainable pace: agents handle grunt work
const agentWork = ['regression', 'data-generation', 'coverage-analysis'];
const humanWork = ['exploratory', 'risk-assessment', 'strategy'];
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/xp-practices/
├── pairing-sessions/*   - Pair/ensemble session logs
├── ci-metrics/*         - CI health metrics
├── velocity/*           - Team velocity data
└── retrospectives/*     - XP retrospective notes
```

### Fleet Coordination
```typescript
const xpFleet = await FleetManager.coordinate({
  strategy: 'xp-workflow',
  agents: [
    'qe-test-generator',   // TDD support
    'qe-test-executor',    // CI integration
    'qe-code-reviewer'     // Collective ownership
  ],
  topology: 'parallel'
});
```

---

## Common Objections

| Objection | Response |
|-----------|----------|
| "Pairing is 2x slower" | 15% slower writing, 15% fewer bugs, net positive |
| "No time for TDD" | Debugging takes longer than testing |
| "CI is hard to setup" | Start simple: one action, one test |
| "Collective ownership = chaos" | Only without tests + CI |

---

## Related Skills
- [tdd-london-chicago](../tdd-london-chicago/) - TDD deep dive
- [refactoring-patterns](../refactoring-patterns/) - Safe refactoring
- [pair-programming](../pair-programming/) - AI-assisted pairing

---

## Remember

**XP practices work as a system.** Don't cherry-pick randomly:
- TDD enables collective ownership
- CI enables small releases
- Pairing enables collective ownership
- Sustainable pace enables everything

**With Agents:** Agents amplify XP. Pair humans with agents. Agents handle repetitive work, humans provide judgment and creativity.
