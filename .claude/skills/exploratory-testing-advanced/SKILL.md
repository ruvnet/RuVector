---
name: exploratory-testing-advanced
description: "Advanced exploratory testing techniques with Session-Based Test Management (SBTM), RST heuristics, and test tours. Use when planning exploration sessions, investigating bugs, or discovering unknown quality risks."
category: testing-methodologies
priority: high
tokenEstimate: 1000
agents: [qe-flaky-test-hunter, qe-visual-tester, qe-quality-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [exploratory, sbtm, rst, heuristics, test-tours, session-based]
trust_tier: 0
validation:

---

# Advanced Exploratory Testing

<default_to_action>
When exploring software or investigating quality risks:
1. CREATE charter with mission, scope, and time-box (45-90 min)
2. APPLY heuristics: SFDIPOT (quality criteria), FEW HICCUPPS (consistency oracles)
3. EXPLORE systematically using test tours (Business District, Bad Neighborhood, Historical)
4. DOCUMENT findings in real-time with notes, screenshots, evidence
5. DEBRIEF: What learned? What's next? Share via agent memory

**Quick Heuristic Selection:**
- What to test → SFDIPOT (Structure, Function, Data, Interfaces, Platform, Operations, Time)
- Recognize problems → FEW HICCUPPS (Familiar, Explainable, World, History, Image, Comparable, Claims, Users, Product, Purpose, Standards)
- Navigate app → Test Tours (12 types for different exploration strategies)

**Critical Success Factors:**
- Exploration is skilled, structured thinking - not random clicking
- Document discoveries, not pre-planned test cases
- Pair testing reveals more than solo exploration
</default_to_action>

## Quick Reference Card

### When to Use
- Investigating new or changed features
- Finding bugs automation misses
- Learning unfamiliar systems
- Risk discovery before test planning

### Session Structure (SBTM)
| Phase | Duration | Activity |
|-------|----------|----------|
| Charter | 5 min | Define mission, scope, focus |
| Explore | 45-75 min | Systematic investigation |
| Note | Continuous | Document findings real-time |
| Debrief | 10-15 min | Summarize, prioritize, share |

### SFDIPOT Heuristic (What to Test)
| Letter | Focus | Example Questions |
|--------|-------|------------------|
| **S**tructure | Is it properly composed? | Code structure, UI layout, data schema |
| **F**unction | Does it do what it should? | Core features work correctly |
| **D**ata | Handles data correctly? | CRUD, validation, persistence |
| **I**nterfaces | Interacts well? | APIs, UI, integrations |
| **P**latform | Works in environment? | Browsers, OS, devices |
| **O**perations | Can be used/managed? | Install, config, monitor |
| **T**ime | Handles timing? | Concurrency, timeouts, scheduling |

### FEW HICCUPPS Oracle (Recognize Problems)
| Consistency With | Check |
|-----------------|-------|
| **F**amiliar problems | Does this look like a known bug pattern? |
| **E**xplainable | Can behavior be explained rationally? |
| **W**orld | Matches real-world expectations? |
| **H**istory | Consistent with prior versions? |
| **I**mage | Matches brand/product image? |
| **C**omparable | Similar to competing products? |
| **C**laims | Matches specs/docs/marketing? |
| **U**sers | Meets user expectations? |
| **P**urpose | Fulfills intended purpose? |
| **S**tatements | Matches what devs said? |

### Test Tours (12 Types)
| Tour | Strategy |
|------|----------|
| Business District | Critical business flows |
| Historical | Where bugs clustered before |
| Bad Neighborhood | Known problem areas |
| Money | Revenue-impacting features |
| Landmark | Navigate by key features |
| Intellectual | Complex, thinking-intensive features |
| FedEx | Follow data through system |
| Garbage Collector | Cleanup and edge cases |
| Museum | Help docs and examples |
| Rained-Out | What happens when things fail? |
| Couch Potato | Minimal effort paths |
| Obsessive-Compulsive | Repetitive actions |

---

## Session Note Template

```markdown
**Charter:** Explore [area] to discover [what] focusing on [heuristic]
**Time-box:** 60 min | **Tester:** [name] | **Date:** [date]

## Session Notes
- [timestamp] Observation/finding
- [timestamp] Bug: [description] - [severity]
- [timestamp] Question: [unclear behavior]

## Findings Summary
- Bugs: X (Critical: Y, Major: Z)
- Questions: X
- Ideas: X

## Coverage
- Areas explored: [list]
- Heuristics used: [SFDIPOT areas]
- % Time on: Bug investigation 30%, Exploration 50%, Setup 20%

## Next Steps
- [ ] Deep dive on [area]
- [ ] Follow up on question about [topic]
```

---

## Agent-Assisted Exploration

```typescript
// Collaborative exploration session
await Task("Exploratory Session", {
  charter: 'Explore checkout flow for payment edge cases',
  duration: '60min',
  heuristics: ['SFDIPOT', 'FEW_HICCUPPS'],
  tour: 'money',
  collaboration: 'human-navigator-agent-driver'
}, "qe-flaky-test-hunter");

// Agent generates test variations while human observes
await Task("Edge Case Generation", {
  area: 'payment-form',
  variations: ['boundary-values', 'invalid-inputs', 'concurrent-submits']
}, "qe-test-generator");

// Visual exploration
await Task("Visual Exploration", {
  tour: 'landmark',
  focus: 'responsive-breakpoints',
  compare: 'baseline-screenshots'
}, "qe-visual-tester");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/exploratory/
├── sessions/*           - Session notes and findings
├── charters/*           - Reusable charter templates
├── bug-clusters/*       - Historical bug patterns
└── heuristic-results/*  - What heuristics revealed
```

### Fleet Coordination
```typescript
const exploratoryFleet = await FleetManager.coordinate({
  strategy: 'exploratory-testing',
  agents: [
    'qe-flaky-test-hunter',   // Pattern recognition
    'qe-visual-tester',       // Visual anomalies
    'qe-quality-analyzer'     // Risk assessment
  ],
  topology: 'mesh'
});
```

---

## Pairing Patterns

| Pattern | Human Role | Agent Role |
|---------|------------|------------|
| Driver-Navigator | Navigate strategy | Execute variations |
| Strong-Style | Dictate actions | Record findings |
| Ping-Pong | Observe one area | Explore another |

---

## Related Skills
- [context-driven-testing](../context-driven-testing/) - RST foundations
- [risk-based-testing](../risk-based-testing/) - Focus exploration on risk
- [agentic-quality-engineering](../agentic-quality-engineering/) - Agent coordination

---

## Remember

**Exploratory testing = simultaneous learning, test design, and test execution.**

Not random clicking. Structured, skilled investigation guided by heuristics and oracles. Document discoveries in real-time. Pair testing amplifies findings.

**With Agents:** Agents generate variations, recognize patterns, and maintain session notes while humans apply judgment and intuition. Combine agent thoroughness with human insight.
