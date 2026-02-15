---
name: context-driven-testing
description: "Apply context-driven testing principles where practices are chosen based on project context, not universal 'best practices'. Use when making testing decisions, questioning dogma, or adapting approaches to specific project needs."
category: testing-methodologies
priority: high
tokenEstimate: 1100
agents: [qe-fleet-commander, qe-regression-risk-analyzer, qe-requirements-validator, qe-quality-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [context-driven, rst, exploratory, heuristics, oracles, skilled-testing]
trust_tier: 0
validation:

---

# Context-Driven Testing

<default_to_action>
When making testing decisions or adapting approaches:
1. ANALYZE context: project goals, constraints, risks, team skills
2. QUESTION practices: "Why this? What risk does it address? What's the cost?"
3. INVESTIGATE not just check: Does software solve the problem, or create new ones?
4. ADAPT approach based on context, not "best practices"
5. DOCUMENT discoveries, not pre-written plans

**Quick Context Analysis:**
- Mission: "Find important problems fast enough to matter" (not "execute test cases")
- Risk: Safety-critical = high rigor; internal tool = lighter touch
- Constraints: Startup with tight timeline ≠ enterprise with compliance
- Skills: Novice needs structure; expert adapts intuitively

**Critical Success Factors:**
- No "best practices" work everywhere - only good practices in context
- Testing is investigation, not script execution
- Context changes; your approach should too
</default_to_action>

## Quick Reference Card

### When to Use
- Making testing decisions for new project
- Questioning "that's how it's done" dogma
- Adapting approach to specific constraints
- Exploratory testing sessions

### Seven Context-Driven Principles
1. Value of any practice depends on its context
2. Good practices in context, no universal best practices
3. People working together are most important
4. Projects unfold in unpredictable ways
5. Product is a solution - if problem not solved, product fails
6. Good testing is challenging intellectual work
7. Judgment and skill determine right things at right times

### Context Factors
| Factor | Questions |
|--------|-----------|
| **Project** | Business goal? User needs? Failure impact? |
| **Constraints** | Timeline? Budget? Team skills? Legacy? |
| **Risk** | Safety-critical? Regulated? High volume? |
| **Technical** | Stack quirks? Integrations? Observability? |

### RST Heuristics
| Heuristic | Application |
|-----------|-------------|
| **SFDIPOT** | Structure, Function, Data, Interfaces, Platform, Operations, Time |
| **Oracles** | Consistency with history, similar products, expectations, docs |
| **Tours** | Business District, Historical, Bad Neighborhood, Tourist, Museum |

---

## Context-Driven Decisions

### Example: Test Automation Level

**Startup Context:**
- Small team, rapid changes, unclear product-market fit
- **Decision:** Light automation on critical paths, heavy exploratory
- **Rationale:** Requirements change too fast for extensive automation

**Enterprise Context:**
- Stable features, regulatory requirements, large team
- **Decision:** Comprehensive automated regression suite
- **Rationale:** Stability allows automation investment to pay off

### Example: Documentation

**Regulated (FDA/medical):**
- **Decision:** Detailed test protocols, traceability matrices
- **Rationale:** Regulatory compliance isn't optional

**Fast-paced startup:**
- **Decision:** Lightweight session notes, risk logs
- **Rationale:** Bureaucracy slows more than it helps

---

## Investigation vs. Checking

| Checking | Testing (Investigation) |
|----------|------------------------|
| Did API return 200? | Does API meet user needs? |
| Does button work? | What happens under load? |
| Match the spec? | Does it solve the problem? |

---

## Red Flags: Not Context-Driven

- Follow process "because that's how it's done"
- Can't explain *why* you're doing something
- Measure test cases executed, not problems found
- Test plan could apply to any project
- Stop thinking once you have a script

---

## Agent-Assisted Context-Driven Testing

```typescript
// Agent analyzes context and recommends approach
const context = await Task("Analyze Context", {
  project: 'e-commerce-platform',
  stage: 'startup',
  constraints: ['timeline: tight', 'budget: limited'],
  risks: ['payment-security', 'high-volume']
}, "qe-fleet-commander");

// Context-aware agent selection
// - qe-security-scanner (critical risk)
// - qe-performance-tester (high volume)
// - Skip: qe-visual-tester (low priority in startup context)

// Adaptive testing strategy
await Task("Generate Tests", {
  context: 'startup',
  focus: 'critical-paths-only',
  depth: 'smoke-tests',
  automation: 'minimal'
}, "qe-test-generator");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/context-driven/
├── context-analysis/*    - Project context snapshots
├── decisions/*           - Testing decisions with rationale
├── discoveries/*         - What was learned during testing
└── adaptations/*         - How approach changed over time
```

### Fleet Coordination
```typescript
const contextFleet = await FleetManager.coordinate({
  strategy: 'context-driven',
  context: {
    type: 'greenfield-saas',
    stage: 'growth',
    compliance: 'gdpr-only'
  },
  agents: ['qe-test-generator', 'qe-security-scanner', 'qe-performance-tester'],
  exclude: ['qe-visual-tester', 'qe-requirements-validator']  // Not priority
});
```

---

## Practical Tips

1. **Start with risk assessment** - List features, ask: How likely to fail? How bad? How hard to test?
2. **Time-box exploration** - 2 hours checkout, 30 min error handling, 15 min per browser
3. **Document discoveries** - Not "Enter invalid email, verify error" but "Payment API returns 500 instead of 400, no user-visible error. Bug filed."
4. **Talk to humans** - Developers, users, support, product
5. **Pair with others** - Different perspectives = different bugs

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - Context-aware agent selection
- [holistic-testing-pact](../holistic-testing-pact/) - Adapt holistic model to context
- [risk-based-testing](../risk-based-testing/) - Context affects risk assessment
- [exploratory-testing-advanced](../exploratory-testing-advanced/) - RST techniques

---

## Remember

**Context drives decisions.** No universal best practices. Skilled testers make informed decisions based on specific goals, constraints, and risks.

You're not a test script executor. You're a skilled investigator helping teams build better products.

**With Agents:** Agents analyze context, adapt strategies, and learn what works in your situation. Use agents to scale context-driven thinking while maintaining human judgment for critical decisions.
