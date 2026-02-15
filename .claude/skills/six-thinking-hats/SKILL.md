---
name: six-thinking-hats
description: "Apply Edward de Bono's Six Thinking Hats methodology to software testing for comprehensive quality analysis. Use when designing test strategies, conducting test retrospectives, analyzing test failures, evaluating testing approaches, or facilitating testing discussions. Each hat provides a distinct testing perspective: facts (White), risks (Black), benefits (Yellow), creativity (Green), emotions (Red), and process (Blue)."
category: methodology
priority: medium
tokenEstimate: 1100
agents: [qe-quality-analyzer, qe-regression-risk-analyzer, qe-test-generator]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [thinking, methodology, decision-making, collaboration, analysis]
trust_tier: 0
validation:

---

# Six Thinking Hats for Testing

<default_to_action>
When analyzing testing decisions:
1. DEFINE focus clearly (specific testing question)
2. APPLY each hat sequentially (5 min each)
3. DOCUMENT insights per hat
4. SYNTHESIZE into action plan

**Quick Hat Rotation (30 min):**
```markdown
🤍 WHITE (5 min) - Facts only: metrics, data, coverage
❤️ RED (3 min) - Gut feelings (no justification needed)
🖤 BLACK (7 min) - Risks, gaps, what could go wrong
💛 YELLOW (5 min) - Strengths, opportunities, what works
💚 GREEN (7 min) - Creative ideas, alternatives
🔵 BLUE (3 min) - Action plan, next steps
```

**Example for "API Test Strategy":**
- 🤍 47 endpoints, 30% coverage, 12 integration tests
- ❤️ Anxious about security, confident on happy paths
- 🖤 No auth tests, rate limiting untested, edge cases missing
- 💛 Good docs, CI/CD integrated, team experienced
- 💚 Contract testing with Pact, chaos testing, property-based
- 🔵 Security tests first, contract testing next sprint
</default_to_action>

## Quick Reference Card

### The Six Hats

| Hat | Focus | Key Question |
|-----|-------|--------------|
| 🤍 **White** | Facts & Data | What do we KNOW? |
| ❤️ **Red** | Emotions | What do we FEEL? |
| 🖤 **Black** | Risks | What could go WRONG? |
| 💛 **Yellow** | Benefits | What's GOOD? |
| 💚 **Green** | Creativity | What ELSE could we try? |
| 🔵 **Blue** | Process | What should we DO? |

### When to Use Each Hat

| Hat | Use For |
|-----|---------|
| 🤍 White | Baseline metrics, test data inventory |
| ❤️ Red | Team confidence check, quality gut feel |
| 🖤 Black | Risk assessment, gap analysis, pre-mortems |
| 💛 Yellow | Strengths audit, quick win identification |
| 💚 Green | Test innovation, new approaches, brainstorming |
| 🔵 Blue | Strategy planning, retrospectives, decision-making |

---

## Hat Details

### 🤍 White Hat - Facts & Data
**Output: Quantitative testing baseline**

Questions:
- What test coverage do we have?
- What is our pass/fail rate?
- What environments exist?
- What is our defect history?

```
Example Output:
Coverage: 67% line, 45% branch
Test Suite: 1,247 unit, 156 integration, 23 E2E
Execution Time: Unit 3min, Integration 12min, E2E 45min
Defects: 23 open (5 critical, 8 major, 10 minor)
```

### 🖤 Black Hat - Risks & Cautions
**Output: Comprehensive risk assessment**

Questions:
- What could go wrong in production?
- What are we NOT testing?
- What assumptions might be wrong?
- Where are the coverage gaps?

```
HIGH RISKS:
- No load testing (production outage risk)
- Auth edge cases untested (security vulnerability)
- Database failover never tested (data loss risk)
```

### 💛 Yellow Hat - Benefits & Optimism
**Output: Strengths and opportunities**

Questions:
- What's working well?
- What strengths can we leverage?
- What quick wins are available?

```
STRENGTHS:
- Strong CI/CD pipeline
- Team expertise in automation
- Stakeholders value quality

QUICK WINS:
- Add smoke tests (reduce incidents)
- Automate manual regression (save 2 days/release)
```

### 💚 Green Hat - Creativity
**Output: Innovative testing ideas**

Questions:
- How else could we test this?
- What if we tried something completely different?
- What emerging techniques could we adopt?

```
IDEAS:
1. AI-powered test generation
2. Chaos engineering for resilience
3. Property-based testing for edge cases
4. Production traffic replay
5. Synthetic monitoring
```

### ❤️ Red Hat - Emotions
**Output: Team gut feelings (NO justification needed)**

Questions:
- How confident do you feel about quality?
- What makes you anxious?
- What gives you confidence?

```
FEELINGS:
- Confident: Unit tests, API tests
- Anxious: Authentication flow, payment processing
- Frustrated: Flaky tests, slow E2E suite
```

### 🔵 Blue Hat - Process
**Output: Action plan with owners and timelines**

Questions:
- What's our strategy?
- How should we prioritize?
- What's the next step?

```
PRIORITIZED ACTIONS:
1. [Critical] Address security testing gap - Owner: Alice
2. [High] Implement contract testing - Owner: Bob
3. [Medium] Reduce flaky tests - Owner: Carol
```

---

## Session Templates

### Solo Session (30 min)
```markdown
# Six Hats Analysis: [Topic]

## 🤍 White Hat (5 min)
Facts: [list metrics, data]

## ❤️ Red Hat (3 min)
Feelings: [gut reactions, no justification]

## 🖤 Black Hat (7 min)
Risks: [what could go wrong]

## 💛 Yellow Hat (5 min)
Strengths: [what works, opportunities]

## 💚 Green Hat (7 min)
Ideas: [creative alternatives]

## 🔵 Blue Hat (3 min)
Actions: [prioritized next steps]
```

### Team Session (60 min)
- Each hat: 10 minutes
- Rotate through hats as group
- Document on shared whiteboard
- Blue Hat synthesizes at end

---

## Agent Integration

```typescript
// Risk-focused analysis (Black Hat)
const risks = await Task("Identify Risks", {
  scope: 'payment-module',
  perspective: 'black-hat',
  includeMitigation: true
}, "qe-regression-risk-analyzer");

// Creative test approaches (Green Hat)
const ideas = await Task("Generate Test Ideas", {
  feature: 'new-auth-system',
  perspective: 'green-hat',
  includeEmergingTechniques: true
}, "qe-test-generator");

// Comprehensive analysis (All Hats)
const analysis = await Task("Six Hats Analysis", {
  topic: 'Q1 Test Strategy',
  hats: ['white', 'black', 'yellow', 'green', 'red', 'blue']
}, "qe-quality-analyzer");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/six-hats/
├── analyses/*        - Complete hat analyses
├── risks/*           - Black hat findings
├── opportunities/*   - Yellow hat findings
└── innovations/*     - Green hat ideas
```

### Fleet Coordination
```typescript
const analysisFleet = await FleetManager.coordinate({
  strategy: 'six-hats-analysis',
  agents: [
    'qe-quality-analyzer',        // White + Blue hats
    'qe-regression-risk-analyzer', // Black hat
    'qe-test-generator'           // Green hat
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [risk-based-testing](../risk-based-testing/) - Black Hat deep dive
- [exploratory-testing-advanced](../exploratory-testing-advanced/) - Green Hat exploration
- [context-driven-testing](../context-driven-testing/) - Adapt to context

---

## Anti-Patterns

| ❌ Avoid | Why | ✅ Instead |
|----------|-----|-----------|
| Mixing hats | Confuses thinking | One hat at a time |
| Justifying Red Hat | Kills intuition | State feelings only |
| Skipping hats | Misses insights | Use all six |
| Rushing | Shallow analysis | 5 min minimum per hat |

---

## Remember

**Separate thinking modes for clarity.** Each hat reveals different insights. Red Hat intuition often catches what Black Hat analysis misses.

**Everyone wears all hats.** This is parallel thinking, not role-based. The goal is comprehensive analysis, not debate.
