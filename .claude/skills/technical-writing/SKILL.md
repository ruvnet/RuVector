---
name: technical-writing
description: "Write clear, engaging technical content from real experience. Use when writing blog posts, documentation, tutorials, or technical articles."
category: communication
priority: medium
tokenEstimate: 800
agents: [qe-quality-analyzer, qe-api-contract-validator]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [writing, documentation, communication, blogs, tutorials]
trust_tier: 1
validation:
  schema_path: schemas/output.json

---

# Technical Writing

<default_to_action>
When writing technical content:
1. LEAD with value (what will reader learn/gain?)
2. SHOW, don't tell (specific examples, code, numbers)
3. STRUCTURE for scanning (headers, bold, short paragraphs)
4. CUT ruthlessly (every sentence must earn its place)
5. BE honest about trade-offs

**Blog Post Structure:**
```markdown
# Title (specific promise)

## Opening (2-3 paragraphs)
- Hook: The problem or insight
- Context: Why this matters
- Promise: What they'll learn

## Body (3-5 sections)
- One clear idea per section
- Support with examples/code/data

## Closing
- Key takeaway (1-2 sentences)
- Action reader can take
```

**Before/After:**
❌ "We implemented a comprehensive testing strategy..."
✅ "We moved exploratory testing into sprint planning. QE now pairs with devs during story refinement."
</default_to_action>

## Quick Reference Card

### Core Principles

| Principle | Bad | Good |
|-----------|-----|------|
| **Lead with value** | "In today's landscape..." | "Here's how we cut bugs 60%" |
| **Show, don't tell** | "We improved testing" | "Bug detection: 12→47 per sprint" |
| **Be specific** | "Performance improved" | "Response time: 2.3s→180ms" |
| **Honest trade-offs** | "This approach is best" | "TDD slowed velocity 20%, reduced bugs 75%" |

### Words to Cut

| Kill | Reason |
|------|--------|
| basically, actually, probably | Hedge words |
| leverage, synergy, paradigm | Corporate speak |
| very, really, quite | Unnecessary qualifiers |
| it should be noted that | Just note it |

---

## Audience-Specific Writing

### For Developers
- Lead with code or concrete problem
- Show implementation details
- Discuss trade-offs and alternatives
- Link to repos or examples

### For QA/QE
- Start with testing challenge
- Show strategy, not just tools
- Include risk assessment
- Provide adaptable heuristics

### For Leadership
- Open with business impact
- Use metrics that matter
- Connect technical to outcomes
- Keep technical details concise

---

## Editing Checklist

Before publishing:
- [ ] Title promises something specific
- [ ] Opening hooks in 30 seconds
- [ ] Claims backed by examples
- [ ] All unnecessary words cut
- [ ] Code examples tested and correct
- [ ] Takeaway crystal clear
- [ ] Would send to respected colleague

---

## Example Transformations

**Before:**
"We decided to implement a more comprehensive testing strategy that would allow us to catch bugs earlier in the development lifecycle."

**After:**
"We moved exploratory testing into sprint planning. QE now pairs with devs during story refinement, identifying risks before code is written."

---

**Before:**
"The benefits of this approach are numerous and include improved quality, faster feedback loops, and better team collaboration."

**After:**
"Three outcomes: bugs found 2 days earlier on average, 30% fewer regression issues, and devs now ask QE for input during design."

---

## Agent Integration

```typescript
// Generate documentation from code
const docs = await Task("Generate Docs", {
  source: 'src/services/PaymentService.ts',
  format: 'markdown',
  includeExamples: true
}, "qe-quality-analyzer");

// Review documentation quality
const review = await Task("Review Docs", {
  files: ['README.md', 'docs/api.md'],
  checkClarity: true,
  checkCodeExamples: true
}, "qe-quality-analyzer");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/technical-writing/
├── generated-docs/*   - Auto-generated documentation
├── reviews/*          - Documentation review findings
└── templates/*        - Reusable doc templates
```

### Fleet Coordination
```typescript
const docsFleet = await FleetManager.coordinate({
  strategy: 'documentation',
  agents: [
    'qe-quality-analyzer',         // Generate and review
    'qe-api-contract-validator'    // API doc accuracy
  ],
  topology: 'sequential'
});
```

---

## Related Skills
- [bug-reporting-excellence](../bug-reporting-excellence/) - Technical bug writing
- [code-review-quality](../code-review-quality/) - Review documentation

---

## Remember

**You're not writing to impress.** You're writing to help people solve problems you've already solved. Be the colleague you wish you'd had.

**Write from experience.** Only write about what you've done in production. If exploring, say so.
