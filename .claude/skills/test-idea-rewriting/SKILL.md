---
name: test-idea-rewriting
description: "Transform passive 'Verify X' test descriptions into active, observable test actions. Use when test ideas lack specificity, use vague language, or fail quality validation. Converts to action-verb format for clearer, more testable descriptions."
category: test-design
priority: medium
tokenEstimate: 800
agents: [qe-test-idea-rewriter]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2026-01-17
dependencies: []
quick_reference_card: true
tags: [test-ideas, rewriting, action-verbs, quality, transformation]
---

# Test Idea Rewriting

<default_to_action>
When transforming test ideas:
1. DETECT all "Verify X" patterns via regex
2. IDENTIFY appropriate action verb category
3. TRANSFORM to "[ACTION] [trigger]; [OBSERVE] [result]" pattern
4. PRESERVE all metadata (IDs, priorities, automation types)
5. VALIDATE zero "Verify" patterns remain
6. OUTPUT in same format as input

**Success Criteria:** `/<td>Verify\s/gi` returns 0 matches
</default_to_action>

## Quick Reference Card

### Transformation Pattern

```
[ACTION VERB] [specific trigger]; [OUTCOME VERB] [observable result]
```

### Action Verb Quick Reference

| Category | Verbs | Use When |
|----------|-------|----------|
| **Interaction** | Click, Type, Submit, Navigate, Scroll | UI actions |
| **Trigger** | Send, Inject, Force, Simulate, Load | API/system actions |
| **Measurement** | Measure, Time, Count, Profile | Performance checks |
| **State** | Set, Configure, Enable, Disable, Toggle | Setup actions |
| **Observation** | Confirm, Assert, Check, Observe | Outcome verification |

### Common Transformations

| Before | After |
|--------|-------|
| Verify login works | Submit valid credentials; confirm session created |
| Verify API returns 200 | Send GET request; assert 200 response within 500ms |
| Verify error displays | Trigger validation error; observe error message |
| Verify data saves | Insert record; query database; confirm fields match |
| Verify performance | Execute 100 requests; measure p99 < 200ms |

---

## Transformation Rules

### Pattern Detection

```regex
/<td>Verify\s/gi     // HTML table cells
/^Verify\s/gim       // Line starts
/"Verify\s[^"]+"/gi  // Quoted strings
```

### Transformation Categories

#### API/Network Tests

| Input Pattern | Output Pattern |
|---------------|----------------|
| Verify API returns X | Send [METHOD] request; assert [STATUS] response |
| Verify endpoint accepts Y | Post [PAYLOAD] to endpoint; confirm [RESPONSE] |
| Verify webhook fires | Trigger [EVENT]; observe webhook received |

#### UI/UX Tests

| Input Pattern | Output Pattern |
|---------------|----------------|
| Verify button works | Click [BUTTON]; observe [EFFECT] |
| Verify form submits | Fill [FIELDS]; submit form; confirm [RESULT] |
| Verify navigation works | Click [LINK]; observe [PAGE] loads |

#### Data Tests

| Input Pattern | Output Pattern |
|---------------|----------------|
| Verify data saves | Insert [RECORD]; query; confirm [MATCH] |
| Verify validation works | Enter [INVALID]; observe [ERROR] |
| Verify format accepted | Submit [FORMAT]; confirm [PROCESSED] |

#### Performance Tests

| Input Pattern | Output Pattern |
|---------------|----------------|
| Verify performance is good | Execute [LOAD]; measure [METRIC] < [THRESHOLD] |
| Verify scalability | Increase [USERS] to [N]; monitor [RESOURCE] |
| Verify timeout works | Inject [DELAY]; confirm timeout after [TIME] |

---

## Action Verb Reference

### Interaction Verbs

| Verb | When to Use | Example |
|------|-------------|---------|
| Click | UI element interaction | Click "Submit" button |
| Type | Text entry | Type "user@example.com" |
| Submit | Form completion | Submit registration form |
| Navigate | Page changes | Navigate to /settings |
| Scroll | Viewport movement | Scroll to page bottom |
| Drag | Drag-and-drop | Drag file to upload zone |
| Hover | Mouse positioning | Hover over tooltip trigger |
| Select | Dropdown/checkbox | Select "Admin" from role dropdown |

### Trigger Verbs

| Verb | When to Use | Example |
|------|-------------|---------|
| Send | HTTP requests | Send POST to /api/orders |
| Inject | Fault injection | Inject 500ms latency |
| Force | State manipulation | Force offline mode |
| Simulate | Event generation | Simulate device rotation |
| Load | Resource loading | Load 50MB test file |
| Execute | Script/command | Execute database migration |
| Invoke | Function/webhook | Invoke payment callback |
| Trigger | Event firing | Trigger scheduled job |

### Measurement Verbs

| Verb | When to Use | Example |
|------|-------------|---------|
| Measure | Quantitative check | Measure response time |
| Time | Duration tracking | Time page render |
| Count | Quantity check | Count search results |
| Profile | Resource analysis | Profile CPU usage |
| Benchmark | Comparison | Benchmark against v1.0 |
| Capture | State recording | Capture network traffic |
| Monitor | Ongoing observation | Monitor memory for 5 minutes |

### Observation Verbs

| Verb | When to Use | Example |
|------|-------------|---------|
| Confirm | Boolean check | Confirm user is logged in |
| Assert | Value comparison | Assert total equals $99.99 |
| Check | State verification | Check cart has 3 items |
| Observe | Behavior watching | Observe spinner appears |
| Validate | Rule compliance | Validate email format |
| Expect | Predicted outcome | Expect redirect to /home |
| Verify (avoid) | Use alternatives | Use confirm/assert instead |

---

## Quality Validation

### Pre-Transform Checks

1. Count "Verify" patterns in input
2. Identify context for each pattern
3. Map to appropriate action verb category

### Post-Transform Checks

1. Regex validation: zero "Verify" matches
2. Every test idea starts with action verb
3. Each test includes observable outcome
4. All metadata preserved unchanged

### Validation Regex

```javascript
// Must return 0 matches for success
const verifyPattern = /<td>Verify\s/gi;
const matches = content.match(verifyPattern);
if (matches && matches.length > 0) {
  throw new Error(`${matches.length} "Verify" patterns remain`);
}
```

---

## Agent Integration

```typescript
// Single file transformation
await Task("Rewrite Test Ideas", {
  inputFile: "assessment.html",
  outputFile: "assessment-rewritten.html",
  preserveFormatting: true
}, "qe-test-idea-rewriter");

// Batch transformation
await Task("Batch Rewrite", {
  inputDir: "./assessments/",
  outputDir: "./assessments-clean/",
  pattern: "*.html"
}, "qe-test-idea-rewriter");
```

---

## Memory Namespace

```
aqe/rewriting/
├── transformations/*  - Transformation logs
├── patterns/*         - Learned patterns
└── vocabulary/*       - Custom verb mappings
```

---

## Related Skills

- [sfdipot-product-factors](../sfdipot-product-factors/) - Assessment generation
- [test-design-techniques](../test-design-techniques/) - Proper test structuring
- [brutal-honesty-review](../brutal-honesty-review/) - Quality validation

---

## Remember

**Every test idea should be actionable.** "Verify X works" tells you nothing about HOW to test. "[Action] X; [Observe] Y" gives clear steps and expected outcomes. Transform passive descriptions into active, observable tests.
