---
name: bug-reporting-excellence
description: "Write high-quality bug reports that get fixed quickly. Use when reporting bugs, training teams on bug reporting, or establishing bug report standards."
category: quality-communication
priority: high
tokenEstimate: 900
agents: [qe-quality-analyzer, qe-production-intelligence, qe-flaky-test-hunter]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [bugs, reporting, communication, triage, quality]
trust_tier: 2
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json

---

# Bug Reporting Excellence

<default_to_action>
When reporting bugs:
1. TITLE: `[Component] fails [Condition] causing [Impact]`
2. DESCRIBE: Expected behavior → Actual behavior → Steps to reproduce
3. INCLUDE: Environment, severity, screenshots/logs, business impact
4. ISOLATE: Narrow down conditions (browser, user, amount thresholds)
5. ONE BUG = ONE REPORT (don't combine issues)

**Bug Report Formula:**
```markdown
## [Component] Issue Title
**Severity:** Critical/High/Medium/Low
**Environment:** Production/Staging/Dev

### Expected Behavior
What should happen

### Actual Behavior
What actually happens (with error messages)

### Steps to Reproduce
1. Step one
2. Step two
3. Observe issue

### Impact
How this affects users/business
```

**Critical Success Factors:**
- Reproducible steps (100%)
- Environment info (100%)
- Business impact stated (90%)
- Screenshots or logs (80%)
</default_to_action>

## Quick Reference Card

### Severity Levels

| Level | Definition | Examples |
|-------|------------|----------|
| **Critical** | System down, data loss, security | DB deleted, payments broken, credentials exposed |
| **High** | Major feature broken, many users | Can't checkout, search broken, dashboard fails |
| **Medium** | Partial break, workaround exists | Filter broken (refresh works), slow export |
| **Low** | Cosmetic, rare edge case | Button wraps on mobile, tooltip wrong color |

### Title Formula
❌ Bad: "Checkout broken"
✅ Good: "Payment fails with Visa cards when order total > $1000"

**Pattern:** `[Component] fails [Condition] causing [Impact]`

---

## Essential Information

### Environment Details
```
Browser: Chrome 120.0.6099.109 (Windows)
OS: Windows 11 Pro
URL: https://example.com/checkout
Date/Time: 2025-10-17 14:23 UTC
User: test@example.com (ID: 12345)
Request ID: abc-123-def-456
```

### Supporting Evidence

**Error Messages:**
```json
{
  "error": "Payment service unavailable",
  "code": "GATEWAY_TIMEOUT",
  "requestId": "abc-123-def-456"
}
```

**Console Logs:**
```
[ERROR] PaymentGateway: Connection timeout after 30000ms
  at PaymentGateway.charge (gateway.js:145)
```

---

## Example: Excellent Bug Report

```markdown
## [Checkout] Payment processing times out for orders > $1000

**Severity:** High
**Environment:** Production
**Affected Users:** ~15% of premium purchases

### Expected Behavior
Payment completes within 5 seconds regardless of amount.

### Actual Behavior
For orders above $1000, payment gateway times out after 30 seconds.
User sees "Payment failed" error. Order not created.

### Steps to Reproduce
1. Add items totaling $1,050 to cart
2. Proceed to checkout
3. Enter payment: Visa 4532 1234 5678 9010
4. Click "Place Order"
5. Wait 30+ seconds
6. Observe timeout error

### Environment
- Browser: Chrome 120 (Windows 11)
- User: test@example.com
- Request ID: abc-123-def-456

### Evidence
Console error: `PaymentGateway timeout: 30000ms exceeded`
Network: /api/checkout: 30.14s (timeout)

### Impact
- Lost revenue: ~$15K/week from failed orders
- 23 support tickets this week
- Affects 15% of orders over $1000

### Additional Context
Started after Oct 15 deployment (v2.3.0)
Possibly related to PR #456 (fraud check)
```

---

## Anti-Patterns

| ❌ Bad | Problem | ✅ Good |
|--------|---------|--------|
| "Checkout is broken" | What doesn't work? | "Payment button doesn't respond when clicked" |
| "I saw an error" | No reproduction steps | Full steps with conditions |
| "Page loads slowly" | No specifics | "Dashboard takes 12s to load (should be <3s)" |
| Multiple bugs in one | Can't track separately | One report per bug |

---

## Agent Integration

```typescript
// Automated bug triage
const triage = await Task("Triage Bug", {
  title: 'Payment fails for orders > $1000',
  description: bugDescription,
  steps: reproductionSteps
}, "qe-quality-analyzer");

// Returns: { severity, priority, component, suggestedAssignee, relatedIssues }

// Duplicate detection
const dupeCheck = await Task("Check Duplicates", {
  bugReport: newBug,
  similarityThreshold: 0.85
}, "qe-quality-analyzer");

// Bug report enhancement
const enhanced = await Task("Enhance Report", {
  originalReport: userSubmittedBug,
  addMissingInfo: true,
  identifyRootCause: true
}, "qe-production-intelligence");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/bug-reports/
├── triaged/*          - Bug triage results
├── duplicates/*       - Duplicate detection
├── patterns/*         - Recurring bug patterns
└── root-cause/*       - Root cause analyses
```

### Fleet Coordination
```typescript
const bugFleet = await FleetManager.coordinate({
  strategy: 'bug-investigation',
  agents: [
    'qe-quality-analyzer',        // Triage and categorize
    'qe-flaky-test-hunter',       // Check if test-related
    'qe-production-intelligence'  // Check production logs
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [technical-writing](../technical-writing/) - Clear bug documentation
- [exploratory-testing-advanced](../exploratory-testing-advanced/) - Finding bugs
- [sherlock-review](../sherlock-review/) - Root cause investigation

---

## Remember

Your bug report is the starting point for someone else's work. Make it **complete** (all info needed), **clear** (anyone can follow), **concise** (no noise), and **actionable** (developer knows next step).

**Good bug reports = Faster fixes = Better product = Happier users**
