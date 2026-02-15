---
name: test-reporting-analytics
description: "Advanced test reporting, quality dashboards, predictive analytics, trend analysis, and executive reporting for QE metrics. Use when communicating quality status, tracking trends, or making data-driven decisions."
category: analytics
priority: high
tokenEstimate: 850
agents: [qe-quality-analyzer, qe-quality-gate, qe-deployment-readiness]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-03
dependencies: []
quick_reference_card: true
tags: [reporting, analytics, dashboards, metrics, trends, predictive]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/test-reporting-analytics.yaml

---

# Test Reporting & Analytics

<default_to_action>
When building test reports:
1. DEFINE audience (dev team vs executives)
2. CHOOSE key metrics (max 5-7)
3. SHOW trends (not just snapshots)
4. HIGHLIGHT actions (what to do about it)
5. AUTOMATE generation

**Dashboard Quick Setup:**
```
+------------------+------------------+------------------+
| Tests Passed     | Code Coverage    | Flaky Tests      |
| 1,247/1,250 ✅   | 82.3% ⬆️ +2.1%  | 1.2% ⬇️ -0.3%   |
+------------------+------------------+------------------+
| Critical Bugs    | Deploy Freq      | MTTR             |
| 0 open ✅        | 12x/day ⬆️       | 2.3h ⬇️          |
+------------------+------------------+------------------+
```

**Key Metrics by Audience:**
- **Dev Team**: Pass rate, flaky %, execution time, coverage gaps
- **QE Team**: Defect detection rate, test velocity, automation ROI
- **Leadership**: Escaped defects, deployment frequency, quality cost
</default_to_action>

## Quick Reference Card

### Essential Metrics

| Category | Metric | Target |
|----------|--------|--------|
| **Execution** | Pass Rate | >98% |
| **Execution** | Flaky Test % | <2% |
| **Execution** | Suite Duration | <10 min |
| **Coverage** | Line Coverage | >80% |
| **Coverage** | Branch Coverage | >70% |
| **Quality** | Escaped Defects | <5/release |
| **Quality** | MTTR | <4 hours |
| **Efficiency** | Automation Rate | >90% |

### Trend Indicators

| Symbol | Meaning | Action |
|--------|---------|--------|
| ⬆️ | Improving | Continue current approach |
| ⬇️ | Declining | Investigate root cause |
| ➡️ | Stable | Maintain or improve |
| ⚠️ | Threshold breach | Immediate attention |

---

## Report Types

### Real-Time Dashboard
```
Live quality status for CI/CD
- Build status (green/red)
- Test results (pass/fail counts)
- Coverage delta
- Flaky test alerts
```

### Sprint Summary
```markdown
## Sprint 47 Quality Summary

### Metrics
| Metric | Value | Trend |
|--------|-------|-------|
| Tests Added | +47 | ⬆️ |
| Coverage | 82.3% | ⬆️ +2.1% |
| Bugs Found | 12 | ➡️ |
| Escaped | 0 | ✅ |

### Highlights
- ✅ Zero escaped defects
- ⚠️ E2E suite now 45min (target: 30min)

### Actions
1. Optimize slow E2E tests
2. Add coverage for payment module
```

### Executive Report
```markdown
## Monthly Quality Report - Oct 2025

### Executive Summary
✅ Production uptime: 99.97% (target: 99.95%)
✅ Deploy frequency: 12x/day (up from 8x)
⚠️ Coverage: 82.3% (target: 85%)

### Business Impact
- Automation saves 120 hrs/month
- Bug cost: $150/bug found vs $5,000 escaped
- Estimated annual savings: $450K

### Recommendations
1. Invest in performance testing tooling
2. Hire senior QE for mobile coverage
```

---

## Predictive Analytics

```typescript
// Predict test failures
const prediction = await Task("Predict Failures", {
  codeChanges: prDiff,
  historicalData: last90Days,
  model: 'gradient-boosting'
}, "qe-quality-analyzer");

// Returns:
// {
//   failureProbability: 0.73,
//   likelyFailingTests: ['payment.test.ts'],
//   suggestedAction: 'Review payment module carefully',
//   confidence: 0.89
// }

// Trend analysis with anomaly detection
const trends = await Task("Analyze Trends", {
  metrics: ['passRate', 'coverage', 'flakyRate'],
  period: '30d',
  detectAnomalies: true
}, "qe-quality-analyzer");
```

---

## Agent Integration

```typescript
// Generate comprehensive quality report
const report = await Task("Generate Quality Report", {
  period: 'sprint',
  audience: 'executive',
  includeROI: true,
  includeTrends: true
}, "qe-quality-analyzer");

// Real-time quality gate check
const gateResult = await Task("Quality Gate Check", {
  metrics: currentMetrics,
  thresholds: qualityPolicy,
  environment: 'production'
}, "qe-quality-gate");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/reporting/
├── dashboards/*      - Dashboard configurations
├── reports/*         - Generated reports
├── trends/*          - Trend analysis data
└── predictions/*     - Predictive model outputs
```

### Fleet Coordination
```typescript
const reportingFleet = await FleetManager.coordinate({
  strategy: 'quality-reporting',
  agents: [
    'qe-quality-analyzer',      // Metrics aggregation
    'qe-quality-gate',          // Threshold validation
    'qe-deployment-readiness'   // Release readiness
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [quality-metrics](../quality-metrics/) - Metric definitions
- [shift-right-testing](../shift-right-testing/) - Production metrics
- [consultancy-practices](../consultancy-practices/) - Client reporting

---

## Remember

**Measure to improve. Report to communicate.**

Good reports:
- Answer "so what?" (actionable insights)
- Show trends (not just snapshots)
- Match audience needs
- Automate where possible

**Data without action is noise. Action without data is guessing.**
