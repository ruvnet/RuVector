---
name: security-visual-testing
description: "Security-first visual testing combining URL validation, PII detection, and visual regression with parallel viewport support. Use when testing web applications that handle sensitive data, need visual regression coverage, or require WCAG accessibility compliance."
category: specialized-testing
priority: high
tokenEstimate: 1400
agents: [qe-visual-tester, qe-security-scanner, qe-accessibility-auditor]
implementation_status: stable
optimization_version: 1.0
last_optimized: 2026-01-21
dependencies: [visual-testing-advanced, security-testing, accessibility-testing]
quick_reference_card: true
tags: [security, visual-testing, pii, accessibility, wcag, parallel-viewport, regression]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/security-visual-testing.yaml

---

# Security Visual Testing

<default_to_action>
When performing security-aware visual testing:
1. VALIDATE URLs before navigation (check for malicious patterns)
2. SCAN for PII before saving screenshots (mask sensitive data)
3. CAPTURE parallel viewports (mobile, tablet, desktop)
4. COMPARE against baselines (detect visual regressions)
5. AUDIT accessibility (WCAG 2.1 AA compliance)

**Quick Security-Visual Checklist:**
- URL validation (no javascript:, data:, file: schemes)
- PII detection (emails, phones, SSN, credit cards, API keys)
- Visual regression (diff threshold < 0.1%)
- Viewport coverage (320px, 768px, 1024px, 1440px)
- Accessibility score (> 90% axe-core pass rate)

**Critical Success Factors:**
- Always mask PII before storing screenshots
- Test across all target viewports in parallel
- Store baselines in version control
- Run accessibility audits on every visual change
</default_to_action>

## Quick Reference Card

### When to Use
| Scenario | Use This Skill? | Why |
|----------|-----------------|-----|
| Testing login pages | Yes | Contains PII (passwords, emails) |
| Visual regression suite | Yes | Parallel viewport + baseline comparison |
| Payment forms | Yes | Credit card data needs masking |
| Public marketing pages | Maybe | Only if sensitive data possible |
| API-only testing | No | Use security-testing skill instead |

### Key Capabilities

| Capability | Description | Performance |
|------------|-------------|-------------|
| URL Validation | Block malicious URLs before navigation | <5ms |
| PII Detection | Find 6+ types of sensitive data | <100ms |
| Parallel Viewports | Test 4 viewports simultaneously | 4x faster |
| Visual Regression | Pixel-diff with configurable threshold | <500ms |
| Accessibility Audit | WCAG 2.1 A/AA/AAA compliance | <2s |

---

## Workflows

### 1. Security Visual Audit (Full Pipeline)

```bash
# Run complete security + visual audit
aqe test visual-audit --url https://example.com --security --accessibility
```

**Steps:**
1. Validate URL security (block malicious schemes)
2. Scan page for security issues (XSS, injection patterns)
3. Capture screenshots across 4 viewports in parallel
4. Compare against stored baselines
5. Run accessibility audit (axe-core)
6. Generate consolidated report

### 2. PII-Safe Screenshot

```bash
# Capture screenshot with automatic PII masking
aqe screenshot --url https://example.com/profile --pii-safe
```

**PII Detection Patterns:**
- Email addresses: `user@example.com`
- Phone numbers: `+1-555-123-4567`
- Credit cards: `4111-1111-1111-1111`
- SSN: `123-45-6789`
- API keys: `sk_live_...`, `AKIA...`
- Passwords: Form fields with type="password"

**Masking Strategy:**
- Default: Blur with high intensity
- Options: `redact` (black box), `pixelate`, `blur`

### 3. Responsive Visual Audit

```bash
# Test visual consistency across viewports
aqe test responsive --url https://example.com --viewports mobile,tablet,desktop
```

**Default Viewports:**
| Name | Width | Height | Device |
|------|-------|--------|--------|
| mobile | 320px | 568px | iPhone SE |
| tablet | 768px | 1024px | iPad |
| desktop | 1440px | 900px | MacBook |
| wide | 1920px | 1080px | Full HD |

---

## Integration with AQE v3

### Using with BrowserSecurityScanner

```typescript
import { BrowserSecurityScanner } from '@agentic-qe/v3';

const scanner = new BrowserSecurityScanner(memory, {
  urlValidation: { enabled: true },
  piiDetection: { enabled: true, maskBeforeSave: true },
  parallelViewports: { maxConcurrent: 4 }
});

const result = await scanner.scanUrl('https://example.com', {
  viewports: ['mobile', 'tablet', 'desktop'],
  accessibility: true
});
```

### Using with TrajectoryAdapter

```typescript
import { TrajectoryAdapter } from '@agentic-qe/v3';

const adapter = new TrajectoryAdapter(memory);

// Record testing trajectory for learning
await adapter.startTrajectory('security-visual-test', {
  url: 'https://example.com',
  testType: 'security-visual'
});

// ... perform tests ...

await adapter.endTrajectory(trajectoryId, {
  success: true,
  piiFound: 3,
  visualRegressions: 0,
  accessibilityScore: 95
});
```

---

## Agent Coordination

### Memory Namespace
```
aqe/security-visual/
├── baselines/*          - Visual regression baselines
├── screenshots/*        - Captured screenshots (PII masked)
├── reports/*            - Audit reports
└── trajectories/*       - Learning trajectories
```

### Fleet Coordination
```typescript
const fleet = await FleetManager.coordinate({
  strategy: 'security-visual-audit',
  agents: [
    'qe-visual-tester',      // Visual regression
    'qe-security-scanner',   // URL/PII scanning
    'qe-accessibility-auditor' // WCAG compliance
  ],
  topology: 'parallel',
  maxConcurrent: 4
});
```

---

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `URL_BLOCKED` | Malicious URL pattern detected | Check URL, remove javascript:/data: |
| `PII_DETECTED` | Sensitive data found in screenshot | Enable masking or redact manually |
| `BASELINE_MISSING` | No baseline for comparison | Run with `--update-baseline` first |
| `VIEWPORT_TIMEOUT` | Browser didn't respond | Increase timeout or reduce parallel |
| `ACCESSIBILITY_FAILED` | WCAG violations found | Review violations, fix issues |

---

## Related Skills
- [visual-testing-advanced](../visual-testing-advanced/) - Pure visual testing without security
- [security-testing](../security-testing/) - Security testing without visual component
- [accessibility-testing](../accessibility-testing/) - Accessibility-only testing
- [qe-visual-accessibility](../qe-visual-accessibility/) - AQE v3 visual domain skill

---

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| URL validation | <5ms | 2ms |
| PII detection | <100ms | 45ms |
| Single viewport capture | <2s | 1.2s |
| 4-viewport parallel | <3s | 2.1s |
| Visual diff | <500ms | 320ms |
| Accessibility audit | <2s | 1.5s |
| Full pipeline | <10s | 7.2s |
