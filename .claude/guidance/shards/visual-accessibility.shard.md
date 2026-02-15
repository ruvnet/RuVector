# Visual Accessibility Domain Shard

**Domain**: visual-accessibility
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **WCAG 2.2 Level AA Minimum**: All user-facing components MUST meet WCAG 2.2 Level AA compliance; Level A alone is insufficient for production.

2. **Automated + Manual Testing**: Accessibility testing MUST combine automated scanning (axe-core) with manual verification; automated-only testing misses critical issues.

3. **Visual Regression Baselines**: Visual regression tests MUST use approved baselines; changes to baselines require explicit approval with accessibility review.

4. **Responsive Design Coverage**: Visual tests MUST cover at minimum desktop (1920x1080), tablet (768x1024), and mobile (375x667) viewports.

5. **EU Compliance (EN 301 549)**: Products targeting EU markets MUST pass EN 301 549 and EAA requirements in addition to WCAG.

6. **Contrast Ratio Enforcement**: All text MUST meet WCAG contrast ratios (4.5:1 normal, 3:1 large text); violations are deployment blockers.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| WCAG AA Compliance | 0.95 | 1.0 | < 0.85 |
| Contrast Compliance | 1.0 | 1.0 | < 0.9 |
| Visual Diff Threshold | < 0.05 | < 0.01 | > 0.1 |
| Responsive Coverage | 3 viewports | 5 viewports | < 3 viewports |
| axe-core Pass Rate | 0.95 | 1.0 | < 0.8 |
| Keyboard Navigation | 1.0 | 1.0 | < 0.9 |

---

## Invariants

```
INVARIANT wcag_aa_compliance:
  FOR ALL component IN user_facing_components:
    component.wcag_level >= 'AA' AND
    component.wcag_violations.critical = 0 AND
    component.wcag_violations.serious = 0
```

```
INVARIANT contrast_ratio_enforcement:
  FOR ALL text_element IN ui_elements:
    IF text_element.type = 'text' THEN
      IF text_element.font_size < 18 THEN
        text_element.contrast_ratio >= 4.5
      ELSE
        text_element.contrast_ratio >= 3.0
```

```
INVARIANT baseline_approval:
  FOR ALL baseline_change IN baseline_changes:
    baseline_change.approved = true AND
    baseline_change.accessibility_reviewed = true AND
    baseline_change.approver_id IS NOT NULL
```

```
INVARIANT responsive_coverage:
  FOR ALL visual_test IN visual_tests:
    visual_test.viewports.length >= 3 AND
    visual_test.viewports INCLUDES ['desktop', 'tablet', 'mobile']
```

```
INVARIANT keyboard_navigability:
  FOR ALL interactive_element IN interactive_elements:
    interactive_element.keyboard_accessible = true AND
    interactive_element.focus_visible = true AND
    interactive_element.tab_order_logical = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/visual-accessibility/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Visual Tester Service | `services/visual-tester.ts` | Visual regression testing |
| Accessibility Tester Service | `services/accessibility-tester.ts` | WCAG compliance testing |
| Responsive Tester Service | `services/responsive-tester.ts` | Multi-viewport testing |
| Viewport Capture Service | `services/viewport-capture.ts` | Screenshot capture |
| Visual Regression Service | `services/visual-regression.ts` | Baseline comparison |
| EU Compliance Service | `services/eu-compliance.ts` | EN 301 549, EAA validation |
| axe-core Integration | `services/axe-core-integration.ts` | Automated accessibility |

**Device Viewports**: Defined in `services/responsive-tester.ts` as `DEVICE_VIEWPORTS`.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-accessibility-auditor` | Full WCAG auditing |
| **Primary** | `qe-visual-tester` | Visual regression testing |
| **Secondary** | `qe-responsive-tester` | Multi-viewport testing |
| **Secondary** | `qe-eu-compliance-validator` | EN 301 549/EAA validation |
| **Approval** | `a11y-team` (human) | Baseline approval |
| **Readonly** | `qe-quality-gate` | Query compliance status |

**Forbidden Actions**: No agent may approve baseline changes without accessibility review.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| WCAG AA compliance < 0.85 | CRITICAL | Block deployment, escalate to Queen Coordinator |
| Contrast violation detected | CRITICAL | Block deployment, require fix |
| Critical axe-core violation | CRITICAL | Block deployment, generate remediation |
| Keyboard navigation failure | CRITICAL | Block deployment, flag for immediate fix |
| Visual diff > 0.1 (10%) | HIGH | Require baseline review |
| Responsive coverage < 3 viewports | HIGH | Add missing viewport tests |
| EU compliance failure (EU markets) | HIGH | Block EU deployment |
| Baseline change without approval | MEDIUM | Revert change, escalate |
| Tab order issues | MEDIUM | Flag for review |

---

## Memory Namespace

- **Namespace**: `qe-patterns/visual-accessibility`
- **Retention**: 90 days (baselines retained longer)
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-execution` | Output | Execute visual/a11y tests |
| `requirements-validation` | Bidirectional | A11y requirements validation |
| `quality-assessment` | Output | Report compliance scores |
| `test-generation` | Output | Generate a11y test cases |
| `learning-optimization` | Bidirectional | Share a11y patterns |

---

## WCAG 2.2 Success Criteria Levels

| Level | Requirement | Deployment Impact |
|-------|-------------|------------------|
| Level A | Minimum compliance | Required for all |
| Level AA | Target compliance | Required for production |
| Level AAA | Enhanced compliance | Recommended for government |

---

## Viewport Presets

| Preset | Width | Height | Use Case |
|--------|-------|--------|----------|
| mobile-sm | 320 | 568 | iPhone SE |
| mobile | 375 | 667 | iPhone 8 |
| mobile-lg | 414 | 896 | iPhone 11 Pro Max |
| tablet | 768 | 1024 | iPad |
| tablet-lg | 1024 | 1366 | iPad Pro |
| desktop | 1280 | 800 | Small desktop |
| desktop-lg | 1920 | 1080 | Full HD |

---

## EN 301 549 Clause Coverage

| Clause | Description | Test Service |
|--------|-------------|--------------|
| 9.1 | Perceivable | AccessibilityTesterService |
| 9.2 | Operable | AccessibilityTesterService |
| 9.3 | Understandable | AccessibilityTesterService |
| 9.4 | Robust | AccessibilityTesterService |
| 10 | Non-web Documents | N/A |
| 11 | Software | EUComplianceService |
| 12 | Documentation | Manual review |

---

## axe-core Configuration

```typescript
const DEFAULT_AXE_CONFIG: AxeCoreConfig = {
  runOnly: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'],
  rules: {
    'color-contrast': { enabled: true },
    'keyboard': { enabled: true },
    'focus-visible': { enabled: true },
    'aria-required-attr': { enabled: true },
  },
  reporter: 'v2',
  resultTypes: ['violations', 'incomplete', 'passes'],
};
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
