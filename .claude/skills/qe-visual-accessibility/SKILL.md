---
name: "QE Visual Accessibility"
description: "Visual regression testing, responsive design validation, and WCAG accessibility compliance testing."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-visual-accessibility.yaml

---

# QE Visual Accessibility

## Purpose

Guide the use of v3's visual and accessibility testing capabilities including screenshot comparison, responsive design validation, and WCAG 2.2 compliance verification.

## Activation

- When testing visual appearance
- When validating responsive design
- When checking accessibility compliance
- When detecting visual regressions
- When testing cross-browser rendering

## Quick Start

```bash
# Visual regression test
aqe visual test --baseline production --current staging

# Responsive design test
aqe visual responsive --url https://example.com --viewports all

# Accessibility audit
aqe a11y audit --url https://example.com --standard wcag22-aa

# Cross-browser test
aqe visual cross-browser --url https://example.com --browsers chrome,firefox,safari
```

## Agent Workflow

```typescript
// Visual regression testing
Task("Run visual regression", `
  Compare staging against production:
  - Capture screenshots of key pages
  - Detect pixel differences
  - Flag significant visual changes
  - Generate visual diff report
`, "qe-visual-tester")

// Accessibility audit
Task("Audit accessibility", `
  Run WCAG 2.2 AA compliance audit:
  - Check color contrast ratios
  - Verify keyboard navigation
  - Test screen reader compatibility
  - Validate ARIA labels
  Generate compliance report with fix suggestions.
`, "qe-accessibility-agent")
```

## Visual Testing Operations

### 1. Visual Regression

```typescript
await visualTester.compareScreenshots({
  baseline: {
    source: 'production',
    pages: ['/', '/login', '/dashboard', '/settings']
  },
  current: {
    source: 'staging',
    pages: ['/', '/login', '/dashboard', '/settings']
  },
  comparison: {
    threshold: 0.1,  // 0.1% pixel difference
    antialiasing: true,
    ignoreRegions: ['#dynamic-content', '.timestamp']
  }
});
```

### 2. Responsive Testing

```typescript
await responsiveTester.test({
  url: 'https://example.com',
  viewports: [
    { name: 'mobile', width: 375, height: 667 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1920, height: 1080 }
  ],
  checks: {
    layoutShift: true,
    contentOverflow: true,
    touchTargets: true,
    fontScaling: true
  }
});
```

### 3. Accessibility Audit

```typescript
await accessibilityAgent.audit({
  url: 'https://example.com',
  standard: 'WCAG22-AA',
  checks: {
    perceivable: {
      colorContrast: true,
      textAlternatives: true,
      captions: true
    },
    operable: {
      keyboardAccessible: true,
      noTimingIssues: true,
      navigable: true
    },
    understandable: {
      readable: true,
      predictable: true,
      inputAssistance: true
    },
    robust: {
      compatible: true,
      parseErrors: true
    }
  }
});
```

### 4. Cross-Browser Testing

```typescript
await visualTester.crossBrowser({
  url: 'https://example.com',
  browsers: ['chrome', 'firefox', 'safari', 'edge'],
  versions: 'latest-2',
  comparisons: {
    betweenBrowsers: true,
    betweenVersions: true,
    againstBaseline: true
  }
});
```

## WCAG 2.2 Checklist

| Level | Criteria | Auto-Testable |
|-------|----------|---------------|
| A | Non-text Content | ✅ |
| A | Info and Relationships | Partial |
| A | Color Contrast (4.5:1) | ✅ |
| A | Keyboard Accessible | ✅ |
| A | Focus Visible | ✅ |
| AA | Reflow | ✅ |
| AA | Text Spacing | ✅ |
| AAA | Enhanced Contrast (7:1) | ✅ |

## Visual Test Report

```typescript
interface VisualReport {
  summary: {
    pagesCompared: number;
    differencesFound: number;
    passRate: number;
  };
  comparisons: {
    page: string;
    viewport: string;
    baseline: string;
    current: string;
    diff: string;
    diffPercentage: number;
    status: 'pass' | 'fail' | 'review';
  }[];
  accessibility: {
    violations: A11yViolation[];
    passes: number;
    incomplete: number;
    score: number;
  };
  responsive: {
    viewport: string;
    issues: ResponsiveIssue[];
  }[];
}
```

## Accessibility Report

```typescript
interface AccessibilityReport {
  summary: {
    score: number;
    violations: number;
    warnings: number;
    passes: number;
  };
  violations: {
    id: string;
    impact: 'critical' | 'serious' | 'moderate' | 'minor';
    description: string;
    wcag: string[];
    elements: {
      selector: string;
      html: string;
      issue: string;
      fix: string;
    }[];
  }[];
  compliance: {
    wcagLevel: 'A' | 'AA' | 'AAA';
    criteriasMet: number;
    criteriasTotal: number;
  };
}
```

## CI/CD Integration

```yaml
visual_testing:
  on_pr:
    - capture_screenshots
    - compare_to_baseline
    - run_a11y_audit

  thresholds:
    visual_diff: 0.1
    a11y_violations: 0

  artifacts:
    - screenshots/
    - diffs/
    - a11y-report.html
```

## Coordination

**Primary Agents**: qe-visual-tester, qe-accessibility-agent, qe-responsive-tester
**Coordinator**: qe-visual-coordinator
**Related Skills**: qe-test-execution, qe-quality-assessment
