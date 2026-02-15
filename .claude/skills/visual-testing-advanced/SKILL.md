---
name: visual-testing-advanced
description: "Advanced visual regression testing with pixel-perfect comparison, AI-powered diff analysis, responsive design validation, and cross-browser visual consistency. Use when detecting UI regressions, validating designs, or ensuring visual consistency."
category: specialized-testing
priority: high
tokenEstimate: 900
agents: [qe-visual-tester, qe-test-executor, qe-quality-gate]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [visual, regression, screenshot, pixel-diff, percy, playwright, responsive]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/visual-testing-advanced.yaml

---

# Advanced Visual Testing

<default_to_action>
When detecting visual regressions or validating UI:
1. CAPTURE baseline screenshots (first run establishes baseline)
2. COMPARE new screenshots against baseline (pixel-by-pixel or AI)
3. MASK dynamic content (timestamps, ads, user counts)
4. TEST across devices (desktop, tablet, mobile viewports)
5. REVIEW and approve intentional changes, fail on regressions

**Quick Visual Testing Steps:**
- Set up baseline on main branch
- Compare feature branch against baseline
- Mask dynamic elements (timestamps, avatars)
- Use AI-powered comparison to reduce false positives
- Integrate in CI/CD to block visual regressions

**Critical Success Factors:**
- Functional tests don't catch visual bugs
- AI-powered tools reduce false positives
- Review diffs, don't just auto-approve
</default_to_action>

## Quick Reference Card

### When to Use
- UI component changes
- CSS/styling modifications
- Responsive design validation
- Cross-browser consistency checks

### Visual Bug Types
| Bug Type | Description |
|----------|-------------|
| Layout shift | Elements moved position |
| Color change | Unintended color modification |
| Font rendering | Typography issues |
| Alignment | Spacing/alignment problems |
| Missing images | Broken image paths |
| Overflow | Content clipping |

### Comparison Algorithms
| Algorithm | Best For |
|-----------|----------|
| **Pixel diff** | Exact match requirement |
| **Structural similarity** | Handle anti-aliasing |
| **AI semantic** | Ignore insignificant changes |

---

## Visual Regression with Playwright

```javascript
import { test, expect } from '@playwright/test';

test('homepage visual regression', async ({ page }) => {
  await page.goto('https://example.com');

  // Capture and compare screenshot
  await expect(page).toHaveScreenshot('homepage.png');
  // First run: saves baseline
  // Subsequent runs: compares to baseline
});

test('responsive design', async ({ page }) => {
  // Mobile viewport
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto('https://example.com');
  await expect(page).toHaveScreenshot('homepage-mobile.png');

  // Tablet viewport
  await page.setViewportSize({ width: 768, height: 1024 });
  await expect(page).toHaveScreenshot('homepage-tablet.png');
});
```

---

## Handling Dynamic Content

```javascript
test('mask dynamic elements', async ({ page }) => {
  await page.goto('https://example.com');

  await expect(page).toHaveScreenshot({
    mask: [
      page.locator('.timestamp'),     // Dynamic time
      page.locator('.user-count'),    // Live counter
      page.locator('.advertisement'), // Ads
      page.locator('.avatar')         // User avatars
    ]
  });
});
```

---

## AI-Powered Visual Testing (Percy)

```javascript
import percySnapshot from '@percy/playwright';

test('AI-powered visual test', async ({ page }) => {
  await page.goto('https://example.com');

  // Percy uses AI to ignore anti-aliasing, minor font differences
  await percySnapshot(page, 'Homepage');
});

test('component visual test', async ({ page }) => {
  await page.goto('https://example.com/components');

  // Snapshot specific component
  const button = page.locator('.primary-button');
  await percySnapshot(page, 'Primary Button', {
    scope: button
  });
});
```

---

## Playwright Configuration

```javascript
// playwright.config.js
export default {
  expect: {
    toHaveScreenshot: {
      maxDiffPixels: 100,      // Allow 100 pixel difference
      maxDiffPixelRatio: 0.01, // Or 1% of image
      threshold: 0.2,          // Color similarity threshold
      animations: 'disabled',  // Disable animations
      caret: 'hide'            // Hide cursor
    }
  }
};
```

---

## Agent-Driven Visual Testing

```typescript
// Comprehensive visual regression
await Task("Visual Regression Suite", {
  baseline: 'main-branch',
  current: 'feature-branch',
  pages: ['homepage', 'product', 'checkout'],
  devices: ['desktop', 'tablet', 'mobile'],
  browsers: ['chrome', 'firefox', 'safari'],
  threshold: 0.01
}, "qe-visual-tester");

// Returns:
// {
//   comparisons: 27,  // 3 pages × 3 devices × 3 browsers
//   differences: 2,
//   report: 'visual-regression-report.html'
// }
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/visual-testing/
├── baselines/*          - Baseline screenshots
├── comparisons/*        - Diff results
├── components/*         - Component snapshots
└── reports/*            - Visual regression reports
```

### Fleet Coordination
```typescript
const visualFleet = await FleetManager.coordinate({
  strategy: 'visual-testing',
  agents: [
    'qe-visual-tester',   // Screenshot comparison
    'qe-test-executor',   // Cross-browser execution
    'qe-quality-gate'     // Block on visual regressions
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [accessibility-testing](../accessibility-testing/) - Visual a11y checks
- [compatibility-testing](../compatibility-testing/) - Cross-browser visuals
- [regression-testing](../regression-testing/) - Regression suite

---

## Remember

**Functional tests don't catch visual bugs.** Layout shifts, color changes, font rendering, alignment issues - all invisible to functional tests but visible to users.

**AI-powered tools reduce false positives.** Percy, Applitools use AI to ignore insignificant differences (anti-aliasing, minor font rendering).

**With Agents:** `qe-visual-tester` automates visual regression across browsers and devices, uses AI to filter noise, and generates visual diff reports. Catches UI regressions before users see them.
