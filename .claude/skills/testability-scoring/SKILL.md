---
name: testability-scoring
description: "AI-powered testability assessment using 10 principles of intrinsic testability with Playwright and optional Vibium integration. Evaluates web applications against Observability, Controllability, Algorithmic Simplicity, Transparency, Stability, Explainability, Unbugginess, Smallness, Decomposability, and Similarity. Use when assessing software testability, evaluating test readiness, identifying testability improvements, or generating testability reports."
category: testing-methodologies
priority: high
tokenEstimate: 1100
agents: [qe-quality-analyzer, qx-partner, qe-visual-tester]
implementation_status: optimized
optimization_version: 2.2
last_optimized: 2025-12-12
dependencies: []
quick_reference_card: true
tags: [testability, scoring, playwright, vibium, assessment, 10-principles, intrinsic-testability, james-bach, michael-bolton]
contributor: "@fndlalit"
vibium_integration: optional
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/testability-scoring.yaml
---

# Testability Scoring

<default_to_action>
When assessing testability:
1. RUN assessment against target URL
2. ANALYZE all 10 principles automatically
3. GENERATE HTML report with radar chart
4. PRIORITIZE improvements by impact/effort
5. INTEGRATE with QX Partner for holistic view

**Quick Assessment:**
```bash
# Run assessment on any URL
TEST_URL='https://example.com/' npx playwright test tests/testability-scoring/testability-scoring.spec.js --project=chromium --workers=1

# Or use shell script wrapper
.claude/skills/testability-scoring/scripts/run-assessment.sh https://example.com/
```

**The 10 Principles at a Glance:**
| Principle | Weight | Key Question |
|-----------|--------|--------------|
| **Observability** | 15% | Can we see what's happening? |
| **Controllability** | 15% | Can we control the application? |
| **Algorithmic Simplicity** | 10% | Are behaviors predictable? |
| **Algorithmic Transparency** | 10% | Can we understand what it does? |
| **Algorithmic Stability** | 10% | Does behavior remain consistent? |
| **Explainability** | 10% | Is the interface understandable? |
| **Unbugginess** | 10% | How error-free is it? |
| **Smallness** | 10% | Are components appropriately sized? |
| **Decomposability** | 5% | Can we test parts in isolation? |
| **Similarity** | 5% | Is the tech stack familiar? |

**Grade Scale:**
- **A (90-100)**: Excellent testability
- **B (80-89)**: Good testability
- **C (70-79)**: Adequate testability
- **D (60-69)**: Below average
- **F (0-59)**: Poor testability
</default_to_action>

## Quick Reference Card

### Running Assessments

| Method | Command | When to Use |
|--------|---------|-------------|
| Shell Script | `./scripts/run-assessment.sh URL` | One-time assessment |
| ENV Override | `TEST_URL='URL' npx playwright test...` | CI/CD integration |
| Config File | Update `tests/testability-scoring/config.js` | Repeated runs |

### Principle Details

#### High Weight (15% each)
| Principle | Measures | Indicators |
|-----------|----------|------------|
| **Observability** | State visibility, logging, monitoring | Console output, network tracking, error visibility |
| **Controllability** | Input control, state manipulation | API access, test data injection, determinism |

#### Medium Weight (10% each)
| Principle | Measures | Indicators |
|-----------|----------|------------|
| **Simplicity** | Predictable behavior | Clear I/O relationships, low complexity |
| **Transparency** | Understanding what system does | Visible processes, readable code |
| **Stability** | Consistent behavior | Change resilience, maintainability |
| **Explainability** | Interface understanding | Good docs, semantic structure, help text |
| **Unbugginess** | Error-free operation | Console errors, warnings, runtime issues |
| **Smallness** | Component size | Element count, script bloat, page complexity |

#### Low Weight (5% each)
| Principle | Measures | Indicators |
|-----------|----------|------------|
| **Decomposability** | Isolation testing | Component separation, modular design |
| **Similarity** | Technology familiarity | Standard frameworks, known patterns |

---

## Assessment Workflow

```
1. Navigate to URL → 2. Collect Metrics → 3. Score Principles
                                              ↓
4. Generate JSON ← 5. Calculate Grades ← 6. Apply Weights
         ↓
7. Generate HTML Report with Radar Chart
         ↓
8. Open in Browser (auto-opens)
```

### Output Files

```
tests/reports/
├── testability-results-<timestamp>.json  # Raw data
├── testability-report-<timestamp>.html   # Visual report
└── latest.json                           # Symlink
```

---

## Integration Examples

### CI/CD Integration
```yaml
# GitHub Actions
- name: Testability Assessment
  run: |
    timeout 180 .claude/skills/testability-scoring/scripts/run-assessment.sh ${{ env.APP_URL }}

- name: Upload Reports
  uses: actions/upload-artifact@v3
  with:
    name: testability-reports
    path: tests/reports/testability-*.html
```

### QX Partner Integration
```typescript
// Combine testability with QX analysis
const qxAnalysis = await Task("QX Analysis", {
  target: 'https://example.com',
  integrateTestability: true
}, "qx-partner");

// Returns combined insights:
// - QX Score: 78/100
// - Testability Integration: Observability 72/100
// - Combined Insight: Low observability may mask UX issues
```

### Programmatic Usage
```typescript
import { runTestabilityAssessment } from './testability';

const results = await runTestabilityAssessment('https://example.com');
console.log(`Overall: ${results.overallScore}/100 (${results.grade})`);
console.log('Recommendations:', results.recommendations);
```

---

## Agent Integration

```typescript
// Run testability assessment
const assessment = await Task("Testability Assessment", {
  url: 'https://example.com',
  generateReport: true,
  openBrowser: true
}, "qe-quality-analyzer");

// Use with QX Partner for holistic analysis
const qxReport = await Task("Full QX Analysis", {
  target: 'https://example.com',
  integrateTestability: true,
  detectOracleProblems: true
}, "qx-partner");
```

---

## Vibium Integration (Optional)

### Overview
Vibium browser automation can be used alongside Playwright for enhanced testability assessment. While **Playwright remains the primary engine**, Vibium offers complementary capabilities for certain metrics.

**Installation:**
```bash
claude mcp add vibium -- npx -y vibium
```

### Vibium-Enhanced Metrics

| Principle | Vibium Enhancement | Benefit |
|-----------|-------------------|---------|
| **Observability** | Auto-wait duration tracking | Measures DOM stability (30s timeout, 100ms polling) |
| **Controllability** | Element interaction success rate | Validates automation readiness via MCP |
| **Stability** | Screenshot consistency | Visual regression detection for layout stability |
| **Explainability** | Element attribute extraction | ARIA labels, semantic HTML validation |

### When to Use Vibium

✅ **USE Vibium for:**
- Element stability metrics (auto-wait duration analysis)
- Visual consistency checks (screenshot comparison)
- MCP-native AI agent integration
- Lightweight Docker images (400MB vs 1.2GB)

❌ **USE Playwright for:**
- Console error detection (Vibium V1 lacks console API)
- Network performance metrics (BiDi network APIs coming in V2)
- Comprehensive browser coverage (Firefox, Safari)
- Production-proven stability (Vibium V1 released Dec 2024)

### Hybrid Assessment Example

```typescript
// Testability assessment using both engines
const assessment = {
  // Playwright: Comprehensive metrics
  playwright: await runPlaywrightAssessment(url),

  // Vibium: Stability metrics
  vibium: {
    elementStability: await measureAutoWaitDuration(url),
    visualConsistency: await compareScreenshots(url),
    accessibilityAttributes: await extractARIALabels(url)
  }
};

// Enhanced Observability Score
const observability =
  (assessment.playwright.consoleErrors * 0.6) +
  (assessment.vibium.elementStability * 0.4);
```

### Vibium MCP Tools for Testability

```typescript
// 1. Element Stability Measurement
const browser = await browser_launch();
await browser_navigate({ url });
const startTime = Date.now();
const element = await browser_find({ selector: ".critical-element" });
const autoWaitDuration = Date.now() - startTime;
// Lower duration = better stability

// 2. Visual Consistency Check
const screenshot1 = await browser_screenshot();
await browser_navigate({ url }); // Reload
const screenshot2 = await browser_screenshot();
const visualDiff = compareImages(screenshot1.png, screenshot2.png);
// Lower diff = better stability

// 3. Accessibility Attribute Extraction
const elements = await browser_find({ selector: "button, a, input" });
const ariaLabels = elements.map(el => el.attributes["aria-label"]);
const semanticScore = (ariaLabels.filter(Boolean).length / elements.length) * 100;
```

### Migration Strategy

**Current (V2.2):** Hybrid approach
- Playwright: Primary engine for all 10 principles
- Vibium: Optional enhancement for stability metrics

**Future (V3.0):** When Vibium V2 ships
- Evaluate Vibium as primary engine if:
  - Console/Network APIs available
  - Production stability proven
  - Community adoption increases

## Agent Coordination Hints

### Memory Namespace
```
aqe/testability/
├── assessments/*       - Assessment results by URL
├── historical/*        - Historical scores for trend analysis
├── recommendations/*   - Improvement recommendations
├── integration/*       - QX integration data
└── vibium/*           - Vibium-specific metrics (optional)
```

### Fleet Coordination
```typescript
const testabilityFleet = await FleetManager.coordinate({
  strategy: 'testability-assessment',
  agents: [
    'qe-quality-analyzer',  // Primary assessment
    'qx-partner',           // UX integration
    'qe-visual-tester'      // Visual validation
  ],
  topology: 'sequential'
});
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Tests timing out | Increase timeout: `timeout 300 ./scripts/run-assessment.sh URL` |
| Partial results | Check console errors, increase network timeout |
| Report not opening | Use `AUTO_OPEN=false`, open manually |
| Config not updating | Use `TEST_URL` env var instead |
| Vibium not available | Install via `claude mcp add vibium -- npx -y vibium` (optional) |
| Hybrid mode errors | Vibium is optional; assessments work without it |

---

## Related Skills
- [accessibility-testing](../accessibility-testing/) - WCAG compliance (overlaps with Explainability)
- [visual-testing-advanced](../visual-testing-advanced/) - UI consistency
- [performance-testing](../performance-testing/) - Load time metrics

---

## Credits & References

### Framework Origin
- **Heuristics for Software Testability** by James Bach and Michael Bolton
- Available at: https://www.satisfice.com/download/heuristics-of-software-testability

### Implementation
- Based on https://github.com/fndlalit/testability-scorer (contributed by [@fndlalit](https://github.com/fndlalit))
- Playwright v1.49.0+ with AI capabilities (primary engine)
- Vibium v1.0+ with MCP integration (optional enhancement)
- Chart.js for radar visualizations

### Vibium Resources
- GitHub: https://github.com/VibiumDev/vibium
- MCP Integration: `claude mcp add vibium -- npx -y vibium`
- Created by Jason Huggins (creator of Selenium/Appium)

---

## Remember

**Testability is an investment, not an afterthought.**

Good testability:
- Reduces debugging time
- Enables faster feedback loops
- Makes defects easier to find
- Supports continuous testing

**Low scores = High risk. Prioritize improvements by weight × impact.**
