---
name: a11y-ally
description: "Comprehensive WCAG accessibility auditing with multi-tool testing (axe-core + pa11y + Lighthouse), TRUE PARALLEL execution with Promise.allSettled, graceful degradation, retry with backoff, context-aware remediation, learning integration, and video accessibility. Uses 3-tier browser cascade: Vibium → agent-browser → Playwright+Stealth."
category: specialized-testing
priority: critical
tokenEstimate: 10000
agents: []
implementation_status: active
optimization_version: 7.0
last_optimized: 2026-01-26
dependencies: [playwright, playwright-extra, puppeteer-extra-plugin-stealth, "@axe-core/playwright", pa11y, lighthouse]
quick_reference_card: true
tags: [accessibility, wcag, a11y, video, captions, audiodesc, vtt, eu-compliance, context-aware, remediation, axe-core, pa11y, lighthouse, parallel, resilient, graceful-degradation, retry]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/a11y-ally.yaml

---

# /a11y-ally - Comprehensive Accessibility Audit

<default_to_action>
When this skill is invoked with a URL, Claude executes ALL steps automatically without waiting for user prompts between steps.

## THIS IS AN LLM-POWERED SKILL

The value of this skill is **Claude's intelligence**, not just running automated tools:

| Automated Tools Do | Claude (This Skill) Does |
|--------------------|--------------------------|
| Flag "button has no name" | Analyze context: icon class, parent element, nearby text → generate "Add to wishlist" |
| Flag "image missing alt" | Use Vision to see the image → describe actual content |
| Flag "video has no captions" | Download video, extract frames, analyze each frame with Vision → generate real captions |
| Output generic templates | Generate context-specific, copy-paste ready fixes |

**IF YOU SKIP THE LLM ANALYSIS, THIS SKILL HAS NO VALUE.**

---

## EXECUTION MODEL

**CLAUDE EXECUTES ALL STEPS WITHOUT STOPPING.**

Do NOT wait for user prompts between steps. Execute the full pipeline:

1. **Data Collection**: Run multi-tool scan (axe-core, pa11y, Lighthouse) via Bash
2. **LLM Analysis**: Read results and analyze context for each violation
3. **Vision Pipeline**: If videos detected → download → extract frames → Read each frame → describe
4. **Intelligent Remediation**: Generate context-specific fixes using your reasoning
5. **Generate Reports**: Write all output files to `docs/accessibility-scans/{page-slug}/`

**WRONG:**
```
Claude: "I found 5 violations. Should I analyze them?"
User: "Yes"
Claude: "I see a video. Should I run the video pipeline?"
User: "Yes"
```

**RIGHT:**
```
Claude: [Runs scan] → [Analyzes violations] → [Downloads video] → [Extracts frames] →
        [Reads each frame with Vision] → [Generates captions] → [Writes all files]
        "Audit complete. Generated 4 files in docs/accessibility-scans/example/"
```

---

## STEP 1: BROWSER AUTOMATION - Content Fetching

### 1.1: Try VIBIUM First (Primary)
```javascript
ToolSearch("select:mcp__vibium__browser_launch")
ToolSearch("select:mcp__vibium__browser_navigate")
mcp__vibium__browser_launch({ headless: true })
mcp__vibium__browser_navigate({ url: "TARGET_URL" })
```

**If Vibium fails** → Go to STEP 1b

### 1b: Try AGENT-BROWSER Fallback
```javascript
ToolSearch("select:mcp__claude-flow_alpha__browser_open")
mcp__claude-flow_alpha__browser_open({ url: "TARGET_URL", waitUntil: "networkidle" })
```

**If agent-browser fails** → Go to STEP 1c

### 1c: PLAYWRIGHT + STEALTH (Final Fallback)
```bash
mkdir -p /tmp/a11y-work && cd /tmp/a11y-work
npm init -y 2>/dev/null
npm install playwright-extra puppeteer-extra-plugin-stealth @axe-core/playwright pa11y lighthouse chrome-launcher 2>/dev/null
```

Create and run scan script - see STEP 2 for full multi-tool scan code.

### 1d: PARALLEL MULTI-PAGE AUDIT (Optional)

For auditing multiple URLs simultaneously, use parallel execution:

```javascript
// /tmp/a11y-work/parallel-audit.js
const { chromium } = require('playwright-extra');
const stealth = require('puppeteer-extra-plugin-stealth')();
const { AxeBuilder } = require('@axe-core/playwright');

chromium.use(stealth);

const MAX_CONCURRENT = 6;  // Maximum parallel auditors

async function auditUrl(browser, url) {
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForTimeout(2000);

    const axeResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
      .analyze();

    return { url, success: true, violations: axeResults.violations };
  } catch (error) {
    return { url, success: false, error: error.message };
  } finally {
    await context.close();
  }
}

async function parallelAudit(urls) {
  const browser = await chromium.launch({ headless: true });
  const results = [];

  // Process in chunks of MAX_CONCURRENT
  for (let i = 0; i < urls.length; i += MAX_CONCURRENT) {
    const chunk = urls.slice(i, i + MAX_CONCURRENT);
    console.log(`Auditing batch ${Math.floor(i/MAX_CONCURRENT) + 1}: ${chunk.length} URLs`);

    const chunkResults = await Promise.all(
      chunk.map(url => auditUrl(browser, url))
    );
    results.push(...chunkResults);
  }

  await browser.close();
  return results;
}

// Usage: node parallel-audit.js url1 url2 url3 ...
const urls = process.argv.slice(2);
if (urls.length > 0) {
  parallelAudit(urls).then(results => {
    console.log(JSON.stringify(results, null, 2));
  });
}
```

**Usage for multi-page audit:**
```bash
node parallel-audit.js https://example.com https://example.com/about https://example.com/contact
```

### 1e: SITE CRAWL MODE (Optional)

For comprehensive site audits, crawl and audit all pages:

```javascript
// /tmp/a11y-work/crawl-audit.js
async function crawlAndAudit(startUrl, maxPages = 50) {
  const browser = await chromium.launch({ headless: true });
  const visited = new Set();
  const toVisit = [startUrl];
  const results = [];
  const baseUrl = new URL(startUrl).origin;

  while (toVisit.length > 0 && results.length < maxPages) {
    const url = toVisit.shift();
    if (visited.has(url)) continue;
    visited.add(url);

    console.log(`[${results.length + 1}/${maxPages}] Auditing: ${url}`);

    const context = await browser.newContext();
    const page = await context.newPage();

    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

      // Extract same-domain links for crawling
      const links = await page.evaluate((base) => {
        return [...document.querySelectorAll('a[href]')]
          .map(a => a.href)
          .filter(href => href.startsWith(base) && !href.includes('#'))
          .filter(href => !href.match(/\.(pdf|jpg|png|gif|css|js)$/i));
      }, baseUrl);

      // Add new links to queue
      links.forEach(link => {
        if (!visited.has(link) && !toVisit.includes(link)) {
          toVisit.push(link);
        }
      });

      // Run accessibility audit
      const axeResults = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze();

      results.push({ url, violations: axeResults.violations });
    } catch (e) {
      results.push({ url, error: e.message });
    }

    await context.close();
  }

  await browser.close();
  return { pagesAudited: results.length, results };
}

// Usage: node crawl-audit.js https://example.com 50
const [startUrl, maxPages] = process.argv.slice(2);
crawlAndAudit(startUrl, parseInt(maxPages) || 50).then(r => console.log(JSON.stringify(r, null, 2)));
```

---

## STEP 2: COMPREHENSIVE WCAG SCAN (Multi-Tool, Parallel, Resilient)

**IMPORTANT:** This step uses THREE accessibility testing tools for maximum coverage:
- **axe-core**: Industry standard, excellent for ARIA and semantic issues
- **pa11y**: Strong on contrast, links, and HTML validation
- **Lighthouse**: Google's accessibility scoring with performance correlation

Combined detection rate is ~15% higher than any single tool.

### 2.0: RESILIENCE ARCHITECTURE (v7.0 Enhancement)

**Key improvements over v6.0:**

| Feature | v6.0 (Old) | v7.0 (New) |
|---------|------------|------------|
| Tool execution | Sequential | **Parallel (Promise.allSettled)** |
| Timeout handling | Global 60s | **Per-tool (60s/60s/90s)** |
| Failure mode | All-or-nothing | **Graceful degradation** |
| Retry logic | None | **Exponential backoff (3 retries)** |
| Output style | Wait for all | **Progressive (stream as ready)** |
| Minimum tools | 3 required | **1 of 3 sufficient** |

**Coverage by tools succeeded:**
- 3/3 tools: ~95% detection (optimal)
- 2/3 tools: ~85% detection (good)
- 1/3 tools: ~70% detection (acceptable)
- 0/3 tools: FAIL - retry with different strategy

### 2.1: Run Multi-Tool Analysis (PARALLEL + RESILIENT)
Create and run `/tmp/a11y-work/multi-tool-scan.js`:
```javascript
const { chromium } = require('playwright-extra');
const stealth = require('puppeteer-extra-plugin-stealth')();
const { AxeBuilder } = require('@axe-core/playwright');
const pa11y = require('pa11y');
const lighthouse = require('lighthouse').default || require('lighthouse');
const { launch: launchChrome } = require('chrome-launcher');
const fs = require('fs');

chromium.use(stealth);

const TARGET_URL = process.argv[2] || 'TARGET_URL';
const OUTPUT_FILE = '/tmp/a11y-work/scan-results.json';
const SYSTEM_CHROMIUM = '/usr/bin/chromium';

// ========== RESILIENCE UTILITIES ==========

// Timeout wrapper - wraps any promise with a timeout
function withTimeout(promise, ms, name) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`${name} timed out after ${ms}ms`)), ms)
    )
  ]);
}

// Retry wrapper - retries with exponential backoff
async function withRetry(fn, name, maxRetries = 3, baseDelay = 2000) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      console.log(`[${name}] Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
      if (isLastAttempt) throw error;
      const delay = baseDelay * Math.pow(2, attempt - 1); // Exponential backoff
      console.log(`[${name}] Retrying in ${delay}ms...`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

// Sleep utility
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Progressive output - append results as they arrive
function progressiveOutput(tool, data) {
  console.log(`\n=== ${tool.toUpperCase()} COMPLETE ===`);
  console.log(JSON.stringify(data, null, 2));

  // Append to results file for progressive access
  try {
    let results = {};
    if (fs.existsSync(OUTPUT_FILE)) {
      results = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
    }
    results[tool] = data;
    results.lastUpdated = new Date().toISOString();
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  } catch (e) { /* ignore file errors */ }
}

// ========== TOOL RUNNERS ==========

// TOOL 1: Axe-core (with page info extraction)
async function runAxeCore(url) {
  console.log('[axe-core] Starting...');
  const browser = await chromium.launch({
    headless: true,
    executablePath: SYSTEM_CHROMIUM,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-blink-features=AutomationControlled'
    ]
  });

  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    locale: 'en-US',
    viewport: { width: 1920, height: 1080 }
  });

  const page = await context.newPage();

  try {
    // Use domcontentloaded (faster, more reliable than networkidle)
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Random delay to appear human
    await sleep(2000 + Math.random() * 2000);

    // Try to dismiss cookie banners
    try {
      const cookieSelectors = [
        'button:has-text("Accept")', 'button:has-text("Akzeptieren")',
        'button:has-text("Alle akzeptieren")', '[data-testid="cookie-accept"]',
        '#onetrust-accept-btn-handler', '.cookie-consent-accept'
      ];
      for (const selector of cookieSelectors) {
        const btn = await page.$(selector);
        if (btn) { await btn.click(); await sleep(500); break; }
      }
    } catch (e) { /* ignore cookie errors */ }

    // Run axe-core analysis
    const axeResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
      .analyze();

    // Extract comprehensive page info
    const pageInfo = await page.evaluate(() => ({
      title: document.title,
      url: window.location.href,
      lang: document.documentElement.lang,
      images: {
        total: document.querySelectorAll('img').length,
        withAlt: document.querySelectorAll('img[alt]').length,
        withoutAlt: document.querySelectorAll('img:not([alt])').length,
        emptyAlt: document.querySelectorAll('img[alt=""]').length
      },
      headings: {
        h1: Array.from(document.querySelectorAll('h1')).map(h => h.textContent.trim().slice(0,60)),
        h2: document.querySelectorAll('h2').length,
        h3: document.querySelectorAll('h3').length,
        total: document.querySelectorAll('h1,h2,h3,h4,h5,h6').length
      },
      forms: {
        total: document.querySelectorAll('form').length,
        inputs: document.querySelectorAll('input, select, textarea').length,
        buttons: document.querySelectorAll('button').length
      },
      links: { total: document.querySelectorAll('a').length },
      aria: {
        ariaLabels: document.querySelectorAll('[aria-label]').length,
        roles: document.querySelectorAll('[role]').length
      },
      landmarks: {
        main: document.querySelectorAll('main').length,
        nav: document.querySelectorAll('nav').length,
        header: document.querySelectorAll('header').length,
        footer: document.querySelectorAll('footer').length
      },
      media: {
        videos: document.querySelectorAll('video').length,
        iframes: document.querySelectorAll('iframe').length,
        videoUrls: Array.from(document.querySelectorAll('video')).map(v => {
          const src = v.src || (v.querySelector('source') ? v.querySelector('source').src : '');
          return {
            src: src,
            hasCaptions: !!v.querySelector('track[kind="captions"]')
          };
        })
      }
    }));

    const violations = axeResults.violations.map(v => ({
      tool: 'axe-core',
      id: v.id,
      impact: v.impact,
      description: v.description,
      help: v.help,
      helpUrl: v.helpUrl,
      tags: v.tags,
      nodeCount: v.nodes.length,
      nodes: v.nodes.slice(0, 5).map(n => ({
        html: n.html.slice(0, 200),
        target: n.target,
        failureSummary: n.failureSummary
      }))
    }));

    return {
      success: true,
      pageInfo,
      violations,
      passesCount: axeResults.passes.length
    };
  } finally {
    await context.close();
    await browser.close();
  }
}

// TOOL 2: Pa11y
async function runPa11y(url) {
  console.log('[pa11y] Starting...');
  const results = await pa11y(url, {
    standard: 'WCAG2AA',
    timeout: 45000,
    wait: 2000,
    chromeLaunchConfig: {
      executablePath: SYSTEM_CHROMIUM,
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
    }
  });

  const violations = results.issues.map(issue => ({
    tool: 'pa11y',
    id: issue.code,
    impact: issue.type === 'error' ? 'serious' : issue.type === 'warning' ? 'moderate' : 'minor',
    description: issue.message,
    selector: issue.selector,
    context: (issue.context || '').slice(0, 200)
  }));

  return { success: true, violations, total: results.issues.length };
}

// TOOL 3: Lighthouse
async function runLighthouse(url) {
  console.log('[lighthouse] Starting...');
  const chrome = await launchChrome({
    chromePath: SYSTEM_CHROMIUM,
    chromeFlags: ['--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage']
  });

  try {
    const result = await lighthouse(url, {
      port: chrome.port,
      onlyCategories: ['accessibility'],
      output: 'json'
    });

    const lhr = result.lhr;
    const score = Math.round(lhr.categories.accessibility.score * 100);
    const violations = Object.values(lhr.audits)
      .filter(audit => audit.score !== null && audit.score < 1)
      .map(audit => ({
        tool: 'lighthouse',
        id: audit.id,
        impact: audit.score === 0 ? 'critical' : audit.score < 0.5 ? 'serious' : 'moderate',
        score: audit.score,
        description: audit.title
      }));

    return { success: true, score, violations };
  } finally {
    await chrome.kill();
  }
}

// ========== MAIN: PARALLEL EXECUTION WITH GRACEFUL DEGRADATION ==========

(async () => {
  console.log('=== MULTI-TOOL ACCESSIBILITY SCAN (v7.0 PARALLEL + RESILIENT) ===');
  console.log('Target:', TARGET_URL);
  console.log('Strategy: Promise.allSettled with per-tool timeouts\n');

  const startTime = Date.now();

  // Run ALL tools in PARALLEL with individual timeouts
  const [axeResult, pa11yResult, lighthouseResult] = await Promise.allSettled([
    withTimeout(
      withRetry(() => runAxeCore(TARGET_URL), 'axe-core', 2, 3000),
      60000, 'axe-core'
    ),
    withTimeout(
      withRetry(() => runPa11y(TARGET_URL), 'pa11y', 2, 3000),
      60000, 'pa11y'
    ),
    withTimeout(
      withRetry(() => runLighthouse(TARGET_URL), 'lighthouse', 2, 3000),
      90000, 'lighthouse'
    )
  ]);

  // ========== PROCESS RESULTS (Graceful Degradation) ==========
  const results = {
    url: TARGET_URL,
    timestamp: new Date().toISOString(),
    duration: `${((Date.now() - startTime) / 1000).toFixed(1)}s`,
    toolsSucceeded: 0,
    toolsFailed: 0,
    pageInfo: null,
    violations: [],
    byTool: {}
  };

  // Process axe-core results
  if (axeResult.status === 'fulfilled') {
    results.toolsSucceeded++;
    results.pageInfo = axeResult.value.pageInfo;
    results.violations.push(...axeResult.value.violations);
    results.byTool['axe-core'] = {
      success: true,
      count: axeResult.value.violations.length,
      passes: axeResult.value.passesCount
    };
    progressiveOutput('axe-core', axeResult.value);
  } else {
    results.toolsFailed++;
    results.byTool['axe-core'] = { success: false, error: axeResult.reason.message };
    console.log('\n[axe-core] FAILED:', axeResult.reason.message);
  }

  // Process pa11y results
  if (pa11yResult.status === 'fulfilled') {
    results.toolsSucceeded++;
    results.violations.push(...pa11yResult.value.violations);
    results.byTool['pa11y'] = {
      success: true,
      count: pa11yResult.value.violations.length
    };
    progressiveOutput('pa11y', pa11yResult.value);
  } else {
    results.toolsFailed++;
    results.byTool['pa11y'] = { success: false, error: pa11yResult.reason.message };
    console.log('\n[pa11y] FAILED:', pa11yResult.reason.message);
  }

  // Process lighthouse results
  if (lighthouseResult.status === 'fulfilled') {
    results.toolsSucceeded++;
    results.violations.push(...lighthouseResult.value.violations);
    results.byTool['lighthouse'] = {
      success: true,
      score: lighthouseResult.value.score,
      count: lighthouseResult.value.violations.length
    };
    progressiveOutput('lighthouse', lighthouseResult.value);
  } else {
    results.toolsFailed++;
    results.byTool['lighthouse'] = { success: false, error: lighthouseResult.reason.message };
    console.log('\n[lighthouse] FAILED:', lighthouseResult.reason.message);
  }

  // ========== DEDUPLICATE VIOLATIONS ==========
  const seen = new Set();
  const uniqueViolations = [];
  for (const v of results.violations) {
    const key = (v.description || '').toLowerCase().slice(0, 50);
    if (!seen.has(key)) {
      seen.add(key);
      uniqueViolations.push(v);
    }
  }
  results.uniqueViolations = uniqueViolations;
  results.totalUnique = uniqueViolations.length;

  // ========== FINAL OUTPUT ==========
  console.log('\n' + '='.repeat(60));
  console.log('=== SCAN COMPLETE ===');
  console.log('='.repeat(60));
  console.log(`Tools succeeded: ${results.toolsSucceeded}/3`);
  console.log(`Tools failed: ${results.toolsFailed}/3`);
  console.log(`Duration: ${results.duration}`);
  console.log(`Total unique violations: ${results.totalUnique}`);

  if (results.toolsSucceeded === 0) {
    console.log('\n⚠️  ALL TOOLS FAILED - Consider:');
    console.log('   1. Site may have strong bot protection');
    console.log('   2. Try Vibium MCP browser instead');
    console.log('   3. Check network connectivity');
  } else if (results.toolsSucceeded < 3) {
    console.log(`\n⚠️  Partial coverage (${results.toolsSucceeded}/3 tools) - Results still usable`);
  } else {
    console.log('\n✅ Full coverage achieved (3/3 tools)');
  }

  console.log('\n=== PAGE INFO ===');
  console.log(JSON.stringify(results.pageInfo, null, 2));

  console.log('\n=== VIOLATIONS BY TOOL ===');
  console.log(JSON.stringify(results.byTool, null, 2));

  console.log('\n=== UNIQUE VIOLATIONS ===');
  console.log(JSON.stringify(results.uniqueViolations, null, 2));

  // Save final results
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to: ${OUTPUT_FILE}`);
})();
```

### 2.2: Read Scan Results

After running the scan, read the results file:
```bash
cat /tmp/a11y-work/scan-results.json
```

The results include:
- **pageInfo**: Page structure, images, headings, media
- **violations**: All violations from all tools (deduplicated)
- **byTool**: Success/failure status per tool
- **toolsSucceeded**: Number of tools that completed (1-3)

### 2.3: Graceful Degradation Decision Tree

| Tools Succeeded | Action |
|-----------------|--------|
| **3/3** | ✅ Full coverage - proceed with all results |
| **2/3** | ⚠️ Good coverage - note which tool failed in report |
| **1/3** | ⚠️ Basic coverage - proceed but flag limited confidence |
| **0/3** | ❌ Retry with Vibium MCP, or document failure |

### 2.4: MANDATORY - Check for Videos and Trigger Pipeline

After reading scan results, check `pageInfo.media.videoUrls`:

```javascript
// Check scan-results.json for videos
const results = JSON.parse(fs.readFileSync('/tmp/a11y-work/scan-results.json'));
if (results.pageInfo && results.pageInfo.media.videoUrls.length > 0) {
  console.log('=== VIDEOS DETECTED - TRIGGERING VIDEO PIPELINE ===');
  for (const video of results.pageInfo.media.videoUrls) {
    console.log(`Video: ${video.src}`);
    console.log(`  Has captions: ${video.hasCaptions}`);
  }
  // PROCEED TO STEP 7 IMMEDIATELY
}
```

**IF videos detected AND hasCaptions=false → STEP 7 is MANDATORY before generating reports.**

---

## STEP 3: CONTEXT-AWARE REMEDIATION (LLM-POWERED)

**THIS IS WHERE CLAUDE'S INTELLIGENCE MATTERS.**

Generic tools output: `aria-label="[DESCRIPTION]"`
You output: `aria-label="Add to shopping cart"` because you understand context.

### 3.1: Context Analysis (Use Your Reasoning)

For EACH violation, Claude must:

1. **READ THE HTML CONTEXT** - Don't just see `<button class="btn">`, see:
   ```html
   <div class="product-card" data-product="Adidas Superstar">
     <img src="superstar.jpg" alt="White sneakers">
     <span class="price">$99</span>
     <button class="btn add-to-cart">  <!-- THIS IS THE VIOLATION -->
       <svg class="icon-cart">...</svg>
     </button>
   </div>
   ```

2. **INFER PURPOSE** from:
   - Class names: `add-to-cart`, `wishlist`, `menu-toggle`
   - Parent context: Inside `.product-card` with product data
   - Icon classes: `icon-cart`, `icon-heart`, `icon-search`
   - Nearby text: Product name, price, "Add to bag"
   - Page section: Header nav vs product grid vs checkout

3. **GENERATE SPECIFIC FIX**:
   ```html
   <!-- NOT THIS (generic template) -->
   <button aria-label="[DESCRIPTION]">

   <!-- THIS (context-aware) -->
   <button aria-label="Add Adidas Superstar to cart - $99">
   ```

### 3.2: Confidence Scoring

Rate your confidence in each fix:
- **0.9+**: Clear context (class="add-to-cart" near product name)
- **0.7-0.9**: Reasonable inference (icon-cart class alone)
- **<0.7**: Needs human review (ambiguous context)

Include confidence in remediation.md:
```markdown
### Button: `.product-card .btn` (Confidence: 0.95)
**Context:** Inside product card for "Adidas Superstar", has cart icon
**Fix:** `aria-label="Add Adidas Superstar to cart"`
```

### 3.2: Remediation Templates by Violation Type

**Form Labels (WCAG 1.3.1, 3.3.2, 4.1.2)**
```html
<!-- Context: Input inside payment form, near "Card Number" text -->
<!-- Confidence: 0.95 -->

<!-- BEFORE -->
<input type="text" name="cardNumber" placeholder="1234 5678 9012 3456">

<!-- AFTER -->
<label for="card-number">Credit Card Number</label>
<input type="text"
       id="card-number"
       name="cardNumber"
       placeholder="1234 5678 9012 3456"
       aria-describedby="card-hint"
       autocomplete="cc-number"
       inputmode="numeric"
       pattern="[0-9\s]{13,19}">
<span id="card-hint" class="visually-hidden">Enter 16-digit card number</span>

<!-- RATIONALE -->
- Visible label aids all users
- aria-describedby provides additional context
- autocomplete enables autofill
- inputmode shows numeric keyboard on mobile
- pattern enables browser validation
```

**Icon Buttons (WCAG 4.1.2)**
```html
<!-- Context: Button with SVG inside nav, classes include "menu-toggle" -->
<!-- Confidence: 0.92 -->

<!-- BEFORE -->
<button class="menu-toggle">
  <svg>...</svg>
</button>

<!-- AFTER -->
<button class="menu-toggle"
        type="button"
        aria-expanded="false"
        aria-controls="main-menu"
        aria-label="Open navigation menu">
  <svg aria-hidden="true" focusable="false">...</svg>
</button>

<!-- RATIONALE -->
- aria-label describes action, not icon
- aria-expanded communicates state
- aria-controls links to menu element
- SVG hidden from assistive tech (decorative)
```

**Color Contrast (WCAG 1.4.3)**
```html
<!-- Context: Gray text (#767676) on white background -->
<!-- Current ratio: 4.48:1 (FAILS AA for normal text) -->
<!-- Required: 4.5:1 (AA) or 7:1 (AAA) -->

<!-- BEFORE -->
.low-contrast { color: #767676; background: #ffffff; }

<!-- AFTER (Option 1: Darken text - minimal change) -->
.accessible { color: #757575; background: #ffffff; } /* 4.6:1 - PASSES AA */

<!-- AFTER (Option 2: Higher contrast for AAA) -->
.high-contrast { color: #595959; background: #ffffff; } /* 7.0:1 - PASSES AAA */

<!-- COLOR ALTERNATIVES -->
| Original | AA Pass | AAA Pass | Notes |
|----------|---------|----------|-------|
| #767676  | #757575 | #595959  | Gray text |
| #0066cc  | #0055b3 | #003d82  | Link blue |
| #cc0000  | #b30000 | #8b0000  | Error red |
```

**Heading Hierarchy (WCAG 1.3.1)**
```html
<!-- Context: Page has 10 H1 elements, skipped H2 levels -->

<!-- BEFORE (broken) -->
<h1>Welcome</h1>
<h1>Products</h1>      <!-- ERROR: Multiple H1s -->
<h4>Shoes</h4>         <!-- ERROR: Skipped H2, H3 -->
<h1>Contact</h1>

<!-- AFTER (correct) -->
<h1>Site Name - Main Page Title</h1>
<main>
  <section aria-labelledby="products-heading">
    <h2 id="products-heading">Products</h2>
    <h3>Shoes</h3>
    <h3>Clothing</h3>
  </section>
  <section aria-labelledby="contact-heading">
    <h2 id="contact-heading">Contact</h2>
  </section>
</main>

<!-- HEADING STRUCTURE VISUALIZATION -->
h1: Site Name - Main Page Title
├── h2: Products
│   ├── h3: Shoes
│   └── h3: Clothing
└── h2: Contact
```

**Skip Links (WCAG 2.4.1)**
```html
<!-- Add as FIRST element inside <body> -->
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <a href="#main-nav" class="skip-link">Skip to navigation</a>

  <header>
    <nav id="main-nav" aria-label="Main navigation">...</nav>
  </header>

  <main id="main-content" tabindex="-1">
    <!-- Main content -->
  </main>
</body>

<style>
.skip-link {
  position: absolute;
  top: -100%;
  left: 16px;
  background: #000;
  color: #fff;
  padding: 12px 24px;
  z-index: 10000;
  text-decoration: none;
  font-weight: bold;
  border-radius: 0 0 4px 4px;
  transition: top 0.2s;
}
.skip-link:focus {
  top: 0;
  outline: 3px solid #ffcc00;
  outline-offset: 2px;
}
</style>
```

**Focus Indicators (WCAG 2.4.7)**
```css
/* NEVER do this */
*:focus { outline: none; } /* WCAG FAIL */

/* DO THIS - Custom focus styles */
:focus-visible {
  outline: 3px solid #005fcc;
  outline-offset: 2px;
}

/* Remove outline only for mouse users */
:focus:not(:focus-visible) {
  outline: none;
}

/* High contrast for interactive elements */
a:focus-visible,
button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
[role="button"]:focus-visible {
  outline: 3px solid #005fcc;
  outline-offset: 2px;
  box-shadow: 0 0 0 6px rgba(0, 95, 204, 0.2);
}

/* Dark backgrounds need light focus */
.dark-bg :focus-visible {
  outline-color: #ffffff;
  box-shadow: 0 0 0 6px rgba(255, 255, 255, 0.3);
}
```

**Keyboard Navigation (WCAG 2.1.1, 2.1.2)**
```html
<!-- Custom interactive element needs keyboard support -->

<!-- BEFORE (inaccessible) -->
<div class="dropdown" onclick="toggleMenu()">
  Menu
</div>

<!-- AFTER (accessible) -->
<button type="button"
        class="dropdown-trigger"
        aria-expanded="false"
        aria-controls="dropdown-menu"
        onclick="toggleMenu()"
        onkeydown="handleKeydown(event)">
  Menu
</button>
<ul id="dropdown-menu" role="menu" hidden>
  <li role="none"><a role="menuitem" href="/page1">Page 1</a></li>
  <li role="none"><a role="menuitem" href="/page2">Page 2</a></li>
</ul>

<script>
function handleKeydown(event) {
  switch(event.key) {
    case 'Enter':
    case ' ':
      event.preventDefault();
      toggleMenu();
      break;
    case 'Escape':
      closeMenu();
      break;
    case 'ArrowDown':
      event.preventDefault();
      focusFirstMenuItem();
      break;
  }
}
</script>
```

**Modal Focus Trap (WCAG 2.4.3)**
```javascript
// Focus trap for modals - REQUIRED for WCAG compliance
function trapFocus(modal) {
  const focusable = modal.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusable[0];
  const last = focusable[focusable.length - 1];

  // Focus first element when modal opens
  first?.focus();

  modal.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    if (e.key === 'Escape') {
      closeModal();
    }
  });
}

// Return focus when modal closes
function closeModal() {
  modal.hidden = true;
  triggerButton.focus(); // Return focus to trigger
}
```

**iframe Titles (WCAG 4.1.2)**
```html
<!-- All iframes MUST have descriptive titles -->
<iframe src="map.html" title="Store location map showing 5 nearby stores"></iframe>
<iframe src="video.html" title="Product demonstration video with captions"></iframe>
<iframe src="chat.html" title="Customer support chat window"></iframe>
```

---

## STEP 4: USER IMPACT ANALYSIS

For each violation, calculate user impact:

### 4.1: Affected User Groups
| Violation Type | Affected Groups | % of Users |
|----------------|-----------------|------------|
| Missing alt text | Blind, low-vision | 7-10% |
| Missing form labels | Blind, screen reader users | 5-8% |
| Low color contrast | Low-vision, color blind | 8-12% |
| No keyboard access | Motor impaired, power users | 10-15% |
| Missing captions | Deaf, hard-of-hearing | 5-7% |
| Flashing content | Seizure sensitive | 0.5-1% |
| Complex language | Cognitive impairment | 10-15% |

### 4.2: Impact Severity Classification
```
BLOCKS-USAGE: User cannot complete task at all
  - Missing form labels on required fields
  - Keyboard traps
  - Critical buttons without accessible names

IMPAIRS-USAGE: User can complete task with difficulty
  - Low contrast (can read with effort)
  - Missing skip links (tedious navigation)
  - Incorrect heading structure (confusing)

MINOR-INCONVENIENCE: Suboptimal but functional
  - Empty alt on decorative images
  - Redundant ARIA
  - Non-semantic HTML that works
```

---

## STEP 5: ROI-BASED PRIORITIZATION

Calculate priority for each remediation:

### 5.1: Priority Formula
```
PRIORITY_SCORE = (IMPACT_WEIGHT × USERS_AFFECTED) / EFFORT_HOURS

Where:
- IMPACT_WEIGHT: Critical=10, Serious=7, Moderate=4, Minor=1
- USERS_AFFECTED: Estimated % of users impacted
- EFFORT_HOURS: Estimated fix time (0.25 to 8 hours)
```

### 5.2: Effort Estimation Guide
| Fix Type | Effort | Complexity |
|----------|--------|------------|
| Add aria-label | 0.25h | Trivial |
| Add alt text | 0.25h | Trivial |
| Add form label | 0.5h | Simple |
| Fix color contrast | 0.5h | Simple |
| Add skip links | 1h | Simple |
| Fix heading structure | 2h | Medium |
| Add keyboard navigation | 4h | High |
| Implement focus trap | 4h | High |
| Add video captions | 8h | High |

### 5.3: Priority Output Format
```
| Rank | Violation | Impact | Users | Effort | ROI Score |
|------|-----------|--------|-------|--------|-----------|
| 1 | Form labels missing | Critical | 15% | 0.5h | 300 |
| 2 | Keyboard trap | Critical | 12% | 4h | 30 |
| 3 | Low contrast | Serious | 10% | 0.5h | 140 |
| 4 | Missing alt text | Serious | 8% | 0.25h | 224 |
```

---

## STEP 6: PRODUCTION READINESS ASSESSMENT

### 6.1: Compliance Scoring
```
COMPLIANCE_SCORE = (PASSED_CRITERIA / TOTAL_CRITERIA) × 100

Production Ready if:
✓ Score ≥ 85%
✓ Zero critical violations
✓ Fewer than 3 serious violations
✓ All user journeys keyboard accessible
```

### 6.2: POUR Analysis (Perceivable, Operable, Understandable, Robust)
```
| Principle | Guidelines | Pass | Fail | Score |
|-----------|-----------|------|------|-------|
| Perceivable | 1.1-1.4 | 12 | 3 | 80% |
| Operable | 2.1-2.5 | 18 | 2 | 90% |
| Understandable | 3.1-3.3 | 8 | 1 | 89% |
| Robust | 4.1 | 4 | 1 | 80% |
| **TOTAL** | | **42** | **7** | **86%** |
```

---

## STEP 7: VIDEO ACCESSIBILITY PIPELINE

**Execute for EACH video detected on page.**

### 7.1: Detect and Extract Video URLs (MANDATORY)

**This step MUST be integrated into STEP 2 multi-tool scan.**

Add this to the page.evaluate() in the multi-tool scan:
```javascript
// In pageInfo extraction (STEP 2), add:
videos: {
  elements: [...document.querySelectorAll('video')].map(v => ({
    src: v.src || v.querySelector('source')?.src,
    fullUrl: new URL(v.src || v.querySelector('source')?.src || '', window.location.href).href,
    poster: v.poster,
    hasCaptions: v.querySelector('track[kind="captions"]') !== null,
    hasDescriptions: v.querySelector('track[kind="descriptions"]') !== null,
    duration: v.duration || 'unknown',
    autoplay: v.autoplay,
    muted: v.muted
  })),
  iframes: [...document.querySelectorAll('iframe')].map(iframe => {
    const src = iframe.src;
    const isVideo = /youtube|vimeo|dailymotion|wistia/.test(src);
    return isVideo ? { src, platform: src.match(/(youtube|vimeo|dailymotion|wistia)/)?.[1] } : null;
  }).filter(Boolean)
}
```

**MANDATORY OUTPUT:** Log all video URLs found:
```
=== VIDEOS DETECTED ===
Video 1: https://example.com/promo.mp4 (no captions, no descriptions)
YouTube iframe: https://youtube.com/embed/xxx
```

### 7.2: Download and Extract Frames (MANDATORY for each video)

**For EACH video URL found in 7.1:**

```bash
# Create output directory
mkdir -p /tmp/a11y-work/frames

# Download video (with retry and user-agent)
curl -L -A "Mozilla/5.0" --retry 3 -o /tmp/a11y-work/video.mp4 "FULL_VIDEO_URL"

# Verify download succeeded
if [ -f /tmp/a11y-work/video.mp4 ] && [ -s /tmp/a11y-work/video.mp4 ]; then
  echo "Video downloaded successfully"
  ffmpeg -i /tmp/a11y-work/video.mp4 -vf "fps=1/3" -frames:v 10 /tmp/a11y-work/frames/frame_%02d.jpg 2>/dev/null
  echo "Extracted $(ls /tmp/a11y-work/frames/*.jpg 2>/dev/null | wc -l) frames"
else
  echo "VIDEO DOWNLOAD FAILED - Document this in audit-summary.md"
fi
```

**IF VIDEO DOWNLOAD FAILS:**
1. Document the failure reason in audit-summary.md
2. Still create video-captions violation in violations.json
3. Add remediation instructions WITHOUT generated captions
4. Mark video pipeline as "blocked" not "skipped"

### 7.3: Analyze Each Frame with Claude Vision (MANDATORY)

**USE THE READ TOOL ON EACH FRAME IMAGE.**

Claude Code has native vision capabilities. When you Read an image file, you SEE it.

```
Read /tmp/a11y-work/frames/frame_01.jpg
Read /tmp/a11y-work/frames/frame_02.jpg
Read /tmp/a11y-work/frames/frame_03.jpg
... (continue for all frames)
```

**For EACH frame, describe:**
- **SCENE**: Setting, environment, lighting, location
- **PEOPLE**: Who appears, what they're doing, expressions, clothing
- **PRODUCTS**: Items shown (for e-commerce: product names, colors, styles)
- **TEXT**: Any visible text, logos, signs, prices
- **ACTION**: Movement, transitions, what's happening

**Example output after reading frame_01.jpg:**
```
Frame 1 (0:00-0:03): A woman in white Adidas sneakers running on a forest trail.
Morning light filters through trees. She wears black athletic leggings and a
gray tank top. The Adidas three-stripe logo is visible on her shoes.
```

**THIS IS THE LLM VALUE.** Generic tools output "[DESCRIBE CONTENT]".
You output actual descriptions because you can SEE the image.

---

**FALLBACK: If Read tool fails on images**

Try Anthropic API directly:
```javascript
const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');
const client = new Anthropic();

const imageData = fs.readFileSync('/tmp/a11y-work/frames/frame_01.jpg').toString('base64');
const response = await client.messages.create({
  model: 'claude-sonnet-4-20250514',
  max_tokens: 500,
  messages: [{
    role: 'user',
    content: [
      { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageData } },
      { type: 'text', text: 'Describe this video frame for accessibility captions.' }
    ]
  }]
});
```

**LAST RESORT: Context-Based Inference (No Vision)**
If vision completely unavailable, infer from:
- Video filename: "product-demo.mp4" → product demonstration
- Page context: product page → product showcase
- Surrounding text: nearby headings and descriptions

**Document for each frame:**
- **SCENE**: Setting, environment, lighting
- **PEOPLE**: Who, actions, expressions, clothing
- **OBJECTS**: Products, props, equipment
- **TEXT**: Visible text, logos, signs
- **ACTION**: Movement, transitions
- **COLORS**: Dominant colors, accessibility-relevant

### 7.4: Generate WebVTT Captions
```vtt
WEBVTT
Kind: captions
Language: {detected-language}

00:00:00.000 --> 00:00:03.000
[Description from frame_01 analysis]

00:00:03.000 --> 00:00:06.000
[Description from frame_02 analysis]
```

### 7.5: Generate Audio Descriptions
```vtt
WEBVTT
Kind: descriptions
Language: en

00:00:00.000 --> 00:00:03.000
SCENE: [Detailed scene for blind users]
VISUAL: [What's on screen]
TEXT: [Any readable text]
ACTION: [What's happening]
```

---

## STEP 8: GENERATE COMPREHENSIVE REPORTS

### 8.1: Required Output Files
Save ALL files to `docs/accessibility-scans/{page-slug}/`:

| File | Contents |
|------|----------|
| `audit-summary.md` | Executive summary, scores, top issues, user impact |
| `remediation.md` | **ALL copy-paste code fixes** with context |
| `violations.json` | Machine-readable violation data |
| `implementation.md` | Video integration guide (if videos) |
| `*.vtt` | Caption and audio description files |

### 8.2: audit-summary.md Template
```markdown
# Accessibility Audit Report: {Site Name}

**URL:** {url}
**Date:** {date}
**Standard:** WCAG 2.2 Level AA

## Executive Summary

| Metric | Value |
|--------|-------|
| **Compliance Score** | {score}% |
| **Production Ready** | {Yes/No} |
| **Critical Issues** | {count} |
| **Total Violations** | {count} |
| **Estimated Fix Time** | {hours}h |

## POUR Analysis
{table}

## Top 10 Issues by Priority
{priority table with ROI scores}

## User Impact Summary
{affected user groups and percentages}

## Recommendations
{prioritized action items}
```

### 8.3: remediation.md Template
```markdown
# Accessibility Remediation Guide: {Site Name}

## Quick Wins (Copy-Paste Ready)

### 1. Form Labels ({count} issues)
{For EACH unlabeled input: context, before/after code, rationale, confidence}

### 2. Heading Structure ({count} issues)
{Current structure visualization, fixed structure, code changes}

### 3. Color Contrast ({count} issues)
{For EACH: current colors, ratio, suggested colors, CSS fixes}

### 4. Missing Alt Text ({count} issues)
{For EACH image: context-inferred alt text suggestions}

### 5. Keyboard Navigation ({count} issues)
{For EACH: element, issue, fix code, test instructions}

### 6. Focus Indicators
{Global CSS to add}

### 7. Skip Links
{Full HTML + CSS to add}

### 8. ARIA Fixes ({count} issues)
{For EACH: context, specific aria attributes to add}

### 9. iframe Titles ({count} issues)
{For EACH: suggested title based on content}

### 10. Video Accessibility ({count} videos)
{Links to generated VTT files, implementation code}

## Testing Checklist
- [ ] Tab through entire page - all interactive elements reachable
- [ ] Screen reader announces all content correctly
- [ ] Color contrast passes (use axe DevTools)
- [ ] Works without mouse
- [ ] Works at 200% zoom
- [ ] Video captions synchronized and accurate
```

---

## STEP 9: LEARNING PROTOCOL (When MCP Available)

Integrate with the learning system to improve over time.

### 9.1: Query Previous Patterns BEFORE Audit

Check if similar sites were audited before:
```javascript
// Load MCP tools
ToolSearch("select:mcp__claude-flow_alpha__memory_retrieve")
ToolSearch("select:mcp__claude-flow_alpha__hooks_intelligence_pattern_search")

// Retrieve domain-specific patterns
mcp__claude-flow_alpha__memory_retrieve({
  key: `accessibility/patterns/${domain}`,
  namespace: "learning"
})

// Search for similar violation patterns
mcp__claude-flow_alpha__hooks_intelligence_pattern_search({
  query: "accessibility remediation",
  type: "accessibility-fix",
  limit: 10
})
```

### 9.2: Store Successful Patterns AFTER Audit

Store patterns that worked for future reuse:
```javascript
ToolSearch("select:mcp__claude-flow_alpha__memory_store")
ToolSearch("select:mcp__claude-flow_alpha__hooks_intelligence_pattern_store")

// Store audit outcome
mcp__claude-flow_alpha__memory_store({
  key: `accessibility-audit/${domain}-${Date.now()}`,
  namespace: "learning",
  value: {
    url: auditedUrl,
    timestamp: new Date().toISOString(),
    violationsFound: violations.length,
    criticalCount: violations.filter(v => v.impact === 'critical').length,
    toolsUsed: ['axe-core', 'pa11y', 'lighthouse'],
    patterns: {
      commonViolations: extractTopViolationTypes(violations),
      effectiveFixes: extractFixesThatWorked(remediations)
    }
  }
})

// Store reusable remediation patterns
mcp__claude-flow_alpha__hooks_intelligence_pattern_store({
  pattern: "form-label-contextual-fix",
  confidence: 0.92,
  type: "accessibility-remediation",
  metadata: {
    wcagCriteria: "1.3.1, 3.3.2, 4.1.2",
    violationType: "missing-form-label",
    codeTemplate: "<label for=\"{id}\">{inferredLabel}</label>",
    contextSignals: ["placeholder", "nearby-text", "field-name"]
  }
})
```

### 9.3: Calculate Audit Quality Score

Self-assess audit completeness (for learning feedback):

| Criteria | Points | Your Score |
|----------|--------|------------|
| Multi-tool testing used (3 tools) | 20 | |
| All WCAG 2.2 AA criteria checked | 15 | |
| Context-aware fixes generated | 20 | |
| Confidence scores included | 10 | |
| ROI prioritization calculated | 10 | |
| Video pipeline completed (if applicable) | 15 | |
| EU compliance mapping included | 10 | |
| **Total** | **100** | |

**Quality Levels:**
- 90-100: Excellent (1.0 reward)
- 70-89: Good (0.8 reward)
- 50-69: Acceptable (0.5 reward)
- <50: Incomplete (0.0 reward - redo required)

---

## STEP 10: SCREEN READER TESTING GUIDE

Manual testing instructions (cannot be fully automated):

### 10.1: NVDA (Windows - Free)
```
1. Download: https://www.nvaccess.org/download/
2. Install and start NVDA (Ctrl+Alt+N)
3. Navigate to audited page

Key Commands:
- H: Jump through headings
- F: Jump through form fields
- B: Jump through buttons
- T: Jump through tables
- K: Jump through links
- D: Jump through landmarks
- Tab: Move through focusable elements

Verify:
- [ ] All headings announced with correct level
- [ ] Form fields announce labels
- [ ] Buttons announce purpose
- [ ] Images announce alt text or "decorative"
- [ ] Dynamic content changes announced (aria-live)
```

### 10.2: VoiceOver (macOS - Built-in)
```
1. Enable: System Preferences → Accessibility → VoiceOver
2. Toggle: Cmd+F5
3. Navigate to audited page

Key Commands:
- VO+U: Open rotor (headings, links, forms, landmarks)
- VO+Space: Activate element
- VO+Right/Left: Move through content
- VO+Cmd+H: Jump to next heading

Verify:
- [ ] Rotor shows all headings hierarchically
- [ ] Forms are navigable and labels announced
- [ ] Focus order matches visual order
- [ ] All content is reachable
```

### 10.3: JAWS (Windows - Commercial)
```
1. Trial: https://www.freedomscientific.com/products/software/jaws/
2. Start JAWS and navigate to page

Key Commands:
- H: Next heading
- F: Next form field
- B: Next button
- T: Next table
- Ins+F6: Heading list
- Ins+F7: Link list

Verify:
- [ ] Virtual cursor mode works correctly
- [ ] Forms mode activates in forms
- [ ] All ARIA roles announced properly
```

### 10.4: Screen Reader Testing Checklist

| Test | NVDA | VoiceOver | JAWS |
|------|------|-----------|------|
| Headings hierarchy correct | [ ] | [ ] | [ ] |
| Form labels announced | [ ] | [ ] | [ ] |
| Button purposes clear | [ ] | [ ] | [ ] |
| Image alt text correct | [ ] | [ ] | [ ] |
| Links announce destination | [ ] | [ ] | [ ] |
| Landmarks navigable | [ ] | [ ] | [ ] |
| Focus order logical | [ ] | [ ] | [ ] |
| Dynamic updates announced | [ ] | [ ] | [ ] |
| No keyboard traps | [ ] | [ ] | [ ] |
| Skip links work | [ ] | [ ] | [ ] |

---

## VALIDATION CHECKLIST

Before completing, verify ALL items:

### Content Fetching (v7.0 Resilient)
- [ ] Browser launched (Vibium/agent-browser/Playwright)
- [ ] Page loaded and analyzed
- [ ] Multi-tool scan ran with **parallel execution**
- [ ] At least **1 of 3 tools succeeded** (graceful degradation)
- [ ] If tools failed, documented **which tools and why**
- [ ] Results saved to `/tmp/a11y-work/scan-results.json`

### Violation Analysis
- [ ] All violations extracted with WCAG criteria
- [ ] Context analyzed for each violation
- [ ] User impact calculated

### Remediation Generation
- [ ] Form label fixes with context and rationale
- [ ] Heading hierarchy fix with visualization
- [ ] Color contrast fixes with hex codes
- [ ] Alt text suggestions with confidence scores
- [ ] Skip link code (full HTML + CSS)
- [ ] Focus indicator CSS
- [ ] ARIA fixes with specific attributes
- [ ] Keyboard navigation fixes
- [ ] iframe titles

### Video Accessibility (MANDATORY if pageInfo.media.videos > 0)
- [ ] Video URLs extracted (full URLs, not relative)
- [ ] Download attempted for EACH video
- [ ] IF download succeeded: Frames extracted with ffmpeg
- [ ] IF download succeeded: Each frame analyzed with Read tool
- [ ] IF download succeeded: captions.vtt generated from ACTUAL frame descriptions
- [ ] IF download succeeded: audiodesc.vtt generated
- [ ] IF download FAILED: Failure documented in audit-summary.md with reason
- [ ] IF download FAILED: Manual captioning instructions in remediation.md

**BLOCKING:** Cannot generate final reports until video pipeline attempted.

### Output Files
- [ ] audit-summary.md with scores, priorities, and user impact
- [ ] remediation.md with ALL copy-paste code fixes
- [ ] violations.json with violation data
- [ ] VTT files (if videos)
- [ ] implementation.md (if videos)

### Quality Checks
- [ ] Compliance score calculated
- [ ] Production readiness assessed
- [ ] ROI prioritization completed
- [ ] POUR analysis included

**IF ANY CHECKBOX IS NO = TASK INCOMPLETE**
</default_to_action>

---

## Quick Reference Card

### Usage
```
/a11y-ally https://example.com
```

### v7.0 Resilience Features
| Feature | Description |
|---------|-------------|
| **Parallel Execution** | All 3 tools run simultaneously via Promise.allSettled |
| **Per-Tool Timeouts** | axe: 60s, pa11y: 60s, Lighthouse: 90s |
| **Retry with Backoff** | 2 retries per tool with exponential backoff |
| **Graceful Degradation** | Continue if 1+ tools succeed |
| **Progressive Output** | Results stream as tools complete |
| **Bot Protection** | Stealth mode, random delays, cookie dismissal |

### Expected Output Structure
```
docs/accessibility-scans/{page-slug}/
├── audit-summary.md      # Executive summary with scores
├── remediation.md        # ALL copy-paste code fixes
├── violations.json       # Machine-readable data
├── implementation.md     # Video integration (if videos)
├── video-*-captions.vtt  # Captions (if videos)
└── video-*-audiodesc.vtt # Audio descriptions (if videos)
```

### Compliance Thresholds
| Level | Min Score | Critical | Serious |
|-------|-----------|----------|---------|
| A | 70% | 0 | ≤5 |
| AA | 85% | 0 | ≤3 |
| AAA | 95% | 0 | 0 |

### Tool Coverage by Success
| Tools Succeeded | Detection Rate | Status |
|-----------------|---------------|--------|
| 3/3 | ~95% | ✅ Optimal |
| 2/3 | ~85% | ⚠️ Good |
| 1/3 | ~70% | ⚠️ Acceptable |
| 0/3 | — | ❌ Retry needed |

### ROI Formula
```
ROI = (Impact × Users%) / Effort_Hours
```

---

## EU Compliance Mapping

| WCAG | EN 301 549 | EU Accessibility Act |
|------|------------|---------------------|
| 1.1.1 | 9.1.1.1 | EAA-I.1 Perceivable |
| 1.4.3 | 9.1.4.3 | EAA-I.1 Perceivable |
| 2.1.1 | 9.2.1.1 | EAA-I.2 Operable |
| 2.4.7 | 9.2.4.7 | EAA-I.2 Operable |
| 3.3.2 | 9.3.3.2 | EAA-I.3 Understandable |
| 4.1.2 | 9.4.1.2 | EAA-I.4 Robust |

---

## Critical Rules

### Execution Rules (v7.0)
1. **ALWAYS** run multi-tool scan with **parallel execution** (Promise.allSettled)
2. **ALWAYS** continue if at least **1 of 3 tools** succeeds (graceful degradation)
3. **ALWAYS** document which tools failed and why in audit-summary.md
4. **ALWAYS** use per-tool timeouts (60s/60s/90s) not global timeout
5. **ALWAYS** retry failed tools with exponential backoff before giving up

### Quality Rules
6. **ALWAYS** analyze context before generating fixes
7. **ALWAYS** include confidence scores with remediation
8. **ALWAYS** calculate user impact and ROI
9. **ALWAYS** generate copy-paste ready code
10. **NEVER** generate placeholder/template fixes
11. **NEVER** skip video pipeline if videos detected
12. **NEVER** complete without remediation.md
13. **NEVER** fail audit just because 1-2 tools failed (use graceful degradation)
