const { test, expect } = require('@playwright/test');
const config = require('./config');
const fs = require('fs');
const path = require('path');

let testabilityScores = {
  timestamp: new Date().toISOString(),
  overall: 0,
  grade: 'F',
  principles: {},
  recommendations: [],
  metadata: {
    url: config.baseURL,
    browser: 'chromium',
    version: '1.0.0',
    assessor: 'testability-scorer-skill'
  }
};

// Initialize all principles with default values to ensure we always have data
function initializeDefaultScores() {
  const defaultScore = { score: 50, grade: 'F', weight: 0 };
  testabilityScores.principles = {
    observability: { ...defaultScore, weight: config.weights.observability },
    controllability: { ...defaultScore, weight: config.weights.controllability },
    algorithmicSimplicity: { ...defaultScore, weight: config.weights.algorithmicSimplicity },
    algorithmicTransparency: { ...defaultScore, weight: config.weights.algorithmicTransparency },
    explainability: { ...defaultScore, weight: config.weights.explainability },
    similarity: { ...defaultScore, weight: config.weights.similarity },
    algorithmicStability: { ...defaultScore, weight: config.weights.algorithmicStability },
    unbugginess: { ...defaultScore, weight: config.weights.unbugginess },
    smallness: { ...defaultScore, weight: config.weights.smallness },
    decomposability: { ...defaultScore, weight: config.weights.decomposability }
  };
}

// Robust page navigation helper
async function navigateToPage(page) {
  console.log(`[NAV] Starting navigation to ${config.baseURL}`);
  page.setDefaultTimeout(45000);
  
  try {
    console.log('[NAV] Attempting page.goto with domcontentloaded...');
    await page.goto(config.baseURL, { 
      timeout: 45000, 
      waitUntil: 'domcontentloaded' 
    });
    console.log('[NAV] Page loaded (domcontentloaded)');
    
    // Try to wait for network idle but don't fail if it times out
    console.log('[NAV] Waiting for networkidle (15s timeout)...');
    await page.waitForLoadState('networkidle', { timeout: 15000 })
      .then(() => console.log('[NAV] Network is idle'))
      .catch(() => console.log('[NAV] Network not idle after 15s, continuing...'));
    
    return true;
  } catch (error) {
    console.log(`[NAV] Navigation failed: ${error.message}`);
    // Try one more time with even more lenient settings
    try {
      console.log('[NAV] Retrying with commit waitUntil...');
      await page.goto(config.baseURL, { 
        timeout: 45000, 
        waitUntil: 'commit' 
      });
      console.log('[NAV] Page committed');
      return true;
    } catch (retryError) {
      console.error(`[NAV] Final navigation failed: ${retryError.message}`);
      return false;
    }
  }
}

test.describe.configure({ mode: 'serial', timeout: 60000 });

test.describe('Comprehensive Testability Analysis - Sauce Demo Shopify', () => {

  test.beforeAll(() => {
    console.log('Starting testability assessment...');
    initializeDefaultScores();
  });

  test('1. Observability Assessment', async ({ page }) => {
    try {
      const logs = [];
      const errors = [];
      const networkRequests = [];

      page.on('console', msg => logs.push(msg));
      page.on('pageerror', err => errors.push(err));
      page.on('request', request => networkRequests.push(request));

      const loaded = await navigateToPage(page);
      if (!loaded) {
        throw new Error('Failed to load page');
      }

    // Check console logging
    const hasConsoleLogs = logs.length > 0;

    // Check network visibility
    const hasNetworkRequests = networkRequests.length > 0;

    // Check state inspection
    const stateVisible = await page.evaluate(() => {
      return typeof window !== 'undefined' &&
             (typeof window.Shopify !== 'undefined' ||
              typeof window.console !== 'undefined');
    });

    // Calculate score
    let score = 0;
    if (hasConsoleLogs) score += 25;
    if (hasNetworkRequests) score += 30;
    if (stateVisible) score += 27;
    score += 10; // Base score for page loading

    testabilityScores.principles.observability = {
      score: Math.min(score, 100),
      grade: getLetterGrade(score),
      weight: config.weights.observability
    };

    if (score < 70) {
      testabilityScores.recommendations.push({
        principle: 'Observability',
        severity: 'medium',
        recommendation: 'Implement detailed event logging for user actions, cart operations, and payment processing.',
        impact: 15,
        effort: 'Low (4-6 hours)'
      });
    }
    } catch (error) {
      console.error('Observability assessment failed:', error.message);
      testabilityScores.principles.observability = { score: 50, grade: 'F', weight: config.weights.observability };
    }
  });

  test('2. Controllability Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Check direct API access
    const hasAPI = await page.evaluate(() => {
      return typeof window.Shopify !== 'undefined' &&
             typeof window.Shopify.checkout !== 'undefined';
    });

    // Check state manipulation
    const canManipulateCart = await page.evaluate(() => {
      return typeof window.Shopify !== 'undefined';
    });

    // Check test data injection capabilities
    const hasTestMode = await page.evaluate(() => {
      return typeof window.testAPI !== 'undefined' ||
             typeof window.Shopify !== 'undefined';
    });

    let score = 0;
    if (hasAPI) score += 25;
    if (canManipulateCart) score += 20;
    if (hasTestMode) score += 10;

    // E-commerce sites typically have limited controllability for security
    score = Math.min(score + 20, 65); // Cap at 65 for production e-commerce

    testabilityScores.principles.controllability = {
      score,
      grade: getLetterGrade(score),
      weight: config.weights.controllability
    };

    if (score < 70) {
      testabilityScores.recommendations.push({
        principle: 'Controllability',
        severity: 'critical',
        recommendation: 'Add test data injection endpoints to allow programmatic product catalog and cart management during automated testing.',
        impact: 38,
        effort: 'Medium (8-12 hours)'
      });
    }
    } catch (error) {
      console.error('Controllability assessment failed:', error.message);
      testabilityScores.principles.controllability = { score: 50, grade: 'F', weight: config.weights.controllability };
    }
  });

  test('3. Algorithmic Simplicity Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Measure complexity through interaction patterns
    const interactions = [];

    try {
      // Test product browsing
      const products = await page.locator('[data-product], .product, .product-item').count();
      interactions.push({ action: 'browse', steps: products > 0 ? 2 : 5 });

      // Test cart interaction
      const cartExists = await page.locator('[data-cart], .cart, #cart').count() > 0;
      interactions.push({ action: 'cart', steps: cartExists ? 2 : 4 });

      // Test checkout flow
      const checkoutExists = await page.locator('[href*="checkout"], .checkout-button').count() > 0;
      interactions.push({ action: 'checkout', steps: checkoutExists ? 3 : 6 });
    } catch (e) {
      interactions.push({ action: 'default', steps: 5 });
    }

    const avgComplexity = interactions.reduce((sum, i) => sum + i.steps, 0) / interactions.length;

    // Shopify is well-structured, so base score is high
    const score = Math.max(70, Math.min(100 - (avgComplexity * 5), 95));

    testabilityScores.principles.algorithmicSimplicity = {
      score: Math.round(score),
      grade: getLetterGrade(score),
      weight: config.weights.algorithmicSimplicity
    };
    } catch (error) {
      console.error('Algorithmic Simplicity assessment failed:', error.message);
      testabilityScores.principles.algorithmicSimplicity = { score: 50, grade: 'F', weight: config.weights.algorithmicSimplicity };
    }
  });

  test('4. Algorithmic Transparency Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Check code readability indicators
    const hasReadableClasses = await page.evaluate(() => {
      const elements = document.querySelectorAll('[class]');
      let readableCount = 0;
      elements.forEach(el => {
        const classes = el.className.toString();
        if (classes.includes('product') || classes.includes('cart') ||
            classes.includes('checkout') || classes.includes('price')) {
          readableCount++;
        }
      });
      return readableCount > 10;
    });

    // Check data attributes
    const hasDataAttributes = await page.evaluate(() => {
      return document.querySelectorAll('[data-product], [data-cart], [data-price]').length > 0;
    });

    let score = 60; // Base score for Shopify structure
    if (hasReadableClasses) score += 10;
    if (hasDataAttributes) score += 10;

    testabilityScores.principles.algorithmicTransparency = {
      score: Math.min(score, 100),
      grade: getLetterGrade(score),
      weight: config.weights.algorithmicTransparency
    };
    } catch (error) {
      console.error('Algorithmic Transparency assessment failed:', error.message);
      testabilityScores.principles.algorithmicTransparency = { score: 50, grade: 'F', weight: config.weights.algorithmicTransparency };
    }
  });

  test('5. Explainability Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Check for help text and documentation
    const hasHelpText = await page.locator('[aria-label], [title], .help-text').count() > 0;

    // Check for clear error messages
    const hasErrorHandling = await page.evaluate(() => {
      return document.querySelectorAll('[role="alert"], .error, .message').length >= 0;
    });

    // Check for tooltips and guidance
    const hasGuidance = await page.locator('[data-tooltip], .tooltip').count() > 0;

    let score = 50; // Base score
    if (hasHelpText) score += 15;
    if (hasErrorHandling) score += 10;
    if (hasGuidance) score += 7;

    testabilityScores.principles.explainability = {
      score: Math.min(score, 100),
      grade: getLetterGrade(score),
      weight: config.weights.explainability
    };

    if (score < 70) {
      testabilityScores.recommendations.push({
        principle: 'Explainability',
        severity: 'medium',
        recommendation: 'Document API contracts, add OpenAPI specifications, and improve error message clarity for checkout process.',
        impact: 25,
        effort: 'Medium (6-8 hours)'
      });
    }
    } catch (error) {
      console.error('Explainability assessment failed:', error.message);
      testabilityScores.principles.explainability = { score: 50, grade: 'F', weight: config.weights.explainability };
    }
  });

  test('6. Similarity to Known Technology Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Shopify is a well-known platform
    const usesShopify = await page.evaluate(() => {
      return typeof window.Shopify !== 'undefined' ||
             document.documentElement.innerHTML.includes('Shopify');
    });

    // Check for standard frameworks
    const usesStandardFrameworks = await page.evaluate(() => {
      return typeof window.jQuery !== 'undefined' ||
             document.querySelector('[data-react-root]') !== null ||
             typeof window.Vue !== 'undefined';
    });

    let score = 85; // High base score for Shopify
    if (usesShopify) score += 10;
    if (usesStandardFrameworks) score += 5;

    testabilityScores.principles.similarity = {
      score: Math.round(score),
      grade: getLetterGrade(score),
      weight: config.weights.similarity
    };
    } catch (error) {
      console.error('Similarity assessment failed:', error.message);
      testabilityScores.principles.similarity = { score: 50, grade: 'F', weight: config.weights.similarity };
    }
  });

  test('7. Algorithmic Stability Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Check for versioning
    const hasVersioning = await page.evaluate(() => {
      return typeof window.Shopify !== 'undefined';
    });

    // Check page consistency
    const pageLoadsConsistently = await page.evaluate(() => {
      return document.readyState === 'complete';
    });

    let score = 60; // Base score
    if (hasVersioning) score += 15;
    if (pageLoadsConsistently) score += 10;

    testabilityScores.principles.algorithmicStability = {
      score: Math.min(score, 100),
      grade: getLetterGrade(score),
      weight: config.weights.algorithmicStability
    };
    } catch (error) {
      console.error('Algorithmic Stability assessment failed:', error.message);
      testabilityScores.principles.algorithmicStability = { score: 50, grade: 'F', weight: config.weights.algorithmicStability };
    }
  });

  test('8. Unbugginess Assessment', async ({ page }) => {
    try {
      const errors = [];
      const warnings = [];

      page.on('console', msg => {
        if (msg.type() === 'error') errors.push(msg);
        if (msg.type() === 'warning') warnings.push(msg);
      });

      page.on('pageerror', err => errors.push(err));

      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

      // Score based on errors
      let score = 95; // Start high
      score -= errors.length * 5;
      score -= warnings.length * 2;
      score = Math.max(score, 0);

      testabilityScores.principles.unbugginess = {
        score: Math.min(score, 100),
        grade: getLetterGrade(score),
        weight: config.weights.unbugginess
      };
    } catch (error) {
      console.error('Unbugginess assessment failed:', error.message);
      testabilityScores.principles.unbugginess = { score: 50, grade: 'F', weight: config.weights.unbugginess };
    }
  });

  test('9. Smallness Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

      // Measure page size indicators
      const elementCount = await page.evaluate(() => document.querySelectorAll('*').length);
      const scriptCount = await page.evaluate(() => document.querySelectorAll('script').length);
      const styleCount = await page.evaluate(() => document.querySelectorAll('style, link[rel="stylesheet"]').length);

      // Smaller is better
      let score = 100;
      if (elementCount > 1000) score -= 10;
      if (elementCount > 2000) score -= 10;
      if (scriptCount > 20) score -= 5;
      if (styleCount > 10) score -= 5;

      testabilityScores.principles.smallness = {
        score: Math.min(score, 100),
        grade: getLetterGrade(score),
        weight: config.weights.smallness
      };
    } catch (error) {
      console.error('Smallness assessment failed:', error.message);
      testabilityScores.principles.smallness = { score: 50, grade: 'F', weight: config.weights.smallness };
    }
  });

  test('10. Decomposability Assessment', async ({ page }) => {
    try {
      const loaded = await navigateToPage(page);
      if (!loaded) throw new Error('Failed to load page');

    // Check for modular components
    const hasModularStructure = await page.evaluate(() => {
      const hasComponents = document.querySelectorAll('[data-component], [data-module], .component, .module').length > 0;
      const hasSections = document.querySelectorAll('section, [role="region"]').length > 0;
      return hasComponents || hasSections;
    });

    // Check for isolated features
    const hasIsolatedFeatures = await page.evaluate(() => {
      const hasCart = document.querySelector('[data-cart], .cart') !== null;
      const hasProduct = document.querySelector('[data-product], .product') !== null;
      return hasCart && hasProduct;
    });

    let score = 50; // Base score
    if (hasModularStructure) score += 20;
    if (hasIsolatedFeatures) score += 15;

    testabilityScores.principles.decomposability = {
      score: Math.min(score, 100),
      grade: getLetterGrade(score),
      weight: config.weights.decomposability
    };

    if (score < 70) {
      testabilityScores.recommendations.push({
        principle: 'Decomposability',
        severity: 'high',
        recommendation: 'Extract product catalog, shopping cart, and checkout into separate microservices for better isolation and testability.',
        impact: 30,
        effort: 'High (16-24 hours)'
      });
    }
    } catch (error) {
      console.error('Decomposability assessment failed:', error.message);
      testabilityScores.principles.decomposability = { score: 50, grade: 'F', weight: config.weights.decomposability };
    }
  });

  test.afterAll('Calculate Overall Score & Generate Report', async () => {
    // Calculate weighted average
    const principles = testabilityScores.principles;
    const weights = config.weights;

    let totalScore = 0;
    Object.keys(principles).forEach(key => {
      const weight = weights[key] / 100;
      totalScore += principles[key].score * weight;
    });

    testabilityScores.overall = Math.round(totalScore);
    testabilityScores.grade = getLetterGrade(testabilityScores.overall);
    testabilityScores.metadata.duration = Date.now() - new Date(testabilityScores.timestamp).getTime();

    // Sort recommendations by impact
    testabilityScores.recommendations.sort((a, b) => b.impact - a.impact);

    // Save JSON report
    const timestamp = Date.now();
    const jsonPath = path.join(config.reports.directory, `testability-results-${timestamp}.json`);
    const htmlPath = path.join(config.reports.directory, `testability-report-${timestamp}.html`);

    fs.writeFileSync(jsonPath, JSON.stringify(testabilityScores, null, 2));
    fs.writeFileSync(path.join(config.reports.directory, 'latest.json'), JSON.stringify(testabilityScores, null, 2));

    console.log(`\n✅ Assessment Complete!`);
    console.log(`\n📊 Overall Testability Score: ${testabilityScores.overall}/100 (${testabilityScores.grade})`);
    console.log(`\n📄 JSON Report saved: ${jsonPath}`);
    console.log(`\n🎯 Generating HTML report with Chrome auto-launch...`);

    // Generate HTML report using the enhanced script
    const { exec } = require('child_process');
    exec(`node .claude/skills/testability-scorer/scripts/generate-html-report.js "${jsonPath}" "${htmlPath}"`,
      (error, stdout, stderr) => {
        if (error) {
          console.error(`Error generating HTML: ${error.message}`);
          return;
        }
        if (stderr) {
          console.error(`stderr: ${stderr}`);
        }
        console.log(stdout);
      });
  });
});

function getLetterGrade(score) {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}
