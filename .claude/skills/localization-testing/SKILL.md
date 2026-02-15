---
name: localization-testing
description: "Internationalization (i18n) and localization (l10n) testing for global products including translations, locale formats, RTL languages, and cultural appropriateness. Use when launching in new markets or building multi-language products."
category: specialized-testing
priority: medium
tokenEstimate: 800
agents: [qe-test-generator, qe-test-executor, qe-visual-tester]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [localization, i18n, l10n, translation, rtl, unicode, locale]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/localization-testing.yaml

---

# Localization & Internationalization Testing

<default_to_action>
When testing multi-language/region support:
1. VERIFY translation coverage (all strings translated)
2. TEST locale-specific formats (date, time, currency, numbers)
3. VALIDATE RTL layout (Arabic, Hebrew)
4. CHECK character encoding (UTF-8, unicode)
5. CONFIRM cultural appropriateness (icons, colors, content)

**Quick i18n Checklist:**
- All user-facing strings externalized
- No hardcoded text in code
- Date/time/currency formatted per locale
- RTL languages flip layout correctly
- Unicode characters display properly

**Critical Success Factors:**
- Don't hardcode strings - externalize everything
- Test with real speakers, not just translation files
- RTL requires mirrored UI layout
</default_to_action>

## Quick Reference Card

### When to Use
- Launching in new markets
- Adding language support
- Before international releases
- After UI changes

### i18n vs l10n
| Term | Full Name | Focus |
|------|-----------|-------|
| **i18n** | Internationalization | Building for localization |
| **l10n** | Localization | Adapting for specific locale |

### Common Locale Formats
| Type | US (en-US) | UK (en-GB) | Japan (ja-JP) |
|------|------------|------------|---------------|
| **Date** | 10/24/2025 | 24/10/2025 | 2025/10/24 |
| **Currency** | $1,234.56 | £1,234.56 | ¥1,235 |
| **Number** | 1,234.56 | 1,234.56 | 1,234.56 |

---

## Translation Coverage Testing

```javascript
test('all strings are translated', () => {
  const enKeys = Object.keys(translations.en);
  const frKeys = Object.keys(translations.fr);
  const esKeys = Object.keys(translations.es);

  // All locales have same keys
  expect(frKeys).toEqual(enKeys);
  expect(esKeys).toEqual(enKeys);
});

test('no missing translation placeholders', async ({ page }) => {
  await page.goto('/?lang=fr');
  const text = await page.textContent('body');

  // Should not see placeholder keys
  expect(text).not.toContain('translation.missing');
  expect(text).not.toMatch(/\{\{.*\}\}/); // {{key}} format
});
```

---

## Date/Time/Currency Formats

```javascript
test('date formats by locale', () => {
  const date = new Date('2025-10-24');

  expect(formatDate(date, 'en-US')).toBe('10/24/2025');
  expect(formatDate(date, 'en-GB')).toBe('24/10/2025');
  expect(formatDate(date, 'ja-JP')).toBe('2025/10/24');
});

test('currency formats by locale', () => {
  const amount = 1234.56;

  expect(formatCurrency(amount, 'en-US', 'USD')).toBe('$1,234.56');
  expect(formatCurrency(amount, 'de-DE', 'EUR')).toBe('1.234,56 €');
  expect(formatCurrency(amount, 'ja-JP', 'JPY')).toBe('¥1,235');
});
```

---

## RTL (Right-to-Left) Testing

```javascript
test('layout flips for RTL languages', async ({ page }) => {
  await page.goto('/?lang=ar'); // Arabic

  const dir = await page.locator('html').getAttribute('dir');
  expect(dir).toBe('rtl');

  // Navigation should be on right
  const nav = await page.locator('nav');
  const styles = await nav.evaluate(el =>
    window.getComputedStyle(el)
  );
  expect(styles.direction).toBe('rtl');
});

test('icons/images appropriate for RTL', async ({ page }) => {
  await page.goto('/?lang=he'); // Hebrew

  // Back arrow should point right in RTL
  const backIcon = await page.locator('.back-icon');
  expect(await backIcon.getAttribute('class')).toContain('rtl-flipped');
});
```

---

## Unicode Character Support

```javascript
test('supports unicode characters', async ({ page }) => {
  // Japanese
  await page.fill('#name', '山田太郎');
  await page.click('#submit');

  const saved = await db.users.findOne({ /* ... */ });
  expect(saved.name).toBe('山田太郎');

  // Arabic
  await page.fill('#name', 'محمد');
  // Emoji
  await page.fill('#bio', '👋🌍');

  expect(saved.bio).toBe('👋🌍');
});
```

---

## Agent-Driven Localization Testing

```typescript
// Comprehensive localization validation
await Task("Localization Testing", {
  url: 'https://example.com',
  locales: ['en-US', 'fr-FR', 'de-DE', 'ja-JP', 'ar-SA'],
  checks: ['translations', 'formats', 'rtl', 'unicode'],
  detectHardcodedStrings: true
}, "qe-test-generator");

// Returns:
// {
//   locales: 5,
//   missingTranslations: 3,
//   formatIssues: 1,
//   rtlIssues: 0,
//   hardcodedStrings: ['button.submit', 'header.title']
// }
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/localization-testing/
├── translations/*       - Translation coverage
├── formats/*            - Locale-specific formats
├── rtl-validation/*     - RTL layout checks
└── unicode/*            - Character encoding tests
```

### Fleet Coordination
```typescript
const l10nFleet = await FleetManager.coordinate({
  strategy: 'localization-testing',
  agents: [
    'qe-test-generator',   // Generate l10n tests
    'qe-test-executor',    // Execute across locales
    'qe-visual-tester'     // RTL visual validation
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [accessibility-testing](../accessibility-testing/) - Language accessibility
- [compatibility-testing](../compatibility-testing/) - Cross-platform i18n
- [visual-testing-advanced](../visual-testing-advanced/) - RTL visual regression

---

## Remember

**Don't hardcode. Externalize all user-facing strings.** Every string visible to users must come from translation files, not code.

**Test with native speakers, not just translation files.** Machine translations and translation files can have context issues that only native speakers catch.

**With Agents:** Agents validate translation coverage, detect hardcoded strings, test locale-specific formatting, and verify RTL layouts automatically across all supported languages.
