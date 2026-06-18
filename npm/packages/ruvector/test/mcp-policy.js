#!/usr/bin/env node

/**
 * Unit tests for the MCP tool-access policy (ADR-256 default-deny posture).
 * Pure module — no MCP stdio server required.
 */

const assert = require('assert');
const {
  MCP_PROFILES,
  parseList,
  buildToolPolicy,
  isToolAllowed,
  filterAllowedTools,
} = require('../bin/mcp-policy.js');

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failed++;
    failures.push({ name, error: err.message || String(err) });
    console.log(`  FAIL  ${name}`);
    console.log(`        ${err.message || err}`);
  }
}

const TOOLS = [
  { name: 'ruvector' },
  { name: 'hooks_route' },
  { name: 'hooks_recall' },
  { name: 'hooks_security_scan' },
  { name: 'hooks_force_learn' },
];

console.log('\n--- MCP tool-access policy (ADR-256) ---\n');

test('parseList handles comma and whitespace separators', () => {
  assert.deepStrictEqual(parseList('a, b ,c'), ['a', 'b', 'c']);
  assert.deepStrictEqual(parseList('a b\tc'), ['a', 'b', 'c']);
  assert.deepStrictEqual(parseList(''), []);
  assert.deepStrictEqual(parseList(undefined), []);
});

test('no env configured → allow-all, not configured', () => {
  const p = buildToolPolicy({});
  assert.strictEqual(p.configured, false);
  assert.strictEqual(p.allowSet, null);
  for (const t of TOOLS) assert.strictEqual(isToolAllowed(t.name, p), true);
  assert.strictEqual(filterAllowedTools(TOOLS, p).length, TOOLS.length);
});

test('RUVECTOR_MCP_ALLOW restricts to the listed tools (deny rest)', () => {
  const p = buildToolPolicy({ RUVECTOR_MCP_ALLOW: 'hooks_route, hooks_recall' });
  assert.strictEqual(p.configured, true);
  assert.strictEqual(isToolAllowed('hooks_route', p), true);
  assert.strictEqual(isToolAllowed('hooks_recall', p), true);
  assert.strictEqual(isToolAllowed('hooks_security_scan', p), false);
  assert.strictEqual(isToolAllowed('ruvector', p), false);
  assert.deepStrictEqual(
    filterAllowedTools(TOOLS, p).map((t) => t.name).sort(),
    ['hooks_recall', 'hooks_route'],
  );
});

test('RUVECTOR_MCP_DENY blocks specific tools, allows the rest', () => {
  const p = buildToolPolicy({ RUVECTOR_MCP_DENY: 'hooks_force_learn' });
  assert.strictEqual(p.configured, true);
  assert.strictEqual(isToolAllowed('hooks_force_learn', p), false);
  assert.strictEqual(isToolAllowed('hooks_route', p), true);
});

test('DENY wins over ALLOW (precedence)', () => {
  const p = buildToolPolicy({
    RUVECTOR_MCP_ALLOW: 'hooks_route,hooks_force_learn',
    RUVECTOR_MCP_DENY: 'hooks_force_learn',
  });
  assert.strictEqual(isToolAllowed('hooks_route', p), true);
  assert.strictEqual(isToolAllowed('hooks_force_learn', p), false);
});

test('RUVECTOR_MCP_PROFILE=readonly applies the curated subset', () => {
  const p = buildToolPolicy({ RUVECTOR_MCP_PROFILE: 'readonly' });
  assert.strictEqual(p.profile, 'readonly');
  assert.strictEqual(p.configured, true);
  for (const name of MCP_PROFILES.readonly) {
    assert.strictEqual(isToolAllowed(name, p), true, `${name} should be allowed in readonly`);
  }
  // A mutating tool not in the profile is denied
  assert.strictEqual(isToolAllowed('hooks_force_learn', p), false);
});

test('profile + extra ALLOW are unioned', () => {
  const p = buildToolPolicy({
    RUVECTOR_MCP_PROFILE: 'readonly',
    RUVECTOR_MCP_ALLOW: 'hooks_force_learn',
  });
  assert.strictEqual(isToolAllowed('hooks_force_learn', p), true);
  assert.strictEqual(isToolAllowed('hooks_stats', p), true);
});

test('unknown profile with no allow/deny → not configured (allow-all)', () => {
  const p = buildToolPolicy({ RUVECTOR_MCP_PROFILE: 'does-not-exist' });
  assert.strictEqual(p.profile, null);
  assert.strictEqual(p.configured, false);
  assert.strictEqual(isToolAllowed('hooks_force_learn', p), true);
});

console.log('\n' + '='.repeat(60));
console.log(`\nResults: ${passed} passed, ${failed} failed\n`);
if (failures.length > 0) {
  console.log('Failures:');
  for (const f of failures) console.log(`  - ${f.name}: ${f.error}`);
  console.log('');
}
if (failed > 0) {
  console.log('SOME TESTS FAILED\n');
  process.exit(1);
} else {
  console.log('ALL TESTS PASSED\n');
}
