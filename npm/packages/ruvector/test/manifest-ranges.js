#!/usr/bin/env node
/**
 * Every dependency range is bounded above. An open-ended `>=x` range let
 * `@ruvector/ruvllm >=2.0.0` resolve to 4.0.1, a different product published
 * under the same name without the runtime the `llm` / `sona` commands use.
 * The 4.0.x line is excluded explicitly.
 */

const assert = require('assert');
const path = require('path');

const pkg = require(path.join(__dirname, '..', 'package.json'));
const failures = [];
for (const section of ['dependencies', 'optionalDependencies', 'peerDependencies', 'devDependencies']) {
  for (const [name, range] of Object.entries(pkg[section] || {})) {
    for (const alt of String(range).split('||').map((s) => s.trim())) {
      const bounded = /^[\^~]\d/.test(alt) || /^\d+\.\d+\.\d+$/.test(alt) || (/>=/.test(alt) && /</.test(alt));
      if (!bounded) failures.push(`${section}.${name}: "${range}" (alternative "${alt}" has no upper bound)`);
    }
  }
}
assert.deepStrictEqual(failures, [], `unbounded ranges:\n  ${failures.join('\n  ')}`);

const ruvllm = pkg.peerDependencies['@ruvector/ruvllm'];
assert.ok(!/(^|\|\|)\s*>=\s*2\.0\.0\s*($|\|\|)/.test(ruvllm), 'ruvllm must not be open-ended');
assert.ok(/<\s*4\.0\.0|\^4\.1\.0/.test(ruvllm), `ruvllm range must exclude 4.0.x: ${ruvllm}`);
console.log(`  PASS  all dependency ranges are bounded (ruvllm: ${ruvllm})`);
