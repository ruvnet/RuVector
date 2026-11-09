/**
 * Packaging test: build, `npm pack`, install the tarball into a scratch
 * project, and load every public entry point through both module systems.
 *
 * Guards the 0.3.4/0.3.5 regression where `require('ruvbot')` worked but
 * `await import('ruvbot')` threw (dist/esm had no {"type":"module"} marker and
 * used directory imports that ESM cannot resolve).
 *
 * Set RUVBOT_PACK_TEST_SKIP_BUILD=1 to reuse an existing dist/.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'child_process';
import { mkdtempSync, rmSync, writeFileSync, readdirSync } from 'fs';
import { tmpdir } from 'os';
import { join, resolve } from 'path';

const pkgDir = resolve(__dirname, '../../..');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

const SUBPATHS = [
  'ruvbot',
  'ruvbot/core',
  'ruvbot/learning',
  'ruvbot/integrations/slack',
  'ruvbot/integrations/webhooks',
];

let scratch: string;

function run(cmd: string, args: string[], cwd: string): string {
  return execFileSync(cmd, args, {
    cwd,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, npm_config_loglevel: 'error' },
  });
}

/** Run a script in the scratch project and return its parsed JSON output. */
function probe(file: string, body: string): Record<string, string> {
  writeFileSync(join(scratch, file), body);
  const out = run(process.execPath, [file], scratch);
  return JSON.parse(out.trim().split('\n').pop() as string);
}

describe('packed tarball entry points', () => {
  beforeAll(() => {
    if (!process.env.RUVBOT_PACK_TEST_SKIP_BUILD) run(npm, ['run', 'build'], pkgDir);

    scratch = mkdtempSync(join(tmpdir(), 'ruvbot-pack-'));
    run(npm, ['pack', '--ignore-scripts', '--pack-destination', scratch], pkgDir);
    const tarball = readdirSync(scratch).find((f) => f.endsWith('.tgz'));
    if (!tarball) throw new Error('npm pack produced no tarball');

    writeFileSync(join(scratch, 'package.json'), JSON.stringify({ name: 'scratch', private: true }));
    run(
      npm,
      [
        'install',
        `./${tarball}`,
        '--no-save',
        '--no-workspaces',
        '--ignore-scripts',
        '--omit=optional',
        '--legacy-peer-deps',
        '--no-audit',
        '--no-fund',
      ],
      scratch
    );
  }, 600_000);

  afterAll(() => {
    if (scratch) rmSync(scratch, { recursive: true, force: true });
  });

  it('loads every entry point with require()', () => {
    const result = probe(
      'probe.cjs',
      `const out = {};
for (const p of ${JSON.stringify(SUBPATHS)}) {
  try { require(p); out[p] = 'ok'; } catch (e) { out[p] = String(e && e.message); }
}
console.log(JSON.stringify(out));`
    );
    for (const p of SUBPATHS) expect(result[p], p).toBe('ok');
  });

  it('loads every entry point with import()', () => {
    const result = probe(
      'probe.mjs',
      `const out = {};
for (const p of ${JSON.stringify(SUBPATHS)}) {
  try { await import(p); out[p] = 'ok'; } catch (e) { out[p] = String(e && e.message); }
}
console.log(JSON.stringify(out));`
    );
    for (const p of SUBPATHS) expect(result[p], p).toBe('ok');
  });

  it('exposes the same runtime exports from ESM and CJS roots', () => {
    const result = probe(
      'probe-keys.mjs',
      `import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const esm = Object.keys(await import('ruvbot')).filter((k) => k !== 'default' && k !== 'module.exports').sort();
const cjs = Object.keys(require('ruvbot')).sort();
console.log(JSON.stringify({ esm: esm.join(','), cjs: cjs.join(','), hasChatEnhancer: String(esm.includes('ChatEnhancer')) }));`
    );
    expect(result.hasChatEnhancer).toBe('true');
    expect(result.esm).toBe(result.cjs);
  });
});
