#!/usr/bin/env node
// Packs this package, installs the tarball into a temp dir and checks that
// require(), import() and a subpath resolve, and that a Bell state measures
// ~50/50 through @ruvector/ruqu-wasm. Run after `npm run build`.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const pkgDir = resolve(import.meta.dirname, '..');
const work = mkdtempSync(join(process.env.PACK_SMOKE_DIR ?? tmpdir(), 'wasm-unified-pack-'));
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
try {
  const tgz = run('npm', ['pack', '--silent', '--pack-destination', work], pkgDir).trim().split('\n').pop();
  run('npm', ['init', '-y'], work);
  run('npm', ['install', '--no-audit', '--no-fund', join(work, tgz)], work);
  writeFileSync(join(work, 'c.cjs'), `
const u = require('@ruvector/wasm-unified');
const ex = require('@ruvector/wasm-unified/exotic');
if (typeof u.createUnifiedEngine !== 'function' || typeof ex.createExoticEngine !== 'function') throw new Error('cjs exports');
(async () => {
  const e = await u.createUnifiedEngine(); await e.init();
  const x = e.exotic; const b = x.quantumCnot(x.quantumHadamard(x.quantumInit(2), 0), 0, 1);
  let zz = 0; const n = 400;
  for (let i = 0; i < n; i++) { const k = x.quantumMeasure(b).bitstring.join(''); if (k !== '00' && k !== '11') throw new Error('bad outcome ' + k); if (k === '00') zz++; }
  if (zz / n < 0.35 || zz / n > 0.65) throw new Error('not ~50/50: ' + zz + '/' + n);
  console.log('cjs ok', zz + '/' + n);
})().catch((err) => { console.error(err); process.exit(1); });
`);
  writeFileSync(join(work, 'm.mjs'), `
import { createUnifiedEngine, initQuantumBackend } from '@ruvector/wasm-unified';
import { createAttentionEngine } from '@ruvector/wasm-unified/attention';
if (typeof createUnifiedEngine !== 'function' || typeof createAttentionEngine !== 'function') throw new Error('esm exports');
await initQuantumBackend();
console.log('esm ok');
`);
  process.stdout.write(run('node', ['c.cjs'], work));
  process.stdout.write(run('node', ['m.mjs'], work));
} finally {
  rmSync(work, { recursive: true, force: true });
}
