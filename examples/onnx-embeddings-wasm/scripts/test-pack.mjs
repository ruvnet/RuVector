#!/usr/bin/env node
/**
 * Packed-tarball smoke test for ruvector-onnx-embeddings-wasm.
 *
 * Packs ./pkg (or installs a given spec), installs it into a fresh directory outside $HOME
 * (so a stray ~/node_modules cannot shadow it), and loads it the way a consumer would:
 *   - require()  through the "node" + "require" condition (CommonJS glue)
 *   - import()   through the "node" + "import" condition (ESM wrapper)
 *   - import()   of the "./web" entry, initialised with initSync({ module: bytes })
 *   - the "./loader" and "./parallel" subpaths
 * Every load path makes real wasm calls (version, cosineSimilarity, normalizeL2, config).
 *
 * Regression guard: 0.1.0-0.1.2 failed both require() and import().
 *
 * Usage:
 *   node scripts/test-pack.mjs                     # pack ./pkg and test it
 *   node scripts/test-pack.mjs --spec <name@ver>   # test a published version
 *   EMBED=1 node scripts/test-pack.mjs ...         # also download all-MiniLM-L6-v2 and embed
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG = 'ruvector-onnx-embeddings-wasm';
const here = dirname(fileURLToPath(import.meta.url));
const specArg = process.argv.indexOf('--spec');
const base = process.env.PACK_TEST_DIR || tmpdir();
const work = mkdtempSync(join(base, 'onnx-wasm-pack-'));

const run = (cmd, args, opts = {}) =>
  execFileSync(cmd, args, { cwd: work, stdio: ['ignore', 'pipe', 'inherit'], encoding: 'utf8', ...opts });

const CHECKS = `
  const fail = (m) => { console.error('FAIL ' + m); process.exit(1); };
  const check = (m, label) => {
    for (const k of ['WasmEmbedder', 'WasmEmbedderConfig', 'cosineSimilarity', 'normalizeL2', 'simd_available', 'version']) {
      if (typeof m[k] !== 'function') fail(label + ': missing export ' + k);
    }
    const v = m.version();
    if (typeof v !== 'string' || !v) fail(label + ': version() returned ' + v);
    const same = m.cosineSimilarity(new Float32Array([1, 0, 0]), new Float32Array([1, 0, 0]));
    const orth = m.cosineSimilarity(new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0]));
    if (Math.abs(same - 1) > 1e-4 || Math.abs(orth) > 1e-4) fail(label + ': cosineSimilarity ' + same + ' ' + orth);
    const n = m.normalizeL2(new Float32Array([3, 4]));
    if (Math.abs(Math.hypot(n[0], n[1]) - 1) > 1e-4) fail(label + ': normalizeL2');
    new m.WasmEmbedderConfig().setMaxLength(128).setNormalize(true).setPooling(0);
    console.log('ok ' + label + ' (version ' + v + ', simd ' + m.simd_available() + ')');
  };
`;

try {
  run('npm', ['init', '-y']);
  let spec;
  if (specArg !== -1) {
    spec = process.argv[specArg + 1];
  } else {
    const pkgDir = resolve(here, '..', 'pkg');
    const out = run('npm', ['pack', pkgDir, '--pack-destination', work, '--json']);
    spec = join(work, JSON.parse(out)[0].filename);
  }
  run('npm', ['install', '--no-audit', '--no-fund', spec]);

  // CommonJS consumer.
  writeFileSync(join(work, 'cjs.cjs'), `${CHECKS}
    check(require('${PKG}'), 'require()');
    const pj = require('${PKG}/package.json');
    if (pj.type !== 'module') fail('package.json type');
  `);
  console.log(run('node', ['cjs.cjs']).trim());

  // ESM consumer.
  writeFileSync(join(work, 'esm.mjs'), `${CHECKS}
    import { readFileSync } from 'node:fs';
    import { createRequire } from 'node:module';
    const require = createRequire(import.meta.url);

    const node = await import('${PKG}');
    await node.default();
    check(node, 'import()');

    const web = await import('${PKG}/web');
    web.initSync({ module: readFileSync(require.resolve('${PKG}/wasm')) });
    check(web, 'import(/web) + initSync');

    const loader = await import('${PKG}/loader');
    if (typeof loader.createEmbedder !== 'function' || !loader.MODELS['all-MiniLM-L6-v2']) fail('loader exports');
    const parallel = await import('${PKG}/parallel');
    if (typeof parallel.ParallelEmbedder !== 'function') fail('parallel exports');
    console.log('ok subpaths ./loader ./parallel');

    if (process.env.EMBED === '1') {
      const emb = await loader.embed('Hello world');
      if (!(emb instanceof Float32Array) || emb.length !== 384) fail('embed length ' + emb?.length);
      const s = await loader.similarity('I love dogs', 'I adore puppies');
      const d = await loader.similarity('I love dogs', 'Quarterly tax filing deadline');
      if (!(s > d)) fail('similarity ordering ' + s + ' <= ' + d);
      console.log('ok embedding dim=' + emb.length + ' sim(related)=' + s.toFixed(4) + ' sim(unrelated)=' + d.toFixed(4));
    }
  `);
  console.log(run('node', ['esm.mjs']).trim());
  console.log(`PASS ${spec}`);
} finally {
  if (!process.env.KEEP) rmSync(work, { recursive: true, force: true });
}
