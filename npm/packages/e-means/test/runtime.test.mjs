import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

test('compiled-module runtime and plain imports load no Node builtins or Darwin SDK', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'e-means-runtime-'));
  try {
    const bytes = await readFile(new URL('../wasm/ruvector_e_means_bg.wasm', import.meta.url));
    const loader = 'data:text/javascript,' + encodeURIComponent(`
      export async function resolve(specifier, context, next) {
        if (specifier.startsWith('node:') || specifier.includes('@metaharness/darwin')) {
          throw new Error('Unexpected Node/SDK import: ' + specifier);
        }
        return next(specifier, context);
      }
    `);
    const file = join(temp, 'consumer.mjs');
    await writeFile(file, `
      const runtime = await import(${JSON.stringify(new URL('../runtime.mjs', import.meta.url).href)});
      let rejected = false;
      try { await runtime.init(); } catch (error) { rejected = /requires WASM bytes/.test(error.message); }
      if (!rejected) throw new Error('Runtime must reject an omitted source before initialization');
      const bytes = Uint8Array.from(atob(${JSON.stringify(bytes.toString('base64'))}), character => character.charCodeAt(0));
      const module = new WebAssembly.Module(bytes);
      await Promise.all([runtime.init(module), runtime.init(module)]);
      const result = runtime.fit(new Float32Array([0, 2]), 1, { clusters: 1 });
      if (result.centroids[0] !== 1 || result.mse !== 1) throw new Error('Compiled module fit differs');
      await import(${JSON.stringify(new URL('../e-means.mjs', import.meta.url).href)});
      const integration = await import(${JSON.stringify(new URL('../integrations/metaharness.mjs', import.meta.url).href)});
      if (typeof integration.createClusteringEvaluator !== 'function') throw new Error('Evaluator export missing');
      console.log('Node-free runtime and lazy SDK imports passed');
    `);
    const child = spawnSync(process.execPath, ['--experimental-loader', loader, file], { encoding: 'utf8' });
    assert.equal(child.status, 0, child.stderr);
    assert.match(child.stdout, /Node-free runtime/);
  } finally { await rm(temp, { recursive: true, force: true }); }
});
